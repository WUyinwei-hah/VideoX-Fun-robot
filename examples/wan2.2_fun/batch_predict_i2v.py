"""
Batch I2V inference script with multi-GPU support.
Based on predict_i2v.py from VideoX-Fun.

Usage:
    # Single process with all GPUs (uses ulysses/ring parallelism within each sample)
    python batch_predict_i2v.py

    # Multi-process: each process handles different samples on different GPU
    # For 8 GPUs, run 8 processes:
    CUDA_VISIBLE_DEVICES=0 python batch_predict_i2v.py --gpu_id 0 --num_gpus 8
    CUDA_VISIBLE_DEVICES=1 python batch_predict_i2v.py --gpu_id 1 --num_gpus 8
    ...
    
    # Or use torchrun for distributed:
    torchrun --nproc_per_node=8 batch_predict_i2v.py --distributed
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import re
import torch.distributed as dist
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer, CLIPModel,
                              WanT5EncoderModel, Wan2_2Transformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2FunInpaintPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                   save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Batch I2V inference with multi-GPU support")
    parser.add_argument("--csv_path", type=str, 
                        default="/gemini/code/FFGO-Video-Customization-main/Data/combined_first_frames/0-data.csv",
                        help="Path to CSV file containing image_path and prompt")
    parser.add_argument("--image_base_path", type=str,
                        default="/gemini/code/FFGO-Video-Customization-main/Data",
                        help="Base path to prepend to image_path from CSV")
    parser.add_argument("--save_path", type=str, 
                        default="samples/batch_i2v_output",
                        help="Directory to save output videos")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID for this process (0-7 for 8 GPUs)")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Total number of GPUs for data parallel")
    parser.add_argument("--distributed", action="store_true",
                        help="Use torch distributed (torchrun)")

    parser.add_argument(
        "--parallel_mode",
        type=str,
        default="data_parallel",
        choices=["data_parallel", "model_parallel"],
        help="data_parallel: one process per GPU with full model; model_parallel: one job sharded across GPUs (ulysses/ring), run with torchrun",
    )
    
    # Model config
    parser.add_argument("--config_path", type=str,
                        default="/gemini/code/VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml")
    parser.add_argument("--model_name", type=str,
                        default="/gemini/code/models/Wan2.2-Fun-A14B-InP")
    
    # Generation params
    parser.add_argument("--video_length", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--target_height", type=int, default=640)
    
    # Multi-GPU inference within single sample (ulysses/ring)
    parser.add_argument("--ulysses_degree", type=int, default=1,
                        help="Ulysses parallel degree (for single sample multi-GPU)")
    parser.add_argument("--ring_degree", type=int, default=1,
                        help="Ring parallel degree (for single sample multi-GPU)")

    parser.add_argument("--no_data_parallel", action="store_true",
                        help="Do not split CSV across processes. Useful for debugging or when using model-parallel.")

    parser.add_argument(
        "--gpu_memory_mode",
        type=str,
        default="model_full_load",
        choices=[
            "model_full_load",
            "model_full_load_and_qfloat8",
            "model_cpu_offload",
            "model_cpu_offload_and_qfloat8",
            "sequential_cpu_offload",
        ],
    )
    parser.add_argument("--fsdp_dit", action="store_true")
    parser.add_argument("--fsdp_text_encoder", action="store_true")

    parser.add_argument("--lora_low_path", type=str, default="/gemini/code/VideoX-Fun/output_dir_low",
                        help="LoRA for low-noise transformer (file .safetensors or a directory containing checkpoint-*.safetensors)")
    parser.add_argument("--lora_high_path", type=str, default="/gemini/code/VideoX-Fun/output_dir_high",
                        help="LoRA for high-noise transformer_2 (file .safetensors or a directory containing checkpoint-*.safetensors)")
    parser.add_argument("--lora_weight_low", type=float, default=1)
    parser.add_argument("--lora_weight_high", type=float, default=1)
    parser.add_argument("--no_lora", action="store_true", help="Disable loading LoRA")
    parser.add_argument("--unmerge_lora", action="store_true", help="Unmerge LoRA weights after inference")
    
    return parser.parse_args()


def setup_distributed(args):
    """Setup for distributed training with torchrun."""
    if args.distributed:
        # For model_parallel, we follow predict_i2v.py and let set_multi_gpus_devices()
        # initialize distributed. Here we only read env vars for logging/routing.
        args.rank = int(os.environ.get("RANK", "0"))
        args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        args.gpu_id = args.rank
        args.num_gpus = args.world_size
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

    # Default behavior: data-parallel splits CSV unless explicitly disabled.
    args.data_parallel = not args.no_data_parallel
    return args


def load_data(csv_path, image_base_path, gpu_id, num_gpus):
    """Load CSV and split data for this GPU."""
    df = pd.read_csv(csv_path)
    
    # Filter out rows with empty prompt or image_path
    df = df.dropna(subset=['image_path', 'prompt'])
    df = df[df['prompt'].str.strip() != '']
    df = df[df['image_path'].str.strip() != '']
    
    # Prepend base path to image_path
    df['full_image_path'] = df['image_path'].apply(
        lambda x: os.path.join(image_base_path, x.lstrip('/'))
    )

    return df


def select_data_subset(df, gpu_id, num_gpus, data_parallel: bool):
    if not data_parallel:
        print(f"[GPU {gpu_id}] data_parallel disabled, processing all {len(df)} samples.")
        return df.reset_index(drop=True)
    
    total_samples = len(df)
    samples_per_gpu = total_samples // num_gpus
    start_idx = gpu_id * samples_per_gpu
    end_idx = start_idx + samples_per_gpu if gpu_id < num_gpus - 1 else total_samples

    df_subset = df.iloc[start_idx:end_idx].reset_index(drop=True)

    print(f"[GPU {gpu_id}] Processing samples {start_idx} to {end_idx-1} ({len(df_subset)} samples)")

    return df_subset


def _extract_checkpoint_step(path: str) -> int:
    m = re.search(r"checkpoint-(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def resolve_lora_path(lora_path: str) -> str:
    if lora_path is None:
        return None
    lora_path = str(lora_path)
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")
    if os.path.isdir(lora_path):
        candidates = []
        for name in os.listdir(lora_path):
            if name.endswith(".safetensors"):
                candidates.append(os.path.join(lora_path, name))
        if not candidates:
            raise FileNotFoundError(f"No .safetensors found under LoRA directory: {lora_path}")
        candidates.sort(key=_extract_checkpoint_step)
        return candidates[-1]
    return lora_path


def get_sample_size(image_path, target_height=640):
    """Calculate sample size based on image aspect ratio."""
    if image_path is not None and os.path.exists(image_path):
        image = Image.open(image_path)
        width, height = image.size
        aspect_ratio = width / height
        
        target_width = int(target_height * aspect_ratio)
        target_width = (target_width // 16) * 16
        
        return [target_height, target_width]
    else:
        return [480, 864]


def build_pipeline(args, device):
    """Build the inference pipeline."""
    
    # GPU memory mode
    GPU_memory_mode = args.gpu_memory_mode
    
    # TeaCache config
    enable_teacache = True
    teacache_threshold = 0.15
    num_skip_start_steps = 5
    teacache_offload = False
    
    # Other configs
    cfg_skip_ratio = 0
    enable_riflex = False
    riflex_k = 6
    sampler_name = "Flow"
    shift = 5
    
    weight_dtype = torch.bfloat16
    
    config = OmegaConf.load(args.config_path)
    boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)
    
    # Load transformer
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(args.model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    
    if config['transformer_additional_kwargs'].get('transformer_combination_type', 'single') == "moe":
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(args.model_name, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
    else:
        transformer_2 = None
    
    # Load VAE
    Chosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8
    }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
    
    vae = Chosen_AutoencoderKL.from_pretrained(
        os.path.join(args.model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )
    
    # Load Text encoder
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()
    
    # Get Scheduler
    Chosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]
    
    scheduler = Chosen_Scheduler(
        **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    
    # Build Pipeline
    pipeline = Wan2_2FunInpaintPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

    # Merge LoRA on CPU BEFORE any multi-gpu sharding/offload.
    # This avoids CUDA "illegal memory access" that can happen when merging into sharded/offloaded modules.
    if getattr(args, "resolved_lora_low", None) is not None:
        pipeline = merge_lora(
            pipeline,
            args.resolved_lora_low,
            args.lora_weight_low,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    if getattr(args, "resolved_lora_high", None) is not None and transformer_2 is not None:
        pipeline = merge_lora(
            pipeline,
            args.resolved_lora_high,
            args.lora_weight_high,
            device=torch.device("cpu"),
            dtype=torch.float32,
            sub_transformer_name="transformer_2",
        )
    
    # Multi-GPU setup for single sample (if using ulysses/ring)
    if args.ulysses_degree > 1 or args.ring_degree > 1:
        from functools import partial
        transformer.enable_multi_gpus_inference()
        if transformer_2 is not None:
            transformer_2.enable_multi_gpus_inference()

        if args.fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)
            if transformer_2 is not None:
                pipeline.transformer_2 = shard_fn(pipeline.transformer_2)

        if args.fsdp_text_encoder:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.text_encoder = shard_fn(pipeline.text_encoder)

    if GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation",], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        if transformer_2 is not None:
            replace_parameters_by_name(transformer_2, ["modulation",], device=device)
            transformer_2.freqs = transformer_2.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)
    
    # Enable TeaCache
    coefficients = get_teacache_coefficients(args.model_name) if enable_teacache else None
    if coefficients is not None:
        print(f"[GPU {args.gpu_id}] Enable TeaCache with threshold {teacache_threshold}")
        pipeline.transformer.enable_teacache(
            coefficients, args.num_inference_steps, teacache_threshold, 
            num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
        )
        if transformer_2 is not None:
            pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)
    
    if cfg_skip_ratio is not None and cfg_skip_ratio > 0:
        pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, args.num_inference_steps)
        if transformer_2 is not None:
            pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)
    
    return pipeline, vae, boundary, weight_dtype


def generate_video(pipeline, vae, args, image_path, prompt, save_name, device, boundary):
    """Generate a single video."""
    
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    
    sample_size = get_sample_size(image_path, args.target_height)
    video_length = args.video_length
    
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    with torch.no_grad():
        video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
        
        input_video, input_video_mask, clip_image = get_image_to_video_latent(
            image_path, None, video_length=video_length, sample_size=sample_size
        )
        
        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            boundary=boundary,
            video=input_video,
            mask_video=input_video_mask,
            shift=5,
        ).videos

    # Save only on rank 0 to avoid multiple ranks writing the same file in model-parallel mode.
    if args.distributed and dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            dist.barrier()
            return None

    os.makedirs(args.save_path, exist_ok=True)
    video_path = os.path.join(args.save_path, f"{save_name}.mp4")
    save_videos_grid(sample, video_path, fps=args.fps)

    if args.distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()

    return video_path


def cleanup_distributed(args):
    if args.distributed and dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()


def main():
    args = parse_args()
    args = setup_distributed(args)

    if args.parallel_mode == "model_parallel":
        if not args.distributed:
            raise ValueError("model_parallel requires torchrun. Please run with: torchrun --nproc_per_node=8 batch_predict_i2v.py --distributed --parallel_mode model_parallel --ulysses_degree 8 --ring_degree 1")
        # In model-parallel, do not split CSV.
        args.data_parallel = False
        # Recommend enabling FSDP sharding (same as predict_i2v.py default in multi-gpu).
        args.fsdp_dit = True if not args.fsdp_dit else args.fsdp_dit
        args.fsdp_text_encoder = True if not args.fsdp_text_encoder else args.fsdp_text_encoder

    print(f"[GPU {args.gpu_id}] Starting batch inference...")
    
    # Set device
    # - Model-parallel: set_multi_gpus_devices expects all GPUs to be visible and ranks to be initialized (torchrun).
    # - Data-parallel w/ CUDA_VISIBLE_DEVICES=...: visible GPU is cuda:0.
    if args.parallel_mode == "model_parallel":
        # Follow predict_i2v.py: set_multi_gpus_devices initializes distributed and returns the device used everywhere.
        device = set_multi_gpus_devices(args.ulysses_degree, args.ring_degree)
        if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
            print(f"Enable model_parallel with ulysses_degree={args.ulysses_degree}, ring_degree={args.ring_degree}")
    else:
        # data_parallel: When using CUDA_VISIBLE_DEVICES, the visible GPU is always cuda:0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
    
    try:
        # Load data
        df_all = load_data(args.csv_path, args.image_base_path, args.gpu_id, args.num_gpus)
        df = select_data_subset(df_all, args.gpu_id, args.num_gpus, args.data_parallel)
        
        if len(df) == 0:
            print(f"[GPU {args.gpu_id}] No samples to process.")
            return
        
        # Build pipeline
        print(f"[GPU {args.gpu_id}] Loading model...")
        args.resolved_lora_low = None
        args.resolved_lora_high = None
        if not args.no_lora:
            args.resolved_lora_low = resolve_lora_path(args.lora_low_path) if args.lora_low_path else None
            args.resolved_lora_high = resolve_lora_path(args.lora_high_path) if args.lora_high_path else None
            if args.resolved_lora_low is not None:
                print(f"[GPU {args.gpu_id}] Will merge LoRA low on CPU: {args.resolved_lora_low} (weight={args.lora_weight_low})")
            if args.resolved_lora_high is not None:
                print(f"[GPU {args.gpu_id}] Will merge LoRA high on CPU: {args.resolved_lora_high} (weight={args.lora_weight_high})")

        pipeline, vae, boundary, weight_dtype = build_pipeline(args, device)
        
        # Process each sample
        for idx, row in df.iterrows():
            image_path = row['full_image_path']
            prompt = row['prompt']
            
            # Generate unique save name
            original_idx = row.name if 'Unnamed: 0' not in row else row['Unnamed: 0']
            save_name = f"sample_{original_idx:04d}_gpu{args.gpu_id}"
            
            print(f"[GPU {args.gpu_id}] Processing {idx+1}/{len(df)}: {os.path.basename(image_path)}")
            
            if not os.path.exists(image_path):
                print(f"[GPU {args.gpu_id}] Warning: Image not found: {image_path}")
                continue
            
            try:
                video_path = generate_video(
                    pipeline, vae, args, image_path, prompt, save_name, device, boundary
                )
                if video_path is not None:
                    print(f"[GPU {args.gpu_id}] Saved: {video_path}")
            except Exception as e:
                print(f"[GPU {args.gpu_id}] Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if args.unmerge_lora and not args.no_lora:
            if args.parallel_mode == "model_parallel":
                print(f"[GPU {args.gpu_id}] Warning: --unmerge_lora is not supported in model_parallel mode, skip.")
            else:
                if args.resolved_lora_low is not None:
                    pipeline = unmerge_lora(pipeline, args.resolved_lora_low, args.lora_weight_low, device=device, dtype=weight_dtype)
                if args.resolved_lora_high is not None and getattr(pipeline, "transformer_2", None) is not None:
                    pipeline = unmerge_lora(
                        pipeline,
                        args.resolved_lora_high,
                        args.lora_weight_high,
                        device=device,
                        dtype=weight_dtype,
                        sub_transformer_name="transformer_2",
                    )

        print(f"[GPU {args.gpu_id}] Batch inference completed!")
    finally:
        cleanup_distributed(args)


if __name__ == "__main__":
    main()
