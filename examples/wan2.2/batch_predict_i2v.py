import os
import sys
import argparse
import re

import pandas as pd
import torch
import torch.distributed as dist
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (
    AutoencoderKLWan,
    AutoencoderKLWan3_8,
    AutoTokenizer,
    CLIPModel,
    WanT5EncoderModel,
    Wan2_2Transformer3DModel,
)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2I2VPipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    replace_parameters_by_name,
    convert_weight_dtype_wrapper,
)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import filter_kwargs, get_image_to_video_latent, save_videos_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Batch I2V inference")

    parser.add_argument(
        "--csv_path",
        type=str,
        default="/gemini/code/FFGO-Video-Customization-main/Data/combined_first_frames/0-data.csv",
    )
    parser.add_argument(
        "--image_base_path",
        type=str,
        default="/gemini/code/FFGO-Video-Customization-main/Data",
    )
    parser.add_argument("--save_path", type=str, default="samples/batch_i2v_output")

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--distributed", action="store_true")

    parser.add_argument(
        "--parallel_mode",
        type=str,
        default="data_parallel",
        choices=["data_parallel", "model_parallel"],
    )

    parser.add_argument("--config_path", type=str, default="config/wan2.2/wan_civitai_i2v.yaml")
    parser.add_argument("--model_name", type=str, default="/gemini/code/models/Wan2.2-I2V-A14B")

    parser.add_argument("--video_length", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--target_height", type=int, default=640)

    parser.add_argument("--negative_prompt", type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

    parser.add_argument(
        "--sampler_name",
        type=str,
        default="Flow_Unipc",
        choices=["Flow", "Flow_Unipc", "Flow_DPM++"],
    )
    parser.add_argument("--shift", type=float, default=5.0)

    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)

    parser.add_argument("--no_data_parallel", action="store_true")

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

    parser.add_argument("--enable_teacache", action="store_true")
    parser.add_argument("--teacache_threshold", type=float, default=0.10)
    parser.add_argument("--teacache_num_skip_start_steps", type=int, default=5)
    parser.add_argument("--teacache_offload", action="store_true")

    parser.add_argument(
        "--lora_low_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--lora_high_path",
        type=str,
        default=None,
    )
    parser.add_argument("--lora_weight_low", type=float, default=0.55)
    parser.add_argument("--lora_weight_high", type=float, default=0.55)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--unmerge_lora", action="store_true")

    return parser.parse_args()


def setup_distributed(args):
    if args.distributed:
        args.rank = int(os.environ.get("RANK", "0"))
        args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        args.gpu_id = args.rank
        args.num_gpus = args.world_size
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

    args.data_parallel = not args.no_data_parallel
    return args


def load_data(csv_path: str, image_base_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["image_path", "prompt"])
    df = df[df["prompt"].astype(str).str.strip() != ""]
    df = df[df["image_path"].astype(str).str.strip() != ""]

    df = df.reset_index(drop=False).rename(columns={"index": "_global_index"})
    df["full_image_path"] = df["image_path"].apply(lambda x: os.path.join(image_base_path, str(x).lstrip("/")))
    return df


def select_data_subset(df: pd.DataFrame, gpu_id: int, num_gpus: int, data_parallel: bool) -> pd.DataFrame:
    if not data_parallel:
        return df.reset_index(drop=True)

    total_samples = len(df)
    samples_per_gpu = total_samples // num_gpus
    start_idx = gpu_id * samples_per_gpu
    end_idx = start_idx + samples_per_gpu if gpu_id < num_gpus - 1 else total_samples
    return df.iloc[start_idx:end_idx].reset_index(drop=True)


def _extract_checkpoint_step(path: str) -> int:
    m = re.search(r"checkpoint-(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def resolve_lora_path(lora_path: str) -> str | None:
    if lora_path is None:
        return None
    lora_path = str(lora_path)
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")
    if os.path.isdir(lora_path):
        candidates = [os.path.join(lora_path, n) for n in os.listdir(lora_path) if n.endswith(".safetensors")]
        if not candidates:
            raise FileNotFoundError(f"No .safetensors found under LoRA directory: {lora_path}")
        candidates.sort(key=_extract_checkpoint_step)
        return candidates[-1]
    return lora_path


def get_sample_size(image_path: str, target_height: int = 640):
    if image_path is not None and os.path.exists(image_path):
        image = Image.open(image_path)
        width, height = image.size
        aspect_ratio = width / height
        target_width = int(target_height * aspect_ratio)
        target_width = (target_width // 16) * 16
        return [target_height, target_width]
    return [512, 512]


def build_pipeline(args, device):
    config = OmegaConf.load(args.config_path)
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.900)

    weight_dtype = torch.bfloat16

    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(
            args.model_name,
            config["transformer_additional_kwargs"].get("transformer_low_noise_model_subpath", "transformer"),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    transformer_2 = None
    if config["transformer_additional_kwargs"].get("transformer_combination_type", "single") == "moe":
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(
                args.model_name,
                config["transformer_additional_kwargs"].get("transformer_high_noise_model_subpath", "transformer"),
            ),
            transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )

    Chosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = Chosen_AutoencoderKL.from_pretrained(
        os.path.join(args.model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")),
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_name, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()

    Chosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[args.sampler_name]

    if args.sampler_name in {"Flow_Unipc", "Flow_DPM++"}:
        config["scheduler_kwargs"]["shift"] = 1

    scheduler = Chosen_Scheduler(**filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config["scheduler_kwargs"])))

    pipeline = Wan2_2I2VPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

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

    if args.gpu_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        if transformer_2 is not None:
            replace_parameters_by_name(transformer_2, ["modulation"], device=device)
            transformer_2.freqs = transformer_2.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation"], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation"], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)

    if args.enable_teacache:
        coefficients = get_teacache_coefficients(args.model_name)
        if coefficients is not None:
            pipeline.transformer.enable_teacache(
                coefficients,
                args.num_inference_steps,
                args.teacache_threshold,
                num_skip_start_steps=args.teacache_num_skip_start_steps,
                offload=args.teacache_offload,
            )
            if transformer_2 is not None:
                pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

    return pipeline, vae, boundary, weight_dtype


def _save_sample(sample, out_path: str, fps: int):
    if sample.shape[2] == 1:
        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).cpu().numpy().astype("uint8")
        Image.fromarray(image).save(out_path)
    else:
        save_videos_grid(sample, out_path, fps=fps)


def generate_video(pipeline, vae, args, image_path: str, prompt: str, save_name: str, device, boundary):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    sample_size = get_sample_size(image_path, args.target_height)

    with torch.no_grad():
        video_length = args.video_length
        video_length = (
            int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
            if video_length != 1
            else 1
        )

        input_video, input_video_mask, _clip_image = get_image_to_video_latent(
            image_path, None, video_length=video_length, sample_size=sample_size
        )

        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=args.negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            boundary=boundary,
            video=input_video,
            mask_video=input_video_mask,
            shift=args.shift,
        ).videos

    if args.distributed and dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            dist.barrier()
            return None

    os.makedirs(args.save_path, exist_ok=True)
    ext = ".png" if sample.shape[2] == 1 else ".mp4"
    out_path = os.path.join(args.save_path, f"{save_name}{ext}")
    _save_sample(sample, out_path, fps=args.fps)

    if args.distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()

    return out_path


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
            raise ValueError("model_parallel requires torchrun with --distributed")
        args.data_parallel = False
        args.fsdp_dit = True if not args.fsdp_dit else args.fsdp_dit
        args.fsdp_text_encoder = True if not args.fsdp_text_encoder else args.fsdp_text_encoder

    if args.parallel_mode == "model_parallel":
        device = set_multi_gpus_devices(args.ulysses_degree, args.ring_degree)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)

    try:
        df_all = load_data(args.csv_path, args.image_base_path)
        df = select_data_subset(df_all, args.gpu_id, args.num_gpus, args.data_parallel)
        if len(df) == 0:
            return

        args.resolved_lora_low = None
        args.resolved_lora_high = None
        if not args.no_lora:
            args.resolved_lora_low = resolve_lora_path(args.lora_low_path) if args.lora_low_path else None
            args.resolved_lora_high = resolve_lora_path(args.lora_high_path) if args.lora_high_path else None

        pipeline, vae, boundary, weight_dtype = build_pipeline(args, device)

        for _i, row in df.iterrows():
            image_path = row["full_image_path"]
            prompt = str(row["prompt"])
            global_idx = int(row.get("_global_index", _i))
            save_name = f"sample_{global_idx:08d}_gpu{args.gpu_id}"

            if not os.path.exists(image_path):
                continue

            try:
                out_path = generate_video(pipeline, vae, args, image_path, prompt, save_name, device, boundary)
                _ = out_path
            except Exception:
                import traceback

                traceback.print_exc()
                continue

        if args.unmerge_lora and not args.no_lora:
            if args.parallel_mode != "model_parallel":
                if args.resolved_lora_low is not None:
                    pipeline = unmerge_lora(
                        pipeline,
                        args.resolved_lora_low,
                        args.lora_weight_low,
                        device=device,
                        dtype=weight_dtype,
                    )
                if args.resolved_lora_high is not None and getattr(pipeline, "transformer_2", None) is not None:
                    pipeline = unmerge_lora(
                        pipeline,
                        args.resolved_lora_high,
                        args.lora_weight_high,
                        device=device,
                        dtype=weight_dtype,
                        sub_transformer_name="transformer_2",
                    )
    finally:
        cleanup_distributed(args)


if __name__ == "__main__":
    main()
