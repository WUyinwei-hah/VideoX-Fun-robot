import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from decord import VideoReader
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.data.dataset_robot import RobotVideoDataset
from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoTokenizer, CLIPModel, AutoencoderKLWan3_8,
                              WanT5EncoderModel, Wan2_2Transformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2FunInpaintPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from safetensors import safe_open
from videox_fun.utils.utils import filter_kwargs, save_videos_grid
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


# SEP = "A hard cut to the next shot, featuring the same character, maintaining identity continuity across shots."
SEP = "ad23r2 A full-frame white flash for an instant, then a hard cut to the next shot."

# SEP = "ad23r2 A hard cut to the next shot, featuring the same character, maintaining identity continuity across shots."

def _ceil_to_multiple(x: int, m: int) -> int:
    if m <= 1:
        return int(x)
    return int(math.ceil(x / m) * m)


def _compute_hw_from_video_keep_ar(
    video_path: str,
    base_resolution: int,
    multiple_h: int,
    multiple_w: int,
) -> Tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid video size: {video_path} ({w}x{h})")

    short_side = min(w, h)
    ratio = float(base_resolution) / float(short_side)
    out_w = max(1, int(round(w * ratio)))
    out_h = max(1, int(round(h * ratio)))

    out_h = max(multiple_h, _ceil_to_multiple(out_h, multiple_h))
    out_w = max(multiple_w, _ceil_to_multiple(out_w, multiple_w))
    return out_h, out_w


def _sample_video_frames_as_tensor_v2(
    video_path: str,
    num_frames: int,
    height: int,
    width: int,
    sampling: str,
) -> torch.Tensor:
    vr = VideoReader(video_path)
    total = int(len(vr))
    if total <= 0:
        raise ValueError(f"No frames in video: {video_path}")
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")

    if num_frames == 1:
        indices = np.array([0], dtype=int)
    else:
        sampling = str(sampling).lower().strip()
        if sampling == "contiguous":
            if total >= int(num_frames):
                start = 0
                indices = np.arange(start, start + int(num_frames), dtype=int)
            else:
                pad = int(num_frames) - total
                indices = np.concatenate(
                    [np.arange(0, total, dtype=int), np.full((pad,), total - 1, dtype=int)], axis=0
                )
        elif sampling == "stride2_then_tail":
            # Always sample from the tail (end of video) so the ending content is included.
            # Prefer stride=2 backwards from the last frame. If the video is not long enough
            # for stride=2, fall back to taking the last num_frames contiguously.
            need = 1 + 2 * (int(num_frames) - 1)
            if total >= need:
                end = total - 1
                indices = np.arange(end - 2 * (int(num_frames) - 1), end + 1, 2, dtype=int)
            else:
                if total >= int(num_frames):
                    start = total - int(num_frames)
                    indices = np.arange(start, start + int(num_frames), dtype=int)
                else:
                    pad = int(num_frames) - total
                    indices = np.concatenate(
                        [np.zeros((pad,), dtype=int), np.arange(0, total, dtype=int)], axis=0
                    )
        elif sampling == "uniform":
            indices = np.linspace(0, total - 1, int(num_frames), dtype=int)
        else:
            raise ValueError(f"Unknown sampling={sampling}. Expected contiguous|uniform|stride2_then_tail")

    frames = vr.get_batch(indices).asnumpy()  # [T,H,W,3] uint8
    del vr

    resized = []
    for i in range(frames.shape[0]):
        img = Image.fromarray(frames[i])
        img = img.resize((int(width), int(height)))
        resized.append(np.array(img))
    frames = np.stack(resized, axis=0)

    video = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).contiguous()  # [1,3,T,H,W]
    video = video.to(dtype=torch.float32) / 255.0
    return video


def get_video_to_video_latent(
    source_video_path: str,
    video_length: int,
    sample_size: List[int],
    src_frame_sampling: str = "contiguous",
):
    height, width = int(sample_size[0]), int(sample_size[1])
    if video_length < 3:
        raise ValueError("video_length must be >= 3 for source + white + edited")

    remaining = int(video_length - 1)
    n_src = int(remaining // 2)
    if n_src <= 0:
        raise ValueError("video_length too small")

    src_video = _sample_video_frames_as_tensor_v2(
        source_video_path, n_src, height=height, width=width, sampling=src_frame_sampling
    )

    input_video = torch.zeros([1, 3, video_length, height, width], dtype=torch.float32)
    input_video[:, :, :n_src] = src_video
    input_video[:, :, n_src : n_src + 1] = 1.0

    input_video_mask = torch.zeros([1, 1, video_length, height, width], dtype=torch.float32)
    input_video_mask[:, :, n_src + 1 :] = 255.0

    return input_video, input_video_mask, None


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser()

    # Eval args
    parser.add_argument("--eval_jsonl_path", type=str, default=None)
    parser.add_argument("--eval_num_samples", type=int, default=16)
    parser.add_argument("--eval_seed", type=int, default=42)

    # Robot dataset args (used when eval_jsonl_path is not provided)
    parser.add_argument("--robot_data_root", type=str, default=None)
    parser.add_argument("--robot_source_data_root", type=str, default=None)
    parser.add_argument("--robot_suites", nargs="+", default=None)

    # Model args
    parser.add_argument("--config_path", type=str, default="config/wan2.2/wan_civitai_5b.yaml")
    parser.add_argument("--model_name", type=str, default="models/Diffusion_Transformer/Wan2.2-Fun-5B-InP/")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--lora_weight", type=float, default=0.55)
    parser.add_argument("--lora_high_path", type=str, default=None)
    parser.add_argument("--lora_high_weight", type=float, default=0.55)

    # Output args
    parser.add_argument("--output_dir", type=str, default="samples/wan-videos-fun-i2v-firstframe-pair-val")

    # Generation args
    parser.add_argument("--sampler_name", type=str, default="Flow", choices=("Flow", "Flow_Unipc", "Flow_DPM++"))
    parser.add_argument("--shift", type=int, default=5)
    parser.add_argument("--base_resolution", type=int, default=640)
    parser.add_argument("--video_length", type=int, default=129)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--negative_prompt", type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
    parser.add_argument("--src_frame_sampling", type=str, default="contiguous", choices=("contiguous", "uniform", "stride2_then_tail"))

    # Memory args
    parser.add_argument("--gpu_memory_mode", type=str, default="sequential_cpu_offload",
                        choices=("model_full_load", "model_full_load_and_qfloat8", "model_cpu_offload", "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"))
    parser.add_argument("--weight_dtype", type=str, default="bf16", choices=("bf16", "fp16", "fp32"))

    # Multi-GPU args
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--fsdp_dit", action="store_true")
    parser.add_argument("--fsdp_text_encoder", action="store_true")
    parser.add_argument("--compile_dit", action="store_true")

    # TeaCache args
    parser.add_argument("--enable_teacache", action="store_true")
    parser.add_argument("--teacache_threshold", type=float, default=0.10)
    parser.add_argument("--num_skip_start_steps", type=int, default=5)
    parser.add_argument("--teacache_offload", action="store_true")

    args = parser.parse_args()

    weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.weight_dtype]

    device = set_multi_gpus_devices(args.ulysses_degree, args.ring_degree)
    config = OmegaConf.load(args.config_path)
    boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)

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

    # Get Vae
    Chosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8
    }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
    vae = Chosen_AutoencoderKL.from_pretrained(
        os.path.join(args.model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Get Text encoder
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
    }[args.sampler_name]
    if args.sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        config['scheduler_kwargs']['shift'] = 1
    scheduler = Chosen_Scheduler(
        **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Pipeline
    pipeline = Wan2_2FunInpaintPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
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

    if args.compile_dit:
        for i in range(len(pipeline.transformer.blocks)):
            pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
        if transformer_2 is not None:
            for i in range(len(pipeline.transformer_2.blocks)):
                pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])

    if args.gpu_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation",], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        if transformer_2 is not None:
            replace_parameters_by_name(transformer_2, ["modulation",], device=device)
            transformer_2.freqs = transformer_2.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)

    coefficients = get_teacache_coefficients(args.model_name) if args.enable_teacache else None
    if coefficients is not None:
        print(f"Enable TeaCache with threshold {args.teacache_threshold} and skip the first {args.num_skip_start_steps} steps.")
        pipeline.transformer.enable_teacache(
            coefficients, args.num_inference_steps, args.teacache_threshold, num_skip_start_steps=args.num_skip_start_steps, offload=args.teacache_offload
        )
        if transformer_2 is not None:
            pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

    def load_lora_state_dict(lora_path):
        """Load LoRA state dict, handling both file and directory paths."""
        if os.path.isdir(lora_path):
            # Find .safetensors file in directory
            safetensors_files = [f for f in os.listdir(lora_path) if f.endswith('.safetensors')]
            if safetensors_files:
                lora_path = os.path.join(lora_path, safetensors_files[0])
            else:
                raise FileNotFoundError(f"No .safetensors file found in {lora_path}")
        
        # Use safe_open to load tensors one by one (more compatible with multi-GPU)
        state_dict = {}
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict

    if args.lora_path is not None:
        print("loading lora")
        lora_state_dict = load_lora_state_dict(args.lora_path)
        pipeline = merge_lora(pipeline, args.lora_path, args.lora_weight, device=device, dtype=weight_dtype, state_dict=lora_state_dict)
        if transformer_2 is not None and args.lora_high_path is not None:
            lora_high_state_dict = load_lora_state_dict(args.lora_high_path)
            pipeline = merge_lora(pipeline, args.lora_high_path, args.lora_high_weight, device=device, dtype=weight_dtype, state_dict=lora_high_state_dict, sub_transformer_name="transformer_2")

    os.makedirs(args.output_dir, exist_ok=True)

    # Read JSONL and sample
    if args.eval_jsonl_path:
        items = _read_jsonl(args.eval_jsonl_path)
    else:
        if not args.robot_data_root:
            raise ValueError("Either --eval_jsonl_path or --robot_data_root must be provided")
        robot_ds = RobotVideoDataset(
            data_root=args.robot_data_root,
            source_data_root=args.robot_source_data_root,
            suites=args.robot_suites,
            enable_bucket=False,
        )
        items = []
        for d in robot_ds.dataset:
            items.append(
                {
                    "negative_video": d.get("negative_video"),
                    "original_video": d.get("original_video"),
                    "instruction": d.get("instruction") or d.get("text"),
                    "suite": d.get("suite"),
                    "task": d.get("task"),
                    "demo": d.get("demo"),
                }
            )
    rng = random.Random(args.eval_seed)
    if args.eval_num_samples < len(items):
        indices = rng.sample(list(range(len(items))), args.eval_num_samples)
        items = [items[i] for i in indices]

    with torch.no_grad():
        video_length = int((args.video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if args.video_length != 1 else 1

        spatial_ratio = int(getattr(vae, "spatial_compression_ratio", vae.config.spatial_compression_ratio))
        patch = getattr(pipeline.transformer.config, "patch_size", None)
        patch_h = int(patch[1]) if isinstance(patch, (list, tuple)) and len(patch) >= 3 else 16
        patch_w = int(patch[2]) if isinstance(patch, (list, tuple)) and len(patch) >= 3 else 16
        multiple_h = spatial_ratio * patch_h
        multiple_w = spatial_ratio * patch_w

        for i, item in enumerate(items):
            # Robot dataset: use negative video as condition (first half), generate the second half.
            neg_path = str(item.get("negative_video_abs") or item.get("negative_video") or "").strip()
            orig_path = str(item.get("original_video_abs") or item.get("original_video") or "").strip()

            if neg_path and os.path.exists(neg_path):
                condition_path = neg_path
            else:
                print(f"Missing negative video: {neg_path}, skipping...")
                continue

            # Build prompt: use original instruction directly (no SEP / edited prompt)
            prompt = str(item.get("instruction") or "").strip()
            if not prompt:
                prompt = str(item.get("text") or "").strip()
            if not prompt:
                prompt = str(item.get("prompt") or "").strip()

            # Compute sample size from video (use source for AR, condition for frames)
            ar_ref_path = orig_path if orig_path and os.path.exists(orig_path) else condition_path
            height, width = _compute_hw_from_video_keep_ar(
                ar_ref_path,
                base_resolution=args.base_resolution,
                multiple_h=multiple_h,
                multiple_w=multiple_w,
            )
            sample_size = [int(height), int(width)]

            # Prepare input video (use foreground/condition path)
            input_video, input_video_mask, _ = get_video_to_video_latent(
                condition_path,
                video_length=video_length,
                sample_size=sample_size,
                src_frame_sampling=args.src_frame_sampling,
            )
            input_video = input_video.to(device=device)
            input_video_mask = input_video_mask.to(device=device)

            item_seed = int(item.get("seed", args.seed + i))
            generator = torch.Generator(device=device).manual_seed(item_seed)

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

            # Save video
            prefix = str(i).zfill(4)
            video_out = os.path.join(args.output_dir, f"firstframe_pair_val_{prefix}.mp4")
            save_videos_grid(sample, video_out, fps=args.fps)

            # Save metadata
            meta_out = os.path.join(args.output_dir, f"firstframe_pair_val_{prefix}.json")
            with open(meta_out, "w", encoding="utf-8") as mf:
                json.dump(
                    {
                        "index": i,
                        "negative_video": neg_path,
                        "original_video": orig_path,
                        "condition_video": condition_path,
                        "prompt": prompt,
                        "height": sample_size[0],
                        "width": sample_size[1],
                        "video_length": video_length,
                        "seed": item_seed,
                        "lora_path": args.lora_path,
                        "lora_weight": args.lora_weight,
                        "lora_high_path": args.lora_high_path,
                        "lora_high_weight": args.lora_high_weight,
                        "fps": args.fps,
                        "src_frame_sampling": args.src_frame_sampling,
                    },
                    mf,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"Saved: {video_out}")

    if args.lora_path is not None:
        pipeline = unmerge_lora(pipeline, args.lora_path, args.lora_weight, device=device, dtype=weight_dtype)
        if transformer_2 is not None and args.lora_high_path is not None:
            pipeline = unmerge_lora(pipeline, args.lora_high_path, args.lora_high_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")


if __name__ == "__main__":
    main()
