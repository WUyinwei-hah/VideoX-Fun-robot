import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from decord import VideoReader
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
    Wan2_2Transformer3DModel,
    WanT5EncoderModel,
)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2TI2VPipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper,
    replace_parameters_by_name,
)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import filter_kwargs, save_videos_grid


SEP = "ad23r2 A full-frame white flash for an instant, then a hard cut to the next shot."


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


def _sample_video_frames_as_tensor(video_path: str, num_frames: int, height: int, width: int) -> torch.Tensor:
    vr = VideoReader(video_path)
    if len(vr) <= 0:
        raise ValueError(f"No frames in video: {video_path}")
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")

    if num_frames == 1:
        indices = np.array([0], dtype=int)
    else:
        indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
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
        elif sampling == "uniform":
            indices = np.linspace(0, total - 1, int(num_frames), dtype=int)
        else:
            raise ValueError(f"Unknown sampling={sampling}. Expected contiguous|uniform")

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


def _get_video_fps(video_path: str) -> Optional[float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 1e-6:
        return None
    return fps


def _get_dist_env() -> Tuple[int, int, int]:
    def _int_env(k: str, default: int) -> int:
        v = os.environ.get(k, None)
        if v is None or v == "":
            return default
        try:
            return int(v)
        except Exception:
            return default

    local_rank = _int_env("LOCAL_RANK", 0)
    rank = _int_env("RANK", 0)
    world_size = _int_env("WORLD_SIZE", 1)
    return local_rank, rank, world_size


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _setup_pipeline(
    config_path: str,
    model_name_or_path: str,
    sampler_name: str,
    shift: int,
    weight_dtype: torch.dtype,
    gpu_memory_mode: str,
    ulysses_degree: int,
    ring_degree: int,
    fsdp_dit: bool,
    fsdp_text_encoder: bool,
    compile_dit: bool,
    enable_teacache: bool,
    teacache_threshold: float,
    num_skip_start_steps: int,
    teacache_offload: bool,
    num_inference_steps: int,
    device: Optional[torch.device] = None,
    enable_model_parallel: bool = True,
):
    if device is None:
        device = set_multi_gpus_devices(ulysses_degree, ring_degree)
    config = OmegaConf.load(config_path)
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)

    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(model_name_or_path, config["transformer_additional_kwargs"].get("transformer_low_noise_model_subpath", "transformer")),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    if config["transformer_additional_kwargs"].get("transformer_combination_type", "single") == "moe":
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(model_name_or_path, config["transformer_additional_kwargs"].get("transformer_high_noise_model_subpath", "transformer")),
            transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
    else:
        transformer_2 = None

    Chosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]

    vae = Chosen_AutoencoderKL.from_pretrained(
        os.path.join(model_name_or_path, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name_or_path, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name_or_path, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    chosen_scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]

    if sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        config["scheduler_kwargs"]["shift"] = 1

    scheduler = chosen_scheduler(
        **filter_kwargs(chosen_scheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
    )

    pipeline = Wan2_2TI2VPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

    if enable_model_parallel and (ulysses_degree > 1 or ring_degree > 1):
        from functools import partial

        transformer.enable_multi_gpus_inference()
        if transformer_2 is not None:
            transformer_2.enable_multi_gpus_inference()
        if fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)
            if transformer_2 is not None:
                pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
        if fsdp_text_encoder:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.text_encoder = shard_fn(pipeline.text_encoder)

    if compile_dit:
        for i in range(len(pipeline.transformer.blocks)):
            pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
        if transformer_2 is not None:
            for i in range(len(pipeline.transformer_2.blocks)):
                pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])

    if gpu_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        if transformer_2 is not None:
            replace_parameters_by_name(transformer_2, ["modulation"], device=device)
            transformer_2.freqs = transformer_2.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif gpu_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation"], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif gpu_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation"], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)

    coefficients = get_teacache_coefficients(model_name_or_path) if enable_teacache else None
    if coefficients is not None:
        pipeline.transformer.enable_teacache(
            coefficients,
            num_inference_steps,
            teacache_threshold,
            num_skip_start_steps=num_skip_start_steps,
            offload=teacache_offload,
        )
        if transformer_2 is not None:
            pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

    return device, pipeline, config, boundary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_jsonl_path", type=str, required=True)
    parser.add_argument("--eval_num_samples", type=int, default=16)
    parser.add_argument("--eval_seed", type=int, default=42)

    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--lora_weight", type=float, default=1.0)

    parser.add_argument("--output_dir", type=str, default="examples/wan2.2/character_pair_val_outputs")

    parser.add_argument("--sampler_name", type=str, default="Flow_Unipc", choices=("Flow", "Flow_Unipc", "Flow_DPM++"))
    parser.add_argument("--shift", type=int, default=5)

    parser.add_argument("--base_resolution", type=int, default=640)
    parser.add_argument("--video_length", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)

    parser.add_argument(
        "--src_frame_sampling",
        type=str,
        default="contiguous",
        choices=("contiguous", "uniform"),
        help="How to sample the source frames. contiguous preserves original motion speed; uniform may look sped up.",
    )
    parser.add_argument(
        "--use_source_fps",
        action="store_true",
        help="If set, save output video using the source video's fps (may better match original playback speed).",
    )

    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--negative_prompt", type=str, default="")

    parser.add_argument(
        "--gpu_memory_mode",
        type=str,
        default="model_full_load",
        choices=(
            "model_full_load",
            "model_full_load_and_qfloat8",
            "model_cpu_offload",
            "model_cpu_offload_and_qfloat8",
            "sequential_cpu_offload",
        ),
    )

    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--fsdp_dit", action="store_true")
    parser.add_argument("--fsdp_text_encoder", action="store_true")
    parser.add_argument("--compile_dit", action="store_true")

    parser.add_argument(
        "--data_parallel",
        action="store_true",
        help="If launched with torchrun and WORLD_SIZE>1, shard eval items across ranks (throughput scaling).",
    )

    parser.add_argument("--enable_teacache", action="store_true")
    parser.add_argument("--teacache_threshold", type=float, default=0.10)
    parser.add_argument("--num_skip_start_steps", type=int, default=5)
    parser.add_argument("--teacache_offload", action="store_true")

    parser.add_argument("--weight_dtype", type=str, default="bf16", choices=("bf16", "fp16", "fp32"))

    args = parser.parse_args()

    local_rank, rank, world_size = _get_dist_env()
    use_data_parallel = bool(args.data_parallel) and int(world_size) > 1

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(local_rank))
        except Exception:
            pass
        explicit_device = torch.device("cuda", int(local_rank))
    else:
        explicit_device = torch.device("cpu")

    weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.weight_dtype]

    device, pipeline, config, boundary = _setup_pipeline(
        config_path=args.config_path,
        model_name_or_path=args.model_name,
        sampler_name=args.sampler_name,
        shift=int(args.shift),
        weight_dtype=weight_dtype,
        gpu_memory_mode=args.gpu_memory_mode,
        ulysses_degree=1 if use_data_parallel else int(args.ulysses_degree),
        ring_degree=1 if use_data_parallel else int(args.ring_degree),
        fsdp_dit=bool(args.fsdp_dit),
        fsdp_text_encoder=bool(args.fsdp_text_encoder),
        compile_dit=bool(args.compile_dit),
        enable_teacache=bool(args.enable_teacache),
        teacache_threshold=float(args.teacache_threshold),
        num_skip_start_steps=int(args.num_skip_start_steps),
        teacache_offload=bool(args.teacache_offload),
        num_inference_steps=int(args.num_inference_steps),
        device=explicit_device if use_data_parallel else None,
        enable_model_parallel=not use_data_parallel,
    )

    if args.lora_path:
        pipeline = merge_lora(pipeline, args.lora_path, float(args.lora_weight), device=device, dtype=weight_dtype)

    os.makedirs(args.output_dir, exist_ok=True)

    items = _read_jsonl(args.eval_jsonl_path)
    rng = random.Random(int(args.eval_seed))
    if int(args.eval_num_samples) < len(items):
        indices = rng.sample(list(range(len(items))), int(args.eval_num_samples))
        items = [items[i] for i in indices]

    enable_model_parallel = (not use_data_parallel) and (int(world_size) > 1)
    save_rank0_only = enable_model_parallel

    with torch.no_grad():
        video_length = int(args.video_length)
        video_length = (
            int((video_length - 1) // pipeline.vae.config.temporal_compression_ratio * pipeline.vae.config.temporal_compression_ratio)
            + 1
            if video_length != 1
            else 1
        )

        for i, item in enumerate(items):
            if use_data_parallel and (i % int(world_size) != int(rank)):
                continue

            source_path = item.get("source_abs")
            if not source_path or not os.path.exists(source_path):
                raise FileNotFoundError(f"Missing source video: {source_path}")

            prompt = str(item.get("prompt") or "").strip()
            if not prompt:
                src_prompt = str(item.get("refined_source_prompt", "")).strip()
                edt_prompt = str(item.get("edited_video_prompt", "")).strip()
                prompt = (src_prompt + " " + SEP + " " + edt_prompt).strip()

            spatial_ratio = int(getattr(pipeline.vae, "spatial_compression_ratio", pipeline.vae.config.spatial_compression_ratio))
            patch = getattr(pipeline.transformer.config, "patch_size", None)
            patch_h = int(patch[1]) if isinstance(patch, (list, tuple)) and len(patch) >= 3 else 16
            patch_w = int(patch[2]) if isinstance(patch, (list, tuple)) and len(patch) >= 3 else 16
            multiple_h = spatial_ratio * patch_h
            multiple_w = spatial_ratio * patch_w

            height, width = _compute_hw_from_video_keep_ar(
                source_path,
                base_resolution=int(args.base_resolution),
                multiple_h=multiple_h,
                multiple_w=multiple_w,
            )
            sample_size = [int(height), int(width)]

            input_video, input_video_mask, _ = get_video_to_video_latent(
                source_path,
                video_length=video_length,
                sample_size=sample_size,
                src_frame_sampling=str(args.src_frame_sampling),
            )
            input_video = input_video.to(device=device)
            input_video_mask = input_video_mask.to(device=device)

            item_seed = int(item.get("seed", int(args.seed) + i))
            generator = torch.Generator(device=device).manual_seed(item_seed)

            sample = pipeline(
                prompt,
                num_frames=video_length,
                negative_prompt=str(args.negative_prompt),
                height=sample_size[0],
                width=sample_size[1],
                generator=generator,
                guidance_scale=float(args.guidance_scale),
                num_inference_steps=int(args.num_inference_steps),
                boundary=boundary,
                video=input_video,
                mask_video=input_video_mask,
                shift=int(args.shift),
            ).videos

            if save_rank0_only and int(rank) != 0:
                continue

            prefix = str(i).zfill(4)
            if use_data_parallel:
                prefix = f"{prefix}_r{int(rank)}"
            video_out = os.path.join(args.output_dir, f"character_pair_val_{prefix}.mp4")
            fps_to_save = float(args.fps)
            if bool(args.use_source_fps):
                src_fps = _get_video_fps(source_path)
                if src_fps is not None:
                    fps_to_save = float(src_fps)
            save_videos_grid(sample, video_out, fps=int(round(fps_to_save)))

            meta_out = os.path.join(args.output_dir, f"character_pair_val_{prefix}.json")
            with open(meta_out, "w", encoding="utf-8") as mf:
                json.dump(
                    {
                        "index": i,
                        "source_video": source_path,
                        "edited_video": item.get("edited_abs"),
                        "character_key": item.get("character_key"),
                        "prompt": prompt,
                        "height": int(sample_size[0]),
                        "width": int(sample_size[1]),
                        "video_length": int(video_length),
                        "seed": item_seed,
                        "lora_path": args.lora_path,
                        "lora_weight": float(args.lora_weight),
                        "rank": int(rank),
                        "world_size": int(world_size),
                        "data_parallel": bool(use_data_parallel),
                        "fps": int(round(fps_to_save)),
                        "src_frame_sampling": str(args.src_frame_sampling),
                    },
                    mf,
                    ensure_ascii=False,
                    indent=2,
                )

    if args.lora_path:
        pipeline = unmerge_lora(pipeline, args.lora_path, float(args.lora_weight), device=device, dtype=weight_dtype)


if __name__ == "__main__":
    main()
