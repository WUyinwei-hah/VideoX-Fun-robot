import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from decord import VideoReader
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

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
from videox_fun.pipeline import Wan2_2I2VPipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper,
    replace_parameters_by_name,
)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import filter_kwargs, get_image_to_video_latent, save_videos_grid


SEP = "ad23r2 the screen suddenly turns white, then the camera view suddenly changes."


EVAL_JSONL_PATH = "/gemini/code/VideoX-Fun/examples/wan2.2/moviebench_character_pair_val.jsonl"
EVAL_NUM_SAMPLES = 16
EVAL_SEED = 2025

CONFIG_PATH = "config/wan2.2/wan_civitai_i2v.yaml"
MODEL_NAME = "/gemini/code/models/Wan2.2-I2V-A14B"

LORA_LOW_PATH = "/gemini/code/VideoX-Fun/output_dir_character_pair_low_filtered_pair/checkpoint-1500.safetensors"
LORA_HIGH_PATH = "/gemini/code/VideoX-Fun/output_dir_character_pair_high_filtered_pair/checkpoint-1500.safetensors"
LORA_LOW_WEIGHT = 1.0
LORA_HIGH_WEIGHT = 1.0

OUTPUT_DIR = "examples/wan2.2/character_pair_14b_i2v_outputs_val"

GPU_MEMORY_MODE = "model_full_load"
ULYSSES_DEGREE = 8
RING_DEGREE = 1
FSDP_DIT = False
FSDP_TEXT_ENCODER = False
COMPILE_DIT = False

DATA_PARALLEL = False

ENABLE_TEACACHE = False
TEACACHE_THRESHOLD = 0.15
NUM_SKIP_START_STEPS = 5
TEACACHE_OFFLOAD = False

WEIGHT_DTYPE = "bf16"

BASE_RESOLUTION = 640
MAX_LONG_SIDE = 640
MAX_TOTAL_PIXELS = 640 * 640
VIDEO_LENGTH = 81
FPS = 16

USE_SOURCE_VIDEO_AS_COND = True
SRC_FRAME_SAMPLING = "contiguous"

COND_VIDEO_START_SEC = 0.0
COND_VIDEO_END_SEC: Optional[float] = None
COND_TARGET_FPS: Optional[float] = None
COND_MAX_FRAMES: Optional[int] = None

GUIDANCE_SCALE = 6.0
NUM_INFERENCE_STEPS = 50
SEED = 43
NEGATIVE_PROMPT = ""

SAMPLER_NAME = "Flow_Unipc"
SHIFT = 5

SAVE_EXTRACTED_FRAMES = True


def _get_dist_env() -> Tuple[int, int, int]:
    def _safe_int(v: Optional[str], default: int) -> int:
        if v is None or v == "":
            return default
        try:
            return int(v)
        except Exception:
            return default

    local_rank = _safe_int(os.environ.get("LOCAL_RANK", "0"), 0)
    rank = _safe_int(os.environ.get("RANK", "0"), 0)
    world_size = _safe_int(os.environ.get("WORLD_SIZE", "1"), 1)
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


def _ceil_to_multiple(x: int, m: int) -> int:
    if m <= 1:
        return int(x)
    return int(math.ceil(x / m) * m)


def _floor_to_multiple(x: int, m: int) -> int:
    if m <= 1:
        return int(x)
    return int(math.floor(x / m) * m)


def _compute_hw_from_image_keep_ar(
    w: int,
    h: int,
    base_resolution: int,
    multiple_h: int,
    multiple_w: int,
) -> Tuple[int, int]:
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {w}x{h}")
    short_side = min(w, h)
    ratio = float(base_resolution) / float(short_side)
    out_w = max(1, int(round(w * ratio)))
    out_h = max(1, int(round(h * ratio)))

    if MAX_LONG_SIDE is not None and int(MAX_LONG_SIDE) > 0:
        long_side = max(out_w, out_h)
        if long_side > int(MAX_LONG_SIDE):
            s = float(int(MAX_LONG_SIDE)) / float(long_side)
            out_w = max(1, int(math.floor(out_w * s)))
            out_h = max(1, int(math.floor(out_h * s)))

    if MAX_TOTAL_PIXELS is not None and int(MAX_TOTAL_PIXELS) > 0:
        total = int(out_w) * int(out_h)
        if total > int(MAX_TOTAL_PIXELS):
            s = math.sqrt(float(int(MAX_TOTAL_PIXELS)) / float(total))
            out_w = max(1, int(math.floor(out_w * s)))
            out_h = max(1, int(math.floor(out_h * s)))

    out_h = max(multiple_h, _floor_to_multiple(out_h, multiple_h))
    out_w = max(multiple_w, _floor_to_multiple(out_w, multiple_w))
    return int(out_h), int(out_w)


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
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.900)

    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(
            model_name_or_path,
            config["transformer_additional_kwargs"].get("transformer_low_noise_model_subpath", "transformer"),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    if config["transformer_additional_kwargs"].get("transformer_combination_type", "single") == "moe":
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(
                model_name_or_path,
                config["transformer_additional_kwargs"].get("transformer_high_noise_model_subpath", "transformer"),
            ),
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
    text_encoder = text_encoder.eval()

    chosen_scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]

    if sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        config["scheduler_kwargs"]["shift"] = 1

    scheduler = chosen_scheduler(**filter_kwargs(chosen_scheduler, OmegaConf.to_container(config["scheduler_kwargs"])))

    pipeline = Wan2_2I2VPipeline(
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
    local_rank, rank, world_size = _get_dist_env()
    use_data_parallel = bool(DATA_PARALLEL) and int(world_size) > 1

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(local_rank))
        except Exception:
            pass
        explicit_device = torch.device("cuda", int(local_rank))
    else:
        explicit_device = torch.device("cpu")

    weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[str(WEIGHT_DTYPE)]

    device, pipeline, _config, boundary = _setup_pipeline(
        config_path=CONFIG_PATH,
        model_name_or_path=MODEL_NAME,
        sampler_name=SAMPLER_NAME,
        shift=int(SHIFT),
        weight_dtype=weight_dtype,
        gpu_memory_mode=str(GPU_MEMORY_MODE),
        ulysses_degree=1 if use_data_parallel else int(ULYSSES_DEGREE),
        ring_degree=1 if use_data_parallel else int(RING_DEGREE),
        fsdp_dit=bool(FSDP_DIT),
        fsdp_text_encoder=bool(FSDP_TEXT_ENCODER),
        compile_dit=bool(COMPILE_DIT),
        enable_teacache=bool(ENABLE_TEACACHE),
        teacache_threshold=float(TEACACHE_THRESHOLD),
        num_skip_start_steps=int(NUM_SKIP_START_STEPS),
        teacache_offload=bool(TEACACHE_OFFLOAD),
        num_inference_steps=int(NUM_INFERENCE_STEPS),
        device=explicit_device if use_data_parallel else None,
        enable_model_parallel=not use_data_parallel,
    )

    if LORA_LOW_PATH:
        pipeline = merge_lora(
            pipeline,
            LORA_LOW_PATH,
            float(LORA_LOW_WEIGHT),
            device=device,
            dtype=weight_dtype,
        )
    if LORA_HIGH_PATH and getattr(pipeline, "transformer_2", None) is not None:
        pipeline = merge_lora(
            pipeline,
            LORA_HIGH_PATH,
            float(LORA_HIGH_WEIGHT),
            device=device,
            dtype=weight_dtype,
            sub_transformer_name="transformer_2",
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    items = _read_jsonl(EVAL_JSONL_PATH)
    rng = random.Random(int(EVAL_SEED))
    if int(EVAL_NUM_SAMPLES) < len(items):
        indices = rng.sample(list(range(len(items))), int(EVAL_NUM_SAMPLES))
        items = [items[i] for i in indices]

    save_rank0_only = (not use_data_parallel) and (int(world_size) > 1)

    with torch.no_grad():
        video_length = int(VIDEO_LENGTH)
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

            prefix = str(i).zfill(4)
            if use_data_parallel:
                prefix = f"{prefix}_r{int(rank)}"

            cap = cv2.VideoCapture(source_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Failed to open video: {source_path}")
            w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            if w0 <= 0 or h0 <= 0:
                _, h0, w0 = _extract_first_frame_as_pil(source_path)

            height, width = _compute_hw_from_image_keep_ar(
                w=w0,
                h=h0,
                base_resolution=int(BASE_RESOLUTION),
                multiple_h=multiple_h,
                multiple_w=multiple_w,
            )
            sample_size = [int(height), int(width)]

            if bool(SAVE_EXTRACTED_FRAMES) and int(rank) == 0:
                frames_dir = os.path.join(OUTPUT_DIR, "_extracted_frames")
                os.makedirs(frames_dir, exist_ok=True)
                frame_path = os.path.join(frames_dir, f"character_i2v_{prefix}.png")
                try:
                    frame_pil, _, _ = _extract_first_frame_as_pil(source_path)
                    frame_pil.save(frame_path)
                except Exception:
                    pass

            if bool(USE_SOURCE_VIDEO_AS_COND):
                input_video, input_video_mask, _clip_image = get_video_to_video_latent(
                    source_path,
                    video_length=int(video_length),
                    sample_size=sample_size,
                    src_frame_sampling=str(SRC_FRAME_SAMPLING),
                )
            else:
                frame_pil, _, _ = _extract_first_frame_as_pil(source_path)
                input_video, input_video_mask, _clip_image = get_image_to_video_latent(
                    [frame_pil], None, video_length=video_length, sample_size=sample_size
                )

            input_video = input_video.to(device=device)
            input_video_mask = input_video_mask.to(device=device)

            item_seed = int(item.get("seed", int(SEED) + i))
            generator = torch.Generator(device=device).manual_seed(item_seed)

            sample = pipeline(
                prompt,
                num_frames=video_length,
                negative_prompt=str(NEGATIVE_PROMPT),
                height=sample_size[0],
                width=sample_size[1],
                generator=generator,
                guidance_scale=float(GUIDANCE_SCALE),
                num_inference_steps=int(NUM_INFERENCE_STEPS),
                boundary=boundary,
                video=input_video,
                mask_video=input_video_mask,
                shift=int(SHIFT),
            ).videos

            if save_rank0_only and int(rank) != 0:
                continue

            video_out = os.path.join(OUTPUT_DIR, f"character_i2v_{prefix}.mp4")
            save_videos_grid(sample, video_out, fps=int(FPS))

            meta_out = os.path.join(OUTPUT_DIR, f"character_i2v_{prefix}.json")
            with open(meta_out, "w", encoding="utf-8") as mf:
                json.dump(
                    {
                        "index": i,
                        "source_video": source_path,
                        "character_key": item.get("character_key"),
                        "prompt": prompt,
                        "height": int(sample_size[0]),
                        "width": int(sample_size[1]),
                        "video_length": int(video_length),
                        "seed": item_seed,
                        "lora_low_path": LORA_LOW_PATH,
                        "lora_low_weight": float(LORA_LOW_WEIGHT),
                        "lora_high_path": LORA_HIGH_PATH,
                        "lora_high_weight": float(LORA_HIGH_WEIGHT),
                        "rank": int(rank),
                        "world_size": int(world_size),
                        "data_parallel": bool(use_data_parallel),
                        "fps": int(FPS),
                        "base_resolution": int(BASE_RESOLUTION),
                        "use_source_video_as_cond": bool(USE_SOURCE_VIDEO_AS_COND),
                        "src_frame_sampling": str(SRC_FRAME_SAMPLING),
                    },
                    mf,
                    ensure_ascii=False,
                    indent=2,
                )

    if LORA_HIGH_PATH and getattr(pipeline, "transformer_2", None) is not None:
        pipeline = unmerge_lora(
            pipeline,
            LORA_HIGH_PATH,
            float(LORA_HIGH_WEIGHT),
            device=device,
            dtype=weight_dtype,
            sub_transformer_name="transformer_2",
        )
    if LORA_LOW_PATH:
        pipeline = unmerge_lora(
            pipeline,
            LORA_LOW_PATH,
            float(LORA_LOW_WEIGHT),
            device=device,
            dtype=weight_dtype,
        )


if __name__ == "__main__":
    import debugpy

    # debugpy.listen(("127.0.0.1", 5678))  # 只监听本机，最安全
    # print("debugpy is listening on 127.0.0.1:5678")
    # debugpy.wait_for_client()            # 等 VSCode 连接后再继续
    # debugpy.breakpoint() 
    main()
