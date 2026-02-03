import os
import sys

import json
import math
import random

import cv2
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from decord import VideoReader

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan3_8, AutoencoderKLWan, WanT5EncoderModel, AutoTokenizer,
                              Wan2_2Transformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2TI2VPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                   save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


SEP = "ad23r2 the screen suddenly turns white, then the camera view suddenly changes."


def _ceil_to_multiple(x: int, m: int) -> int:
    if m <= 1:
        return int(x)
    return int(math.ceil(x / m) * m)


def _compute_hw_from_video_keep_ar(
    video_path: str,
    base_resolution: int = 640,
    multiple_h: int = 16,
    multiple_w: int = 16,
):
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


def _sample_video_frames_as_tensor(video_path: str, num_frames: int, height: int, width: int):
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


def get_video_to_video_latent(source_video_path: str, video_length: int, sample_size):
    height, width = int(sample_size[0]), int(sample_size[1])
    if video_length < 3:
        raise ValueError("video_length must be >= 3 for source + white + edited")

    remaining = int(video_length - 1)
    n_src = int(remaining // 2)
    if n_src <= 0:
        raise ValueError("video_length too small")

    src_video = _sample_video_frames_as_tensor(source_video_path, n_src, height=height, width=width)

    input_video = torch.zeros([1, 3, video_length, height, width], dtype=torch.float32)
    input_video[:, :, :n_src] = src_video
    input_video[:, :, n_src : n_src + 1] = 1.0

    input_video_mask = torch.zeros([1, 1, video_length, height, width], dtype=torch.float32)
    input_video_mask[:, :, n_src + 1 :] = 255.0

    return input_video, input_video_mask, None


def _sample_jsonl_items(jsonl_path: str, k: int, seed: int = 0):
    rng = random.Random(seed)
    reservoir = []
    seen = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            seen += 1
            if len(reservoir) < k:
                reservoir.append(item)
            else:
                j = rng.randint(1, seen)
                if j <= k:
                    reservoir[j - 1] = item
    if len(reservoir) < k:
        raise ValueError(f"jsonl only has {len(reservoir)} items, but k={k}")
    return reservoir

# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
GPU_memory_mode     = "model_full_load"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# TeaCache config
enable_teacache     = True
# Recommended to be set between 0.05 and 0.30. A larger threshold can cache more steps, speeding up the inference process, 
# but it may cause slight differences between the generated content and the original content.
# # --------------------------------------------------------------------------------------------------- #
# | Model Name          | threshold | Model Name          | threshold |
# | Wan2.2-T2V-A14B     | 0.10~0.15 | Wan2.2-I2V-A14B     | 0.15~0.20 |
# # --------------------------------------------------------------------------------------------------- #
teacache_threshold  = 0.10
# The number of steps to skip TeaCache at the beginning of the inference process, which can
# reduce the impact of TeaCache on generated video quality.
num_skip_start_steps = 5
# Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
teacache_offload    = False

# Skip some cfg steps in inference
# Recommended to be set between 0.00 and 0.25
cfg_skip_ratio      = 0

# Riflex config
enable_riflex       = False
# Index of intrinsic frequency
riflex_k            = 6

# Config and model path
config_path         = "config/wan2.2/wan_civitai_5b.yaml"
# model path
model_name          = "/gemini/code/models/Wan2.2-TI2V-5B"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow_Unipc"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 
# Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
shift               = 5

# Load pretrained model if need
# The transformer_path is used for low noise model, the transformer_high_path is used for high noise model.
# Since Wan2.2-5b consists of only one model, only transformer_path is used.
transformer_path        = None
transformer_high_path   = None
vae_path                = None
# Load lora model if need
# The lora_path is used for low noise model, the lora_high_path is used for high noise model.
# Since Wan2.2-5b consists of only one model, only lora_path is used.
lora_path               = "/gemini/code/VideoX-Fun/output_dir_editing_5b/checkpoint-2350.safetensors"
lora_high_path          = None

# Other params
base_resolution     = 640
sample_size         = None
video_length        = 121
fps                 = 24

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16
# If you want to generate from text, please set the validation_image_start = None and validation_image_end = None
validation_image_start  = "asset/1.png"

# prompts
prompt              = "一只棕色的狗摇着头，坐在舒适房间里的浅色沙发上。在狗的后面，架子上有一幅镶框的画，周围是粉红色的花朵。房间里柔和温暖的灯光营造出舒适的氛围。"
negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
guidance_scale      = 6.0
seed                = 43
num_inference_steps = 50
# The lora_weight is used for low noise model, the lora_high_weight is used for high noise model.
lora_weight         = 1
lora_high_weight    = 1
save_path           = "samples/wan-videos-t2v-editing"

eval_jsonl_path     = "/gemini/code/VideoX-Fun/samples/wan-videos-t2v-editing/test.jsonl"
eval_num_samples    = 2
eval_seed           = 42

device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)
boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)

transformer = Wan2_2Transformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
if config['transformer_additional_kwargs'].get('transformer_combination_type', 'single') == "moe":
    transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
else:
    transformer_2 = None

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

if transformer_2 is not None:
    if transformer_high_path is not None:
        print(f"From checkpoint: {transformer_high_path}")
        if transformer_high_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(transformer_high_path)
        else:
            state_dict = torch.load(transformer_high_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer_2.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
Chosen_AutoencoderKL = {
    "AutoencoderKLWan": AutoencoderKLWan,
    "AutoencoderKLWan3_8": AutoencoderKLWan3_8
}[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
vae = Chosen_AutoencoderKL.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

# Get Scheduler
Chosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
    config['scheduler_kwargs']['shift'] = 1
scheduler = Chosen_Scheduler(
    **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline
pipeline = Wan2_2TI2VPipeline(
    transformer=transformer,
    transformer_2=transformer_2,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)
if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if transformer_2 is not None:
        transformer_2.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        if transformer_2 is not None:
            pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    if transformer_2 is not None:
        for i in range(len(pipeline.transformer_2.blocks)):
            pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])
    print("Add Compile")

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

coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )
    if transformer_2 is not None:
        pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
    if transformer_2 is not None:
        pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

if lora_path is not None:
    print("loading lora")
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
    if transformer_2 is not None:
        pipeline = merge_lora(pipeline, lora_high_path, lora_high_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")

with torch.no_grad():
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

    if enable_riflex:
        pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)
        if transformer_2 is not None:
            pipeline.transformer_2.enable_riflex(k=riflex_k, L_test=latent_frames)

    os.makedirs(save_path, exist_ok=True)
    eval_items = _sample_jsonl_items(eval_jsonl_path, eval_num_samples, seed=eval_seed)
    for i, item in enumerate(eval_items):
        source_path = item.get("source_abs")
        edited_path = item.get("edited_abs")
        src_prompt = str(item.get("refined_source_prompt", "")).strip()
        edt_prompt = str(item.get("edited_video_prompt", "")).strip()
        prompt = (src_prompt + " " + SEP + " " + edt_prompt).strip()
        plrint(prompt)

        if not source_path or not os.path.exists(source_path):
            raise FileNotFoundError(f"Missing source video: {source_path}")

        spatial_ratio = int(getattr(vae, "spatial_compression_ratio", vae.config.spatial_compression_ratio))
        patch_h = int(pipeline.transformer.config.patch_size[1])
        patch_w = int(pipeline.transformer.config.patch_size[2])
        multiple_h = spatial_ratio * patch_h
        multiple_w = spatial_ratio * patch_w
        height, width = _compute_hw_from_video_keep_ar(
            source_path,
            base_resolution=base_resolution,
            multiple_h=multiple_h,
            multiple_w=multiple_w,
        )
        sample_size = [height, width]

        input_video, input_video_mask, _ = get_video_to_video_latent(source_path, video_length=video_length, sample_size=sample_size)
        input_video = input_video.to(device=device)
        input_video_mask = input_video_mask.to(device=device)

        generator = torch.Generator(device=device).manual_seed(int(seed) + i)
        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            boundary=boundary,
            video=input_video,
            mask_video=input_video_mask,
            shift=shift,
        ).videos

        prefix = str(i).zfill(4)
        video_path = os.path.join(save_path, f"ditto_eval_{prefix}.mp4")
        save_videos_grid(sample, video_path, fps=fps)
        meta_path = os.path.join(save_path, f"ditto_eval_{prefix}.json")
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(
                {
                    "index": i,
                    "source_video": source_path,
                    "edited_video": edited_path,
                    "refined_source_prompt": src_prompt,
                    "edited_video_prompt": edt_prompt,
                    "instruction": item.get("edit_instruction", None),
                    "height": int(sample_size[0]),
                    "width": int(sample_size[1]),
                    "video_length": int(video_length),
                    "prompt": prompt,
                    "seed": int(seed) + i,
                },
                mf,
                ensure_ascii=False,
                indent=2,
            )

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
    if transformer_2 is not None:
        pipeline = unmerge_lora(pipeline, lora_high_path, lora_high_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        pass