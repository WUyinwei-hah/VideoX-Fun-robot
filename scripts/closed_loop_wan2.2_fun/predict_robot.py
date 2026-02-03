import argparse
import os
import random
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.data.libero_dataset import LiberoDataset
from videox_fun.data.utils import get_random_mask_for_robot
from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import AutoTokenizer, WanT5EncoderModel, AutoencoderKLWan, AutoencoderKLWan3_8
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.models.wan_transformer3d_robot import Wan2_2Transformer3DModel
from videox_fun.pipeline import Wan2_2RobotInpaintPipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    replace_parameters_by_name,
    convert_weight_dtype_wrapper,
)
from videox_fun.utils.lora_utils import merge_lora
from videox_fun.utils.utils import filter_kwargs, save_videos_grid


def _load_robot_extra_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]):
    for _name in ["action_embedding", "proprio_embedding", "action_head", "propriohead"]:
        if not hasattr(model, _name):
            continue
        _mod = getattr(model, _name)
        prefix = f"{_name}."
        sub = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if sub:
            _mod.load_state_dict(sub, strict=False)


def _pixel_values_to_video_tensor(pixel_values: Any, height: int, width: int) -> torch.Tensor:
    if isinstance(pixel_values, np.ndarray):
        frames = torch.from_numpy(pixel_values).to(dtype=torch.float32)  # [F,H,W,3]
        frames = frames.permute(0, 3, 1, 2).contiguous()  # [F,3,H,W]
        frames = frames / 255.0
    elif torch.is_tensor(pixel_values):
        frames = pixel_values.to(dtype=torch.float32)
        if frames.dim() != 4 or frames.shape[1] != 3:
            raise ValueError(f"Expected torch pixel_values [F,3,H,W], got {tuple(frames.shape)}")
        frames = (frames * 0.5 + 0.5).clamp(0, 1)
    else:
        raise TypeError(f"Unsupported pixel_values type: {type(pixel_values)}")

    if frames.shape[-2] != height or frames.shape[-1] != width:
        frames = F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)
    video = frames.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # [1,3,F,H,W]
    return video


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default="config/wan2.2/wan_civitai_5b.yaml")
    parser.add_argument("--model_name", type=str, default="/gemini/code/models/Wan2.2-Fun-5B-InP")

    parser.add_argument("--libero_data_root", type=str, required=True)
    parser.add_argument(
        "--robot_suites",
        type=str,
        nargs="+",
        default=["libero_90", "libero_10", "libero_goal", "libero_object", "libero_spatial"],
    )
    parser.add_argument("--dataset_stats_path", type=str, default=None)
    parser.add_argument("--libero_first_frame_prob", type=float, default=0.0)
    parser.add_argument("--enable_bucket", action="store_true")

    parser.add_argument("--output_dir", type=str, default="samples/wan-robot")

    parser.add_argument("--sampler_name", type=str, default="Flow", choices=("Flow", "Flow_Unipc", "Flow_DPM++"))
    parser.add_argument("--shift", type=int, default=5)
    parser.add_argument("--video_length", type=int, default=45)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative_prompt", type=str, default="")

    parser.add_argument("--eval_num_samples", type=int, default=16)
    parser.add_argument("--eval_start_idx", type=int, default=0)

    parser.add_argument("--robot_extra_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--lora_weight", type=float, default=1.0)

    parser.add_argument(
        "--gpu_memory_mode",
        type=str,
        default="sequential_cpu_offload",
        choices=(
            "model_full_load",
            "model_full_load_and_qfloat8",
            "model_cpu_offload",
            "model_cpu_offload_and_qfloat8",
            "sequential_cpu_offload",
        ),
    )
    parser.add_argument("--weight_dtype", type=str, default="bf16", choices=("bf16", "fp16", "fp32"))

    parser.add_argument("--enable_teacache", action="store_true")
    parser.add_argument("--teacache_threshold", type=float, default=0.10)
    parser.add_argument("--num_skip_start_steps", type=int, default=5)
    parser.add_argument("--teacache_offload", action="store_true")

    args = parser.parse_args()

    weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.weight_dtype]

    os.makedirs(args.output_dir, exist_ok=True)

    device = set_multi_gpus_devices(ulysses_degree=1, ring_degree=1)
    config = OmegaConf.load(args.config_path)
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.900)

    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(
            args.model_name,
            config["transformer_additional_kwargs"].get("transformer_low_noise_model_subpath", "transformer"),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

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
    else:
        transformer_2 = None

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
    if args.sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        config["scheduler_kwargs"]["shift"] = 1
    scheduler = Chosen_Scheduler(**filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config["scheduler_kwargs"])))

    pipeline = Wan2_2RobotInpaintPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

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

    def load_lora_state_dict(lora_path: str) -> Dict[str, torch.Tensor]:
        if os.path.isdir(lora_path):
            safetensors_files = [f for f in os.listdir(lora_path) if f.endswith(".safetensors")]
            if safetensors_files:
                lora_path = os.path.join(lora_path, safetensors_files[0])
            else:
                raise FileNotFoundError(f"No .safetensors file found in {lora_path}")

        state_dict: Dict[str, torch.Tensor] = {}
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict

    if args.lora_path is not None:
        lora_state_dict = load_lora_state_dict(args.lora_path)
        pipeline = merge_lora(
            pipeline,
            args.lora_path,
            args.lora_weight,
            device=device,
            dtype=weight_dtype,
            state_dict=lora_state_dict,
        )

    if args.robot_extra_path is not None and os.path.exists(args.robot_extra_path):
        robot_sd = load_file(args.robot_extra_path, device="cpu")
        _load_robot_extra_state_dict(pipeline.transformer, robot_sd)
        if pipeline.transformer_2 is not None:
            _load_robot_extra_state_dict(pipeline.transformer_2, robot_sd)

    coefficients = get_teacache_coefficients(args.model_name) if args.enable_teacache else None
    if coefficients is not None:
        pipeline.transformer.enable_teacache(
            coefficients,
            args.num_inference_steps,
            args.teacache_threshold,
            num_skip_start_steps=args.num_skip_start_steps,
            offload=args.teacache_offload,
        )
        if pipeline.transformer_2 is not None:
            pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

    dataset = LiberoDataset(
        data_root=args.libero_data_root,
        video_sample_size=256,
        video_sample_n_frames=args.video_length,
        use_wrist_images=True,
        use_third_person_images=True,
        enable_bucket=bool(args.enable_bucket),
        suites=args.robot_suites,
        first_frame_prob=float(args.libero_first_frame_prob),
        dataset_stats_path=args.dataset_stats_path,
    )

    rng = random.Random(args.seed)
    all_indices = list(range(len(dataset)))
    if args.eval_start_idx > 0:
        all_indices = all_indices[args.eval_start_idx :]
    if args.eval_num_samples < len(all_indices):
        all_indices = rng.sample(all_indices, args.eval_num_samples)

    height = 256
    width = 512

    with torch.no_grad():
        for out_i, ds_idx in enumerate(all_indices):
            sample: Dict[str, Any] = dataset[ds_idx]

            pixel_values = sample["pixel_values"]  # np.uint8 [F,H,W,3]
            actions = sample["actions"]  # np.float32 [F,7], normalized
            proprio = sample["proprio"]  # np.float32 [F,9], normalized
            prompt = sample.get("text", "")

            video_full = _pixel_values_to_video_tensor(pixel_values, height=height, width=width)
            f = int(video_full.shape[2])

            mask = get_random_mask_for_robot((f, 3, height, width)).to(dtype=torch.float32)  # [F,1,H,W] 0/1
            mask_video = (mask * 255.0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # [1,1,F,H,W]

            chunk_len = int((f - 1) // 4) // 3 * 4 + 1
            input_video = torch.zeros_like(video_full)
            input_video[:, :, :chunk_len] = video_full[:, :, :chunk_len]

            actions_t = torch.from_numpy(actions).to(dtype=weight_dtype).unsqueeze(0)
            proprio_t = torch.from_numpy(proprio).to(dtype=weight_dtype).unsqueeze(0)

            video_full = video_full.to(device=device)
            mask_video = mask_video.to(device=device)
            input_video = input_video.to(device=device)
            actions_t = actions_t.to(device=device)
            proprio_t = proprio_t.to(device=device)

            generator = torch.Generator(device=device).manual_seed(int(args.seed + out_i))

            out = pipeline(
                prompt,
                num_frames=f,
                negative_prompt=args.negative_prompt,
                height=height,
                width=width,
                video=input_video,
                mask_video=mask_video,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                boundary=boundary,
                shift=args.shift,
                generator=generator,
                output_type="numpy",
                actions=actions_t,
                proprio=proprio_t,
            )

            prefix = str(out_i).zfill(4)
            video_out_path = os.path.join(args.output_dir, f"robot_{prefix}.mp4")
            save_videos_grid(out.videos, video_out_path, fps=args.fps)

            if out.actions is not None:
                np.save(os.path.join(args.output_dir, f"robot_{prefix}_actions.npy"), out.actions.detach().cpu().float().numpy())
            if out.proprio is not None:
                np.save(os.path.join(args.output_dir, f"robot_{prefix}_proprio.npy"), out.proprio.detach().cpu().float().numpy())


if __name__ == "__main__":
    main()
