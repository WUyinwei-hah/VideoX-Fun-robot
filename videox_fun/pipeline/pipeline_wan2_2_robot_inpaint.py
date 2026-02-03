import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange

from ..models import AutoencoderKLWan, AutoTokenizer, WanT5EncoderModel, Wan2_2Transformer3DModel
from ..utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom timesteps."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom sigmas."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )

        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )
    return resized_mask


@dataclass
class WanRobotPipelineOutput(BaseOutput):
    videos: torch.Tensor
    actions: Optional[torch.Tensor] = None
    proprio: Optional[torch.Tensor] = None


class Wan2_2RobotInpaintPipeline(DiffusionPipeline):
    _optional_components = ["transformer_2"]
    model_cpu_offload_seq = "text_encoder->transformer_2->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: Wan2_2Transformer3DModel,
        transformer_2: Wan2_2Transformer3DModel = None,
        scheduler: FlowMatchEulerDiscreteScheduler = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            transformer_2=transformer_2,
            scheduler=scheduler,
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.spatial_compression_ratio,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance, noise_aug_strength):
        if mask is not None:
            mask = mask.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask = []
            for i in range(0, mask.shape[0], bs):
                mask_bs = mask[i : i + bs]
                mask_bs = self.vae.encode(mask_bs)[0]
                mask_bs = mask_bs.mode()
                new_mask.append(mask_bs)
            mask = torch.cat(new_mask, dim=0)

        if masked_image is not None:
            masked_image = masked_image.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask_pixel_values = []
            for i in range(0, masked_image.shape[0], bs):
                mask_pixel_values_bs = masked_image[i : i + bs]
                mask_pixel_values_bs = self.vae.encode(mask_pixel_values_bs)[0]
                mask_pixel_values_bs = mask_pixel_values_bs.mode()
                new_mask_pixel_values.append(mask_pixel_values_bs)
            masked_image_latents = torch.cat(new_mask_pixel_values, dim=0)
        else:
            masked_image_latents = None

        return mask, masked_image_latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        frames = frames.cpu().float().numpy()
        return frames

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @staticmethod
    def _split_by_mask(mask_video: torch.Tensor, actions: Optional[torch.Tensor], proprio: Optional[torch.Tensor]):
        split_idx = None
        if mask_video is not None:
            if mask_video.dim() != 5:
                raise ValueError(f"mask_video must be 5D [B,1,F,H,W], got {mask_video.shape}")
            m = mask_video[:, 0, :, 0, 0]
            split_idx_list = []
            for bi in range(m.size(0)):
                nz = (m[bi] > 0).nonzero(as_tuple=False)
                split_idx_list.append(int(nz[0].item()) if nz.numel() > 0 else int(m.size(1)))
            if len(split_idx_list) > 0:
                split_idx = split_idx_list[0]

        actions_cond, actions_target = actions, None
        proprio_cond, proprio_target = proprio, None
        if split_idx is not None:
            if actions is not None:
                actions_cond = actions[:, :split_idx]
                actions_target = actions[:, split_idx:]
            if proprio is not None:
                proprio_cond = proprio[:, :split_idx]
                proprio_target = proprio[:, split_idx:]

        return split_idx, actions_cond, actions_target, proprio_cond, proprio_target

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        video: Union[torch.FloatTensor] = None,
        mask_video: Union[torch.FloatTensor] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        boundary: float = 0.875,
        comfyui_progressbar: bool = False,
        shift: int = 5,
        actions: Optional[torch.FloatTensor] = None,
        proprio: Optional[torch.FloatTensor] = None,
    ) -> Union[WanRobotPipelineOutput, Tuple]:
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        if do_classifier_free_guidance:
            in_prompt_embeds = negative_prompt_embeds + prompt_embeds
        else:
            in_prompt_embeds = prompt_embeds

        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, mu=1)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(self.scheduler, device=device, sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        if video is not None:
            video_length = video.shape[2]
            init_video = self.image_processor.preprocess(rearrange(video, "b c f h w -> (b f) c h w"), height=height, width=width)
            init_video = init_video.to(dtype=torch.float32)
            init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)
        else:
            init_video = None

        latent_channels = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )

        mask = None
        if init_video is not None and mask_video is not None:
            if (mask_video == 255).all():
                mask_latents = torch.tile(torch.zeros_like(latents)[:, :1].to(device, weight_dtype), [1, 4, 1, 1, 1])
                masked_video_latents = torch.zeros_like(latents).to(device, weight_dtype)
                if self.vae.spatial_compression_ratio >= 16:
                    mask = torch.ones_like(latents).to(device, weight_dtype)[:, :1].to(device, weight_dtype)
            else:
                bs, _, video_length, h, w = video.size()
                mask_condition = self.mask_processor.preprocess(
                    rearrange(mask_video, "b c f h w -> (b f) c h w"), height=height, width=width
                )
                mask_condition = mask_condition.to(dtype=torch.float32)
                mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length)

                masked_video = init_video * (torch.tile(mask_condition, [1, 3, 1, 1, 1]) < 0.5)
                _, masked_video_latents = self.prepare_mask_latents(
                    None,
                    masked_video,
                    batch_size,
                    height,
                    width,
                    weight_dtype,
                    device,
                    generator,
                    do_classifier_free_guidance,
                    noise_aug_strength=None,
                )

                mask_condition = torch.concat(
                    [torch.repeat_interleave(mask_condition[:, :, 0:1], repeats=4, dim=2), mask_condition[:, :, 1:]],
                    dim=2,
                )
                mask_condition = mask_condition.view(bs, mask_condition.shape[2] // 4, 4, h, w)
                mask_condition = mask_condition.transpose(1, 2)
                mask_latents = resize_mask(1 - mask_condition, masked_video_latents, True).to(device, weight_dtype)

                if self.vae.spatial_compression_ratio >= 16:
                    mask = F.interpolate(mask_condition[:, :1], size=latents.size()[-3:], mode="trilinear", align_corners=True).to(device, weight_dtype)
                    if not mask[:, :, 0, :, :].any():
                        mask[:, :, 1:, :, :] = 1
                        latents = (1 - mask) * masked_video_latents + mask * latents
        else:
            mask_latents = None
            masked_video_latents = None

        split_idx, actions_cond, actions_target, proprio_cond, proprio_target = self._split_by_mask(mask_video, actions, proprio)

        actions_latents = None
        proprio_latents = None
        if actions_target is not None:
            actions_latents = randn_tensor(actions_target.shape, generator=generator, device=device, dtype=weight_dtype)
        if proprio_target is not None:
            proprio_latents = randn_tensor(proprio_target[:, -1:].shape, generator=generator, device=device, dtype=weight_dtype)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (
            self.vae.latent_channels,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            width // self.vae.spatial_compression_ratio,
            height // self.vae.spatial_compression_ratio,
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2])
            * target_shape[1]
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.transformer.current_steps = i

                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                y = None
                if init_video is not None and mask_latents is not None and masked_video_latents is not None:
                    mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                    masked_video_latents_input = (
                        torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                    )
                    y = torch.cat([mask_input, masked_video_latents_input], dim=1).to(device, weight_dtype)

                if self.vae.spatial_compression_ratio >= 16 and init_video is not None and mask is not None:
                    temp_ts = ((mask[0][0][:, ::2, ::2]) * t).flatten()
                    temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * t])
                    temp_ts = temp_ts.unsqueeze(0)
                    timestep = temp_ts.expand(latent_model_input.shape[0], temp_ts.size(1))
                else:
                    timestep = t.expand(latent_model_input.shape[0])

                if timestep.dim() == 2:
                    bsz = timestep.size(0)
                    orig_timesteps = timestep[:, -1:]
                    extra_ts_list = []
                    if actions_cond is not None:
                        extra_ts_list.append(timestep.new_zeros(bsz, actions_cond.size(1)))
                    if actions_target is not None:
                        extra_ts_list.append(orig_timesteps.expand(-1, actions_target.size(1)))
                    if proprio_cond is not None:
                        extra_ts_list.append(timestep.new_zeros(bsz, 1))
                    if proprio_target is not None:
                        extra_ts_list.append(orig_timesteps.expand(-1, 1))
                    if extra_ts_list:
                        extra_ts = torch.cat(extra_ts_list, dim=1)
                        timestep = torch.cat([timestep, extra_ts], dim=1)

                if self.transformer_2 is not None:
                    if t >= boundary * self.scheduler.config.num_train_timesteps:
                        local_transformer = self.transformer_2
                    else:
                        local_transformer = self.transformer
                else:
                    local_transformer = self.transformer

                actions_in = None
                proprio_in = None
                if actions_cond is not None or actions_latents is not None:
                    if do_classifier_free_guidance:
                        zeros_cond = torch.zeros_like(actions_cond) if actions_cond is not None else None
                        zeros_target = torch.zeros_like(actions_latents) if actions_latents is not None else None
                        actions_cond_cfg = torch.cat([zeros_cond, actions_cond], dim=0) if actions_cond is not None else None
                        actions_target_cfg = torch.cat([zeros_target, actions_latents], dim=0) if actions_latents is not None else None
                        actions_in = (actions_cond_cfg, actions_target_cfg)
                    else:
                        actions_in = (actions_cond, actions_latents)

                if proprio_cond is not None or proprio_latents is not None:
                    if do_classifier_free_guidance:
                        zeros_cond = torch.zeros_like(proprio_cond) if proprio_cond is not None else None
                        zeros_target = torch.zeros_like(proprio_latents) if proprio_latents is not None else None
                        proprio_cond_cfg = torch.cat([zeros_cond, proprio_cond], dim=0) if proprio_cond is not None else None
                        proprio_target_cfg = torch.cat([zeros_target, proprio_latents], dim=0) if proprio_latents is not None else None
                        proprio_in = (proprio_cond_cfg, proprio_target_cfg)
                    else:
                        proprio_in = (proprio_cond, proprio_latents)

                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                    noise_pred, action_pred, proprio_pred = local_transformer(
                        x=latent_model_input,
                        context=in_prompt_embeds,
                        t=timestep,
                        seq_len=seq_len,
                        y=y,
                        actions=actions_in,
                        proprio=proprio_in,
                    )

                if do_classifier_free_guidance:
                    if self.transformer_2 is not None and (isinstance(self.guidance_scale, (list, tuple))):
                        sample_guide_scale = (
                            self.guidance_scale[1]
                            if t >= self.transformer_2.config.boundary * self.scheduler.config.num_train_timesteps
                            else self.guidance_scale[0]
                        )
                    else:
                        sample_guide_scale = self.guidance_scale
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_text - noise_pred_uncond)

                    if action_pred is not None:
                        action_pred_uncond, action_pred_text = action_pred.chunk(2)
                        action_pred = action_pred_uncond + sample_guide_scale * (action_pred_text - action_pred_uncond)

                    if proprio_pred is not None:
                        proprio_pred_uncond, proprio_pred_text = proprio_pred.chunk(2)
                        proprio_pred = proprio_pred_uncond + sample_guide_scale * (proprio_pred_text - proprio_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if actions_latents is not None and action_pred is not None:
                    actions_latents = self.scheduler.step(
                        action_pred, t, actions_latents, **extra_step_kwargs, return_dict=False
                    )[0]

                if proprio_latents is not None and proprio_pred is not None:
                    proprio_latents = self.scheduler.step(
                        proprio_pred, t, proprio_latents, **extra_step_kwargs, return_dict=False
                    )[0]

                if self.vae.spatial_compression_ratio >= 16 and mask is not None and not mask[:, :, 0, :, :].any():
                    latents = (1 - mask) * masked_video_latents + mask * latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "numpy":
            video_out = self.decode_latents(latents)
        elif not output_type == "latent":
            video_out = self.decode_latents(latents)
            video_out = self.video_processor.postprocess_video(video=video_out, output_type=output_type)
        else:
            video_out = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            video_out = torch.from_numpy(video_out) if isinstance(video_out, np.ndarray) else video_out

        actions_out = None
        proprio_out = None
        if actions_cond is not None or actions_latents is not None:
            if actions_cond is None:
                actions_out = actions_latents
            elif actions_latents is None:
                actions_out = actions_cond
            else:
                actions_out = torch.cat([actions_cond, actions_latents], dim=1)
        if proprio_cond is not None or proprio_latents is not None:
            if proprio_cond is None:
                proprio_out = proprio_latents
            elif proprio_latents is None:
                proprio_out = proprio_cond
            else:
                # model only predicts last timestep for proprio target, keep full cond + predicted last
                proprio_out = torch.cat([proprio_cond, proprio_latents], dim=1)

        return WanRobotPipelineOutput(videos=video_out, actions=actions_out, proprio=proprio_out)
