#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
export MODEL_NAME="/gemini/code/models/Wan2.2-Fun-5B-InP"
export PROJECT_ROOT="/gemini/code/VideoX-Fun"

# Robot dataset paths
export ROBOT_DATA_ROOT="/gemini/code/datasets/libero/perturbed"
export LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-"/gemini/code/datasets/LIBERO-Cosmos-Policy/success_only"}
ROBOT_SUITES=(libero_90 libero_10 libero_goal libero_object libero_spatial)

# Avoid writing caches/temp files to unexpected locations with quotas
# export HF_HOME="/gemini/code/.cache/huggingface"
# export TRANSFORMERS_CACHE="/gemini/code/.cache/huggingface/transformers"
# export DIFFUSERS_CACHE="/gemini/code/.cache/huggingface/diffusers"
# export TMPDIR="/gemini/code/.tmp"
# mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$DIFFUSERS_CACHE" "$TMPDIR"

# accelerate launch --mixed_precision="bf16" --num_processes 8 --num_machines 1 ${PROJECT_ROOT}/scripts/closed_loop_wan2.2_fun/train_lora_robot.py \
accelerate launch --mixed_precision="bf16" --num_processes 8 --num_machines 1 ${PROJECT_ROOT}/scripts/closed_loop_wan2.2_fun/train_lora_robot.py \
  --config_path="${PROJECT_ROOT}/config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_type="robot" \
  --robot_data_root=$ROBOT_DATA_ROOT \
  --robot_suites "${ROBOT_SUITES[@]}" \
  --libero_data_root=$LIBERO_DATA_ROOT \
  --image_sample_size=256 \
  --video_sample_size=256 \
  --enable_bucket \
  --token_sample_size=256 \
  --video_sample_stride=1 \
  --video_sample_n_frames=45 \
  --train_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=0 \
  --num_train_epochs=10000 \
  --checkpointing_steps=500 \
  --save_state \
  --resume_from_checkpoint="latest" \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="/gemini/code/VideoX-Fun/scripts/closed_loop_wan2.2_fun/output_dir_robot_libero" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --uniform_sampling \
  --boundary_type="full" \
  --rank=128 \
  --network_alpha=128 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --train_mode="inpaint" \
  --low_vram
