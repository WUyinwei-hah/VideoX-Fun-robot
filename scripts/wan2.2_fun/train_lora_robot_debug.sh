#!/bin/bash
export MODEL_NAME="/gemini/code/models/Wan2.2-Fun-5B-InP"
export PROJECT_ROOT="/gemini/code/VideoX-Fun"

# Robot dataset paths
export ROBOT_DATA_ROOT="/gemini/code/datasets/libero/perturbed"
export ROBOT_SOURCE_DATA_ROOT="/gemini/code/datasets/libero/datasets"

accelerate launch --mixed_precision="bf16" --num_processes 1 --num_machines 1 ${PROJECT_ROOT}/scripts/wan2.2_fun/train_lora_robot.py \
  --config_path="${PROJECT_ROOT}/config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_type="robot" \
  --robot_data_root=$ROBOT_DATA_ROOT \
  --robot_source_data_root=$ROBOT_SOURCE_DATA_ROOT \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --enable_bucket \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=0 \
  --num_train_epochs=10000 \
  --checkpointing_steps=500 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_robot_libero" \
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
