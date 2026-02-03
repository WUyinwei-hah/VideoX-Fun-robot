export MODEL_NAME="/gemini/code/models/Wan2.2-TI2V-5B"

export PROJECT_ROOT="/gemini/code/VideoX-Fun"

# Original videos (symlinked)
export VIDEO_ROOT="/gemini/code/datasets/MovieBench_ByCharacter"
# Recaptioned json root
export RECAPTION_ROOT="/gemini/code/datasets/MovieBench_ByCharacter_recaptioned"

accelerate launch --mixed_precision="bf16" ${PROJECT_ROOT}/scripts/wan2.2/train_lora_editing.py \
  --config_path="${PROJECT_ROOT}/config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_type="character_pair" \
  --train_data_dir=$VIDEO_ROOT \
  --recaption_root=$RECAPTION_ROOT \
  --video_root=$VIDEO_ROOT \
  --recaption_key="Recaption Prompt" \
  --pairing_strategy="random" \
  --max_pair_candidates=64 \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --enable_bucket \
  --token_sample_size=640 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=0 \
  --num_train_epochs=500 \
  --checkpointing_steps=150 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_character_pair_5b_filtered_pair" \
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
  --train_mode="ti2v" \
  --low_vram
