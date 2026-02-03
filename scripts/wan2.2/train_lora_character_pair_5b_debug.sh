export MODEL_NAME="/gemini/code/models/Wan2.2-TI2V-5B"

# Original videos (symlinked)
export VIDEO_ROOT="/gemini/code/datasets/MovieBench_ByCharacter"
# Recaptioned json root
export RECAPTION_ROOT="/gemini/code/datasets/MovieBench_ByCharacter_recaptioned"

accelerate launch --num_processes 1 --num_machines 1 --mixed_precision="bf16" scripts/wan2.2/train_lora_editing.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
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
  --token_sample_size=640 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=0 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_character_pair_5b" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --enable_bucket \
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
