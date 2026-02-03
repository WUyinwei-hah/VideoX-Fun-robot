export MODEL_NAME="/gemini/code/models/Wan2.2-Fun-A14B-InP"

export PROJECT_ROOT="/gemini/code/VideoX-Fun"

export DATASET_CSV="/gemini/code/datasets/moviebench_human_final/dataset.csv"
export VIDEO_ROOT="/gemini/code/datasets/moviebench_human_final"
export VALIDATION_JSONL="/gemini/code/VideoX-Fun/examples/wan2.2/moviebench_firstframe_pair_test.jsonl"
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --mixed_precision="bf16" --num_machines 1 ${PROJECT_ROOT}/scripts/wan2.2_fun/train_lora_firstframe.py \
  --config_path="${PROJECT_ROOT}/config/wan2.2/wan_civitai_i2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_type="character_pair" \
  --train_data_meta=$DATASET_CSV \
  --video_root=$VIDEO_ROOT \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --enable_bucket \
  --token_sample_size=512 \
  --validation_steps=100000000 \
  --save_state \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=0 \
  --num_train_epochs=50000 \
  --checkpointing_steps=200 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_moviebench_firstframe_pair_14b_high" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --uniform_sampling \
  --boundary_type="high" \
  --rank=128 \
  --network_alpha=128 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --train_mode="inpaint" \
  --low_vram



# export MODEL_NAME="/gemini/code/models/Wan2.2-Fun-5B-InP"

# export PROJECT_ROOT="/gemini/code/VideoX-Fun"

# export DATASET_CSV="/gemini/code/datasets/moviebench_human_final/dataset.csv"
# export VIDEO_ROOT="/gemini/code/datasets/moviebench_human_final"

# accelerate launch --mixed_precision="bf16" --num_processes 8 --num_machines 1 ${PROJECT_ROOT}/scripts/wan2.2_fun/train_lora_firstframe.py \
#   --config_path="${PROJECT_ROOT}/config/wan2.2/wan_civitai_5b.yaml" \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_type="character_pair" \
#   --train_data_meta=$DATASET_CSV \
#   --video_root=$VIDEO_ROOT \
#   --image_sample_size=640 \
#   --validation_data_jsonl=$VALIDATION_JSONL \
#   --validation_steps=1000000 \
#   --video_sample_size=640 \
#   --enable_bucket \
#   --token_sample_size=640 \
#   --video_sample_stride=1 \
#   --video_sample_n_frames=81 \
#   --train_batch_size=4 \
#   --gradient_accumulation_steps=1 \
#   --dataloader_num_workers=0 \
#   --num_train_epochs=500 \
#   --checkpointing_steps=150 \
#   --resume_from_checkpoint="/gemini/code/VideoX-Fun/output_dir_moviebench_firstframe_pair_5b_2.2fun/checkpoint-1500" \
#   --learning_rate=1e-04 \
#   --seed=42 \
#   --output_dir="output_dir_moviebench_firstframe_pair_5b_2.2fun_from1500" \
#   --gradient_checkpointing \
#   --mixed_precision="bf16" \
#   --adam_weight_decay=3e-2 \
#   --adam_epsilon=1e-10 \
#   --vae_mini_batch=1 \
#   --max_grad_norm=0.05 \
#   --random_hw_adapt \
#   --training_with_video_token_length \
#   --uniform_sampling \
#   --boundary_type="full" \
#   --rank=128 \
#   --network_alpha=128 \
#   --target_name="q,k,v,ffn.0,ffn.2" \
#   --use_peft_lora \
#   --train_mode="inpaint" \
#   --low_vram