export MODEL_NAME="/gemini/code/models/Wan2.2-Fun-A14B-InP"
export DATASET_META_NAME="/gemini/code/datasets/moviebench_human_final/dataset.csv"
export DATASET_VIDEO_ROOT="/gemini/code/datasets/moviebench_human_final"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# accelerate launch --num_processes 1 --num_machines 1 --mixed_precision="bf16" scripts/wan2.2_fun/train_lora_pair_5b.py \
#   --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_meta=$DATASET_META_NAME \
#   --dataset_video_root=$DATASET_VIDEO_ROOT \
#   --image_sample_size=640 \
#   --video_sample_size=640 \
#   --token_sample_size=640 \
#   --video_sample_stride=1 \
#   --video_sample_n_frames=81 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --dataloader_num_workers=0 \
#   --num_train_epochs=100 \
#   --checkpointing_steps=150 \
#   --learning_rate=1e-04 \
#   --seed=42 \
#   --output_dir="output_dir_pair_5b" \
#   --gradient_checkpointing \
#   --mixed_precision="bf16" \
#   --adam_weight_decay=3e-2 \
#   --adam_epsilon=1e-10 \
#   --vae_mini_batch=1 \
#   --max_grad_norm=0.05 \
#   --random_hw_adapt \
#   --training_with_video_token_length \
#   --enable_bucket \
#   --uniform_sampling \
#   --train_mode="inpaint" \
#   --boundary_type="low" \
#   --rank=128 \
#   --network_alpha=128 \
#   --target_name="q,k,v,ffn.0,ffn.2" \
#   --use_peft_lora \
#   --low_vram \
#   --use_pair_dataset

export MODEL_NAME="/gemini/code/models/Wan2.2-Fun-5B-InP"


accelerate launch --num_processes 8 --num_machines 1 --mixed_precision="bf16" scripts/wan2.2_fun/train_lora_pair_5b.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --dataset_video_root=$DATASET_VIDEO_ROOT \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=0 \
  --num_train_epochs=10000 \
  --checkpointing_steps=200 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_pair_5b" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --train_mode="inpaint" \
  --boundary_type="full" \
  --rank=128 \
  --network_alpha=128 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --low_vram \
  --use_pair_dataset
