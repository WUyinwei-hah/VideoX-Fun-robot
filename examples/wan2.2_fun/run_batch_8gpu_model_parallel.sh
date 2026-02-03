#!/bin/bash
# 8-GPU model-parallel batch inference (ulysses/ring) for Wan2.2 Fun I2V
# This runs a SINGLE job sharded across 8 GPUs, similar to predict_i2v.py (ulysses_degree=8, ring_degree=1).

CSV_PATH="/gemini/code/FFGO-Video-Customization-main/Data/combined_first_frames/0-data.csv"
IMAGE_BASE_PATH="/gemini/code/FFGO-Video-Customization-main/Data"
SAVE_PATH="samples/batch_i2v_output_128"

LORA_LOW_PATH="/gemini/code/VideoX-Fun/output_dir_low_128/checkpoint-800.safetensors"
LORA_HIGH_PATH="/gemini/code/VideoX-Fun/output_dir_high_128/checkpoint-800.safetensors"

ULYSSES_DEGREE=8
RING_DEGREE=1

echo "Starting 8-GPU MODEL-PARALLEL batch inference..."
echo "CSV: $CSV_PATH"
echo "Save path: $SAVE_PATH"
echo "LoRA low: $LORA_LOW_PATH"
echo "LoRA high: $LORA_HIGH_PATH"

mkdir -p logs

# Important:
# - DO NOT set CUDA_VISIBLE_DEVICES per rank here.
# - torchrun will spawn 8 processes and manage LOCAL_RANK/RANK/WORLD_SIZE.

torchrun --nproc_per_node=8 /gemini/code/VideoX-Fun/examples/wan2.2_fun/batch_predict_i2v.py \
  --distributed \
  --parallel_mode model_parallel \
  --ulysses_degree $ULYSSES_DEGREE \
  --ring_degree $RING_DEGREE \
  --csv_path "$CSV_PATH" \
  --image_base_path "$IMAGE_BASE_PATH" \
  --save_path "$SAVE_PATH" \
  --lora_low_path "$LORA_LOW_PATH" \
  --lora_high_path "$LORA_HIGH_PATH" \
  --fsdp_dit \
  --fsdp_text_encoder \
  2>&1 | tee -a logs/model_parallel.log
