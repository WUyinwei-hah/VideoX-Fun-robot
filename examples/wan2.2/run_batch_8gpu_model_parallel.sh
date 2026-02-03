#!/bin/bash
# 8-GPU model-parallel batch inference (ulysses/ring) for Wan2.2 I2V

CSV_PATH="/gemini/code/FFGO-Video-Customization-main/Data/combined_first_frames/0-data.csv"
IMAGE_BASE_PATH="/gemini/code/FFGO-Video-Customization-main/Data"
SAVE_PATH="samples/batch_i2v_output_wan2.2"

LORA_LOW_PATH="/gemini/code/FFGO-Video-Customization-main/FFGO-Lora-Adapter/low"
LORA_HIGH_PATH="/gemini/code/FFGO-Video-Customization-main/FFGO-Lora-Adapter/high"

ULYSSES_DEGREE=8
RING_DEGREE=1

mkdir -p logs

echo "Starting 8-GPU MODEL-PARALLEL batch inference..."
echo "CSV: $CSV_PATH"
echo "Save path: $SAVE_PATH"

torchrun --nproc_per_node=8 /gemini/code/VideoX-Fun/examples/wan2.2/batch_predict_i2v.py \
  --distributed \
  --parallel_mode model_parallel \
  --ulysses_degree $ULYSSES_DEGREE \
  --ring_degree $RING_DEGREE \
  --csv_path "$CSV_PATH" \
  --image_base_path "$IMAGE_BASE_PATH" \
  --save_path "$SAVE_PATH" \
  ${LORA_LOW_PATH:+--lora_low_path "$LORA_LOW_PATH"} \
  ${LORA_HIGH_PATH:+--lora_high_path "$LORA_HIGH_PATH"} \
  --fsdp_dit \
  --fsdp_text_encoder \
  2>&1 | tee -a logs/model_parallel_i2v.log
