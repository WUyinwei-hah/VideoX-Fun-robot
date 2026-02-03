#!/bin/bash
# Run batch I2V inference across 8 GPUs in parallel
# Each GPU processes a different subset of the data

CSV_PATH="/gemini/code/FFGO-Video-Customization-main/Data/combined_first_frames/0-data.csv"
IMAGE_BASE_PATH="/gemini/code/FFGO-Video-Customization-main/Data"
SAVE_PATH="samples/batch_i2v_output"
LORA_LOW_PATH="/gemini/code/VideoX-Fun/output_dir_low_128/checkpoint-800.safetensors"
LORA_HIGH_PATH="/gemini/code/VideoX-Fun/output_dir_high_128/checkpoint-800.safetensors"
NUM_GPUS=8

echo "Starting 8-GPU batch inference..."
echo "CSV: $CSV_PATH"
echo "Save path: $SAVE_PATH"

# Create logs directory
mkdir -p logs

# Launch 8 parallel processes, one per GPU
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    echo "Launching GPU $gpu_id..."
    CUDA_VISIBLE_DEVICES=$gpu_id python /gemini/code/VideoX-Fun/examples/wan2.2_fun/batch_predict_i2v.py \
        --csv_path "$CSV_PATH" \
        --image_base_path "$IMAGE_BASE_PATH" \
        --save_path "$SAVE_PATH" \
        --gpu_id $gpu_id \
        --num_gpus $NUM_GPUS \
        --lora_low_path "$LORA_LOW_PATH" \
        --lora_high_path "$LORA_HIGH_PATH" \
        --ulysses_degree 1 \
        --ring_degree 1 \
        2>&1 | tee -a "logs/gpu_${gpu_id}.log" &
done

echo "All processes launched. Waiting for completion..."
wait

echo "Batch inference completed!"
