#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

export PROJECT_ROOT="/gemini/code/VideoX-Fun"
export MODEL_NAME=${MODEL_NAME:-"/gemini/code/models/Wan2.2-Fun-5B-InP"}
export LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-"/gemini/code/datasets/LIBERO-Cosmos-Policy/success_only"}

ROBOT_SUITES=(libero_90 libero_10 libero_goal libero_object libero_spatial)

python ${PROJECT_ROOT}/scripts/closed_loop_wan2.2_fun/predict_robot.py \
  --config_path="${PROJECT_ROOT}/config/wan2.2/wan_civitai_5b.yaml" \
  --model_name="$MODEL_NAME" \
  --libero_data_root="$LIBERO_DATA_ROOT" \
  --robot_suites "${ROBOT_SUITES[@]}" \
  --video_length=45 \
  --guidance_scale=6.0 \
  --num_inference_steps=50 \
  --shift=5 \
  --sampler_name=Flow \
  --weight_dtype=bf16 \
  --gpu_memory_mode=sequential_cpu_offload \
  --enable_bucket \
  --libero_first_frame_prob=0.0 \
  --eval_num_samples=16 \
  --seed=42 \
  --output_dir="${PROJECT_ROOT}/scripts/closed_loop_wan2.2_fun/samples_robot" \
  "$@"
