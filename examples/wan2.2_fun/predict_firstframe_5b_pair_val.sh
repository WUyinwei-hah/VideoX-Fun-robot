NUM_GPUS=${NUM_GPUS:-1}
ULYSSES_DEGREE=${ULYSSES_DEGREE:-1}
RING_DEGREE=${RING_DEGREE:-1}

EVAL_JSONL_PATH=${EVAL_JSONL_PATH:-/gemini/code/VideoX-Fun/examples/wan2.2_fun/wan_i2v_staircase_test_prompts.jsonl}
EVAL_NUM_SAMPLES=${EVAL_NUM_SAMPLES:-100}
EVAL_SEED=${EVAL_SEED:-42}

CONFIG_PATH=${CONFIG_PATH:-/gemini/code/VideoX-Fun/config/wan2.2/wan_civitai_5b.yaml}
MODEL_NAME=${MODEL_NAME:-/gemini/code/models/Wan2.2-Fun-5B-InP/}
LORA_PATH=${LORA_PATH:-/gemini/code/VideoX-Fun/output_dir_moviebench_firstframe_pair_5b_whithout_white_frame/checkpoint-4500.safetensors}
LORA_WEIGHT=${LORA_WEIGHT:-1}

OUTPUT_DIR=${OUTPUT_DIR:-/gemini/code/VideoX-Fun/examples/wan2.2_fun/firstframe_pair_val_outputs_2.2fun_5b_without_white_frame_5000}
VIDEO_LENGTH=${VIDEO_LENGTH:-81}
BASE_RESOLUTION=${BASE_RESOLUTION:-640}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-6.0}
SAMPLER_NAME=${SAMPLER_NAME:-Flow}
SHIFT=${SHIFT:-5}
WEIGHT_DTYPE=${WEIGHT_DTYPE:-bf16}

RUNNER="python"
DIST_ARGS=""
if [ "${NUM_GPUS}" -gt 1 ]; then
  if [ $((ULYSSES_DEGREE * RING_DEGREE)) -ne "${NUM_GPUS}" ]; then
    echo "ERROR: ULYSSES_DEGREE(${ULYSSES_DEGREE}) * RING_DEGREE(${RING_DEGREE}) must equal NUM_GPUS(${NUM_GPUS})." 1>&2
    exit 1
  fi
  RUNNER="torchrun --nproc-per-node=${NUM_GPUS}"
  DIST_ARGS="--ulysses_degree ${ULYSSES_DEGREE} --ring_degree ${RING_DEGREE}"
fi

if [ -n "${LORA_PATH}" ]; then
  ${RUNNER} /gemini/code/VideoX-Fun/examples/wan2.2_fun/predict_i2v_5b_firstframe_pair_val.py \
    --eval_jsonl_path "${EVAL_JSONL_PATH}" \
    --eval_num_samples "${EVAL_NUM_SAMPLES}" \
    --eval_seed "${EVAL_SEED}" \
    --config_path "${CONFIG_PATH}" \
    --model_name "${MODEL_NAME}" \
    --lora_path "${LORA_PATH}" \
    --lora_weight "${LORA_WEIGHT}" \
    --output_dir "${OUTPUT_DIR}" \
    --video_length "${VIDEO_LENGTH}" \
    --base_resolution "${BASE_RESOLUTION}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --sampler_name "${SAMPLER_NAME}" \
    --shift "${SHIFT}" \
    --weight_dtype "${WEIGHT_DTYPE}" \
    ${DIST_ARGS}
else
  ${RUNNER} /gemini/code/VideoX-Fun/examples/wan2.2_fun/predict_i2v_5b_firstframe_pair_val.py \
    --eval_jsonl_path "${EVAL_JSONL_PATH}" \
    --eval_num_samples "${EVAL_NUM_SAMPLES}" \
    --eval_seed "${EVAL_SEED}" \
    --config_path "${CONFIG_PATH}" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --video_length "${VIDEO_LENGTH}" \
    --base_resolution "${BASE_RESOLUTION}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --sampler_name "${SAMPLER_NAME}" \
    --shift "${SHIFT}" \
    --weight_dtype "${WEIGHT_DTYPE}" \
    ${DIST_ARGS}
fi
