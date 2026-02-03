NUM_GPUS=${NUM_GPUS:-1}
ULYSSES_DEGREE=${ULYSSES_DEGREE:-2}
RING_DEGREE=${RING_DEGREE:-4}

PRESET=${PRESET:-5B}

EVAL_JSONL_PATH=${EVAL_JSONL_PATH:-/gemini/code/VideoX-Fun/examples/wan2.2/moviebench_character_pair_test1.jsonl}
EVAL_NUM_SAMPLES=${EVAL_NUM_SAMPLES:-8}
EVAL_SEED=${EVAL_SEED:-42}

DEFAULT_CONFIG_PATH=/gemini/code/VideoX-Fun/config/wan2.2/wan_civitai_5b.yaml
DEFAULT_MODEL_NAME=/gemini/code/models/Wan2.2-TI2V-5B
DEFAULT_LORA_PATH=/gemini/code/VideoX-Fun/output_dir_character_pair_5b_filtered_pair/checkpoint-2400.safetensors

if [ "${PRESET}" = "14B" ] || [ "${PRESET}" = "A14B" ]; then
  DEFAULT_CONFIG_PATH=/gemini/code/VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml
  DEFAULT_MODEL_NAME=/gemini/code/models/Wan2.2-I2V-A14B
  DEFAULT_LORA_PATH=""
fi

CONFIG_PATH=${CONFIG_PATH:-${DEFAULT_CONFIG_PATH}}
MODEL_NAME=${MODEL_NAME:-${DEFAULT_MODEL_NAME}}
LORA_PATH=${LORA_PATH:-${DEFAULT_LORA_PATH}}
LORA_WEIGHT=${LORA_WEIGHT:-1.0}

OUTPUT_DIR=${OUTPUT_DIR:-/gemini/code/VideoX-Fun/examples/wan2.2/character_pair_test1_outputs}
VIDEO_LENGTH=${VIDEO_LENGTH:-81}
BASE_RESOLUTION=${BASE_RESOLUTION:-640}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-6.0}
SAMPLER_NAME=${SAMPLER_NAME:-Flow_Unipc}
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
  ${RUNNER} /gemini/code/VideoX-Fun/examples/wan2.2/predict_ti2v_character_pair_val.py \
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
  ${RUNNER} /gemini/code/VideoX-Fun/examples/wan2.2/predict_ti2v_character_pair_val.py \
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