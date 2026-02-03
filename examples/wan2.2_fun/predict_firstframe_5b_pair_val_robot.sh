NUM_GPUS=${NUM_GPUS:-1}
ULYSSES_DEGREE=${ULYSSES_DEGREE:-1}
RING_DEGREE=${RING_DEGREE:-1}

# Robot dataset args (used when EVAL_JSONL_PATH is empty)
ROBOT_DATA_ROOT=${ROBOT_DATA_ROOT:-/gemini/code/datasets/libero/perturbed}
ROBOT_SOURCE_DATA_ROOT=${ROBOT_SOURCE_DATA_ROOT:-/gemini/code/datasets/libero/datasets}
ROBOT_SUITES=${ROBOT_SUITES:-}

# Optional jsonl path (if provided, will be used instead of robot dataset traversal)
EVAL_JSONL_PATH=${EVAL_JSONL_PATH:-}
EVAL_NUM_SAMPLES=${EVAL_NUM_SAMPLES:-100}
EVAL_SEED=${EVAL_SEED:-42}

CONFIG_PATH=${CONFIG_PATH:-/gemini/code/VideoX-Fun/config/wan2.2/wan_civitai_5b.yaml}
MODEL_NAME=${MODEL_NAME:-/gemini/code/models/Wan2.2-Fun-5B-InP/}
LORA_PATH=${LORA_PATH:-/gemini/code/VideoX-Fun/output_dir_robot_libero/checkpoint-16500.safetensors}
LORA_WEIGHT=${LORA_WEIGHT:-1}

OUTPUT_DIR=${OUTPUT_DIR:-/gemini/code/VideoX-Fun/examples/wan2.2_fun/firstframe_pair_val_outputs_2.2fun_5b_robot}
VIDEO_LENGTH=${VIDEO_LENGTH:-129}
BASE_RESOLUTION=${BASE_RESOLUTION:-640}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-6.0}
SAMPLER_NAME=${SAMPLER_NAME:-Flow}
SHIFT=${SHIFT:-5}
WEIGHT_DTYPE=${WEIGHT_DTYPE:-bf16}
SRC_FRAME_SAMPLING=${SRC_FRAME_SAMPLING:-stride2_then_tail}

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

BASE_ARGS=(
  --eval_num_samples "${EVAL_NUM_SAMPLES}"
  --eval_seed "${EVAL_SEED}"
  --config_path "${CONFIG_PATH}"
  --model_name "${MODEL_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --video_length "${VIDEO_LENGTH}"
  --base_resolution "${BASE_RESOLUTION}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --sampler_name "${SAMPLER_NAME}"
  --shift "${SHIFT}"
  --weight_dtype "${WEIGHT_DTYPE}"
  --src_frame_sampling "${SRC_FRAME_SAMPLING}"
)

LORA_ARGS=()
if [ -n "${LORA_PATH}" ]; then
  LORA_ARGS+=(--lora_path "${LORA_PATH}" --lora_weight "${LORA_WEIGHT}")
fi

DATA_ARGS=()
if [ -n "${EVAL_JSONL_PATH}" ]; then
  DATA_ARGS+=(--eval_jsonl_path "${EVAL_JSONL_PATH}")
else
  DATA_ARGS+=(--robot_data_root "${ROBOT_DATA_ROOT}" --robot_source_data_root "${ROBOT_SOURCE_DATA_ROOT}")
  if [ -n "${ROBOT_SUITES}" ]; then
    DATA_ARGS+=(--robot_suites ${ROBOT_SUITES})
  fi
fi

${RUNNER} /gemini/code/VideoX-Fun/examples/wan2.2_fun/predict_i2v_5b_firstframe_pair_val_robot.py \
  ${BASE_ARGS[@]} \
  ${LORA_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${DIST_ARGS}
