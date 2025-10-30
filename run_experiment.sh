#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"meta-llama/Meta-Llama-3-8B"}
DATA_DIR=${DATA_DIR:-"./data"}
SIMILARITY_MATRIX=${SIMILARITY_MATRIX:-"./similarity_matrix.pkl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./experiments/learnable_alpha"}
CAL_SAMPLES=${CAL_SAMPLES:-1000}
TRAIN_STEPS=${TRAIN_STEPS:-500}
LR=${LR:-1e-4}
BATCH_SIZE=${BATCH_SIZE:-4}

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/plots"

echo "[1/2] Running evaluation suite..."
python evaluate_methods.py \
  --model_path "${MODEL_PATH}" \
  --data_dir "${DATA_DIR}" \
  --similarity_matrix "${SIMILARITY_MATRIX}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_calibration_samples ${CAL_SAMPLES} \
  --num_training_steps ${TRAIN_STEPS} \
  --learning_rate ${LR} \
  --batch_size ${BATCH_SIZE}

RESULTS_FILE="${OUTPUT_DIR}/results.json"
LEARNED_ALPHAS_FILE="${OUTPUT_DIR}/learned_alpha/learnable_alpha_metrics.json"

if [ -f "${LEARNED_ALPHAS_FILE}" ]; then
  echo "[2/2] Generating plots..."
  python plot_results.py \
    --results_file "${RESULTS_FILE}" \
    --learned_alphas "${LEARNED_ALPHAS_FILE}" \
    --output_dir "${OUTPUT_DIR}/plots"
else
  echo "Skipped plotting: learned alpha metrics not found at ${LEARNED_ALPHAS_FILE}."
fi

echo "Done. Outputs stored in ${OUTPUT_DIR}."
