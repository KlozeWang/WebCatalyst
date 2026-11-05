#!/usr/bin/env bash
# Generate one response per dataset row on B200 GPUs, then evaluate with Agens.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment_name=${EXPERIMENT_NAME:-qwen3_6_35b_a3b_inference}
default_input_file=${INPUT_FILE:-"${repo_root}/agens_data/train_game.jsonl"}
default_output_root=${OUTPUT_ROOT:-"${repo_root}/inference/results/${experiment_name}"}

input_file=${default_input_file}
if (( $# >= 1 )) && [[ $1 != --* ]]; then
    input_file=$1
    shift
fi
output_root=${default_output_root}
if (( $# >= 1 )) && [[ $1 != --* ]]; then
    output_root=$1
    shift
fi

caller_pwd=${PWD}
if [[ "${input_file}" != /* ]]; then
    input_file="${caller_pwd}/${input_file}"
fi
if [[ "${output_root}" != /* ]]; then
    output_root="${caller_pwd}/${output_root}"
fi

if [[ ! -f "${input_file}" ]]; then
    echo "Input dataset does not exist: ${input_file}" >&2
    exit 1
fi
case "${input_file}" in
    *.parquet|*.pq|*.jsonl) ;;
    *)
        echo "Input must be a .parquet, .pq, or .jsonl file: ${input_file}" >&2
        exit 1
        ;;
esac

cd "${repo_root}"

if [[ -f "${repo_root}/.env" ]]; then
    # shellcheck disable=SC1091
    set -a
    source "${repo_root}/.env"
    set +a
fi

agens_root=${AGENS_ROOT:-"${repo_root}/agens-cli"}
python_bin=${PYTHON_BIN:-python3}
export AGENS_SIM_JUDGE_API_KEY="${AGENS_SIM_JUDGE_API_KEY:-}"
export AGENS_ROOT="${agens_root}"
export AGENS_WIRE_API=${AGENS_WIRE_API:-responses}
export AGENS_SIM_JUDGE_WIRE_API=${AGENS_SIM_JUDGE_WIRE_API:-responses}
export AGENS_JUDGE_WIRE_API=${AGENS_JUDGE_WIRE_API:-responses}
: "${AGENS_SIM_JUDGE_BASE_URL:?Set AGENS_SIM_JUDGE_BASE_URL to an OpenAI-compatible judge endpoint.}"
export AGENS_SIM_JUDGE_BASE_URL
export AGENS_SIM_JUDGE_MODEL=${AGENS_SIM_JUDGE_MODEL:-gpt-5.6-terra}
export AGENS_SIM_EXTRA_GLOBAL_JUDGE_MODEL=${AGENS_SIM_EXTRA_GLOBAL_JUDGE_MODEL:-${AGENS_SIM_JUDGE_MODEL}}
export AGENS_SIM_JUDGE_DISABLE_TEMPERATURE=${AGENS_SIM_JUDGE_DISABLE_TEMPERATURE:-true}
export AGENS_JUDGE_BASE_URL=${AGENS_JUDGE_BASE_URL:-${AGENS_SIM_JUDGE_BASE_URL}}
export AGENS_JUDGE_API_KEY=${AGENS_JUDGE_API_KEY:-${AGENS_SIM_JUDGE_API_KEY:-}}
export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=${PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD:-1}
export VLLM_USE_V1=${VLLM_USE_V1:-1}

if ! command -v node >/dev/null 2>&1; then
    echo "Node.js 24 or newer is required. Use docker/Dockerfile.agens.qwen3_6.megatron.vllm." >&2
    exit 1
fi
node_major=$(node -p 'Number(process.versions.node.split(".")[0])')
if ((node_major < 24)); then
    echo "Node.js 24 or newer is required; found $(node --version)." >&2
    exit 1
fi
if [[ ! -f "${agens_root}/package-lock.json" ]]; then
    echo "Agens CLI was not found at ${agens_root}." >&2
    exit 1
fi
if [[ ! -d "${agens_root}/node_modules" ]]; then
    echo "Agens dependencies are missing at ${agens_root}/node_modules; install them before running inference." >&2
    exit 1
fi
npm run build --prefix "${agens_root}"

nnodes=${NNODES:-1}
n_gpus_per_node=${NGPUS_PER_NODE:-8}
if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
    "${python_bin}" -c \
        'import sys, torch; expected = int(sys.argv[1]); count = torch.cuda.device_count(); assert count >= expected, f"B200 inference requires {expected} visible GPUs, found {count}"' \
        "${n_gpus_per_node}"
fi

resume_args=()
if [[ "${RESUME:-0}" == "1" ]]; then
    resume_args+=(--resume)
fi

exec "${python_bin}" "${repo_root}/inference/infer_and_simulate.py" \
    --input "${input_file}" \
    --output-root "${output_root}" \
    --model-path "${MODEL_PATH:-Qwen/Qwen3.6-35B-A3B}" \
    --nnodes "${nnodes}" \
    --gpus-per-node "${n_gpus_per_node}" \
    --rollout-tp "${ROLLOUT_TP:-1}" \
    --rollout-gpu-memory-utilization "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH:-2048}" \
    --max-response-length "${MAX_RESPONSE_LENGTH:-32768}" \
    --rollout-max-model-len "${ROLLOUT_MAX_MODEL_LEN:-34816}" \
    --rollout-max-num-batched-tokens "${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-34816}" \
    --rollout-max-num-seqs "${ROLLOUT_MAX_NUM_SEQS:-16}" \
    --rollout-moe-backend "${ROLLOUT_MOE_BACKEND:-triton}" \
    --generation-concurrency "${GENERATION_CONCURRENCY:-0}" \
    --temperature "${TEMPERATURE:-0.6}" \
    --top-p "${TOP_P:-0.95}" \
    --top-k "${TOP_K:-20}" \
    --agens-cli "${agens_root}/dist/cli.js" \
    --agens-cwd "${agens_root}" \
    --agens-turn-limit "${AGENS_TURN_LIMIT:-3}" \
    --agens-concurrency "${AGENS_CONCURRENCY:-8}" \
    --agens-attempt-timeout "${AGENS_ATTEMPT_TIMEOUT:-300}" \
    --agens-total-timeout "${AGENS_TOTAL_TIMEOUT:-330}" \
    --agens-max-retries "${AGENS_MAX_RETRIES:-0}" \
    "${resume_args[@]}" \
    "$@"
