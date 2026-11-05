#!/usr/bin/env bash
set -euo pipefail
unset AGENS_ROOT
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
agens_root=${AGENS_ROOT:-"${repo_root}/agens-cli"}
python_bin=${PYTHON_BIN:-python3}

cd "${repo_root}"

if [[ -f "${repo_root}/.env" ]]; then
    # .env is gitignored and may contain the judge endpoint and credentials.
    # shellcheck disable=SC1091
    set -a
    source "${repo_root}/.env"
    set +a
fi

unset WANDB_RUN_ID
unset WANDB_RESUME

: "${AGENS_SIM_JUDGE_BASE_URL:?Set AGENS_SIM_JUDGE_BASE_URL to an OpenAI-compatible judge endpoint.}"

export AGENS_ROOT="${agens_root}"
export AGENS_WIRE_API=${AGENS_WIRE_API:-responses}
export AGENS_SIM_JUDGE_WIRE_API=${AGENS_SIM_JUDGE_WIRE_API:-responses}
export AGENS_JUDGE_WIRE_API=${AGENS_JUDGE_WIRE_API:-responses}

# consider using these to enable more RPM by adding other keys.
# export AGENS_SIM_JUDGE_BASE_URL_ADDITION=
# export AGENS_SIM_JUDGE_API_KEY_ADDITION=
# export AGENS_SIM_JUDGE_ROUTING_RATIO=0.5

export AGENS_SIM_JUDGE_API_KEY="${AGENS_SIM_JUDGE_API_KEY:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"
export AGENS_SIM_JUDGE_MODEL=${AGENS_SIM_JUDGE_MODEL:-gpt-5.6-terra}
export AGENS_SIM_EXTRA_GLOBAL_JUDGE_MODEL=${AGENS_SIM_EXTRA_GLOBAL_JUDGE_MODEL:-${AGENS_SIM_JUDGE_MODEL}}
export AGENS_SIM_JUDGE_DISABLE_TEMPERATURE=${AGENS_SIM_JUDGE_DISABLE_TEMPERATURE:-true}
export AGENS_JUDGE_BASE_URL=${AGENS_JUDGE_BASE_URL:-${AGENS_SIM_JUDGE_BASE_URL}}
export AGENS_JUDGE_API_KEY=${AGENS_JUDGE_API_KEY:-${AGENS_SIM_JUDGE_API_KEY:-}}
export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=${PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD:-1}
export PROJECT_NAME=${PROJECT_NAME:-webcatalyst_qwen36}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
export MOUNT_ROOT=${MOUNT_ROOT:-/mnt/nfs/webcatalyst}

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
    npm ci --prefix "${agens_root}"
fi
npm run build --prefix "${agens_root}"

if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
    "${python_bin}" -c 'import torch; count = torch.cuda.device_count(); assert count >= 8, f"Qwen3.6 launcher requires 8 visible GPUs, found {count}"'
fi

exec bash "${repo_root}/examples/grpo_trainer/run_agens_sim_reward_qwen3_6_35b_a3b_megatron.sh" "$@"
