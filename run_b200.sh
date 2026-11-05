#!/usr/bin/env bash
# B200 rollout deployment for the training job launched by run.sh.
#
# Run one complete rollout replica per B200. vLLM's automatic Blackwell choice,
# FlashInfer TRTLLM BF16 MoE, converts expert weights to a block layout that is
# incompatible with this job's live Megatron weight reload; at TP=8 it also
# produces 64-wide expert shards while that kernel requires a multiple of 128.
# Force the Triton MoE backend to retain the reloadable standard weight layout.
# The remaining overrides increase only vLLM's KV-cache and scheduler capacity.
# Training batch sizes, sampling settings, sequence lengths, and optimizer
# settings continue to come unchanged from the shared launcher.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
export MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-34816}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-34816}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-34816}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-16}
export PRE_ROLLOUT_RESULTS_ROOT=${PRE_ROLLOUT_RESULTS_ROOT:-/workspace/webcatalyst/inference/results}
rollout_moe_backend=${ROLLOUT_MOE_BACKEND:-triton}
agens_failure_pause_threshold=${AGENS_FAILURE_PAUSE_THRESHOLD:-10}
# run.sh loads the judge endpoint and credentials from the root .env or environment.

if [[ ! "${agens_failure_pause_threshold}" =~ ^[1-9][0-9]*$ ]]; then
    echo "AGENS_FAILURE_PAUSE_THRESHOLD must be a positive integer; got ${agens_failure_pause_threshold}." >&2
    exit 1
fi

exec bash "${repo_root}/run.sh" \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=${rollout_moe_backend}" \
    "+reward.agens_failure_filter.pause_after_dropped_samples=${agens_failure_pause_threshold}" \
    "$@"
