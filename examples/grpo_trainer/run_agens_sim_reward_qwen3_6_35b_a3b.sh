#!/usr/bin/env bash
# Qwen3.6-35B-A3B GRPO with Megatron training, vLLM rollout, and Agens simulation reward.
# Defaults target one node with eight 80GB NVIDIA GPUs.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
agens_root=${AGENS_ROOT:-"${repo_root}/agens-cli"}
python_bin=${PYTHON_BIN:-python3}
mount_root=${MOUNT_ROOT:-/mnt/nfs/webcatalyst}

model_path=${MODEL_PATH:-Qwen/Qwen3.6-35B-A3B}
train_file=${TRAIN_FILE:-"${repo_root}/agens_data/train_game.jsonl"}
val_file=${VAL_FILE:-"${repo_root}/agens_data/val_game.jsonl"}
dataset_adapter="${repo_root}/scripts/html_game_dataset.py"

nnodes=${NNODES:-1}
n_gpus_per_node=${NGPUS_PER_NODE:-8}
tp=${TP:-2}
pp=${PP:-1}
cp=${CP:-1}
ep=${EP:-8}
etp=${ETP:-1}
rollout_tp=${ROLLOUT_TP:-8}
all_offload=${ALL_OFFLOAD:-True}

train_batch_size=${TRAIN_BATCH_SIZE:-16}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
rollout_n=${ROLLOUT_N:-5}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-20480}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-4}

max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-16384}
max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU:-24576}
filter_overlong_prompts_workers=${FILTER_OVERLONG_PROMPTS_WORKERS:-1}

actor_lr=${ACTOR_LR:-1e-6}
total_epochs=${TOTAL_EPOCHS:-2}
save_freq=${SAVE_FREQ:-10}
test_freq=${TEST_FREQ:-5}

reward_num_workers=${REWARD_NUM_WORKERS:-6}
reward_max_concurrent=${REWARD_MAX_CONCURRENT:-1}
reward_timeout=${REWARD_TIMEOUT:-330}
agens_attempt_timeout=${AGENS_ATTEMPT_TIMEOUT:-300}
agens_max_retries=${AGENS_MAX_RETRIES:-0}
agens_turn_limit=${AGENS_TURN_LIMIT:-8}
agens_reference_penalty_lambda=${AGENS_REFERENCE_PENALTY_LAMBDA:-1.0}
min_valid_rollouts=${MIN_VALID_ROLLOUTS:-4}
agens_trace_advantage_enabled=${AGENS_TRACE_ADVANTAGE_ENABLED:-True}
agens_trace_threshold=${AGENS_TRACE_THRESHOLD:-70}
agens_trace_beta=${AGENS_TRACE_BETA:-0.5}
agens_trace_gamma=${AGENS_TRACE_GAMMA:-0.9} # or 0.7
agens_trace_max_pause_events=${AGENS_TRACE_MAX_PAUSE_EVENTS_PER_ACTION:-4096}
trainer_logger=${TRAINER_LOGGER:-'["console"]'}

project_name=${PROJECT_NAME:-agens_training_qwen3_6_35b}
experiment_name=${EXPERIMENT_NAME:-qwen3_6_35b_a3b_grpo_agens_$(date +%Y%m%d_%H%M%S)}
checkpoint_dir=${CHECKPOINT_DIR:-"${repo_root}/results/${experiment_name}"}
validation_data_dir=${VALIDATION_DATA_DIR:-"${mount_root}/run_log/${experiment_name}/validation/${experiment_name}"}
agens_output_root=${AGENS_OUTPUT_ROOT:-"${mount_root}/run_log/${experiment_name}/agens_reward"}
pre_rollout_results_root=${PRE_ROLLOUT_RESULTS_ROOT:-}

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}
export VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS=${VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS:-30}

: "${AGENS_SIM_JUDGE_BASE_URL:?Set AGENS_SIM_JUDGE_BASE_URL to an OpenAI-compatible judge endpoint.}"

if ! "${python_bin}" - "${agens_reference_penalty_lambda}" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and 0 <= value <= 1 else 1)
PY
then
    echo "AGENS_REFERENCE_PENALTY_LAMBDA must be a finite number between 0 and 1; got ${agens_reference_penalty_lambda}." >&2
    exit 1
fi

if [[ ! "${rollout_n}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ROLLOUT_N must be a positive integer; got ${rollout_n}." >&2
    exit 1
fi
if [[ ! "${min_valid_rollouts}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MIN_VALID_ROLLOUTS must be a positive integer; got ${min_valid_rollouts}." >&2
    exit 1
fi
if ((min_valid_rollouts > rollout_n)); then
    echo "MIN_VALID_ROLLOUTS=${min_valid_rollouts} cannot exceed ROLLOUT_N=${rollout_n}." >&2
    exit 1
fi

if [[ "${nnodes}" -ne 1 ]]; then
    echo "This launcher currently supports one node; got NNODES=${nnodes}." >&2
    exit 1
fi

world_size=$((nnodes * n_gpus_per_node))
dense_model_parallel_size=$((tp * pp * cp))
expert_model_parallel_size=$((etp * ep * pp))
if ((world_size % dense_model_parallel_size != 0)); then
    echo "Invalid dense mesh: world size ${world_size} must be divisible by TP*PP*CP=${dense_model_parallel_size}." >&2
    exit 1
fi
if ((world_size % expert_model_parallel_size != 0)); then
    echo "Invalid expert mesh: world size ${world_size} must be divisible by ETP*EP*PP=${expert_model_parallel_size}." >&2
    exit 1
fi
if ((world_size % rollout_tp != 0)); then
    echo "Invalid rollout mesh: world size ${world_size} must be divisible by ROLLOUT_TP=${rollout_tp}." >&2
    exit 1
fi

for required_path in \
    "${train_file}" \
    "${val_file}" \
    "${dataset_adapter}" \
    "${agens_root}/dist/cli.js"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 1
    fi
done

if [[ -n "${pre_rollout_results_root}" ]]; then
    if [[ ! -d "${pre_rollout_results_root}" ]]; then
        echo "Pre-rollout results root does not exist: ${pre_rollout_results_root}" >&2
        exit 1
    fi
    pre_rollout_results_root=$(
        "${python_bin}" -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).expanduser().resolve())' \
            "${pre_rollout_results_root}"
    )
    "${python_bin}" "${repo_root}/scripts/validate_agens_pre_rollouts.py" \
        --dataset "${train_file}" \
        --results-root "${pre_rollout_results_root}"
fi

if [[ "${SKIP_RUNTIME_CHECK:-0}" != "1" ]]; then
    if ! "${python_bin}" -c 'import mbridge, transfer_queue, transformers, vllm; import megatron.core; from fla.modules import FusedRMSNormGated; from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule; from transformers import Qwen3_5MoeForConditionalGeneration'; then
        echo "Qwen3.6 runtime dependency check failed. Rebuild docker/Dockerfile.agens.qwen3_6.megatron.vllm so matching flash-linear-attention and fla-core packages are installed." >&2
        exit 1
    fi
fi

config_only=0
for argument in "$@"; do
    if [[ "${argument}" == "--cfg" || "${argument}" == --cfg=* || "${argument}" == "--info" || "${argument}" == --info=* ]]; then
        config_only=1
        break
    fi
done

if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" && "${config_only}" != "1" ]]; then
    echo "Resolving a complete local safetensors snapshot for ${model_path}..." >&2
    model_path=$(
        "${python_bin}" - "${model_path}" <<'PY'
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


model_ref = sys.argv[1]
candidate = Path(model_ref).expanduser()
if candidate.is_dir():
    snapshot = candidate.resolve()
elif model_ref.startswith(("/", "./", "../", "~")):
    raise SystemExit(f"Local MODEL_PATH does not exist or is not a directory: {candidate}")
else:
    snapshot = Path(snapshot_download(repo_id=model_ref)).resolve()

index_path = snapshot / "model.safetensors.index.json"
if index_path.is_file():
    with index_path.open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file).get("weight_map", {})
    shard_names = set(weight_map.values())
    missing = sorted(name for name in shard_names if not (snapshot / name).is_file())
    if missing:
        preview = ", ".join(missing[:3])
        raise SystemExit(
            f"Incomplete model snapshot {snapshot}: {len(missing)} safetensors shards are missing; first: {preview}"
        )
elif not any(path.is_file() for path in snapshot.glob("*.safetensors")):
    raise SystemExit(
        f"No safetensors weights found in {snapshot}. "
        "MBridge requires a fully downloaded local snapshot, not config-only cache files."
    )

print(snapshot)
PY
    )
    echo "Using model snapshot: ${model_path}" >&2
fi

data_config=(
    "data.train_files=${train_file}"
    "data.val_files=${val_file}"
    "data.custom_cls.path=${dataset_adapter}"
    data.custom_cls.name=HtmlGameDataset
    data.prompt_key=prompt
    data.return_raw_chat=True
    "data.train_batch_size=${train_batch_size}"
    "data.max_prompt_length=${max_prompt_length}"
    "data.max_response_length=${max_response_length}"
    data.filter_overlong_prompts=True
    "data.filter_overlong_prompts_workers=${filter_overlong_prompts_workers}"
    data.truncation=error
    +data.apply_chat_template_kwargs.enable_thinking=True
)

model_config=(
    "actor_rollout_ref.model.path=${model_path}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
)

actor_config=(
    "actor_rollout_ref.actor.optim.lr=${actor_lr}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token_len_per_gpu}"
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${tp}"
    "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${pp}"
    "actor_rollout_ref.actor.megatron.context_parallel_size=${cp}"
    "actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ep}"
    "actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${etp}"
    "actor_rollout_ref.actor.megatron.param_offload=${all_offload}"
    "actor_rollout_ref.actor.megatron.optimizer_offload=${all_offload}"
    "actor_rollout_ref.actor.megatron.grad_offload=${all_offload}"
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
)

rollout_config=(
    actor_rollout_ref.rollout.name=vllm
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization}"
    "actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}"
    "actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}"
    "actor_rollout_ref.rollout.n=${rollout_n}"
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_token_len_per_gpu}"
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True
)

ref_config=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_token_len_per_gpu}"
    "actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${tp}"
    "actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${pp}"
    "actor_rollout_ref.ref.megatron.context_parallel_size=${cp}"
    "actor_rollout_ref.ref.megatron.expert_model_parallel_size=${ep}"
    "actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${etp}"
    "actor_rollout_ref.ref.megatron.param_offload=${all_offload}"
)

reward_config=(
    "reward.custom_reward_function.path=${repo_root}/verl/utils/reward_score/agens_sim_reward.py"
    reward.custom_reward_function.name=compute_score
    "+reward.custom_reward_function.reward_kwargs.agens_cli=${agens_root}/dist/cli.js"
    "+reward.custom_reward_function.reward_kwargs.agens_cwd=${agens_root}"
    "+reward.custom_reward_function.reward_kwargs.output_root=${agens_output_root}"
    "+reward.custom_reward_function.reward_kwargs.turn_limit=${agens_turn_limit}"
    "+reward.custom_reward_function.reward_kwargs.attempt_timeout=${agens_attempt_timeout}"
    "+reward.custom_reward_function.reward_kwargs.max_retries=${agens_max_retries}"
    "+reward.custom_reward_function.reward_kwargs.reference_penalty_lambda=${agens_reference_penalty_lambda}"
    reward.reward_manager.name=rate_limited
    reward.reward_model.enable=False
    "reward.num_workers=${reward_num_workers}"
    "+reward.max_concurrent=${reward_max_concurrent}"
    "+reward.timeout=${reward_timeout}"
    +reward.agens_failure_filter.enabled=True
    "+reward.agens_failure_filter.min_valid_rollouts=${min_valid_rollouts}"
    "reward.agens_trace_advantage.enabled=${agens_trace_advantage_enabled}"
    "reward.agens_trace_advantage.threshold=${agens_trace_threshold}"
    "reward.agens_trace_advantage.beta=${agens_trace_beta}"
    "reward.agens_trace_advantage.gamma=${agens_trace_gamma}"
    "reward.agens_trace_advantage.max_pause_events_per_action=${agens_trace_max_pause_events}"
)
if [[ -n "${pre_rollout_results_root}" ]]; then
    reward_config+=(
        "+reward.custom_reward_function.reward_kwargs.pre_rollout_results_root=${pre_rollout_results_root}"
    )
fi

trainer_config=(
    trainer.use_v1=True
    trainer.critic_warmup=0
    trainer.val_before_train=True
    trainer.max_actor_ckpt_to_keep=1
    "trainer.test_freq=${test_freq}"
    "trainer.validation_data_dir=${validation_data_dir}"
    "trainer.save_freq=${save_freq}"
    "trainer.total_epochs=${total_epochs}"
    "trainer.default_local_dir=${checkpoint_dir}"
    trainer.balance_batch=True
    "trainer.logger=${trainer_logger}"
    "trainer.project_name=${project_name}"
    "trainer.experiment_name=${experiment_name}"
    "trainer.n_gpus_per_node=${n_gpus_per_node}"
    "trainer.nnodes=${nnodes}"
)

"${python_bin}" -m verl.trainer.main_ppo \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    "${data_config[@]}" \
    "${model_config[@]}" \
    "${actor_config[@]}" \
    "${rollout_config[@]}" \
    "${ref_config[@]}" \
    "${reward_config[@]}" \
    "${trainer_config[@]}" \
    "$@"
