#!/usr/bin/env bash
# Resume the newest complete B200 checkpoint under ./results.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
results_root=${RESUME_RESULTS_ROOT:-"${repo_root}/results"}

if [[ ! -d "${results_root}" ]]; then
    echo "Checkpoint results directory does not exist: ${results_root}" >&2
    exit 1
fi

latest_experiment=""
latest_checkpoint=""
latest_step=""
latest_mtime=-1

shopt -s nullglob
for tracker in "${results_root}"/*/latest_checkpointed_iteration.txt; do
    experiment_dir=${tracker%/latest_checkpointed_iteration.txt}
    step=""
    IFS= read -r step < "${tracker}" || [[ -n "${step}" ]] || continue
    if [[ ! "${step}" =~ ^[0-9]+$ ]]; then
        echo "Ignoring invalid checkpoint tracker ${tracker}: ${step}" >&2
        continue
    fi

    checkpoint_dir="${experiment_dir}/global_step_${step}"
    required_files=(
        "${checkpoint_dir}/data.pt"
        "${checkpoint_dir}/actor/ckpt_contents.json"
        "${checkpoint_dir}/actor/model/huggingface/model.safetensors.index.json"
        "${checkpoint_dir}/actor/optimizer/dist_ckpt/.metadata"
        "${checkpoint_dir}/actor/extra/dist_ckpt/.metadata"
    )
    complete=true
    for required_file in "${required_files[@]}"; do
        if [[ ! -f "${required_file}" ]]; then
            complete=false
            echo "Ignoring incomplete checkpoint ${checkpoint_dir}: missing ${required_file}" >&2
            break
        fi
    done
    if [[ "${complete}" != true ]]; then
        continue
    fi

    tracker_mtime=$(stat -c %Y "${tracker}")
    if ((tracker_mtime > latest_mtime)); then
        latest_mtime=${tracker_mtime}
        latest_experiment=${experiment_dir}
        latest_checkpoint=${checkpoint_dir}
        latest_step=${step}
    fi
done
shopt -u nullglob

if [[ -z "${latest_checkpoint}" ]]; then
    echo "No complete resumable checkpoint found under ${results_root}." >&2
    exit 1
fi

experiment_name=$(basename "${latest_experiment}")
export CHECKPOINT_DIR=${latest_experiment}
export EXPERIMENT_NAME=${experiment_name}

echo "Resuming B200 training from global step ${latest_step}:"
echo "  checkpoint: ${latest_checkpoint}"
echo "  save root:  ${latest_experiment}"
echo "  experiment: ${experiment_name}"

if [[ "${RESUME_DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

# Put resume overrides last so the wrapper cannot accidentally start from
# scratch because of a conflicting caller-provided Hydra override.
exec bash "${repo_root}/run_b200.sh" \
    "$@" \
    "trainer.default_local_dir=${latest_experiment}" \
    "trainer.experiment_name=${experiment_name}" \
    trainer.resume_mode=resume_path \
    "trainer.resume_from_path=${latest_checkpoint}"
