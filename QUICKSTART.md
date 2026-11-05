# Quick Start Guide

Requires **8 NVIDIA B200 GPUs**, Linux, and Docker with GPU support. Start from the repository root.

## 1. Enter the runtime

On the host:

The prebuilt `klozewang/agens-verl:0805-qwen36` image is available for non-GB GPUs. This B200 workflow rebuilds the image with the FLA 0.5.2 packages required by NVIDIA GB-series hardware.

```bash
docker build -f docker/Dockerfile.agens.qwen3_6.megatron.vllm \
  --build-arg FLASH_LINEAR_ATTENTION_VERSION=0.5.2 \
  -t webcatalyst:b200 .
docker run --gpus all --ipc=host -it --rm \
  -v "$PWD":/workspace/webcatalyst \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace/webcatalyst webcatalyst:b200 bash
```

Run all remaining commands inside the container, in the same shell:

```bash
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps -e .
```

## 2. Configure the judge

Copy `.env.example` to `.env` and edit it with your own provider settings. Both the training and inference launchers load this file and require an explicit judge endpoint. The endpoint must support images and the Responses API for the configuration below:

```bash
export AGENS_SIM_JUDGE_BASE_URL="https://your-provider.example/v1"
export AGENS_SIM_JUDGE_API_KEY="YOUR_JUDGE_API_KEY"
export AGENS_SIM_JUDGE_MODEL="gpt-5.6-terra"
export AGENS_SIM_EXTRA_GLOBAL_JUDGE_MODEL="gpt-5.6-terra"
export AGENS_SIM_JUDGE_WIRE_API="responses"
export TRAINER_LOGGER='["console"]'
export MOUNT_ROOT="/workspace/webcatalyst/run_artifacts"
```

Keys are preserved when the launchers load `.env`. Omit the judge key only for an endpoint that does not require authentication. For a Chat Completions endpoint, set `AGENS_SIM_JUDGE_WIRE_API=completions` instead.

## 3. Prepare Visual Anchors

Download and extract the paper's references:

```bash
mkdir -p inference/results/paper_bundle
curl -fL --retry 3 \
  <ZIP_TO_BE_UPLOADED>  \
  -o inference/results/inference_gpt55.zip
python3 -m zipfile -e inference/results/inference_gpt55.zip inference/results/paper_bundle
PRE_ROLLOUT_RESULTS_ROOT=$(python3 - <<'PY'
from pathlib import Path
roots = {p.parent.parent.resolve() for p in Path("inference/results/paper_bundle").rglob("state/run-config-*.json")}
assert len(roots) == 1, f"Expected one reference root; found {roots}"
print(roots.pop())
PY
)
export PRE_ROLLOUT_RESULTS_ROOT
```

## 4. Run training

```bash
export EXPERIMENT_NAME="webcatalyst_b200_$(date +%Y%m%d_%H%M%S)"
bash run_b200.sh trainer.total_training_steps=40
```

The launcher builds Agens and validates the references automatically. Checkpoints go to `results/$EXPERIMENT_NAME/`. For reference generation and other options, see [README.md](README.md).
