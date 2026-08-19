#!/bin/bash
# Serve one registered language model with vLLM and print its endpoint.
#
#   scripts/llm_server.sh patho-r1-7b              # sbatch, print endpoint
#   scripts/llm_server.sh patho-r1-7b --foreground # run here (interactive node)
#
# vLLM pins torch hard and this benchmark pins torch for its methods, so the
# two must not share an interpreter. Point PGVL_VLLM_ENV at a separate conda
# environment holding vllm; nothing in common/llm imports it.
set -uo pipefail

MODEL="${1:?usage: llm_server.sh <registered-model> [--foreground]}"; shift || true
FOREGROUND=0
[[ "${1:-}" == "--foreground" ]] && FOREGROUND=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${REPO}/.env" ]]; then
    set -a
    source "${REPO}/.env"
    set +a
fi
PORT="${PGVL_VLLM_PORT:-8000}"
VLLM_ENV="${PGVL_VLLM_ENV:-${PGVL_STORAGE_ROOT}/dchanda/envs/pgvl-vllm}"
export HF_HOME="${HF_HOME:-${PGVL_STORAGE_ROOT}/.cache_huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Python adds ~/.local/lib/pythonX.Y/site-packages ahead of the environment,
# so a stray user-installed package can shadow one of vLLM's pinned deps.
# The whole point of a separate environment is that it is not shared.
export PYTHONNOUSERSITE=1

# Compute nodes ship an older libstdc++ than the login node (CXXABI_1.3.13 vs
# the 1.3.15+ that conda's icu/sqlite stack needs). The environment carries a
# newer copy; make the loader prefer it, or the server dies during import with
# a CXXABI error that looks nothing like a vLLM problem.
[[ -n "${VLLM_ENV}" ]] && export LD_LIBRARY_PATH="${VLLM_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Resolve the repo id and revision from the registry, so the served model is
# the one the client will later verify against rather than a hand-typed path.
read -r MODEL_ID REVISION CACHED < <(
  cd "${REPO}" && python - "$MODEL" <<'PY'
import sys
sys.path.insert(0, ".")
from common.llm.registry import get_spec
spec = get_spec(sys.argv[1])
print(spec.model_id, spec.revision or "main", int(spec.cached))
PY
) || { echo "[llm] FATAL: unknown model '${MODEL}'" >&2; exit 2; }

if [[ "${CACHED}" != "1" ]]; then
    echo "[llm] WARNING: ${MODEL_ID} is not in the shared cache." >&2
    echo "[llm] Compute nodes are offline; download it from a login node first:" >&2
    echo "[llm]   HF_HOME=${HF_HOME} hf download ${MODEL_ID}" >&2
fi

# Activating by prefix needs conda on PATH, which a batch shell does not get.
# Call the environment's own interpreter instead: no activation, no module load,
# and no chance of picking up the benchmark environment by accident.
serve_cmd() {
    local python_bin="python"
    [[ -n "${VLLM_ENV}" ]] && python_bin="${VLLM_ENV}/bin/python"
    cat <<CMD
${python_bin} -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_ID} \
  --revision ${REVISION} \
  --served-model-name ${MODEL} \
  --port ${PORT} \
  --download-dir ${HF_HOME}/hub \
  ${PGVL_VLLM_EAGER:-} \
  ${PGVL_VLLM_ARGS:-}
CMD
}

if [[ "${FOREGROUND}" == "1" ]]; then
    echo "[llm] serving ${MODEL_ID}@${REVISION} as '${MODEL}' on port ${PORT}"
    echo "[llm] endpoint: http://$(hostname):${PORT}/v1"
    eval "$(serve_cmd)"
    exit $?
fi

# NOTE: flashinfer's comm/fd_exchange.py annotates `array.array[int]` without
# postponed evaluation. array.array is not subscriptable, so the annotation
# raises TypeError at import -- and vLLM guards that import with
# `except ImportError`, which does not catch it, so the engine dies at startup.
# The environment's copy is patched with `from __future__ import annotations`
# (original kept as fd_exchange.py.orig). Reinstalling the vLLM environment
# loses that patch and the failure returns. Eager mode does NOT avoid it: the
# import arrives through the model registry, not the compile backend.
#
# vLLM stages the whole checkpoint through host RAM before moving it to the
# device, so a GPU request alone is not enough: without --mem the default
# allocation is a fraction of what a 7B model needs and the job is OOM-killed
# during load, with no vLLM error of its own.
JOB_SCRIPT="$(mktemp)"
cat > "${JOB_SCRIPT}" <<JOB
#!/bin/bash
#SBATCH --job-name=vllm-${MODEL}
#SBATCH --account=${PGVL_SLURM_ACCOUNT:-bhwm-delta-gpu}
#SBATCH --partition=${PGVL_SLURM_PARTITION:-gpuA100x4}
#SBATCH --gres=gpu:${PGVL_VLLM_GPUS:-1}
#SBATCH --cpus-per-task=${PGVL_VLLM_CPUS:-16}
#SBATCH --mem=${PGVL_VLLM_MEM:-96G}
#SBATCH --time=${PGVL_VLLM_WALLTIME:-08:00:00}
#SBATCH --output=logs/vllm-${MODEL}-%j.out
set -uo pipefail
export HF_HOME="${HF_HOME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE}"
# Without these the batch job is silent for the whole model load, and the user
# site-packages guard applies only to the submitting shell rather than the job
# that actually serves the model.
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${VLLM_ENV}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
echo "[llm] endpoint: http://\$(hostname):${PORT}/v1"
echo "[llm] loading ${MODEL_ID} -- first startup reads ~16 GB from /work"
$(serve_cmd)
JOB

mkdir -p logs
JOB_ID="$(sbatch --parsable "${JOB_SCRIPT}")" || {
    echo "[llm] FATAL: sbatch failed" >&2; rm -f "${JOB_SCRIPT}"; exit 3; }
rm -f "${JOB_SCRIPT}"
echo "[llm] submitted job ${JOB_ID} serving '${MODEL}'"
echo "[llm] the endpoint is printed in logs/vllm-${MODEL}-${JOB_ID}.out once the"
echo "[llm] node is allocated; wait for it with:"
echo "      until grep -m1 -o 'http://[^ ]*' logs/vllm-${MODEL}-${JOB_ID}.out; do sleep 10; done"
