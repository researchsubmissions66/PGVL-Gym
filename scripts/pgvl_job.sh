#!/bin/bash
# Job body for exactly one PGVL-Gym run.
#
# Runs identically under sbatch and by hand, so a failed benchmark row can be
# reproduced interactively without reconstructing the environment:
#
#     scripts/pgvl_job.sh pathpt benchmarks/tcga_brca/configs/pathpt/brca_4shot.yaml
#
# Exit codes
#   0   train.py completed
#   78  environment could not be prepared (never a modelling failure)
#   *   whatever train.py returned
#
# `set -e` is deliberately omitted: the launcher classifies runs by exit code,
# so this script must always reach its final `exit`.
set -uo pipefail

METHOD="${1:?usage: pgvl_job.sh <method> <config> [device] [--rerun]}"
CONFIG="${2:?usage: pgvl_job.sh <method> <config> [device] [--rerun]}"
DEVICE="${3:-cuda:0}"
RERUN="${4:-}"
if [[ -n "${RERUN}" && "${RERUN}" != "--rerun" ]]; then
    echo "[pgvl] FATAL: fourth argument must be --rerun, got '${RERUN}'" >&2
    exit 78
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${REPO}/.env" ]]; then
    set -a
    source "${REPO}/.env"
    set +a
fi
cd "$REPO" || exit 78

if [[ -z "${PGVL_STORAGE_ROOT:-}" ]]; then
    echo "[pgvl] FATAL: PGVL_STORAGE_ROOT is unset; configure .env first" >&2
    exit 78
fi

# --- environment ------------------------------------------------------------
module load pytorch-conda 2>/dev/null || true

# PGVL_CONDA_ENV selects the benchmark environment created from environment.yml.
# When unset the module's default interpreter is used, which is enough for the
# methods whose extras are already present.
if [[ -n "${PGVL_CONDA_ENV:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
    fi
    conda activate "${PGVL_CONDA_ENV}" || {
        echo "[pgvl] FATAL: cannot activate conda env '${PGVL_CONDA_ENV}'" >&2
        exit 78
    }
fi

# Every weight this benchmark needs is already in the shared project cache.
# Compute nodes have no outbound network, so point Hugging Face at it and make
# a cache miss fail immediately instead of hanging on a connection attempt.
export HF_HOME="${HF_HOME:-${PGVL_STORAGE_ROOT}/.cache_huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

# PLIP's default revision publishes only pytorch_model.bin, and transformers
# refuses to torch.load a .bin below torch 2.6 (CVE-2025-32434). A directory
# holding the same weights as model.safetensors loads through a path that never
# calls torch.load. Exported only when it is actually populated, so an
# unprepared site falls back to the HF cache instead of failing on a bad path.
# See docs/environment.md, "PLIP below torch 2.6".
PLIP_LOCAL="${PLIP_CKPT:-${PGVL_STORAGE_ROOT}/dchanda/model_cache/plip}"
if [[ -f "${PLIP_LOCAL}/model.safetensors" ]]; then
    export PLIP_CKPT="${PLIP_LOCAL}"
else
    echo "[pgvl] note: no PLIP safetensors at ${PLIP_LOCAL};" \
         "PLIP/MAPLE/MSCPT-PLIP will fail on torch <2.6" >&2
fi

# Bio_ClinicalBERT ships only pytorch_model.bin, so WSI-FiVE's text tower hits
# the same torch<2.6 refusal as PLIP. Its safetensors copy drops the tied MLM
# head, which AutoModel discards anyway. Same guard: exported only when present.
BERT_LOCAL="${CLINICALBERT_CKPT:-${PGVL_STORAGE_ROOT}/dchanda/model_cache/bio_clinicalbert}"
if [[ -f "${BERT_LOCAL}/model.safetensors" ]]; then
    export CLINICALBERT_CKPT="${BERT_LOCAL}"
else
    echo "[pgvl] note: no BioClinicalBERT safetensors at ${BERT_LOCAL};" \
         "WSI-FiVE will fail on torch <2.6" >&2
fi

echo "[pgvl] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[pgvl] method=${METHOD}"
echo "[pgvl] config=${CONFIG}"
echo "[pgvl] python=$(command -v python)"
echo "[pgvl] HF_HOME=${HF_HOME} HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "[pgvl] FATAL: config does not exist: ${CONFIG}" >&2
    exit 78
fi

python - <<'PY' || exit 78
import sys
try:
    import torch
except Exception as error:                                   # noqa: BLE001
    print(f"[pgvl] FATAL: torch import failed: {error}", file=sys.stderr)
    raise SystemExit(1)
print(f"[pgvl] torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
PY

# --- run --------------------------------------------------------------------
train_args=(
    train.py --method "${METHOD}" --config "${CONFIG}" --device "${DEVICE}"
)
if [[ "${RERUN}" == "--rerun" ]]; then
    train_args+=(--rerun)
fi
python "${train_args[@]}"
status=$?
echo "[pgvl] train.py exited with status ${status}"
exit ${status}
