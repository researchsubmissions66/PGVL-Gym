#!/bin/bash
# Launch the PGVL-Gym campaign.
#
# Thin wrapper over scripts/run_benchmark.py, which is resume-aware: it submits
# only the runs that are ready and not already complete or queued, and skips the
# rest with a recorded reason instead of stopping.
#
#   ./launch_pgvl.sh --dry-run              # show the plan, submit nothing
#   ./launch_pgvl.sh                        # submit everything outstanding
#   ./launch_pgvl.sh --regenerate           # rebuild configs first, then submit
#   ./launch_pgvl.sh --cohort brca --limit 3   # canary batch
#
# Safe to re-run: finished runs are left alone, interrupted runs resume.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set PGVL_CONDA_ENV to the environment built from environment.yml once it
# exists; scripts/pgvl_job.sh activates it on the compute node.
export PGVL_CONDA_ENV="${PGVL_CONDA_ENV:-}"

exec python3 "${REPO}/scripts/run_benchmark.py" "$@"
