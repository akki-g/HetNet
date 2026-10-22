#!/bin/bash
# Submit from the frozen checkout with BASH_ENV/ENV removed.
#SBATCH --job-name=pcp-recovery
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0-5%6
#SBATCH --output=pcp-recovery-%A_%a.out
#SBATCH --error=pcp-recovery-%A_%a.err
set -Eeuo pipefail
trap 'printf "pcp-recovery failed at line %s (exit %s)\n" "$LINENO" "$?" >&2' ERR
(( $# == 2 || $# == 3 )) || { echo 'Usage: reconstruct|run MANIFEST [--resume]' >&2; exit 2; }
phase=$1
manifest=$2
shift 2
case "$phase" in
  reconstruct)
    [[ ${SLURM_ARRAY_TASK_ID:-} =~ ^[01]$ && $# == 0 ]] || { echo 'Reconstruct expects index 0..1 and no resume' >&2; exit 2; } ;;
  run)
    [[ ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-5]$ ]] || { echo 'Run expects index 0..5' >&2; exit 2; }
    (( $# == 0 )) || [[ $1 == --resume ]] || { echo 'Only --resume is supported for run' >&2; exit 2; } ;;
  *) echo 'Expected phase reconstruct or run' >&2; exit 2 ;;
esac
[[ ${SLURM_ARRAY_JOB_ID:-} =~ ^[1-9][0-9]*$ ]] || { echo 'Expected a Slurm array job ID' >&2; exit 2; }
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen repository root}"
[[ -f $manifest ]] || { echo 'Recovery manifest does not exist' >&2; exit 2; }
export CONDA_SHLVL=0
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
for conda_prefix_name in "${!CONDA_PREFIX_@}"; do unset "$conda_prefix_name"; done
if ! type module >/dev/null 2>&1; then
  for module_init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
    if [[ -r $module_init ]]; then
      set +u
      source "$module_init"
      set -u
      type module >/dev/null 2>&1 && break
    fi
  done
fi
type module >/dev/null 2>&1 || { echo 'pcp-recovery: module initialization unavailable' >&2; exit 2; }
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
[[ -x $HETNET_PYTHON ]] || { echo 'pcp-recovery: pinned interpreter is not executable' >&2; exit 2; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:$PWD/envs"
printf 'pcp-recovery: job=%s phase=%s index=%s node=%s\n' \
  "$SLURM_ARRAY_JOB_ID" "$phase" "$SLURM_ARRAY_TASK_ID" "${SLURM_JOB_NODELIST:-unknown}" >&2
exec srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole.pcp_recovery "$phase" \
  --manifest "$manifest" --index "$SLURM_ARRAY_TASK_ID" "$@"
