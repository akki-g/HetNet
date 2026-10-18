#!/bin/bash
# Submit from the repository root; omit inherited BASH_ENV/ENV at submission.
#SBATCH --job-name=hetnet-resume
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs_1/hetnet-resume-%A_%a.out
#SBATCH --error=logs_1/hetnet-resume-%A_%a.err
set -Eeuo pipefail
trap 'printf "hetnet-resume failed at line %s (exit %s)\n" "$LINENO" "$?" >&2' ERR
printf 'hetnet-resume starting: job=%s node=%s\n' "${SLURM_JOB_ID:-local}" "${SLURM_JOB_NODELIST:-unknown}" >&2
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

# A batch shell has not activated the submitting shell's Conda environment.
# Stale activation counters can make module initialization repeatedly deactivate.
export CONDA_SHLVL=0
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
for conda_prefix_name in "${!CONDA_PREFIX_@}"; do
  unset "$conda_prefix_name"
done
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
type module >/dev/null 2>&1 || { echo 'hetnet-resume: module initialization unavailable' >&2; exit 2; }
echo 'hetnet-resume: loading anaconda/anaconda-2024.10' >&2
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch
printf 'hetnet-resume: launching %s\n' "$HETNET_PYTHON" >&2
exec srun --cpu-bind=cores "$HETNET_PYTHON" -u -m publication_reconstruction resume --wall-seconds 165600 --episode-log stdout "$@"
