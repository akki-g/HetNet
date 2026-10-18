#!/bin/bash
# Fresh seeds 3,4,5: main 0..5; optional beta0.5 stability group 6..8.
# Submit from the frozen checkout with inherited BASH_ENV/ENV removed.
#SBATCH --job-name=pcp-add-seeds
#SBATCH --account=cenyioha
#SBATCH --partition=normal
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0-5%6
#SBATCH --output=pcp-add-seeds-%A_%a.out
#SBATCH --error=pcp-add-seeds-%A_%a.err
set -Eeuo pipefail
trap 'printf "pcp-add-seeds failed at line %s (exit %s)\n" "$LINENO" "$?" >&2' ERR
(( $# == 0 )) || { echo 'No script arguments: scientific settings are fixed; select indices with the scheduler array option.' >&2; exit 2; }
[[ ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-8]$ ]] || { echo 'Expected array index 0..8' >&2; exit 2; }
[[ ${SLURM_ARRAY_JOB_ID:-} =~ ^[1-9][0-9]*$ ]] || { echo 'Expected a Slurm array job ID' >&2; exit 2; }
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

# Same startup recovery as publication_resume.sbatch; no login shell or conda init.
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
type module >/dev/null 2>&1 || { echo 'pcp-add-seeds: module initialization unavailable' >&2; exit 2; }
module load anaconda/anaconda-2024.10
export HETNET_PYTHON="${HETNET_PYTHON:-$PWD/.venv/bin/python}"
[[ -x $HETNET_PYTHON ]] || { echo 'pcp-add-seeds: pinned interpreter is not executable' >&2; exit 2; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 DGLBACKEND=pytorch PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:$PWD/envs"
printf 'pcp-add-seeds: job=%s index=%s node=%s\n' "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "${SLURM_JOB_NODELIST:-unknown}" >&2
exec srun --cpu-bind=cores "$HETNET_PYTHON" -u -m softrole.pcp_additional_seeds \
  --index "$SLURM_ARRAY_TASK_ID" \
  --run-root "$PWD/runs/pcp_additional_seeds_v1_${SLURM_ARRAY_JOB_ID}" \
  --capcon-manifest "${PCP_ADDITIONAL_BASE_MANIFEST:-$PWD/runs/capcon_l1_40m_20261017/manifest.json}"
