#!/usr/bin/env bash
set -x

# Check if SLURM_JOB_NAME contains "interactive" or "debug"
if [[ "$SLURM_JOB_NAME" == *interactive* || "$SLURM_JOB_NAME" == *debug* ]]; then
    export REMOTE_DEBUG=1
else
    export REMOTE_DEBUG=0
fi

# export REMOTE_DEBUG=0
echo "REMOTE_DEBUG: $REMOTE_DEBUG"

START_TIME_LOCAL=$(date +%s)
OFFSET=$((10 * 3600 + 30 * 60))
TARGET_TIME=$((START_TIME_LOCAL + OFFSET))
echo "START TIME: $(date -d "@$TARGET_TIME")"
export START_TIME=$(date +%s)

############### DATA ################
SRUN_ARGS="--ntasks=$SLURM_NNODES --ntasks-per-node=1"
XDG_CACHE_HOME_TEMP=${SLURM_TMPDIR}/.cache
XDG_CONFIG_HOME_TEMP=${SLURM_TMPDIR}/.config
TRITON_CACHE_DIR_TEMP=${SLURM_TMPDIR}/.cache/triton
TMPDIR_TEMP=${SLURM_TMPDIR}/.cache/tmp
VLLM_CACHE_ROOT_TEMP=${SLURM_TMPDIR}/vllm_cache
VLLM_CONFIG_ROOT_TEMP=${SLURM_TMPDIR}/vllm_config
VLLM_ASSETS_CACHE_TEMP=${SLURM_TMPDIR}/assets_cache
FLASHINFER_WORKSPACE_BASE_TEMP=${SLURM_TMPDIR}/flashinfer
srun $SRUN_ARGS mkdir -p $XDG_CACHE_HOME_TEMP
srun $SRUN_ARGS mkdir -p $XDG_CONFIG_HOME_TEMP
srun $SRUN_ARGS mkdir -p $TRITON_CACHE_DIR_TEMP
srun $SRUN_ARGS mkdir -p $TMPDIR_TEMP
srun $SRUN_ARGS mkdir -p $VLLM_CACHE_ROOT_TEMP
srun $SRUN_ARGS mkdir -p $VLLM_CONFIG_ROOT_TEMP
srun $SRUN_ARGS mkdir -p $VLLM_ASSETS_CACHE_TEMP
srun $SRUN_ARGS mkdir -p $FLASHINFER_WORKSPACE_BASE_TEMP
export XDG_CACHE_HOME=${XDG_CACHE_HOME_TEMP}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME_TEMP}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR_TEMP}
export TMPDIR=${TMPDIR_TEMP}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT_TEMP}
export VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT_TEMP}
export VLLM_ASSETS_CACHE=${VLLM_ASSETS_CACHE_TEMP}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE_TEMP}
################################## ENV ##################################
if [[ "$HEADER" == *"narv"* ]]; then
    echo "narv"
    VENVS_DIR=${PROJECTS_DIR}/venvs
else
    VENVS_DIR=${SCRATCH}/venvs
    echo "not narv"
fi

LOGIN_HOST="${SLURM_SUBMIT_HOST}"
ssh "$LOGIN_HOST" "
    nohup ssh '$node1' </dev/null >/dev/null 2>&1 &
    nohup proxy --hostname 0.0.0.0 --port 8899 </dev/null >/dev/null 2>&1 &
" </dev/null >/dev/null 2>&1

export http_proxy=http://${node1}:8899
export https_proxy=$http_proxy
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$http_proxy
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

cd ${PROJECTS_DIR}/RESTORE
deactivate
module purge all
module load StdEnv/2023 gcc/12.3
module load cuda/12.6
module load java
source ${VENVS_DIR}/qwen/bin/activate

K="64"
TASKS=refcoco,refcoco+,refcocog
OPTIONAL_ARGS=""
CANCEL=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --k)
      K="$2"
      shift 2
      ;;
    --t|--tasks)
      TASKS="$2"
      shift 2
      ;;
    --c|--cancel)
      CANCEL="$2"
      shift 2
      ;;
    --a|--args)
      OPTIONAL_ARGS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# ========================================================

REPO_ROOT=${PROJECTS_DIR}/RESTORE
LMMS_ROOT="${REPO_ROOT}/lmms-eval"

NPROC="${NPROC:-4}"
PORT="${PORT:-29641}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUT_ROOT=/scratch/rohhs/projects/voco_llava/RESTORE

OUT="${OUT_ROOT}/${TASKS}/k${K}"

export BBOX_COORD_FORMAT=pixel
export REFCOCO_PROMPT_STYLE=nuwa

export PYTHONPATH="${REPO_ROOT}:${LMMS_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH="${REPO_ROOT}/tokenreduction:${LMMS_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH="${REPO_ROOT}/transformers/src/transformers:${LMMS_ROOT}:${PYTHONPATH:-}"

MODEL_ARGS_LIST=(
  "pretrained=${MODEL}"
  "use_cache=true"
)
MODEL_ARGS="$(IFS=,; echo "${MODEL_ARGS_LIST[*]}")"

echo "[RESTORE] Qwen2.5-VL-7B | tasks=${TASKS} | K=${K} | output=${OUT}"
cd "${LMMS_ROOT}"

OUTPUT_DIR=${SCRATCH}/projects/voco_llava
export SAVE_LOGS_DIR=${OUTPUT_DIR}/RESTORE/eval_logs

export GPUS_PER_NODE=$(nvidia-smi -L | wc -l)

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export LMMS_EVAL_DATASETS_CACHE="/scratch/rohhs/huggingface/datasets"

export QA_ID="RESTORE_k${K}_refcoco_refcocoplus_refcocog_pixel_nuwa"

NUM_MACHINES=${SLURM_NNODES}
NUM_PROCESSES=$((GPUS_PER_NODE * NUM_MACHINES))
MAIN_PROCESS_IP=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MAIN_PROCESS_PORT=29500
MACHINE_RANK=${SLURM_NODEID}

export BASE="HoloV"
export N_VIS=$K
export RESTORE=1

accelerate launch --num_processes=${GPUS_PER_NODE} -m lmms_eval \
  --model qwen2_5_vl \
  --model_args ${MODEL_ARGS} \
  --tasks ${TASKS} \
  --batch_size 1 \
  --log_samples \
  --output_path ${SAVE_LOGS_DIR} \
  ${OPTIONAL_ARGS}


if (( CANCEL == 1 )); then
  scancel $SLURM_JOB_ID
fi


################ SBATCH
# cat scripts/headers/${HEADER}.sh scripts/eval_qwen2_5_vl_refcoco.sh > myjob.sh
# k=128
# sbatch --nodes=2 --job-name=${HEADER}_RESTORE_k${k} --time=12:00:00 myjob.sh --k $k

# cat scripts/headers/${HEADER}.sh scripts/eval_qwen2_5_vl_refcoco.sh > myjob.sh
# k=256
# sbatch --nodes=2 --job-name=${HEADER}_RESTORE_k${k} --time=12:00:00 myjob.sh --k $k

# cat scripts/headers/${HEADER}.sh scripts/eval_qwen2_5_vl_refcoco.sh > myjob.sh
# k=512
# sbatch --nodes=2 --job-name=${HEADER}_RESTORE_k${k} --time=12:00:00 myjob.sh --k $k

################ INTERACTIVE

##### bash ${PROJECTS_DIR}/RESTORE/scripts/eval_qwen2_5_vl_refcoco.sh --k 128 --c 1 --tasks refcoco  2>&1 | tee "$SLURM_OUTPUTS/RESTORE_k128_refcoco.log"
##### bash ${PROJECTS_DIR}/RESTORE/scripts/eval_qwen2_5_vl_refcoco.sh --k 128 --c 1 --tasks refcoco+ 2>&1 | tee "$SLURM_OUTPUTS/RESTORE_k128_refcoco+.log"
##### bash ${PROJECTS_DIR}/RESTORE/scripts/eval_qwen2_5_vl_refcoco.sh --k 128 --c 1 --tasks refcocog 2>&1 | tee "$SLURM_OUTPUTS/RESTORE_k128_refcocog.log"

##### bash ${PROJECTS_DIR}/RESTORE/scripts/eval_qwen2_5_vl_refcoco.sh --k 256 --c 1 --tasks refcoco  2>&1 | tee "$SLURM_OUTPUTS/RESTORE_k256_refcoco.log"
##### bash ${PROJECTS_DIR}/RESTORE/scripts/eval_qwen2_5_vl_refcoco.sh --k 256 --c 1 --tasks refcoco+ 2>&1 | tee "$SLURM_OUTPUTS/RESTORE_k256_refcoco+.log"
##### bash ${PROJECTS_DIR}/RESTORE/scripts/eval_qwen2_5_vl_refcoco.sh --k 256 --c 1 --tasks refcocog 2>&1 | tee "$SLURM_OUTPUTS/RESTORE_k256_refcocog.log"

##### bash ${PROJECTS_DIR}/RESTORE/scripts/eval_qwen2_5_vl_refcoco.sh --k 512 --c 1 --tasks refcoco  2>&1 | tee "$SLURM_OUTPUTS/RESTORE_k512_refcoco.log"
##### bash ${PROJECTS_DIR}/RESTORE/scripts/eval_qwen2_5_vl_refcoco.sh --k 512 --c 1 --tasks refcoco+ 2>&1 | tee "$SLURM_OUTPUTS/RESTORE_k512_refcoco+.log"
##### bash ${PROJECTS_DIR}/RESTORE/scripts/eval_qwen2_5_vl_refcoco.sh --k 512 --c 1 --tasks refcocog 2>&1 | tee "$SLURM_OUTPUTS/RESTORE_k512_refcocog.log"

# bash ${PROJECTS_DIR}/RESTORE/scripts/eval_qwen2_5_vl_refcoco.sh --k 1048 --c 1 --tasks refcoco+ 2>&1 | tee "$SLURM_OUTPUTS/RESTORE_k1048_refcoco+.log"

################ RECURSIVE

# k=512
# bash ${PROJECTS_DIR}/RESTORE/recursive_interactive_eval.sh --k $k
