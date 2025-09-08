#!/bin/bash
#SBATCH --account=CDERID0047
#SBATCH -J DeepSeek          # Job name
#SBATCH -o logfiles/%x_%j.out         # Standard output and error log (%x = job name, %j = job ID)
#SBATCH -e logfiles/%x_%j.err         # Error log
#SBATCH --nodes=1            # Request N nodes
#SBATCH --cpus-per-task=10     # Use 100 CPU cores per task
#SBATCH --mem=60GB             # Adjust memory as needed
#SBATCH --time=72:00:00                # Time limit (24 hours)
#SBATCH --array=5-100
##SBATCH --exclude=bc002
##SBATCH --constraint=gpu_cc_80
#SBATCH --gres=gpu:4



# Print job details for debugging
echo "Running on node $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job Name: $SLURM_JOB_NAME"


echo check for gpu: nvidia-smi output:
nvidia-smi
echo

# Get start of job information
START_TIME=`date +%s`

CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home/seyedhamed.meshkin/anaconda3/bin/python3 DeepSeek_AI2ARC_multichoice.py  --output AI2ARC_validation_results${SLURM_ARRAY_TASK_ID} --run $SLURM_ARRAY_TASK_ID  >& logfiles/"$SLURM_JOB_ID".${SLURM_JOB_NAME}.${SLURM_ARRAY_TASK_ID}.txt

# Get end of job information
EXIT_STATUS=$?
END_TIME=`date +%s`
