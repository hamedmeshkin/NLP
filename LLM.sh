#!/bin/bash
#SBATCH --account=CDERID0047
#SBATCH -J DeepSeek          # Job name
#SBATCH -o logfiles/DeepSeek_%j.out         # Standard output and error log (%x = job name, %j = job ID)
#SBATCH -e logfiles/DeepSeek_%j.err         # Error log
#SBATCH --nodes=1            # Request N nodes
#SBATCH --cpus-per-task=1    # Use 100 CPU cores per task
#SBATCH --mem=70GB           # Adjust memory as needed
#SBATCH --time=10:00:00      # Time limit (10 hours)

####SBATCH --exclude=bc002
#SBATCH --constraint=gpu_mem_80
#SBATCH --gres=gpu:a100:4


# Print job details for debugging
echo "Running on node $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"



echo check for gpu: nvidia-smi output:
nvidia-smi
echo

# Get start of job information
START_TIME=`date +%s`

CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


#below from my old scripts, using Mike's python and libraries
#source /projects/mikem/applications/centos7/ananconda3/set-env.sh
#source /projects/mikem/opt/cuda/set_env.sh
#export LD_LIBRARY_PATH=/app/GPU/CUDA-Toolkit/cuda-11.1/cudnn/cuda/lib64:$LD_LIBRARY_PATH

#below from Hamed
source /projects01/mikem/Python-3.11/set_env.sh
export PYTHONPATH=/projects01/dars-comp-bio/anaconda3/lib/python3.11/site-packages/:$PYTHONPATH
LD_LIBRARY_PATH=/home/mikem/lib:$LD_LIBRARY_PATH

mkdir logfiles

python3 DeepSeek.py  --output AI2ARC_validation_results  >& logfiles/"$SLURM_JOB_ID".o.txt

# Get end of job information
EXIT_STATUS=$?
END_TIME=`date +%s`
