#!/bin/bash
#SBATCH --account=CDERID0047
#SBATCH --job-name=llm_model
#SBATCH --nodes=1               # Request 4 nodes
#SBATCH --ntasks-per-node=1     # 1 task per node
#SBATCH --cpus-per-task=48      # Use 100 CPU cores per task
#SBATCH --mem=300GB             # Adjust memory as needed
#SBATCH --time=24:00:00         # Set max runtime (HH:MM:SS)
#SBATCH --output=LOGs/output_%j.log         # Output log file
#SBATCH --error=LOGs/llm_error.log   # Log errors


# Print job details for debugging
echo "Running on node $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
#echo "Task ID: $SLURM_ARRAY_TASK_ID"

#module load anaconda/2023.10  # Load Conda if needed
#source activate myenv         # Activate virtual environment

# Set MASTER_ADDR and MASTER_PORT
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501  # Can be any free port

# Set distributed training variables
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

echo "SLURM_JOB_NODELIST is " $SLURM_JOB_NODELIST
echo "MASTER_ADDR is " $MASTER_ADDR
echo "MASTER_PORT is " $MASTER_PORT
echo "WORLD_SIZE is " $WORLD_SIZE
echo "RANK is " $RANK
echo "SLURM_NTASKS_PER_NODE is " $SLURM_NTASKS_PER_NODE
echo "SLURM_NNODES is " $SLURM_NNODES

srun hostname
# Run the model using torchrun for multi-node execution
# Launch PyTorch Distributed
srun ~/anaconda3/bin/torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_NTASKS_PER_NODE --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT cpu.py
