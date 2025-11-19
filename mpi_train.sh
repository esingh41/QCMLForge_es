#!/bin/bash
#SBATCH --job-name=qcml_ddp_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=4                    # Number of nodes
#SBATCH --ntasks-per-node=4          # Number of processes per node (GPUs per node or CPU processes)
#SBATCH --cpus-per-task=8            # CPU cores per process (for OMP_NUM_THREADS)
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --mem=32GB                   # Memory per node
#SBATCH --partition=gpu              # Partition name (adjust for your cluster)
# #SBATCH --gres=gpu:4               # Uncomment if using GPUs (4 GPUs per node)

# ============================================================================
# QCMLForge Multi-Node Distributed Data Parallel Training Script
# ============================================================================
# 
# Usage:
#   sbatch mpi_train.sh
#
# Configuration:
#   - Edit SLURM directives above to match your cluster configuration
#   - Adjust training parameters in the Python command below
#   - Set environment variables as needed
#
# Examples:
#   2 nodes, 4 procs/node = 8 total processes (world_size=8)
#   1 node,  4 procs/node = 4 total processes (world_size=4)
# ============================================================================

echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Working Directory: $(pwd)"
echo "========================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================================================
# Environment Setup
# ============================================================================

# Load required modules (adjust for your cluster)
# module purge
# module load python/3.10
# module load cuda/11.8  # If using GPUs
# module load openmpi/4.1.1

# Activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate qcml

# Set OMP_NUM_THREADS to match cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to: $OMP_NUM_THREADS"

# PyTorch distributed environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo ""

# ============================================================================
# Training Configuration
# ============================================================================

# Dataset configuration
DATA_ROOT="data_dimer_2"
SPEC_TYPE=10
MAX_SIZE=None  # Set to None for full dataset or a number for subset
USE_LMDB=true
PRECOMPUTE_HFVR=true

# Pre-trained model paths
ATP_MODEL_PATH="models/ap3_ensemble/1/atp_mpnn_1.pt"

# Training hyperparameters
N_EPOCHS=500
BATCH_SIZE=16
LEARNING_RATE=5e-5
SPLIT_PERCENT=0.9

# Output model path
MODEL_SAVE_PATH="models/ap3_ensemble/2/aidm_sNN_lr_ddp_2.pt"
mkdir -p $(dirname $MODEL_SAVE_PATH)

# Dataloader configuration
NUM_WORKERS=2  # Number of dataloader workers per process
PIN_MEMORY=true

# ============================================================================
# Run Training with srun (SLURM's MPI-like launcher)
# ============================================================================

echo "========================================"
echo "Starting Distributed Training"
echo "========================================"
echo "Training with $WORLD_SIZE processes across $SLURM_JOB_NUM_NODES nodes"
echo ""

# Use srun to launch the training script on all nodes/tasks
srun python -u train_ddp_slurm.py \
    --data_root "$DATA_ROOT" \
    --atp_model_path "$ATP_MODEL_PATH" \
    --spec_type $SPEC_TYPE \
    --max_size "$MAX_SIZE" \
    --use_lmdb $USE_LMDB \
    --precompute_hfvr $PRECOMPUTE_HFVR \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --split_percent $SPLIT_PERCENT \
    --model_save_path "$MODEL_SAVE_PATH" \
    --num_workers $NUM_WORKERS \
    --omp_num_threads $OMP_NUM_THREADS

# Capture exit code
EXIT_CODE=$?

echo ""
echo "========================================"
echo "Training Complete"
echo "========================================"
echo "Exit Code: $EXIT_CODE"
echo "Model saved to: $MODEL_SAVE_PATH"
echo "========================================"

exit $EXIT_CODE
