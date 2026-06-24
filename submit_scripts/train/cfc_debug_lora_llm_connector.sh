#!/bin/bash

#SBATCH --job-name=cfc_debug_lora_llm_connector
#SBATCH --partition=vision-beery
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=28
#SBATCH --time=3-00:00:00
#SBATCH --mem=400G
#SBATCH --output=/data/vision/beery/scratch/kai/molmo2/logs/cfc_debug_lora_llm_connector_%j.log

export PYTHONPATH=$PWD
export MASTER_PORT=$((29500 + RANDOM % 1000))
export NCCL_TIMEOUT_MINUTES=20
export MOLMO_DATA_DIR="data"
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
NAME=cfc_all_real
DEBUG_NAME=cfc_debug_lora_llm_connector
CKPT=Molmo2-8B

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
  --format=csv -l 30 > logs/gpu_monitor_${SLURM_JOB_ID}.csv &
NVIDIA_PID=$!

# Node-level CPU memory monitor (10s cadence). Columns: ts,total,used,free,available (MB)
(
  echo "ts,total_mb,used_mb,free_mb,available_mb"
  while true; do
    free -m | awk -v ts="$(date +%s)" '/^Mem:/ {print ts","$2","$3","$4","$7}'
    sleep 10
  done
) > logs/mem_monitor_${SLURM_JOB_ID}.csv &
MEM_PID=$!

torchrun --nproc-per-node 4 \
 --master_port=$MASTER_PORT \
 launch_scripts/sft.py \
 $CKPT \
 $NAME \
 --device_batch_size 1 \
 --name $DEBUG_NAME \
 --save_folder runs/$DEBUG_NAME \
 --num_workers 2 \
 --lora --lora_rank 64 \
 --lora_connector \
 --train_split 'train-v2' \
 --val_split 'validation-v2' \
 max_duration=1

kill $NVIDIA_PID 2>/dev/null
kill $MEM_PID 2>/dev/null
