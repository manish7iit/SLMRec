#!/bin/bash

# Get the output directory from command line argument
OUTPUT_DIR=$1

# Check if the output directory parameter is provided
if [ -z "$OUTPUT_DIR" ]; then
  echo "Error: Output directory is required."
  echo "Usage: $0 <output_directory>"
  exit 1
fi

# Run the command with the provided output directory
NCCL_P2P_DISABLE=1 CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=1 --master_port=1234 finetune.py \
    --task_type sequential \
    --cache_dir cace_dir/ \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 256 \
    --micro_batch_size 32 \
    --num_epochs 3 \
    --learning_rate 0.001 \
    --cutoff_len 4096 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[gate_proj, down_proj, up_proj]' \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name alpaca \
    --warmup_steps 100 \
    --lr_scheduler 'cosine' \
    --llama_decoder_nums 8
