#!/bin/bash

OUTPUT_DIR=$1

if [ -z "$OUTPUT_DIR" ]; then
  echo "Error: Output directory is required."
  exit 1
fi

CUDA_LAUNCH_BLOCKING=1 python finetune.py \
    --task_type sequential \
    --cache_dir cache_dir/ \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 32 \
    --micro_batch_size 2 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[gate_proj, down_proj, up_proj]' \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name alpaca \
    --warmup_steps 500 \
    --lr_scheduler 'cosine' \
    --llama_decoder_nums 16
