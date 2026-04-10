TEACHER_CHECKPOINT=$1
OUTPUT_DIR=$2

if [ -z "$TEACHER_CHECKPOINT" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <teacher_checkpoint_path> <output_directory>"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 python distill.py \
    --task_type sequential \
    --cache_dir cache_dir/ \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 32 \
    --micro_batch_size 2 \
    --num_epochs 3 \
    --max_steps 1000 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules "gate_proj,down_proj,up_proj" \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name alpaca \
    --warmup_steps 500 \
    --lr_scheduler cosine \
    --llama_decoder_nums_teacher 8 \
    --llama_decoder_nums_student 4 \
    --teacher_resume_from_checkpoint "$TEACHER_CHECKPOINT" \
    --distill_lambda 0.5 \
    --save_steps 500 \
    --domain_type music