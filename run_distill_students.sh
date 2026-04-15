TEACHER_CHECKPOINT=$1
OUTPUT_ROOT=$2

if [ -z "$TEACHER_CHECKPOINT" ] || [ -z "$OUTPUT_ROOT" ]; then
  echo "Usage: $0 <teacher_checkpoint_path> <output_root>"
  exit 1
fi

bash run_distill.sh "$TEACHER_CHECKPOINT" "$OUTPUT_ROOT/qwen25_3b_layers4_lambda05_seed42" 4 0.5 42 "Qwen/Qwen2.5-3B"
bash run_distill.sh "$TEACHER_CHECKPOINT" "$OUTPUT_ROOT/phi2_layers6_lambda03_seed123" 6 0.3 123 "  "
bash run_distill.sh "$TEACHER_CHECKPOINT" "$OUTPUT_ROOT/stablelm3b_layers8_lambda07_seed2024" 8 0.7 2024 "stabilityai/stablelm-3b-4e1t"
