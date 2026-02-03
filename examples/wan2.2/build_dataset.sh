python /gemini/code/VideoX-Fun/examples/wan2.2/build_moviebench_character_pair_val_manifest.py \
  --recaption_root /gemini/code/datasets/MovieBench_ByCharacter_recaptioned \
  --video_root /gemini/code/datasets/MovieBench_ByCharacter \
  --output_jsonl /gemini/code/VideoX-Fun/examples/wan2.2/moviebench_character_pair_val.jsonl \
  --num_samples 16 \
  --seed 42 \
  --pairing_strategy random