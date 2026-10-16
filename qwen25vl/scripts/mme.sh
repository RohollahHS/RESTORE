#!/bin/bash

CKPT="Qwen2.5-VL-7B-Instruct"
ROOT=$(pwd)

python -m qwen.eval.model_vqa_loader \
    --model-path Qwen/$CKPT \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$CKPT.jsonl \
    --temperature 0 \
    "$@"

cd ./playground/data/eval/MME

python "$ROOT"/scripts/convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py --results_dir answers/$CKPT
