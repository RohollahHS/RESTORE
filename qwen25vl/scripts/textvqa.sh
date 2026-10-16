#!/bin/bash

CKPT="Qwen2.5-VL-7B-Instruct"

python -m qwen.eval.model_vqa_loader \
    --model-path Qwen/$CKPT \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl \
    --temperature 0 \
    "$@"

python -m qwen.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl
