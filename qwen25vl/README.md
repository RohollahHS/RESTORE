# RESTORE for Qwen2.5-VL

This folder applies RESTORE to [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (Table 3 of the paper).
It is self-contained and uses its own conda environment, separate from the LLaVA-1.5 code in the repository root.

## Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Results](#results)
- [Implementation](#implementation)

## Installation

1. Clone this repository and move to this folder

```bash
git clone https://github.com/cvlab-yonsei/RESTORE.git
cd RESTORE/qwen25vl
```

2. Create and set up a conda environment

   Qwen2.5-VL requires a newer `transformers` than LLaVA, so please use a separate environment:

```bash
# Create conda env
conda create -n restore_qwen python=3.10 -y
conda activate restore_qwen

# PyTorch (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Qwen2.5-VL
pip install transformers==4.51.3 accelerate==1.13.0 qwen-vl-utils==0.0.14

# Evaluation
pip install pandas openpyxl shortuuid scikit-learn tqdm
```

## Dataset

We use the same benchmarks as the LLaVA-1.5 experiments and follow the evaluation protocol of [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).
Please download and prepare each benchmark as described there. If you have already prepared `RESTORE/playground` for LLaVA-1.5, you can simply link it:

```bash
# in RESTORE/qwen25vl
ln -s ../playground playground
```

The Qwen2.5-VL experiments use five benchmarks, and the directory should look like:

```
qwen25vl/
└── playground/
    └── data/
        └── eval/
            ├── mmbench/
            │   └── mmbench_dev_20230712.tsv
            ├── MME/
            │   ├── llava_mme.jsonl
            │   ├── MME_Benchmark_release_version/
            │   └── eval_tool/
            ├── pope/
            │   ├── llava_pope_test.jsonl
            │   ├── val2014/
            │   └── coco/
            ├── scienceqa/
            │   ├── llava_test_CQM-A.json
            │   ├── images/test/
            │   ├── problems.json
            │   └── pid_splits.json
            └── textvqa/
                ├── llava_textvqa_val_v051_ocr.jsonl
                ├── train_images/
                └── TextVQA_0.5.1_val.json
```

## Evaluation

We recommend setting the following environment variables so that the Qwen2.5-VL-7B-Instruct checkpoint is downloaded (and cached) to the path of your choice:

```bash
export HF_HOME=<path_to_checkpoint>
export HF_HUB_CACHE=<path_to_checkpoint>
```

Each benchmark is evaluated through the scripts under `scripts/`, run from this folder (`RESTORE/qwen25vl`). The reduction behavior is controlled by three arguments:

- `--base` &mdash; the visual token reduction baseline: `HoloV`, `DivPrune`, or `VisPruner`.
- `--retain-ratio` &mdash; the fraction of retained visual tokens (e.g., `0.333`, `0.222`, `0.111`). Qwen2.5-VL encodes images at their native resolution, so the number of visual tokens varies per image and a ratio is used instead of `--n_vis`.
- `--restore` &mdash; when present, applies **RESTORE** on top of the baseline (rectifying distortions via token size and distance calibration). When omitted, the plain baseline reduction is used.

Every benchmark runs on a single GPU.

```bash
# MME with RESTORE
CUDA_VISIBLE_DEVICES=0 bash scripts/mme.sh \
    --base HoloV --retain-ratio 0.111 --restore

# Baseline only (drop --restore)
CUDA_VISIBLE_DEVICES=0 bash scripts/mme.sh \
    --base HoloV --retain-ratio 0.111
```

Available benchmark scripts: `mmbench.sh`, `mme.sh`, `pope.sh`, `sqa.sh`, `textvqa.sh`.
For MMBench, the [submission server](https://mmbench.opencompass.org.cn/mmbench-submission) has been closed, so following its guidance, we evaluate the predictions in `playground/data/eval/mmbench/answers/` with [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) (`MMBench_DEV_EN`, CircularEval, GPT-4o-2024-11-20 as the choice extractor).

## Results

The numbers below are obtained with this code and differ from those reported in Table 3 of the paper. Please refer to these numbers.

| Method | Average | MMB | MME | POPE | SQA | TextVQA |
|---|---:|---:|---:|---:|---:|---:|
| ***Using all visual tokens*** | | | | | | |
| Qwen2.5-VL-7B-Instruct | 100.0% | 83.5 | 2327 | 86.7 | 78.2 | 77.6 |
| ***Retain 33.3% visual tokens*** | | | | | | |
| DivPrune (CVPR 25) | 95.9% | 79.6 | 2169 | 84.5 | 77.0 | 74.0 |
| &nbsp;&nbsp;+ RESTORE | 97.5% | 81.2 | 2197 | 86.1 | 79.2 | 73.8 |
| VisPruner (ICCV 25) | 96.4% | 80.5 | 2161 | 84.8 | 77.4 | 74.5 |
| &nbsp;&nbsp;+ RESTORE | 97.5% | 81.1 | 2193 | 86.0 | 78.3 | 74.9 |
| HoloV (NeurIPS 25) | 96.4% | 79.9 | 2134 | 85.2 | 77.8 | 75.1 |
| &nbsp;&nbsp;+ RESTORE | **97.5%** | 80.8 | 2206 | 86.0 | 78.3 | 75.0 |
| ***Retain 22.2% visual tokens*** | | | | | | |
| DivPrune (CVPR 25) | 93.4% | 78.4 | 2071 | 83.6 | 75.3 | 71.3 |
| &nbsp;&nbsp;+ RESTORE | 94.7% | 78.4 | 2121 | 85.0 | 77.5 | 70.9 |
| VisPruner (ICCV 25) | 93.9% | 77.6 | 2077 | 83.4 | 76.9 | 72.0 |
| &nbsp;&nbsp;+ RESTORE | **95.4%** | 79.8 | 2128 | 85.0 | 77.9 | 71.8 |
| HoloV (NeurIPS 25) | 93.8% | 77.1 | 2067 | 83.3 | 77.0 | 72.3 |
| &nbsp;&nbsp;+ RESTORE | 95.3% | 79.5 | 2121 | 85.1 | 77.9 | 71.8 |
| ***Retain 11.1% visual tokens*** | | | | | | |
| DivPrune (CVPR 25) | 88.4% | 72.0 | 1971 | 79.9 | 73.9 | 65.4 |
| &nbsp;&nbsp;+ RESTORE | **90.2%** | 76.5 | 1998 | 81.8 | 76.4 | 63.2 |
| VisPruner (ICCV 25) | 86.4% | 71.5 | 1832 | 78.0 | 73.5 | 65.0 |
| &nbsp;&nbsp;+ RESTORE | 89.6% | 74.4 | 1984 | 81.3 | 76.9 | 63.4 |
| HoloV (NeurIPS 25) | 87.3% | 71.6 | 1849 | 80.5 | 74.2 | 65.0 |
| &nbsp;&nbsp;+ RESTORE | 90.1% | 74.6 | 1997 | 82.1 | 76.9 | 64.1 |

- Average is the mean of each benchmark score divided by that of the full model (Qwen2.5-VL-7B-Instruct with all visual tokens), as in the paper.
- MMB is MMBench-dev (EN), evaluated with VLMEvalKit as described in [Evaluation](#evaluation).
- SQA is the median of three runs, since its decoding samples with a near-zero temperature.

## Implementation

- [`qwen/model/modeling_qwen2_5_vl.py`](qwen/model/modeling_qwen2_5_vl.py) is the Qwen2.5-VL model with a single hook for visual token reduction.
  - The vision encoder also returns the last-layer attention score of each visual token, which the attention-based baselines (HoloV, VisPruner) use.
  - Right after the image features are placed in the input embeddings, `model.reduction` reduces the visual tokens of the prefill.
- [`qwen/tore`](qwen/tore) implements the baseline visual token reduction methods and RESTORE.
  - [`holov.py`](qwen/tore/holov.py), [`divprune.py`](qwen/tore/divprune.py), [`vispruner.py`](qwen/tore/vispruner.py): baselines. Each is a `select(image_tokens, image_attn, n_retain)` function registered in `BASES` ([`__init__.py`](qwen/tore/__init__.py)).
  - [`merge.py`](qwen/tore/merge.py): our merging with distinctive anchor token selection.
  - [`restore.py`](qwen/tore/restore.py): RESTORE for M-RoPE.
    - Each merged token is placed at the mean of its source tokens' 3D (temporal, height, width) positions.
    - The attention calibration uses the M-RoPE decay of Appendix A, `D = Σ_g w_g D_g` over the three axes (`mrope_section = [16, 24, 24]`).
  - [`reduction.py`](qwen/tore/reduction.py): applies a baseline and, optionally, RESTORE to the visual tokens. With `--restore`, half of the budget is selected by the baseline and the other half is merged from the pruned tokens.
    The prefill attention mask is always passed to SDPA explicitly (the calibrated mask with `--restore`, a plain causal mask without it), so both settings run the same attention kernel.
- [`qwen/eval`](qwen/eval) contains the inference scripts (`model_vqa_*.py`) and the scoring scripts taken from LLaVA (`eval_*.py`).
- [`scripts/convert_answer_to_mme.py`](scripts/convert_answer_to_mme.py) is the MME conversion script of LLaVA, modified to handle answers that contain line breaks.

The baselines are organized in a modular form, so a new baseline reduction method can be added by registering its `select` function in `BASES`.
