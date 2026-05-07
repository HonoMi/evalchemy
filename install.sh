#!/bin/bash

pyenv_name=evalchemy-DF

python -m pip install -U pip setuptools wheel

# torch / torchvision / torchaudio は先に CUDA 12.8 で揃える
python -m pip install -U \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.10.0" \
    "torchvision==0.25.0" \
    "torchaudio==2.10.0"

# CUDA 12.8 環境では、vLLM 0.19.1 の generic wheel が libcudart.so.12 を使う
python -m pip install -U "vllm==0.19.1"

# transformers は別で最新 stable を入れる
python -m pip install -U "transformers==5.5.4" accelerate sentencepiece

pip install -e .
pip install -e eval/chat_benchmarks/alpaca_eval
pip install func_timeout  # to use LiveCodeBench

# lm-evaluation-harness は最後に editable install して、
# evalchemy 本体の依存で入った lm-eval をローカル版で上書きする。
if [ ! -d lm-evaluation-harness ]; then
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
fi
cd lm-evaluation-harness
pip install -e .
