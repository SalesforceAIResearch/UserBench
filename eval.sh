#!/bin/bash
# Set your OpenAI API key
export OPENAI_API_KEY=${OPENAI_API_KEY:-"your-openai-key-here"}

python eval.py \
    --model_name gpt-4o \
    --port 8000 \
    --max_turns 20 \
    --pass_k 1 \
    --temperature 0.0 \
    --envs travel22 travel33 travel44 \
    --save_name travel_gpt-4o


python eval.py \
    --model_name ${MODEL_PATH:-"path/to/your/model"} \
    --port 8000 \
    --max_turns 20 \
    --pass_k 1 \
    --temperature 0.0 \
    --envs travel22 travel33 travel44 \
    --save_name travel_local_model


python eval.py \
    --model_name gpt-4o-mini \
    --port 8000 \
    --max_turns 20 \
    --pass_k 1 \
    --temperature 0.0 \
    --envs travel22 travel33 travel44 \
    --save_name travel_gpt-4o-mini