# ISL=3000
# OSL=1000
# RATIO=0
PORT=10001
# PORT=40015
MODEL_PATH=/shared-inference/models_blog/Qwen3-0.6B

CONCURRENCY=1  # "8 16 32 64 128"
PROMPTS=1
     vllm bench serve  \
        --dataset-name sharegpt \
        --dataset-path /shared-inference/models_blog/ShareGPT_V3_unfiltered_cleaned_split.json \
        --model  QWEN \
         --tokenizer $MODEL_PATH \
        --num-prompt $PROMPTS \
        --base-url "http://127.0.0.1:$PORT" \
        --backend vllm \
        --max-concurrency $CONCURRENCY