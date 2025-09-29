# ISL=4096
ISL=20480
#1000有问题,300没问题
OSL=3
RATIO=0
# PORT=10001
PORT=40005

CONCURRENCY=1 # "8 16 32 64 128"
PROMPTS=1
      vllm bench serve  \
        --dataset-name random \
        --model  deepseek-ai/DeepSeek-R1 \
        --random-input-len $ISL \
        --random-output-len $OSL \
        --num-prompt $PROMPTS \
        --random-range-ratio $RATIO \
        --base-url "http://127.0.0.1:$PORT" \
        --backend vllm \
        --max-concurrency $CONCURRENCY \
        # --profile
        # --dataset-path /nfs/users/mingzliu/ShareGPT_V3_unfiltered_cleaned_split.json \
#1bench  0.6B 16384  3  128ms 114ms.32 114.62ms
#              20480 3   206ms 192ms  199ms