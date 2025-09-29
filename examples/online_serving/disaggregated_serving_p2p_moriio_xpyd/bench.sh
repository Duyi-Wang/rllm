# ISL=4096*8
ISL=8192
#1000有问题,300没问题   #但是为什么16384数据才能对的上
OSL=3  #TTFT不受OSL影响  验证了
# OSL=128
RATIO=0
PORT=10001
# PORT=40005
export VLLM_TORCH_PROFILER_DIR=/nfs/users/mingzliu/vllm/examples/online_serving/disaggregated_serving_p2p_moriio_xpyd/zlogs
CONCURRENCY=1 #"8 16 32 64 128"
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
        # --profile \
        # --dataset-path /nfs/users/mingzliu/ShareGPT_V3_unfiltered_cleaned_split.json \
#bench  0.6B 16384 3  , 303ms, 269ms,254ms
   #         20480 3  , 398ms, 365ms , 342ms        profile1000ms



