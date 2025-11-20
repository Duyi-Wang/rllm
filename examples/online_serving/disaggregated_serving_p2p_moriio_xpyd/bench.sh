 # ISL=4096*8
ISL=4090
#1000有问题,300没问题   #但是为什么16384数据才能对的上
OSL=3 #TTFT不受OSL影响  验证了
# OSL=128
RATIO=0
PORT=10001
# PORT=8023
# PORT=40005
export VLLM_TORCH_PROFILER_DIR=/nfs/users/mingzliu/vllm/examples/online_serving/disaggregated_serving_p2p_moriio_xpyd/zlogs
CONCURRENCY=1 #"8 16 32 64 128"
# MODEL_PATH=/shared-inference/models_blog/Qwen3-0.6B
MODEL_PATH=/mnt/m2m_nobackup/models/deepseek-ai/DeepSeek-V3
PROMPTS=1
      vllm bench serve  \
        --dataset-name random \
        --model  $MODEL_PATH \
        --random-input-len $ISL \
        --random-output-len $OSL \
        --tokenizer $MODEL_PATH \
        --num-prompt $PROMPTS \
        --random-range-ratio $RATIO \
        --base-url "http://127.0.0.1:$PORT" \
        --backend vllm \
        --max-concurrency $CONCURRENCY \
        # --profile \
        # --dataset-path /nfs/users/mingzliu/ShareGPT_V3_unfiltered_cleaned_split.json \




#0.6B
   #1layer youhua    16384    230ms,220ms,220ms   #perf 254ms
                    # 20480   308ms,297ms,308ms 



#32B           16384 3   421ms,401ms,412ms
#              20480 3   548ms,545ms,548ms #profile 593



#修复tokenizer问题后：
                  #268m's 287ms(70ms request延迟，0~30ms发送延迟)，266ms
                    #249ms( 总时间0.31， 32ms发送延迟) 
                    #260ms(总时间0.3 reqest发送70ms,while等待<1ms,merge<1ms,通信0，


   #         19.53.0456 bench       ，接受request70ms（19.53.1160), 100ms prefill（19.53.2128）prefill完成 发送，
   # 19.54.7694