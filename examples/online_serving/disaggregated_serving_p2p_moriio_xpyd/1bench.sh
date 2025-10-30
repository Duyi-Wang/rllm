# ISL=4096
ISL=4096
#1000有问题,300没问题
OSL=3
RATIO=0
# PORT=10001
PORT=50005

CONCURRENCY=1 # "8 16 32 64 128"
PROMPTS=1
      vllm bench serve  \
        --dataset-name random \
        --model  QWEN \
        --tokenizer /nfs/data/Qwen3-32B \
        --random-input-len $ISL \
        --random-output-len $OSL \
        --num-prompt $PROMPTS \
        --random-range-ratio $RATIO \
        --base-url "http://10.194.132.10:$PORT" \
        --backend vllm \
        --max-concurrency $CONCURRENCY \
        # --profile
        # --dataset-path /nfs/users/mingzliu/ShareGPT_V3_unfiltered_cleaned_split.json \
#1bench  0.6B 16384  3  128ms 114ms.32 114.62ms  perf176ms
#              20480 3   206ms 192ms  199ms

#1bench 32B  16384  3    194ms, 184ms, 183ms
           # 20480 3     275ms, 273ms, 277ms


           #25600 3   219ms,257ms,249ms



#修复tokenizer问题后：
      #16384    #145ms 161ms  158ms 142(0.22)  143(0.22) ，     #远端发送   147ms（发送60ms，总时间0.22)，162ms(发送60ms,总时间0.24)， 148ms（发送60ms，总时间0.22）,144.08(本机发送6ms,144ms，总时间0.18)，119ms(本机发送5ms,总时间0.17)


               #162ms(发送60模式，总时间24ms, 结束到发回来35ms)


      #20480   #167ms(0.17)   170ms(0.17)
