
MODEL_PATH=/mnt/m2m_nobackup/models/deepseek-ai/DeepSeek-V3
# MODEL_PATH=/shared-inference/models_blog/Qwen3-0.6B
# pip install lm-eval[api]==0.4.9.1
lm_eval --model local-completions --model_args model=QWEN,tokenizer=${MODEL_PATH},base_url=http://127.0.0.1:10001/v1/completions,num_concurrent=32,max_retries=1,max_gen_toks=2048 --tasks gsm8k --num_fewshot 5 --batch_size auto  --apply_chat_template
# lm_eval --model local-completions --model_args model=QWEN,tokenizer=${MODEL_PATH},base_url=http://127.0.0.1:50005/v1/completions,num_concurrent=256,max_retries=1,max_gen_toks=1024 --tasks gsm8k --num_fewshot 5 --batch_size auto  --apply_chat_template
#PD result
# lm_eval --model local-completions --model_args model=QWEN,tokenizer=${MODEL_PATH},base_url=http://127.0.0.1:10001/v1/completions,num_concurrent=64,max_retries=1,max_gen_toks=1024 --tasks gsm8k --num_fewshot 5 --batch_size auto  --apply_chat_template

# #|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.5898|±  |0.0135|
# |     |       |strict-match    |     5|exact_match|↑  |0.3048|±  |0.0127|
# # lm_eval --model local-completions --model_args model=QWEN,tokenizer=/nfs/data/Qwen3-32B,base_url=http://127.0.0.1:50005/v1/completions,num_concurrent=256,max_retries=1,max_gen_toks=32 --tasks gsm8k --num_fewshot 5 --batch_size auto  --apply_chat_template
#2p2d
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.5785|±  |0.0136|
# |     |       |strict-match    |     5|exact_match|↑  |0.2896|±  |0.0125|


#single node
# lm_eval --model local-completions --model_args model=QWEN,tokenizer=${MODEL_PATH},base_url=http://127.0.0.1:50005/v1/completions,num_concurrent=256,max_retries=1,max_gen_toks=1024 --tasks gsm8k --num_fewshot 5 --batch_size auto  --apply_chat_template

# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.5936|±  |0.0135|
# |     |       |strict-match    |     5|exact_match|↑  |0.3033|±  |0.0127|
