# curl -X POST -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "deepseek-ai/DeepSeek-R1","prompt": "Hi,how are you?","max_tokens": 10,"temperature": 0, "top_k":1}' | awk -F'"' '{print $22}'
# curl -X POST -s http://10.235.192.61:36367/v1/completions -H "Content-Type: application/json" -d '{"model": "deepseek-ai/DeepSeek-R1","prompt": "Hi,how are you?","max_tokens": 100,"temperature": 0, "top_k":1}' | awk -F'"' '{print $22}'


curl -X POST -s http://127.0.0.1:10001/v1/completions -H "Content-Type: application/json" -d '{"model": "QWEN","prompt": "the us is ?","max_tokens": 10,"temperature": 0, "top_k":1}' | awk -F'"' '{print $22}'


# curl -X POST -s http://127.0.0.1:10001/v1/completions -H "Content-Type: application/json" -d '{"model": "deepseek-ai/DeepSeek-R1","prompt": "the us is ?","max_tokens": 32000,"temperature": 0, "top_k":1}' | awk -F'"' '{print $22}'
# timeout 130 curl -X POST -s http://127.0.0.1:10001/v1/completions -H "Content-Type: application/json" -d '{"model": "deepseek-ai/DeepSeek-R1","prompt": "the us is ?","max_tokens": 32000,"temperature": 0, "top_k":1}' --max-time 10 
# echo "Start curl: $(date '+%Y-%m-%d %H:%M:%S') ($(date +%s))" 
# curl -X POST -s http://127.0.0.1:10001/v1/completions -H "Content-Type: application/json" -d '{"model": "deepseek-ai/DeepSeek-R1","prompt": "the us is ?","max_tokens": 32000,"temperature": 0, "top_k":1}' --max-time 80 -w "\nExit code: %{exitcode}\nError: %{errormsg}\n"
# echo "end curl: $(date '+%Y-%m-%d %H:%M:%S') ($(date +%s))" 

# curl -X POST -s http://127.0.0.1:10001/v1/completions -H "Content-Type: application/json" -d '{"model": "deepseek-ai/DeepSeek-R1", "prompt": "Do you know the book Traction by Gino Wickman", "temperature": 0.0, "max_tokens": 122,"top_k":1,"repetition_penalty": 1.0}' | awk -F'"' '{print $22}'

# curl -X POST -s http://127.0.0.1:10001/v1/completions -H "Content-Type: application/json" -d '{"model": "deepseek-ai/DeepSeek-R1", "prompt": "Do you know the book Traction by Gino Wickman", "temperature": 0.0, "max_tokens": 122,"top_k":1,"repetition_penalty": 1.0,"stream": True, "stream_options": {"include_usage": True}}' | awk -F'"' '{print $22}'


#ok
# curl -X POST -s http://127.0.0.1:10001/v1/completions -H "Content-Type: application/json" -d '{"model": "deepseek-ai/DeepSeek-R1", "prompt": "Do you know the book Traction by Gino Wickman", "temperature": 0.0, "max_tokens": 122,"top_k":1,"repetition_penalty": 1.0,"stream": true}' | awk -F'"' '{print $22}'
#ok
# curl -X POST -s http://127.0.0.1:10001/v1/completions -H "Content-Type: application/json" -d '{"model": "deepseek-ai/DeepSeek-R1", "prompt": "Do you know the book Traction by Gino Wickman", "temperature": 0.0, "max_tokens": 122,"top_k":1,"repetition_penalty": 1.0}' | awk -F'"' '{print $22}'
#ok
# curl -X POST -s http://127.0.0.1:10001/v1/completions -H "Content-Type: application/json" -d '{"model": "deepseek-ai/DeepSeek-R1", "prompt": "Do you know the book Traction by Gino Wickman", "temperature": 0.0, "max_tokens": 122,"top_k":1,"repetition_penalty": 1.0,"stream": true,"stream_options": {"include_usage": true}}' | awk -F'"' '{print $22}'




# curl -X POST -s http://127.0.0.1:10001/v1/completions -H "Content-Type: application/json" -d '{"model": "deepseek-ai/DeepSeek-R1","prompt": "Do you know the book Traction by Gino Wickman","max_tokens": 10,"temperature": 0, "top_k":1}' | awk -F'"' '{print $22}'

# curl -X POST -s http://127.0.0.1:10001/v1/completions -H "Content-Type: application/json" -d '{'model': 'deepseek-ai/DeepSeek-R1', 'prompt': 'Do you know the book Traction by Gino Wickman', 'temperature': 0.0, 'repetition_penalty': 1.0, 'max_tokens': 122, 'logprobs': None, 'stream': True, 'stream_options': {'include_usage': True}}' | awk -F'"' '{print $22}'


