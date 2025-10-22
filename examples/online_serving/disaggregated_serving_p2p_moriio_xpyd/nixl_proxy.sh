python3 "/opt/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
    --host 0.0.0.0 \
    --port 10001 \
    --prefiller-hosts 10.224.2.178 \
    --prefiller-ports 20005 \
    --decoder-hosts 10.224.3.45 \
    --decoder-ports 20005