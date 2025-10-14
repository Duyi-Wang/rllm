#!/bin/bash

set -ex
curl -X POST -s http://0.0.0.0:30000/v1/completions -H "Content-Type: application/json" -d '{"model": "/apps/data/models/DSR1","prompt": "Write a summary of why scarcity and urgency are the strongest mental triggers and have been the driving force behind many of our best performing campaigns over the last 8 years.","max_tokens": 200,"temperature": 0, "top_k":1}' | awk -F'"' '{print $22}'
