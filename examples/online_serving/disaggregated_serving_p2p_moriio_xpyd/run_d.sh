#!/bin/bash
# taskset -c 0 "$0" "$@"
# pkill -9 -f python
sleep 3
# git pull
# TARGET_GUID="30786afffee60ec9"

# CURRENT_GUID=$(ibv_devices | awk '/bnxt_re_bond0/ {print $2}')

# if [ -z "$CURRENT_GUID" ]; then
#     echo "Error: Could not find bnxt_re_bond0 in ibv_devices output."
#     exit 1
# fi

# if [ "$CURRENT_GUID" == "$TARGET_GUID" ]; then
#     echo "I am Prefill instance.............................."
#     bash proxy.sh
#     bash prefill.sh

# else
echo "I am Decode instance.............................."
bash decode.sh
# fi