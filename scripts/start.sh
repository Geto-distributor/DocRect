#!/bin/bash
# Reliable (re)start of the OmniX OCR backend on :6006.
# Kills any previous instance first (fuser is NOT installed on this box, so we
# kill by pidfile + by scanning ps — never by `pkill -f uvicorn`, which would
# self-match the shell running this script).
export PATH=/root/miniconda3/bin:$PATH
# let onnxruntime-gpu (ISNet) find paddle's bundled CUDA/cuDNN libs
NV=/root/miniconda3/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(echo $NV/*/lib | tr ' ' ':'):$LD_LIBRARY_PATH
cd /root || exit 1
PIDFILE=/root/ocrsvc.pid

if [ -f "$PIDFILE" ]; then
  kill -9 "$(cat "$PIDFILE")" 2>/dev/null
fi
for p in $(ps -eo pid,cmd | grep "[u]vicorn app:app" | awk '{print $1}'); do
  kill -9 "$p" 2>/dev/null
done
sleep 2

nohup python -m uvicorn app:app --host 0.0.0.0 --port 6006 > /root/ocrsvc.log 2>&1 &
echo $! > "$PIDFILE"
sleep 6
if curl -s -m 5 http://127.0.0.1:6006/health | grep -q ok; then
  echo "OCR backend started OK, pid $(cat "$PIDFILE")"
else
  echo "WARN: health check failed; tail log:"; tail -n 20 /root/ocrsvc.log
fi
