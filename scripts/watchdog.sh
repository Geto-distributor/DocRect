#!/bin/bash
# Keep the OCR backend alive. Debounced: only restarts on a SUSTAINED outage,
# so it won't fight a normal in-progress restart (which takes a few seconds).
export PATH=/root/miniconda3/bin:$PATH

healthy() { curl -s -m 5 http://127.0.0.1:6006/health 2>/dev/null | grep -q ok; }

while true; do
  if ! healthy; then
    sleep 10                       # grace: let an in-progress restart come up
    if ! healthy; then
      echo "$(date) health still down after grace -> restart" >> /root/ocr_watchdog.log
      bash /root/start_ocr.sh >> /root/ocr_watchdog.log 2>&1
    fi
  fi
  sleep 30
done
