#!/bin/bash
cd /home/jay/Desktop/new-model-zigzag
# wait for aggregation to finish (or die)
while true; do
  grep -q "ALL AGGREGATION DONE" experiments/atlas_retrain_like_dow/aggregate.log 2>/dev/null && break
  pgrep -f aggregate_8y >/dev/null 2>&1 || { echo "AGG PROCESS GONE"; break; }
  sleep 10
done
echo "=== aggregation finished, starting 8y retrain ==="
ls -la data/m1_xau_orderflow_8y.parquet data/m1_btc_orderflow_8y.parquet 2>/dev/null
python3 experiments/atlas_retrain_like_dow/train.py > experiments/atlas_retrain_like_dow/retrain_8y.log 2>&1
echo "=== CHAIN DONE ==="
