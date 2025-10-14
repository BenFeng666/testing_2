#!/bin/bash
# Monitor training progress

echo "=== Job Queue Status ==="
squeue -u li003385

echo ""
echo "=== Recent Training Logs ==="
if [ -f logs/train_*.out ]; then
    latest_log=$(ls -t logs/train_*.out 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo "Latest log: $latest_log"
        echo "--- Last 30 lines ---"
        tail -30 "$latest_log"
    fi
else
    echo "No training logs found yet. Job may still be queued."
fi

echo ""
echo "=== Training Status Summary ==="
if [ -d "qwen_lora_finetuned" ]; then
    echo "✓ Model checkpoint directory exists"
    ls -lh qwen_lora_finetuned/ 2>/dev/null | head -10
else
    echo "⏳ Waiting for training to complete..."
fi

echo ""
echo "=== Commands ==="
echo "Check queue:        squeue -u li003385"
echo "View training log:  tail -f logs/train_*.out"
echo "Cancel job:         scancel <job_id>"
echo "Submit eval job:    sbatch evaluate_job.sh"

