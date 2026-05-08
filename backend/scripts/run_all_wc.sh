#!/bin/bash
# Run all 4 strategy walk-forward backtests
# Usage: cd backend && bash scripts/run_all_wc.sh

STRATEGIES=("Swing_Trading" "Momentum_Trading" "Hybrid_Trading" "Nitin_Triple_Confirm_Retracement")
PERIOD="5y"
MC_ITER=8
MAX_STOCKS=100

cd "$(dirname "$0")/.."

mkdir -p logs

for strategy in "${STRATEGIES[@]}"; do
    logfile="logs/wc_${strategy}_${PERIOD}.log"
    echo "Launching: $strategy -> $logfile"
    nohup python scripts/run_portfolio_backtest.py \
        --strategy "$strategy" \
        --period "$PERIOD" \
        --max-stocks "$MAX_STOCKS" \
        --walk-forward \
        --mc-iterations "$MC_ITER" \
        > "$logfile" 2>&1 &
    echo "  PID: $!"
    sleep 2
done

echo ""
echo "All 4 backtests launched. Check logs/ for output."
echo "To monitor: tail -f logs/wc_<Strategy>_5y.log"
