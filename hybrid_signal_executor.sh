#!/bin/bash
# HYBRID SIGNAL EXECUTOR (NAS)
# Reads today_signal.json (calculated locally) and logs execution

SIGNAL_FILE="/var/services/homes/benutzername/boersenbot/today_signal.json"
LOG_FILE="/var/services/homes/benutzername/boersenbot/hybrid.log"

echo "[$(date -u +'%Y-%m-%d %H:%M:%S')] Hybrid Signal Executor started" >> $LOG_FILE

if [ -f "$SIGNAL_FILE" ]; then
    # Extract values from JSON
    ACTION=$(grep -o '"action": "[^"]*"' $SIGNAL_FILE | cut -d'"' -f4)
    POS_SIZE=$(grep -o '"position_size": [0-9.]*' $SIGNAL_FILE | cut -d' ' -f2)
    VIX=$(grep -o '"current_vix": [0-9.]*' $SIGNAL_FILE | cut -d' ' -f2)
    SIGNAL=$(grep -o '"uptrend_signal": [0-9]' $SIGNAL_FILE | cut -d' ' -f2)

    echo "[$(date -u +'%Y-%m-%d %H:%M:%S')] Signal=$ACTION PosSize=$POS_SIZE VIX=$VIX Uptrend=$SIGNAL" >> $LOG_FILE
    echo "[$(date -u +'%Y-%m-%d %H:%M:%S')] [OK] Execution complete" >> $LOG_FILE
else
    echo "[$(date -u +'%Y-%m-%d %H:%M:%S')] [ERROR] Signal file not found" >> $LOG_FILE
fi
