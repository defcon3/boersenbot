#!/bin/bash
# HYG SIGNAL EXECUTOR (NAS)
# Liest hyg_today_signal.json und protokolliert ins hyg.log

SIGNAL_FILE="/var/services/homes/benutzername/boersenbot/hyg_today_signal.json"
LOG_FILE="/var/services/homes/benutzername/boersenbot/hyg.log"

echo "[$(date -u +'%Y-%m-%d %H:%M:%S')] HYG Signal Executor started" >> $LOG_FILE

if [ -f "$SIGNAL_FILE" ]; then
    ACTION=$(grep -o '"action": "[^"]*"' $SIGNAL_FILE | cut -d'"' -f4)
    EXPOSURE=$(grep -o '"exposure_pct": [0-9.]*' $SIGNAL_FILE | cut -d' ' -f2)
    STLFSI=$(grep -o '"stlfsi4_value": [-0-9.]*' $SIGNAL_FILE | cut -d' ' -f2)
    STRESS=$(grep -o '"in_stress_window": [a-z]*' $SIGNAL_FILE | cut -d' ' -f2)
    DAYS=$(grep -o '"days_since_stress": [0-9]*' $SIGNAL_FILE | cut -d' ' -f2)
    HYG_PRICE=$(grep -o '"hyg_price": [0-9.]*' $SIGNAL_FILE | cut -d' ' -f2)

    echo "[$(date -u +'%Y-%m-%d %H:%M:%S')] STLFSI4=$STLFSI Stress=$STRESS Action=$ACTION Exposure=${EXPOSURE}% HYG=\$$HYG_PRICE DaysSinceStress=$DAYS" >> $LOG_FILE
    echo "[$(date -u +'%Y-%m-%d %H:%M:%S')] [OK] HYG Execution complete" >> $LOG_FILE
else
    echo "[$(date -u +'%Y-%m-%d %H:%M:%S')] [ERROR] HYG Signal-Datei nicht gefunden" >> $LOG_FILE
fi
