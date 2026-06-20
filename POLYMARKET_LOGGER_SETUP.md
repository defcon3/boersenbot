# Polymarket Arbitrage Logger — Setup & Usage

## Overview

Three scripts work together to collect data for arbitrage analysis:

1. **`polymarket_logger.py`** — Polls Gamma API (Polymarket prices)
2. **`bookmaker_logger.py`** — Polls Betfair + Pinnacle (Bookmaker odds)
3. **`spread_calculator.py`** — Matches events, calculates spreads, finds arbs

**Phase 1 Goal:** Run read-only (zero capital) for 2 weeks, collect real spread distribution.

---

## Setup

### 1. Install Dependencies

```bash
pip install aiohttp
```

### 2. Create Data Directories

```bash
mkdir -p logs polymarket_raw bookmaker_raw arbitrage_analysis
```

### 3. Polymarket Logger (Easy — No Auth Required)

```bash
python polymarket_logger.py
```

**What it does:**
- Polls `https://gamma-api.polymarket.com/markets` every 5 minutes
- Fetches order book (prices, depth) for active markets
- Saves JSON snapshots to `polymarket_raw/snapshot_*.json`
- Logs to `logs/polymarket.log`

**Expected output:**
```
2026-06-20 16:15:00 - INFO - === Poll cycle start ===
2026-06-20 16:15:05 - INFO - Fetched 47 active markets
2026-06-20 16:15:12 - INFO - Saved snapshot: polymarket_raw/snapshot_2026-06-20T16-15-12_345678.json
2026-06-20 16:15:12 - INFO - === Poll cycle done (20 markets) ===
```

**Troubleshooting:**
- If `Fetched 0 active markets`: Gamma API might be down, wait & retry
- Check `logs/polymarket.log` for full errors

---

### 4. Bookmaker Logger (Requires API Keys)

#### 4a. Betfair Setup

1. **Register** at https://www.betfair.com/
2. **Get API Key:** https://developer.betfair.com/
3. **Login & Get Session Token:**
   ```bash
   curl -X POST https://api.betfair.com/exchange/account/json-rpc/v1 \
     -H "X-Application: YOUR_APP_KEY" \
     -d '{"jsonrpc":"2.0","method":"AccountAPING/v1.0/login","params":{"username":"YOUR_USERNAME","password":"YOUR_PASSWORD","applicationKey":"YOUR_APP_KEY"},"id":1}'
   ```
   Extract `sessionToken` from response.

4. **Set Environment Variable:**
   ```bash
   export BETFAIR_SESSION_TOKEN="your_token_here"
   export BETFAIR_APP_KEY="your_app_key_here"
   ```

#### 4b. Pinnacle Setup

1. **Register** at https://www.pinnacle.com/
2. **Get API Key** from account settings
3. **Set Environment Variable:**
   ```bash
   export PINNACLE_API_KEY="your_api_key_here"
   ```

#### 4c. Run Logger

```bash
python bookmaker_logger.py
```

**What it does:**
- Polls Betfair API for live tennis/sports odds
- Polls Pinnacle API for live sports odds
- Saves JSON snapshots to `bookmaker_raw/snapshot_*.json`
- Logs to `logs/bookmaker.log`

**Current Limitations:**
- Only scrapes Tennis (Betfair) + Tennis (Pinnacle)
- Can expand to other sports (Football, Boxing, etc.)
- Betfair requires login token (expires after 24h, needs refresh)

---

### 5. Spread Calculator

**Run manually after each poll cycle (or on schedule):**

```bash
python spread_calculator.py
```

**What it does:**
1. Loads latest Polymarket snapshot
2. Loads latest Bookmaker snapshot
3. **Event Matching:** Fuzzy-matches Poly titles to bookie market names
4. **Spread Calc:** `|(Bookie Odds) - (Poly Odds)| / avg_odds`
5. **Friction:** Subtracts Poly fee (0.2%) + Wettsteuer (5%) + Commissions
6. **Net Arb:** Remaining spread after friction
7. **Output:** CSV `arbitrage_analysis/arbitrage_spreads.csv`

**Example Output (CSV):**
```
timestamp,event_name,poly_yes_price,bookie_yes_price,spread_yes_pct,friction_yes_pct,net_arb_yes_pct,arbitrageable
2026-06-20T16:15:00,Djokovic vs Sinner,2.10,2.05,2.34,5.20,-2.86,NO
2026-06-20T16:15:00,Alcaraz vs Medvedev,1.95,2.08,6.34,5.20,1.14,YES
```

---

## Running the Full Pipeline

### Option 1: Manual (for testing)

```bash
# Terminal 1: Polymarket logger
python polymarket_logger.py

# Terminal 2: Bookmaker logger
python bookmaker_logger.py

# Terminal 3: Run calculator every 5 minutes (manual)
while true; do
  python spread_calculator.py
  sleep 300
done
```

### Option 2: Scheduled (Recommended for Phase 1)

**Windows (Task Scheduler):**
```batch
# Schedule Python task every 5 minutes
schtasks /create /tn "polymarket_logger" /tr "python C:\projekte\boersenbot\polymarket_logger.py" /sc minute /mo 5 /f
schtasks /create /tn "bookmaker_logger" /tr "python C:\projekte\boersenbot\bookmaker_logger.py" /sc minute /mo 5 /f
schtasks /create /tn "spread_calculator" /tr "python C:\projekte\boersenbot\spread_calculator.py" /sc minute /mo 5 /f
```

**Linux/macOS (cron):**
```bash
# Run calculator every 5 minutes
*/5 * * * * cd /home/veit/boersenbot && python spread_calculator.py >> logs/scheduler.log 2>&1
```

### Option 3: Docker (for VPS deployment)

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY polymarket_logger.py bookmaker_logger.py spread_calculator.py ./
RUN pip install aiohttp
CMD ["python", "polymarket_logger.py"]
```

```bash
docker build -t boersenbot-polymarket .
docker run -v $(pwd)/logs:/app/logs -v $(pwd)/polymarket_raw:/app/polymarket_raw boersenbot-polymarket
```

---

## Monitoring & Troubleshooting

### Check Log Files

```bash
# Real-time log tailing
tail -f logs/polymarket.log
tail -f logs/bookmaker.log
tail -f logs/spread_calculator.log
```

### Data Quality Checks

```bash
# Count snapshots collected (should grow over time)
ls -la polymarket_raw/ | wc -l
ls -la bookmaker_raw/ | wc -l

# Check latest spread analysis
head -20 arbitrage_analysis/arbitrage_spreads.csv
```

### Common Issues

| Issue | Cause | Fix |
|---|---|---|
| "No snapshots found" | Loggers haven't run yet | Wait 5 min, check logs/ |
| Betfair 401 Unauthorized | Session token expired | Re-login, get new token |
| Pinnacle API 403 | API key invalid | Check env var, re-register |
| 0 matches between Poly/Bookie | Event names don't align | Adjust fuzzy matching threshold |
| "arbitrageable: NO" for all rows | Friction too high | Check fee/tax calculations |

---

## Phase 1 Success Criteria

**After 2 weeks of continuous logging:**

1. ✅ Collected 500+ Poly market snapshots
2. ✅ Collected 300+ Bookie snapshots
3. ✅ Matched 100+ Poly ↔ Bookie event pairs
4. ✅ Distribution of spreads calculated
5. ✅ Arbs > 2% netto visible in 5% of events (minimum threshold to proceed)

**Output File:**
- `arbitrage_analysis/arbitrage_spreads.csv` — Full data for G1 gate
- `logs/polymarket.log`, `logs/bookmaker.log` — Full audit trail

---

## Next Steps

**If spreads look promising (> 2% netto in 10%+ of events):**
→ Move to Phase 2: Risk Validation (G1–G5 Gates)

**If spreads are small (< 1% netto):**
→ Pivot to Phase 2B: Polymarket as Signal Source for SPY Trading

---

## Development Notes

- **Expand to more sports:** Modify `list_events()` in `bookmaker_logger.py` (add Football, Boxing, etc.)
- **Improve event matching:** Replace Levenshtein with NLP model (spaCy) for better accuracy
- **Add real-time alerts:** Email/Slack when arb > 3% appears
- **Database:** Consider switching from JSON files to SQLite for faster queries
- **Cloud deployment:** Push to VPS once logging is stable

