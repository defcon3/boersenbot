"""Holt SPY 30-Min-Bars (Alpaca SIP, 2016->heute) und speichert als pkl.
LAEUFT AUF DEM VPS (config.py mit Keys liegt nur dort). Danach pkl scp-en."""
import pickle
import time
import requests
from config import ALPACA_CONFIG

H = {"APCA-API-KEY-ID": ALPACA_CONFIG["api_key"],
     "APCA-API-SECRET-KEY": ALPACA_CONFIG["secret_key"]}
BASE = "https://data.alpaca.markets/v2/stocks/SPY/bars"
OUT = "spy_30min_sip_2016_2026.pkl"


def fetch_all(start, end, tf="30Min", feed="sip"):
    bars, token, page = [], None, 0
    while True:
        params = {"timeframe": tf, "start": start, "end": end, "feed": feed,
                  "limit": 10000, "adjustment": "raw"}
        if token:
            params["page_token"] = token
        r = requests.get(BASE, headers=H, params=params, timeout=60)
        j = r.json()
        if r.status_code != 200:
            raise SystemExit(f"HTTP {r.status_code}: {j}")
        got = j.get("bars") or []
        bars += got
        page += 1
        print(f"  page {page}: +{len(got)} (total {len(bars)})")
        token = j.get("next_page_token")
        if not token:
            break
        time.sleep(0.2)
    return bars


if __name__ == "__main__":
    print("Fetch SPY 30Min SIP 2016-01-01 .. 2026-06-23 ...")
    bars = fetch_all("2016-01-01T00:00:00Z", "2026-06-23T00:00:00Z")
    with open(OUT, "wb") as f:
        pickle.dump(bars, f)
    print(f"OK: {len(bars)} Bars -> {OUT}")
    print(f"   first={bars[0]['t']}  last={bars[-1]['t']}")
