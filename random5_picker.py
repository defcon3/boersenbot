#!/usr/bin/env python3
"""
Einmaliger Picker: 5 zufaellige S&P-500-Aktien, die shortable sind.
Schreibt random5_universe.json. Ergebnis ist fix.
"""
import json
import os
import random
import sys

import pandas as pd
import requests

ALPACA_KEY = "PK7C52Q5VZXZ5DDOIDCEEY7CKD"
ALPACA_SECRET = "BqgRvkRyUeanetTS8AEzvnTp5GMaPHcuWqFkCDjTgyLa"
ALPACA_BASE = "https://paper-api.alpaca.markets"

HDR = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random5_universe.json")


def fetch_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compat. bot)"} , timeout=20)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_html(StringIO(r.text))[0]
    syms = [s.replace(".", "-") for s in df["Symbol"].tolist()]  # BRK.B -> BRK-B fuer Alpaca
    print(f"S&P 500 geladen: {len(syms)} Tickers")
    return syms


def check_shortable(symbol):
    r = requests.get(f"{ALPACA_BASE}/v2/assets/{symbol}", headers=HDR, timeout=10)
    if not r.ok:
        return False, f"HTTP {r.status_code}"
    a = r.json()
    if a.get("tradable") and a.get("shortable") and a.get("easy_to_borrow"):
        return True, a
    return False, a


def main():
    if os.path.exists(OUT):
        print(f"FEHLER: {OUT} existiert bereits. Bitte loeschen wenn neu wuerfeln gewollt.")
        sys.exit(1)

    syms = fetch_sp500()
    random.shuffle(syms)

    picked = []
    for s in syms:
        if len(picked) >= 5:
            break
        ok, info = check_shortable(s)
        if ok:
            print(f"  OK: {s} ({info.get('name','?')[:40]})")
            picked.append(s)
        else:
            reason = info if isinstance(info, str) else f"tradable={info.get('tradable')} shortable={info.get('shortable')} etb={info.get('easy_to_borrow')}"
            print(f"  skip {s}: {reason}")

    if len(picked) < 5:
        print("FEHLER: weniger als 5 shortable gefunden")
        sys.exit(1)

    universe = {
        "picked_at": pd.Timestamp.utcnow().isoformat(),
        "universe": "SP500",
        "symbols": picked,
        "position_size_usd": 500,
        "leverage": 2,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(universe, f, indent=2)
    print(f"\nGeschrieben: {OUT}")
    print(f"5er-Universum: {picked}")


if __name__ == "__main__":
    main()
