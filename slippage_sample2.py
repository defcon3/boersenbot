#!/usr/bin/env python3
"""SLIPPAGE-FORENSIK v2 (2026-05-20, Nachbesserung).

v1 (slippage_sample.py) hat gegen Pre-Open-Mid gemessen -- methodisch
ungeeignet, weil Pre-Open-Quotes Stub-Quotes/Auktions-Indikationen sind
(Halb-Spreads Median 46 bps vs 2 bps im regulaeren Handel).

v2: Effective Spread gegen den naechsten handelbaren Mid NACH der
Auktion (bzw. VOR dem Close, was sauber funktioniert).

  Entry-Slip = (yfOpen  - PostOpenMid)  / PostOpenMid  * 1e4
  Exit-Slip  = (yfClose - PreCloseMid)  / PreCloseMid  * 1e4
  Round-trip = Entry-Slip - Exit-Slip   (positiv = Kosten Long-Korb)

PRE-REG-GATE v2:
  (S1) Median round-trip < 10 bps
  (S2) Q90    round-trip < 25 bps
  (S3) Datenqualitaet >=80%: PostOpenMid + PreCloseMid mit Halb-Spread<10
"""
import os
import sys
import json
import time
import random
import pickle
import warnings
import datetime as dt
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import pytz

sys.path.insert(0, "/home/veit/boersenbot")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from config import ALPACA_CONFIG
except ImportError:
    print("FATAL: config.py mit ALPACA_CONFIG nicht gefunden. "
          "Lauf nur auf VPS.")
    sys.exit(2)

# Parameter
GAP_THR       = -0.025
MIN_KORB      = 10
OOS_START     = pd.Timestamp("2022-01-01")
STK_HISTORY   = "2021-06-01"
N_SAMPLE_DAYS = 30
RNG_SEED      = 20260520               # gleicher Seed wie v1 fuer Vergleichbarkeit
QUALITY_HALF_SPREAD_BPS = 10.0         # filter fuer "handelbare" Mids

ET = pytz.timezone("America/New_York")
UTC = pytz.UTC

CACHE_FILE = Path(os.path.dirname(os.path.abspath(__file__))) / \
             "slippage_universe.pkl"   # gleicher Cache wie v1

HEADERS = {
    "APCA-API-KEY-ID":     ALPACA_CONFIG["api_key"],
    "APCA-API-SECRET-KEY": ALPACA_CONFIG["secret_key"],
}
QUOTES_URL = "https://data.alpaca.markets/v2/stocks/{sym}/quotes"


def load_universe():
    import io
    hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    html = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=hdr, timeout=30).text
    sp = pd.read_html(io.StringIO(html))[0]
    sp["Symbol"] = sp["Symbol"].astype(str).str.replace(".", "-",
                                                       regex=False)
    return sp["Symbol"].tolist()


def load_ohlc(tickers):
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            print(f"  Cache gefunden: {len(cache)} Ticker", flush=True)
            return cache
        except Exception as ex:
            print(f"  Cache defekt ({ex}) -- neu laden", flush=True)
    print(f"  Lade {len(tickers)} Ticker ab {STK_HISTORY} ...", flush=True)
    cache = {}
    BATCH = 60
    for i in range(0, len(tickers), BATCH):
        ch = tickers[i:i+BATCH]
        try:
            data = yf.download(ch, start=STK_HISTORY, interval="1d",
                               progress=False, group_by="ticker",
                               threads=True, auto_adjust=True)
        except Exception as ex:
            print(f"  Batch {i}: FEHLER {ex}", flush=True)
            continue
        for t in ch:
            try:
                d = data[t][["Open", "Close"]].dropna()
                if len(d) < 30:
                    continue
                o = d["Open"].astype(float).values.ravel()
                c = d["Close"].astype(float).values.ravel()
                dts = pd.to_datetime(d.index).normalize()
                pc = np.append([np.nan], c[:-1])
                gap = o / pc - 1.0
                cache[t] = pd.DataFrame({
                    "date": dts, "open": o, "close": c, "gap": gap,
                })
            except Exception:
                pass
        print(f"    {min(i+BATCH, len(tickers))}/{len(tickers)} "
              f"({len(cache)} ok)", flush=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    return cache


def baskets_oos(ohlc):
    rows = []
    for t, df in ohlc.items():
        m = df["gap"] <= GAP_THR
        if not m.any():
            continue
        sub = df[m & (df["date"] >= OOS_START)]
        for d in sub["date"].values:
            rows.append((pd.Timestamp(d), t))
    by_day = {}
    for d, t in rows:
        by_day.setdefault(d, []).append(t)
    return {d: ts for d, ts in by_day.items() if len(ts) >= MIN_KORB}


def fetch_quotes(symbol, start_iso, end_iso, limit=500):
    params = {"start": start_iso, "end": end_iso, "limit": limit,
              "feed": "sip"}
    url = QUOTES_URL.format(sym=symbol)
    for attempt in range(3):
        r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        if r.status_code == 200:
            return r.json().get("quotes") or []
        if r.status_code == 429:
            time.sleep(2.0 + attempt * 2.0)
            continue
        return None
    return None


def to_utc_iso(date_local_ny, hh, mm, ss=0):
    naive = dt.datetime(date_local_ny.year, date_local_ny.month,
                        date_local_ny.day, hh, mm, ss)
    local = ET.localize(naive)
    utc = local.astimezone(UTC)
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def quality_mid(quotes, t_start_et, t_end_et, max_half_bps):
    """Liefere Median-Mid aus Quotes im Fenster [t_start_et, t_end_et],
    die einen Halb-Spread <= max_half_bps haben. Sonst None.
    Auch Median-Halb-Spread, Anzahl handelbarer Quotes zurueck."""
    if not quotes:
        return None, None, 0
    t_start_utc = ET.localize(t_start_et).astimezone(UTC)
    t_end_utc   = ET.localize(t_end_et).astimezone(UTC)
    good = []
    for q in quotes:
        ts = q.get("t")
        if not ts:
            continue
        try:
            t_utc = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            continue
        if not (t_start_utc <= t_utc < t_end_utc):
            continue
        bp, ap = q.get("bp", 0), q.get("ap", 0)
        if not (bp and ap and ap > bp):
            continue
        mid = (bp + ap) / 2
        half = (ap - bp) / 2 / mid * 1e4
        if half > max_half_bps:
            continue
        good.append((mid, half))
    if not good:
        return None, None, 0
    mids = [m for m, _ in good]
    halfs = [h for _, h in good]
    return float(np.median(mids)), float(np.median(halfs)), len(good)


def sample_one(symbol, date, df_ohlc):
    row = df_ohlc[df_ohlc["date"] == date]
    if row.empty:
        return None
    daily_open = float(row["open"].iloc[0])
    daily_close = float(row["close"].iloc[0])
    gap = float(row["gap"].iloc[0])

    py_date = date.to_pydatetime().date() if hasattr(date, "to_pydatetime") \
              else date.date() if hasattr(date, "date") else date

    # POST-OPEN-Fenster 09:30:30 - 09:31:30 ET (Auktion vorbei, Spreads eng)
    post_open_start = to_utc_iso(py_date, 9, 30, 0)
    post_open_end   = to_utc_iso(py_date, 9, 32, 0)
    # PRE-CLOSE-Fenster 15:58:30 - 15:59:30 ET (vor Closing-Auktion, eng)
    pre_close_start = to_utc_iso(py_date, 15, 58, 0)
    pre_close_end   = to_utc_iso(py_date, 16,  0, 0)

    q_post_open  = fetch_quotes(symbol, post_open_start,  post_open_end)
    q_pre_close  = fetch_quotes(symbol, pre_close_start,  pre_close_end)
    if q_post_open is None or q_pre_close is None:
        return None

    post_open_mid, half_open_bps, n_open = quality_mid(
        q_post_open,
        dt.datetime(py_date.year, py_date.month, py_date.day, 9, 30, 30),
        dt.datetime(py_date.year, py_date.month, py_date.day, 9, 31, 30),
        QUALITY_HALF_SPREAD_BPS)
    pre_close_mid, half_close_bps, n_close = quality_mid(
        q_pre_close,
        dt.datetime(py_date.year, py_date.month, py_date.day, 15, 58, 30),
        dt.datetime(py_date.year, py_date.month, py_date.day, 15, 59, 30),
        QUALITY_HALF_SPREAD_BPS)

    entry_slip_bps = None
    exit_slip_bps = None
    if post_open_mid and post_open_mid > 0:
        entry_slip_bps = (daily_open - post_open_mid) / post_open_mid * 1e4
    if pre_close_mid and pre_close_mid > 0:
        exit_slip_bps = (daily_close - pre_close_mid) / pre_close_mid * 1e4

    rt = None
    if entry_slip_bps is not None and exit_slip_bps is not None:
        rt = entry_slip_bps - exit_slip_bps

    return {
        "symbol": symbol,
        "date": py_date.isoformat(),
        "gap_pct": gap * 100,
        "open": daily_open,
        "close": daily_close,
        "post_open_mid":  post_open_mid,
        "pre_close_mid":  pre_close_mid,
        "entry_slip_bps": entry_slip_bps,
        "exit_slip_bps":  exit_slip_bps,
        "round_trip_bps": rt,
        "half_open_bps":  half_open_bps,
        "half_close_bps": half_close_bps,
        "n_quality_open":  n_open,
        "n_quality_close": n_close,
        "n_quotes_open":  len(q_post_open),
        "n_quotes_close": len(q_pre_close),
    }


def summarise(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        print("Keine Datenpunkte gesammelt.")
        return

    print(f"\nGesamt-Datenpunkte: {len(df)}  "
          f"(distinct Tage {df['date'].nunique()}, "
          f"distinct Symbole {df['symbol'].nunique()})")

    # Datenqualitaet
    qual_open = df["post_open_mid"].notna().mean()
    qual_close = df["pre_close_mid"].notna().mean()
    qual_both = (df["post_open_mid"].notna() &
                 df["pre_close_mid"].notna()).mean()
    print(f"\nDatenqualitaet (Mid mit Halb-Spread<{QUALITY_HALF_SPREAD_BPS}):")
    print(f"  PostOpenMid verfuegbar : {qual_open:.1%}")
    print(f"  PreCloseMid verfuegbar : {qual_close:.1%}")
    print(f"  beide verfuegbar       : {qual_both:.1%}  "
          f"(Gate S3 >= 80%: {'PASS' if qual_both >= 0.80 else 'FAIL'})")

    def stats(col, label):
        s = df[col].dropna()
        if s.empty:
            print(f"  {label:<30} -- keine Werte")
            return
        med = np.median(s); q90 = np.quantile(s, 0.9)
        q10 = np.quantile(s, 0.1)
        print(f"  {label:<30} n={len(s):>4}  Ø {s.mean():+7.2f}  "
              f"Median {med:+7.2f}  Q10 {q10:+7.2f}  "
              f"Q90 {q90:+7.2f}  min {s.min():+7.2f}  max {s.max():+7.2f}")

    print("\n-- Halb-Spreads gefilterte Mids (bps) --")
    stats("half_open_bps",  "POST-OPEN  half-spread")
    stats("half_close_bps", "PRE-CLOSE  half-spread")

    print("\n-- Slippage vs handelbarer Mid (bps) --")
    stats("entry_slip_bps", "Entry slip (open-auction vs postMid)")
    stats("exit_slip_bps",  "Exit slip  (close-auction vs preMid)")
    stats("round_trip_bps", "Round-trip (entry - exit)")

    rt = df["round_trip_bps"].dropna()
    if not rt.empty:
        med = np.median(rt); q90 = np.quantile(rt, 0.9)
        s1 = med < 10
        s2 = q90 < 25
        s3 = qual_both >= 0.80
        print(f"\n-- Pre-Reg-Gate v2 --")
        print(f"  (S1) Median round-trip {med:+.2f} bps  (< 10)   "
              f"{'PASS' if s1 else 'FAIL'}")
        print(f"  (S2) Q90 round-trip    {q90:+.2f} bps  (< 25)   "
              f"{'PASS' if s2 else 'FAIL'}")
        print(f"  (S3) Datenqualitaet    {qual_both:.1%}      (>= 80%)  "
              f"{'PASS' if s3 else 'FAIL'}")
        print(f"  GESAMT: {'PASS' if (s1 and s2 and s3) else 'FAIL'}")

    df["month"] = df["date"].str[:7]
    print("\n-- Round-trip nach Monat (alle Tage) --")
    g = df.groupby("month")["round_trip_bps"].agg(
        ["count", "mean", "median"]).dropna()
    g = g.sort_index()
    print(g.to_string(formatters={"mean": "{:+.2f}".format,
                                  "median": "{:+.2f}".format}))


def main():
    print("="*78)
    print("SLIPPAGE-SAMPLE v2  (2026-05-20)")
    print(f"  Korb: gap<={GAP_THR*100:.1f}%, mink>={MIN_KORB}, OOS>={OOS_START.date()}")
    print(f"  Methode: Effective Spread vs Post-Auction-Mid")
    print(f"  N_SAMPLE_DAYS={N_SAMPLE_DAYS}  SEED={RNG_SEED}")
    print(f"  Quality-Filter: Halb-Spread < {QUALITY_HALF_SPREAD_BPS} bps")
    print("="*78, flush=True)

    print("\n[1/4] Universum laden...", flush=True)
    tickers = load_universe()
    print(f"      {len(tickers)} Ticker", flush=True)

    print("\n[2/4] OHLC + Gap je Ticker (Cache aus v1)...", flush=True)
    ohlc = load_ohlc(tickers)
    print(f"      {len(ohlc)} Ticker mit Daten", flush=True)

    print("\n[3/4] OOS-Gap-Koerbe (>= MIN_KORB) ermitteln...", flush=True)
    by_day = baskets_oos(ohlc)
    print(f"      {len(by_day)} OOS-Tage mit Korb >= {MIN_KORB}", flush=True)
    if not by_day:
        print("Keine Korb-Tage gefunden — abbruch.")
        return

    rng = random.Random(RNG_SEED)
    days = sorted(by_day.keys())
    sample_days = rng.sample(days, min(N_SAMPLE_DAYS, len(days)))

    print("\n[4/4] Quotes ziehen (Alpaca SIP)...", flush=True)
    rows = []
    start = time.time()
    total = sum(len(by_day[d]) for d in sample_days)
    done = 0
    for d in sorted(sample_days):
        for sym in by_day[d]:
            r = sample_one(sym, d, ohlc[sym])
            done += 1
            if r is not None:
                rows.append(r)
            time.sleep(0.7)
            if done % 25 == 0:
                el = time.time() - start
                print(f"      {done}/{total}  ({el:.0f}s, "
                      f"{len(rows)} ok)", flush=True)

    out = Path(os.path.dirname(os.path.abspath(__file__))) / \
          "slippage_sample2.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSample geschrieben: {out}  (n={len(rows)})")

    summarise(rows)


if __name__ == "__main__":
    main()
