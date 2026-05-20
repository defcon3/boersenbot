#!/usr/bin/env python3
"""SLIPPAGE-FORENSIK Schritt 2 (Spec siehe Memory veitluther-de-roadmap-open).

Sample ~300 Datenpunkte echter Alpaca-SIP-Quotes um Open + Close fuer
OOS-Gap-Korb-Tage. Ziel: ehrlicher round-trip-Slippage-Median pruefen
gegen Pre-Reg-Gate (Median < 10 bps, Q90 < 25 bps).

Korb-Definition (konservativstes Set aus decay-Lauf):
    gap <= -2.5 %   und   >= 10 Aktien im Tageskorb
OOS: ab 2022-01-01.  Sample-Phase A: 30 zufaellige Tage, alle Korbnamen.

Quote-Windows (SIP-Feed, limit=200):
    OPEN  09:25-09:35 ET
    CLOSE 15:55-16:05 ET
Zeitzonen-konvertierung via pytz (EDT/EST automatisch).

Slippage-Definition:
    Pre-Open-Mid  = Mittelwert (bid+ask)/2 der letzten 5 Quotes vor 09:30 ET
    Pre-Close-Mid = Mittelwert (bid+ask)/2 der letzten 5 Quotes vor 16:00 ET
    Entry-Slip = (yfinance-Open  - PreOpenMid)  / PreOpenMid  in bps
    Exit-Slip  = (yfinance-Close - PreCloseMid) / PreCloseMid in bps
    Round-trip = Entry-Slip - Exit-Slip  (positiv = Kosten fuer Long-Korb)
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
          "Lokal nur als Vorlage syntaktisch pruefen, Lauf nur auf VPS.")
    sys.exit(2)

# --------------------------------------------------------------------------
# Parameter
# --------------------------------------------------------------------------
GAP_THR       = -0.025
MIN_KORB      = 10
OOS_START     = pd.Timestamp("2022-01-01")
STK_HISTORY   = "2021-06-01"           # genug Vorlauf, OOS startet 2022-01
N_SAMPLE_DAYS = 30
RNG_SEED      = 20260520               # Datum als Seed -> reproduzierbar

ET = pytz.timezone("America/New_York")
UTC = pytz.UTC

CACHE_FILE = Path(os.path.dirname(os.path.abspath(__file__))) / \
             "slippage_universe.pkl"

HEADERS = {
    "APCA-API-KEY-ID":     ALPACA_CONFIG["api_key"],
    "APCA-API-SECRET-KEY": ALPACA_CONFIG["secret_key"],
}
QUOTES_URL = "https://data.alpaca.markets/v2/stocks/{sym}/quotes"

# --------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------
def load_universe():
    """S&P-500-Liste -> Tickerliste. (Wikipedia, gleich wie decay-Lauf.)"""
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
    """Lade Tages-OHLC fuer alle Ticker ab STK_HISTORY.
    Cached pickle (~80 MB), damit zweite Laeufe nicht wieder herunterladen.
    Rueckgabe: dict ticker -> DataFrame [date, open, close, gap]."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            print(f"  Cache gefunden: {len(cache)} Ticker aus "
                  f"{CACHE_FILE}", flush=True)
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
    print(f"  Cache geschrieben: {CACHE_FILE}", flush=True)
    return cache


def baskets_oos(ohlc):
    """Ermittle OOS-Gap-Tage mit Korb >= MIN_KORB Aktien.
    Rueckgabe: dict pd.Timestamp(date) -> list[ticker]."""
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


def fetch_quotes(symbol, start_iso, end_iso, limit=200):
    """Alpaca SIP-Quotes. Bei 429 throttlen, max 3 Versuche."""
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
        print(f"    {symbol} {start_iso} HTTP {r.status_code} "
              f"{r.text[:120]}", flush=True)
        return None
    return None


def to_utc_iso(date_local_ny, hh, mm):
    """Wandle (date in America/New_York, HH:MM ET) -> UTC-ISO-String.
    DST automatisch korrekt."""
    naive = dt.datetime(date_local_ny.year, date_local_ny.month,
                        date_local_ny.day, hh, mm)
    local = ET.localize(naive)
    utc = local.astimezone(UTC)
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def mid_from_quotes(quotes, until_naive_et, last_n=5):
    """Mittelwert (bid+ask)/2 der letzten `last_n` Quotes VOR until_naive_et.
    `until_naive_et` ist naive datetime im ET (z.B. 09:30:00).
    Auch Halb-Spread-Median in bps wird zurueckgegeben."""
    if not quotes:
        return None, None
    cutoff_utc = ET.localize(until_naive_et).astimezone(UTC)
    rows = []
    for q in quotes:
        ts = q.get("t")
        if not ts:
            continue
        try:
            t_utc = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            continue
        bp, ap = q.get("bp", 0), q.get("ap", 0)
        if not (bp and ap and ap > bp):
            continue
        rows.append((t_utc, bp, ap))
    rows.sort(key=lambda r: r[0])
    before = [r for r in rows if r[0] < cutoff_utc]
    if not before:
        return None, None
    use = before[-last_n:]
    mids = [(bp + ap) / 2 for _, bp, ap in use]
    half_bps = [(ap - bp) / 2 / ((ap + bp) / 2) * 1e4
                for _, bp, ap in use]
    return float(np.mean(mids)), float(np.median(half_bps))


def sample_one(symbol, date, df_ohlc):
    """Hol Open- und Close-Window, berechne Slippage-Felder.
    Rueckgabe: dict mit allen Feldern (oder None bei Datenausfall)."""
    row = df_ohlc[df_ohlc["date"] == date]
    if row.empty:
        return None
    daily_open = float(row["open"].iloc[0])
    daily_close = float(row["close"].iloc[0])
    gap = float(row["gap"].iloc[0])

    py_date = date.to_pydatetime().date() if hasattr(date, "to_pydatetime") \
              else date.date() if hasattr(date, "date") else date
    # OPEN-Fenster 09:25-09:35 ET
    open_start = to_utc_iso(py_date, 9, 25)
    open_end   = to_utc_iso(py_date, 9, 35)
    # CLOSE-Fenster 15:55-16:05 ET
    close_start = to_utc_iso(py_date, 15, 55)
    close_end   = to_utc_iso(py_date, 16, 5)

    q_open  = fetch_quotes(symbol, open_start,  open_end)
    q_close = fetch_quotes(symbol, close_start, close_end)
    if q_open is None or q_close is None:
        return None

    pre_open_mid, half_open_bps = mid_from_quotes(
        q_open, dt.datetime(py_date.year, py_date.month, py_date.day,
                            9, 30, 0))
    pre_close_mid, half_close_bps = mid_from_quotes(
        q_close, dt.datetime(py_date.year, py_date.month, py_date.day,
                             16, 0, 0))

    entry_slip_bps = None
    exit_slip_bps = None
    if pre_open_mid and pre_open_mid > 0:
        entry_slip_bps = (daily_open - pre_open_mid) / pre_open_mid * 1e4
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
        "pre_open_mid":  pre_open_mid,
        "pre_close_mid": pre_close_mid,
        "entry_slip_bps": entry_slip_bps,
        "exit_slip_bps":  exit_slip_bps,
        "round_trip_bps": rt,
        "half_open_bps":  half_open_bps,
        "half_close_bps": half_close_bps,
        "n_quotes_open":  len(q_open),
        "n_quotes_close": len(q_close),
    }


def summarise(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        print("Keine Datenpunkte gesammelt.")
        return
    print(f"\nGesamt-Datenpunkte: {len(df)}  "
          f"(distinct Tage {df['date'].nunique()}, "
          f"distinct Symbole {df['symbol'].nunique()})")

    def stats(col, label):
        s = df[col].dropna()
        if s.empty:
            print(f"  {label:<30} -- keine Werte")
            return
        med = np.median(s); q90 = np.quantile(s, 0.9)
        print(f"  {label:<30} n={len(s):>4}  Ø {s.mean():+7.2f}  "
              f"Median {med:+7.2f}  Q90 {q90:+7.2f}  "
              f"min {s.min():+7.2f}  max {s.max():+7.2f}")

    print("\n-- Halb-Spreads (bps) --")
    stats("half_open_bps",  "OPEN  half-spread")
    stats("half_close_bps", "CLOSE half-spread")

    print("\n-- Slippage (Auktion vs Pre-Auktion-Mid, bps) --")
    stats("entry_slip_bps", "Entry slip (open vs preMid)")
    stats("exit_slip_bps",  "Exit slip  (close vs preMid)")
    stats("round_trip_bps", "Round-trip (entry - exit)")

    # Pre-Reg-Gate (Memory: Median < 10 bps, Q90 < 25 bps)
    rt = df["round_trip_bps"].dropna()
    if not rt.empty:
        med = np.median(rt); q90 = np.quantile(rt, 0.9)
        gate = (med < 10) and (q90 < 25)
        print(f"\n-- Pre-Reg-Gate (Memory) --")
        print(f"  Median round-trip {med:+.2f} bps  (Gate: < 10)  "
              f"{'PASS' if med < 10 else 'FAIL'}")
        print(f"  Q90 round-trip    {q90:+.2f} bps  (Gate: < 25)  "
              f"{'PASS' if q90 < 25 else 'FAIL'}")
        print(f"  GESAMT-GATE: {'PASS' if gate else 'FAIL'}")

    # Konzentration in Stress-Monaten
    df["month"] = df["date"].str[:7]
    print("\n-- Round-trip nach Monat (Top 8 nach |Median|) --")
    g = df.groupby("month")["round_trip_bps"].agg(
        ["count", "mean", "median"]).dropna()
    g = g.iloc[(g["median"].abs()).argsort()[::-1]].head(8)
    print(g.to_string(formatters={"mean": "{:+.2f}".format,
                                  "median": "{:+.2f}".format}))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("="*78)
    print("SLIPPAGE-SAMPLE  (Schritt 2 Slippage-Forensik 2026-05-20)")
    print(f"  Korb: gap<={GAP_THR*100:.1f}%, mink>={MIN_KORB}, OOS>={OOS_START.date()}")
    print(f"  N_SAMPLE_DAYS={N_SAMPLE_DAYS}  SEED={RNG_SEED}")
    print("="*78, flush=True)

    print("\n[1/4] Universum laden...", flush=True)
    tickers = load_universe()
    print(f"      {len(tickers)} Ticker", flush=True)

    print("\n[2/4] OHLC + Gap je Ticker...", flush=True)
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
    print(f"      sample {len(sample_days)} Tage")
    for d in sorted(sample_days):
        print(f"        {d.date()}  Korb={len(by_day[d])}")

    print("\n[4/4] Quotes ziehen (Alpaca SIP, ~3-5 Min)...", flush=True)
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
            # 200 req/min Alpaca-Free => 2 Quotes-Calls je sym -> max 100
            # sym/min. Sicherheitsabstand:
            time.sleep(0.7)
            if done % 25 == 0:
                el = time.time() - start
                print(f"      {done}/{total}  ({el:.0f}s, "
                      f"{len(rows)} ok)", flush=True)

    out = Path(os.path.dirname(os.path.abspath(__file__))) / \
          "slippage_sample.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSample geschrieben: {out}  (n={len(rows)})")

    summarise(rows)


if __name__ == "__main__":
    main()
