#!/usr/bin/env python3
"""
GAP-KORB-DECAY-FORENSIK (Reanimations-Pruefung 2026-05-20).

Ausgangslage: gap_bounce_study (2026-05-18) -- Aktien-Korb GAP<-2% ist
in-sample robust, OOS schwaecht aber, Status gelb. Reanimations-Backlog
verlangt: rollierende 12M-t + Schwellen-Robustheit + Sektor-Filter +
realistischere Auktions-Kosten.

PRE-REGISTRIERTES REANIMATIONS-GATE (alle drei muessen erfuellt sein):
 (G1) Rollierende 12M-Tagesportfolio-t bleibt in >=60% der Fenster >= 1.
 (G2) Mindestens ein Schwellen/Mindestkorb-Set hat OOS netto@8bps Ø>0
      und Tages-t > 1.5 bei minKorb>=8.
 (G3) Edge ist NICHT auf einen einzelnen GICS-Sektor konzentriert
      (Top-Sektor traegt < 50% der OOS-Tages-PnL-Summe).

Sonst: Faden zu, Arc schliessen oder bezahlte Daten.
"""
import warnings; warnings.filterwarnings("ignore")
import io, sys, time, json
import numpy as np, pandas as pd, yfinance as yf, requests

STK_START   = "2015-01-01"
SPLIT       = pd.Timestamp("2022-01-01")
COVID_A     = pd.Timestamp("2020-02-15")
COVID_B     = pd.Timestamp("2020-04-30")
# Schwellen-Grid fuer Test 2
GAP_THRS    = [-0.010, -0.015, -0.020, -0.025, -0.030]
MIN_KORB    = [5, 8, 10, 15]
COSTS_FLAT  = [3.0, 5.0, 8.0, 12.0]   # bps round-trip
ROLL_DAYS   = 252                     # 12 Monate
ROLL_STEP   = 21                      # ~Monatsende neu


def tstat(r):
    r = np.asarray(r, float)
    n = len(r)
    if n < 2: return float("nan")
    s = r.std(ddof=1)
    return r.mean()/(s/np.sqrt(n)) if s > 0 else float("nan")


def maxdd(r):
    if len(r) == 0: return 0.0
    eq = np.cumprod(1.0 + np.asarray(r, float))
    peak = np.maximum.accumulate(eq)
    return float((eq/peak - 1.0).min())


# -- 1) Universum + Sektoren von Wikipedia ----------------------------
print("Lade S&P-500-Liste + Sektoren...", flush=True)
hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
html = requests.get(
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    headers=hdr, timeout=30).text
sp = pd.read_html(io.StringIO(html))[0]
sp["Symbol"] = sp["Symbol"].astype(str).str.replace(".", "-", regex=False)
sector_of = dict(zip(sp["Symbol"], sp["GICS Sector"]))
tickers = sp["Symbol"].tolist()
print(f"  {len(tickers)} Ticker, {len(set(sector_of.values()))} Sektoren",
      flush=True)


# -- 2) Daten laden in Batches: Open/Close/High/Low fuer Spread-Proxy --
print(f"Lade Tages-OHLC ab {STK_START} in Batches...", flush=True)
# Pro Aktie: Liste (date, gap, intr, range_pct, prev_close)
per_ticker = {}
BATCH = 60
for i in range(0, len(tickers), BATCH):
    ch = tickers[i:i+BATCH]
    try:
        data = yf.download(ch, start=STK_START, interval="1d",
                           progress=False, group_by="ticker", threads=True,
                           auto_adjust=True)
    except Exception as ex:
        print(f"  Batch {i}: FEHLER {ex}", flush=True); continue
    for t in ch:
        try:
            d = data[t][["Open","High","Low","Close"]].dropna()
            if len(d) < 60: continue
            o = d["Open"].astype(float).values.ravel()
            h = d["High"].astype(float).values.ravel()
            l = d["Low"].astype(float).values.ravel()
            c = d["Close"].astype(float).values.ravel()
            dts = pd.to_datetime(d.index).normalize()
            pc = np.append([np.nan], c[:-1])
            gap = o/pc - 1.0
            intr = c/o - 1.0
            rng_pct = (h - l)/c   # Tages-Range relativ Close (Spread-Proxy)
            per_ticker[t] = (dts, gap, intr, rng_pct)
        except Exception:
            pass
    print(f"  {min(i+BATCH, len(tickers))}/{len(tickers)} "
          f"({len(per_ticker)} ok)", flush=True)


# -- 3) Pro (Schwelle, MinKorb) das Tages-Portfolio bauen --------------
# Effizient: einmal alle (datum, ticker, gap, intr, rng) flach sammeln,
# dann pro Tag filtern.
print("\nFlachstruktur bauen...", flush=True)
all_rows = []
for t, (dts, gap, intr, rng_pct) in per_ticker.items():
    sec = sector_of.get(t, "Unknown")
    finite = np.isfinite(gap) & np.isfinite(intr) & np.isfinite(rng_pct)
    for j in np.flatnonzero(finite):
        all_rows.append((dts[j], t, sec, gap[j], intr[j], rng_pct[j]))
df = pd.DataFrame(all_rows, columns=["date","tkr","sec","gap","intr","rng"])
print(f"  {len(df)} Aktie-Tage gesamt", flush=True)


# -- 4) Hilfsfunktion: Tages-Portfolio fuer (thr, mink) ----------------
def daily_basket(thr, mink, sec_filter=None):
    sub = df[df["gap"] <= thr]
    if sec_filter is not None:
        sub = sub[sub["sec"] == sec_filter]
    grp = sub.groupby("date")
    rows = []
    for d, g in grp:
        n = len(g)
        if n < mink: continue
        rows.append((d, g["intr"].mean(), g["rng"].mean(), n))
    rows.sort()
    if not rows: return None
    out = pd.DataFrame(rows, columns=["date","ret","rng","n"])
    return out


# -- 5) HEADLINE (Baseline-Bestaetigung: thr=-2%, mink=5) --------------
print("\n" + "="*86)
print("0) BASELINE  thr=-2%, minKorb=5  -- soll die 2026-05-18-Zahlen treffen")
print("="*86)
b = daily_basket(-0.02, 5)
if b is not None:
    r = b["ret"].values
    is_mask = b["date"].values < SPLIT.to_datetime64()
    print(f"  n={len(r)} Tage, MedKorb={int(b['n'].median())}, "
          f"Ø {r.mean()*1e4:+.2f} bps, t={tstat(r):+.2f}, MaxDD={maxdd(r):.1%}")
    for c in [0,3,5,8,12]:
        rn = r - c/1e4
        print(f"    netto@{c:>2}: Ø {rn.mean()*1e4:+6.2f} t={tstat(rn):+5.2f}")
    for seg, m in (("IS  <=2021", is_mask), ("OOS >=2022", ~is_mask)):
        rs = r[m]
        if len(rs):
            print(f"  [{seg}] n={len(rs):>4} Ø {rs.mean()*1e4:+6.2f} bps "
                  f"t={tstat(rs):+5.2f}  netto@5 t={tstat(rs-5/1e4):+5.2f}")


# -- 6) TEST 1: rollierende 12M-t ---------------------------------------
print("\n" + "="*86)
print("TEST 1) ROLLIERENDE 12M-PORTFOLIO-t  (thr=-2%, minKorb=8)")
print("="*86)
b = daily_basket(-0.02, 8)
if b is None or len(b) < ROLL_DAYS:
    print("  zu wenige Tage fuer rollierendes 12M-Fenster")
    g1_pass = False
else:
    dates = pd.to_datetime(b["date"].values)
    r = b["ret"].values
    rows = []
    i = 0
    while i + ROLL_DAYS <= len(r):
        w = r[i:i+ROLL_DAYS]
        rows.append((dates[i], dates[i+ROLL_DAYS-1], len(w),
                     w.mean()*1e4, tstat(w),
                     (w - 5/1e4).mean()*1e4, tstat(w - 5/1e4)))
        i += ROLL_STEP
    rt = pd.DataFrame(rows, columns=["from","to","n","brutto_bps","t",
                                     "netto5_bps","t_netto5"])
    print(rt.to_string(index=False,
        formatters={"brutto_bps":"{:+.2f}".format,
                    "netto5_bps":"{:+.2f}".format,
                    "t":"{:+.2f}".format,
                    "t_netto5":"{:+.2f}".format,
                    "from":lambda d: pd.Timestamp(d).strftime("%Y-%m"),
                    "to"  :lambda d: pd.Timestamp(d).strftime("%Y-%m")}))
    share_pos = (rt["t"] >= 1.0).mean()
    share_pos_oos = (rt[rt["from"] >= SPLIT]["t"] >= 1.0).mean() \
        if (rt["from"] >= SPLIT).any() else float("nan")
    print(f"\n  Anteil 12M-Fenster mit t>=1.0  GESAMT: {share_pos:.1%}  "
          f"OOS-Start>=2022: {share_pos_oos:.1%}")
    g1_pass = share_pos >= 0.60


# -- 7) TEST 2: Schwellen/Mindestkorb-Grid (Headline IS+OOS+netto) -----
print("\n" + "="*86)
print("TEST 2) SCHWELLEN x MIN-KORB  (Brutto, OOS Tages-t, OOS netto@8)")
print("="*86)
g2_pass = False
g2_winners = []
hdr_row = "thr    \\ mink  " + " ".join(f"  {m:>3}" for m in MIN_KORB)
def fmt_cell(val):
    if val is None: return "   --  "
    return f"{val:+5.2f}"
# Print: OOS Tages-t
print("\nOOS  Tages-t  (>=2022, brutto):")
print(hdr_row)
for thr in GAP_THRS:
    row = [f"{thr*100:+5.1f}%       "]
    for mink in MIN_KORB:
        b = daily_basket(thr, mink)
        if b is None:
            row.append(fmt_cell(None)); continue
        m = b["date"].values >= SPLIT.to_datetime64()
        rs = b["ret"].values[m]
        row.append(fmt_cell(tstat(rs)) if len(rs) >= 10 else "   --  ")
    print(" ".join(row))

print("\nOOS  Ø netto@8 bps  (>=2022):")
print(hdr_row)
for thr in GAP_THRS:
    row = [f"{thr*100:+5.1f}%       "]
    for mink in MIN_KORB:
        b = daily_basket(thr, mink)
        if b is None: row.append(fmt_cell(None)); continue
        m = b["date"].values >= SPLIT.to_datetime64()
        rs = b["ret"].values[m]
        if len(rs) < 10: row.append("   --  "); continue
        rn = rs - 8/1e4
        row.append(f"{rn.mean()*1e4:+5.2f}")
        if mink >= 8 and rn.mean() > 0 and tstat(rs) > 1.5:
            g2_winners.append((thr, mink, rn.mean()*1e4, tstat(rs), len(rs)))
    print(" ".join(row))

print("\n  Treffer OOS minKorb>=8 mit Ø netto@8>0 und brutto-Tages-t>1.5:")
if g2_winners:
    for thr, mk, nbps, tt, n in g2_winners:
        print(f"    thr={thr*100:+.1f}%  mink={mk}  netto@8 Ø {nbps:+.2f} bps  "
              f"t_brutto={tt:+.2f}  n={n} Tage")
    g2_pass = True
else:
    print("    -- keine --")


# -- 8) TEST 3: GICS-Sektor-Konzentration ------------------------------
print("\n" + "="*86)
print("TEST 3) SEKTOR-KONZENTRATION  (thr=-2%, minKorb>=3 je Sektor)")
print("="*86)
b_all = daily_basket(-0.02, 5)
# Pro Sektor: sammle Tage mit >=3 Namen Gap<-2% innerhalb Sektor
g3_pass = False
sec_pnl_oos = {}
for sec in sorted(set(sector_of.values())):
    bs = daily_basket(-0.02, 3, sec_filter=sec)
    if bs is None or len(bs) < 20: continue
    is_mask = bs["date"].values < SPLIT.to_datetime64()
    oos = bs["ret"].values[~is_mask]
    isb = bs["ret"].values[is_mask]
    sec_pnl_oos[sec] = oos.sum() if len(oos) else 0.0
    if len(oos) >= 10:
        print(f"  {sec:<25}  n_is={len(isb):>3} "
              f"Ø {isb.mean()*1e4:+6.2f} t={tstat(isb):+4.2f} | "
              f"n_oos={len(oos):>3} Ø {oos.mean()*1e4:+6.2f} "
              f"t={tstat(oos):+4.2f}  netto@8 Ø {(oos-8/1e4).mean()*1e4:+6.2f}")
total_oos = sum(abs(v) for v in sec_pnl_oos.values())
if total_oos > 0:
    top_sec, top_val = max(sec_pnl_oos.items(), key=lambda kv: kv[1])
    print(f"\n  Top-Sektor nach OOS-PnL-Summe: {top_sec}  "
          f"({top_val*1e4:+.1f} bp-Tage = {top_val/total_oos:.1%} von |Sum-|OOS|)")
    g3_pass = (top_val/total_oos) < 0.50
else:
    print("\n  keine OOS-Sektor-PnL")
    g3_pass = False


# -- 9) TEST 4: Spread-Proxy-Kosten (statt Flat-3 bps) -----------------
print("\n" + "="*86)
print("TEST 4) REALISTISCHERE KOSTEN  (thr=-2%, minKorb=8)")
print("="*86)
b = daily_basket(-0.02, 8)
if b is not None:
    r = b["ret"].values
    rng = b["rng"].values   # Ø Tages-Range-pct im Korb
    is_mask = b["date"].values < SPLIT.to_datetime64()
    # Annahme: round-trip-Slippage ~ k * Ø-Tages-Range (k konservativ).
    for k in [0.05, 0.10, 0.15, 0.20]:
        slip = k * rng           # entry+exit zusammen, in Returnseinheit
        rn = r - slip
        print(f"  k={k:.2f} (slip Ø {slip.mean()*1e4:5.2f} bps):  "
              f"GES Ø {rn.mean()*1e4:+6.2f} t={tstat(rn):+5.2f}  | "
              f"OOS Ø {rn[~is_mask].mean()*1e4:+6.2f} "
              f"t={tstat(rn[~is_mask]):+5.2f}")
    for c in COSTS_FLAT:
        rn = r - c/1e4
        print(f"  flat {c:>2}bps:                       GES Ø {rn.mean()*1e4:+6.2f} "
              f"t={tstat(rn):+5.2f}  | OOS Ø {rn[~is_mask].mean()*1e4:+6.2f} "
              f"t={tstat(rn[~is_mask]):+5.2f}")


# -- 10) GATE-AUSWERTUNG -----------------------------------------------
print("\n" + "="*86)
print("PRE-REGISTRIERTE GATE-AUSWERTUNG (Reanimation Gap-Korb)")
print("="*86)
print(f"  (G1) >=60% rollierende 12M-Fenster mit t>=1     : "
      f"{'PASS' if g1_pass else 'FAIL'}")
print(f"  (G2) >=1 Schwellen-Set OOS netto@8>0 + t>1.5    : "
      f"{'PASS' if g2_pass else 'FAIL'}")
print(f"  (G3) Top-OOS-Sektor < 50% der |OOS-PnL-Summe|    : "
      f"{'PASS' if g3_pass else 'FAIL'}")
print()
if g1_pass and g2_pass and g3_pass:
    print("  ==> GATE ERFUELLT. Gruenes Licht fuer Alpaca-Paper-Trading.")
else:
    print("  ==> GATE NICHT ERFUELLT. Faden ehrlich schliessen oder neue")
    print("      Hypothese / bezahlte Daten. KEIN Paper-Trading.")
