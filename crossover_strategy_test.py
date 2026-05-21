#!/usr/bin/env python3
"""
MA-CROSSOVER STRATEGIE-TEST (Plan veitluther-de-roadmap-open, NAECHSTER
SCHRITT 1, fixiert 2026-05-21).

Ehrlicher Long-Backtest auf den im Grid identifizierten MA-Crossover-Edge.
Klon von cc_meanrev_excess.py, aber Signal ersetzt:
  Streak >= N  --->  Golden Cross MA_fast / MA_slow

Festgenagelte Parameter (kein Grid mehr in diesem Skript):
  fast=20, slow=50, horizon=10 Handelstage  (Sweet-Spot aus crossover_grid)
  -> +1.82 pp Hit, +0.21 % Mean-Diff in den Roh-Aggregaten 2014-2026

Methodik (identisch zu cc_meanrev_excess.py):
  (A) EXCESS vs SPY per Einstiegstag (ehrliches n=#Tage),
      OOS-Split 2022-01-01, COVID-Ausschluss 2020-02-15..04-30,
      Kosten {3,5,10} bps round-trip, Monatsblock-t als Sanity.
  (B) Reales taeglich rebalanciertes OVERLAP-Portfolio (Roh-Long),
      echter MaxDD, exakte Kosten -(C/h) je Aktiv-Tag.

Datenquelle: crossover_universe.pkl (VPS-Cache aus crossover_grid.py).
Survivorship-Hinweis EXPLIZIT: Universum = aktuelle S&P-500-Liste, also
Bias zugunsten der Long-Strategie bekannt -> Edge wird tendenziell
ueberschaetzt.

=================================================================
PRE-REGISTRIERTES ERFOLGSKRITERIUM (fixiert VOR dem Lauf, 2026-05-21)
=================================================================
Gates, alle 5 muessen halten:
  (G1) IS  Excess vs SPY Mittel > 0, t-Stat (pro Einstiegstag) > +2.0
  (G2) OOS Excess vs SPY Mittel > 0, t-Stat                    > +1.5
  (G3) Netto @ 5 bps round-trip: OOS-Mittel noch > 0, t > +1.0
  (G4) COVID-Ausschluss-Variante: OOS-Mittel bleibt > 0
  (G5) Median offener Positionen pro Tag im (B)-Portfolio >= 5
Fail in EINEM Gate => Edge nicht greenlight, Faden ehrlich schliessen.
"""
import warnings; warnings.filterwarnings("ignore")
import io, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf

START      = "2014-01-01"
SPLIT      = pd.Timestamp("2022-01-01")
COVID_A    = pd.Timestamp("2020-02-15")
COVID_B    = pd.Timestamp("2020-04-30")
FAST       = 20
SLOW       = 50
H          = 10
COSTS      = [0.0, 3.0, 5.0, 10.0]   # bps round-trip
CACHE_FILE = Path("/home/veit/boersenbot/crossover_universe.pkl")


def tstat(r):
    r = np.asarray(r, float); n = len(r)
    if n < 2: return float("nan")
    s = r.std(ddof=1)
    return r.mean() / (s / np.sqrt(n)) if s > 0 else float("nan")


def maxdd(r):
    if len(r) == 0: return 0.0
    eq = np.cumprod(1.0 + np.asarray(r, float))
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def monthly_t(dates, r):
    s = pd.Series(r, index=pd.to_datetime(dates)).resample("ME").sum()
    s = s[s != 0]
    return tstat(s.values), len(s)


def gate(label, cond, detail):
    flag = "PASS" if cond else "FAIL"
    print(f"  [{flag}] {label}  -- {detail}")
    return bool(cond)


print("="*78)
print("PRE-REG (fixiert vor Lauf): f=20, s=50, h=10  | OOS=2022-01-01")
print("Gates G1..G5 wie im Skript-Header. Survivorship: aktuelle SP500-Liste.")
print("="*78, flush=True)

# 1) Universum
print("\n[1/4] Universum + OHLC laden ...", flush=True)
if CACHE_FILE.exists():
    with open(CACHE_FILE, "rb") as f:
        ohlc = pickle.load(f)
    print(f"  Cache: {len(ohlc)} Ticker (aus crossover_universe.pkl)", flush=True)
else:
    print("  Cache fehlt -- Notfall-Download via yfinance.", flush=True)
    hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers=hdr, timeout=30).text
    tickers = [str(t).replace(".", "-")
               for t in pd.read_html(io.StringIO(html))[0]["Symbol"].tolist()]
    ohlc = {}
    BATCH = 60
    for i in range(0, len(tickers), BATCH):
        ch = tickers[i:i+BATCH]
        try:
            data = yf.download(ch, start=START, interval="1d", progress=False,
                               group_by="ticker", threads=True, auto_adjust=True)
        except Exception as ex:
            print(f"  Batch {i}: {ex}", flush=True); continue
        for t in ch:
            try:
                d = data[t][["Close"]].dropna()
                if len(d) < 280: continue
                c = d["Close"].astype(float).values.ravel()
                dts = pd.to_datetime(d.index).normalize()
                ohlc[t] = pd.DataFrame({"date": dts, "close": c})
            except Exception:
                pass
        print(f"  {min(i+BATCH,len(tickers))}/{len(tickers)} "
              f"({len(ohlc)} ok)", flush=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(ohlc, f)

# 2) SPY (Benchmark, fuer Excess)
print("\n[2/4] SPY (Benchmark) ...", flush=True)
spy_raw = yf.download("SPY", start=START, interval="1d", progress=False,
                      auto_adjust=True)["Close"]
spy = pd.Series(np.asarray(spy_raw, float).ravel(),
                index=pd.to_datetime(spy_raw.index).normalize()).dropna()
print(f"  SPY: {len(spy)} Tage {spy.index.min().date()}..{spy.index.max().date()}",
      flush=True)

# 3) Events sammeln
print(f"\n[3/4] Golden-Cross-Events f={FAST}/s={SLOW}/h={H} sammeln ...",
      flush=True)
pfx = {}            # pfx[date_t] = [sum_excess, sum_raw, count]
dly = {}            # dly[date_d] = [sum_r1, count]
base_n, base_sum = 0, 0.0          # unbedingte h-Tages-Basis ueber alle Aktien-Tage

skipped = 0
for sym, df in ohlc.items():
    c = df["close"].values.astype(float)
    dts = pd.to_datetime(df["date"]).values
    dts = pd.DatetimeIndex(dts).normalize()
    n = len(c)
    if n < SLOW + H + 1:
        skipped += 1; continue
    sp = spy.reindex(dts).values
    # MAs
    ma_f = pd.Series(c).rolling(FAST).mean().values
    ma_s = pd.Series(c).rolling(SLOW).mean().values
    diff = ma_f - ma_s
    prev = diff[:-1]; cur = diff[1:]
    gold_mask = np.isfinite(prev) & np.isfinite(cur) & (prev <= 0) & (cur > 0)
    gold_idx  = np.flatnonzero(gold_mask) + 1   # Crossover an Index t
    # h-Tages Roh-Return
    fwd = np.full(n, np.nan)
    fwd[:n-H] = c[H:] / c[:n-H] - 1.0
    spf = np.full(n, np.nan)
    if np.isfinite(sp).any():
        spf[:n-H] = sp[H:] / sp[:n-H] - 1.0
    # 1d-Returns fuer (B)-Overlap
    cc = np.empty(n); cc[0] = np.nan
    cc[1:] = c[1:] / c[:-1] - 1.0
    # unbedingte Basis (alle gueltigen Aktie-Tage)
    okb = np.isfinite(fwd)
    base_n  += int(okb.sum())
    base_sum += float(np.nansum(fwd[okb]))
    # Event-Filter: nur Indizes, an denen fwd UND spf gueltig sind
    for i in gold_idx:
        if i >= n: continue
        if not np.isfinite(fwd[i]) or not np.isfinite(spf[i]):
            continue
        dt = dts[i]
        ex = fwd[i] - spf[i]
        a = pfx.setdefault(dt, [0.0, 0.0, 0])
        a[0] += ex; a[1] += fwd[i]; a[2] += 1
        # (B) Overlap: Position lebt an Tagen i+1 .. i+H
        for j in range(1, H + 1):
            k = i + j
            if k >= n or not np.isfinite(cc[k]): continue
            b = dly.setdefault(dts[k], [0.0, 0])
            b[0] += cc[k]; b[1] += 1

print(f"  Ticker verarbeitet: {len(ohlc)-skipped} (skipped wegen Laenge: {skipped})",
      flush=True)
print(f"  Einstiegstage gesamt: {len(pfx)}", flush=True)
basem = base_sum / base_n if base_n else 0.0
print(f"  Unbedingte Basis h={H}: Mittel {basem*100:+.3f}% (n={base_n:,})",
      flush=True)

# 4) Auswertung
print("\n" + "="*100)
print("(A) EXCESS vs SPY -- per Einstiegstag (ehrliches n=#Tage)")
print("="*100)
items = sorted(pfx.items())
dts_all = pd.DatetimeIndex([d for d, _ in items])
ex_all  = np.array([v[0] / v[2] for _, v in items])  # Korb-Mittel Excess/Tag
raw_all = np.array([v[1] / v[2] for _, v in items])
cnt_all = np.array([v[2] for _, v in items], int)
rb_all  = raw_all - basem                              # raw - Basis als Sanity
isb     = dts_all < SPLIT

print(f"Tage gesamt={len(ex_all)}  MedianKorb={int(np.median(cnt_all))}  "
      f"MaxKorb={int(cnt_all.max())}  MinKorb={int(cnt_all.min())}")

results = {}   # fuer Gate-Check
for seg, m in (("IS ", isb), ("OOS", ~isb)):
    xs, ds, cs = ex_all[m], dts_all[m], cnt_all[m]
    if len(xs) == 0:
        print(f"  {seg}: keine Tage"); continue
    mt, mnn = monthly_t(ds, xs)
    nocov = ~((ds >= COVID_A) & (ds <= COVID_B))
    xc = xs[nocov]
    print(f"\n  {seg}: Tage={len(xs):>4} | MedianKorb={int(np.median(cs))} | "
          f"exØ {xs.mean()*100:+6.3f}% | t={tstat(xs):+5.2f} | "
          f"MB-t={mt:+5.2f}(n={mnn}) | %+={(xs>0).mean():4.0%}")
    print(f"        rawMB Ø {rb_all[m].mean()*100:+.3f}%  (Sanity)")
    print(f"        COVID-exkl: Tage={len(xc):>4} | exØ {xc.mean()*100:+6.3f}% "
          f"| t={tstat(xc):+5.2f}")
    seg_costs = {}
    for cnet in COSTS:
        xn = xs - cnet / 1e4
        seg_costs[cnet] = (xn.mean(), tstat(xn), float((xn > 0).mean()))
        print(f"        netto@{cnet:>2.0f}bps: exØ {xn.mean()*100:+6.3f}% "
              f"| t={tstat(xn):+5.2f} | %+={(xn>0).mean():4.0%}")
    results[seg.strip()] = {
        "ex_mean": xs.mean(),
        "ex_t":    tstat(xs),
        "mb_t":    mt,
        "covid_excl_mean": xc.mean() if len(xc) else float("nan"),
        "net5":     seg_costs[5.0],
    }

# (B) Reales Overlap-Portfolio
print("\n" + "="*100)
print("(B) REALES taeglich rebalanciertes OVERLAP-Portfolio (Roh-Long)")
print("="*100)
it_b   = sorted(dly.items())
dd_b   = pd.DatetimeIndex([d for d, _ in it_b])
rg_b   = np.array([v[0] / v[1] for _, v in it_b])     # Roh-Tagesrendite Portfolio
m_b    = np.array([v[1] for _, v in it_b])            # offene Positionen/Tag
isb_b  = dd_b < SPLIT
print(f"Aktive Tage gesamt={len(rg_b)}  Median offen={int(np.median(m_b))}  "
      f"Max offen={int(m_b.max())}  Min offen={int(m_b.min())}")
overlap_med_oos = None
for seg, mask in (("IS ", isb_b), ("OOS", ~isb_b)):
    rs, ds = rg_b[mask], dd_b[mask]
    if len(rs) == 0:
        print(f"  {seg}: keine Tage"); continue
    med_open = int(np.median(m_b[mask]))
    if seg.strip() == "OOS":
        overlap_med_oos = med_open
    print(f"\n  {seg}: Tage={len(rs):>4} | Ø/Tag {rs.mean()*1e4:+5.2f}bps | "
          f"t={tstat(rs):+5.2f} | ann Ø {rs.mean()*252*100:+5.1f}% | "
          f"REAL MaxDD={maxdd(rs):5.1%} | MedOffen={med_open}")
    for cnet in COSTS:
        rn = rs - (cnet / 1e4) / H
        print(f"        netto@{cnet:>2.0f}bps: Ø/Tag {rn.mean()*1e4:+5.2f}bps "
              f"| t={tstat(rn):+5.2f} | ann {rn.mean()*252*100:+5.1f}% "
              f"| MaxDD={maxdd(rn):5.1%}")
    nocov = ~((ds >= COVID_A) & (ds <= COVID_B))
    rc = rs[nocov]
    print(f"        COVID-exkl: Tage={len(rc):>4} | Ø/Tag "
          f"{rc.mean()*1e4:+5.2f}bps | t={tstat(rc):+5.2f} "
          f"| netto@5 t={tstat(rc-(5/1e4)/H):+5.2f}")

# Gate-Check
print("\n" + "="*100)
print("PRE-REG GATE-CHECK (alle 5 muessen halten)")
print("="*100)
passed = []
if "IS" in results and "OOS" in results:
    is_r, oos_r = results["IS"], results["OOS"]
    passed.append(gate("G1 IS Excess > 0, t > +2.0",
                       is_r["ex_mean"] > 0 and is_r["ex_t"] > 2.0,
                       f"Ø={is_r['ex_mean']*100:+.3f}% t={is_r['ex_t']:+.2f}"))
    passed.append(gate("G2 OOS Excess > 0, t > +1.5",
                       oos_r["ex_mean"] > 0 and oos_r["ex_t"] > 1.5,
                       f"Ø={oos_r['ex_mean']*100:+.3f}% t={oos_r['ex_t']:+.2f}"))
    n5_mean, n5_t, _ = oos_r["net5"]
    passed.append(gate("G3 OOS netto@5bps > 0, t > +1.0",
                       n5_mean > 0 and n5_t > 1.0,
                       f"Ø={n5_mean*100:+.3f}% t={n5_t:+.2f}"))
    passed.append(gate("G4 OOS COVID-exkl Mean > 0",
                       oos_r["covid_excl_mean"] > 0,
                       f"Ø={oos_r['covid_excl_mean']*100:+.3f}%"))
if overlap_med_oos is not None:
    passed.append(gate("G5 OOS MedianOffen >= 5",
                       overlap_med_oos >= 5,
                       f"MedOffen={overlap_med_oos}"))

print("\n" + "="*100)
if all(passed) and len(passed) == 5:
    print("ERGEBNIS: ALLE 5 GATES PASS  -- Edge greenlight fuer naechsten Schritt.")
    print("Sinnvoll: Paper-Trading-Sim mit echten Daten, Sektor-Konzentrations-Check.")
else:
    print(f"ERGEBNIS: {sum(passed)}/5 Gates  -- Edge NICHT greenlight.")
    print("Faden ehrlich schliessen oder Variante mit Conditioner (EMA/Sektor/Vola)")
    print("nach SEPARATER Pre-Reg testen, nicht hier nachfrisieren.")
print("="*100)
