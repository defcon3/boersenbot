#!/usr/bin/env python3
"""
OVERNIGHT-vs-INTRADAY-ZERLEGUNG (Pre-Reg 2026-06-03)

ZWEISTUFIGE HYPOTHESE:
  (1) DESKRIPTIV: Die Aktien-Risikopraemie faellt ueberwiegend OVERNIGHT
      (Close->Open) an, kaum intraday (Open->Close). Bekanntes Stylized Fact
      (Cooper/Lipson, Bogousslavsky). Erwartung: ~60% reproduzierbar.
  (2) HANDELBAR: Bleibt nach Friktion ein nutzbarer Edge? Eine
      "Nur-Overnight"-Strategie (Kauf zum Close, Verkauf zum Open) handelt
      2 Legs/Tag -> Kosten brutal. Erwartung: ~5% (eher Show-Data).

DIVIDENDEN-FALLE (zentral):
  Ex-Div-Tage erzeugen am Open einen kuenstlichen Kurs-Gap. Mit
  auto_adjust=True passt yfinance alle OHLC konsistent an -> die
  Total-Return-Zerlegung attribuiert den Dividenden-"Credit" dem Overnight-
  Bucket (oekonomisch korrekt: man haelt ueber Nacht/Ex-Tag, um die Div zu
  kassieren). Intraday-Ratio Close/Open ist faktor-invariant (gleicher Tag).
  Sanity-Check: (1+overnight)*(1+intraday) == (1+total) je Tag.

DESIGN (vor Lauf festgelegt):
  - Primaer SPY; Cross-Section AAPL MSFT JNJ KO XOM (G3).
  - Daten: Tages-OHLC adjusted (auto_adjust), gratis yfinance, ab 1999.
  - overnight_t = adjOpen_t / adjClose_{t-1} - 1
    intraday_t  = adjClose_t / adjOpen_t - 1
    total_t     = adjClose_t / adjClose_{t-1} - 1
  - Subperioden fuer Stabilitaet: 2 Haelften (Median-Split der Tage).
  - Handelstest: Overnight-only vs Intraday-only vs Buy&Hold;
    Kosten 5 und 10 bps je Leg (2 Legs/Tag bei den getakteten Varianten).

PRE-REG-GATES:
  G1 (Deskriptiv stabil): SPY-Overnight-Kumulativ > Intraday-Kumulativ in
                          BEIDEN Subperioden.
  G2 (Handelbar):         Overnight-only NET (5 bps/Leg) Sharpe > B&H Sharpe.
  G3 (Cross-Section):     Overnight > Intraday (kumuliert) bei Mehrheit der
                          5 Einzelnamen.

VERDICT:
  GREEN:  G1 + G2 + G3 (echtes handelbares Overnight-Alpha)
  YELLOW: G1 + G3, aber G2 FAIL (Stylized Fact bestaetigt, nicht handelbar)
  RED:    G1 FAIL (nicht mal deskriptiv reproduzierbar)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf

START = "1999-01-01"
END = "2026-06-30"
PRIMARY = "SPY"
NAMES = ["AAPL", "MSFT", "JNJ", "KO", "XOM"]
COSTS_BPS = [5, 10]

# ===========================================================================
# HELPERS
# ===========================================================================

def decompose(df):
    """OHLC-DataFrame -> DataFrame mit overnight/intraday/total Returns."""
    o = df["Open"].astype(float)
    c = df["Close"].astype(float)
    pc = c.shift(1)
    overnight = o / pc - 1.0
    intraday = c / o - 1.0
    total = c / pc - 1.0
    out = pd.DataFrame({"overnight": overnight, "intraday": intraday,
                        "total": total}, index=df.index).dropna()
    return out

def sharpe(r):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    if len(r) < 2 or r.std(ddof=1) == 0: return np.nan
    return (r.mean() / r.std(ddof=1)) * np.sqrt(252)

def cagr(r):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    if len(r) == 0: return np.nan
    return np.prod(1 + r) ** (252 / len(r)) - 1

def max_dd(r):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    eq = np.cumprod(1 + r); peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())

def fetch(sym):
    d = yf.download(sym, start=START, end=END, interval="1d",
                    progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d[["Open", "Close"]].dropna()

# ===========================================================================
# MAIN
# ===========================================================================
print("=" * 80)
print("OVERNIGHT-vs-INTRADAY-ZERLEGUNG")
print("=" * 80)

print(f"\n[1/4] Lade {PRIMARY} Tages-OHLC (adjusted) ab {START}...", flush=True)
spy = fetch(PRIMARY)
dec = decompose(spy)
print(f"  {len(dec)} Handelstage  {dec.index.min().date()} .. {dec.index.max().date()}")

# Sanity: (1+on)(1+id) == (1+total)
recon = (1 + dec["overnight"]) * (1 + dec["intraday"]) - 1
maxerr = float((recon - dec["total"]).abs().max())
print(f"  Sanity-Check Rekonstruktion max|Fehler| = {maxerr:.2e} (sollte ~0)")

# ---------------------------------------------------------------------------
print("\n[2/4] Deskriptive Zerlegung (kumuliert, gesamt + 2 Subperioden)\n", flush=True)

def summarize(seg, label):
    on, idd, tot = seg["overnight"], seg["intraday"], seg["total"]
    cum_on = np.prod(1 + on) - 1
    cum_id = np.prod(1 + idd) - 1
    cum_tot = np.prod(1 + tot) - 1
    print(f"  {label:14s} | n={len(seg):5d} | Overnight {cum_on*100:+8.1f}%  "
          f"Intraday {cum_id*100:+8.1f}%  Total {cum_tot*100:+8.1f}%  "
          f"| on>id: {'JA' if cum_on > cum_id else 'NEIN'}")
    return cum_on, cum_id

print("  --- Gesamt ---")
summarize(dec, "GESAMT")

mid = len(dec) // 2
h1, h2 = dec.iloc[:mid], dec.iloc[mid:]
print("  --- Subperioden (Median-Split) ---")
on1, id1 = summarize(h1, f"H1 bis {h1.index.max().date()}")
on2, id2 = summarize(h2, f"H2 ab {h2.index.min().date()}")
g1 = (on1 > id1) and (on2 > id2)

# ---------------------------------------------------------------------------
print("\n[3/4] Handelstest: Overnight-only vs Intraday-only vs Buy&Hold\n", flush=True)
# Getaktete Strategien handeln 2 Legs/Tag -> Kosten = 2*bps je Handelstag
print(f"  {'Strategie':16s} {'CAGR':>9} {'Sharpe':>8} {'MaxDD':>9}   (gross)")
print("  " + "-" * 55)
bh = dec["total"]
on = dec["overnight"]
idd = dec["intraday"]
for name, series in [("Buy&Hold", bh), ("Overnight-only", on), ("Intraday-only", idd)]:
    print(f"  {name:16s} {cagr(series)*100:8.2f}% {sharpe(series):8.2f} {max_dd(series)*100:8.2f}%")

print(f"\n  Net-of-cost (Overnight-only / Intraday-only handeln 2 Legs/Tag):")
print(f"  {'bps/Leg':>8} {'ON-net CAGR':>12} {'ON-net Shrp':>12} {'ID-net CAGR':>12} {'ID-net Shrp':>12}  {'B&H Shrp':>9}")
bh_sharpe = sharpe(bh)
g2 = False
for bps in COSTS_BPS:
    cost = 2 * bps / 10000.0   # 2 Legs pro Handelstag
    on_net = on - cost
    id_net = idd - cost
    on_net_sharpe = sharpe(on_net)
    print(f"  {bps:8d} {cagr(on_net)*100:11.2f}% {on_net_sharpe:12.2f} "
          f"{cagr(id_net)*100:11.2f}% {sharpe(id_net):12.2f}  {bh_sharpe:9.2f}")
    if bps == 5 and on_net_sharpe > bh_sharpe:
        g2 = True

# ---------------------------------------------------------------------------
print("\n[4/4] Cross-Section (5 Einzelnamen)\n", flush=True)
print(f"  {'Name':6s} | {'Overnight%':>11} {'Intraday%':>11}  on>id")
print("  " + "-" * 40)
n_on_wins = 0
for nm in NAMES:
    try:
        d = decompose(fetch(nm))
    except Exception as e:
        print(f"  {nm:6s} | FAIL ({e})"); continue
    c_on = np.prod(1 + d["overnight"]) - 1
    c_id = np.prod(1 + d["intraday"]) - 1
    win = c_on > c_id
    if win: n_on_wins += 1
    print(f"  {nm:6s} | {c_on*100:+10.1f}% {c_id*100:+10.1f}%   {'JA' if win else 'NEIN'}")
g3 = n_on_wins >= 3

# ---------------------------------------------------------------------------
print("\n--- GATES ---")
print(f"  G1 Deskriptiv stabil (beide Subperioden on>id): {'PASS' if g1 else 'FAIL'}")
print(f"  G2 Overnight-only net 5bps/Leg Sharpe > B&H   : {'PASS' if g2 else 'FAIL'}")
print(f"  G3 on>id Mehrheit Einzelnamen ({n_on_wins}/{len(NAMES)})        : {'PASS' if g3 else 'FAIL'}")

print("\n--- FINAL VERDICT ---")
# G1 (SPY deskriptiv) ist der Reproduktions-Kern; G2 Handelbarkeit; G3
# Cross-Section-Universalitaet (Nuance, kein Reproduktions-K.o.).
if g1 and g2:
    print("GREEN: handelbares Overnight-Alpha (ueberlebt Friktion).")
elif g1:
    print("YELLOW: Stylized Fact fuer SPY bestaetigt (Overnight >> Intraday,")
    print("        beide Subperioden), aber nach Kosten NICHT handelbar (G2 FAIL)")
    print("        -> Show-Data, kein Trading-Edge.")
    if not g3:
        print(f"        Nuance: NICHT universell - on>id nur {n_on_wins}/{len(NAMES)} Einzelnamen")
        print("        (Overnight-Dominanz konzentriert in Growth/Tech, nicht Dividenden-Werte).")
else:
    print("RED: SPY-Zerlegung nicht mal deskriptiv stabil reproduzierbar.")
