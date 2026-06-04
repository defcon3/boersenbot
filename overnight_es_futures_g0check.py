#!/usr/bin/env python3
"""
G0-DATENCHECK fuer die ES-Futures-Pre-Reg (preregs/overnight_es_futures_2026_06_04.md)

FRAGE (G0): Kann Gratis-yfinance ES=F die CASH-EQUITY-Overnight-Session
(US-Cash-Close 16:00 ET -> naechster Cash-Open 09:30 ET) sauber von der
RTH-/Globex-Session trennen? Nur dann ist der Overnight-Test ueberhaupt valide.

DIAGNOSE (zwei harte Belege):
  (1) Tages-OHLC-Sessiongrenze: Wenn ES=F-Tagesbars die VOLLE Globex-Session
      abdecken (~18:00 ET Vortag -> 17:00 ET), ist die Close->Open-"Luecke"
      nur der winzige Globex-Reopen-Gap, NICHT die Cash-Overnight-Strecke.
      Test: reproduziert die ES=F-Overnight/Intraday-Zerlegung das bei SPY
      bekannte Muster (Overnight >> Intraday)? Wenn NICHT, sind die Sessions
      nicht cash-aligned -> der "Overnight"-Bucket misst etwas anderes.
  (2) Freie Intraday-Abdeckung: yfinance gibt 1m nur ~7 Tage, 1h ~730 Tage.
      Damit lassen sich keine session-sauberen Overnight-Strecken ueber 2010+
      bauen. Test: wie viel Historie liefert 1m/1h wirklich?

VERDIKT: G0 PASS nur, wenn (1) cash-aligned UND (2) genug Intraday-Historie.
Sonst G0 FAIL -> Pre-Reg dokumentiert verworfen (kein Rechnen mit Approximation).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf

START = "2010-01-01"
END = "2026-06-30"


def decompose(df):
    o, c = df["Open"].astype(float), df["Close"].astype(float)
    pc = c.shift(1)
    return pd.DataFrame({"overnight": o / pc - 1.0, "intraday": c / o - 1.0},
                        index=df.index).dropna()


def fetch_daily(sym):
    d = yf.download(sym, start=START, end=END, interval="1d",
                    progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d[["Open", "Close"]].dropna()


def cum(s):
    return float(np.prod(1 + s.dropna()) - 1)


print("=" * 74)
print("G0-DATENCHECK — ES=F Session-Sauberkeit (yfinance gratis)")
print("=" * 74)

# --- (1) Tages-Sessiongrenze: ES=F vs SPY Overnight/Intraday-Zerlegung -----
print("\n[1] Reproduziert ES=F die Cash-Overnight-Dominanz (wie SPY)?\n", flush=True)
spy = decompose(fetch_daily("SPY"))
es = decompose(fetch_daily("ES=F"))
idx = spy.index.intersection(es.index)
spy, es = spy.loc[idx], es.loc[idx]
print(f"  gemeinsame Handelstage: {len(idx)}  ({idx.min().date()}..{idx.max().date()})")

rows = []
for name, dec in [("SPY (Cash-ETF)", spy), ("ES=F (Futures)", es)]:
    c_on, c_id = cum(dec["overnight"]) * 100, cum(dec["intraday"]) * 100
    share = c_on / (c_on + c_id) * 100 if (c_on + c_id) != 0 else float("nan")
    rows.append((name, c_on, c_id, c_on > c_id))
    print(f"  {name:16s} | Overnight {c_on:+8.1f}%  Intraday {c_id:+8.1f}%  "
          f"on>id: {'JA' if c_on > c_id else 'NEIN'}")

spy_on_dom = rows[0][3]
es_on_dom = rows[1][3]
# Korrelation der Tages-Overnight-Returns: bei cash-aligned Sessions hoch,
# bei Globex-Vollsession niedrig (ES misst eine andere Strecke).
corr_on = float(np.corrcoef(spy["overnight"], es["overnight"])[0, 1])
corr_id = float(np.corrcoef(spy["intraday"], es["intraday"])[0, 1])
print(f"\n  Korrelation SPY vs ES=F  Overnight-Returns: {corr_on:+.2f}")
print(f"  Korrelation SPY vs ES=F  Intraday-Returns : {corr_id:+.2f}")
print("  (cash-aligned => beide hoch ~0.9+. Niedrige Overnight-Korr => ES=F")
print("   'Overnight' ist NICHT die Cash-Overnight-Strecke.)")
sessions_aligned = (corr_on > 0.8) and es_on_dom

# --- (2) Freie Intraday-Abdeckung -----------------------------------------
print("\n[2] Wie viel Intraday-Historie liefert yfinance gratis fuer ES=F?\n", flush=True)
cov = {}
for interval in ["1h", "5m", "1m"]:
    try:
        d = yf.download("ES=F", period="730d" if interval == "1h" else "60d",
                        interval=interval, progress=False, auto_adjust=False)
        if d.empty:
            cov[interval] = (0, None, None)
        else:
            cov[interval] = (len(d), d.index.min(), d.index.max())
    except Exception as e:
        cov[interval] = (-1, str(e)[:40], None)
    n = cov[interval][0]
    if n > 0:
        span = (cov[interval][2] - cov[interval][1]).days
        print(f"  {interval:>3}: {n:>6} Bars, {cov[interval][1].date()}..{cov[interval][2].date()} "
              f"(~{span} Tage Historie)")
    else:
        print(f"  {interval:>3}: keine/zuwenig Daten ({cov[interval][1]})")

# 1m-Historie in Tagen (braucht 2010+ fuer den Test -> ~5800 Tage)
h1_days = (cov["1h"][2] - cov["1h"][1]).days if cov["1h"][0] > 0 else 0
enough_history = h1_days > 3650          # mind. ~10 Jahre noetig

# --- VERDIKT ---------------------------------------------------------------
print("\n" + "=" * 74)
print("G0-VERDIKT")
print("=" * 74)
print(f"  (1) ES=F-Tagesbars cash-aligned (Overnight-Korr>0.8 & on>id): "
      f"{'JA' if sessions_aligned else 'NEIN'}")
print(f"  (2) Genug freie Intraday-Historie (>10J): "
      f"{'JA' if enough_history else 'NEIN'} (1h reicht ~{h1_days} Tage zurueck)")
g0_pass = sessions_aligned and enough_history
print(f"\n  G0: {'PASS — ES-Test darf rechnen' if g0_pass else 'FAIL — Pre-Reg verwerfen'}")
if not g0_pass:
    print("  -> Gratis-yfinance trennt die Cash-Overnight-Session nicht sauber")
    print("     (Globex-Vollsession bei Tagesbars; Intraday-Historie zu kurz).")
    print("     Saubere Sessions brauchen bezahlte Intraday-Daten (Polygon/CME).")
    print("     Disziplin: lieber nicht testen als mit Approximation Selbstbetrug.")
