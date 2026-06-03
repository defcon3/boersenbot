#!/usr/bin/env python3
"""
Export der Overnight-vs-Intraday-Zerlegung als JSON fuer die Web-Show-Data-Seite.
Reproduziert overnight_intraday_test.py, dumpt Kurven+Kennzahlen nach
overnight_intraday_data.json (wird ins Template inline-eingebettet).
"""
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np
import pandas as pd
import yfinance as yf

START = "1999-01-01"
END = "2026-06-30"
PRIMARY = "SPY"
NAMES = ["AAPL", "MSFT", "JNJ", "KO", "XOM"]
COSTS_BPS = [5, 10]


def decompose(df):
    o = df["Open"].astype(float); c = df["Close"].astype(float); pc = c.shift(1)
    return pd.DataFrame({"overnight": o / pc - 1.0, "intraday": c / o - 1.0,
                         "total": c / pc - 1.0}, index=df.index).dropna()

def sharpe(r):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    if len(r) < 2 or r.std(ddof=1) == 0: return None
    return float((r.mean() / r.std(ddof=1)) * np.sqrt(252))

def cagr(r):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    if len(r) == 0: return None
    return float(np.prod(1 + r) ** (252 / len(r)) - 1)

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


print("Lade SPY...", flush=True)
spy = fetch(PRIMARY)
dec = decompose(spy)

# Kumulative Equity-Kurven (Start=100), monatlich downgesampled fuers Chart
eq = pd.DataFrame({
    "overnight": (1 + dec["overnight"]).cumprod() * 100,
    "intraday": (1 + dec["intraday"]).cumprod() * 100,
    "total": (1 + dec["total"]).cumprod() * 100,
})
eq_m = eq.resample("ME").last().dropna()
curve = {
    "dates": [d.strftime("%Y-%m") for d in eq_m.index],
    "overnight": [round(v, 1) for v in eq_m["overnight"]],
    "intraday": [round(v, 1) for v in eq_m["intraday"]],
    "total": [round(v, 1) for v in eq_m["total"]],
}

# Subperioden-Stabilitaet
mid = len(dec) // 2
def cum(seg, col): return float(np.prod(1 + seg[col]) - 1)
subperiods = []
for label, seg in [("Gesamt 1999–2026", dec),
                   (f"H1 bis {dec.index[mid].date()}", dec.iloc[:mid]),
                   (f"H2 ab {dec.index[mid].date()}", dec.iloc[mid:])]:
    subperiods.append({"label": label, "n": int(len(seg)),
                       "overnight": round(cum(seg, "overnight") * 100, 1),
                       "intraday": round(cum(seg, "intraday") * 100, 1),
                       "total": round(cum(seg, "total") * 100, 1)})

# Strategie-Kennzahlen (gross)
strat = {}
for name, col in [("Buy&Hold", "total"), ("Overnight-only", "overnight"),
                  ("Intraday-only", "intraday")]:
    s = dec[col]
    strat[name] = {"cagr": round(cagr(s) * 100, 2), "sharpe": round(sharpe(s), 2),
                   "maxdd": round(max_dd(s) * 100, 1)}

# Net-of-cost (2 Legs/Tag)
bh_sharpe = sharpe(dec["total"])
netcost = []
for bps in COSTS_BPS:
    cost = 2 * bps / 10000.0
    on_net = dec["overnight"] - cost
    id_net = dec["intraday"] - cost
    netcost.append({"bps": bps,
                    "on_cagr": round(cagr(on_net) * 100, 2), "on_sharpe": round(sharpe(on_net), 2),
                    "id_cagr": round(cagr(id_net) * 100, 2), "id_sharpe": round(sharpe(id_net), 2)})

# Cross-Section
print("Lade Einzelnamen...", flush=True)
xsec = []
for nm in NAMES:
    d = decompose(fetch(nm))
    xsec.append({"name": nm,
                 "overnight": round(cum(d, "overnight") * 100, 1),
                 "intraday": round(cum(d, "intraday") * 100, 1)})

# Sensitivitaet: adjusted (Total-Return) vs price-only (Dividende raus).
# SPY hat nie gesplittet -> auto_adjust=False unterscheidet sich von True NUR
# durch die Dividende (kein Split-Confound). Siehe overnight_intraday_sensitivity.py.
print("Sensitivitaet adjusted vs price-only (SPY)...", flush=True)
raw = yf.download(PRIMARY, start=START, end=END, interval="1d",
                  progress=False, auto_adjust=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
draw = decompose(raw[["Open", "Close"]].dropna())
sidx = dec.index.intersection(draw.index)
on_adj = float(np.prod(1 + dec.loc[sidx, "overnight"]) - 1) * 100
on_raw = float(np.prod(1 + draw.loc[sidx, "overnight"]) - 1) * 100
id_adj = float(np.prod(1 + dec.loc[sidx, "intraday"]) - 1) * 100
id_raw = float(np.prod(1 + draw.loc[sidx, "intraday"]) - 1) * 100
sensitivity = {
    "overnight_adj": round(on_adj, 1), "overnight_raw": round(on_raw, 1),
    "intraday_adj": round(id_adj, 1), "intraday_raw": round(id_raw, 1),
    "div_share_pct": round((on_adj - on_raw) / on_adj * 100),   # % der adj-Overnight = Dividende
    "price_share_pct": round(on_raw / on_adj * 100),
    "still_dominates": bool(on_raw > id_raw),
}

data = {
    "generated": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "n_days": int(len(dec)),
    "range": [dec.index.min().strftime("%Y-%m-%d"), dec.index.max().strftime("%Y-%m-%d")],
    "curve": curve,
    "subperiods": subperiods,
    "strat": strat,
    "bh_sharpe": round(bh_sharpe, 2),
    "netcost": netcost,
    "xsec": xsec,
    "sensitivity": sensitivity,
}
with open("overnight_intraday_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"Geschrieben: overnight_intraday_data.json ({len(curve['dates'])} Monatspunkte)")
