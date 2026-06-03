#!/usr/bin/env python3
"""
SENSITIVITAET zu overnight_intraday_test.py (Reviewer-Frage 2026-06-03):
"Was passiert mit UNADJUSTIERTEN Preisen (Dividenden-Effekt raus)?"

Idee: Die adjustierte Zerlegung (auto_adjust=True) ist eine TOTAL-RETURN-
Zerlegung und schreibt Ex-Div-Gaps dem Overnight-Bucket zu. Hier rechnen wir
dieselbe Zerlegung mit PREIS-ONLY-Kursen (Dividende NICHT zurueckgerechnet)
und vergleichen, wie viel der Overnight-Dominanz reine Dividende ist.

WICHTIG (Split-Falle): unadjustierte Rohpreise wuerden bei Aktiensplits
riesige kuenstliche Overnight-Gaps erzeugen. yfinance auto_adjust=False liefert
Open/Close SPLIT-bereinigt, aber Dividenden-UNbereinigt -> genau was wir wollen.
SPY hat NIE gesplittet -> fuer SPY ist der Vergleich confound-frei (einziger
Unterschied True/False = Dividende). Darum nur SPY.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf

START, END, SYM = "1999-01-01", "2026-06-30", "SPY"


def flat(d):
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d


def decompose(o, c):
    pc = c.shift(1)
    out = pd.DataFrame({"overnight": o / pc - 1, "intraday": c / o - 1,
                        "total": c / pc - 1}).dropna()
    return out


def cum(s):
    return float(np.prod(1 + s) - 1)


# --- adjusted (Total-Return, = Hauptanalyse) ---
adj = flat(yf.download(SYM, start=START, end=END, interval="1d",
                       progress=False, auto_adjust=True))
da = decompose(adj["Open"].astype(float), adj["Close"].astype(float))

# --- price-only (split-bereinigt, dividenden-UNbereinigt) ---
raw = flat(yf.download(SYM, start=START, end=END, interval="1d",
                       progress=False, auto_adjust=False))
dr = decompose(raw["Open"].astype(float), raw["Close"].astype(float))

# auf gemeinsame Tage alignen
idx = da.index.intersection(dr.index)
da, dr = da.loc[idx], dr.loc[idx]

print("=" * 72)
print(f"SENSITIVITAET ADJUSTED vs PRICE-ONLY  ({SYM}, {len(idx)} Tage,")
print(f"  {idx.min().date()}..{idx.max().date()})")
print("=" * 72)

print(f"\n{'Bucket':12s} {'ADJUSTED (TR)':>16s} {'PRICE-ONLY':>14s} {'Differenz':>14s}")
print("-" * 60)
for col in ["overnight", "intraday", "total"]:
    a, p = cum(da[col]) * 100, cum(dr[col]) * 100
    print(f"{col:12s} {a:+15.1f}% {p:+13.1f}% {a-p:+13.1f}%")

# interne Konsistenz: intraday darf sich kaum unterscheiden (Div beruehrt c/o nicht)
id_gap = abs(cum(da["intraday"]) - cum(dr["intraday"])) * 100
print(f"\nKonsistenz-Check Intraday-Differenz: {id_gap:.2f} pp "
      f"(erwartet ~0, da Dividende same-day c/o nicht beruehrt)")

# Kernfrage: dominiert Overnight auch PRICE-ONLY noch ueber Intraday?
on_p, id_p = cum(dr["overnight"]) * 100, cum(dr["intraday"]) * 100
print(f"\nKERNFRAGE: Overnight > Intraday auch PRICE-ONLY?")
print(f"  Overnight price-only = {on_p:+.1f}%  vs  Intraday price-only = {id_p:+.1f}%")
print(f"  -> {'JA, Effekt ueberlebt' if on_p > id_p else 'NEIN, war groesstenteils Dividende'}")

# Wie viel der adjustierten Overnight-Performance ist Dividende?
on_a = cum(da["overnight"]) * 100
div_share = (on_a - on_p) / on_a * 100 if on_a != 0 else float('nan')
print(f"\nDividenden-Anteil an adjustierter Overnight-Kumulierung: {div_share:.0f}%")
print(f"  (adjustiert {on_a:+.1f}% - price-only {on_p:+.1f}% = {on_a-on_p:+.1f}% Dividende)")
