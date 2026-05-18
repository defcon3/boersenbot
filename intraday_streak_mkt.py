#!/usr/bin/env python3
"""
Intraday-Streak (N,k)-GITTER, MARKT-BEREINIGT -- Plan Schritt 2 / Stufe-2.

Stufe-2-Faktor: Markt-/Index-Kontext (SPY). Statt der Roh-Rendite der
Einzelaktie wird die IDIOSYNKRATISCHE (marktrelative) Rendite verwendet:

    excess_i(t) = logret_i(t) - logret_SPY(t)        (gleiche Minute)

Streak UND Vorwaerts-Kumulrendite werden auf dieser Excess-Serie
gebildet. Hypothese: Die in der Roh-EDA (intraday_streak.py) gefundene
Reversion ist groesstenteils Markt-Mean-Reversion / Bid-Ask-Bounce der
Einzelaktie -- auf marktrelativer Ebene koennte Continuation existieren
("Aktie laeuft dem Markt N Minuten davon -> laeuft sie weiter davon?").

Regeln unveraendert: kein Look-ahead (Streak nur Vergangenheit), kein
Overnight (Streak + alle k Folgeminuten im selben Handelstag). SPY und
Einzelwert werden je Minuten-Timestamp ausgerichtet (inner join).

HINWEIS: Excess-Edge in bps ist NICHT 1:1 handelbar (impliziert
SPY-Hedge). Diese EDA klaert nur, OB auf idiosynkratischer Ebene
ueberhaupt Continuation-Signal existiert.
"""
import warnings; warnings.filterwarnings("ignore")
import io, numpy as np, pandas as pd, yfinance as yf, requests
from scipy import stats

K_LIST = [1, 2, 3, 5, 10]
N_MAX  = 15
COST_BPS = 2.0

print("Lade S&P-500-Liste...", flush=True)
hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                    headers=hdr, timeout=20).text
tickers = [str(t).replace(".", "-")
           for t in pd.read_html(io.StringIO(html))[0]["Symbol"].tolist()]

print("Lade SPY 1-Min (Markt-Kontext, 7 Tage)...", flush=True)
spy = yf.download("SPY", period="7d", interval="1m", progress=False)
spy_close = spy["Close"].dropna()
spy_close.index = pd.to_datetime(spy_close.index)
spy_close = spy_close.sort_index()
spy_close = spy_close[~spy_close.index.duplicated(keep="first")]
spy_p = spy_close.values.astype(float).ravel()
spy_lr = pd.Series(np.append([np.nan], np.log(spy_p[1:] / spy_p[:-1])),
                   index=spy_close.index)
print(f"  SPY: {len(spy_lr):,} Minuten. {len(tickers)} Ticker, Batches...",
      flush=True)

BATCH = 60
base = {k: [0, 0] for k in K_LIST}
dn   = {}
up   = {}


def process_day(x):
    """x: 1-D np.array der EXCESS-Log-Renditen EINES Handelstages.
    x[0] (1. Min/Tag) ist NaN."""
    L = len(x)
    if L < 30:
        return
    finite = ~np.isnan(x)
    down = np.zeros(L, bool); up_m = np.zeros(L, bool)
    down[finite] = x[finite] < 0
    up_m[finite] = x[finite] > 0

    ds = np.zeros(L, int); us = np.zeros(L, int)
    for i in range(1, L):
        ds[i] = ds[i-1] + 1 if down[i] else 0
        us[i] = us[i-1] + 1 if up_m[i] else 0

    xf = np.where(np.isnan(x), 0.0, x)
    cs = np.cumsum(xf)
    for k in K_LIST:
        F = np.full(L, np.nan)
        hi = L - k
        if hi > 0:
            idx = np.arange(hi)
            F[idx] = cs[idx + k] - cs[idx]
        valid = ~np.isnan(F)
        fup = (F > 0)
        base[k][0] += int(valid.sum())
        base[k][1] += int(np.count_nonzero(fup & valid))
        for tbl, streak, cont, edge in (
            (dn, ds, ~fup, -F),
            (up, us,  fup,  F),
        ):
            for N in range(1, N_MAX + 1):
                sel = ((streak == N) if N < N_MAX else (streak >= N)) & valid
                n = int(sel.sum())
                if n == 0:
                    continue
                b = tbl.setdefault((N, k), [0, 0, 0.0])
                b[0] += n
                b[1] += int(np.count_nonzero(cont & sel))
                b[2] += float(np.sum(edge[sel]))


def process_symbol(close, idx):
    s = pd.Series(np.asarray(close).ravel(), index=pd.to_datetime(idx)).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    p = s.values.astype(float)
    L = len(p)
    if L < 2:
        return
    sym_lr = np.full(L, np.nan)
    sym_lr[1:] = np.log(p[1:] / p[:-1])
    # Markt-Kontext auf die Symbol-Timestamps ausrichten
    spy_aligned = spy_lr.reindex(s.index).values.astype(float)
    excess = sym_lr - spy_aligned                # idiosynkratische Rendite
    day = s.index.normalize()
    boundary = np.empty(L, bool)
    boundary[0] = True
    boundary[1:] = day[1:].asi8 != day[:-1].asi8
    excess[boundary] = np.nan
    starts = np.flatnonzero(boundary)
    ends = np.append(starts[1:], L)
    for a, b in zip(starts, ends):
        process_day(excess[a:b])


for i in range(0, len(tickers), BATCH):
    ch = tickers[i:i + BATCH]
    try:
        data = yf.download(ch, period="7d", interval="1m", progress=False,
                           group_by="ticker", threads=True)
    except Exception as ex:
        print(f"Batch {i}: {ex}", flush=True); continue
    for t in ch:
        if t == "SPY":
            continue
        try:
            c = data[t]["Close"].dropna()
            if len(c) > 100:
                process_symbol(c, c.index)
        except Exception:
            pass
    print(f"  {min(i + BATCH, len(tickers))}/{len(tickers)}", flush=True)


def show(tbl, titel, base_cont):
    print("\n" + "=" * 78)
    print(titel)
    print("=" * 78)
    head = "{:>4} | {:>3} | {:>9} | {:>11} | {:>12} | {:>4} | {:>6}".format(
        "N", "k", "Faelle", "P(cont)", "O Exc bps", "Sig", "Trade?")
    print(head)
    print("-" * len(head))
    for N in range(1, N_MAX + 1):
        for k in K_LIST:
            key = (N, k)
            if key not in tbl:
                continue
            n, w, se = tbl[key]
            p = w / n
            edge_bps = se / n * 1e4
            null = base_cont[k]
            pv = stats.binomtest(int(w), int(n), null).pvalue
            st = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""
            trade = "JA" if (edge_bps > COST_BPS and pv < 0.05) else ""
            tag = (">=" + str(N)) if N == N_MAX else str(N)
            print("{:>4} | {:>3} | {:>9} | {:>11.4f} | {:>+12.3f} | {:>4} | {:>6}"
                  .format(tag, k, n, p, edge_bps, st, trade))
        print("-" * len(head))


tot = sum(base[k][0] for k in K_LIST)
print(f"\nMARKT-BEREINIGT. Gesamt: ~{len(tickers)} Aktien, 7 Tage, "
      f"{tot:,} (N,k)-Beobachtungen")
print("\nUnbedingte Basisraten je Horizont k (Excess > 0):")
base_up, base_down = {}, {}
for k in K_LIST:
    nv, nu = base[k]
    bu = nu / nv if nv else float("nan")
    base_up[k] = bu
    base_down[k] = 1.0 - bu
    print(f"  k={k:>2}: P(Excess-Kumul > 0) = {bu:.4f}  (n={nv:,})")

show(dn, "DOWN-Excess-Streak -> laeuft Aktie weiter UNTER Markt? (Put)", base_down)
show(up, "UP-Excess-Streak -> laeuft Aktie weiter UEBER Markt? (Call)", base_up)

print(f"\nKosten-Huerde COST_BPS = {COST_BPS} bps (Orientierung; Excess-Edge")
print("ist nicht 1:1 handelbar -> impliziert SPY-Hedge). 'Trade?'=JA nur")
print("wenn O Exc bps > Huerde UND p<0.05. Kein JA / negativer Excess-Edge")
print("=> auch idiosynkratisch keine Continuation -> Stufe-2 erschoepft.")
