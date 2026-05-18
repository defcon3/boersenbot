#!/usr/bin/env python3
"""
Intraday-Streak (N,k)-GITTER -- Plan-Schritt 1 (EDA, keine Modelle).

Frage: Liegt zum Zeitpunkt t eine Streak von N gleichgerichteten
1-Min-Bewegungen vor -- setzt sich die Bewegung ueber die naechsten
k Minuten (kumulierte Rendite) ueberzufaellig fort?

Pro (N, k, Richtung):
  - Fallzahl
  - P(Continuation) = Anteil, in dem die k-Min-Kumulrendite in
    Streak-Richtung zeigt (Down-Streak -> Put, Up-Streak -> Call)
  - O Edge in bps = mittlere Rendite IN Trade-Richtung (positiv = gut)
  - Binomialtest gegen die unbedingte Richtungs-Basisrate fuer dieses k
  - Markierung, ob der Edge die Kosten-Huerde (COST_BPS) klar uebersteigt

Regeln (wie gehabt): kein Look-ahead (Streak nur aus Vergangenheit),
kein Overnight (Streak UND alle k Folgeminuten strikt innerhalb eines
Handelstages). yfinance liefert 1m nur ~7 Kalendertage -- das ist die
Kalender-Breite-Grenze dieses Laufs (Plan-Schritt 2: breitere Quelle).
"""
import warnings; warnings.filterwarnings("ignore")
import io, numpy as np, pandas as pd, yfinance as yf, requests
from scipy import stats

K_LIST = [1, 2, 3, 5, 10]      # Halte-Horizonte (Minuten)
N_MAX  = 15                    # N=15 bedeutet ">=15"
COST_BPS = 2.0                 # Kosten-Huerde lt. Plan (Spread+Gebuehr Derivat)

print("Lade S&P-500-Liste...", flush=True)
hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                    headers=hdr, timeout=20).text
tickers = [str(t).replace(".", "-")
           for t in pd.read_html(io.StringIO(html))[0]["Symbol"].tolist()]
print(f"{len(tickers)} Ticker. Lade 1-Min-Daten (7 Tage, Batches)...", flush=True)

BATCH = 60
# Akkumulatoren:
#   base[k] = [n_valid, n_fwd_up]                 (unbedingte k-Min-Basis)
#   dn[(N,k)] / up[(N,k)] = [faelle, wins_cont, sum_edge_signed]
base = {k: [0, 0] for k in K_LIST}
dn   = {}
up   = {}


def process_day(x):
    """x: 1-D np.array der Minuten-Log-Renditen EINES Handelstages.
    x[0] ist NaN (kein Overnight-Gap als Bewegung)."""
    L = len(x)
    if L < 30:
        return
    finite = ~np.isnan(x)
    down = np.zeros(L, bool); up_m = np.zeros(L, bool)
    down[finite] = x[finite] < 0
    up_m[finite] = x[finite] > 0

    # Streak-Laenge je Minute (nur aus Vergangenheit; reicht bis t)
    ds = np.zeros(L, int); us = np.zeros(L, int)
    for i in range(1, L):           # i=0 ist NaN -> Streak 0
        ds[i] = ds[i-1] + 1 if down[i] else 0
        us[i] = us[i-1] + 1 if up_m[i] else 0

    # Vorwaerts-Kumul-Logrendite: F_k[i] = x[i+1]+...+x[i+k]
    # x[0]=NaN kommt in keinem Vorwaertsfenster vor (Fenster startet bei i+1>=1)
    xf = np.where(np.isnan(x), 0.0, x)
    cs = np.cumsum(xf)              # cs[j] = sum x[0..j]
    for k in K_LIST:
        F = np.full(L, np.nan)
        hi = L - k                  # gueltig fuer i in 0..hi-1 (i+k <= L-1)
        if hi > 0:
            idx = np.arange(hi)
            F[idx] = cs[idx + k] - cs[idx]
        valid = ~np.isnan(F)
        fup = (F > 0)
        base[k][0] += int(valid.sum())
        base[k][1] += int(np.count_nonzero(fup & valid))

        for tbl, streak, cont, edge in (
            (dn, ds, ~fup, -F),     # Down-Streak: Continuation = F<0, Put-Edge = -F
            (up, us,  fup,  F),     # Up-Streak:   Continuation = F>0, Call-Edge = +F
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
    s = pd.Series(close.values, index=pd.to_datetime(idx)).sort_index()
    p = s.values.astype(float)
    L = len(p)
    if L < 2:
        return
    logr = np.full(L, np.nan)
    logr[1:] = np.log(p[1:] / p[:-1])
    day = s.index.normalize()                 # tz-konsistent (DatetimeIndex)
    boundary = np.empty(L, bool)
    boundary[0] = True
    boundary[1:] = day[1:].asi8 != day[:-1].asi8   # neuer Handelstag?
    logr[boundary] = np.nan                    # 1. Min/Tag = kein Overnight-Move
    starts = np.flatnonzero(boundary)
    ends = np.append(starts[1:], L)
    for a, b in zip(starts, ends):
        process_day(logr[a:b])


for i in range(0, len(tickers), BATCH):
    ch = tickers[i:i + BATCH]
    try:
        data = yf.download(ch, period="7d", interval="1m", progress=False,
                           group_by="ticker", threads=True)
    except Exception as ex:
        print(f"Batch {i}: {ex}", flush=True); continue
    for t in ch:
        try:
            c = data[t]["Close"].dropna()
            if len(c) > 100:
                process_symbol(c, c.index)
        except Exception:
            pass
    print(f"  {min(i + BATCH, len(tickers))}/{len(tickers)}", flush=True)


def show(tbl, titel, base_cont):
    """base_cont[k] = unbedingte P(Continuation) fuer diese Richtung+k."""
    print("\n" + "=" * 78)
    print(titel)
    print("=" * 78)
    head = "{:>4} | {:>3} | {:>9} | {:>11} | {:>12} | {:>4} | {:>6}".format(
        "N", "k", "Faelle", "P(cont)", "O Edge bps", "Sig", "Trade?")
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
print(f"\nGesamt: ~{len(tickers)} Aktien, 7 Tage, {tot:,} (N,k)-Beobachtungen")
print("\nUnbedingte Basisraten je Horizont k:")
base_up   = {}
base_down = {}
for k in K_LIST:
    nv, nu = base[k]
    bu = nu / nv if nv else float("nan")
    base_up[k] = bu
    base_down[k] = 1.0 - bu
    print(f"  k={k:>2}: P(k-Min-Kumul > 0) = {bu:.4f}  (n={nv:,})")

show(dn, "DOWN-Streak -> setzt sich der FALL fort? (Trade: Put)", base_down)
show(up, "UP-Streak -> setzt sich der ANSTIEG fort? (Trade: Call)", base_up)

print(f"\nKosten-Huerde COST_BPS = {COST_BPS} bps. 'Trade?'=JA nur, wenn")
print("O Edge bps > Huerde UND Binomialtest p<0.05 gegen Basisrate.")
print("Kein JA in der breiten EDA => Hypothese mit diesen Faktoren nicht")
print("handelbar (Plan-Abbruchkriterium) -> Stufe-2-Faktoren oder verwerfen.")
