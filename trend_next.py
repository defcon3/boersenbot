#!/usr/bin/env python3
"""
LOCKERER Trend statt strikter Streak.

Frage (User, 2026-05-18): Wenn der Kurs in den letzten W Minuten
UEBERWIEGEND gestiegen ist -- wie wahrscheinlich steigt er in der
naechsten Minute?

Methode: pro Minute t zaehle, wie viele der letzten W 1-Min-Returns
positiv waren (u = 0..W). Bucketn nach u. Pro Bucket:
P(naechste Minute > 0), O naechste Rendite in bps, Fallzahl.
Steigt P mit u  -> Momentum. Faellt P mit u -> Reversion.

Regeln: kein Look-ahead (Fenster nur Vergangenheit), kein Overnight
(Fenster UND naechste Minute strikt im selben Handelstag). Eine
Minute mit Return == 0 zaehlt weder up noch down.
"""
import warnings; warnings.filterwarnings("ignore")
import io, numpy as np, pandas as pd, yfinance as yf, requests
from scipy import stats

WINDOWS = [5, 10, 20]

print("Lade S&P-500-Liste...", flush=True)
hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                    headers=hdr, timeout=20).text
tickers = [str(t).replace(".", "-")
           for t in pd.read_html(io.StringIO(html))[0]["Symbol"].tolist()]
print(f"{len(tickers)} Ticker. Lade 1-Min-Daten (7 Tage, Batches)...", flush=True)

BATCH = 60
# acc[W][u] = [faelle, wins_next_up, sum_next_ret]
acc = {W: {u: [0, 0, 0.0] for u in range(W + 1)} for W in WINDOWS}
base_n = 0
base_up = 0


def process_day(x):
    """x: Minuten-Log-Returns EINES Tages. x[0] = NaN (1. Min/Tag)."""
    global base_n, base_up
    L = len(x)
    if L < max(WINDOWS) + 2:
        return
    up = (x > 0).astype(float)          # NaN -> False -> 0
    up[np.isnan(x)] = np.nan
    nret = np.append(x[1:], np.nan)     # Return der naechsten Minute
    nx_up = (nret > 0).astype(float)
    valid_next = ~np.isnan(nret)        # naechste Minute existiert (selber Tag)

    # Basisrate (alle Minuten mit gueltiger Folgeminute, ab Minute 1)
    bvalid = valid_next & ~np.isnan(x)
    base_n += int(bvalid.sum())
    base_up += int(np.nansum(nx_up[bvalid]))

    for W in WINDOWS:
        if L < W + 2:
            continue
        # Anzahl Aufwaerts-Minuten in den letzten W Minuten (t-W+1 .. t),
        # nur aus Vergangenheit -> Vorhersage fuer t+1
        upc = pd.Series(up).rolling(W).sum().to_numpy()   # NaN wenn Fenster unvollst.
        ok = ~np.isnan(upc) & valid_next
        ui = np.where(ok, upc, -1).astype(int)
        for u in range(W + 1):
            m = ok & (ui == u)
            n = int(m.sum())
            if n == 0:
                continue
            b = acc[W][u]
            b[0] += n
            b[1] += int(np.nansum(nx_up[m]))
            b[2] += float(np.nansum(nret[m]))


def process_symbol(close, idx):
    s = pd.Series(np.asarray(close).ravel(), index=pd.to_datetime(idx)).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    p = s.values.astype(float)
    L = len(p)
    if L < 2:
        return
    logr = np.full(L, np.nan)
    logr[1:] = np.log(p[1:] / p[:-1])
    day = s.index.normalize()
    boundary = np.empty(L, bool)
    boundary[0] = True
    boundary[1:] = day[1:].asi8 != day[:-1].asi8
    logr[boundary] = np.nan
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


base = base_up / base_n
print(f"\nBasisrate P(naechste Minute steigt) = {base:.4f}  (n={base_n:,})")
print("Lies die Kurve: steigt P(next up) mit 'Auf-Min'  -> MOMENTUM,")
print("faellt sie -> REVERSION. O bps = mittlere Rendite der Folgeminute.\n")

for W in WINDOWS:
    print("=" * 70)
    print(f"FENSTER W = {W} Minuten   (Auf-Min = wie viele der letzten {W} stiegen)")
    print("=" * 70)
    print("{:>7} | {:>10} | {:>11} | {:>11} | {:>4}".format(
        "Auf-Min", "Faelle", "P(next up)", "O next bps", "Sig"))
    print("-" * 56)
    for u in range(W + 1):
        n, w, sr = acc[W][u]
        if n == 0:
            continue
        p = w / n
        avg = sr / n * 1e4
        pv = stats.binomtest(int(w), int(n), base).pvalue
        st = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""
        print("{:>3}/{:<3} | {:>10} | {:>11.4f} | {:>+11.3f} | {:>4}"
              .format(u, W, n, p, avg, st))
    print()

print("Hinweis: Folge-Renditen pro Minute sind winzig (~3-4 bps Ø-|Move|).")
print("Ein Edge ist nur real, wenn P(next up) deutlich UND O bps klar von")
print("der Basisrate weg zeigt -- und groesser als Kosten (~1-5 bps).")
