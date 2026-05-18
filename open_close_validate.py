#!/usr/bin/env python3
"""
VALIDIERUNG der Down-Streak-Open->Close-Strategie.
Plan-Abschnitt "Spec Validierungsschritt" (2026-05-18).

Strategie: Tag t, wenn Vortags-CC-Abwaerts-Streak (Ende t-1) >= Nthr
-> long zum Open t, Exit zum Close t. PnL = Close/Open - 1 - Kosten.

Drei Pruefungen:
 1. OOS-Split: Train <= 2021-12-31, Test >= 2022-01-01.
 2. Kosten: netto bei round-trip 3 und 5 bps.
 3. Clustering: echte Einheit = TAGES-PORTFOLIO (Gleichgewicht-Korb
    aller Signal-Aktien je Tag), Statistik ueber #Tage statt Aktie-Tage.

Daten: Tages-OHLC ~503 S&P-500, gratis yfinance, ab 2015.
"""
import warnings; warnings.filterwarnings("ignore")
import io, numpy as np, pandas as pd, yfinance as yf, requests
from scipy import stats

START   = "2015-01-01"
SPLIT   = pd.Timestamp("2022-01-01")
NTHRS   = [2, 4, 5, 6]
COSTS   = [0.0, 3.0, 5.0]                 # bps round-trip
GAPKEY  = "GAP<-2%"                        # Zusatz-Variante

print("Lade S&P-500-Liste...", flush=True)
hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                    headers=hdr, timeout=20).text
tickers = [str(t).replace(".", "-")
           for t in pd.read_html(io.StringIO(html))[0]["Symbol"].tolist()]
print(f"{len(tickers)} Ticker. Lade Tages-OHLC ab {START}, Batches...", flush=True)

BATCH = 60
keys = [str(N) for N in NTHRS] + [GAPKEY]

# stock-day acc: sd[key][seg] = [n, sum, sumsq, npos]   seg in IS/OOS
def newacc(): return [0, 0.0, 0.0, 0]
sd = {k: {"IS": newacc(), "OOS": newacc()} for k in keys}
# tages-portfolio: pf[key][date] = [sum_intraday, count]
pf = {k: {} for k in keys}


def process(df):
    o = df["Open"].astype(float).values.ravel()
    c = df["Close"].astype(float).values.ravel()
    dates = pd.to_datetime(df.index).normalize()
    n = len(o)
    if n < 40:
        return
    pc = np.append([np.nan], c[:-1])
    intraday = c / o - 1.0
    cc = c / pc - 1.0
    gap = o / pc - 1.0
    down = cc < 0
    run = np.zeros(n, int)
    for i in range(1, n):
        run[i] = run[i-1] + 1 if down[i] else 0
    streak_prev = np.append([0], run[:-1])      # bekannt zum Open t

    for i in range(1, n):
        x = intraday[i]
        if not np.isfinite(x):
            continue
        dt = dates[i]
        seg = "OOS" if dt >= SPLIT else "IS"
        sig = []
        for N in NTHRS:
            if streak_prev[i] >= N:
                sig.append(str(N))
        if np.isfinite(gap[i]) and gap[i] < -0.02:
            sig.append(GAPKEY)
        for k in sig:
            a = sd[k][seg]
            a[0] += 1; a[1] += x; a[2] += x * x
            if x > 0: a[3] += 1
            b = pf[k].setdefault(dt, [0.0, 0])
            b[0] += x; b[1] += 1


for i in range(0, len(tickers), BATCH):
    ch = tickers[i:i + BATCH]
    try:
        data = yf.download(ch, start=START, interval="1d", progress=False,
                           group_by="ticker", threads=True, auto_adjust=True)
    except Exception as ex:
        print(f"Batch {i}: {ex}", flush=True); continue
    for t in ch:
        try:
            d = data[t][["Open", "Close"]].dropna()
            if len(d) > 60:
                process(d)
        except Exception:
            pass
    print(f"  {min(i + BATCH, len(tickers))}/{len(tickers)}", flush=True)


def line(label, n, mean, sd_, npos, cost):
    if n == 0:
        print(f"  {label:>22} | n=0"); return
    net = mean - cost / 1e4
    t = mean / (sd_ / np.sqrt(n)) if sd_ > 0 else float("nan")
    pv = stats.binomtest(int(npos), int(n), 0.5).pvalue
    st = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""
    print("  {:>22} | {:>8,} | brutto {:>+7.2f} | netto@{:>2.0f} {:>+7.2f} bps"
          " | t={:>+6.2f} | %+={:>6.2%} {:>3}"
          .format(label, n, mean * 1e4, cost, net * 1e4, t, npos / n, st))


print("\n" + "=" * 92)
print("1) + 2)  STOCK-DAY-EBENE  (naiv, OOS-Split, Kosten)  -- n = Aktie-Tage")
print("=" * 92)
for k in keys:
    print(f"\n[{k}{'+ down' if k != GAPKEY else ''}]")
    for seg in ("IS", "OOS"):
        n, s, sq, npos = sd[k][seg]
        m = s / n if n else 0.0
        v = (sq / n - m * m) if n else 0.0
        sdv = np.sqrt(max(v, 0.0))
        for cost in COSTS:
            line(f"{seg} cost={cost:.0f}", n, m, sdv, npos, cost)


def port_stats(dates_vals, label, cost):
    """dates_vals: list of (date, port_ret_gross). cost in bps."""
    if not dates_vals:
        print(f"  {label:>26} | keine Tage"); return
    dts = np.array([d for d, _ in dates_vals])
    r = np.array([v for _, v in dates_vals]) - cost / 1e4
    n = len(r); m = r.mean(); s = r.std(ddof=1) if n > 1 else 0.0
    t = m / (s / np.sqrt(n)) if s > 0 else float("nan")
    pos = (r > 0).mean()
    months = len({(pd.Timestamp(d).year, pd.Timestamp(d).month) for d in dts})
    tot = r.sum()
    sr = np.sort(r)
    conc = ""
    if abs(tot) > 1e-9 and n >= 10:
        top5 = sr[-5:].sum() / tot
        bot5 = sr[:5].sum() / tot
        conc = f" | Top5={top5:>5.0%} Bot5={bot5:>5.0%} v.Summe"
    print("  {:>26} | Tage={:>4} Mon={:>3} | Ø {:>+7.2f} bps | t={:>+6.2f}"
          " | %Tage+={:>6.2%}{}"
          .format(label, n, months, m * 1e4, t, pos, conc))


print("\n" + "=" * 92)
print("3)  TAGES-PORTFOLIO-EBENE  (ehrliches n = #Tage, Clustering-fest)")
print("=" * 92)
for k in keys:
    print(f"\n[{k}{'+ down' if k != GAPKEY else ''}]  Korb = Gleichgewicht aller "
          f"Signal-Aktien je Tag")
    items = sorted(pf[k].items())
    is_v  = [(d, v[0] / v[1]) for d, v in items if d < SPLIT and v[1] > 0]
    oos_v = [(d, v[0] / v[1]) for d, v in items if d >= SPLIT and v[1] > 0]
    for cost in COSTS:
        port_stats(is_v,  f"IS  cost={cost:.0f}", cost)
        port_stats(oos_v, f"OOS cost={cost:.0f}", cost)

print("\nLesart: Edge gilt nur als real, wenn er (a) OOS haelt, (b) netto")
print("nach Kosten >0 bleibt UND (c) auf TAGES-Portfolio-Ebene signifikant")
print("ist (nicht nur Aktie-Tag). Hohe Top5-Konzentration / wenige Monate")
print("= Edge haengt an wenigen Crash-Episoden -> fragil.")
