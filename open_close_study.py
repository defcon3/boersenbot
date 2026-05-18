#!/usr/bin/env python3
"""
OPEN->CLOSE-STUDIE -- "morgens kaufen, abends verkaufen" (1 Pos/Tag,
kein Overnight). Plan-Abschnitt PIVOT 2026-05-18.

Frage: Ist Eroeffnung->Schluss im Schnitt profitabel -- unkonditioniert
und konditioniert auf Overnight-Gap / Vortags-Abwaerts-Streak /
SPY-Gap? Gibt es ein Regime ueber der Kosten-Huerde (~3-5 bps)?

Daten: Tages-OHLC ~503 S&P-500, gratis via yfinance, mehrere Jahre.
Strategie-PnL je Aktie/Tag = intraday = Close/Open - 1 (long).
Negativer Schnitt in einem Regime => dort waere SHORT (Open->Close)
profitabel, ebenfalls ohne Overnight.

Regeln: alles aus Vergangenheit bekannt zum Open (Gap, Streak, SPY-Gap),
keine Overnight-Position. kein Look-ahead.
"""
import warnings; warnings.filterwarnings("ignore")
import io, numpy as np, pandas as pd, yfinance as yf, requests
from scipy import stats

START = "2015-01-01"

print("Lade S&P-500-Liste...", flush=True)
hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                    headers=hdr, timeout=20).text
tickers = [str(t).replace(".", "-")
           for t in pd.read_html(io.StringIO(html))[0]["Symbol"].tolist()]

print(f"Lade SPY Tages-OHLC ab {START}...", flush=True)
spy = yf.download("SPY", start=START, interval="1d", progress=False,
                  auto_adjust=True)
spy_open = spy["Open"].astype(float).values.ravel()
spy_close = spy["Close"].astype(float).values.ravel()
spy_dates = pd.to_datetime(spy.index).normalize()
spy_pc = np.append([np.nan], spy_close[:-1])
spy_gap = spy_open / spy_pc - 1.0
spy_gap_by_date = pd.Series(spy_gap, index=spy_dates)
print(f"  SPY {len(spy_dates)} Tage. {len(tickers)} Ticker, Batches...",
      flush=True)

BATCH = 60

# Akkumulator-Helfer: [n, sum_x, sumsq_x, n_pos]   (x = intraday-Return)
def newacc():
    return [0, 0.0, 0.0, 0]

def add(a, x):
    a[0] += 1; a[1] += x; a[2] += x * x
    if x > 0:
        a[3] += 1

uncond   = newacc()           # intraday
uncond_on = newacc()          # overnight (prevClose->Open), Kontrast
GAP_BINS = ["<-2%", "-2..-0.5%", "-0.5..0.5%", "0.5..2%", ">2%"]
by_gap   = {b: newacc() for b in GAP_BINS}
by_gap_sign = {"Gap<0": newacc(), "Gap~0": newacc(), "Gap>0": newacc()}
by_streak = {N: newacc() for N in range(0, 11)}      # 10 = ">=10"
by_spygap = {"SPY<0": newacc(), "SPY~0": newacc(), "SPY>0": newacc()}
days_seen = set()


def gap_bin(g):
    if g < -0.02:  return "<-2%"
    if g < -0.005: return "-2..-0.5%"
    if g <=  0.005: return "-0.5..0.5%"
    if g <=  0.02: return "0.5..2%"
    return ">2%"


def process(df):
    o = df["Open"].astype(float).values.ravel()
    c = df["Close"].astype(float).values.ravel()
    dates = pd.to_datetime(df.index).normalize()
    n = len(o)
    if n < 30:
        return
    pc = np.append([np.nan], c[:-1])           # Vortagesschluss
    intraday = c / o - 1.0                      # Open->Close (Strategie)
    overnight = o / pc - 1.0                    # prevClose->Open
    gap = overnight.copy()                      # = Overnight-Gap
    cc = c / pc - 1.0                           # Close->Close (Vortagsmove)

    # Abwaerts-Streak Close-to-Close, bekannt ZUM heutigen Open:
    # Anzahl aufeinanderfolgender Down-CC-Tage, die GESTERN endeten.
    down = cc < 0
    run = np.zeros(n, int)
    for i in range(1, n):
        run[i] = run[i-1] + 1 if down[i] else 0
    streak_prev = np.append([0], run[:-1])      # Streak bis t-1 (Stand zum Open t)

    sg = spy_gap_by_date.reindex(dates).values.astype(float)

    for i in range(1, n):
        x = intraday[i]
        if not np.isfinite(x):
            continue
        days_seen.add(dates[i])
        add(uncond, x)
        if np.isfinite(overnight[i]):
            add(uncond_on, overnight[i])
        g = gap[i]
        if np.isfinite(g):
            add(by_gap[gap_bin(g)], x)
            key = "Gap<0" if g < -0.001 else ("Gap>0" if g > 0.001 else "Gap~0")
            add(by_gap_sign[key], x)
        N = min(int(streak_prev[i]), 10)
        add(by_streak[N], x)
        s = sg[i]
        if np.isfinite(s):
            k = "SPY<0" if s < -0.001 else ("SPY>0" if s > 0.001 else "SPY~0")
            add(by_spygap[k], x)


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


def stat(a, label, base_pos=None):
    n, sx, sxx, npos = a
    if n == 0:
        print(f"{label:>14} |  n=0"); return
    m = sx / n
    sd = np.sqrt(max(sxx / n - m * m, 0.0))
    t = m / (sd / np.sqrt(n)) if sd > 0 else float("nan")
    pos = npos / n
    bp = 0.5 if base_pos is None else base_pos
    pv = stats.binomtest(int(npos), int(n), bp).pvalue
    st = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""
    print("{:>14} | {:>9,} | {:>+9.2f} bps | t={:>+6.2f} | %+={:>6.2%} {:>3}"
          .format(label, n, m * 1e4, t, pos, st))


print("\n" + "=" * 78)
print("OPEN->CLOSE  (Strategie: morgens kaufen, abends verkaufen, long)")
print("=" * 78)
print(f"Distinkte Handelstage: {len(days_seen):,}   Zeitraum ab {START}")
print("\n--- ZERLEGUNG (unkonditioniert) ---")
stat(uncond,    "Intraday O->C")
stat(uncond_on, "Overnight pC->O")
basep = uncond[3] / uncond[0] if uncond[0] else 0.5

print("\n--- nach OVERNIGHT-GAP (Vorzeichen) ---  [Trade-Schwelle ~3-5 bps]")
for k in ("Gap<0", "Gap~0", "Gap>0"):
    stat(by_gap_sign[k], k, basep)
print("\n--- nach OVERNIGHT-GAP (Groesse) ---")
for b in GAP_BINS:
    stat(by_gap[b], b, basep)
print("\n--- nach VORTAGS-ABWAERTS-STREAK (Close-to-Close, validierter Edge) ---")
for N in range(0, 11):
    stat(by_streak[N], (">=10" if N == 10 else str(N)) + " down", basep)
print("\n--- nach SPY-OVERNIGHT-GAP (Index-Regime, Tag = Einheit!) ---")
for k in ("SPY<0", "SPY~0", "SPY>0"):
    stat(by_spygap[k], k, basep)

print("\nLesart: Ø bps = mittlerer Open->Close-Return je Tag (long).")
print("Positiv & |.|>Kosten => long handelbar; deutlich negativ => short")
print("handelbar (auch ohne Overnight). t = mean/SE, %+ = Trefferquote.")
print("WARNUNG Clustering: SPY-Gap & marktweite Gap-Tage sind je Tag")
print("korreliert -> effektive n ~ #Tage, nicht #Aktie-Tage. Aktien-")
print("eigener Gap/Streak querschnittlich breiter, aber Common-Factor.")
