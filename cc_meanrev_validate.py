#!/usr/bin/env python3
"""
CC-MEAN-REVERSION-VALIDIERUNG (Plan-Abschnitt AKTIV 2026-05-18).

No-Overnight-Regel gestrichen -> validierter Tages-Mean-Reversion-Edge
als handelbare Close-to-Close-Long-Strategie (Cash-Aktien, kein CFD).

Setup: Tag t mit N-Tage-CC-Abwaerts-Streak (inkl. Tag t, bekannt zum
Close t) -> long zum Close t, Exit Close t+h. PnL = Close_{t+h}/Close_t-1.
Gitter N in {3,4,5,6,7}, h in {1,2,3,5}.

Vier Gates: (1) TAGES-PORTFOLIO (Gleichgewicht-Korb je Einstiegstag,
ehrliches n=#Tage, Clustering-fest), (2) OOS <=2021 / >=2022,
(3) Kosten {0,5,10,20} bps round-trip, (4) COVID-Ausschluss +
Monatsblock-t + MaxDD; plus Stock-Day naiv als Kontrast und unbedingte
Basis-Folgerendite. h>1 = ueberlappende Positionen (Autokorr.) -> per
Einstiegstag, Monatsblock-t daempft, explizit gekennzeichnet.
"""
import warnings; warnings.filterwarnings("ignore")
import io, numpy as np, pandas as pd, yfinance as yf, requests

START  = "2015-01-01"
SPLIT  = pd.Timestamp("2022-01-01")
COVID_A, COVID_B = pd.Timestamp("2020-02-15"), pd.Timestamp("2020-04-30")
NS     = [3, 4, 5, 6, 7]
HS     = [1, 2, 3, 5]
COSTS  = [0.0, 5.0, 10.0, 20.0]            # bps round-trip


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


print("Lade S&P-500-Liste...", flush=True)
hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                    headers=hdr, timeout=20).text
tickers = [str(t).replace(".", "-")
           for t in pd.read_html(io.StringIO(html))[0]["Symbol"].tolist()]
print(f"{len(tickers)} Ticker, Tages-OHLC ab {START}, Batches...", flush=True)

BATCH = 60
# stock-day: sd[(N,h)][seg] = [n,sum,sumsq,npos]
sd = {(N, h): {"IS": [0,0.0,0.0,0], "OOS": [0,0.0,0.0,0]} for N in NS for h in HS}
# portfolio: pf[(N,h)][date_t] = [sum_fwd, count]
pf = {(N, h): {} for N in NS for h in HS}
# unbedingte Basis je h: base[h] = [n,sum]
base = {h: [0, 0.0] for h in HS}


def process(df):
    c = df["Close"].astype(float).values.ravel()
    dts = pd.to_datetime(df.index).normalize()
    n = len(c)
    if n < 60: return
    cc = np.empty(n); cc[0] = np.nan; cc[1:] = c[1:] / c[:-1] - 1.0
    down = cc < 0
    streak = np.zeros(n, int)              # Abwaerts-Tage in Folge BIS inkl. t
    for i in range(1, n):
        streak[i] = streak[i-1] + 1 if down[i] else 0
    for h in HS:
        fwd = np.full(n, np.nan)
        fwd[:n-h] = c[h:] / c[:n-h] - 1.0   # Close_{t+h}/Close_t - 1
        ok0 = np.isfinite(fwd)
        for i in np.flatnonzero(ok0):
            base[h][0] += 1; base[h][1] += fwd[i]
        for N in NS:
            sel = ok0 & (streak >= N)
            for i in np.flatnonzero(sel):
                x = fwd[i]; dt = dts[i]
                seg = "OOS" if dt >= SPLIT else "IS"
                a = sd[(N, h)][seg]
                a[0]+=1; a[1]+=x; a[2]+=x*x
                if x > 0: a[3]+=1
                b = pf[(N, h)].setdefault(dt, [0.0, 0])
                b[0]+=x; b[1]+=1


for i in range(0, len(tickers), BATCH):
    ch = tickers[i:i+BATCH]
    try:
        data = yf.download(ch, start=START, interval="1d", progress=False,
                           group_by="ticker", threads=True, auto_adjust=True)
    except Exception as ex:
        print(f"Batch {i}: {ex}", flush=True); continue
    for t in ch:
        try:
            d = data[t][["Close"]].dropna()
            if len(d) > 80: process(d)
        except Exception:
            pass
    print(f"  {min(i+BATCH,len(tickers))}/{len(tickers)}", flush=True)


print("\n" + "=" * 100)
print("UNBEDINGTE BASIS-Folgerendite (alle Aktie-Tage) -- zum Vergleich")
print("=" * 100)
for h in HS:
    n, s = base[h]
    print(f"  h={h}: Ø {s/n*100:+.3f}%  ({s/n*1e4:+.1f} bps)  n={n:,}")

print("\n" + "=" * 100)
print("STOCK-DAY-EBENE (naiv, korreliert -> nur Kontrast)  | Ø in %, t, %+")
print("=" * 100)
print("  {:>3} {:>2} | {:>9} | {:>8} {:>7} {:>6} | {:>8} {:>7} {:>6}".format(
    "N","h","","IS Ø%","IS t","%+","OOS Ø%","OOS t","%+"))
for N in NS:
    for h in HS:
        row=""
        for seg in ("IS","OOS"):
            nn,ss,sq,npx = sd[(N,h)][seg]
            if nn:
                m=ss/nn; v=max(sq/nn-m*m,0); t=m/(np.sqrt(v)/np.sqrt(nn)) if v>0 else float('nan')
                row+=" | {:>7.3f} {:>7.2f} {:>5.1%}".format(m*100,t,npx/nn)
            else:
                row+=" | {:>7} {:>7} {:>6}".format("-","-","-")
        print("  {:>3} {:>2} | n={:>7,}{}".format(N,h,sd[(N,h)]['IS'][0]+sd[(N,h)]['OOS'][0],row))

print("\n" + "=" * 100)
print("TAGES-PORTFOLIO (Gleichgewicht-Korb je Einstiegstag; EHRLICHES n=#Tage)")
print("h>1 = ueberlappende Positionen (Autokorr.) -> Monatsblock-t maßgeblich")
print("=" * 100)
for N in NS:
  for h in HS:
    items = sorted(pf[(N,h)].items())
    if not items: continue
    dts = pd.to_datetime([d for d,_ in items])
    r   = np.array([v[0]/v[1] for _,v in items])           # Korb-Ø je Tag (brutto)
    isb = dts < SPLIT
    print(f"\n[N>={N}, h={h}]  Einstiegs-Tage={len(r)}  Median Korb="
          f"{int(np.median([v[1] for _,v in items]))}")
    for seg,mask in (("IS ",isb),("OOS",~isb)):
        rs=r[mask]; ds=dts[mask]
        if len(rs)==0:
            print(f"  {seg}: keine Tage"); continue
        mt,mn = monthly_t(ds,rs)
        line=(f"  {seg}: Tage={len(rs):>4} | Ø {rs.mean()*100:+6.3f}% | "
              f"t={tstat(rs):+5.2f} | Monatsblock-t={mt:+5.2f}(n={mn}) | "
              f"%Tage+={ (rs>0).mean():4.0%} | MaxDD={maxdd(rs):4.0%}")
        print(line)
        for cnet in COSTS:
            rn=rs-cnet/1e4
            print(f"      netto@{cnet:>2.0f}bps: Ø {rn.mean()*100:+6.3f}% | "
                  f"t={tstat(rn):+5.2f} | %+={ (rn>0).mean():4.0%}")
        nocov = ~((ds>=COVID_A)&(ds<=COVID_B))
        rc=rs[nocov]
        print(f"      COVID-exkl.: Tage={len(rc):>4} | Ø {rc.mean()*100:+6.3f}%"
              f" | t={tstat(rc):+5.2f} | netto@10 t={tstat(rc-10/1e4):+5.2f}")

print("\nErfolg: Portfolio-Edge haelt OOS UND netto nach Kosten UND nicht")
print("nur COVID. Monatsblock-t ist bei h>1 das ehrlichste Maß. Stock-Day-")
print("Spalte NUR Kontrast (aufgeblasenes n durch Korrelation in Selloffs).")
