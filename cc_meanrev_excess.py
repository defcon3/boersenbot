#!/usr/bin/env python3
"""
CC-MEAN-REVERSION ENTSCHEIDER-TEST (Plan "NAECHSTE AUFGABE" 2026-05-18).

Klon von cc_meanrev_validate.py mit den drei Korrekturen, die der
fruehere Lauf offen liess (Benchmark-Fehler + Schein-MaxDD):

1. EXCESS statt Rohrendite. Je Trade (Long Close t -> Close t+h):
     ret_Aktie(t->t+h) - ret_SPY(t->t+h)   (SPY 1d, auto_adjust, gleicher
     Kalenderzeitraum). Signifikanz wird gegen 0 DIESER Excess-Groesse
     gemessen (nicht gegen Roh-0). Zusaetzlich Kontroll-Spalte
     raw - unbedingte_h-Tages-Basis.

2. UEBERLAPPUNGS-KORREKTES Portfolio. Echte taeglich rebalancierte
   Overlap-Strategie (Jegadeesh-Titman-Stil): an Kalendertag d ist
   Kapital gleichverteilt auf ALLE an d offenen Positionen (Eintritt in
   den letzten h Tagen). Daraus echte Tages-Portfolio-Rendite und
   REALER MaxDD (nicht das Artefakt aus naivem h>1-Kompounding).
   Netto: da je (N,h) alle Positionen dasselbe h haben, faellt der
   Korb-Mittelwert exakt um (C/h) pro Aktiv-Tag -> exakte Kostenrechnung.

3. KORB-MINDESTGROESSE-GITTER. N in {4,5,6,7} x h in {2,3,5}; je Zelle
   Auswertung nur fuer Einstiegstage mit >= {1,5,8} Signal-Namen.
   Ziel: Zelle finden, die Alpha (vs SPY) UND Diversifikation zugleich
   hat.

Gates wie gehabt: OOS <=2021 / >=2022, Kosten {0,5,10,20} bps round-trip,
COVID-Ausschluss, Monatsblock-t. Erfolg = Excess-Alpha haelt OOS,
netto@10, nicht nur COVID, mit handelbarer Korbgroesse (>=~8 Namen).
"""
import warnings; warnings.filterwarnings("ignore")
import io, numpy as np, pandas as pd, yfinance as yf, requests

START  = "2015-01-01"
SPLIT  = pd.Timestamp("2022-01-01")
COVID_A, COVID_B = pd.Timestamp("2020-02-15"), pd.Timestamp("2020-04-30")
NS      = [4, 5, 6, 7]
HS      = [2, 3, 5]
COSTS   = [0.0, 5.0, 10.0, 20.0]            # bps round-trip
MINNMS  = [1, 5, 8]                          # Korb-Mindestgroesse-Schwellen


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
print(f"{len(tickers)} Ticker, Tages-OHLC ab {START}.", flush=True)

print("Lade SPY (Benchmark)...", flush=True)
spy_raw = yf.download("SPY", start=START, interval="1d", progress=False,
                      auto_adjust=True)["Close"]
spy = pd.Series(np.asarray(spy_raw, float).ravel(),
                index=pd.to_datetime(spy_raw.index).normalize()).dropna()
print(f"SPY: {len(spy)} Tage {spy.index.min().date()}..{spy.index.max().date()}",
      flush=True)

BATCH = 60
# A) per Einstiegstag EXCESS: pfx[(N,h)][date_t] = [sum_excess, sum_raw, count]
pfx = {(N, h): {} for N in NS for h in HS}
# B) reales taegl. Overlap-Portfolio: dly[(N,h)][date_d] = [sum_r1, count]
dly = {(N, h): {} for N in NS for h in HS}
# unbedingte Basis je h: base[h] = [n, sum]
base = {h: [0, 0.0] for h in HS}


def process(df, sym):
    c = df["Close"].astype(float).values.ravel()
    dts = pd.to_datetime(df.index).normalize()
    n = len(c)
    if n < 80: return
    # SPY auf dieselben Handelstage mappen
    sp = spy.reindex(dts).values
    cc = np.empty(n); cc[0] = np.nan; cc[1:] = c[1:] / c[:-1] - 1.0   # 1d-Ret
    down = cc < 0
    streak = np.zeros(n, int)              # Abwaerts-Tage in Folge BIS inkl. t
    for i in range(1, n):
        streak[i] = streak[i-1] + 1 if down[i] else 0
    for h in HS:
        fwd = np.full(n, np.nan)
        fwd[:n-h] = c[h:] / c[:n-h] - 1.0          # Stock Close_{t+h}/Close_t-1
        spf = np.full(n, np.nan)
        if np.isfinite(sp).any():
            spf[:n-h] = sp[h:] / sp[:n-h] - 1.0    # SPY gleicher Zeitraum
        ok0 = np.isfinite(fwd)
        for i in np.flatnonzero(ok0):
            base[h][0] += 1; base[h][1] += fwd[i]
        for N in NS:
            sel = ok0 & np.isfinite(spf) & (streak >= N)
            for i in np.flatnonzero(sel):
                dt = dts[i]
                ex = fwd[i] - spf[i]                # EXCESS vs SPY
                a = pfx[(N, h)].setdefault(dt, [0.0, 0.0, 0])
                a[0] += ex; a[1] += fwd[i]; a[2] += 1
                # B) Position lebt an Tagen i+1..i+h -> 1d-Returns dort
                for j in range(1, h + 1):
                    k = i + j
                    if k >= n or not np.isfinite(cc[k]): continue
                    b = dly[(N, h)].setdefault(dts[k], [0.0, 0])
                    b[0] += cc[k]; b[1] += 1


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
            if len(d) > 80: process(d, t)
        except Exception:
            pass
    print(f"  {min(i+BATCH,len(tickers))}/{len(tickers)}", flush=True)


print("\n" + "=" * 100)
print("UNBEDINGTE BASIS-Folgerendite (alle Aktie-Tage) -- Kontroll-Benchmark")
print("=" * 100)
basem = {}
for h in HS:
    nn, ss = base[h]
    basem[h] = ss / nn if nn else 0.0
    print(f"  h={h}: Ø {basem[h]*100:+.3f}%  ({basem[h]*1e4:+.1f} bps)  n={nn:,}")

print("\n" + "=" * 100)
print("(A) EXCESS vs SPY -- per Einstiegstag (ehrliches n=#Tage)")
print("Edge muss gegen 0 DER EXCESS-GROESSE signifikant sein.")
print("rawMB = Ø(raw - unbed. Basis_h)  als zweite Kontrolle.")
print("=" * 100)
for N in NS:
  for h in HS:
    items = sorted(pfx[(N, h)].items())
    if not items: continue
    allcnt = np.array([v[2] for _, v in items])
    print(f"\n[N>={N}, h={h}]  Einstiegs-Tage gesamt={len(items)}  "
          f"Median Korb={int(np.median(allcnt))}  max={int(allcnt.max())}")
    for mn in MINNMS:
        f = [(d, v) for d, v in items if v[2] >= mn]
        if not f:
            print(f"  minKorb>={mn}: keine Tage"); continue
        dts = pd.to_datetime([d for d, _ in f])
        ex  = np.array([v[0] / v[2] for _, v in f])          # Korb-Ø Excess/Tag
        rb  = np.array([v[1] / v[2] for _, v in f]) - basem[h]  # raw - Basis
        isb = dts < SPLIT
        med = int(np.median([v[2] for _, v in f]))
        print(f"  --- minKorb>={mn}: Tage={len(ex)}  MedianKorb={med} ---")
        for seg, m in (("IS ", isb), ("OOS", ~isb)):
            xs, ds = ex[m], dts[m]
            if len(xs) == 0:
                print(f"   {seg}: keine Tage"); continue
            mt, mnn = monthly_t(ds, xs)
            nocov = ~((ds >= COVID_A) & (ds <= COVID_B))
            xc = xs[nocov]
            print(f"   {seg}: Tage={len(xs):>4} | exØ {xs.mean()*100:+6.3f}% "
                  f"| t={tstat(xs):+5.2f} | MB-t={mt:+5.2f}(n={mnn}) "
                  f"| %+={ (xs>0).mean():4.0%} | rawMB Ø {rb[m].mean()*100:+.3f}%")
            print(f"        COVID-exkl: Tage={len(xc):>4} | exØ "
                  f"{xc.mean()*100:+6.3f}% | t={tstat(xc):+5.2f}")
            for cnet in COSTS:
                xn = xs - cnet / 1e4
                print(f"        netto@{cnet:>2.0f}bps: exØ {xn.mean()*100:+6.3f}%"
                      f" | t={tstat(xn):+5.2f} | %+={ (xn>0).mean():4.0%}")

print("\n" + "=" * 100)
print("(B) REALES taeglich rebalanciertes OVERLAP-Portfolio (Roh-Long).")
print("Kapital je Tag gleichverteilt auf alle offenen Positionen ->")
print("ECHTE Tagesserie, ECHTER MaxDD. Netto: -(C/h) je Aktiv-Tag (exakt).")
print("=" * 100)
for N in NS:
  for h in HS:
    it = sorted(dly[(N, h)].items())
    if not it: continue
    dd = pd.to_datetime([d for d, _ in it])
    rg = np.array([v[0] / v[1] for _, v in it])     # Roh-Tagesrendite Portfolio
    m  = np.array([v[1] for _, v in it])            # offene Positionen/Tag
    isb = dd < SPLIT
    print(f"\n[N>={N}, h={h}]  aktive Tage={len(rg)}  Median offen="
          f"{int(np.median(m))}  max offen={int(m.max())}")
    for seg, mask in (("IS ", isb), ("OOS", ~isb)):
        rs, ds = rg[mask], dd[mask]
        if len(rs) == 0:
            print(f"  {seg}: keine Tage"); continue
        print(f"  {seg}: Tage={len(rs):>4} | Ø/Tag {rs.mean()*1e4:+5.2f}bps "
              f"| t={tstat(rs):+5.2f} | annualisiert Ø "
              f"{rs.mean()*252*100:+5.1f}% | REAL MaxDD={maxdd(rs):5.1%} "
              f"| MedOffen={int(np.median(m[mask]))}")
        for cnet in COSTS:
            rn = rs - (cnet / 1e4) / h               # exakte Kostenamortisation
            print(f"      netto@{cnet:>2.0f}bps: Ø/Tag {rn.mean()*1e4:+5.2f}bps "
                  f"| t={tstat(rn):+5.2f} | ann {rn.mean()*252*100:+5.1f}% "
                  f"| MaxDD={maxdd(rn):5.1%}")
        nocov = ~((ds >= COVID_A) & (ds <= COVID_B))
        rc = rs[nocov]
        print(f"      COVID-exkl: Tage={len(rc):>4} | Ø/Tag "
              f"{rc.mean()*1e4:+5.2f}bps | t={tstat(rc):+5.2f} "
              f"| netto@10 t={tstat(rc-(10/1e4)/h):+5.2f}")

print("\nERFOLG = (A) Excess-Ø vs SPY > 0 UND OOS-signifikant UND netto@10")
print("UND nicht nur COVID UND bei handelbarer Korbgroesse (minKorb>=8),")
print("(B) realer MaxDD ertraeglich. Sonst: No-Overnight-Edge widerlegt /")
print("Alpha lebt nur in winzigen Koerben (undiversifizierbar).")
