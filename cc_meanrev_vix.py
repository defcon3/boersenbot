#!/usr/bin/env python3
"""
DOWN-STREAK x ANGST-CONDITIONER (Kapitulations-Hypothese), 2026-05-18.

Frage: Der Close-to-Close-Down-Streak-MR hat sauber gemessen KEIN
handelbares OOS-Alpha vs SPY (cc_meanrev_excess.py). Hypothese: Er
*konzentriert* sich vielleicht auf Einstiege, an denen gleichzeitig
markt-weite ANGST hoch ist (Kapitulations-Tief) — und ist in ruhigen
Phasen Rauschen. Wenn das stimmt, muss das FEAR/SPIKE-Subset das
ungefilterte ALL-Subset OOS, netto, bei handelbarem Korb DEUTLICH
schlagen — UND nicht nur aus 2–3 Crash-Episoden bestehen.

Conditioner (alle point-in-time, am Close t bekannt = Einstieg, KEIN
Look-ahead; freie lange Gratis-Historie via yfinance):
  - VIXPCT  = Rang von VIX_t in trailing 252d (min 60) -> Regime:
              CALM (<=0.40), FEAR (>=0.80), MID sonst
  - SPIKE   = VIX_t / Mittel(VIX letzte 20d) - 1 >= 0.15  (akuter Sprung)
  - BACKWARD= VIX_t > VIX3M_t  (Backwardation = akuter Stress)

Aufbau identisch zur Gate-Disziplin von cc_meanrev_excess.py:
(A) Excess je Trade ret_Aktie - ret_SPY, per Einstiegstag (ehrliches
    n=#Tage), je Regime; OOS-Split, Kosten {0,5,10,20}, COVID-excl,
    Monatsblock-t, Korb-Mindestgröße {1,8}, PLUS Clustering-Sichtbarkeit:
    #distinkte Monate + Top5-Tage-PnL-Konzentration (hier zentral, weil
    Fear-Conditioning Einstiege in Crash-Episoden ballt).
(B) Reales täglich rebalanciertes Overlap-Portfolio (realer MaxDD) je
    Regime in {ALL,FEAR,SPIKE}.

ERFOLG nur wenn FEAR (oder SPIKE) OOS bei minKorb>=8 das ALL-Subset
materiell schlägt, OOS-signifikant, netto@10>0, NICHT nur COVID, mit
nicht-winziger Monatszahl. „Edge nur in FEAR" + winzige #Monate =
„Edge nur in 2008/2020" = FALSIFIKATION, kein Erfolg. (Viele Zellen
getestet -> Multiple-Testing bewusst; nur dieses strenge Kriterium zählt.)
"""
import warnings; warnings.filterwarnings("ignore")
import io, numpy as np, pandas as pd, yfinance as yf, requests

START  = "2015-01-01"
SPLIT  = pd.Timestamp("2022-01-01")
COVID_A, COVID_B = pd.Timestamp("2020-02-15"), pd.Timestamp("2020-04-30")
NS      = [4, 5, 6, 7]
HS      = [2, 3, 5]
COSTS   = [0.0, 5.0, 10.0, 20.0]
MINNMS  = [1, 8]                              # undiversifiziert vs handelbar
REGS    = ["ALL", "CALM", "FEAR", "SPIKE", "BACKWARD"]
REGS_B  = ["ALL", "FEAR", "SPIKE"]            # fuer reales Overlap-Portfolio
PCT_CALM, PCT_FEAR = 0.40, 0.80
SPK_THR = 0.15


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

def conc(r):
    """Top5-Tage-Anteil an der Gesamtsumme (Clustering-Sichtbarkeit)."""
    r = np.asarray(r, float); tot = r.sum()
    if len(r) < 6 or abs(tot) < 1e-9: return float("nan")
    return float(np.sort(r)[-5:].sum() / tot)

def nmonths(dates):
    d = pd.to_datetime(dates)
    return len({(x.year, x.month) for x in d})


print("Lade S&P-500-Liste...", flush=True)
hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                    headers=hdr, timeout=20).text
tickers = [str(t).replace(".", "-")
           for t in pd.read_html(io.StringIO(html))[0]["Symbol"].tolist()]
print(f"{len(tickers)} Ticker, Tages-OHLC ab {START}.", flush=True)


def dl_close(sym):
    s = yf.download(sym, start=START, interval="1d", progress=False,
                    auto_adjust=False)["Close"]
    return pd.Series(np.asarray(s, float).ravel(),
                     index=pd.to_datetime(s.index).normalize()).dropna()

print("Lade SPY / ^VIX / ^VIX3M ...", flush=True)
spy   = dl_close("SPY")
vix   = dl_close("^VIX")
vix3m = dl_close("^VIX3M")
print(f"SPY {len(spy)} | VIX {len(vix)} ({vix.index.min().date()}.."
      f"{vix.index.max().date()}) | VIX3M {len(vix3m)}", flush=True)

# --- point-in-time Angst-Features auf dem VIX-Kalender ---
vix_pct = vix.rolling(252, min_periods=60).apply(
    lambda w: float((w <= w[-1]).mean()), raw=True)            # Rang heute
vix_ma20 = vix.rolling(20, min_periods=10).mean()
vix_spk = vix / vix_ma20 - 1.0                                 # vs eigener MA20
bwd = (vix.reindex(vix.index) > vix3m.reindex(vix.index)).astype(float)
bwd[vix3m.reindex(vix.index).isna()] = np.nan                  # ohne VIX3M = NA

feat = pd.DataFrame({"pct": vix_pct, "spk": vix_spk, "bwd": bwd})

BATCH = 60
# A) pfx[(N,h,reg)][date_t] = [sum_excess, sum_raw, count]
pfx = {(N, h, r): {} for N in NS for h in HS for r in REGS}
# B) dly[(N,h,reg)][date_d] = [sum_r1, count]
dly = {(N, h, r): {} for N in NS for h in HS for r in REGS_B}
base = {h: [0, 0.0] for h in HS}


def process(df):
    c = df["Close"].astype(float).values.ravel()
    dts = pd.to_datetime(df.index).normalize()
    n = len(c)
    if n < 80: return
    sp = spy.reindex(dts).values
    fp = feat.reindex(dts)
    vp = fp["pct"].values; vs = fp["spk"].values; vb = fp["bwd"].values
    cc = np.empty(n); cc[0] = np.nan; cc[1:] = c[1:] / c[:-1] - 1.0
    down = cc < 0
    streak = np.zeros(n, int)
    for i in range(1, n):
        streak[i] = streak[i-1] + 1 if down[i] else 0
    for h in HS:
        fwd = np.full(n, np.nan)
        fwd[:n-h] = c[h:] / c[:n-h] - 1.0
        spf = np.full(n, np.nan)
        if np.isfinite(sp).any():
            spf[:n-h] = sp[h:] / sp[:n-h] - 1.0
        ok0 = np.isfinite(fwd)
        for i in np.flatnonzero(ok0):
            base[h][0] += 1; base[h][1] += fwd[i]
        for N in NS:
            sel = ok0 & np.isfinite(spf) & (streak >= N)
            for i in np.flatnonzero(sel):
                dt = dts[i]; ex = fwd[i] - spf[i]
                p = vp[i]
                if not np.isfinite(p):       # ohne Regime -> nicht raten
                    continue
                regs = ["ALL"]
                if p <= PCT_CALM: regs.append("CALM")
                if p >= PCT_FEAR: regs.append("FEAR")
                if np.isfinite(vs[i]) and vs[i] >= SPK_THR: regs.append("SPIKE")
                if np.isfinite(vb[i]) and vb[i] >= 0.5:     regs.append("BACKWARD")
                for rg in regs:
                    a = pfx[(N, h, rg)].setdefault(dt, [0.0, 0.0, 0])
                    a[0] += ex; a[1] += fwd[i]; a[2] += 1
                    if rg in REGS_B:
                        for j in range(1, h + 1):
                            k = i + j
                            if k >= n or not np.isfinite(cc[k]): continue
                            b = dly[(N, h, rg)].setdefault(dts[k], [0.0, 0])
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
            if len(d) > 80: process(d)
        except Exception:
            pass
    print(f"  {min(i+BATCH,len(tickers))}/{len(tickers)}", flush=True)


print("\n" + "=" * 100)
print("KONTROLLE: unbedingte Basis-Folgerendite je h (alle Aktie-Tage)")
print("=" * 100)
basem = {}
for h in HS:
    nn, ss = base[h]; basem[h] = ss / nn if nn else 0.0
    print(f"  h={h}: Ø {basem[h]*100:+.3f}%  n={nn:,}")

print("\n" + "=" * 100)
print("(A) EXCESS vs SPY per Einstiegstag, je ANGST-REGIME (ehrliches n).")
print("ENTSCHEIDEND: schlaegt FEAR/SPIKE das ALL-Subset OOS bei minKorb>=8,")
print("netto@10>0, mit GENUG #Monaten (nicht nur Crash-Episoden)?")
print("=" * 100)
for N in NS:
  for h in HS:
    head_done = False
    for mn in MINNMS:
      for rg in REGS:
        items = sorted(pfx[(N, h, rg)].items())
        items = [(d, v) for d, v in items if v[2] >= mn]
        if not items: continue
        if not head_done:
            tot = len(pfx[(N, h, "ALL")])
            print(f"\n========== [N>={N}, h={h}]  ALL-Einstiegstage={tot} =========="); head_done = True
        dts = pd.to_datetime([d for d, _ in items])
        ex  = np.array([v[0] / v[2] for _, v in items])
        rb  = np.array([v[1] / v[2] for _, v in items]) - basem[h]
        isb = dts < SPLIT
        med = int(np.median([v[2] for _, v in items]))
        print(f" --- {rg:8s} minKorb>={mn} | Tage={len(ex)} MedKorb={med} "
              f"#Mon={nmonths(dts)} ---")
        for seg, m in (("IS ", isb), ("OOS", ~isb)):
            xs, ds = ex[m], dts[m]
            if len(xs) == 0:
                print(f"   {seg}: keine Tage"); continue
            mt, mnn = monthly_t(ds, xs)
            nocov = ~((ds >= COVID_A) & (ds <= COVID_B))
            xc = xs[nocov]
            n10 = xs - 10 / 1e4
            print(f"   {seg}: Tage={len(xs):>4} exØ {xs.mean()*100:+6.3f}% "
                  f"t={tstat(xs):+5.2f} MB-t={mt:+5.2f}(n={mnn}) "
                  f"%+={ (xs>0).mean():3.0%} #Mon={nmonths(ds)} "
                  f"Top5={conc(xs):+5.2f} | rawMB {rb[m].mean()*100:+.3f}%")
            print(f"        netto@10 exØ {n10.mean()*100:+6.3f}% "
                  f"t={tstat(n10):+5.2f} | COVID-excl exØ "
                  f"{xc.mean()*100:+6.3f}% t={tstat(xc):+5.2f}")
            for cnet in (5.0, 20.0):
                xn = xs - cnet / 1e4
                print(f"        netto@{cnet:>4.0f} exØ {xn.mean()*100:+6.3f}% "
                      f"t={tstat(xn):+5.2f}")

print("\n" + "=" * 100)
print("(B) REALES taegl. Overlap-Portfolio (Roh-Long) je Regime — REALER MaxDD")
print("Netto: -(C/h) je Aktiv-Tag (exakt, je (N,h,reg) gleiches h).")
print("=" * 100)
for N in NS:
  for h in HS:
    for rg in REGS_B:
        it = sorted(dly[(N, h, rg)].items())
        if not it: continue
        dd = pd.to_datetime([d for d, _ in it])
        rgd = np.array([v[0] / v[1] for _, v in it])
        m  = np.array([v[1] for _, v in it])
        isb = dd < SPLIT
        print(f"\n[N>={N}, h={h}, {rg}] Tage={len(rgd)} MedOffen={int(np.median(m))}")
        for seg, mask in (("IS ", isb), ("OOS", ~isb)):
            rs, ds = rgd[mask], dd[mask]
            if len(rs) == 0:
                print(f"  {seg}: keine Tage"); continue
            rn = rs - (10 / 1e4) / h
            print(f"  {seg}: Tage={len(rs):>4} Ø/Tag {rs.mean()*1e4:+5.2f}bps "
                  f"t={tstat(rs):+5.2f} ann {rs.mean()*252*100:+5.1f}% "
                  f"REAL-MaxDD={maxdd(rs):5.1%} | netto@10 Ø/Tag "
                  f"{rn.mean()*1e4:+5.2f}bps t={tstat(rn):+5.2f} "
                  f"MaxDD={maxdd(rn):5.1%}")

print("\nLESEART: Kapitulations-Hypothese NUR bestaetigt, wenn FEAR/SPIKE")
print("OOS @minKorb>=8 das ALL-Subset materiell + signifikant + netto@10>0")
print("schlaegt UND #Mon nicht winzig + Top5 nicht ~1.0 (sonst = nur")
print("Crash-Episoden = Falsifikation). Multiple-Testing bewusst.")
