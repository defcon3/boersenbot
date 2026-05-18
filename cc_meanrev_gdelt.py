#!/usr/bin/env python3
"""
DOWN-STREAK x NEWS-STIMMUNG (Option C, 2026-05-18).

Analog zu cc_meanrev_vix.py, aber Conditioner = MARKT-WEITE NEWS-
TAGESSTIMMUNG aus GDELT (gdelt_market_tone.csv, gebaut von
gdelt_market_tone_etl.py). Frage: Ist der (sauber gemessen tote)
Down-Streak-MR gut speziell dann, wenn die Markt-Nachrichtenlage
sehr negativ ist (Kapitulation IN DEN NEWS) — das News-Pendant zur
bereits falsifizierten VIX-Kapitulations-These?

LOOK-AHEAD-DISZIPLIN (zentral): Die Tone-Features werden auf einen
TAGES-Kalender ge-ffill-t und um 1 Tag GESHIFTET. Entscheidung am
Close t nutzt damit ausschliesslich News bis EINSCHLIESSLICH t-1 —
unzweideutig kein Look-ahead. Regime point-in-time (Trailing-Perzentil
nur aus Vergangenheit).

Conditioner-Regime (auf tone_all_mean; econ-Variante parallel):
  - pct  = Rang der Tagesstimmung in trailing 252d (min 60)
  - NEG  = pct <= 0.20  (sehr schlechte Nachrichtenlage / Angst)
  - POS  = pct >= 0.80
  - DROP = tone - Mittel(letzte 20d) <= -DROP_THR  (Stimmungs-Schock)
  - NEGE = econ-gefilterte Stimmung pct <= 0.20

Gate-Disziplin identisch zu cc_meanrev_vix.py: Excess vs SPY, per
Einstiegstag (ehrliches n), OOS-Split, Kosten {0,5,10,20}, COVID-excl,
Monatsblock-t, Korb-Mindestgröße {1,8}, #distinkte Monate + Top5-Tage-
Konzentration. Vor-registriertes Erfolgskriterium: NEG/DROP muss das
ALL-Subset OOS @minKorb>=8 materiell + signifikant + netto@10>0
schlagen, mit genug #Monaten. „Nur in Crash-Monaten" = Falsifikation.
Viele Zellen -> Multiple-Testing bewusst, nur dieses Kriterium zählt.

Lauf erst sinnvoll, wenn gdelt_market_tone.csv genug Historie hat;
Skript ist robust gegen Teil-Historie und weist Abdeckung aus.
"""
import warnings; warnings.filterwarnings("ignore")
import io, os, numpy as np, pandas as pd, yfinance as yf, requests

START   = "2015-01-01"
SPLIT   = pd.Timestamp("2022-01-01")
COVID_A, COVID_B = pd.Timestamp("2020-02-15"), pd.Timestamp("2020-04-30")
NS      = [4, 5, 6, 7]
HS      = [2, 3, 5]
COSTS   = [0.0, 5.0, 10.0, 20.0]
MINNMS  = [1, 8]
TONE_CSV = "/home/veit/boersenbot/gdelt_market_tone.csv"
PCT_NEG, PCT_POS = 0.20, 0.80
DROP_THR = 0.50
REGS    = ["ALL", "NEG", "POS", "DROP", "NEGE"]
REGS_B  = ["ALL", "NEG", "DROP"]


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
    r = np.asarray(r, float); tot = r.sum()
    if len(r) < 6 or abs(tot) < 1e-9: return float("nan")
    return float(np.sort(r)[-5:].sum() / tot)

def nmonths(dates):
    d = pd.to_datetime(dates)
    return len({(x.year, x.month) for x in d})


if not os.path.exists(TONE_CSV):
    raise SystemExit(f"FEHLT: {TONE_CSV} — erst gdelt_market_tone_etl.py laufen lassen.")
tn = pd.read_csv(TONE_CSV)
tn["date"] = pd.to_datetime(tn["date"], format="%Y%m%d")
tn = tn.sort_values("date").set_index("date")
print(f"GDELT-Tone-Historie: {len(tn)} Tage "
      f"{tn.index.min().date()}..{tn.index.max().date()}", flush=True)
# GUARD: Auf unvollstaendiger CSV entartet das Regime (ffill-Schwanz ->
# pct kuenstlich 1.0). Lieber laut abbrechen als still Unsinn rechnen.
_stale = (pd.Timestamp("today").normalize() - tn.index.max()).days
if _stale > 5:
    raise SystemExit(
        f"ABBRUCH: Tone-CSV endet {tn.index.max().date()} ({_stale} Tage "
        f"alt). ETL noch nicht fertig/aktuell -> erst gdelt_market_tone_"
        f"etl.py durchlaufen lassen (Backtest auf Teil-Historie = Müll).")

def pit_pct(s):
    return s.rolling(252, min_periods=60).apply(
        lambda w: float((w <= w[-1]).mean()), raw=True)

ta = tn["tone_all_mean"].astype(float)
te = pd.to_numeric(tn["tone_econ_mean"], errors="coerce")
feat = pd.DataFrame({
    "pct":  pit_pct(ta),
    "drop": ta - ta.rolling(20, min_periods=10).mean(),
    "pcte": pit_pct(te),
})
# Tages-Kalender, ffill, dann +1 Tag SHIFT -> Wert an Handelstag t = Stand t-1
cal = pd.date_range(feat.index.min(), pd.Timestamp("today").normalize())
feat_lag = feat.reindex(cal).ffill().shift(1)

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

print("Lade SPY ...", flush=True)
spy = dl_close("SPY")

BATCH = 60
pfx = {(N, h, r): {} for N in NS for h in HS for r in REGS}
dly = {(N, h, r): {} for N in NS for h in HS for r in REGS_B}
base = {h: [0, 0.0] for h in HS}
cov  = [0, 0]                                  # Signale gesamt / mit Regime


def process(df):
    c = df["Close"].astype(float).values.ravel()
    dts = pd.to_datetime(df.index).normalize()
    n = len(c)
    if n < 80: return
    sp = spy.reindex(dts).values
    fp = feat_lag.reindex(dts)
    vp = fp["pct"].values; vd = fp["drop"].values; vpe = fp["pcte"].values
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
                cov[0] += 1
                p = vp[i]
                if not np.isfinite(p):
                    continue                       # ohne Regime -> nicht raten
                cov[1] += 1
                dt = dts[i]; ex = fwd[i] - spf[i]
                regs = ["ALL"]
                if p <= PCT_NEG: regs.append("NEG")
                if p >= PCT_POS: regs.append("POS")
                if np.isfinite(vd[i]) and vd[i] <= -DROP_THR: regs.append("DROP")
                if np.isfinite(vpe[i]) and vpe[i] <= PCT_NEG: regs.append("NEGE")
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
print(f"ABDECKUNG: {cov[1]:,}/{cov[0]:,} Down-Streak-Signale mit gueltigem "
      f"News-Regime ({(cov[1]/max(cov[0],1)):.0%}). Rest vor GDELT-Start / "
      f"Warmup -> ausgeschlossen.")
print("KONTROLLE unbedingte Basis je h:")
basem = {}
for h in HS:
    nn, ss = base[h]; basem[h] = ss / nn if nn else 0.0
    print(f"  h={h}: Ø {basem[h]*100:+.3f}%  n={nn:,}")

print("\n" + "=" * 100)
print("(A) EXCESS vs SPY per Einstiegstag, je NEWS-REGIME (ehrliches n).")
print("ENTSCHEIDEND: schlaegt NEG/DROP das ALL-Subset OOS @minKorb>=8,")
print("netto@10>0, signifikant, mit GENUG #Monaten (nicht nur Crashs)?")
print("=" * 100)
for N in NS:
  for h in HS:
    head = False
    for mn in MINNMS:
      for rg in REGS:
        items = sorted(pfx[(N, h, rg)].items())
        items = [(d, v) for d, v in items if v[2] >= mn]
        if not items: continue
        if not head:
            print(f"\n===== [N>={N}, h={h}]  ALL-Tage={len(pfx[(N,h,'ALL')])} ====="); head = True
        dts = pd.to_datetime([d for d, _ in items])
        ex  = np.array([v[0] / v[2] for _, v in items])
        rb  = np.array([v[1] / v[2] for _, v in items]) - basem[h]
        isb = dts < SPLIT
        med = int(np.median([v[2] for _, v in items]))
        print(f" --- {rg:5s} minKorb>={mn} | Tage={len(ex)} MedKorb={med} "
              f"#Mon={nmonths(dts)} ---")
        for seg, m in (("IS ", isb), ("OOS", ~isb)):
            xs, ds = ex[m], dts[m]
            if len(xs) == 0:
                print(f"   {seg}: keine Tage"); continue
            mt, mnn = monthly_t(ds, xs)
            nocov = ~((ds >= COVID_A) & (ds <= COVID_B))
            xc = xs[nocov]; n10 = xs - 10 / 1e4
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
print("=" * 100)
for N in NS:
  for h in HS:
    for rg in REGS_B:
        it = sorted(dly[(N, h, rg)].items())
        if not it: continue
        dd = pd.to_datetime([d for d, _ in it])
        rgd = np.array([v[0] / v[1] for _, v in it])
        m  = np.array([v[1] for _, v in it]); isb = dd < SPLIT
        print(f"\n[N>={N}, h={h}, {rg}] Tage={len(rgd)} MedOffen={int(np.median(m))}")
        for seg, mask in (("IS ", isb), ("OOS", ~isb)):
            rs, ds = rgd[mask], dd[mask]
            if len(rs) == 0:
                print(f"  {seg}: keine Tage"); continue
            rn = rs - (10 / 1e4) / h
            print(f"  {seg}: Tage={len(rs):>4} Ø/Tag {rs.mean()*1e4:+5.2f}bps "
                  f"t={tstat(rs):+5.2f} ann {rs.mean()*252*100:+5.1f}% "
                  f"REAL-MaxDD={maxdd(rs):5.1%} | netto@10 Ø/Tag "
                  f"{rn.mean()*1e4:+5.2f}bps t={tstat(rn):+5.2f}")

print("\nLESEART: News-Kapitulations-These NUR bestaetigt, wenn NEG/DROP")
print("OOS @minKorb>=8 das ALL-Subset materiell + signifikant + netto@10>0")
print("schlaegt UND #Mon nicht winzig + Top5 nicht ~1.0. Sonst falsifiziert")
print("(wie schon VIX-Variante). Multiple-Testing bewusst.")
