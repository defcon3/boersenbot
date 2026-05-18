#!/usr/bin/env python3
"""
GAP-NACHFASS-STUDIE (Option 1) -- Plan-Abschnitt "Spec GAP-Nachfass".

A) SPY-ONLY: Tage mit SPY-Overnight-Gap <= Schwelle -> SPY zum Open
   kaufen, zum Close verkaufen (1 Instrument, kein Overnight, tiefste
   Auktions-Liquiditaet, keine Survivorship-/Basket-Probleme).
B) Aktien-Korb GAP<-2 % (sekundaer): roh + SPY-intraday-gehedged.

Pruefungen je Variante: OOS-Split (<=2021 / >=2022), Per-Jahr,
COVID-Fenster (2020-02-15..2020-04-30) exkludiert, Netto nach Kosten,
Max-Drawdown, Monatsblock-t (Autokorrelation daempfen).
"""
import warnings; warnings.filterwarnings("ignore")
import io, numpy as np, pandas as pd, yfinance as yf, requests

SPY_START   = "2000-01-01"
STK_START   = "2015-01-01"
SPLIT       = pd.Timestamp("2022-01-01")
COVID_A     = pd.Timestamp("2020-02-15")
COVID_B     = pd.Timestamp("2020-04-30")
SPY_THRS    = [-0.01, -0.02, -0.03]
COSTS       = [0.0, 1.0, 2.0, 3.0, 5.0]      # bps round-trip


def tstat(r):
    r = np.asarray(r, float)
    n = len(r)
    if n < 2:
        return float("nan")
    s = r.std(ddof=1)
    return r.mean() / (s / np.sqrt(n)) if s > 0 else float("nan")


def maxdd(r):
    if len(r) == 0:
        return 0.0
    eq = np.cumprod(1.0 + np.asarray(r, float))
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def monthly_block_t(dates, r):
    df = pd.DataFrame({"r": r}, index=pd.to_datetime(dates))
    m = df["r"].resample("ME").sum()
    m = m[m != 0]
    return tstat(m.values), len(m)


def report(name, dates, r):
    dates = pd.to_datetime(np.asarray(dates))
    r = np.asarray(r, float)
    order = np.argsort(dates.values)
    dates, r = dates[order], r[order]
    print("\n" + "-" * 86)
    print(f"### {name}   |  Signal-Tage gesamt: {len(r)}")
    if len(r) == 0:
        print("  keine Tage"); return
    mb_t, mb_n = monthly_block_t(dates, r)
    print(f"  Ø brutto {r.mean()*1e4:+.2f} bps | t={tstat(r):+.2f} | "
          f"%Tage+={ (r>0).mean():.1%} | MaxDD={maxdd(r):.1%} | "
          f"Monatsblock-t={mb_t:+.2f} (n={mb_n} Mon)")
    for c in COSTS:
        rn = r - c / 1e4
        print(f"    netto@{c:>2.0f}bps: Ø {rn.mean()*1e4:+6.2f} | "
              f"t={tstat(rn):+5.2f} | %+={ (rn>0).mean():.1%}")
    # OOS-Split
    isb = dates < SPLIT
    for seg, mask in (("IS  <=2021", isb), ("OOS >=2022", ~isb)):
        rs = r[mask]
        if len(rs):
            print(f"  [{seg}] n={len(rs):>4} | Ø {rs.mean()*1e4:+6.2f} bps | "
                  f"t={tstat(rs):+5.2f} | netto@3 t="
                  f"{tstat(rs-3/1e4):+5.2f} | %+={ (rs>0).mean():.1%}")
    # COVID exkludiert
    nocov = ~((dates >= COVID_A) & (dates <= COVID_B))
    rc = r[nocov]
    print(f"  [COVID exkl.] n={len(rc):>4} | Ø {rc.mean()*1e4:+6.2f} bps | "
          f"t={tstat(rc):+5.2f} | netto@3 t={tstat(rc-3/1e4):+5.2f}")
    # Per Jahr
    yr = pd.Series(r, index=dates).groupby(dates.year)
    cells = [f"{y}:{g.mean()*1e4:+.1f}({len(g)})" for y, g in yr]
    print("  Jahr Ø bps(n): " + "  ".join(cells))


print(f"Lade SPY Tages-OHLC ab {SPY_START}...", flush=True)
spy = yf.download("SPY", start=SPY_START, interval="1d", progress=False,
                  auto_adjust=True)
so = spy["Open"].astype(float).values.ravel()
sc = spy["Close"].astype(float).values.ravel()
sdt = pd.to_datetime(spy.index).normalize()
spc = np.append([np.nan], sc[:-1])
sgap = so / spc - 1.0
sintr = sc / so - 1.0
spy_intr_by_date = pd.Series(sintr, index=sdt)

print("=" * 86)
print("A) SPY-ONLY  -- SPY-Gap-Tag: Open kaufen, Close verkaufen")
print("=" * 86)
for thr in SPY_THRS:
    m = np.isfinite(sgap) & np.isfinite(sintr) & (sgap <= thr)
    report(f"SPY-Gap <= {thr*100:.0f}%", sdt[m], sintr[m])

print("\n\n" + "=" * 86)
print("B) AKTIEN-KORB GAP<-2 %  (sekundaer; SURVIVORSHIP-BIAS: aktuelle")
print("   S&P-500-Liste rueckwirkend -> Ergebnis eher zu optimistisch)")
print("=" * 86)
hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                    headers=hdr, timeout=20).text
tickers = [str(t).replace(".", "-")
           for t in pd.read_html(io.StringIO(html))[0]["Symbol"].tolist()]
print(f"{len(tickers)} Ticker, Tages-OHLC ab {STK_START}, Batches...", flush=True)

BATCH = 60
# basket[date] = [sum_intraday, count]
basket = {}
for i in range(0, len(tickers), BATCH):
    ch = tickers[i:i + BATCH]
    try:
        data = yf.download(ch, start=STK_START, interval="1d", progress=False,
                           group_by="ticker", threads=True, auto_adjust=True)
    except Exception as ex:
        print(f"Batch {i}: {ex}", flush=True); continue
    for t in ch:
        try:
            d = data[t][["Open", "Close"]].dropna()
            if len(d) < 60:
                continue
            o = d["Open"].astype(float).values.ravel()
            c = d["Close"].astype(float).values.ravel()
            dts = pd.to_datetime(d.index).normalize()
            pc = np.append([np.nan], c[:-1])
            gap = o / pc - 1.0
            intr = c / o - 1.0
            sel = np.isfinite(gap) & np.isfinite(intr) & (gap < -0.02)
            for j in np.flatnonzero(sel):
                b = basket.setdefault(dts[j], [0.0, 0])
                b[0] += intr[j]; b[1] += 1
        except Exception:
            pass
    print(f"  {min(i + BATCH, len(tickers))}/{len(tickers)}", flush=True)

items = sorted((d, v[0] / v[1], v[1]) for d, v in basket.items() if v[1] >= 5)
bd = pd.to_datetime([d for d, _, _ in items])
braw = np.array([m for _, m, _ in items])
sz = np.array([n for _, _, n in items])
shedge = spy_intr_by_date.reindex(bd).values.astype(float)
bhx = braw - shedge
print(f"\nKorb-Tage (>=5 Namen): {len(items)}, Median Korbgroesse "
      f"{int(np.median(sz)) if len(sz) else 0}")
report("Korb GAP<-2% ROH (long Korb)", bd, braw)
report("Korb GAP<-2% SPY-intraday-gehedged", bd, bhx)

print("\nLesart: Kandidat nur, wenn (a) OOS haelt, (b) netto>0 nach")
print("Kosten, (c) NICHT nur COVID-2020, (d) Monatsblock-t bleibt klar.")
print("Bricht der SPY-hedge die Korb-Variante -> war nur Markt-Beta")
print("(dann ist SPY-only die ehrliche, einzig relevante Variante).")
