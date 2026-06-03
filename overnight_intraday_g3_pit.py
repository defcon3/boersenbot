#!/usr/bin/env python3
"""
G3 ROBUST: Point-in-Time-Index-Mitgliedschaft (Reviewer-Nachforderung 2026-06-03)

Ersetzt den schwachen G3 aus overnight_intraday_test.py (5 handverlesene
Survivor) durch einen echten Querschnittstest ueber die HISTORISCHE
S&P-500-Zusammensetzung:
  - Mitgliedschaft tagesgenau aus oeffentlichem Constituents-Changes-Dataset
    (fja05680/sp500), d.h. eine Aktie zaehlt nur an den Tagen, an denen sie
    TATSAECHLICH im Index war (keine Rueckschau-Selektion).
  - Zwei Auswertungen:
      (P) Gleichgewichtetes PIT-Mitglieder-Portfolio: Overnight vs Intraday
          kumuliert, gesamt + 2 Haelften.
      (S) Pro-Aktie ueber die eigenen Mitgliedschaftstage: Win-Rate on>id.

SURVIVORSHIP-EHRLICHKEIT: Spaeter entfernte/uebernommene Namen sind teils
bei yfinance nicht (mehr) abrufbar. Wir protokollieren die Coverage
(Anteil der Mitglieds-Slots mit echten Kursdaten). Das Ergebnis ist also
"so survivorship-frei wie Gratis-Daten es zulassen", nicht perfekt.

GATE G3-PIT (PASS, wenn beides):
  - Portfolio overnight>intraday kumuliert in BEIDEN Haelften
  - Pro-Aktie Win-Rate(on>id) > 50% (Mehrheit), membership-day-gewichtet konsistent
"""
import warnings; warnings.filterwarnings("ignore")
import csv, io, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf

START = pd.Timestamp("2010-01-01")
END = pd.Timestamp("2026-01-14")          # = letzter Membership-Snapshot
HERE = Path(__file__).parent
COMP_CSV = HERE / "sp500_hist_components.csv"
OHLC_CACHE = HERE / "g3_pit_ohlc.pkl"
MIN_MEMBER_DAYS = 60                        # Pro-Aktie-Test braucht Mindesthistorie
BATCH = 50


def load_membership():
    """-> sortierte Liste (Timestamp, frozenset(tickers)); nur clean Tickers."""
    if not COMP_CSV.exists():
        url = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
               "S%26P%20500%20Historical%20Components%20%26%20Changes(01-17-2026).csv")
        COMP_CSV.write_text(requests.get(url, timeout=30,
                            headers={"User-Agent": "Mozilla/5.0"}).text, encoding="utf-8")
    rows = list(csv.reader(io.StringIO(COMP_CSV.read_text(encoding="utf-8"))))[1:]
    snaps = []
    for d, tickers in rows:
        ts = pd.Timestamp(d)
        clean = frozenset(t.strip().replace(".", "-") for t in tickers.split(",")
                          if t.strip() and not t.strip().rsplit("-", 1)[-1].isdigit())
        snaps.append((ts, clean))
    snaps.sort(key=lambda x: x[0])
    return snaps


def fetch_ohlc(tickers):
    if OHLC_CACHE.exists():
        with open(OHLC_CACHE, "rb") as f:
            return pickle.load(f)
    out = {}
    tickers = sorted(tickers)
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i:i + BATCH]
        try:
            data = yf.download(" ".join(chunk), start=START, end=END + pd.Timedelta(days=2),
                               interval="1d", progress=False, group_by="ticker",
                               auto_adjust=True, threads=True)
        except Exception as e:
            print(f"  Batch {i//BATCH}: FAIL {e}", flush=True); continue
        for t in chunk:
            try:
                sub = data[t] if len(chunk) > 1 else data
                oc = sub[["Open", "Close"]].dropna()
                if len(oc) > MIN_MEMBER_DAYS:
                    out[t] = oc
            except Exception:
                pass
        print(f"  {min(i+BATCH,len(tickers))}/{len(tickers)} -> {len(out)} ok", flush=True)
    with open(OHLC_CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def decompose_oc(oc):
    o, c = oc["Open"].astype(float), oc["Close"].astype(float)
    pc = c.shift(1)
    return o / pc - 1.0, c / o - 1.0          # overnight, intraday


def cumr(s):
    s = s.dropna()
    return float(np.prod(1 + s) - 1)

def sharpe(s):
    s = np.asarray(s.dropna(), float)
    if len(s) < 2 or s.std(ddof=1) == 0: return np.nan
    return s.mean() / s.std(ddof=1) * np.sqrt(252)


# ===========================================================================
print("=" * 74)
print("G3 ROBUST — Point-in-Time S&P-500-Mitgliedschaft")
print("=" * 74)

print("\n[1/5] Lade historische Mitgliedschaft...", flush=True)
snaps = load_membership()
win_snaps = [(d, s) for d, s in snaps if d <= END]
union = sorted(set().union(*[s for d, s in win_snaps if d >= START - pd.Timedelta(days=400)]))
print(f"  {len(win_snaps)} Snapshots bis {END.date()}; Union-Ticker (clean): {len(union)}")

print("\n[2/5] Lade Tages-OHLC (auto_adjust, Cache wenn vorhanden)...", flush=True)
ohlc = fetch_ohlc(union)
fetched = sorted(ohlc.keys())
print(f"  abrufbar: {len(fetched)}/{len(union)} ({len(fetched)/len(union)*100:.0f}%)")

print("\n[3/5] Baue Return-Panels + PIT-Membership-Maske...", flush=True)
on_cols, id_cols = {}, {}
for t in fetched:
    on, idd = decompose_oc(ohlc[t])
    on_cols[t], id_cols[t] = on, idd
on_df = pd.DataFrame(on_cols).sort_index()
id_df = pd.DataFrame(id_cols).sort_index()
# auf Fenster + handelbare Tage beschneiden
mask_dates = (on_df.index >= START) & (on_df.index <= END)
on_df, id_df = on_df.loc[mask_dates], id_df.loc[mask_dates]
dates = on_df.index

# Glitch-Filter: |Tagesreturn| > 50% ist bei S&P-500-Large-Caps praktisch
# immer ein Daten-Fehler (Schrott-Print/Split-Luecke bei delisteten Tickern).
# Wenn eine Seite kaputt ist, ist die Gegenseite (gleicher o bzw. c) mit
# betroffen -> ganze Tag-Zelle verwerfen.
CLIP = 0.50
bad = (on_df.abs() > CLIP) | (id_df.abs() > CLIP)
n_cells = int(on_df.notna().values.sum())
n_bad = int(bad.values.sum())
on_df = on_df.mask(bad)
id_df = id_df.mask(bad)
print(f"  Glitch-Filter |r|>{CLIP:.0%}: {n_bad:,} von {n_cells:,} Zellen verworfen "
      f"({n_bad/n_cells*100:.3f}%)")

# is_member ueber ALLE Union-Ticker (fuer Coverage), bool-Matrix
is_member_all = pd.DataFrame(False, index=dates, columns=union)
snap_dates = [d for d, s in win_snaps]
for i, (d, S) in enumerate(win_snaps):
    d_next = win_snaps[i + 1][0] if i + 1 < len(win_snaps) else END + pd.Timedelta(days=1)
    rng = (dates >= d) & (dates < d_next)
    if not rng.any():
        continue
    cols = [t for t in S if t in is_member_all.columns]
    is_member_all.loc[rng, cols] = True

# Coverage: Anteil der Mitglieds-Slots mit echten Kursdaten
total_slots = int(is_member_all.values.sum())
covered_slots = int(is_member_all[fetched].values.sum())
print(f"  Mitglieds-Slots gesamt: {total_slots:,} | mit Kursdaten: {covered_slots:,} "
      f"({covered_slots/total_slots*100:.1f}%)")
print(f"  -> {total_slots-covered_slots:,} Slots fehlen (Delisting/yfinance-Luecke = Survivorship-Rest)")

is_member = is_member_all[fetched]

print("\n[4/5] (P) Gleichgewichtetes PIT-Mitglieder-Portfolio...", flush=True)
# nur Mitglieds-Zellen behalten, Tagesmittel ueber aktuelle Mitglieder
port_on = on_df.where(is_member).mean(axis=1)
port_id = id_df.where(is_member).mean(axis=1)
n_members = is_member.sum(axis=1)
port_on, port_id = port_on[n_members >= 20], port_id[n_members >= 20]   # robuste Korbgroesse

def report_seg(seg_on, seg_id, label):
    con, cid = cumr(seg_on), cumr(seg_id)
    print(f"  {label:22s} n={len(seg_on):5d} | Overnight {con*100:+9.1f}%  "
          f"Intraday {cid*100:+9.1f}%  | on>id: {'JA' if con > cid else 'NEIN'}")
    return con > cid

print(f"  mittlere Korbgroesse: {n_members[n_members>=20].mean():.0f} Aktien/Tag")
report_seg(port_on, port_id, "GESAMT")
mid = len(port_on) // 2
h1on, h1id = port_on.iloc[:mid], port_id.iloc[:mid]
h2on, h2id = port_on.iloc[mid:], port_id.iloc[mid:]
g_h1 = report_seg(h1on, h1id, f"H1 bis {port_on.index[mid].date()}")
g_h2 = report_seg(h2on, h2id, f"H2 ab {port_on.index[mid].date()}")
print(f"  Portfolio Sharpe: Overnight {sharpe(port_on):.2f}  Intraday {sharpe(port_id):.2f}")
portfolio_pass = g_h1 and g_h2

print("\n[5/5] (S) Pro-Aktie ueber eigene Mitgliedschaftstage...", flush=True)
wins = 0; tested = 0; wins_wd = 0; total_wd = 0
for t in fetched:
    mday = is_member[t]
    if mday.sum() < MIN_MEMBER_DAYS:
        continue
    on_t = on_df.loc[mday.values, t]
    id_t = id_df.loc[mday.values, t]
    if on_t.dropna().empty or id_t.dropna().empty:
        continue
    tested += 1
    md = int(mday.sum())
    total_wd += md
    if cumr(on_t) > cumr(id_t):
        wins += 1; wins_wd += md
print(f"  getestete Aktien (>= {MIN_MEMBER_DAYS} Mitgliedstage): {tested}")
print(f"  Win-Rate on>id (ungewichtet):          {wins}/{tested} = {wins/tested*100:.1f}%")
print(f"  Win-Rate on>id (membership-day-gewicht): {wins_wd/total_wd*100:.1f}%")
stock_pass = (wins / tested) > 0.5

print("\n" + "=" * 74)
print("GATE G3-PIT")
print("=" * 74)
print(f"  Portfolio on>id beide Haelften : {'PASS' if portfolio_pass else 'FAIL'}")
print(f"  Pro-Aktie Win-Rate > 50%       : {'PASS' if stock_pass else 'FAIL'} ({wins/tested*100:.1f}%)")
print(f"\n  G3-PIT GESAMT: {'PASS' if (portfolio_pass and stock_pass) else 'FAIL'}")
print(f"  (vs. alter handverlesener G3: 2/5 = FAIL)")

# Export fuer evtl. Web-Einbettung
import json
res = {
    "union": len(union), "fetched": len(fetched),
    "coverage_pct": round(covered_slots / total_slots * 100, 1),
    "missing_slots": total_slots - covered_slots,
    "basket_avg": int(n_members[n_members >= 20].mean()),
    "port_overnight_pct": round(cumr(port_on) * 100, 1),
    "port_intraday_pct": round(cumr(port_id) * 100, 1),
    "port_sharpe_on": round(sharpe(port_on), 2), "port_sharpe_id": round(sharpe(port_id), 2),
    "h1_on_pct": round(cumr(h1on) * 100, 1), "h1_id_pct": round(cumr(h1id) * 100, 1),
    "h2_on_pct": round(cumr(h2on) * 100, 1), "h2_id_pct": round(cumr(h2id) * 100, 1),
    "split_date": str(port_on.index[mid].date()),
    "h1_on_gt_id": bool(g_h1), "h2_on_gt_id": bool(g_h2),
    "stock_winrate_pct": round(wins / tested * 100, 1),
    "stock_winrate_wd_pct": round(wins_wd / total_wd * 100, 1),
    "stocks_tested": tested,
    "g3_pit_pass": bool(portfolio_pass and stock_pass),
}
(HERE / "overnight_intraday_g3_pit.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("\nGespeichert: overnight_intraday_g3_pit.json")
