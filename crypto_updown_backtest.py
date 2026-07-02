#!/usr/bin/env python3
"""
crypto_updown_backtest.py — Backtest der "Underdog-Sinus"-These auf den
bb_CryptoUpDown15m-Ticks (Jupiter 15-Min Up/Down, btc/eth/sol/doge/bnb).

VORREGISTRIERTE HYPOTHESE (2026-06-30, vor Datensammlung): Der hintenliegende
(Underdog-)Kontrakt pendelt im Band ~0,20-0,40 ("Sinuskurve"). Ist das als
Mean-Reversion handelbar: kaufe niedrig (~0,20-0,30), verkaufe hoch (~0,35-0,45),
nach Spread + Fee positiv? Gegen-Einwände, die der Test klären muss:
(1) Klippe — Runde löst binär 0/1 auf, Ausbruch = Totalverlust;
(2) Preis = Wahrscheinlichkeit, "Sinus" nur Abbildung des Spot-Random-Walks;
(3) Spread + Fee fressen die enge Range.

METHODIK
 A) Kalibrierung (parameterfrei, Kern-Test von Einwand 2): realisierte
    Gewinnrate des Underdogs vs. Quoten-Mid je Preis-Bucket x Restzeit-Terzil.
    Dedup: je (Event, Bucket, Terzil) nur der erste Tick (gegen Autokorrelation).
    Ist der Preis kalibriert, existiert kein Free Lunch fuer simple Regeln.
 B) Trade-Sim-Grid: Entry = erster Tick mit Underdog-Ask <= X (Mid < 0,5),
    Exit = erster spaeterer Tick mit Bid >= Y, sonst Settlement 1/0.
    Kosten ehrlich: Kauf zum Ask + Fee, Verkauf zum Bid - Fee,
    Fee = FEE_RATE * min(p, 1-p) je Kontrakt und Seite (Audit-Modell autopilot.py,
    Default 0.07; Settlement-Claim ohne Fee). Ein Trade je Event.
    IS/OOS-Split nach 15-Min-Fenstern (erste Haelfte IS). Bonferroni ueber das
    Grid (12 Settings -> IS-Schwelle |t| > 2.87). Signifikanz via Cluster-t auf
    15-Min-Fenster-Ebene, weil die 5 Assets synchron aufloesen (Crypto-Beta,
    effektives N = Fenster, nicht Events).
 C) Deskription: Asset-Synchronitaet, Choppiness (Fuehrungswechsel je Range),
    Spot-Kopplung (Korrelation dMid ~ dSpot-Distanz; hoch = Einwand 2 bestaetigt).

BEFUND: siehe CRYPTO_UPDOWN_FINDINGS.md (wird nach dem Lauf committet — auch FAIL).

Aufruf:
  python crypto_updown_backtest.py                 # kompletter Report
  python crypto_updown_backtest.py --fee-rate 0.0  # Brutto (Spread only)
"""

import argparse
import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import pymssql
except ImportError:
    pymssql = None

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Hardcodierte Centron-Creds — bewusste Projektentscheidung (siehe CLAUDE.md).
DB_CONFIG = {
    "server": "158.181.48.77",
    "database": "dbdata",
    "user": "326773",
    "password": "Extaler11!",
}

FEE_RATE = 0.07          # Audit-Modell: fee = rate * Stueck * min(p, 1-p)
ENTRIES = [0.20, 0.25, 0.30, 0.33]
TARGETS = [0.35, 0.40, 0.45]
BONFERRONI_T = 2.87      # zweiseitig, alpha=0.05/12 Settings


# ---------------------------------------------------------------- Daten

def load_ticks():
    conn = pymssql.connect(**DB_CONFIG)
    df = pd.read_sql(
        """
        SELECT asset, event_id, range_start_utc, range_end_utc, ts_utc,
               secs_to_close, up_buy, up_sell, down_buy, down_sell,
               result, spot, price_to_beat
        FROM bb_CryptoUpDown15m
        WHERE settled = 1 AND result IN ('Up', 'Down')
          AND secs_to_close BETWEEN 0 AND 900
        ORDER BY event_id, ts_utc
        """,
        conn,
    )
    conn.close()
    # Degenerierte / kaputte Preiszeilen raus (nach Range-Ende meldet die API 1.000/1.000).
    px = ["up_buy", "up_sell", "down_buy", "down_sell"]
    df = df.dropna(subset=px)
    for c in px:
        df = df[(df[c] > 0.0005) & (df[c] < 0.9995)]
    df["up_mid"] = (df.up_buy + df.up_sell) / 2.0
    df["down_mid"] = (df.down_buy + df.down_sell) / 2.0
    # 15-Min-Fenster-ID: Range-Start (Assets im selben Fenster loesen synchron auf).
    df["window"] = df.range_start_utc
    return df.reset_index(drop=True)


def fee(p, rate):
    return rate * min(p, 1.0 - p)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ---------------------------------------------------------------- A) Kalibrierung

def calibration(df):
    print("\n" + "=" * 78)
    print("A) KALIBRIERUNG: Underdog-Mid vs. realisierte Gewinnrate")
    print("   (je Event x Bucket x Terzil nur der ERSTE Tick; kalibriert = kein Edge)")
    print("=" * 78)

    d = df.copy()
    # Underdog je Tick: Seite mit Mid < 0.5.
    is_down_dog = d.down_mid < d.up_mid
    d["dog_mid"] = np.where(is_down_dog, d.down_mid, d.up_mid)
    d["dog_side"] = np.where(is_down_dog, "Down", "Up")
    d = d[d.dog_mid < 0.5]
    d["dog_won"] = (d.dog_side == d.result).astype(int)
    d["bucket"] = (d.dog_mid / 0.05).astype(int) * 0.05          # 0.05er-Schritte
    d["terzil"] = pd.cut(d.secs_to_close, [0, 300, 600, 900],
                         labels=["spaet 0-5min", "mitte 5-10min", "frueh 10-15min"],
                         include_lowest=True)

    first = (d.sort_values("ts_utc")
              .groupby(["event_id", "bucket", "terzil"], observed=True)
              .first().reset_index())

    print(f"\n{'Bucket':>10} {'Terzil':>14} {'N':>5} {'Mid(avg)':>9} "
          f"{'real.Rate':>10} {'Diff':>7}  Wilson-95%-CI")
    for (b, t), g in first.groupby(["bucket", "terzil"], observed=True):
        n = len(g)
        if n < 30:
            continue
        k = g.dog_won.sum()
        rate, mid = k / n, g.dog_mid.mean()
        lo, hi = wilson_ci(k, n)
        flag = "  <-- Abweichung" if (mid < lo or mid > hi) else ""
        print(f"{b:>10.2f} {str(t):>14} {n:>5} {mid:>9.3f} {rate:>10.3f} "
              f"{rate - mid:>+7.3f}  [{lo:.3f}, {hi:.3f}]{flag}")

    # Gesamt-Edge (won - mid) mit Cluster-SE auf Fenster-Ebene ('window' steckt
    # schon in first, weil groupby().first() alle uebrigen Spalten mitnimmt).
    resid = first.assign(e=first.dog_won - first.dog_mid).groupby("window").e.mean()
    t = resid.mean() / (resid.std(ddof=1) / math.sqrt(len(resid)))
    print(f"\nGesamt (won - mid): {(first.dog_won - first.dog_mid).mean():+.4f}"
          f" | Cluster-t (n={len(resid)} Fenster): {t:+.2f}")
    print("-> |t| < 2 heisst: Underdog-Preise sind kalibriert, kein systematischer")
    print("   Rueckschwung-Bonus. Positiv+signifikant waere: Underdogs UNTERbepreist.")


# ---------------------------------------------------------------- B) Trade-Sim

def simulate(df, entry, target, rate):
    """Ein Trade je Event: kaufe Underdog-Ask<=entry, verkaufe Bid>=target, sonst 1/0."""
    trades = []
    for (eid, asset, window, result), g in df.groupby(
            ["event_id", "asset", "window", "result"], sort=False):
        g = g.sort_values("ts_utc")
        buy_i = side = None
        for i, r in enumerate(g.itertuples()):
            if r.up_mid < 0.5 and r.up_buy <= entry:
                side = "Up"
            elif r.down_mid < 0.5 and r.down_buy <= entry:
                side = "Down"
            else:
                continue
            buy_i = i
            break
        if buy_i is None:
            continue
        row = g.iloc[buy_i]
        ask = row.up_buy if side == "Up" else row.down_buy
        cost = ask + fee(ask, rate)
        exit_px, kind = None, None
        for r in g.iloc[buy_i + 1:].itertuples():
            bid = r.up_sell if side == "Up" else r.down_sell
            if bid >= target:
                exit_px, kind = bid - fee(bid, rate), "target"
                break
        if exit_px is None:
            exit_px, kind = (1.0, "cliff_win") if result == side else (0.0, "cliff_loss")
        trades.append({"event_id": eid, "asset": asset, "window": window,
                       "entry_secs": int(row.secs_to_close), "side": side,
                       "cost": cost, "exit": exit_px, "kind": kind,
                       "pnl": exit_px - cost})
    return pd.DataFrame(trades)


def cluster_t(trades):
    """t-Statistik auf 15-Min-Fenster-Summen (Assets synchron -> Cluster noetig)."""
    if trades.empty:
        return 0.0, 0
    w = trades.groupby("window").pnl.sum()
    n = len(w)
    if n < 3 or w.std(ddof=1) == 0:
        return 0.0, n
    return w.mean() / (w.std(ddof=1) / math.sqrt(n)), n


def sim_report(trades, label):
    n = len(trades)
    if n == 0:
        return f"{label}: keine Trades"
    pnl = trades.pnl.sum()
    t, nw = cluster_t(trades)
    kinds = trades.kind.value_counts()
    cliff = kinds.get("cliff_loss", 0)
    return (f"{label}: N={n:>4} | PnL {pnl:>+7.2f} $/Kontrakt | avg {pnl/n:>+7.4f} | "
            f"Cluster-t {t:>+5.2f} (n={nw} Fenster) | "
            f"target {kinds.get('target', 0):>3} / cliff_win {kinds.get('cliff_win', 0):>3} "
            f"/ CLIFF_LOSS {cliff:>3} ({cliff/n*100:.0f}%)")


def run_grid(df, rate):
    print("\n" + "=" * 78)
    print(f"B) TRADE-SIM-GRID (Fee-Rate {rate}) — kaufe Underdog-Ask<=X, verkaufe Bid>=Y")
    print(f"   IS = erste Haelfte der 15-Min-Fenster, OOS = zweite. "
          f"Bonferroni-IS-Schwelle |t| > {BONFERRONI_T}")
    print("=" * 78)

    windows = sorted(df.window.unique())
    cut = windows[len(windows) // 2]
    is_df, oos_df = df[df.window < cut], df[df.window >= cut]
    print(f"IS: {df[df.window < cut].event_id.nunique()} Events bis {cut} | "
          f"OOS: {oos_df.event_id.nunique()} Events danach")

    results = []
    for entry in ENTRIES:
        for target in TARGETS:
            tr_is = simulate(is_df, entry, target, rate)
            tr_oos = simulate(oos_df, entry, target, rate)
            t_is, _ = cluster_t(tr_is)
            t_oos, _ = cluster_t(tr_oos)
            results.append({"entry": entry, "target": target,
                            "is_n": len(tr_is), "is_pnl": tr_is.pnl.sum() if len(tr_is) else 0,
                            "is_t": t_is,
                            "oos_n": len(tr_oos), "oos_pnl": tr_oos.pnl.sum() if len(tr_oos) else 0,
                            "oos_t": t_oos,
                            "tr_is": tr_is, "tr_oos": tr_oos})

    print(f"\n{'entry':>6} {'target':>7} | {'IS-N':>5} {'IS-PnL':>8} {'IS-t':>6} | "
          f"{'OOS-N':>5} {'OOS-PnL':>8} {'OOS-t':>6} | Vorzeichen")
    for r in results:
        sign = "stabil" if (r["is_pnl"] > 0) == (r["oos_pnl"] > 0) else "DREHT"
        star = " *G1?*" if r["is_t"] > BONFERRONI_T else (
            " (sig. NEGATIV)" if r["is_t"] < -BONFERRONI_T else "")
        print(f"{r['entry']:>6.2f} {r['target']:>7.2f} | {r['is_n']:>5} "
              f"{r['is_pnl']:>+8.2f} {r['is_t']:>+6.2f} | {r['oos_n']:>5} "
              f"{r['oos_pnl']:>+8.2f} {r['oos_t']:>+6.2f} | {sign}{star}")

    best = max(results, key=lambda r: r["is_t"])
    print(f"\nBestes IS-Setting: entry {best['entry']} / target {best['target']} "
          f"(IS-t {best['is_t']:+.2f})")
    print("  IS : " + sim_report(best["tr_is"], "IS "))
    print("  OOS: " + sim_report(best["tr_oos"], "OOS"))

    print("\nPro-Asset-Konsistenz des besten Settings (IS+OOS zusammen):")
    all_tr = pd.concat([best["tr_is"], best["tr_oos"]], ignore_index=True)
    if not all_tr.empty:
        for asset, g in all_tr.groupby("asset"):
            print("  " + sim_report(g, f"{asset:>4}"))
    return results


def favorite_addendum(df, rate):
    """EXPLORATIV — NICHT vorregistriert (Test Nr. 13+, Bonferroni beachten!).

    Die Kalibrierung zeigt Underdogs leicht UEBERteuert (Longshot-Bias, am
    staerksten in den letzten 5 Minuten). Spiegeltest: Favorit spaet kaufen
    (erster Tick mit secs_to_close <= 300 und Favorit-Ask im Band), bis
    Settlement halten (keine Verkaufs-Fee, Claim kostenlos). Dient als
    Bias-Quantifizierung fuer eine ETWAIGE neue Pre-Reg — KEIN Trade-Signal.
    """
    print("\n" + "=" * 78)
    print(f"ADDENDUM (EXPLORATIV): Favorit spaet kaufen & halten (Fee-Rate {rate})")
    print("=" * 78)

    def sim_band(sub, lo, hi):
        trades = []
        for (eid, window, result), g in sub.groupby(
                ["event_id", "window", "result"], sort=False):
            g = g[g.secs_to_close <= 300].sort_values("ts_utc")
            for r in g.itertuples():
                side, ask = ("Up", r.up_buy) if r.up_mid > 0.5 else ("Down", r.down_buy)
                if not (lo <= ask <= hi):
                    continue
                cost = ask + fee(ask, rate)
                pay = 1.0 if result == side else 0.0
                trades.append({"window": window, "pnl": pay - cost,
                               "won": int(pay > 0), "cost": cost})
                break
        return pd.DataFrame(trades)

    windows = sorted(df.window.unique())
    cut = windows[len(windows) // 2]
    for lo, hi in [(0.55, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 0.97), (0.55, 0.97)]:
        parts = []
        for label, sub in [("IS ", df[df.window < cut]), ("OOS", df[df.window >= cut])]:
            tr = sim_band(sub, lo, hi)
            if tr.empty:
                parts.append(f"{label}: n=0")
                continue
            t, nw = cluster_t(tr)
            roi = tr.pnl.sum() / tr.cost.sum()
            parts.append(f"{label}: N={len(tr):>3} win {tr.won.mean():.3f} "
                         f"PnL {tr.pnl.sum():>+6.2f} ROI {roi*100:>+5.1f}% t {t:>+5.2f}")
        print(f"  Ask {lo:.2f}-{hi:.2f} | " + " | ".join(parts))
    print("  -> Interpretation: positiv+stabil in IS UND OOS noetig; selbst dann")
    print("     erst Pre-Reg + frische Daten (Logger laeuft weiter), dann Paper.")


# ---------------------------------------------------------------- C) Deskription

def synchrony(df):
    print("\n" + "=" * 78)
    print("C1) ASSET-SYNCHRONITAET (Crypto-Beta -> effektives N)")
    print("=" * 78)
    res = df.drop_duplicates("event_id")[["window", "asset", "result"]]
    piv = res.pivot_table(index="window", columns="asset", values="result",
                          aggfunc="first")
    full = piv.dropna()
    same = (full.nunique(axis=1) == 1).mean()
    up_rate = (res.result == "Up").mean()
    # mittlere paarweise Uebereinstimmung
    pairs, agree = 0, 0
    cols = list(full.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs += len(full)
            agree += (full[cols[i]] == full[cols[j]]).sum()
    print(f"Fenster mit allen 5 Assets: {len(full)} | Up-Rate gesamt: {up_rate:.3f}")
    print(f"Alle 5 Assets gleiches Ergebnis: {same*100:.1f}%  (unabhaengig waeren ~6.25%)")
    print(f"Paarweise Uebereinstimmung: {agree/pairs*100:.1f}%  (unabhaengig ~50%)")
    print(f"-> Events: {df.event_id.nunique()} | unabhaengige Wetten eher ~{len(full)} Fenster")


def choppiness(df):
    print("\n" + "=" * 78)
    print("C2) CHOPPINESS: Fuehrungswechsel je Range (Sinus-Optik vs. frueh entschieden)")
    print("=" * 78)
    stats = []
    for eid, g in df.groupby("event_id"):
        g = g.sort_values("ts_utc")
        lead = np.sign(g.up_mid - 0.5)
        lead = lead[lead != 0]
        changes = int((lead.diff().fillna(0) != 0).sum())
        stats.append(changes)
    s = pd.Series(stats)
    print(f"Ranges: {len(s)} | Fuehrungswechsel median {s.median():.0f}, "
          f"mean {s.mean():.1f}, max {s.max()}")
    for lo, hi, lab in [(0, 0, "0 (nie gewechselt, frueh entschieden)"),
                        (1, 2, "1-2"), (3, 6, "3-6 (choppy)"), (7, 999, ">=7 (sehr choppy)")]:
        m = ((s >= lo) & (s <= hi)).mean()
        print(f"  {lab:>38}: {m*100:5.1f}%")


def spot_coupling(df):
    print("\n" + "=" * 78)
    print("C3) SPOT-KOPPLUNG: korreliert dUp-Mid mit d(Spot - price_to_beat)?")
    print("    (hoch = Quote bildet nur den Spot-Random-Walk ab -> Einwand 2)")
    print("=" * 78)
    d = df.dropna(subset=["spot", "price_to_beat"])
    for asset, g in d.groupby("asset"):
        cors = []
        for eid, e in g.groupby("event_id"):
            e = e.sort_values("ts_utc")
            if len(e) < 20:
                continue
            dm = e.up_mid.diff().dropna()
            ds = (e.spot - e.price_to_beat).diff().dropna()
            if dm.std() == 0 or ds.std() == 0:
                continue
            cors.append(np.corrcoef(dm, ds)[0, 1])
        if cors:
            print(f"  {asset:>5}: mittlere Tick-Korrelation dMid~dSpotDist "
                  f"{np.mean(cors):+.3f} (n={len(cors)} Ranges)")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Backtest Underdog-Sinus-These (bb_CryptoUpDown15m)")
    ap.add_argument("--fee-rate", type=float, default=FEE_RATE,
                    help=f"Fee je Seite: rate*min(p,1-p) (default {FEE_RATE}; 0 = brutto)")
    args = ap.parse_args()

    df = load_ticks()
    print(f"Geladen: {len(df)} valide In-Range-Ticks | {df.event_id.nunique()} gesettlete "
          f"Events | {df.ts_utc.min()} -> {df.ts_utc.max()}")

    calibration(df)
    run_grid(df, args.fee_rate)
    favorite_addendum(df, args.fee_rate)
    synchrony(df)
    choppiness(df)
    spot_coupling(df)


if __name__ == "__main__":
    main()
