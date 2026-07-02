#!/usr/bin/env python3
"""
crypto_updown_streaks.py — Streak-/Persistenz-Analyse der 15-Min-Up/Down-Ergebnisse
(bb_CryptoUpDown15m). Nutzerfrage 2026-07-02: "Wie oft in Folge war die gleiche
Richtung nacheinander?" — und die dahinterliegende Edge-Frage: Ist die Richtungs-
Serie seriell abhaengig (Momentum/Reversal ueber Ranges hinweg), und falls ja,
preist der Markt das am Start der Folge-Range bereits ein?

EXPLORATIV (keine Pre-Reg — Analyse auf bereits gesammelten Daten). Faellt hier
ein Signal auf, ist der saubere Weg: Regel einfrieren und auf den frischen Daten
03.–10.07. (Logger laeuft bis 10.07.) konfirmieren.

METHODIK
 A) Streak-Verteilung je Asset (nur lueckenlose 15-Min-Nachbarn verkettet;
    Logging-Luecken brechen die Serie). Zufalls-Baseline via PERMUTATIONSTEST
    (Shuffle der Ergebnisse bei fixer Segmentstruktur -> kontrolliert p(Up) und N
    exakt). Caveat: Shuffle zerstoert auch langsame Regime-Drift — lokale
    Trendphasen erscheinen als "Persistenz", obwohl bedingt aufs Regime nichts
    da ist; Schiedsrichter ist deshalb der oekonomische Test (D).
 B) Fortsetzungsrate P(X_t = X_{t-1}) gesamt und nach Vorgaenger-Streak-Laenge,
    Wilson-CI. Caveat Synchronitaet: die 5 Asset-Serien sind zu ~80 % dieselbe
    Serie (Crypto-Beta) — "5x bestaetigt" ist KEINE unabhaengige Replikation.
 C) Markt-Einpreisung: Up-Start-Quote (Mid des ersten In-Range-Ticks,
    secs_to_close >= 870) bedingt auf Vorgaenger-Ergebnis vs. realisierte P(Up).
 D) Netto-Sim: Momentum-Regel (kaufe Vorgaenger-Richtung am Range-Start) und
    Reversal-Regel (Gegenseite), zusaetzlich bedingt auf Streak >= 2 / >= 3;
    Kauf zum Ask + Fee (0.07*min(p,1-p)), halten bis Settlement (Claim frei).
    Signifikanz: Cluster-t auf 15-Min-Fenster (Assets synchron).
 E) Spot-Anker: AC(1) der 15-Min-Log-Returns aus price_to_beat (Binance-Open
    je Range) — die Up/Down-Serie ist das Vorzeichen dieser Returns.

BEFUND: siehe CRYPTO_UPDOWN_FINDINGS.md (Abschnitt Streak-Addendum).

Aufruf: python crypto_updown_streaks.py [--fee-rate 0.07] [--nperm 10000]
"""

import argparse
import sys

import numpy as np
import pandas as pd

from crypto_updown_backtest import FEE_RATE, cluster_t, fee, load_ticks, wilson_ci

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------- Aufbereitung

def build_events(df):
    """Je Event: result, Fenster, Start-Quoten (nur wenn wirklich am Range-Anfang
    erwischt, secs_to_close >= 870 — sonst waere die 'Start'-Quote Leakage aus dem
    Range-Verlauf), price_to_beat. Segment-IDs: Bruch bei Logging-Luecke (!=900s)."""
    ev = df.drop_duplicates("event_id")[["event_id", "asset", "window", "result"]]

    start = (df[df.secs_to_close <= 895].sort_values("ts_utc")
             .groupby("event_id")[["up_buy", "up_sell", "down_buy", "down_sell",
                                   "secs_to_close"]].first().reset_index())
    ev = ev.merge(start, on="event_id", how="left")
    ev["start_ok"] = ev.secs_to_close >= 870

    ptb = df.dropna(subset=["price_to_beat"]).groupby("event_id").price_to_beat.first()
    ev = ev.merge(ptb.rename("ptb"), on="event_id", how="left")

    ev = ev.sort_values(["asset", "window"]).reset_index(drop=True)
    gap = ev.groupby("asset").window.diff().dt.total_seconds()
    ev["seg"] = (gap != 900.0).cumsum()          # NaN am Asset-Anfang -> neues Segment

    # Vorgaenger (nur innerhalb Segment gueltig)
    ev["prev_result"] = ev.result.shift(1)
    ev["same_seg"] = ev.seg.eq(ev.seg.shift(1))

    # laufende Streak-Laenge je Position (innerhalb Segment)
    streaks = np.ones(len(ev), dtype=int)
    for i in range(1, len(ev)):
        if ev.same_seg.iat[i] and ev.result.iat[i] == ev.result.iat[i - 1]:
            streaks[i] = streaks[i - 1] + 1
    ev["streak"] = streaks
    ev["prev_streak"] = ev.streak.shift(1)
    return ev


def run_lengths(x, seg):
    """Run-Laengen (numpy): Bruch bei Wertwechsel ODER Segmentwechsel."""
    if len(x) == 0:
        return np.array([], dtype=int)
    brk = np.where((x[1:] != x[:-1]) | (seg[1:] != seg[:-1]))[0]
    return np.diff(np.concatenate([[-1], brk, [len(x) - 1]]))


# ---------------------------------------------------------------- A) Verteilung

def streak_report(ev, nperm, rng):
    print("\n" + "=" * 78)
    print("A) STREAK-VERTEILUNG (nur lueckenlose 15-Min-Ketten; Baseline = Shuffle)")
    print("=" * 78)
    print(f"\n{'asset':>6} {'N':>4} {'p(Up)':>6} | {'Runs':>4} {'~Shuffle':>8} | "
          f"{'O-Run':>5} | {'max':>3} {'~Shuffle':>8} | {'Fortsetzg.':>10} {'~Shuffle':>8} {'p-Wert':>7}")

    pooled_obs = np.zeros(30)
    pooled_exp = np.zeros(30)
    for asset, g in ev.groupby("asset"):
        x = (g.result == "Up").to_numpy().astype(np.int8)
        seg = g.seg.to_numpy()
        rl = run_lengths(x, seg)
        same = (seg[1:] == seg[:-1])
        t_obs = float(np.mean(x[1:][same] == x[:-1][same]))

        n_runs_p, max_p, cont_p = [], [], []
        for _ in range(nperm):
            xp = rng.permutation(x)
            rlp = run_lengths(xp, seg)
            n_runs_p.append(len(rlp))
            max_p.append(rlp.max())
            cont_p.append(np.mean(xp[1:][same] == xp[:-1][same]))
            pooled_exp += np.bincount(np.minimum(rlp, 29), minlength=30) / nperm
        cont_p = np.asarray(cont_p)
        # zweiseitiger Permutations-p fuer die Fortsetzungsrate
        pval = float(np.mean(np.abs(cont_p - cont_p.mean()) >= abs(t_obs - cont_p.mean())))
        pooled_obs += np.bincount(np.minimum(rl, 29), minlength=30)

        print(f"{asset:>6} {len(g):>4} {x.mean():>6.3f} | {len(rl):>4} {np.mean(n_runs_p):>8.1f} | "
              f"{rl.mean():>5.2f} | {rl.max():>3} {np.mean(max_p):>8.1f} | "
              f"{t_obs:>10.3f} {cont_p.mean():>8.3f} {pval:>7.3f}")

    print("\nStreak-Laengen gepoolt ueber 5 Assets (CAVEAT: ~80 % synchron = 1 Serie x5):")
    print(f"{'Laenge':>7} {'beobachtet':>11} {'~Shuffle':>9}")
    for L in range(1, 30):
        if pooled_obs[L] == 0 and pooled_exp[L] < 0.5:
            continue
        print(f"{L:>7} {int(pooled_obs[L]):>11} {pooled_exp[L]:>9.1f}")


# ---------------------------------------------------------------- B) Fortsetzung

def continuation_report(ev):
    print("\n" + "=" * 78)
    print("B) FORTSETZUNGSRATE nach Vorgaenger-Streak-Laenge (gepoolt, Wilson-95%-CI)")
    print("=" * 78)
    pairs = ev[ev.same_seg & ev.prev_result.notna()].copy()
    pairs["cont"] = (pairs.result == pairs.prev_result).astype(int)
    pairs["k"] = pairs.prev_streak.clip(upper=4).astype(int)   # 4 = "4+"
    print(f"\n{'Streak k':>9} {'N':>5} {'P(weiter)':>10}  Wilson-CI            je Asset (btc/eth/sol/doge/bnb)")
    for k, g in pairs.groupby("k"):
        lo, hi = wilson_ci(g.cont.sum(), len(g))
        per_asset = " ".join(f"{gg.cont.mean():.2f}" for _, gg in g.groupby("asset"))
        lab = f"{k}+" if k == 4 else f"{k}"
        print(f"{lab:>9} {len(g):>5} {g.cont.mean():>10.3f}  [{lo:.3f}, {hi:.3f}]   {per_asset}")
    lo, hi = wilson_ci(pairs.cont.sum(), len(pairs))
    print(f"{'gesamt':>9} {len(pairs):>5} {pairs.cont.mean():>10.3f}  [{lo:.3f}, {hi:.3f}]")
    print("(0,5 = Muenze; >0,5 Momentum, <0,5 Reversal. CI ignoriert Asset-Synchronitaet!)")
    return pairs


# ---------------------------------------------------------------- C) Einpreisung

def pricing_report(ev):
    print("\n" + "=" * 78)
    print("C) MARKT-EINPREISUNG: Up-Start-Quote vs. realisierte Up-Rate, bedingt auf Vorgaenger")
    print("=" * 78)
    d = ev[ev.same_seg & ev.prev_result.notna() & ev.start_ok &
           ev.up_buy.notna() & ev.up_sell.notna()].copy()
    d["up_mid0"] = (d.up_buy + d.up_sell) / 2
    d["up_won"] = (d.result == "Up").astype(int)
    print(f"\n{'Vorgaenger':>10} {'N':>5} {'Start-Mid(Up)':>13} {'real.P(Up)':>11} {'Diff':>7}")
    for prev, g in d.groupby("prev_result"):
        print(f"{prev:>10} {len(g):>5} {g.up_mid0.mean():>13.3f} {g.up_won.mean():>11.3f} "
              f"{g.up_won.mean() - g.up_mid0.mean():>+7.3f}")
    print("(Diff > 0: Up in dieser Gruppe UNTERbepreist -> dort saesse der Edge.)")


# ---------------------------------------------------------------- D) Netto-Sim

def sim_report(ev, rate):
    print("\n" + "=" * 78)
    print(f"D) SIM am Range-Start kaufen, bis Settlement halten (netto Fee {rate} | brutto)")
    print("   Cluster-t auf 15-Min-Fenster (Assets synchron -> effektives N = Fenster)")
    print("=" * 78)
    base = ev[ev.same_seg & ev.prev_result.notna() & ev.start_ok &
              ev.up_buy.notna() & ev.down_buy.notna()].copy()

    print(f"\n{'Regel':>22} {'N':>5} {'win':>6} {'O-Ask':>6} | "
          f"{'netto-PnL':>9} {'ROI':>7} {'t':>6} | {'brutto-PnL':>10} {'t':>6}")
    for mode in ("momentum", "reversal"):
        for k in (1, 2, 3):
            d = base[base.prev_streak >= k].copy()
            if mode == "momentum":
                d["side"] = d.prev_result
            else:
                d["side"] = np.where(d.prev_result == "Up", "Down", "Up")
            d["ask"] = np.where(d.side == "Up", d.up_buy, d.down_buy)
            d = d[(d.ask > 0.02) & (d.ask < 0.98)]
            d["won"] = (d.result == d.side).astype(int)
            d["cost_n"] = d.ask + d.ask.map(lambda a: fee(a, rate))
            d["pnl"] = d.won - d.cost_n
            d["pnl_b"] = d.won - d.ask
            t_n, nw = cluster_t(d[["window", "pnl"]])
            t_b, _ = cluster_t(d[["window", "pnl_b"]].rename(columns={"pnl_b": "pnl"}))
            roi = d.pnl.sum() / d.cost_n.sum() if len(d) else 0.0
            print(f"{mode + f' (Streak>={k})':>22} {len(d):>5} {d.won.mean():>6.3f} "
                  f"{d.ask.mean():>6.3f} | {d.pnl.sum():>+9.2f} {roi * 100:>+6.1f}% {t_n:>+6.2f} | "
                  f"{d.pnl_b.sum():>+10.2f} {t_b:>+6.2f}")
    print("(6 Regeln x2 Kosten-Sichten explorativ -> Bonferroni im Kopf behalten: |t|>~2.6)")


def sim_asym_report(ev, rate):
    """Die von Block C nahegelegte scharfe Variante: Momentum GETRENNT nach
    Vorgaenger-Richtung, plus Haelften-Split (Stabilitaets-Check — beide Haelften
    sind gesehen, das ist KEIN echtes OOS)."""
    print("\n" + "=" * 78)
    print("D2) ASYMMETRIE + STABILITAET: Momentum getrennt nach Vorgaenger-Richtung")
    print("=" * 78)
    base = ev[ev.same_seg & ev.prev_result.notna() & ev.start_ok &
              ev.up_buy.notna() & ev.down_buy.notna()].copy()
    cut = base.window.median()

    print(f"\n{'Regel':>24} {'N':>4} {'win':>6} {'O-Ask':>6} | {'netto-PnL':>9} "
          f"{'ROI':>7} {'t':>6} | {'t H1':>6} {'t H2':>6}")
    for label, prev, side in [("nach Up -> kaufe Up", "Up", "Up"),
                              ("nach Down -> kaufe Down", "Down", "Down")]:
        d = base[base.prev_result == prev].copy()
        d["ask"] = np.where(side == "Up", d.up_buy, d.down_buy)
        d = d[(d.ask > 0.02) & (d.ask < 0.98)]
        d["won"] = (d.result == side).astype(int)
        d["cost"] = d.ask + d.ask.map(lambda a: fee(a, rate))
        d["pnl"] = d.won - d.cost
        t_all, _ = cluster_t(d[["window", "pnl"]])
        t_h1, _ = cluster_t(d[d.window <= cut][["window", "pnl"]])
        t_h2, _ = cluster_t(d[d.window > cut][["window", "pnl"]])
        roi = d.pnl.sum() / d.cost.sum() if len(d) else 0.0
        print(f"{label:>24} {len(d):>4} {d.won.mean():>6.3f} {d.ask.mean():>6.3f} | "
              f"{d.pnl.sum():>+9.2f} {roi * 100:>+6.1f}% {t_all:>+6.2f} | "
              f"{t_h1:>+6.2f} {t_h2:>+6.2f}")


def binance_anchor(days=60):
    """Langzeit-Anker: Fortsetzungsrate von sign(close>=open) und AC(1) der
    15-Min-Kerzen-Returns direkt aus Binance-Klines. Beantwortet: ist das
    beobachtete 15m-Momentum ein stabiles Krypto-Phaenomen oder 2-Tage-Regime?"""
    import time

    import requests

    print("\n" + "=" * 78)
    print(f"F) BINANCE-ANKER ({days} Tage 15m-Klines): Fortsetzungsrate & AC(1)")
    print("   (erwartete Fortsetzung unter i.i.d. = p^2+(1-p)^2 mit p = Up-Rate)")
    print("=" * 78)
    symbol = {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT",
              "doge": "DOGEUSDT", "bnb": "BNBUSDT"}
    end = int(time.time() * 1000)
    start0 = end - days * 86400 * 1000
    print(f"\n{'asset':>6} {'n':>5} {'p(Up)':>6} | {'Fortsetzg.':>10} {'~iid':>6} "
          f"{'z':>6} | {'AC(1)':>7} {'2SE':>6} | {'Forts. letzte 2T':>16}")
    for asset, sym in symbol.items():
        kl, t = [], start0
        while t < end:
            r = requests.get("https://api.binance.com/api/v3/klines",
                             params={"symbol": sym, "interval": "15m",
                                     "startTime": t, "limit": 1000}, timeout=20)
            r.raise_for_status()
            chunk = r.json()
            if not chunk:
                break
            kl += chunk
            t = chunk[-1][6] + 1
            if len(chunk) < 1000:
                break
        o = np.array([float(k[1]) for k in kl])
        c = np.array([float(k[4]) for k in kl])
        up = (c >= o).astype(np.int8)
        n = len(up)
        cont = float(np.mean(up[1:] == up[:-1]))
        p = float(up.mean())
        iid = p * p + (1 - p) * (1 - p)
        z = (cont - iid) / np.sqrt(iid * (1 - iid) / (n - 1))
        lr = np.log(c / o)
        ac1 = float(np.corrcoef(lr[1:], lr[:-1])[0, 1])
        cont2d = float(np.mean(up[-192:][1:] == up[-192:][:-1]))
        print(f"{asset:>6} {n:>5} {p:>6.3f} | {cont:>10.3f} {iid:>6.3f} {z:>+6.2f} | "
              f"{ac1:>+7.3f} {2 / np.sqrt(n):>6.3f} | {cont2d:>16.3f}")


# ---------------------------------------------------------------- E) Spot-Anker

def spot_ac1(ev):
    print("\n" + "=" * 78)
    print("E) SPOT-ANKER: AC(1) der 15-Min-Log-Returns (price_to_beat-Serie, Binance)")
    print("=" * 78)
    for asset, g in ev.groupby("asset"):
        g = g.sort_values("window")
        r = np.log(g.ptb).diff()
        ok = g.same_seg & r.notna()
        pair = pd.DataFrame({"r": r[ok], "r_prev": r.shift(1)[ok],
                             "seg_ok": g.same_seg.shift(1)[ok]}).dropna()
        pair = pair[pair.seg_ok.astype(bool)]
        if len(pair) < 30:
            print(f"  {asset:>5}: zu wenig price_to_beat-Paare (n={len(pair)})")
            continue
        ac = float(np.corrcoef(pair.r, pair.r_prev)[0, 1])
        print(f"  {asset:>5}: AC(1) = {ac:+.3f}  (n={len(pair)}, ~2SE = {2 / np.sqrt(len(pair)):.3f})")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Streak-/Persistenz-Analyse bb_CryptoUpDown15m")
    ap.add_argument("--fee-rate", type=float, default=FEE_RATE)
    ap.add_argument("--nperm", type=int, default=10000)
    args = ap.parse_args()
    rng = np.random.default_rng(42)

    df = load_ticks()
    ev = build_events(df)
    n_pairs = int((ev.same_seg & ev.prev_result.notna()).sum())
    print(f"Events: {len(ev)} | verwertbare 15-Min-Nachbar-Paare: {n_pairs} | "
          f"Zeitraum {ev.window.min()} -> {ev.window.max()}")

    streak_report(ev, args.nperm, rng)
    continuation_report(ev)
    pricing_report(ev)
    sim_report(ev, args.fee_rate)
    sim_asym_report(ev, args.fee_rate)
    spot_ac1(ev)
    binance_anchor()


if __name__ == "__main__":
    main()
