"""
EXECUTION-OVERLAY — was spart der Hybrid-Bot durch Schlussauktions-Fills? (2026-06-23)

Folgt aus dem Volumen-U-Form-Befund: Close = tiefste Liquiditaet. Der Hybrid-Bot
(hybrid_simple.py) rebalanciert TAEGLICH zum Close (pos_size*close-zu-close-Rendite).
Frage: wie viel Slippage spart MOC (Close-Auktion) ggü. Mittags-/Open-Ausfuehrung,
und wie viel Tracking-Error entsteht, wenn man NICHT zum Close handelt?

Methodik:
  1. Hybrid pos_size + Turnover rekonstruieren (yfinance daily SPY+VIX, ab 2014).
  2. Liquiditaet je Tageszeit-Fenster aus SPY 30-Min SIP (spy_30min_sip_2016_2026.pkl).
  3. Square-Root-Impact-Modell: cost ~ 0.5*spread + eta*sigma*sqrt(Q/V_fenster).
     -> Ersparnis Close vs Mittag je Kontogroesse (% des ADV).
  4. Overnight-Gap auf realen Rebalance-Tagen = Tracking-Error bei Open- statt
     Close-Ausfuehrung.
"""
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import warnings; warnings.filterwarnings("ignore")

ETA = 0.5            # Impact-Koeffizient (typische Kalibrierung 0.3-1.0)
HALF_SPREAD_BPS = 0.1  # SPY ~1 Cent auf ~500$ = 0.2bp -> halber Spread 0.1bp


# ---------------- 1) Hybrid pos_size + Turnover ----------------
def hybrid_positions():
    spy = yf.download("SPY", start="2014-01-01", progress=False)
    vix = yf.download("^VIX", start="2014-01-01", progress=False)
    # SPY & VIX auf gemeinsame Handelstage alignen (unterschiedliche Laengen)
    j = pd.DataFrame({"close": spy["Close"].values.flatten(),
                      }, index=spy.index).join(
        pd.DataFrame({"vix": vix["Close"].values.flatten()}, index=vix.index),
        how="inner").dropna()
    close = j["close"].values
    vix_close = j["vix"].values
    dates = j.index
    ma50 = pd.Series(close).rolling(50).mean().values
    ma200 = pd.Series(close).rolling(200).mean().values
    uptrend = (ma50 > ma200).astype(float)
    vix_mean = np.mean(vix_close[:len(vix_close) // 2])
    vix_norm = np.clip((vix_close - vix_mean) / (vix_mean + 1) * 0.1, -0.5, 0.5)
    pos = uptrend * np.clip(1 - vix_norm, 0.2, 1.0)
    return pd.Series(pos, index=dates).dropna()


# ---------------- 2) Liquiditaet je Fenster aus 30-Min-Daten ----------------
def window_liquidity():
    with open("spy_30min_sip_2016_2026.pkl", "rb") as f:
        bars = pickle.load(f)
    df = pd.DataFrame(bars)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["et"] = df["t"].dt.tz_convert("America/New_York")
    df["date"] = df["et"].dt.date
    df["tod"] = df["et"].dt.strftime("%H:%M")
    # nur regulaere Session 09:30..15:30
    rth = [f"{h:02d}:{m:02d}" for h in range(9, 16) for m in (0, 30)
           if not (h == 9 and m == 0)]
    df = df[df["tod"].isin(rth)]
    # $-Volumen je Bar
    df["dollar"] = df["v"] * df["c"]
    # Tages-ADV ($)
    adv = df.groupby("date")["dollar"].sum().mean()
    # mittleres $-Volumen je 30-Min-Fenster
    per = df.groupby("tod")["dollar"].mean()
    return adv, per


# ---------------- 3) Impact-Modell ----------------
def impact_bps(Q_dollar, V_window_dollar, sigma_daily):
    """Square-Root-Impact in bps fuer Order Q im Fenster mit Volumen V."""
    if Q_dollar <= 0:
        return 0.0
    part = ETA * sigma_daily * np.sqrt(Q_dollar / V_window_dollar)
    return (HALF_SPREAD_BPS / 1e4 + part) * 1e4  # -> bps


def main():
    print("=" * 84)
    print("EXECUTION-OVERLAY: Ersparnis durch Schlussauktions-Fills (Hybrid-Bot)")
    print("=" * 84)

    pos = hybrid_positions()
    # taeglicher Turnover = |Aenderung der Zielgewichtung|
    turn = pos.diff().abs().dropna()
    years = (pos.index[-1] - pos.index[0]).days / 365.25
    ann_turnover = turn.sum() / years
    big = turn[turn > 0.3]
    print(f"\n--- 1) HYBRID-TURNOVER (2014-2026, {years:.1f} Jahre) ---")
    print(f"  Tage gesamt: {len(pos)}  |  Tage mit Handel (>0.01): {(turn>0.01).sum()}")
    print(f"  Jaehrlicher Turnover: {ann_turnover*100:.1f}% des Portfolios")
    print(f"  Grosse Rebalances (>30% an einem Tag, MA-Crossover): {len(big)}")
    print(f"  Mittlerer Tages-Turnover: {turn.mean()*100:.3f}%  median {turn.median()*100:.3f}%")

    adv, per = window_liquidity()
    sigma_daily = 0.0095  # SPY Tagesvol ~0.95%
    close_win = per["15:30"]      # tiefstes regulaeres Fenster + naehe Auktion
    open_win = per["09:30"]
    midday = per[["12:00", "12:30", "13:00"]].mean()
    print(f"\n--- 2) LIQUIDITAET je 30-Min-Fenster ($-Volumen, SPY 2016-2026) ---")
    print(f"  ADV gesamt: ${adv/1e9:.1f} Mrd")
    print(f"  Open 09:30:   ${open_win/1e6:,.0f} Mio  (Faktor {open_win/midday:.1f}x Mittag)")
    print(f"  Mittag 12-13: ${midday/1e6:,.0f} Mio")
    print(f"  Close 15:30:  ${close_win/1e6:,.0f} Mio  (Faktor {close_win/midday:.1f}x Mittag)")
    print(f"  Hinweis: Schlussauktion ist zusaetzlicher Einzel-Print (oft 3-7% des ADV)")

    print(f"\n--- 3) IMPACT & ERSPARNIS je Kontogroesse ---")
    print(f"  (taegl. Order = Konto x mittlerer Turnover {turn.mean()*100:.3f}%; "
          f"eta={ETA}, sigma={sigma_daily*100:.2f}%)")
    print(f"\n  {'Konto':>12} {'Order/Tag':>12} {'%ADV':>8} {'Close-Imp':>10} "
          f"{'Mittag-Imp':>11} {'Ersparnis':>10} {'€/Jahr':>10}")
    n_trade_days = (turn > 0.001).sum()
    for acct in [1e4, 1e5, 1e6, 1e7, 1e8]:
        q = acct * turn.mean()           # mittlere Order pro Tag in $
        pct_adv = q / adv * 100
        c_close = impact_bps(q, close_win, sigma_daily)
        c_mid = impact_bps(q, midday, sigma_daily)
        save_bps = c_mid - c_close
        # €/Jahr: Ersparnis bps auf das jaehrlich gehandelte Volumen
        ann_vol = acct * ann_turnover
        save_eur = save_bps / 1e4 * ann_vol
        print(f"  {acct:>12,.0f} {q:>12,.0f} {pct_adv:>7.4f}% {c_close:>9.3f}b "
              f"{c_mid:>10.3f}b {save_bps:>9.3f}b {save_eur:>9,.0f}€")

    print(f"\n--- 4) TRACKING-ERROR: Open- statt Close-Ausfuehrung ---")
    # Overnight-Gap = (open[t+1]-close[t])/close[t] auf allen Tagen
    spy = yf.download("SPY", start="2016-01-01", progress=False)
    o = np.asarray(spy["Open"].values).flatten()
    c = np.asarray(spy["Close"].values).flatten()
    gap = (o[1:] - c[:-1]) / c[:-1]
    print(f"  Overnight-Gap (Close->naechster Open), SPY 2016-2026:")
    print(f"    mean={gap.mean()*1e4:+.2f}bps  std={gap.std()*1e4:.1f}bps  "
          f"|gap|-median={np.median(np.abs(gap))*1e4:.1f}bps")
    print(f"  -> Wer den Close-Trade auf den naechsten Open verschiebt, handelt sich")
    print(f"     pro Rebalance ~{gap.std()*1e4:.0f}bps ZUFALLS-Tracking-Error ein")
    print(f"     (mean~0, aber reine Varianz ggü. der Backtest-Annahme Close-zu-Close).")

    print("\n" + "=" * 84)
    print("FAZIT")
    print("=" * 84)
    print("- IMPACT: SPY ist so liquide, dass fuer Privat-/kleine Konten die")
    print("  Auktions-Ersparnis SUB-Basispunkt ist (faktisch 0). Erst ab ~8-stelligem")
    print("  Volumen wird der Close-vs-Mittag-Unterschied spuerbar.")
    print("- DER ECHTE GRUND fuer MOC ist NICHT Impact, sondern Benchmark-Treue:")
    print("  Der Backtest unterstellt Close-zu-Close. MOC = 0 Tracking-Error.")
    print("  Open-Ausfuehrung = Overnight-Gap-Varianz (s.o.) als reines Zusatzrauschen.")
    print("- Empfehlung: MOC/LOC-Order beibehalten (falls Broker unterstuetzt),")
    print("  Eroeffnungsminuten meiden. Kein Geld 'gespart', aber Risiko/Tracking sauber.")


if __name__ == "__main__":
    main()
