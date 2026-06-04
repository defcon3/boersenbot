#!/usr/bin/env python3
"""
OVERNIGHT-vs-INTRADAY — ROLLING-STABILITAET + BOOTSTRAP + STRUKTURBRUCH
(Reviewer-Nachforderung 2026-06-04, Grok-Review)

WARUM:
  Der bisherige G1 (overnight_intraday_test.py) zeigt Stabilitaet ueber EINEN
  Median-Split in 2 Haelften. Das ist ein schwacher Stabilitaetsnachweis:
    - ein einziger Split-Zeitpunkt,
    - kein Konfidenzintervall (Punktschaetzung der kumulierten Differenz),
    - keine Aussage, OB/WANN sich das Regime aendert.
  Dieses Skript ersetzt den Median-Split-Beleg durch drei robustere Belege,
  alle auf dem TAGES-Unterschied d_t = overnight_t - intraday_t (SPY):

  (A) ROLLING: 3- und 5-Jahres-Fenster. Anteil der Fenster mit on>id
      (kumuliert), plus annualisierter Mittelwert von d_t je Fenster.
      -> zeigt Stabilitaet als Verteilung statt als 1 Zahl.

  (B) BOOTSTRAP: Stationary Bootstrap (Politis-Romano 1994), erhaelt die
      Serien-/Querkorrelation. 95%-KI + zweiseitiger p-Wert fuer
        - annualisierten Mittelwert von d_t,
        - Sharpe(overnight) - Sharpe(intraday).
      Gepaartes Resampling (gleiche Block-Indizes fuer on & id) -> die
      Korrelation zwischen den Buckets bleibt erhalten.

  (C) STRUKTURBRUCH: sup-Wald / QLR-Test (Quandt-Andrews) fuer EINEN
      unbekannten Bruch im Mittel von d_t. Kritische Werte per Block-
      Bootstrap unter H0 (kein Bruch) -> Autokorrelation korrekt behandelt,
      keine iid-Annahme. Liefert geschaetztes Bruchdatum + p-Wert.

LESART (wichtig, KEIN Gate-Hacking):
  Das ist ein STABILITAETS-NACHWEIS fuer den DESKRIPTIVEN Befund (G1), nicht
  fuer Handelbarkeit. G2 (Kosten) bleibt unberuehrt. Ein signifikanter
  Strukturbruch WIDERLEGT G1 nicht automatisch — er praezisiert nur, dass das
  Niveau der Overnight-Dominanz nicht konstant ist (was die Literatur, z.B.
  Tug-of-War zwischen Clienteles, ohnehin erwartet).
"""
import warnings; warnings.filterwarnings("ignore")
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

PRIMARY = "SPY"
START = "1999-01-01"
END = "2026-06-30"
HERE = Path(__file__).parent
CACHE = HERE / "spy_overnight_ohlc.pkl"

N_BOOT = 2000           # Bootstrap-Replikate fuer KI/p-Werte
N_BOOT_BREAK = 1000     # Bootstrap-Replikate fuer QLR-Kritikwerte (teurer)
EXP_BLOCK = 21          # erw. Blocklaenge ~ 1 Handelsmonat (Serienkorrelation)
TRIM = 0.15             # Randtrimmung fuer Strukturbruch-Suche
SEED = 20260604
RNG = np.random.default_rng(SEED)
TDAYS = 252


# ---------------------------------------------------------------------------
# DATEN
# ---------------------------------------------------------------------------
def fetch_spy():
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    d = yf.download(PRIMARY, start=START, end=END, interval="1d",
                    progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    oc = d[["Open", "Close"]].dropna()
    with open(CACHE, "wb") as f:
        pickle.dump(oc, f)
    return oc


def decompose(oc):
    o, c = oc["Open"].astype(float), oc["Close"].astype(float)
    pc = c.shift(1)
    on = o / pc - 1.0
    idd = c / o - 1.0
    out = pd.DataFrame({"overnight": on, "intraday": idd}, index=oc.index).dropna()
    return out


def sharpe(r):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return r.mean() / r.std(ddof=1) * np.sqrt(TDAYS)


# ---------------------------------------------------------------------------
# (B) STATIONARY BOOTSTRAP
# ---------------------------------------------------------------------------
def ppw_block_length(x):
    """Patton-Politis-White (2009) optimale Blocklaenge fuer Stationary
    Bootstrap. Flat-Top-Kernel + datengetriebenes M ueber implizite
    Autokorrelations-Tests. Ersetzt die fixe Blocklaenge=21 als Robustheits-
    Check (Grok-Vorschlag)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    kn = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    mmax = int(np.ceil(np.sqrt(n))) + kn
    acov = np.array([np.dot(x[:n - k], x[k:]) / n for k in range(mmax + 1)])
    rho = acov / acov[0]
    crit = 2.0 * np.sqrt(np.log10(n) / n)        # implizite Signifikanzschwelle
    a = np.abs(rho)
    mhat = None
    for m in range(1, mmax - kn + 1):            # erstes Fenster mit kn ruhigen Lags
        if np.all(a[m + 1: m + 1 + kn] < crit):
            mhat = m
            break
    if mhat is None:
        sig = np.where(a[1:] >= crit)[0]
        mhat = int(sig[-1] + 1) if len(sig) else 1
    M = max(1, min(2 * mhat, mmax))
    ks = np.arange(-M, M + 1)
    lam = np.where(np.abs(ks) / M <= 0.5, 1.0,
                   np.where(np.abs(ks) / M <= 1.0, 2.0 * (1.0 - np.abs(ks) / M), 0.0))
    g = acov[np.abs(ks)]
    Ghat = np.sum(lam * np.abs(ks) * g)
    Dsb = 2.0 * np.sum(lam * g) ** 2
    b = (2.0 * Ghat ** 2 / Dsb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    bmax = np.ceil(min(3.0 * np.sqrt(n), n / 3.0))
    return float(np.clip(b, 1.0, bmax))


def stationary_bootstrap_indices(n, exp_block, rng):
    """Politis-Romano: geometrische Blocklaengen, zirkulaer."""
    p = 1.0 / exp_block
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(n)
    coin = rng.random(n) < p
    jumps = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = jumps[t] if coin[t] else (idx[t - 1] + 1) % n
    return idx


def bootstrap_cis(on, idd, n_boot, exp_block, rng):
    n = len(on)
    d = on - idd
    mean_d = np.empty(n_boot)
    sharpe_diff = np.empty(n_boot)
    cum_diff = np.empty(n_boot)
    for b in range(n_boot):
        ix = stationary_bootstrap_indices(n, exp_block, rng)
        on_b, id_b, d_b = on[ix], idd[ix], d[ix]
        mean_d[b] = d_b.mean() * TDAYS
        s_on = on_b.mean() / on_b.std(ddof=1) * np.sqrt(TDAYS)
        s_id = id_b.mean() / id_b.std(ddof=1) * np.sqrt(TDAYS)
        sharpe_diff[b] = s_on - s_id
        cum_diff[b] = (np.prod(1 + on_b) - 1) - (np.prod(1 + id_b) - 1)
    return mean_d, sharpe_diff, cum_diff


def ci_pval(boot, point):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # zweiseitiger Bootstrap-p-Wert gegen H0: Groesse <= 0
    frac_le0 = np.mean(boot <= 0)
    p_two = 2 * min(frac_le0, 1 - frac_le0)
    return lo, hi, p_two


# ---------------------------------------------------------------------------
# (C) STRUKTURBRUCH (sup-Wald / QLR auf Mittel von d_t)
# ---------------------------------------------------------------------------
def supwald_path(d, trim):
    """sup-Wald ueber alle Bruchkandidaten via Kumulativsummen (O(n))."""
    n = len(d)
    cs = np.concatenate([[0.0], np.cumsum(d)])
    cs2 = np.concatenate([[0.0], np.cumsum(d * d)])
    tot, tot2 = cs[-1], cs2[-1]
    k0, k1 = int(trim * n), int((1 - trim) * n)
    ks = np.arange(max(k0, 1), min(k1, n - 1) + 1)
    n1 = ks.astype(float)
    n2 = (n - ks).astype(float)
    s1 = cs[ks]
    s2 = tot - s1
    m1 = s1 / n1
    m2 = s2 / n2
    rss = (cs2[ks] - n1 * m1**2) + (tot2 - cs2[ks] - n2 * m2**2)
    s2v = rss / (n - 2)
    f = (m1 - m2) ** 2 / (s2v * (1.0 / n1 + 1.0 / n2))
    j = int(np.argmax(f))
    return float(f[j]), int(ks[j])


def break_test(d, n_boot, exp_block, trim, rng):
    qlr_obs, k_obs = supwald_path(d, trim)
    e = d - d.mean()                       # H0: konstantes Mittel
    boot = np.empty(n_boot)
    for b in range(n_boot):
        ix = stationary_bootstrap_indices(len(e), exp_block, rng)
        boot[b], _ = supwald_path(e[ix], trim)
    p = float(np.mean(boot >= qlr_obs))
    return qlr_obs, k_obs, p


# ===========================================================================
print("=" * 78)
print("OVERNIGHT vs INTRADAY — ROLLING + BOOTSTRAP + STRUKTURBRUCH (SPY)")
print("=" * 78)

print(f"\n[1/4] Lade {PRIMARY} Tages-OHLC (auto_adjust)...", flush=True)
dec = decompose(fetch_spy())
on = dec["overnight"].to_numpy(float)
idd = dec["intraday"].to_numpy(float)
d = on - idd
dates = dec.index
print(f"  {len(dec)} Handelstage  {dates.min().date()} .. {dates.max().date()}")
print(f"  d_t = overnight - intraday:  Mittel {d.mean()*1e4:+.2f} bps/Tag  "
      f"(annualisiert {d.mean()*TDAYS*100:+.1f}%)")

# --- (A) ROLLING -----------------------------------------------------------
print("\n[2/4] (A) Rolling-Stabilitaet (kumuliert on>id je Fenster)\n", flush=True)
# log-Differenz: rolling-Summe>0  <=>  kumuliert overnight>intraday im Fenster
g = np.log1p(on) - np.log1p(idd)
g_ser = pd.Series(g, index=dates)
roll = {}
for years in (3, 5):
    W = years * TDAYS
    rs = g_ser.rolling(W).sum().dropna()
    frac = float((rs > 0).mean())
    ann_mean = pd.Series(d, index=dates).rolling(W).mean().dropna() * TDAYS
    roll[years] = {"windows": int(len(rs)), "frac_on_gt_id": round(frac, 3),
                   "ann_mean_min": round(float(ann_mean.min()) * 100, 1),
                   "ann_mean_med": round(float(ann_mean.median()) * 100, 1),
                   "ann_mean_max": round(float(ann_mean.max()) * 100, 1)}
    print(f"  {years}J-Fenster (W={W}): {len(rs)} Fenster | on>id in "
          f"{frac*100:5.1f}% | ann. d_t  min {ann_mean.min()*100:+5.1f}%  "
          f"med {ann_mean.median()*100:+5.1f}%  max {ann_mean.max()*100:+5.1f}%")

# --- (B) BOOTSTRAP ---------------------------------------------------------
pt_mean = d.mean() * TDAYS
pt_sd = sharpe(on) - sharpe(idd)
pt_cd = (np.prod(1 + on) - 1) - (np.prod(1 + idd) - 1)

# PW-automatische Blocklaenge als Robustheits-Check gegen die fixe 21d
pw_block = ppw_block_length(d)
print(f"\n[3/4] (B) Stationary Bootstrap (N={N_BOOT})", flush=True)
print(f"  Blocklaenge: fix={EXP_BLOCK}d  vs  Patton-Politis-White-auto="
      f"{pw_block:.1f}d\n", flush=True)

results_by_block = {}
for tag, blk in [("fix21", EXP_BLOCK), ("pw_auto", pw_block)]:
    mb, sb, cb = bootstrap_cis(on, idd, N_BOOT, blk, RNG)
    results_by_block[tag] = {
        "block": round(blk, 1),
        "mean_d": ci_pval(mb, pt_mean),
        "sharpe_diff": ci_pval(sb, pt_sd),
        "cum_diff": ci_pval(cb, pt_cd),
    }
    print(f"  --- Block {blk:.1f}d ({tag}) ---")
    for label, key, pt, unit in [
            ("ann. Mittel d_t        ", "mean_d", pt_mean, "%"),
            ("Sharpe(on) - Sharpe(id)", "sharpe_diff", pt_sd, ""),
            ("kum. (on - id)         ", "cum_diff", pt_cd, "%")]:
        lo, hi, p = results_by_block[tag][key]
        s = 100 if unit == "%" else 1
        print(f"    {label}: Punkt {pt*s:+8.2f}{unit}  95%-KI "
              f"[{lo*s:+8.2f}, {hi*s:+8.2f}]{unit}  p(2s)={p:.3f}")

# fix21 bleibt die Referenz fuer Fazit/Export (vorregistrierter Default)
lo_m, hi_m, p_m = results_by_block["fix21"]["mean_d"]
lo_s, hi_s, p_s = results_by_block["fix21"]["sharpe_diff"]
ci_excl0_mean = (lo_m > 0) or (hi_m < 0)
ci_excl0_sharpe = (lo_s > 0) or (hi_s < 0)
# PW-Robustheit: kippt die Schlussfolgerung bei datengetriebener Blocklaenge?
pw_p_mean = results_by_block["pw_auto"]["mean_d"][2]
pw_p_sharpe = results_by_block["pw_auto"]["sharpe_diff"][2]
print(f"\n  Robustheit (PW vs fix): Mittel-p {p_m:.3f}->{pw_p_mean:.3f} | "
      f"Sharpe-p {p_s:.3f}->{pw_p_sharpe:.3f}  "
      f"=> Schluss {'STABIL' if (pw_p_sharpe<0.05)==(p_s<0.05) else 'KIPPT'}")

# --- (C) STRUKTURBRUCH -----------------------------------------------------
print(f"\n[4/4] (C) Strukturbruch sup-Wald/QLR (Block-Bootstrap-Kritikwerte, "
      f"N={N_BOOT_BREAK})\n", flush=True)
qlr, k_break, p_break = break_test(d, N_BOOT_BREAK, EXP_BLOCK, TRIM, RNG)
break_date = dates[k_break]
print(f"  sup-Wald (QLR) = {qlr:.2f}   geschaetzter Bruch: {break_date.date()} "
      f"(Tag {k_break}/{len(d)})")
print(f"  Bootstrap-p (H0 kein Bruch) = {p_break:.3f}  -> "
      f"{'SIGNIFIKANTER Bruch' if p_break < 0.05 else 'kein signifikanter Bruch'}")
pre = d[:k_break].mean() * TDAYS
post = d[k_break:].mean() * TDAYS
print(f"  ann. d_t vor Bruch {pre*100:+.1f}%   nach Bruch {post*100:+.1f}%")

# ===========================================================================
print("\n" + "=" * 78)
print("FAZIT ROLLING/BOOTSTRAP")
print("=" * 78)
print(f"  Rolling on>id (5J-Fenster):     {roll[5]['frac_on_gt_id']*100:.0f}% der Fenster")
print(f"  ann. Mittel d_t KI schliesst 0 aus: {'JA' if ci_excl0_mean else 'NEIN'}")
print(f"  Sharpe-Differenz KI schliesst 0 aus: {'JA' if ci_excl0_sharpe else 'NEIN'}")
print(f"  Strukturbruch (p<0.05):         {'JA @ '+str(break_date.date()) if p_break < 0.05 else 'NEIN'}")
g1_robust = ci_excl0_mean and roll[5]["frac_on_gt_id"] >= 0.8
print(f"\n  G1-ROBUST (KI>0 & >=80% der 5J-Fenster on>id): "
      f"{'PASS' if g1_robust else 'TEILWEISE/FAIL'}")

# --- EXPORT ----------------------------------------------------------------
res = {
    "n_days": int(len(dec)),
    "date_start": str(dates.min().date()),
    "date_end": str(dates.max().date()),
    "d_mean_bps_day": round(float(d.mean() * 1e4), 2),
    "d_mean_ann_pct": round(float(d.mean() * TDAYS * 100), 1),
    "rolling": roll,
    "boot_n": N_BOOT, "boot_exp_block": EXP_BLOCK,
    "pw_block_auto": round(pw_block, 1),
    "pw_mean_d_p": round(pw_p_mean, 3),
    "pw_sharpe_diff_p": round(pw_p_sharpe, 3),
    "mean_d_ann_pct": round(pt_mean * 100, 1),
    "mean_d_ci_pct": [round(lo_m * 100, 1), round(hi_m * 100, 1)],
    "mean_d_p": round(p_m, 3),
    "sharpe_on": round(sharpe(on), 2), "sharpe_id": round(sharpe(idd), 2),
    "sharpe_diff": round(pt_sd, 2),
    "sharpe_diff_ci": [round(lo_s, 2), round(hi_s, 2)],
    "sharpe_diff_p": round(p_s, 3),
    "cum_diff_pct": round(pt_cd * 100, 1),
    "cum_diff_ci_pct": [round(results_by_block["fix21"]["cum_diff"][0] * 100, 1),
                        round(results_by_block["fix21"]["cum_diff"][1] * 100, 1)],
    "break_qlr": round(qlr, 2),
    "break_date": str(break_date.date()),
    "break_p": round(p_break, 3),
    "break_significant": bool(p_break < 0.05),
    "d_ann_pre_break_pct": round(pre * 100, 1),
    "d_ann_post_break_pct": round(post * 100, 1),
    "g1_robust": bool(g1_robust),
}
(HERE / "overnight_intraday_rolling_bootstrap.json").write_text(
    json.dumps(res, indent=2), encoding="utf-8")
print("\nGespeichert: overnight_intraday_rolling_bootstrap.json")
