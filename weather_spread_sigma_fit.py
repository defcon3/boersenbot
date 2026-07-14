# -*- coding: utf-8 -*-
"""weather_spread_sigma_fit.py — Auswertung der Pre-Reg
`preregs/weather_spread_sigma_2026_07_14.md` (G1-G4).

Hypothese: sigma_city(s) = a_city + b*s  (s = Spanne der 5 rohen Modell-Tageshochs)
statt festem Sigma je Stadt. Wird das Modell dadurch auch an zerklüfteten Tagen
ehrlich, kann der harte Spannen-Veto (sperrt 37 % aller Tage) durch korrekte
Preise ersetzt werden.

Alles Wesentliche war VOR dem Lauf festgelegt: Modellform, Split (zeitlich 70/30),
Gates, sogar die Vorab-Erwartung. Siehe Pre-Reg.

Aufruf:
  python weather_spread_sigma_fit.py                 # nutzt Cache, sonst holt er Daten
  python weather_spread_sigma_fit.py --refetch
"""
import argparse
import math
import pickle
import sys
import time
from pathlib import Path

import airportsdata
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from weather_source_compare import (STATIONS, MODELS, fetch_model_daily_extreme,
                                    fetch_actual_daily_extreme)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

CACHE = Path("g3_spread_sigma_cache.pkl")   # reproduzierbar aus diesem Skript, gitignorable
MIN_DAYS = 300          # Pre-Reg: Stadt fliegt raus unter 300 nutzbaren Tagen
IS_FRAC = 0.70          # Pre-Reg: aelteste 70 % = In-Sample
SIGMA_FLOOR = 0.3       # Pre-Reg
LAY_ZONE = 0.10         # Pre-Reg: "Lay-Zone" = Buckets mit Modell-P <= 10 %
G3_TOL = 1.25           # Pre-Reg: realisiert <= 1,25 x vorhergesagt
SPREAD_CUT = 3.0        # Veto-Schwelle der Screens
AP = airportsdata.load("ICAO")


# ---------------------------------------------------------------- Daten
def load_data(days, refetch):
    if CACHE.exists() and not refetch:
        return pickle.loads(CACHE.read_bytes())
    from datetime import datetime, timedelta, timezone
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=days)
    rows = []   # (city, dayiso, spread, ens_raw, ist)
    for city, icao in STATIONS.items():
        st = AP.get(icao)
        if not st:
            continue
        try:
            daily, tz = fetch_model_daily_extreme(icao, st["lat"], st["lon"],
                                                  start.isoformat(), end.isoformat(), max)
            actual = fetch_actual_daily_extreme(icao, start, end, tz, max)
        except Exception as ex:
            print(f"  {city:14} Fehler {str(ex)[:50]} -> skip")
            continue
        n0 = len(rows)
        for day, ist in actual.items():
            vals = [daily[m][day] for m in MODELS if day in daily.get(m, {})]
            if len(vals) < 5:
                continue
            rows.append((city, day, max(vals) - min(vals), sum(vals) / 5.0, ist))
        print(f"  {city:14} {len(rows) - n0:4d} Tage")
        time.sleep(1)
    CACHE.write_bytes(pickle.dumps(rows))
    return rows


# ---------------------------------------------------------------- Fit
def fit_b(err, spread, cidx, ncity):
    """MLE fuer sigma_i = a_city + b*spread_i (Gauss). a_city gegeben b in
    geschlossener Naeherung nicht loesbar -> innere 1D-Suche je Stadt."""
    def a_hat(b):
        a = np.zeros(ncity)
        for c in range(ncity):
            m = cidx == c
            e, s = err[m], spread[m]

            def nll_a(av):
                sig = np.maximum(av + b * s, SIGMA_FLOOR)
                return np.sum(np.log(sig) + e ** 2 / (2 * sig ** 2))
            a[c] = minimize_scalar(nll_a, bounds=(0.01, 6.0), method="bounded").x
        return a

    def nll_b(b):
        a = a_hat(b)
        sig = np.maximum(a[cidx] + b * spread, SIGMA_FLOOR)
        return np.sum(np.log(sig) + err ** 2 / (2 * sig ** 2))

    r = minimize_scalar(nll_b, bounds=(-0.1, 1.0), method="bounded")
    return r.x, a_hat(r.x)


def block_bootstrap_se(err, spread, cidx, ncity, days_idx, n=40, block=30, seed=7):
    """Block-Bootstrap (30 zusammenhaengende Tage) — Wetter ist autokorreliert,
    ein naiver t waere aufgeblasen."""
    rng = np.random.default_rng(seed)
    N = len(err)
    order = np.argsort(days_idx)
    bs = []
    nblocks = max(1, N // block)
    for _ in range(n):
        pick = rng.integers(0, max(1, N - block), size=nblocks)
        idx = np.concatenate([order[p:p + block] for p in pick])
        idx = idx[idx < N]
        try:
            b, _ = fit_b(err[idx], spread[idx], cidx[idx], ncity)
            bs.append(b)
        except Exception:
            pass
    return float(np.std(bs, ddof=1)) if len(bs) > 2 else float("nan")


# ---------------------------------------------------------------- Bucket-P
def bucket_p_norm(k, mu, sig):
    return norm.cdf((k + 0.5 - mu) / sig) - norm.cdf((k - 0.5 - mu) / sig)


def bucket_p_emp(k, mu, sig, zs):
    """Empirische z-Quantile statt Normal (Sekundaermodell der Pre-Reg)."""
    lo = np.searchsorted(zs, (k - 0.5 - mu) / sig) / len(zs)
    hi = np.searchsorted(zs, (k + 0.5 - mu) / sig) / len(zs)
    return max(hi - lo, 1e-9)


def evaluate(name, rows, mu, sig, pfun, extra=None):
    """Log-Loss + Reliability in der Lay-Zone, getrennt nach Spannen-Regime."""
    ll, zone = [], []          # zone: (p, hit, spread)
    for i, (_, _, spread, _, ist) in enumerate(rows):
        k_ist = math.floor(ist + 0.5)
        ks = range(int(math.floor(mu[i])) - 6, int(math.ceil(mu[i])) + 7)
        ps = {k: pfun(k, mu[i], sig[i], extra) for k in ks}
        tot = sum(ps.values())
        p_ist = ps.get(k_ist, 1e-9) / max(tot, 1e-9)
        ll.append(-math.log(max(p_ist, 1e-9)))
        for k, p in ps.items():
            pn = p / max(tot, 1e-9)
            if pn <= LAY_ZONE:
                zone.append((pn, 1.0 if k == k_ist else 0.0, spread))
    z = np.array(zone)
    out = {"name": name, "logloss": float(np.mean(ll)), "regimes": {}}
    for lbl, m in (("ruhig (<3°)", z[:, 2] < SPREAD_CUT), ("zerklueftet (>=3°)", z[:, 2] >= SPREAD_CUT)):
        pred, real = float(z[m, 0].mean()), float(z[m, 1].mean())
        out["regimes"][lbl] = (pred, real, real / max(pred, 1e-9), int(m.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=700)
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()

    print("Lade Daten (Cache: %s) ..." % CACHE)
    rows = load_data(args.days, args.refetch)
    cities = sorted({r[0] for r in rows})
    cities = [c for c in cities if sum(1 for r in rows if r[0] == c) >= MIN_DAYS]
    rows = [r for r in rows if r[0] in cities]
    rows.sort(key=lambda r: r[1])
    print(f"\n{len(rows)} Stadt-Tage, {len(cities)} Staedte (>= {MIN_DAYS} Tage)")

    days = sorted({r[1] for r in rows})
    cut = days[int(len(days) * IS_FRAC)]
    is_rows = [r for r in rows if r[1] < cut]
    oos_rows = [r for r in rows if r[1] >= cut]
    print(f"Split (Pre-Reg, zeitlich): IS bis {cut} (n={len(is_rows)}), OOS ab {cut} (n={len(oos_rows)})")

    ci = {c: i for i, c in enumerate(cities)}
    dayi = {d: i for i, d in enumerate(days)}

    def arrs(rs):
        return (np.array([r[2] for r in rs]), np.array([r[3] for r in rs]),
                np.array([r[4] for r in rs]), np.array([ci[r[0]] for r in rs]),
                np.array([dayi[r[1]] for r in rs]))

    s_is, ens_is, ist_is, c_is, d_is = arrs(is_rows)
    s_oo, ens_oo, ist_oo, c_oo, _ = arrs(oos_rows)

    # Bias je Stadt aus IS (Forecast - Ist), wie im Bestand
    bias = np.array([np.mean(ens_is[c_is == c] - ist_is[c_is == c]) for c in range(len(cities))])
    err_is = ist_is - (ens_is - bias[c_is])
    err_oo = ist_oo - (ens_oo - bias[c_oo])

    # ---------------- G1 ----------------
    print("\n" + "=" * 78 + "\nG1 (IS): Ist Sigma spannen-abhaengig?\n" + "=" * 78)
    for lo, hi in ((0, 1.5), (1.5, 3), (3, 5), (5, 99)):
        m = (s_is >= lo) & (s_is < hi)
        if m.sum():
            print(f"  Spanne {lo:>4.1f}-{hi:<4.1f}  n={m.sum():5d}  empirisches Sigma {err_is[m].std():.3f}°")
    b, a = fit_b(err_is, s_is, c_is, len(cities))
    se = block_bootstrap_se(err_is, s_is, c_is, len(cities), d_is)
    t = b / se if se == se and se > 0 else float("nan")
    print(f"\n  MLE: sigma_city(s) = a_city + {b:.3f}*s   (Block-Bootstrap SE {se:.3f} -> t = {t:.1f})")
    print(f"  a_city: min {a.min():.2f}  median {np.median(a):.2f}  max {a.max():.2f}")
    g1 = (b > 0) and (t > 4)
    print(f"  ==> G1 {'BESTANDEN' if g1 else 'GESCHEITERT'} (verlangt: b>0 und t>4)")

    # ---------------- Modelle auf OOS ----------------
    sig_fix = np.array([err_is[c_is == c].std() for c in range(len(cities))])[c_oo]  # Status quo
    sig_new = np.maximum(a[c_oo] + b * s_oo, SIGMA_FLOOR)
    mu_oo = ens_oo - bias[c_oo]

    z_is = np.sort(err_is / np.maximum(a[c_is] + b * s_is, SIGMA_FLOOR))  # empirische z (IS!)

    res = [
        evaluate("Status quo (festes Sigma, Normal)", oos_rows, mu_oo, sig_fix,
                 lambda k, m, s, e: bucket_p_norm(k, m, s)),
        evaluate("NEU: sigma(s), Normal", oos_rows, mu_oo, sig_new,
                 lambda k, m, s, e: bucket_p_norm(k, m, s)),
        evaluate("NEU: sigma(s), empirische z", oos_rows, mu_oo, sig_new,
                 lambda k, m, s, e: bucket_p_emp(k, m, s, z_is)),
    ]

    print("\n" + "=" * 78 + "\nG2 (OOS): Log-Loss der Bucket-Wahrscheinlichkeiten\n" + "=" * 78)
    for r in res:
        print(f"  {r['name']:38} {r['logloss']:.4f}")
    g2 = res[1]["logloss"] < res[0]["logloss"]
    print(f"  ==> G2 {'BESTANDEN' if g2 else 'GESCHEITERT'} "
          f"(sigma(s) besser als festes Sigma: {res[0]['logloss'] - res[1]['logloss']:+.4f})")

    print("\n" + "=" * 78 + f"\nG3 (OOS): Kalibrierung in der LAY-ZONE (Modell-P <= {LAY_ZONE:.0%})"
          f"\n          verlangt: realisiert <= {G3_TOL} x vorhergesagt, in BEIDEN Regimen\n" + "=" * 78)
    print(f"  {'Modell':38} {'Regime':20} {'vorherg.':>9} {'real.':>8} {'Faktor':>8}")
    print("  " + "-" * 88)
    verdict = {}
    for r in res:
        ok = True
        for lbl, (pred, real, fac, n) in r["regimes"].items():
            flag = "OK " if fac <= G3_TOL else "FAIL"
            ok &= fac <= G3_TOL
            print(f"  {r['name']:38} {lbl:20} {pred*100:8.2f}% {real*100:7.2f}% {fac:7.2f}x {flag}")
        verdict[r["name"]] = ok
        print()
    print("  ==> G3: " + " | ".join(f"{n.split(':')[0]}: {'BESTANDEN' if v else 'GESCHEITERT'}"
                                    for n, v in verdict.items()))
    return b, a, bias, cities, verdict


if __name__ == "__main__":
    main()
