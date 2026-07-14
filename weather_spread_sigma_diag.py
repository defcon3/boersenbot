# -*- coding: utf-8 -*-
"""Diagnostik zur Pre-Reg `weather_spread_sigma_2026_07_14.md`.

WICHTIG — Status dieser Auswertungen:
  * Der vorab deklarierte Sommer-OOS-Schnitt ist Teil der Pre-Reg.
  * Die BIN-WEISE Reliability und der Beijing-Gegencheck sind NACHTRAeGLICH
    (das vorregistrierte G3 hat aggregiert und war damit zu stumpf: es liess auch
    den Status quo durch, den wir als fehlkalibriert kennen). Sie sind daher
    EXPLORATIV und werden als solche berichtet — kein nachgeschobener Gate-Pass.
"""
import math
import sys

import numpy as np
from scipy.stats import norm

from weather_spread_sigma_fit import (load_data, fit_b, bucket_p_norm, bucket_p_emp,
                                      MIN_DAYS, IS_FRAC, SIGMA_FLOOR, SPREAD_CUT)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

rows = load_data(700, False)
cities = sorted({r[0] for r in rows})
cities = [c for c in cities if sum(1 for r in rows if r[0] == c) >= MIN_DAYS]
rows = [r for r in rows if r[0] in cities]
rows.sort(key=lambda r: r[1])
days = sorted({r[1] for r in rows})
cut = days[int(len(days) * IS_FRAC)]
is_rows = [r for r in rows if r[1] < cut]
oos_rows = [r for r in rows if r[1] >= cut]
ci = {c: i for i, c in enumerate(cities)}


def arrs(rs):
    return (np.array([r[2] for r in rs]), np.array([r[3] for r in rs]),
            np.array([r[4] for r in rs]), np.array([ci[r[0]] for r in rs]))


s_is, ens_is, ist_is, c_is = arrs(is_rows)
s_oo, ens_oo, ist_oo, c_oo = arrs(oos_rows)
bias = np.array([np.mean(ens_is[c_is == c] - ist_is[c_is == c]) for c in range(len(cities))])
err_is = ist_is - (ens_is - bias[c_is])
b, a = fit_b(err_is, s_is, c_is, len(cities))
sig_fix_c = np.array([err_is[c_is == c].std() for c in range(len(cities))])
z_is = np.sort(err_is / np.maximum(a[c_is] + b * s_is, SIGMA_FLOOR))

mu_oo = ens_oo - bias[c_oo]
sig_fix = sig_fix_c[c_oo]
sig_new = np.maximum(a[c_oo] + b * s_oo, SIGMA_FLOOR)

MODELLE = [
    ("Status quo (festes Sigma)", sig_fix, lambda k, m, s: bucket_p_norm(k, m, s)),
    ("NEU sigma(s), Normal", sig_new, lambda k, m, s: bucket_p_norm(k, m, s)),
    ("NEU sigma(s), empir. z", sig_new, lambda k, m, s: bucket_p_emp(k, m, s, z_is)),
]
BINS = [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20)]


def reliability(mask, titel):
    print(f"\n{'='*94}\n{titel}\n{'='*94}")
    print(f"{'Modell':26} {'P-Bin':>10} {'Regime':>18} {'vorherg.':>9} {'real.':>8} {'Faktor':>8} {'n':>7}")
    print("-" * 94)
    for name, sig, pf in MODELLE:
        for blo, bhi in BINS:
            for rlbl, rmask in (("ruhig", s_oo < SPREAD_CUT), ("zerklueftet", s_oo >= SPREAD_CUT)):
                pred, hit = [], []
                for i in np.where(mask & rmask)[0]:
                    k_ist = math.floor(ist_oo[i] + 0.5)
                    ks = range(int(mu_oo[i]) - 6, int(mu_oo[i]) + 7)
                    ps = {k: pf(k, mu_oo[i], sig[i]) for k in ks}
                    tot = sum(ps.values())
                    for k, p in ps.items():
                        pn = p / tot
                        if blo < pn <= bhi:
                            pred.append(pn)
                            hit.append(1.0 if k == k_ist else 0.0)
                if len(pred) < 30:
                    continue
                pr, re = float(np.mean(pred)), float(np.mean(hit))
                fac = re / pr
                flag = "" if fac <= 1.25 else "  <-- ueberkonfident"
                print(f"{name:26} {blo*100:3.0f}-{bhi*100:3.0f}% {rlbl:>18} "
                      f"{pr*100:8.2f}% {re*100:7.2f}% {fac:7.2f}x {len(pred):7d}{flag}")
        print()


# ---- 1) Bin-weise Reliability (EXPLORATIV, nicht das vorregistrierte Gate) ----
reliability(np.ones(len(oos_rows), bool),
            "OOS, BIN-WEISE (explorativ — das vorregistrierte G3 aggregierte und war zu stumpf)")

# ---- 2) Sommer-OOS (VORAB deklariert in der Pre-Reg) ----
summer = np.array([r[1][5:7] in ("06", "07", "08") for r in oos_rows])
if summer.sum() > 200:
    reliability(summer, f"OOS SOMMER (Jun-Aug, vorab deklariert) — n={summer.sum()} Stadt-Tage")

# ---- 3) Der Gegencheck: haette sigma(s) den Beijing-33-Trade gestoppt? ----
print("\n" + "=" * 94)
print("GEGENCHECK: Beijing 33°C (Zieltag 14.07.) — haette sigma(s) allein den Trade verhindert?")
print("=" * 94)
bj = cities.index("Beijing")
raw = {"GFS": 34.7, "ICON": 34.4, "UKMO": 34.0, "JMA": 37.1, "ECMWF": 33.5}
spread_bj = max(raw.values()) - min(raw.values())
ens_bj = sum(raw.values()) / 5
ens_rob = sum(v for k, v in raw.items() if k != "JMA") / 4
sig_bj_fix = sig_fix_c[bj]
sig_bj_new = max(a[bj] + b * spread_bj, SIGMA_FLOOR)
print(f"  Beijing: a_city {a[bj]:.2f}, festes Sigma {sig_bj_fix:.2f}, Spanne {spread_bj:.1f}° "
      f"-> sigma(s) = {a[bj]:.2f} + {b:.3f}*{spread_bj:.1f} = {sig_bj_new:.2f}")
print(f"  (Zum Vergleich: Beijings Median-Spanne = {np.median(s_is[c_is == bj]):.1f}°)\n")
print(f"  {'mu-Quelle':28} {'Sigma':>7} {'P(33er)':>9}   Urteil bei BE 21 %")
print("  " + "-" * 74)
for mlbl, ens in (("volles ENS (5 Modelle)", ens_bj), ("ohne JMA (Ausreisser raus)", ens_rob)):
    for slbl, sg in (("festes Sigma", sig_bj_fix), ("sigma(s)", sig_bj_new)):
        # 700d-Bias aus diesem Datensatz
        mu = ens - bias[bj]
        p = bucket_p_norm(33, mu, sg)
        print(f"  {mlbl+' / '+slbl:28} {sg:7.2f} {p*100:8.1f}%   "
              f"{'+EV' if p < 0.21 else '-EV'}  (mu {mu:.2f})")
print("\n  Ist-Hoch war 33,0 °C -> der Bucket wurde GETROFFEN.")
