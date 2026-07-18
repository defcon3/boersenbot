#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""weather_calib_prio235_eval.py — Auswertung Backlog Prio 2/3/5 (18.07.2026).

Liest den Residuen-Dump aus weather_source_compare.py --dump-residuals
(preregs/weather_residuals_lead1_2026_07_18.csv.gz) und wertet die drei
VOR der Auswertung fixierten Gates aus:
preregs/weather_calib_prio235_prereg_2026_07_18.md

  H2 (Prio 2): bias(doy)-Harmonik vs 700d-Konstante vs 40d-Fenster,
               walk-forward, ENS5-Residuen. Gate: harm1 schlaegt BEIDE,
               Stadt-t > 2.
  H3 (Prio 3): 8 statt 5 Modelle. Gate: walk-forward-debiastes ens8 vs ens5
               auf identischer Tagesmenge, gepoolt >= 3 % MAE-Gewinn, >= 60 %
               der Staedte, Archiv-Tiefe n >= 350 fuer alle 3 neuen in >= 20
               Staedten. Plus deskriptiv: Ranking, Heimmodell-These.
  H5 (Prio 5): Inverse-Kovarianz-Gewichtung (Gleichkorrelations-Sigma,
               monatlicher Refit) vs Gleichgewichtung, 5er-Basis.
               Gate: >= 5 % MAE-Gewinn gepoolt UND Stadt-t > 2.
               Zusatzarm (nur Bericht): JMA+UKMO raus.

Umsetzungs-Festlegungen (vor dem ersten Lauf entschieden, nicht getunt):
  - Anlauf = 180 SERIEN-Tage der jeweiligen Stadt-Serie (Tage mit Daten),
    bewertet wird ab Serien-Tag 181.
  - roll40 = Kalenderfenster [t-40d, t); unter 15 Werten Fallback auf const.
  - H3 "auswertbare Stadt" = Alle-8-Schnittserie >= 280 Tage (180 Anlauf
    + >= 100 Bewertungstage).
  - H5-Refit alle 30 Serien-Tage; Gewichte negativ geklippt + renormiert.
  - Shenzhen ist ueberall ausgeschlossen (Settlement-Ziel ist WU, Prio 0).
"""
import sys

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

DUMP = "preregs/weather_residuals_lead1_2026_07_18.csv.gz"
BASE5 = ["gfs_seamless", "icon_seamless", "ukmo_seamless", "jma_seamless", "ecmwf_ifs025"]
EXT3 = ["gem_seamless", "meteofrance_seamless", "cma_grapes_global"]
ALL8 = BASE5 + EXT3
SHORT = {"gfs_seamless": "GFS", "icon_seamless": "ICON", "ukmo_seamless": "UKMO",
         "jma_seamless": "JMA", "ecmwf_ifs025": "ECMWF", "gem_seamless": "GEM",
         "meteofrance_seamless": "MF", "cma_grapes_global": "CMA"}
WARMUP = 180
ROLL_DAYS = 40
ROLL_MIN = 15
PERIOD = 365.25
HOME = {"gem_seamless": ["Toronto"], "meteofrance_seamless": ["Paris"],
        "cma_grapes_global": ["Beijing", "Shanghai", "Chengdu", "Wuhan"]}


def city_matrix(df, city, models):
    """Tage x Modelle Residuen-Matrix (nur Tage, an denen ALLE models liefern).
    Rueckgabe: dates (np.datetime64[D]), doy, R (nTage x nModelle)."""
    sub = df[(df.city == city) & (df.model.isin(models))]
    piv = sub.pivot_table(index="date", columns="model", values="resid").dropna()
    piv = piv.sort_index()
    if piv.empty:
        return None, None, None
    dates = piv.index.values.astype("datetime64[D]")
    doy = piv.index.dayofyear.values.astype(float)
    return dates, doy, piv[models].values


def harm_design(doy, order):
    cols = [np.ones_like(doy)]
    for k in range(1, order + 1):
        w = 2.0 * np.pi * k * doy / PERIOD
        cols += [np.sin(w), np.cos(w)]
    return np.column_stack(cols)


def walkforward_bias(dates, doy, r):
    """Walk-forward-Bias-Schaetzer fuer eine Residuen-Serie r[t].
    Liefert dict est -> bias_hat-Array (NaN im Anlauf)."""
    n = len(r)
    out = {e: np.full(n, np.nan) for e in ("const", "roll40", "harm1", "harm2")}
    X1 = harm_design(doy, 1)
    X2 = harm_design(doy, 2)
    csum = np.cumsum(r)
    for i in range(WARMUP, n):
        out["const"][i] = csum[i - 1] / i
        lo = dates[i] - np.timedelta64(ROLL_DAYS, "D")
        mask = (dates[:i] >= lo)
        out["roll40"][i] = r[:i][mask].mean() if mask.sum() >= ROLL_MIN else out["const"][i]
        for est, X in (("harm1", X1), ("harm2", X2)):
            beta, *_ = np.linalg.lstsq(X[:i], r[:i], rcond=None)
            out[est][i] = float(X[i] @ beta)
    return out


def paired_city_t(deltas):
    """t-Statistik ueber Stadt-Mittel (Staedte als unabhaengige Einheiten)."""
    d = np.array(deltas, float)
    if len(d) < 3:
        return float("nan")
    return d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))


def main():
    df = pd.read_csv(DUMP, parse_dates=["date"])
    df["resid"] = df.forecast - df.actual
    df = df[df.city != "Shenzhen"]
    cities = sorted(df.city.unique())
    print(f"Dump: {len(df)} Zeilen, {len(cities)} Staedte (Shenzhen ausgeschlossen), "
          f"{df.date.min().date()} .. {df.date.max().date()}")

    # ================= H2: Saison-Harmonik (ENS5) =================
    print("\n" + "=" * 78)
    print("H2 (Prio 2) — bias(doy)-Harmonik vs const(700d-Logik) vs roll40, walk-forward")
    print("=" * 78)
    mae = {e: {} for e in ("const", "roll40", "harm1", "harm2")}
    pooled = {e: [] for e in mae}
    for city in cities:
        dates, doy, R = city_matrix(df, city, BASE5)
        if dates is None or len(dates) < WARMUP + 50:
            continue
        r = R.mean(axis=1)  # ENS5-Residuum = Mittel der Modell-Residuen
        est = walkforward_bias(dates, doy, r)
        ok = ~np.isnan(est["const"])
        for e in mae:
            errs = np.abs(r[ok] - est[e][ok])
            mae[e][city] = errs.mean()
            pooled[e].append(errs)
    common = sorted(set(mae["const"]) & set(mae["harm1"]))
    print(f"{'Stadt':13} {'const':>7} {'roll40':>7} {'harm1':>7} {'harm2':>7}   harm1-Gewinn vs const")
    for c in common:
        gain = (mae["const"][c] - mae["harm1"][c]) / mae["const"][c] * 100
        print(f"{c:13} {mae['const'][c]:7.3f} {mae['roll40'][c]:7.3f} "
              f"{mae['harm1'][c]:7.3f} {mae['harm2'][c]:7.3f}   {gain:+5.1f} %")
    pool_mae = {e: np.concatenate(pooled[e]).mean() for e in mae}
    print(f"\n{'GEPOOLT':13} " + " ".join(f"{pool_mae[e]:7.3f}" for e in ("const", "roll40", "harm1", "harm2")))
    d_const = [mae["const"][c] - mae["harm1"][c] for c in common]
    d_roll = [mae["roll40"][c] - mae["harm1"][c] for c in common]
    t_const = paired_city_t(d_const)
    t_roll = paired_city_t(d_roll)
    n_beat_c = sum(1 for x in d_const if x > 0)
    n_beat_r = sum(1 for x in d_roll if x > 0)
    g_const = (pool_mae["const"] - pool_mae["harm1"]) / pool_mae["const"] * 100
    g_roll = (pool_mae["roll40"] - pool_mae["harm1"]) / pool_mae["roll40"] * 100
    print(f"\nharm1 vs const : gepoolt {g_const:+.2f} %  Stadt-t {t_const:+.2f}  ({n_beat_c}/{len(common)} Staedte besser)")
    print(f"harm1 vs roll40: gepoolt {g_roll:+.2f} %  Stadt-t {t_roll:+.2f}  ({n_beat_r}/{len(common)} Staedte besser)")
    h2_green = (pool_mae["harm1"] < pool_mae["const"] and pool_mae["harm1"] < pool_mae["roll40"]
                and t_const > 2 and t_roll > 2)
    print(f"GATE G-H2: {'GRUEN' if h2_green else 'ROT'} "
          f"(harm1 < beide UND beide Stadt-t > 2 gefordert)")

    # ================= H3: 8 statt 5 Modelle =================
    print("\n" + "=" * 78)
    print("H3 (Prio 3) — ens8 vs ens5 (walk-forward-debiast, identische Tagesmenge)")
    print("=" * 78)
    nn = df.groupby(["model", "city"]).size().unstack(fill_value=0)
    depth_ok = {m: int((nn.loc[m] >= 350).sum()) for m in EXT3}
    print("Archiv-Tiefe (Staedte mit n>=350): "
          + ", ".join(f"{SHORT[m]} {depth_ok[m]}/{len(cities)}" for m in EXT3))
    gains, d5_all, d8_all = {}, [], []
    ranks = {m: [] for m in ALL8}
    home_rows = []
    for city in cities:
        dates, doy, R = city_matrix(df, city, ALL8)
        if dates is None or len(dates) < WARMUP + 100:
            continue
        n = len(dates)
        # walk-forward-const-Debias je Modell, dann Mittel ueber 5 bzw. 8
        deb = np.full_like(R, np.nan)
        csums = np.cumsum(R, axis=0)
        for i in range(WARMUP, n):
            deb[i] = R[i] - csums[i - 1] / i
        ok = ~np.isnan(deb[:, 0])
        e5 = np.abs(deb[ok][:, :5].mean(axis=1))
        e8 = np.abs(deb[ok].mean(axis=1))
        gains[city] = (e5.mean() - e8.mean()) / e5.mean() * 100
        d5_all.append(e5)
        d8_all.append(e8)
        # deskriptives Ranking (voller Zeitraum, roher MAE je Modell)
        maes = np.abs(R).mean(axis=0)
        order = maes.argsort().argsort() + 1  # Rang 1 = bester
        for k, m in enumerate(ALL8):
            ranks[m].append((city, int(order[k])))
    e5p = np.concatenate(d5_all).mean()
    e8p = np.concatenate(d8_all).mean()
    g_pool = (e5p - e8p) / e5p * 100
    n_better = sum(1 for g in gains.values() if g > 0)
    print(f"Staedte auswertbar: {len(gains)}; ens8 besser in {n_better} "
          f"({n_better / max(len(gains), 1) * 100:.0f} %); gepoolter MAE-Gewinn {g_pool:+.2f} % "
          f"(ens5 {e5p:.3f} -> ens8 {e8p:.3f})")
    top3 = sorted(gains, key=gains.get, reverse=True)[:3]
    bot3 = sorted(gains, key=gains.get)[:3]
    print("  beste Staedte: " + ", ".join(f"{c} {gains[c]:+.1f} %" for c in top3))
    print("  schlechteste : " + ", ".join(f"{c} {gains[c]:+.1f} %" for c in bot3))
    print("\nMittlerer MAE-Rang (1=best, ueber Alle-8-Schnitt): "
          + ", ".join(f"{SHORT[m]} {np.mean([r for _, r in ranks[m]]):.2f}" for m in ALL8))
    print("Heimmodell-These:")
    for m, homes in HOME.items():
        hr = [r for c, r in ranks[m] if c in homes]
        fr = [r for c, r in ranks[m] if c not in homes]
        if hr:
            print(f"  {SHORT[m]:5}: Heim-Rang {np.mean(hr):.1f} ({', '.join(f'{c} {r}' for c, r in ranks[m] if c in homes)}) "
                  f"vs Fremd-Median {np.median(fr):.1f}")
    arch_ok = all(depth_ok[m] >= 20 for m in EXT3)
    h3_green = g_pool >= 3.0 and n_better / max(len(gains), 1) >= 0.60 and arch_ok
    print(f"GATE G-H3: {'GRUEN' if h3_green else 'ROT'} "
          f"(>=3 % gepoolt, >=60 % Staedte, Archiv n>=350 in >=20 Staedten)")

    # ================= H5: Gewichtung =================
    print("\n" + "=" * 78)
    print("H5 (Prio 5) — Inverse-Kovarianz-Gewichtung vs Gleichgewichtung (5er, walk-forward)")
    print("=" * 78)
    res_rows = []
    ew_all, iv_all, drop_all = [], [], []
    for city in cities:
        dates, doy, R = city_matrix(df, city, BASE5)
        if dates is None or len(dates) < WARMUP + 100:
            continue
        n, k = R.shape
        w = np.full(k, 1.0 / k)
        bias = R[:WARMUP].mean(axis=0)
        e_ew, e_iv, e_dr = [], [], []
        drop_idx = [i for i, m in enumerate(BASE5) if m not in ("jma_seamless", "ukmo_seamless")]
        for i in range(WARMUP, n):
            if (i - WARMUP) % 30 == 0:
                hist = R[:i]
                bias = hist.mean(axis=0)
                E = hist - bias
                sd = E.std(axis=0, ddof=1)
                C = np.corrcoef(E.T)
                iu = np.triu_indices(k, 1)
                rho = float(np.clip(C[iu].mean(), 0.0, 0.98))
                Sig = np.outer(sd, sd) * (rho * np.ones((k, k)) + (1 - rho) * np.eye(k))
                try:
                    wi = np.linalg.solve(Sig, np.ones(k))
                except np.linalg.LinAlgError:
                    wi = 1.0 / np.maximum(sd, 1e-6) ** 2
                wi = np.clip(wi, 0.0, None)
                w = wi / wi.sum() if wi.sum() > 0 else np.full(k, 1.0 / k)
            deb = R[i] - bias
            e_ew.append(deb.mean())
            e_iv.append(float(w @ deb))
            e_dr.append(deb[drop_idx].mean())
        e_ew, e_iv, e_dr = (np.abs(np.array(x)) for x in (e_ew, e_iv, e_dr))
        res_rows.append((city, e_ew.mean(), e_iv.mean(), e_dr.mean(),
                         (e_ew.mean() - e_iv.mean()) / e_ew.mean() * 100))
        ew_all.append(e_ew)
        iv_all.append(e_iv)
        drop_all.append(e_dr)
    print(f"{'Stadt':13} {'gleich':>7} {'invKov':>7} {'-JMA/UKMO':>9}   invKov-Gewinn")
    for city, a, b, c, g in sorted(res_rows, key=lambda x: -x[4]):
        mark = "  <- Seoul" if city == "Seoul" else ""
        print(f"{city:13} {a:7.3f} {b:7.3f} {c:9.3f}   {g:+5.1f} %{mark}")
    ewp = np.concatenate(ew_all).mean()
    ivp = np.concatenate(iv_all).mean()
    drp = np.concatenate(drop_all).mean()
    g5 = (ewp - ivp) / ewp * 100
    gd = (ewp - drp) / ewp * 100
    t5 = paired_city_t([a - b for _, a, b, _, _ in res_rows])
    td = paired_city_t([a - c for _, a, _, c, _ in res_rows])
    n_bet = sum(1 for *_, g in res_rows if g > 0)
    print(f"\ninvKov  vs gleich: gepoolt {g5:+.2f} %  Stadt-t {t5:+.2f}  ({n_bet}/{len(res_rows)} besser)")
    print(f"-JMA/UKMO vs gleich (Zusatzarm, Bonferroni x2 falls Empfehlung): "
          f"gepoolt {gd:+.2f} %  Stadt-t {td:+.2f}")
    h5_green = g5 >= 5.0 and t5 > 2
    print(f"GATE G-H5: {'GRUEN' if h5_green else 'ROT'} (>=5 % gepoolt UND Stadt-t > 2)")

    print("\nHinweis: Gates + Konsequenzen in preregs/weather_calib_prio235_prereg_2026_07_18.md;"
          "\nnur H2-GRUEN fuehrt in dieser Session zu einem Screen-Einbau (Zusatz-Sicht).")


if __name__ == "__main__":
    main()
