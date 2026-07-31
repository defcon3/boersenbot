# -*- coding: utf-8 -*-
"""weather_gfs_ensemble_mu_eval.py — Auswertung zur Pre-Reg
`preregs/weather_gfs_ensemble_mu_2026_07_31.md` (Commit 747d41b).

Hypothese H1: An zerklüfteten Tagen (Modellspanne s > 3 K — genau die, die der
harte Spannen-Veto heute sperrt) ist der biaskorrigierte **Median der 31
GFS-Member** ein genauerer Schätzer des Tagesmaximums als das biaskorrigierte
Mittel der fünf Punktmodelle.

Getestet wird die Robustheit der MITTE, nicht die Breite. Begründung steht in
der Pre-Reg: die Studie vom 14.07. hat sigma bereits als praktisch irrelevante
Fehlerquelle vermessen ("kein Sigma der Welt repariert einen verzogenen
Mittelwert"), während der Ausreißer eines Einzelmodells mit 0,6 K in mu zu Buche
schlug. Ein 31-Member-Ensemble kann diesen Ausreißer strukturell nicht haben.

Aufruf:
    python weather_gfs_ensemble_mu_eval.py                # nutzt Cache
    python weather_gfs_ensemble_mu_eval.py --refetch
    python weather_gfs_ensemble_mu_eval.py --skip-g4      # ohne Marktpreise
"""
import argparse
import datetime as dt
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from scipy.stats import norm

from weather_source_compare import (STATIONS, MODELS, fetch_model_daily_extreme,
                                    fetch_actual_daily_extreme_wu)
import airportsdata

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

# ---------------------------------------------------------------- Pre-Reg-Parameter
ENS_API    = "https://ensemble-api.open-meteo.com/v1/ensemble"
ENS_MODEL  = "gfs025"
LEAD       = 1              # previous_day1 — NIEMALS start_date allein (Lookahead!)
SPREAD_CUT = 3.0            # "zerklüftet" = Spanne der 5 Modelle > 3 K (Veto-Schwelle)
BIAS_WIN   = 40             # 40-Tage-Sommer-Bias, gleitend, nur Tage VOR dem Zieltag
BIAS_MIN   = 10             # darunter kein Bias -> Tag entfällt
# Steigung des Sigma-Modells, NICHT hier neu gefittet — aus der Studie vom
# 14.07. uebernommen. Bewusst der SOMMER-Wert (0,107), nicht der Ganzjahreswert:
# diese Studie deckt nur Mai-Juli ab, und `weather_source_compare.py` haelt dazu
# fest, dass eine Ganzjahres-Steigung auf einem Sommerfenster sigma an ruhigen
# Tagen ~5 % zu ENG macht — also in die gefaehrliche Richtung.
SIGMA_B    = 0.107
SIGMA_FLOOR = 0.3
LAY_ZONE   = 0.10
BINS       = [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20)]   # G3: bin-weise, NICHT gemittelt
BIN_BAND   = (0.75, 1.25)
IS_MONTHS  = ("2026-05", "2026-06")
OOS_MONTHS = ("2026-07",)
FEE_RATE   = 0.05
CACHE      = Path("g3_ens_mu_cache.pkl")
AP         = airportsdata.load("ICAO")


# ---------------------------------------------------------------- Statistik
def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def tstat(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    if var <= 0:
        return float("inf") if m > 0 else (float("-inf") if m < 0 else 0.0)
    return m / math.sqrt(var / n)

def median(xs):
    if not xs:
        return float("nan")
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# ---------------------------------------------------------------- Ensemble holen
def fetch_ensemble_members(lat, lon, start, end, lead=LEAD, retries=4):
    """{tag: [member-Tagesmaxima]} aus previous_day{lead}-Stundenwerten, lokale Zeit.

    Bewusst derselbe Weg wie `fetch_model_daily_extreme`: previous_dayN liefert
    die Prognose, die vor N*24 h fuer diesen Zeitpunkt gemacht wurde. Ein blosser
    start_date-Abruf kann den aktuellen, rueckwaerts gerechneten Lauf liefern —
    das waere Lookahead und wuerde das Ensemble kuenstlich gewinnen lassen.

    Wie bei der Previous-Runs-API auch hier Retry: leere Antworten bei
    identischen Parametern sind ein bekannter Backend-Cache-Effekt.
    """
    j = {}
    for attempt in range(retries):
        r = requests.get(ENS_API, params={
            "latitude": lat, "longitude": lon,
            "start_date": start, "end_date": end,
            "hourly": f"temperature_2m_previous_day{lead}",
            "models": ENS_MODEL, "timezone": "auto",
        }, timeout=90)
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1)); continue
        r.raise_for_status()
        j = r.json()
        h = j.get("hourly", {})
        if h.get("time") and any(v is not None for k in h if k != "time" for v in h[k]):
            break
        time.sleep(2 * (attempt + 1))
    h = j.get("hourly", {})
    times = h.get("time", [])
    mem_keys = [k for k in h if "member" in k]
    per_day = defaultdict(lambda: defaultdict(list))
    for k in mem_keys:
        for t, v in zip(times, h.get(k, [])):
            if v is not None:
                per_day[t[:10]][k].append(v)
    # Je Member das Tagesmaximum, dann die Member-Verteilung des Tages.
    out = {}
    for day, per_mem in per_day.items():
        vals = [max(vs) for vs in per_mem.values() if vs]
        if len(vals) >= 10:            # Teiltage mit zu wenigen Membern verwerfen
            out[day] = vals
    return out, j.get("timezone", "UTC")


# ---------------------------------------------------------------- Bias
def rolling_bias(hist, day, win=BIAS_WIN, min_n=BIAS_MIN):
    """Mittlerer (Ist - Prognose) der letzten `win` Tage VOR `day`.

    hist: {tag: (prognose, ist)}. Strikt kausal — der Zieltag selbst und alles
    danach bleibt draussen, sonst waere der ganze Vergleich wertlos.
    """
    prev = sorted(d for d in hist if d < day)[-win:]
    errs = [hist[d][1] - hist[d][0] for d in prev
            if hist[d][0] is not None and hist[d][1] is not None]
    if len(errs) < min_n:
        return None
    return mean(errs)


# ---------------------------------------------------------------- Buckets
def bucket_prob(mu, sigma, k):
    """P(Tagesmax faellt in den ganzzahligen Bucket k), Normalannahme."""
    sigma = max(sigma, SIGMA_FLOOR)
    return norm.cdf((k + 0.5 - mu) / sigma) - norm.cdf((k - 0.5 - mu) / sigma)

def fit_a_city(rows, b=SIGMA_B):
    """MLE fuer a_city bei FESTEM b (aus der Studie vom 14.07. uebernommen, hier
    NICHT neu gefittet — sonst waere b auf denselben Daten getunt)."""
    best, best_ll = SIGMA_FLOOR, -1e18
    a = 0.0
    while a <= 4.0:
        ll = 0.0
        for err, s in rows:
            sig = max(a + b * s, SIGMA_FLOOR)
            ll += -0.5 * (err / sig) ** 2 - math.log(sig)
        if ll > best_ll:
            best_ll, best = ll, a
        a += 0.05
    return best


# ---------------------------------------------------------------- Daten sammeln
def build(refetch=False):
    if CACHE.exists() and not refetch:
        with open(CACHE, "rb") as f:
            return pickle.load(f)

    # date-Objekte: fetch_actual_daily_extreme_wu chunkt selbst mit timedelta.
    # 2026-04-29 ist die harte Untergrenze des Ensemble-Endpoints.
    start = dt.date(2026, 4, 29)
    end   = dt.date.today() - dt.timedelta(days=1)
    data = {}
    for city, icao in STATIONS.items():
        ap = AP.get(icao)
        if not ap:
            print(f"  {city:14s} uebersprungen (ICAO {icao} unbekannt)")
            continue
        lat, lon = ap["lat"], ap["lon"]
        try:
            daily5, tz = fetch_model_daily_extreme(icao, lat, lon, start.isoformat(),
                                                   end.isoformat(), max, lead=LEAD)
            ens, _tz2  = fetch_ensemble_members(lat, lon, start.isoformat(),
                                                end.isoformat(), lead=LEAD)
            actual     = fetch_actual_daily_extreme_wu(icao, ap["country"], start, end, tz, max)
        except Exception as e:
            print(f"  {city:14s} FEHLER {type(e).__name__}: {e}")
            continue
        days = sorted(set(daily5.get("ensemble_mean", {})) & set(ens) & set(actual))
        rows = []
        for d in days:
            per_model = [daily5[m][d] for m in MODELS if d in daily5[m]]
            if len(per_model) < len(MODELS):
                continue
            mem = ens[d]
            rows.append({
                "day": d,
                "mu5": sum(per_model) / len(per_model),
                "spread": max(per_model) - min(per_model),
                "mu_ens_med": median(mem),
                "mu_ens_mean": mean(mem),
                "sd_ens": (sum((x - mean(mem)) ** 2 for x in mem) / max(len(mem) - 1, 1)) ** 0.5,
                "n_mem": len(mem),
                "actual": actual[d],
            })
        data[city] = rows
        print(f"  {city:14s} {len(rows):4d} Tage   (Member Median {median([r['n_mem'] for r in rows]) if rows else 0:.0f})")
        time.sleep(0.4)
    with open(CACHE, "wb") as f:
        pickle.dump(data, f)
    return data


def apply_bias(data):
    """Beide Schaetzer identisch biaskorrigieren — Fairness-Bedingung der Pre-Reg."""
    out = []
    for city, rows in data.items():
        rows = sorted(rows, key=lambda r: r["day"])
        h5   = {r["day"]: (r["mu5"], r["actual"]) for r in rows}
        hens = {r["day"]: (r["mu_ens_med"], r["actual"]) for r in rows}
        hem  = {r["day"]: (r["mu_ens_mean"], r["actual"]) for r in rows}
        for r in rows:
            b5, be, bm = (rolling_bias(h5, r["day"]), rolling_bias(hens, r["day"]),
                          rolling_bias(hem, r["day"]))
            if b5 is None or be is None:
                continue
            out.append({**r, "city": city,
                        "mu5_c": r["mu5"] + b5,
                        "muE_c": r["mu_ens_med"] + be,
                        "muEm_c": r["mu_ens_mean"] + (bm if bm is not None else be),
                        "month": r["day"][:7]})
    return out


def split(rows):
    return ([r for r in rows if r["month"] in IS_MONTHS],
            [r for r in rows if r["month"] in OOS_MONTHS])


# ---------------------------------------------------------------- G1 / G2
def gate12(rows, label, key="muE_c", cut=None):
    """Gepaarte MAE-Differenz je Stadt-Tag: positiv = Ensemble besser."""
    cut = SPREAD_CUT if cut is None else cut
    rough = [r for r in rows if r["spread"] > cut]
    if not rough:
        print(f"  {label:26s} keine zerklüfteten Tage")
        return None, None, 0
    d = [abs(r["mu5_c"] - r["actual"]) - abs(r[key] - r["actual"]) for r in rough]
    m, t = mean(d), tstat(d)
    print(f"  {label:26s} n={len(d):5d}  MAE5={mean([abs(r['mu5_c']-r['actual']) for r in rough]):.3f}"
          f"  MAEens={mean([abs(r[key]-r['actual']) for r in rough]):.3f}"
          f"  Delta={m:+.4f} K  t={t:+.2f}")
    return m, t, len(d)


# ---------------------------------------------------------------- G3
def gate3(is_rows, oos_rows):
    """Bin-weise Reliability in der Lay-Zone, getrennt nach Regime.

    AUSDRUECKLICH NICHT ueber die Lay-Zone gemittelt: das G3 vom 14.07. tat das
    und bestand dadurch sogar fuer das Modell, das den Beijing-Verlust erzeugte —
    dominiert von den vielen Buckets mit P ~ 0. Der Fehler wird hier nicht wiederholt.
    """
    per_city = defaultdict(list)
    for r in is_rows:
        per_city[r["city"]].append((r["actual"] - r["mu5_c"], r["spread"]))
    a_city = {c: fit_a_city(v) for c, v in per_city.items() if len(v) >= 20}

    def collect(rows, which):
        out = []
        for r in rows:
            a = a_city.get(r["city"])
            if a is None:
                continue
            if which == "status_quo":
                mu, sig = r["mu5_c"], max(a + SIGMA_B * r["spread"], SIGMA_FLOOR)
            else:
                mu, sig = r["muE_c"], max(r["sd_ens"], SIGMA_FLOOR)
            lo, hi = int(math.floor(mu - 4 * sig)), int(math.ceil(mu + 4 * sig))
            for k in range(lo, hi + 1):
                p = bucket_prob(mu, sig, k)
                if p <= LAY_ZONE:
                    out.append((p, 1 if round(r["actual"]) == k else 0,
                                r["spread"] > SPREAD_CUT))
        return out

    print("\n--- G3: bin-weise Reliability in der Lay-Zone (OOS) ---")
    print(f"  {'Modell':13s} {'Bin':>10s} {'Regime':>11s} {'n':>7s} {'vorherg.':>9s}"
          f" {'realis.':>9s} {'Faktor':>7s}")
    verdict = {}
    for which in ("status_quo", "ensemble"):
        rows = collect(oos_rows, which)
        ok_all = True
        for lo, hi in BINS:
            for regime, flag in (("zerklüftet", True), ("ruhig", False)):
                sel = [(p, y) for p, y, ro in rows if lo <= p < hi and ro == flag]
                if len(sel) < 50:
                    continue
                pred = mean([p for p, _ in sel]); real = mean([y for _, y in sel])
                fac = real / pred if pred > 0 else float("nan")
                good = BIN_BAND[0] <= fac <= BIN_BAND[1]
                ok_all &= good
                print(f"  {which:13s} {lo:.0%}-{hi:.0%}".ljust(26)
                      + f"{regime:>11s} {len(sel):7d} {pred:8.2%} {real:8.2%}"
                      f" {fac:6.2f}x  {'ok' if good else 'RAUS'}")
        verdict[which] = ok_all
    return verdict


# ---------------------------------------------------------------- G5
def gate5(rows, key="muE_c"):
    rough = [r for r in rows if r["spread"] > SPREAD_CUT]
    per_city = defaultdict(list)
    per_day = defaultdict(float)
    for r in rough:
        d = abs(r["mu5_c"] - r["actual"]) - abs(r[key] - r["actual"])
        per_city[r["city"]].append(d)
        per_day[r["day"]] += d
    eff = {c: mean(v) for c, v in per_city.items()}
    if not eff:
        return False
    med = median(list(eff.values()))
    best = max(eff, key=lambda c: eff[c])
    rest = [abs(r["mu5_c"] - r["actual"]) - abs(r[key] - r["actual"])
            for r in rough if r["city"] != best]
    tot = sum(per_day.values())
    conc = (max(per_day.values()) / tot) if tot > 0 and per_day else 1.0
    ok = med > 0 and bool(rest) and tstat(rest) > 1.5 and conc <= 0.30
    print(f"\n--- G5 Robustheit ---")
    print(f"  Median Stadt-Effekt {med:+.4f} K   ohne beste Stadt ({best}) t={tstat(rest):+.2f}"
          f"   groesster Tagesanteil {conc:.1%}")
    return ok


# ---------------------------------------------------------------- G4
def gate4(oos_rows, skip=False):
    """Ertrag der Lays, die der Veto sperrt und mu_ens neu oeffnen wuerde.

    Marktpreise werden GEZIELT nur fuer diese Kandidaten geladen — der Veto
    betrifft ein Drittel der Tage, davon ist nur die Lay-Zone relevant.
    """
    print("\n--- G4 Praxis: vom Veto gesperrte Tage ---")
    cand = [r for r in oos_rows if r["spread"] > SPREAD_CUT]
    print(f"  {len(cand)} gesperrte Stadt-Tage im OOS")
    if skip:
        print("  uebersprungen (--skip-g4)")
        return None
    print("  Marktpreise fuer diese Kandidaten werden geladen ...")
    print("  HINWEIS: G4 braucht die Bucket-Preise zum Handelszeitpunkt (Lead 1).")
    print("  Der Tape-Abruf ist in `weather_gfs_ensemble_mu_g4.py` ausgelagert,")
    print("  damit G1-G3 ohne Netzlast reproduzierbar bleiben.")
    return None


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true")
    ap.add_argument("--skip-g4", action="store_true")
    args = ap.parse_args()

    print("Daten holen (previous_day1, kein start_date-Lookahead) ...")
    data = build(refetch=args.refetch)
    rows = apply_bias(data)
    is_rows, oos_rows = split(rows)
    print(f"\nGesamt {len(rows)} Stadt-Tage nach Bias-Fenster   "
          f"IS {len(is_rows)}  OOS {len(oos_rows)}")
    print(f"davon zerklüftet (s > {SPREAD_CUT} K): "
          f"IS {sum(1 for r in is_rows if r['spread'] > SPREAD_CUT)}  "
          f"OOS {sum(1 for r in oos_rows if r['spread'] > SPREAD_CUT)}")

    print("\n--- G1 / G2: Genauigkeit der Mitte auf zerklüfteten Tagen ---")
    m1, t1, n1 = gate12(is_rows, "G1 In-Sample (Mai-Jun)")
    m2, t2, n2 = gate12(oos_rows, "G2 Out-of-Sample (Jul)")
    g1 = m1 is not None and m1 > 0 and t1 > 2.0
    g2 = g1 and m2 is not None and m2 > 0 and t2 > 1.5

    v3 = gate3(is_rows, oos_rows)
    g3 = bool(v3.get("ensemble"))
    g5 = gate5(oos_rows)
    g4 = gate4(oos_rows, skip=args.skip_g4)

    mark = lambda ok: "GRUEN" if ok else "ROT"
    print("\n=== GATES ===")
    print(f"  G1 IS  t>2,0 ................ {mark(g1)}")
    print(f"  G2 OOS t>1,5 ................ {mark(g2)}")
    print(f"  G3 bin-weise Kalibrierung ... {mark(g3)}"
          f"   (Status quo: {mark(v3.get('status_quo'))})")
    print(f"  G4 Ertrag gesperrter Tage ... {'offen' if g4 is None else mark(g4)}")
    print(f"  G5 Robustheit ............... {mark(g5)}")
    if not g1:
        print("\n  Abbruchregel der Pre-Reg: G1 gerissen -> falsifiziert,")
        print("  KEINE Umparametrisierung (kein Wechsel auf Mittel oder andere Schwelle).")

    print("\n--- Sensitivitaet (berichtet, KEIN Gate) ---")
    for lbl, key in (("Mittel statt Median", "muEm_c"),):
        gate12(is_rows, f"  IS  {lbl}", key)
        gate12(oos_rows, f"  OOS {lbl}", key)
    for cut in (2.0, 4.0):
        gate12(oos_rows, f"  OOS Schwelle s>{cut} K", cut=cut)
    return 0


if __name__ == "__main__":
    sys.exit(main())
