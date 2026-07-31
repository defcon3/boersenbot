# -*- coding: utf-8 -*-
"""weather_cell_selection_eval.py — Auswertung zur Pre-Reg
`preregs/weather_cell_selection_2026_07_31.md` (Commit dcea366).

Frage: Ist Prognosegenauigkeit eine **stabile Eigenschaft einer Zelle**
(Stadt x Metrik) — oder Rauschen? Nur wenn sie stabil ist, darf man nach ihr
Maerkte auswaehlen. Das ist eine Frage ueber die *Rangliste als Ganzes*
(Spearman IS gegen OOS), nicht ueber eine einzelne Zelle. Der Anlasswert
(London-Minimum, MAE 0,44 K) stammt aus 16 angeschauten Zellen und darf
ausdruecklich **kein** Gate erfuellen.

Aufruf:
    python weather_cell_selection_eval.py            # nutzt Cache
    python weather_cell_selection_eval.py --refetch  # Cache neu aufbauen (~25 min)
"""
import argparse
import datetime as dt
import math
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr, norm

from weather_source_compare import (STATIONS, MODELS, fetch_model_daily_extreme,
                                    fetch_actual_daily_extreme_wu)
from weather_stations import station_info

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

# ---------------------------------------------------------------- Pre-Reg-Parameter
START      = dt.date(2024, 8, 1)
END        = dt.date(2026, 7, 23)
IS_FROM, IS_TO   = "2024-08", "2025-07"      # voller Jahreszyklus
OOS_FROM, OOS_TO = "2025-08", "2026-07"      # voller Jahreszyklus
BIAS_WIN   = 40
BIAS_MIN   = 10
MIN_DAYS   = 120          # Zelle faellt raus, wenn ein Fenster duenner ist
LAY_ZONE   = 0.10
BINS       = [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20)]
BIN_BAND   = (0.75, 1.25)
RHO_GATE   = 0.50
LAY_PRICE  = 0.90         # s. Hinweis bei gate4()
SIGMA_FLOOR = 0.3
SPREAD_CUT = 3.0
CACHE      = Path("g3_cell_selection_cache.pkl")
N_PERM     = 2000


def mean(xs):  return sum(xs) / len(xs) if xs else float("nan")
def median(xs):
    if not xs: return float("nan")
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
def stdev(xs):
    if len(xs) < 2: return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
def tstat(xs):
    if len(xs) < 2: return float("nan")
    s = stdev(xs)
    return mean(xs) / (s / math.sqrt(len(xs))) if s > 0 else 0.0


# ---------------------------------------------------------------- Daten
def build(refetch=False):
    """{(stadt, metrik): {tag: (mu5, spread, ist)}} — je Stadt gecacht."""
    cache = {}
    if CACHE.exists() and not refetch:
        with open(CACHE, "rb") as f:
            cache = pickle.load(f)
    todo = [c for c in STATIONS if c not in cache]
    if todo:
        print(f"Cache: {len(cache)} Staedte vorhanden, {len(todo)} zu holen ...")
    for city in todo:
        icao = STATIONS[city]
        try:
            info = station_info(icao)
        except Exception as e:
            print(f"  {city:14s} uebersprungen ({icao}: {type(e).__name__})"); continue
        entry = {}
        for metrik, agg in (("min", min), ("max", max)):
            try:
                d5, tz = fetch_model_daily_extreme(icao, info["lat"], info["lon"],
                                                   START.isoformat(), END.isoformat(),
                                                   agg, lead=1)
                ist = fetch_actual_daily_extreme_wu(icao, info["country"],
                                                    START, END, tz, agg)
            except Exception as e:
                print(f"  {city:14s} {metrik}: FEHLER {type(e).__name__}: {e}")
                continue
            rows = {}
            for day in sorted(set(d5.get("ensemble_mean", {})) & set(ist)):
                vals = [d5[m][day] for m in MODELS if day in d5.get(m, {})]
                if len(vals) < len(MODELS):
                    continue
                rows[day] = (sum(vals) / len(vals), max(vals) - min(vals), ist[day])
            entry[metrik] = rows
        cache[city] = entry
        print(f"  {city:14s} min {len(entry.get('min',{})):4d} Tage   "
              f"max {len(entry.get('max',{})):4d} Tage")
        with open(CACHE, "wb") as f:      # nach jeder Stadt sichern
            pickle.dump(cache, f)
        time.sleep(0.3)
    out = {}
    for city, entry in cache.items():
        for metrik, rows in (entry or {}).items():
            if rows:
                out[(city, metrik)] = rows
    return out


def biased(rows):
    """[(tag, fehler, spanne, ist, mu_korr)] mit strikt kausalem 40d-Bias."""
    days = sorted(rows)
    out = []
    for i, day in enumerate(days):
        prev = days[max(0, i - BIAS_WIN):i]
        errs = [rows[d][2] - rows[d][0] for d in prev]
        if len(errs) < BIAS_MIN:
            continue
        b = mean(errs)
        mu, spread, ist = rows[day]
        out.append((day, (mu + b) - ist, spread, ist, mu + b))
    return out


def window(recs, lo, hi):
    return [r for r in recs if lo <= r[0][:7] <= hi]


# ---------------------------------------------------------------- G1
def gate1(cells_is):
    maes = {c: mean([abs(r[1]) for r in v]) for c, v in cells_is.items()}
    lo_c = min(maes, key=lambda c: maes[c]); hi_c = max(maes, key=lambda c: maes[c])
    faktor = maes[hi_c] / maes[lo_c]
    # Permutationstest: Tage ueber die Zellen mischen, Spannweite der MAEs
    pool = [(c, abs(r[1])) for c, v in cells_is.items() for r in v]
    sizes = [(c, len(v)) for c, v in cells_is.items()]
    vals = [x for _c, x in pool]
    rng = random.Random(20260731)
    echt = max(maes.values()) - min(maes.values())
    hits = 0
    for _ in range(N_PERM):
        rng.shuffle(vals)
        i = 0; sim = []
        for _c, n in sizes:
            sim.append(mean(vals[i:i + n])); i += n
        if (max(sim) - min(sim)) >= echt:
            hits += 1
    p = (hits + 1) / (N_PERM + 1)
    print(f"\n--- G1 IS-Struktur ---")
    print(f"  beste Zelle  {lo_c[0]:14s} {lo_c[1]:3s}  MAE {maes[lo_c]:.3f} K")
    print(f"  schlechteste {hi_c[0]:14s} {hi_c[1]:3s}  MAE {maes[hi_c]:.3f} K")
    print(f"  Faktor {faktor:.2f}   Permutationstest p = {p:.4f}  (n={N_PERM})")
    return (faktor > 2.0 and p < 0.01), maes


# ---------------------------------------------------------------- G2
def gate2(maes_is, maes_oos):
    gem = sorted(set(maes_is) & set(maes_oos))
    a = [maes_is[c] for c in gem]; b = [maes_oos[c] for c in gem]
    rho, p = spearmanr(a, b)
    print(f"\n--- G2 OOS-Stabilitaet (Kern der These) ---")
    print(f"  {len(gem)} Zellen in beiden Fenstern")
    print(f"  Spearman rho = {rho:+.3f}   p = {p:.2e}   (gefordert rho > {RHO_GATE}, p < 0,01)")
    ra = {c: i for i, c in enumerate(sorted(gem, key=lambda c: maes_is[c]))}
    rb = {c: i for i, c in enumerate(sorted(gem, key=lambda c: maes_oos[c]))}
    print(f"\n  {'Zelle':22s} {'IS-MAE':>7s} {'Rang':>5s} {'OOS-MAE':>8s} {'Rang':>5s}")
    for c in sorted(gem, key=lambda c: maes_is[c])[:8]:
        print(f"  {c[0]+' '+c[1]:22s} {maes_is[c]:7.3f} {ra[c]:5d} {maes_oos[c]:8.3f} {rb[c]:5d}")
    print("  ...")
    for c in sorted(gem, key=lambda c: maes_is[c])[-3:]:
        print(f"  {c[0]+' '+c[1]:22s} {maes_is[c]:7.3f} {ra[c]:5d} {maes_oos[c]:8.3f} {rb[c]:5d}")
    return (rho > RHO_GATE and p < 0.01), rho, gem


# ---------------------------------------------------------------- Buckets
def bucket_probs(mu, sigma, ist):
    sigma = max(sigma, SIGMA_FLOOR)
    lo, hi = int(math.floor(mu - 4 * sigma)), int(math.ceil(mu + 4 * sigma))
    for k in range(lo, hi + 1):
        p = norm.cdf((k + 0.5 - mu) / sigma) - norm.cdf((k - 0.5 - mu) / sigma)
        yield k, p, (1 if round(ist) == k else 0)


def sigma_of(cell_recs):
    """Ein sigma je Zelle aus der IS-Fehlerstreuung — kein Freiheitsgrad je Tag."""
    return max(stdev([r[1] for r in cell_recs]), SIGMA_FLOOR)


# ---------------------------------------------------------------- G3
def gate3(cells_is, cells_oos, terzile):
    print(f"\n--- G3 Kalibrierung, BIN-WEISE (OOS) ---")
    print("  (nicht ueber die Lay-Zone gemittelt — das G3 vom 14.07. bestand so")
    print("   sogar fuer das Modell, das den Beijing-Verlust erzeugte)")
    sig = {c: sigma_of(cells_is[c]) for c in cells_is}
    res = {}
    for name, cs in terzile.items():
        rows = []
        for c in cs:
            if c not in cells_oos or c not in sig:
                continue
            for r in cells_oos[c]:
                for _k, p, hit in bucket_probs(r[4], sig[c], r[3]):
                    if p <= LAY_ZONE:
                        rows.append((p, hit))
        ok = True
        print(f"\n  {name}:")
        for lo, hi in BINS:
            sel = [(p, h) for p, h in rows if lo <= p < hi]
            if len(sel) < 100:
                print(f"    {lo:.0%}-{hi:.0%}  n={len(sel)} — zu duenn, kein Urteil"); continue
            pred = mean([p for p, _ in sel]); real = mean([h for _, h in sel])
            fac = real / pred if pred > 0 else float("nan")
            good = BIN_BAND[0] <= fac <= BIN_BAND[1]
            ok &= good
            print(f"    {lo:.0%}-{hi:.0%}  n={len(sel):6d}  vorherg. {pred:6.2%}  "
                  f"realis. {real:6.2%}  Faktor {fac:5.2f}x  {'ok' if good else 'RAUS'}")
        res[name] = ok
    return res


# ---------------------------------------------------------------- G4
def gate4(cells_oos, terzile, sig):
    """Lay-Ertrag in der Lay-Zone.

    ACHTUNG, Abweichung von der Pre-Reg: historische Marktpreise ueber zwei
    Jahre liegen nicht vor. Statt eines echten ROI wird ein **fester Lay-Preis**
    von 0,90 angenommen (Break-even also genau 10 %, die Grenze der Lay-Zone).
    Damit misst G4 nicht den Marktertrag, sondern die **Kalibrierungsqualitaet
    bei fairer Bepreisung** — die Groesse, die den Ertrag treibt. Das ist
    schwaecher als vorregistriert und wird als solches ausgewiesen.
    """
    print(f"\n--- G4 Lay-Ertrag (fester Preis {LAY_PRICE}, s. Hinweis im Code) ---")
    def buch(cs):
        n = 0; pnl = 0.0
        for c in cs:
            if c not in cells_oos or c not in sig: continue
            for r in cells_oos[c]:
                for _k, p, hit in bucket_probs(r[4], sig[c], r[3]):
                    if p <= LAY_ZONE:
                        n += 1
                        pnl += (1 - LAY_PRICE) if not hit else -LAY_PRICE
        return n, pnl
    alle = [c for c in cells_oos]
    n_a, p_a = buch(alle)
    n_t, p_t = buch(terzile["bestes Terzil"])
    roi_a = p_a / (n_a * LAY_PRICE) if n_a else float("nan")
    roi_t = p_t / (n_t * LAY_PRICE) if n_t else float("nan")
    anteil = n_t / n_a if n_a else 0
    print(f"  volles Buch      n={n_a:7d}  ROI {roi_a:+7.2%}")
    print(f"  bestes Terzil    n={n_t:7d}  ROI {roi_t:+7.2%}   Signalanteil {anteil:.1%}")
    print(f"  (gefordert: ROI besser UND Signalanteil >= 40 %)")
    return (roi_t > roi_a and anteil >= 0.40), roi_a, roi_t, anteil


# ---------------------------------------------------------------- G5
def gate5(maes_is, maes_oos, gem, rho):
    best = min(gem, key=lambda c: maes_is[c])
    rest = [c for c in gem if c != best]
    rho2, p2 = spearmanr([maes_is[c] for c in rest], [maes_oos[c] for c in rest])
    print(f"\n--- G5 Robustheit ---")
    print(f"  ohne beste Zelle ({best[0]} {best[1]}): rho = {rho2:+.3f}  p = {p2:.2e}")
    ok_metrik = True
    for metrik in ("min", "max"):
        sub = [c for c in gem if c[1] == metrik]
        if len(sub) < 6: continue
        r3, p3 = spearmanr([maes_is[c] for c in sub], [maes_oos[c] for c in sub])
        good = r3 > 0
        ok_metrik &= good
        print(f"  nur {metrik}: rho = {r3:+.3f}  p = {p3:.2e}  ({len(sub)} Zellen)  "
              f"{'ok' if good else 'RAUS'}")
    return (rho2 > RHO_GATE and p2 < 0.01 and ok_metrik)


# ---------------------------------------------------------------- Haerte
def haertetest(maes_is, terzile):
    print(f"\n--- Haertetest (deskriptiv, KEIN Gate) ---")
    print("  Liegen die dokumentierten Verlust-Zellen im schlechtesten Terzil?")
    for stadt, metrik in (("Munich", "max"), ("Beijing", "max")):
        c = (stadt, metrik)
        if c not in maes_is:
            print(f"    {stadt} {metrik}: nicht im Datensatz"); continue
        wo = next((n for n, cs in terzile.items() if c in cs), "?")
        print(f"    {stadt:10s} {metrik}  IS-MAE {maes_is[c]:.3f} K  ->  {wo}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()

    data = build(refetch=args.refetch)
    cells_is, cells_oos = {}, {}
    for c, rows in data.items():
        recs = biased(rows)
        a = window(recs, IS_FROM, IS_TO); b = window(recs, OOS_FROM, OOS_TO)
        if len(a) >= MIN_DAYS and len(b) >= MIN_DAYS:
            cells_is[c] = a; cells_oos[c] = b
    print(f"\nZellen mit >= {MIN_DAYS} Tagen in BEIDEN Fenstern: {len(cells_is)} "
          f"(von {len(data)} geladenen)")
    if len(cells_is) < 10:
        print("Zu wenige Zellen — Auswertung abgebrochen."); return 1

    g1, maes_is = gate1(cells_is)
    maes_oos = {c: mean([abs(r[1]) for r in v]) for c, v in cells_oos.items()}
    g2, rho, gem = gate2(maes_is, maes_oos)

    rang = sorted(gem, key=lambda c: maes_is[c])
    k = max(len(rang) // 3, 1)
    terzile = {"bestes Terzil": rang[:k], "mittleres Terzil": rang[k:2*k],
               "schlechtestes Terzil": rang[2*k:]}
    sig = {c: sigma_of(cells_is[c]) for c in cells_is}

    v3 = gate3(cells_is, cells_oos, terzile)
    g3 = bool(v3.get("bestes Terzil"))
    g4, roi_a, roi_t, anteil = gate4(cells_oos, terzile, sig)
    g5 = gate5(maes_is, maes_oos, gem, rho)
    haertetest(maes_is, terzile)

    mark = lambda ok: "GRUEN" if ok else "ROT"
    print("\n=== GATES ===")
    print(f"  G1 IS-Struktur ............... {mark(g1)}")
    print(f"  G2 OOS-Stabilitaet (rho>{RHO_GATE}) .. {mark(g2)}   rho = {rho:+.3f}")
    print(f"  G3 Kalibrierung bin-weise .... {mark(g3)}   "
          f"(schlechtestes Terzil: {mark(v3.get('schlechtestes Terzil'))})")
    print(f"  G4 Ertrag + Breite ........... {mark(g4)}   Signalanteil {anteil:.1%}")
    print(f"  G5 Robustheit ................ {mark(g5)}")
    if not g2:
        print("\n  Abbruchregel: G2 gerissen -> Genauigkeit ist keine handelbare")
        print("  Zelleneigenschaft. KEIN Ausweichen auf RMSE, Trefferquote o. ae.")
    elif not g4:
        print("\n  G2 haelt, G4 reisst -> Filter als GEWICHTUNG statt Sperre")
        print("  weiterverfolgen, in eigener Pre-Reg.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
