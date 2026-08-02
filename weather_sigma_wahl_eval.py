#!/usr/bin/env python3
"""weather_sigma_wahl_eval.py — Auswertung zu
`preregs/weather_sigma_wahl_2026_08_02.md`.

FRAGE: Welche der drei sigma-Definitionen ist am besten kalibriert?

  A  Ganzjahres-sigma  — Standardabweichung aller bias-korrigierten Residuen des
     IS-Jahres. So rechnen Ladder-Logger (700d) und Zellen-Eval heute. REFERENZ.
  B  gleitendes 40-Tage-sigma — aus den Residuen der 40 Tage VOR dem Zieltag.
  C  sigma(s) = a + b * Modellspanne — a,b aus dem IS-Jahr gefittet, angewendet
     auf die Tagesspanne. Seit 17.07. in den Screens, nicht im Logger.

WARUM UEBERHAUPT: zwei unabhaengige Messungen am 02.08. zeigen dasselbe Muster —
zu wenig Masse im Zentrum (30,5 % modelliert gegen 32,6 % real), zu viel an den
Raendern (9,3 % gegen 5,8 %). Das ist die Signatur einer zu breiten Verteilung,
nicht einer falschen Verteilungsform: waere die Form schuld, muesste das Zentrum
stimmen.

DER BIAS BLEIBT IN ALLEN DREI VARIANTEN IDENTISCH (gleitend 40 Tage). Variiert
wird ausschliesslich sigma — sonst misst der Test zwei Aenderungen gleichzeitig.

KEIN LOOK-AHEAD: die vorhandenen 40d-Kalibrier-CSVs sind auf ein FESTES
Sommerfenster (Stand 17.07.) gefittet und enthalten fuer Juli-Zieltage Daten aus
der Zukunft. Sie werden hier NICHT verwendet. B und C rechnen ausschliesslich aus
dem Cache, mit Tagen vor dem Zieltag.

BEWERTET WIRD der mittlere absolute Abstand des Kalibrierungsfaktors zu 1,0 —
nicht "kleiner ist besser". Eine Ueberschlagsrechnung sagt, dass die volle
40d-Reduktion ueberschiesst (Faktor ~0,43 statt der gemessenen 0,70); ein sigma
kann also auch zu KLEIN sein.

Aufruf:
  python weather_sigma_wahl_eval.py
"""

import math
import pickle
import statistics
import sys
from pathlib import Path

from scipy.stats import norm

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

CACHE = Path("g3_cell_selection_cache.pkl")
IS_FROM, IS_TO = "2024-08", "2025-07"
OOS_FROM, OOS_TO = "2025-08", "2026-07"
BIAS_WIN, BIAS_MIN = 40, 10
SIGMA_FLOOR = 0.3
BINS = [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20)]
TRAGEND = [(0.02, 0.05), (0.05, 0.10)]     # 10-20 % war im Zellenlauf leer
BAND = (0.75, 1.25)
MIN_ZELLEN, MIN_JE_BIN, MIN_TAGE = 50, 2000, 120
MIN_ZELL_N = 50                            # je Zelle und Bin fuer G3
G2_SCHWELLE, G3_ANTEIL, G4_MAX_PP = 0.10, 0.70, 3.0
VARIANTEN = ("A", "B", "C")


def monat(d):
    return d[:7]


def lade():
    """{(stadt, metrik): [(datum, mu_roh, spanne, ist), ...]} chronologisch."""
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    zellen = {}
    for stadt, metriken in cache.items():
        for metrik, tage in metriken.items():
            reihe = sorted((d, v[0], v[1], v[2]) for d, v in tage.items()
                           if v and v[0] is not None and v[2] is not None)
            if reihe:
                zellen[(stadt, metrik)] = reihe
    return zellen


def mit_bias(reihe):
    """Gleitender 40-Tage-Bias, nur aus Tagen VOR dem Zieltag. Identisch fuer
    alle drei Varianten — hier wird bewusst nichts variiert."""
    out = []
    for i, (d, mu, spanne, ist) in enumerate(reihe):
        vor = reihe[max(0, i - BIAS_WIN):i]
        if len(vor) < BIAS_MIN:
            continue
        bias = statistics.mean(v[1] - v[3] for v in vor)
        mu_k = mu - bias
        out.append((d, mu_k, spanne, ist, mu_k - ist))
    return out


def fit_ab(recs):
    """sigma(s) = a + b*Spanne, ueber Spannen-Quartile gefittet — robuster als
    eine Regression auf Residuenquadrate."""
    if len(recs) < 40:
        return None
    nach_s = sorted(recs, key=lambda r: r[2])
    q = len(nach_s) // 4
    punkte = []
    for i in range(4):
        teil = nach_s[i * q:(i + 1) * q] if i < 3 else nach_s[3 * q:]
        if len(teil) < 8:
            continue
        punkte.append((statistics.mean(t[2] for t in teil),
                       statistics.pstdev([t[4] for t in teil]), len(teil)))
    if len(punkte) < 3:
        return None
    sw = sum(p[2] for p in punkte)
    mx = sum(p[0] * p[2] for p in punkte) / sw
    my = sum(p[1] * p[2] for p in punkte) / sw
    den = sum(p[2] * (p[0] - mx) ** 2 for p in punkte)
    if den <= 0:
        return None
    b = sum(p[2] * (p[0] - mx) * (p[1] - my) for p in punkte) / den
    return my - b * mx, b


def sigma_wahl(variante, i_ges, alle, sigma_a, ab, spanne):
    if variante == "A":
        return sigma_a
    if variante == "B":
        vor = alle[max(0, i_ges - BIAS_WIN):i_ges]
        if len(vor) < BIAS_MIN:
            return sigma_a                     # Anlauf: Referenz statt Luecke
        return max(statistics.pstdev([v[4] for v in vor]), SIGMA_FLOOR)
    if ab is None:
        return sigma_a                         # ohne Fit faellt C auf A zurueck
    return max(ab[0] + ab[1] * spanne, SIGMA_FLOOR)


def bucket_probs(mu, sigma, ist):
    sigma = max(sigma, SIGMA_FLOOR)
    lo, hi = int(math.floor(mu - 4 * sigma)), int(math.ceil(mu + 4 * sigma))
    for k in range(lo, hi + 1):
        p = norm.cdf((k + 0.5 - mu) / sigma) - norm.cdf((k - 0.5 - mu) / sigma)
        yield k, p, (1 if round(ist) == k else 0)


def faktor(topf):
    """realisierte durch vorhergesagte Rate; None wenn zu duenn."""
    summe, n, hits = topf
    if not n:
        return None
    pred = summe / n
    return (hits / n) / pred if pred > 0 else None


def abstand_zu_eins(toepfe, mindest):
    """Mittlerer |Faktor - 1| ueber die tragenden Bins."""
    fs = [faktor(toepfe[b]) for b in TRAGEND if toepfe[b][1] >= mindest]
    fs = [f for f in fs if f is not None]
    return statistics.mean(abs(f - 1.0) for f in fs) if fs else None


def main():
    if not CACHE.exists():
        print("Cache fehlt — erst weather_cell_selection_eval.py laufen lassen.")
        return
    print("=" * 78)
    print("WELCHES SIGMA?   A Ganzjahr (Referenz) · B gleitend 40d · C a+b*Spanne")
    print("=" * 78)

    daten = {}
    for zelle, reihe in lade().items():
        recs = mit_bias(reihe)
        is_r = [r for r in recs if IS_FROM <= monat(r[0]) <= IS_TO]
        oos_r = [r for r in recs if OOS_FROM <= monat(r[0]) <= OOS_TO]
        if len(is_r) < MIN_TAGE or len(oos_r) < MIN_TAGE:
            continue
        daten[zelle] = {"is": is_r, "oos": oos_r,
                        "sigma_a": max(statistics.pstdev([r[4] for r in is_r]),
                                       SIGMA_FLOOR),
                        "ab": fit_ab(is_r)}
    print(f"Zellen mit >= {MIN_TAGE} Tagen in beiden Fenstern: {len(daten)}")
    ohne = sum(1 for v in daten.values() if v["ab"] is None)
    if ohne:
        print(f"  ohne a/b-Fit (fallen fuer C auf A zurueck): {ohne}")

    # -------------------------------------- EINE Schleife sammelt alles
    ges = {v: {b: [0.0, 0, 0] for b in BINS} for v in VARIANTEN}   # [Σp, n, Treffer]
    je_zelle = {v: {} for v in VARIANTEN}
    fav = {v: [0.0, 0, 0] for v in VARIANTEN}                      # [Σp_max, Treffer, n]
    for zelle, d in daten.items():
        alle = d["is"] + d["oos"]
        start = len(d["is"])
        for v in VARIANTEN:
            lokal = {b: [0.0, 0, 0] for b in BINS}
            for j, r in enumerate(d["oos"]):
                sig = sigma_wahl(v, start + j, alle, d["sigma_a"], d["ab"], r[2])
                best_p, best_hit = -1.0, 0
                for _k, p, hit in bucket_probs(r[1], sig, r[3]):
                    for b in BINS:
                        if b[0] <= p < b[1]:
                            for topf in (ges[v][b], lokal[b]):
                                topf[0] += p
                                topf[1] += 1
                                topf[2] += hit
                            break
                    if p > best_p:
                        best_p, best_hit = p, hit
                fav[v][0] += best_p
                fav[v][1] += best_hit
                fav[v][2] += 1
            je_zelle[v][zelle] = lokal

    # ---------------------------------------------------------------- G0
    g0 = (len(daten) >= MIN_ZELLEN
          and all(ges[v][b][1] >= MIN_JE_BIN for v in VARIANTEN for b in TRAGEND))
    print(f"\nG0  BASIS   >= {MIN_ZELLEN} Zellen, >= {MIN_JE_BIN} je tragendem Bin"
          f"  ->  {'BESTANDEN' if g0 else 'GERISSEN'}")
    if not g0:
        print("  Vor G0 wird nicht ausgewertet — so vorregistriert.")
        return

    # ---------------------------------------------------------- G1 / G2
    print(f"\nG1  KALIBRIERUNG (OOS)   Band [{BAND[0]:.2f}; {BAND[1]:.2f}]")
    abst = {}
    for v, name in (("A", "A Ganzjahr (Ref.)"), ("B", "B gleitend 40d"),
                    ("C", "C a+b*Spanne")):
        print(f"  {name}")
        for b in BINS:
            summe, n, hits = ges[v][b]
            if n < 200:
                print(f"    {b[0]:.0%}-{b[1]:.0%}  n={n:7d}  zu duenn, kein Urteil")
                continue
            pred, real = summe / n, hits / n
            f = real / pred
            marke = "ok" if BAND[0] <= f <= BAND[1] else "RAUS"
            zusatz = "" if b in TRAGEND else "  (kein Gate)"
            print(f"    {b[0]:.0%}-{b[1]:.0%}  n={n:7d}  vorherg. {pred:6.2%}  "
                  f"realis. {real:6.2%}  Faktor {f:5.2f}x  {marke}{zusatz}")
        abst[v] = abstand_zu_eins(ges[v], 200)
        print(f"    mittlerer Abstand zu 1,0: "
              f"{abst[v]:.3f}" if abst[v] is not None else "    kein Abstand")

    kandidaten = [v for v in ("B", "C") if abst.get(v) is not None]
    if not kandidaten or abst.get("A") is None:
        print("\n  Nicht auswertbar.")
        return
    beste = min(kandidaten, key=lambda v: abst[v])
    im_band = all(BAND[0] <= faktor(ges[beste][b]) <= BAND[1] for b in TRAGEND)
    g1 = im_band and abst[beste] < abst["A"]
    print(f"\n  Gewinner unter B/C: {beste}   Abstand {abst[beste]:.3f} "
          f"gegen A {abst['A']:.3f}")
    print(f"  G1: beide tragenden Bins im Band UND besser als A  ->  "
          f"{'BELEGT' if g1 else 'NICHT BELEGT'}")
    g2 = (abst["A"] - abst[beste]) >= G2_SCHWELLE
    print(f"  G2: Vorsprung >= {G2_SCHWELLE:.2f}  (ist "
          f"{abst['A'] - abst[beste]:+.3f})  ->  {'BELEGT' if g2 else 'NICHT BELEGT'}")

    # ---------------------------------------------------------------- G3
    print(f"\nG3  BREITE — ist {beste} auch JE ZELLE besser kalibriert als A?")
    besser, zaehlbar = 0, 0
    for zelle in daten:
        a_ab = abstand_zu_eins(je_zelle["A"][zelle], MIN_ZELL_N)
        b_ab = abstand_zu_eins(je_zelle[beste][zelle], MIN_ZELL_N)
        if a_ab is None or b_ab is None:
            continue
        zaehlbar += 1
        if b_ab < a_ab:
            besser += 1
    if zaehlbar:
        print(f"  {besser} von {zaehlbar} auswertbaren Zellen "
              f"({100*besser/zaehlbar:.0f} %)")
        g3 = besser / zaehlbar >= G3_ANTEIL
    else:
        print("  keine Zelle mit genug Beobachtungen je Bin")
        g3 = False
    print(f"  G3 verlangt: >= {G3_ANTEIL:.0%}  ->  "
          f"{'BELEGT' if g3 else 'NICHT BELEGT'}")

    # ---------------------------------------------------------------- G4
    print(f"\nG4  KEIN SCHADEN AM ZENTRUM — Favoriten-Bucket")
    for v in ("A", beste):
        p, h, n = fav[v]
        print(f"  {v}: vorhergesagt {p/n:6.2%}  realisiert {h/n:6.2%}  "
              f"({(h/n - p/n)*100:+.1f} pp)")
    p, h, n = fav[beste]
    abw = abs(h / n - p / n) * 100
    g4 = abw <= G4_MAX_PP
    print(f"  G4 verlangt: |Abweichung| <= {G4_MAX_PP:.0f} pp  (ist {abw:.1f})  ->  "
          f"{'BELEGT' if g4 else 'NICHT BELEGT'}")

    print("\n" + "=" * 78)
    mark = lambda ok: "GRUEN" if ok else "ROT"
    print(f"  G1 {mark(g1)} · G2 {mark(g2)} · G3 {mark(g3)} · G4 {mark(g4)}")
    print("Aus dieser Messung folgt ein VORSCHLAG, keine Aenderung. Die Screens")
    print("sind live; ein anderes sigma veraendert sofort, welche Kandidaten sie")
    print("zeigen. Am Autobuy aendert sich nichts — er verwendet sigma nicht.")


if __name__ == "__main__":
    main()
