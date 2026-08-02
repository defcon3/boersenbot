#!/usr/bin/env python3
"""weather_anker_divergenz_eval.py — Auswertung zu
`preregs/weather_anker_divergenz_2026_08_02.md`.

FRAGE: Traegt die Kalibrierungs-Divergenz D = bias_700d - bias_40d Information
ueber den Ausgang eines -1-Lays — und zwar VORZEICHENRICHTIG?

DIE EINSEITIGE THESE: Gelayt wird der Bucket UNTER unserem Favoriten.
  D < 0  ->  unser mu ist waermer als die Sommersicht, der Favorit sitzt zu hoch,
             das Lay-Ziel rutscht in die Naehe des WAHREN Favoriten  ->  schlecht
  D > 0  ->  das Lay-Ziel rutscht weiter weg  ->  harmlos bis nuetzlich
Ein symmetrisches |D|-Gate bildet das nicht ab. Die Gegenprobe H3 entscheidet:
laeuft auch D > +0,7 schlechter, misst |D| nur "unsichere Stadt".

FENSTER R (12.07.-01.08.) traegt KEIN Gate. Zwei Extremstaedte sind bekannt
(Beijing D=-1,00/d=-1,09, Taipei D=+1,20/d=+1,45), beide vorzeichenkonform — wer
daraus einen Beleg macht, belegt seine eigene Vorauswahl. Der Beleg ist
FENSTER F ab 03.08.

D IST EINE KONSTANTE JE STADT. Effektiv gibt es ~31 unabhaengige Beobachtungen,
nicht 323. Deshalb laeuft der Haupttest ueber Stadt-Mittelwerte; wer positionsweise
rechnet, ist um etwa sqrt(10) zu optimistisch.

NICHT ANGEFASST: die 700d-Basis bleibt (40d im Lay-Buch falsifiziert am 28.07.,
Commit 593d99c), und die 40d-CSVs werden waehrend des Forward-Fensters nicht neu
gerechnet — sonst misst der Test zwei Dinge gleichzeitig.

Aufruf:
  python weather_anker_divergenz_eval.py               # Fenster F (Gates)
  python weather_anker_divergenz_eval.py --fenster R   # Bezifferung, kein Gate
"""

import argparse
import csv
import glob
import math
import os
import statistics
import sys
from collections import defaultdict

import pymssql

from weather_ladder_logger import DB_CONFIG
from weather_stations import favorit_k

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

USD, FEE = 5.0, 0.07
SCHWELLE = 0.7                      # aus der bestehenden Doktrin, NICHT gesucht
CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preregs")
FENSTER = {"R": ("2026-07-12", "2026-08-01"), "F": ("2026-08-03", "2099-12-31")}
MIN_TAGE, MIN_POS, MIN_STAEDTE, MIN_DIVERGENT = 30, 400, 20, 4


def lade_bias(vierzig):
    """bias je Stadt (ensemble_mean, max) — Vorrangregel wie load_calib()."""
    out = {}
    for path in sorted(glob.glob(os.path.join(CALIB_DIR, "weather_source_calib*.csv"))):
        name = os.path.basename(path)
        if ("calib40" in name) != vierzig or "_lead" in name or "_min_" in name:
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["model"] == "ensemble_mean":
                    out[row["city"]] = float(row["bias"])
    return out


def pnl_lay(no, verloren):
    n = USD / no
    fee = FEE * n * min(no, 1 - no)
    return (-USD - fee) if verloren else (n - USD - fee)


def breakeven_q(no):
    n = USD / no
    fee = FEE * n * min(no, 1 - no)
    g, v = n - USD - fee, USD + fee
    return g / (g + v)


def lade_kandidaten(von, bis):
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor(as_dict=True)
    cur.execute(
        "SELECT snapshot_utc, target_date, city, k, buy_no, offset_fav, settle_k "
        "FROM bb_WeatherLadders WHERE var='max' AND kind='eq' AND offset_fav=-1 "
        "AND buy_no IS NOT NULL AND target_date BETWEEN %s AND %s", (von, bis))
    roh = cur.fetchall()
    conn.close()

    neueste = {}
    for r in roh:
        if (r["target_date"] - r["snapshot_utc"].date()).days != 1:
            continue
        key = (str(r["target_date"]), r["city"])
        if key not in neueste or r["snapshot_utc"] > neueste[key]:
            neueste[key] = r["snapshot_utc"]

    posten = []
    for r in roh:
        key = (str(r["target_date"]), r["city"])
        if ((r["target_date"] - r["snapshot_utc"].date()).days != 1
                or r["snapshot_utc"] != neueste.get(key) or r["settle_k"] is None):
            continue
        k0 = int(r["k"]) + 1                      # offset_fav = -1  ->  k0 = k + 1
        no = float(r["buy_no"])
        d = int(r["settle_k"]) - k0
        posten.append({"tag": key[0], "city": r["city"], "no": no, "d": d,
                       "verloren": d == -1, "pnl": pnl_lay(no, d == -1),
                       "be": breakeven_q(no)})
    return sorted(posten, key=lambda p: (p["tag"], p["city"]))


def lade_leitern(von, bis):
    """Volle eq-Leiter je Stadt-Tag (Lead 1, neuester Snapshot) — fuer H5.

    Ohne die ganze Leiter laesst sich das korrigierte Lay-Ziel nicht bepreisen:
    verschiebt sich k0 um einen Bucket, wird ein ANDERER Markt gelayt, und dessen
    Preis steht nur da, weil der Ladder-Logger die komplette Leiter schreibt.
    """
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor(as_dict=True)
    cur.execute(
        "SELECT snapshot_utc, target_date, city, k, buy_no, offset_fav, mu_ens, "
        "settle_k FROM bb_WeatherLadders WHERE var='max' AND kind='eq' "
        "AND target_date BETWEEN %s AND %s", (von, bis))
    roh = cur.fetchall()
    conn.close()

    neueste = {}
    for r in roh:
        if (r["target_date"] - r["snapshot_utc"].date()).days != 1:
            continue
        key = (str(r["target_date"]), r["city"])
        if key not in neueste or r["snapshot_utc"] > neueste[key]:
            neueste[key] = r["snapshot_utc"]

    je = {}
    for r in roh:
        key = (str(r["target_date"]), r["city"])
        if ((r["target_date"] - r["snapshot_utc"].date()).days != 1
                or r["snapshot_utc"] != neueste.get(key)):
            continue
        L = je.setdefault(key, {"tag": key[0], "city": key[1], "k0": None,
                                "mu": None, "settle_k": None, "preise": {}})
        if r["mu_ens"] is not None:
            L["mu"] = float(r["mu_ens"])
        if r["settle_k"] is not None:
            L["settle_k"] = int(r["settle_k"])
        if r["offset_fav"] is not None:
            L["k0"] = int(r["k"]) - int(r["offset_fav"])
        if r["buy_no"]:
            L["preise"][int(r["k"])] = float(r["buy_no"])
    return je


def buch(leitern, D, korrigiert):
    """-1-Lay je Stadt-Tag, wahlweise mit korrigiertem Anker k0' = fav(mu + D).

    mu = ens_raw - bias, also mu_40d = mu_700d + D. Die Korrektur ist damit
    festgelegt und NICHT gefittet.
    """
    posten = []
    for L in leitern.values():
        if L["k0"] is None or L["settle_k"] is None or L["mu"] is None:
            continue
        d = D.get(L["city"])
        if d is None:
            continue
        k0 = favorit_k(L["mu"] + d, L["city"]) if korrigiert else L["k0"]
        ziel = k0 - 1
        no = L["preise"].get(ziel)
        if not no:
            continue
        verloren = (L["settle_k"] == ziel)
        posten.append({"tag": L["tag"], "city": L["city"], "no": no,
                       "d": L["settle_k"] - k0, "verloren": verloren,
                       "pnl": pnl_lay(no, verloren), "be": breakeven_q(no)})
    return posten


def kennzahl(menge):
    if not menge:
        return None
    p = sum(x["pnl"] for x in menge)
    v = sum(1 for x in menge if x["verloren"])
    return {"n": len(menge), "roi": 100 * p / (USD * len(menge)),
            "quote": 100 * v / len(menge),
            "be": 100 * statistics.mean(x["be"] for x in menge),
            "d": statistics.mean(x["d"] for x in menge)}


def gew_korrelation(paare):
    """Pearson ueber Staedte, gewichtet mit der Kandidatenzahl."""
    if len(paare) < 4:
        return float("nan"), float("nan"), len(paare)
    sw = sum(w for _, _, w in paare)
    mx = sum(w * x for x, _, w in paare) / sw
    my = sum(w * y for _, y, w in paare) / sw
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in paare)
    sxx = sum(w * (x - mx) ** 2 for x, _, w in paare)
    syy = sum(w * (y - my) ** 2 for _, y, w in paare)
    if sxx <= 0 or syy <= 0:
        return float("nan"), float("nan"), len(paare)
    r = sxy / math.sqrt(sxx * syy)
    n = len(paare)
    t = r * math.sqrt((n - 2) / (1 - r ** 2)) if abs(r) < 1 else float("inf")
    return r, t, n


def gepaartes_t_tage(a, b):
    """t ueber Tagesdifferenzen zweier Gruppen auf gemeinsamen Zieltagen."""
    def je_tag(m):
        e = defaultdict(list)
        for x in m:
            e[x["tag"]].append(x["pnl"])
        return {t: 100 * sum(v) / (USD * len(v)) for t, v in e.items()}
    ra, rb = je_tag(a), je_tag(b)
    tage = sorted(set(ra) & set(rb))
    diff = [ra[t] - rb[t] for t in tage]
    if len(diff) < 3:
        return float("nan"), len(diff), float("nan")
    m, sd = statistics.mean(diff), statistics.stdev(diff)
    if sd == 0:
        return float("inf"), len(diff), m
    return m / (sd / math.sqrt(len(diff))), len(diff), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fenster", choices=["R", "F"], default="F")
    a = ap.parse_args()
    von, bis = FENSTER[a.fenster]

    b700, b40 = lade_bias(False), lade_bias(True)
    D = {c: b700[c] - b40[c] for c in set(b700) & set(b40)}
    posten = [p for p in lade_kandidaten(von, bis) if p["city"] in D]
    tage = sorted({p["tag"] for p in posten})
    staedte = sorted({p["city"] for p in posten})

    print("=" * 78)
    print(f"ANKER-DIVERGENZ   Fenster {a.fenster}   {von} .. "
          f"{bis if a.fenster == 'R' else 'offen'}")
    print("=" * 78)
    if not posten:
        print("Noch keine gesettelten -1-Kandidaten in diesem Fenster.")
        return
    print(f"Kandidaten {len(posten)} | Zieltage {len(tage)} "
          f"({tage[0]} .. {tage[-1]}) | Staedte {len(staedte)}")
    if a.fenster == "R":
        print("\n*** FENSTER R TRAEGT KEIN GATE — Bezifferung. Zwei Extremstaedte")
        print("*** sind bekannt und vorzeichenkonform; der Beleg ist Fenster F.")

    neg = [p for p in posten if D[p["city"]] < -SCHWELLE]
    pos = [p for p in posten if D[p["city"]] > SCHWELLE]
    mitte = [p for p in posten if abs(D[p["city"]]) <= SCHWELLE]
    n_div = len({p["city"] for p in posten if abs(D[p["city"]]) > SCHWELLE})

    # ---------------------------------------------------------------- G0
    if a.fenster == "F":
        g0 = (len(tage) >= MIN_TAGE and len(posten) >= MIN_POS
              and len(staedte) >= MIN_STAEDTE and n_div >= MIN_DIVERGENT)
        print(f"\nG0  BASIS   Verlangt: >= {MIN_TAGE} Zieltage, >= {MIN_POS} "
              f"Kandidaten, >= {MIN_STAEDTE} Staedte, >= {MIN_DIVERGENT} divergente")
        print(f"  ->  {'BESTANDEN' if g0 else 'GERISSEN'}"
              + ("" if g0 else f"  (fehlen {max(0, MIN_TAGE-len(tage))} Zieltage, "
                               f"{max(0, MIN_POS-len(posten))} Kandidaten)"))
        if not g0:
            print("  Vor G0 wird nicht ausgewertet — so vorregistriert.")
            return

    # ---------------------------------------------------------------- H1
    print(f"\nG1  STETIG — sagt D die Trefferquote vorzeichenrichtig voraus?")
    je_stadt = defaultdict(list)
    for p in posten:
        je_stadt[p["city"]].append(p)
    paare = [(D[c], 100 * sum(1 for x in v if x["verloren"]) / len(v), len(v))
             for c, v in je_stadt.items() if len(v) >= 3]
    r, t1, n1 = gew_korrelation(paare)
    print(f"  gewichtete Korrelation D gegen Trefferquote ueber {n1} Staedte:")
    print(f"    r = {r:+.3f}   t = {t1:+.2f}")
    g1 = r <= -0.40 and abs(t1) > 2.0 and r < 0
    print(f"  Verlangt: r <= -0,40 UND |t| > 2,0 UND r negativ  ->  "
          f"{'BELEGT' if g1 else 'NICHT BELEGT'}")
    print("  (positives r widerlegt die These, es dreht sie nicht um)")

    # ---------------------------------------------------------- G2 / G3
    print(f"\nG2/G3  GRUPPEN — die einseitige These gegen die Gegenprobe")
    print(f"  {'Gruppe':<22}{'n':>5}{'Staedte':>9}{'Treffer':>10}{'be':>8}"
          f"{'ROI':>10}{'d-quer':>9}")
    for name, menge in (("D < -0,7 (mu zu warm)", neg),
                        ("|D| <= 0,7", mitte),
                        ("D > +0,7 (mu zu kalt)", pos)):
        k = kennzahl(menge)
        if k:
            ns = len({p["city"] for p in menge})
            print(f"  {name:<22}{k['n']:5d}{ns:9d}{k['quote']:9.1f} %"
                  f"{k['be']:7.1f} %{k['roi']:+9.2f} %{k['d']:+9.2f}")
        else:
            print(f"  {name:<22}    —")

    k_neg, k_mit, k_pos = kennzahl(neg), kennzahl(mitte), kennzahl(pos)
    if k_neg and k_mit:
        t2, n2, m2 = gepaartes_t_tage(neg, mitte)
        ab = k_mit["roi"] - k_neg["roi"]
        print(f"\n  G2: D < -0,7 liegt {ab:+.2f} pp unter der Mitte "
              f"(Tagesmittel {-m2:+.2f} pp, t = {-t2:+.2f} ueber {n2} Tage)")
        g2 = ab >= 6.0 and -t2 > 2.0
        print(f"      Verlangt: >= 6 pp UND t > 2,0  ->  "
              f"{'BELEGT' if g2 else 'NICHT BELEGT'}")
    if k_pos and k_mit:
        ab_p = k_mit["roi"] - k_pos["roi"]
        g3 = ab_p < 6.0
        print(f"  G3: D > +0,7 liegt {ab_p:+.2f} pp unter der Mitte  ->  "
              f"{'GEGENPROBE BESTANDEN' if g3 else 'GEGENPROBE GERISSEN'}")
        if not g3:
            print("      Beide Richtungen schlechter -> |D| misst 'unsichere Stadt',")
            print("      nicht die gerichtete Verschiebung. G1 und G2 gelten damit")
            print("      als NICHT belegt, unabhaengig von ihren eigenen Zahlen.")

    # ---------------------------------------------------------------- H4
    print(f"\nH4  VOLLSTAENDIGKEIT (diagnostisch, kein Gate)")
    dq = {c: statistics.mean(x["d"] for x in v)
          for c, v in je_stadt.items() if len(v) >= 3}
    if len(dq) >= 4:
        rest = {c: dq[c] - D[c] for c in dq}
        print(f"  sd der Stadt-Verschiebung d-quer : {statistics.pstdev(dq.values()):.3f} Bucket")
        print(f"  sd nach Abzug von D              : {statistics.pstdev(rest.values()):.3f} Bucket")
        print("  Sinkt die zweite Zahl deutlich, erklaert D die Verschiebung;")
        print("  bleibt sie gleich, ist D nur ein Begleiter.")

    # ---------------------------------------------------------------- G4
    if a.fenster == "F" and k_neg and k_mit:
        print(f"\nG4  ROBUSTHEIT")
        h = len(tage) // 2
        for lbl, tg in (("1. Haelfte", tage[:h]), ("2. Haelfte", tage[h:])):
            kn = kennzahl([p for p in neg if p["tag"] in tg])
            km = kennzahl([p for p in mitte if p["tag"] in tg])
            if kn and km:
                print(f"  {lbl}: D<-0,7 {kn['roi']:+.2f} %  gegen Mitte "
                      f"{km['roi']:+.2f} %  (Abstand {km['roi']-kn['roi']:+.2f} pp)")
        je_c = defaultdict(float)
        for p in neg:
            je_c[p["city"]] += p["pnl"]
        if je_c:
            schwach = min(je_c, key=je_c.get)
            ohne = kennzahl([p for p in neg if p["city"] != schwach])
            if ohne:
                print(f"  ohne die staerkste Einzelstadt ({schwach}): "
                      f"D<-0,7 ROI {ohne['roi']:+.2f} %")

    # ---------------------------------------------------------------- G5
    print(f"\nG5  VERWERTUNG — Ankerkorrektur k0' = fav(mu + D) statt Sperre")
    leitern = lade_leitern(von, bis)
    div_staedte = {c for c in D if abs(D[c]) > SCHWELLE}
    nur_div = {k: L for k, L in leitern.items() if L["city"] in div_staedte}
    heute, korr = buch(nur_div, D, False), buch(nur_div, D, True)
    k_h, k_k = kennzahl(heute), kennzahl(korr)
    if k_h and k_k:
        gewechselt = sum(1 for a_, b_ in zip(sorted(heute, key=lambda x: (x["tag"], x["city"])),
                                             sorted(korr, key=lambda x: (x["tag"], x["city"])))
                         if a_["no"] != b_["no"]) if len(heute) == len(korr) else None
        print(f"  Staedte mit |D| > {SCHWELLE}: {len(div_staedte)}   "
              f"Stadt-Tage: heute {k_h['n']}, korrigiert {k_k['n']}"
              + (f"   Lay-Ziel gewechselt: {gewechselt}" if gewechselt is not None else ""))
        print(f"    heute      : Treffer {k_h['quote']:5.1f} %  be {k_h['be']:5.1f} %  "
              f"ROI {k_h['roi']:+7.2f} %")
        print(f"    korrigiert : Treffer {k_k['quote']:5.1f} %  be {k_k['be']:5.1f} %  "
              f"ROI {k_k['roi']:+7.2f} %")
        t5, n5, m5 = gepaartes_t_tage(korr, heute)
        print(f"  Differenz {k_k['roi']-k_h['roi']:+.2f} pp   "
              f"(Tagesmittel {m5:+.2f} pp, t = {t5:+.2f} ueber {n5} Tage)")

        beidseitig_ok = True
        for lbl, sel in (("D < -0,7", lambda c: D[c] < -SCHWELLE),
                         ("D > +0,7", lambda c: D[c] > SCHWELLE)):
            hh = [p for p in heute if sel(p["city"])]
            kk = [p for p in korr if sel(p["city"])]
            a_h, a_k = kennzahl(hh), kennzahl(kk)
            if a_h and a_k:
                tt, _, _ = gepaartes_t_tage(kk, hh)
                schlechter = (a_k["roi"] - a_h["roi"]) < 0 and tt < -2.0
                if schlechter:
                    beidseitig_ok = False
                print(f"    {lbl}: heute {a_h['roi']:+7.2f} %  korrigiert "
                      f"{a_k['roi']:+7.2f} %  ({a_k['roi']-a_h['roi']:+.2f} pp, "
                      f"t = {tt:+.2f}){'  <- signifikant schlechter' if schlechter else ''}")
        if a.fenster == "F":
            g5 = (k_k["roi"] - k_h["roi"]) >= 4.0 and t5 > 2.0 and beidseitig_ok
            print(f"  Verlangt: >= 4 pp UND t > 2,0 UND in KEINER Richtung "
                  f"signifikant schlechter  ->  {'BELEGT' if g5 else 'NICHT BELEGT'}")
        else:
            print("  (Fenster R: kein Gate)")

    print("\n" + "=" * 78)
    if a.fenster == "R":
        print("Bezifferung, kein Beleg. Die Entscheidung faellt in Fenster F.")
    else:
        print("Aus dieser Messung folgt KEIN automatischer Einbau. Bestehen die")
        print("Gates, folgt ein Vorschlag: den Anker der divergenten Staedte um D")
        print("korrigieren, zuerst als Schattenbuch — mit dem Vorbehalt, dass die")
        print("Wirkung an der Aktualitaet der 40d-CSVs haengt.")
    print("KEINE Stadt faellt weg, in keinem Ausgang (Betreiber-Entscheidung 02.08.).")


if __name__ == "__main__":
    main()
