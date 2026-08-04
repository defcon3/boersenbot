#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""weather_konvektiv_sigma_eval.py — Auswertung zu
`preregs/weather_konvektiv_sigma_2026_08_03.md`.

FRAGE: Traegt die vortags erkennbare Konvektionslage Information ueber die
Streuung, die INNERHALB GLEICHER MODELLSPANNE noch nicht enthalten ist?

sigma ist seit dem 14.07. spannen-konditioniert. Ein Konvektionseffekt, der nur
ueber die Spanne wirkt, ist deshalb kein Fund, sondern eine Umbenennung — genau
darum wird alles innerhalb von Spannen-Terzilen gemessen.

ZIELGROESSE je Stadttag: z = (actual - mu_ens) / sigma_ens, mit mu_ens und
sigma_ens wie im Live-Screen (robust_mean nach OUTLIER_DEG, minus Bias;
sigma(s) = max(a_city + b*s, SIGMA_FLOOR)). Ist sigma richtig, gilt sd(z) = 1.
Bekannt ist sd(z) < 1 global — sigma ist zu gross.

H: sd(z|konvektiv) / sd(z|klar) > 1.

DIE BEIDEN FALLEN AUS DER PRE-REG, UND WIE SIE ENTSCHAERFT SIND:

  1. SAISON. Konvektionstage sind ueberwiegend Sommertage; wer ueber das ganze
     Jahr vergleicht, misst den Jahresgang von sigma. Deshalb steht in G5 die
     Paarung innerhalb Stadt-Monat, und sie ist keine Zugabe, sondern die
     Bedingung dafuer, dass G1 ueberhaupt etwas bedeutet.
  2. ZIRKULARITAET. Wer z mit einem sigma bildet, das auf DENSELBEN Tagen
     gefittet wurde, unterschaetzt die Streuung. Deshalb werden a_city, b und
     der Bias AUSSCHLIESSLICH im IS-Fenster gefittet und im OOS eingefroren.
     Die ausgelieferten Kalibrier-CSVs werden NICHT benutzt — sie sind auf
     Fenster gefittet, die das OOS ueberlappen.

UMSETZUNGS-FESTLEGUNGEN (vor dem ersten Blick auf die Zielgroesse):

  a) Spannen-Terzile GLOBAL aus der IS-Spannenverteilung, im OOS eingefroren.
     Je Terzil ein Verhaeltnis r_t; das Gesamt-r ist deren mit n_konvektiv
     gewichtetes Mittel — die konvektive Gruppe ist die knappe, sie bestimmt,
     wie belastbar ein Terzil ist.
  b) G4 zaehlt Buckets in einem FESTEN Fenster mu +- 4 K. Ein sigma-abhaengiges
     Fenster (etwa +-4 sigma) wuerde die Zellenzahl selbst von sigma abhaengig
     machen und den Gate-Vergleich zirkulaer.
  c) sigma-Faktoren des konditionalen Modells = sd(z_IS) je Gruppe, also genau
     die Korrektur, die das IS verlangt. Zwei Faktoren, keine freie Form.

ZWEI STELLEN, AN DENEN DIE PRE-REG UNSCHARF IST — hier offengelegt, nicht
stillschweigend ausgelegt (Pflichtuebung 02.08.):

  * G4 verlangt, dass "die realisierte Trefferquote nicht UNTER Break-even
    22,6 % faellt". Im Lay-Buch ist eine NIEDRIGE Trefferquote das Gute: der
    gelegte Bucket soll gerade nicht eintreten, und 22,6 % ist die Quote, ab der
    das Lay verliert. Woertlich gelesen wuerde das Gate also Erfolg bestrafen.
    Gewertet wird die inhaltliche Fassung: die Quote muss BEI ODER UNTER 22,6 %
    bleiben.
  * "Zusaetzlich handelbare Zellen" ist die Zaehlgroesse, aber die Trefferquote
    ueber ALLE handelbaren Zellen ist keine sinnvolle Kontrolle — je Stadttag
    tritt genau ein Bucket ein, bei ~6 Zellen liegt sie also strukturell bei
    ~15 % und besteht das Gate immer. Gewertet wird deshalb die Quote der NEU
    geoeffneten Zellen; das ist die Frage, die G4 stellt. Die Quote ueber alle
    Zellen wird zusaetzlich berichtet.

Aufruf:
  python weather_konvektiv_sigma_eval.py
"""
import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm

from weather_outlier_screen import (MAX_PMODEL, OUTLIER_DEG, SIGMA_FLOOR,
                                    robust_mean)
from weather_source_compare import _fit_sigma_model

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

DUMP = "preregs/weather_konvektiv_sigma_residuen_2026_08_03.csv.gz"
BEDINGUNG = "preregs/weather_konvektiv_sigma_bedingung_2026_08_04.csv.gz"
BASE5 = ["gfs_seamless", "icon_seamless", "ukmo_seamless", "jma_seamless",
         "ecmwf_ifs025"]
IS_VON, IS_BIS = "2024-09", "2025-12"
OOS_VON, OOS_BIS = "2026-01", "2026-08"
REGEN_MIN, WOLKEN_MIN = 1.0, 60.0

G1_R, G2_R = 1.15, 1.10
G4_MEHR, G4_BREAKEVEN = 0.10, 0.226
G5_ANTEIL = 0.80
FENSTER_K = 4.0          # Bucket-Fenster mu +- 4 K fuer G4
MIN_JE_GRUPPE = 30       # je Terzil und Gruppe, sonst kein Verhaeltnis


def monat(d):
    return d[:7]


def baue_tage():
    """Ein Stadttag = (city, date, mu_roh, spread, actual, konvektiv)."""
    df = pd.read_csv(DUMP)
    bed = pd.read_csv(BEDINGUNG)
    bed["konv"] = ((bed.regen_mm >= REGEN_MIN)
                   & (bed.wolken_tag >= WOLKEN_MIN))
    kmap = {(r.city, r.date): bool(r.konv) for r in bed.itertuples()}

    fc = df.pivot_table(index=["city", "date"], columns="model",
                        values="forecast")
    ist = df.groupby(["city", "date"]).actual.first()
    tage = []
    fehlt = 0
    for (city, datum), reihe in fc.iterrows():
        if reihe[BASE5].isna().any():
            continue
        if (city, datum) not in kmap:
            fehlt += 1
            continue
        roh = {m: float(reihe[m]) for m in BASE5}
        mu, _ = robust_mean(roh)
        werte = list(roh.values())
        tage.append({"city": city, "date": datum, "mu": mu,
                     "spread": max(werte) - min(werte),
                     "actual": float(ist[(city, datum)]),
                     "konv": kmap[(city, datum)]})
    return tage, fehlt


def fit_is(tage):
    """Bias je Stadt und sigma(s) = a_city + b*s, AUSSCHLIESSLICH aus dem
    IS-Fenster. Rueckgabe: (bias, a, b)."""
    is_tage = [t for t in tage if IS_VON <= monat(t["date"]) <= IS_BIS]
    bias = {}
    for city in {t["city"] for t in is_tage}:
        g = [t["mu"] - t["actual"] for t in is_tage if t["city"] == city]
        bias[city] = float(np.mean(g))
    paare = defaultdict(list)
    for t in is_tage:
        paare[t["city"]].append((t["mu"] - bias[t["city"]] - t["actual"],
                                 t["spread"]))
    b, a = _fit_sigma_model(paare)
    return bias, a, b


def sd(werte):
    return float(np.std(np.asarray(werte, float), ddof=1)) if len(werte) > 1 \
        else float("nan")


def verhaeltnis(zs, grenzen, label=""):
    """Gewichtetes r = sd(z|konv)/sd(z|klar) INNERHALB der Spannen-Terzile.
    zs: [(z, spread, konv)]. Liefert (r, zeilen)."""
    terzile = defaultdict(lambda: {"k": [], "c": []})
    for z, s, konv in zs:
        t = 0 if s < grenzen[0] else (1 if s < grenzen[1] else 2)
        terzile[t]["k" if konv else "c"].append(z)
    zaehler, nenner, zeilen = 0.0, 0.0, []
    for t in sorted(terzile):
        k, c = terzile[t]["k"], terzile[t]["c"]
        if len(k) < MIN_JE_GRUPPE or len(c) < MIN_JE_GRUPPE:
            zeilen.append((t, len(k), len(c), float("nan"), float("nan"),
                           float("nan")))
            continue
        sk, sc = sd(k), sd(c)
        r = sk / sc
        zeilen.append((t, len(k), len(c), sk, sc, r))
        zaehler += r * len(k)
        nenner += len(k)
    return (zaehler / nenner if nenner else float("nan")), zeilen


def zelle_zaehlen(mu, sigma, actual):
    """(handelbare Buckets, getroffener Bucket) im festen Fenster mu +- 4 K.
    Handelbar = Modell gibt dem Bucket weniger als MAX_PMODEL."""
    lo, hi = math.floor(mu - FENSTER_K + 0.5), math.ceil(mu + FENSTER_K - 0.5)
    treffer = math.floor(actual + 0.5)
    handelbar = set()
    for k in range(lo, hi + 1):
        p = norm.cdf((k + 0.5 - mu) / sigma) - norm.cdf((k - 0.5 - mu) / sigma)
        if p < MAX_PMODEL:
            handelbar.add(k)
    return handelbar, treffer


def main():
    tage, fehlt = baue_tage()
    print("=" * 96)
    print("KONDITIONALES SIGMA — traegt die Konvektion mehr als die Modellspanne?")
    print("=" * 96)
    print(f"Stadttage mit Modell UND Bedingung: {len(tage)}"
          + (f"  (ohne Bedingung verworfen: {fehlt})" if fehlt else ""))

    bias, a, b = fit_is(tage)
    print(f"IS-Fit ({IS_VON}..{IS_BIS}): b = {b:+.4f}, a je Stadt "
          f"{min(a.values()):.2f}..{max(a.values()):.2f}, {len(a)} Staedte")

    for t in tage:
        if t["city"] not in a:
            t["z"] = None
            continue
        sig = max(a[t["city"]] + b * t["spread"], SIGMA_FLOOR)
        t["sigma"] = sig
        t["mu_k"] = t["mu"] - bias[t["city"]]
        t["z"] = (t["actual"] - t["mu_k"]) / sig
    tage = [t for t in tage if t.get("z") is not None]

    is_t = [t for t in tage if IS_VON <= monat(t["date"]) <= IS_BIS]
    oos_t = [t for t in tage if OOS_VON <= monat(t["date"]) <= OOS_BIS]
    nk_is = sum(1 for t in is_t if t["konv"])
    nk_oos = sum(1 for t in oos_t if t["konv"])
    print(f"IS  {len(is_t):>6} Stadttage, davon konvektiv {nk_is} "
          f"({nk_is/max(len(is_t),1):.1%})")
    print(f"OOS {len(oos_t):>6} Stadttage, davon konvektiv {nk_oos} "
          f"({nk_oos/max(len(oos_t),1):.1%})")
    print(f"sd(z) gesamt IS {sd([t['z'] for t in is_t]):.3f} · "
          f"OOS {sd([t['z'] for t in oos_t]):.3f}   (1,0 waere richtig kalibriert)")

    # Terzilgrenzen aus der IS-Spannenverteilung, danach eingefroren
    sp = np.array([t["spread"] for t in is_t])
    grenzen = (float(np.quantile(sp, 1 / 3)), float(np.quantile(sp, 2 / 3)))
    print(f"Spannen-Terzile (IS, eingefroren): < {grenzen[0]:.2f} K · "
          f"< {grenzen[1]:.2f} K · darueber")

    # ------------------------------------------------------------------ G1/G2
    ergebnis = {}
    for name, menge, gate in (("G1  IS ", is_t, G1_R), ("G2  OOS", oos_t, G2_R)):
        zs = [(t["z"], t["spread"], t["konv"]) for t in menge]
        r, zeilen = verhaeltnis(zs, grenzen)
        print(f"\n{name}   r = sd(z|konvektiv) / sd(z|klar), innerhalb der Terzile")
        print(f"   {'Terzil':<8}{'n konv':>8}{'n klar':>8}{'sd konv':>10}"
              f"{'sd klar':>10}{'r':>8}")
        for t, nk, nc, sk, sc, rt in zeilen:
            if math.isnan(rt):
                print(f"   {t+1:<8}{nk:>8}{nc:>8}     zu duenn (< "
                      f"{MIN_JE_GRUPPE} je Gruppe)")
            else:
                print(f"   {t+1:<8}{nk:>8}{nc:>8}{sk:>10.3f}{sc:>10.3f}{rt:>8.3f}")
        ok = r >= gate
        ergebnis[name.split()[0]] = ok
        print(f"   gewichtet r = {r:.3f}   verlangt >= {gate:.2f}  ->  "
              f"{'BELEGT' if ok else 'NICHT BELEGT'}")

    # -------------------------------------------------------------------- G3
    print(f"\nG3  NUTZEN GEGEN DIE FAIRE REFERENZ (OOS)")
    f_konv = sd([t["z"] for t in is_t if t["konv"]])
    f_klar = sd([t["z"] for t in is_t if not t["konv"]])
    f_glob = sd([t["z"] for t in is_t])
    print(f"   IS-Faktoren: konvektiv {f_konv:.3f} · klar {f_klar:.3f} · "
          f"global {f_glob:.3f}")

    def bewerte(skalierung):
        """|sd(z)-1| je Gruppe + Abdeckung des 80-%-Intervalls, OOS."""
        raus = {}
        for grp, menge in (("konvektiv", [t for t in oos_t if t["konv"]]),
                           ("klar", [t for t in oos_t if not t["konv"]])):
            zz = [t["z"] / skalierung(t) for t in menge]
            deckung = np.mean([abs(z) <= norm.ppf(0.9) for z in zz])
            raus[grp] = (sd(zz), abs(sd(zz) - 1.0), float(deckung))
        return raus

    varianten = {
        "heutiges sigma": lambda t: 1.0,
        "globaler Faktor": lambda t: f_glob,
        "konditional": lambda t: f_konv if t["konv"] else f_klar,
    }
    print(f"   {'Variante':<18}{'sd konv':>9}{'sd klar':>9}"
          f"{'Summe |sd-1|':>14}{'Deckung konv':>14}{'Deckung klar':>14}")
    punkte = {}
    for name, f in varianten.items():
        r = bewerte(f)
        summe = r["konvektiv"][1] + r["klar"][1]
        d_ab = (abs(r["konvektiv"][2] - 0.8) + abs(r["klar"][2] - 0.8))
        punkte[name] = (summe, d_ab)
        print(f"   {name:<18}{r['konvektiv'][0]:>9.3f}{r['klar'][0]:>9.3f}"
              f"{summe:>14.3f}{r['konvektiv'][2]:>13.1%}{r['klar'][2]:>14.1%}")
    g3 = (punkte["konditional"][0] < punkte["heutiges sigma"][0]
          and punkte["konditional"][0] < punkte["globaler Faktor"][0]
          and punkte["konditional"][1] < punkte["globaler Faktor"][1])
    print(f"   verlangt: besser als BEIDE Referenzen (|sd-1| und 80-%-Deckung)"
          f"  ->  {'BELEGT' if g3 else 'NICHT BELEGT'}")

    # -------------------------------------------------------------------- G4
    print(f"\nG4  HANDELSNUTZEN an KLAREN Tagen (OOS, P_modell < {MAX_PMODEL})")
    klar_oos = [t for t in oos_t if not t["konv"]]
    alt_n, neu_n, neu_treffer, alt_treffer, neu_zellen = 0, 0, 0, 0, 0
    for t in klar_oos:
        h_alt, hit = zelle_zaehlen(t["mu_k"], t["sigma"], t["actual"])
        h_neu, _ = zelle_zaehlen(t["mu_k"], t["sigma"] * f_klar, t["actual"])
        alt_n += len(h_alt)
        neu_n += len(h_neu)
        alt_treffer += 1 if hit in h_alt else 0
        neu_treffer += 1 if hit in h_neu else 0
        zusatz = h_neu - h_alt
        neu_zellen += len(zusatz)
        t["_zusatz_hit"] = 1 if hit in zusatz else 0
    mehr = (neu_n - alt_n) / alt_n if alt_n else float("nan")
    zusatz_hits = sum(t.get("_zusatz_hit", 0) for t in klar_oos)
    q_zusatz = zusatz_hits / neu_zellen if neu_zellen else float("nan")
    print(f"   klare Stadttage OOS: {len(klar_oos)}")
    print(f"   handelbare Zellen: heute {alt_n} -> konditional {neu_n} "
          f"({mehr:+.1%})")
    print(f"   davon NEU geoeffnet: {neu_zellen}, davon getroffen "
          f"{zusatz_hits} = {q_zusatz:.1%}")
    print(f"   Trefferquote ueber ALLE Zellen (nur Bericht): heute "
          f"{alt_treffer/max(alt_n,1):.1%} -> konditional "
          f"{neu_treffer/max(neu_n,1):.1%}")
    g4 = mehr >= G4_MEHR and (math.isnan(q_zusatz) or q_zusatz <= G4_BREAKEVEN)
    print(f"   verlangt >= {G4_MEHR:.0%} mehr Zellen UND neue Zellen bei oder "
          f"unter {G4_BREAKEVEN:.1%}  ->  {'BELEGT' if g4 else 'NICHT BELEGT'}")

    # -------------------------------------------------------------------- G5
    print(f"\nG5  ROBUSTHEIT")
    staedte = sorted({t["city"] for t in oos_t})
    haelt = 0
    zaehlbar = 0
    for c in staedte:
        zs = [(t["z"], t["spread"], t["konv"]) for t in oos_t if t["city"] != c]
        r, _ = verhaeltnis(zs, grenzen)
        if not math.isnan(r):
            zaehlbar += 1
            haelt += 1 if r > 1.0 else 0
    anteil = haelt / zaehlbar if zaehlbar else float("nan")
    print(f"   leave-one-city-out: Vorzeichen haelt in {haelt}/{zaehlbar} "
          f"({anteil:.0%}), verlangt >= {G5_ANTEIL:.0%}")

    paare, plus = [], 0
    je_sm = defaultdict(lambda: {"k": [], "c": []})
    for t in tage:
        je_sm[(t["city"], monat(t["date"]))]["k" if t["konv"] else "c"].append(t["z"])
    for schluessel, g in je_sm.items():
        if len(g["k"]) >= 5 and len(g["c"]) >= 5:
            paare.append(sd(g["k"]) - sd(g["c"]))
            plus += 1 if sd(g["k"]) > sd(g["c"]) else 0
    if paare:
        m = float(np.mean(paare))
        t_stat = m / (np.std(paare, ddof=1) / math.sqrt(len(paare)))
        print(f"   gepaart innerhalb Stadt-Monat: {len(paare)} Paare, "
              f"sd(konv)-sd(klar) = {m:+.3f}, t = {t_stat:+.2f}, "
              f"{plus}/{len(paare)} positiv ({plus/len(paare):.0%})")
        g5 = anteil >= G5_ANTEIL and m > 0 and t_stat > 2.0
    else:
        print("   gepaart innerhalb Stadt-Monat: keine Paare mit >= 5 je Gruppe")
        g5 = False
    print(f"   G5  ->  {'BELEGT' if g5 else 'NICHT BELEGT'}")

    # ---------------------------------------------------------------- Bilanz
    print("\n" + "=" * 96)
    mark = lambda ok: "GRUEN" if ok else "ROT"
    print(f"  G1 {mark(ergebnis['G1'])} · G2 {mark(ergebnis['G2'])} · "
          f"G3 {mark(g3)} · G4 {mark(g4)} · G5 {mark(g5)}")
    print("PASS hiesse: sigma DARF konditional gesetzt werden — nicht, dass es")
    print("gesetzt wird. Vor jeder Live-Aenderung laeuft ein Forward-Test.")
    print("FAIL hiesse: die Modellspanne enthaelt die Konvektionsinformation")
    print("bereits, und der 16,7-%-Nebenbefund war ein Kleinserien-Artefakt.")
    print("=" * 96)


if __name__ == "__main__":
    main()
