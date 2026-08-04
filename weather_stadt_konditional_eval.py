#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""weather_stadt_konditional_eval.py — Auswertung zu
`preregs/weather_stadt_konditional_2026_08_03.md`.

FRAGE: Trennt ein mechanisches, vorab festgelegtes Kriterium die Staedte, in
denen Modellgewichtung hilft, von denen, in denen sie schadet? Am 18.07. ist die
PAUSCHALE Gewichtung knapp gescheitert — gepoolt +5,63 %, Stadt-t 1,91 (Gate > 2).
Der Nutzen sass konsistent in Staedten mit grossem Modell-Guete-Gefaelle.

DAS KRITERIUM (rangbasiert, damit es keine Schwelle zu drehen gibt):

    G_Stadt = sd(ENS-Fehler, gleichgewichtet) / min_m sd(Fehler Modell m)

G > 1 heisst: das gleichgewichtete Ensemble ist schlechter als das beste
Einzelmodell — dort hat Umgewichtung Luft. Auswahl = oberstes Quartil nach G,
geschaetzt AUSSCHLIESSLICH aus Tagen vor dem Bewertungsmonat.

UMSETZUNGS-FESTLEGUNGEN (vor dem ersten Blick auf die Zielgroesse entschieden):

  1. ZIELGROESSE = mittlerer absoluter debiaster Ensemble-Fehler (MAE), exakt
     wie am 18.07. Die Pre-Reg nennt sie "sigma-Reduktion"; die Gegenrechnung in
     G1 (Seoul +43,1 · Muenchen +19,5 · Tel Aviv +18,4 · Wuhan +12,7 ·
     Jeddah +10,3) stammt aus der MAE-Spalte jenes Laufs. Es MUSS dieselbe
     Groesse sein, sonst ist die Gegenrechnung wertlos. Die echte
     Streuungsreduktion (sd) wird als zweite Spalte mitberichtet, ohne Gate.
  2. ANLAUF = 180 Serien-Tage je Stadt (wie 18.07.), bewertet ab Serien-Tag 181.
  3. REFIT an KALENDER-Monatsgrenzen, aus allen Tagen strikt VOR dem Monat.
     Am 18.07. lief der Refit alle 30 Serien-Tage; hier verlangt die Pre-Reg
     "aus den Daten vor dem jeweiligen Bewertungsmonat", und die Auswahl ist
     ohnehin monatlich. Beide Arme teilen denselben Refit — die Aenderung
     betrifft Gewichtung UND Referenz gleich.
  4. QUARTIL = round(n_Staedte / 4) Staedte je Monat, also 7 von 29. So steht es
     in der Pre-Reg ("waehlt immer ~7 von 29").
  5. Gewichte wie 18.07.: Gleichkorrelations-Kovarianz, negativ geklippt,
     renormiert.

REPRODUKTIONSPROBE ohne Gate: der PAUSCHALE Gewinn ueber alle Staedte muss in
der Groessenordnung der +5,63 % vom 18.07. liegen. Anderer Dump (29 statt 27
Staedte, anderes Fenster, Shenzhen jetzt gegen METAR statt ausgeschlossen), also
keine exakte Zahl — aber ein Wert von +20 % oder -5 % hiesse, dass die Mechanik
nicht dieselbe ist und die Gegenrechnung nicht greift.

KEIN LOOK-AHEAD an keiner Stelle: G, Bias und Gewichte eines Monats stammen
ausschliesslich aus frueheren Tagen. Die Auswahl ist damit selbst walk-forward
und nicht die Sechser-Liste vom 18.07. — die dient nur als Validierung.

Aufruf:
  python weather_stadt_konditional_eval.py
  python weather_stadt_konditional_eval.py --ohne-g4     (ohne DB-Zugriff)
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from weather_stations import favorit_k

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

DUMP = "preregs/weather_konvektiv_sigma_residuen_2026_08_03.csv.gz"
BASE5 = ["gfs_seamless", "icon_seamless", "ukmo_seamless", "jma_seamless",
         "ecmwf_ifs025"]
WARMUP = 180
QUARTIL_TEILER = 4
DB_CONFIG = {"server": "158.181.48.77", "database": "dbdata",
             "user": "326773", "password": "Extaler11!"}

# Gate-Schwellen, woertlich aus der Pre-Reg
G1_REDUKTION, G1_T = 10.0, 2.0
G2_MAX_SCHLECHTER = 0.20
G3_RHO = 0.40
G4_BUCKET = 0.2
G5_REDUKTION, G5_T = 8.0, 1.5

# Validierungsliste vom 18.07. — NICHT das Kriterium, nur der Abgleich
LISTE_1807 = ["Seoul", "Tel Aviv", "Munich", "Jeddah", "Wuhan", "Beijing"]


def stadt_matrix(df, city):
    """(dates, R) mit R = Tage x Modelle Residuen (forecast - actual), nur Tage,
    an denen ALLE fuenf Modelle liefern."""
    sub = df[(df.city == city) & (df.model.isin(BASE5))]
    piv = sub.pivot_table(index="date", columns="model", values="resid").dropna()
    if piv.empty:
        return None, None, None
    piv = piv.sort_index()
    return piv.index.values.astype("datetime64[D]"), piv[BASE5].values, piv.index


def invkov_gewichte(E, sd):
    """Inverse-Kovarianz-Gewichte mit Gleichkorrelations-Sigma — identisch zum
    H5-Lauf vom 18.07., damit die Gegenrechnung traegt."""
    k = E.shape[1]
    C = np.corrcoef(E.T)
    iu = np.triu_indices(k, 1)
    rho = float(np.clip(C[iu].mean(), 0.0, 0.98))
    Sig = np.outer(sd, sd) * (rho * np.ones((k, k)) + (1 - rho) * np.eye(k))
    try:
        wi = np.linalg.solve(Sig, np.ones(k))
    except np.linalg.LinAlgError:
        wi = 1.0 / np.maximum(sd, 1e-6) ** 2
    wi = np.clip(wi, 0.0, None)
    return wi / wi.sum() if wi.sum() > 0 else np.full(k, 1.0 / k)


def walkforward(dates, R, monate):
    """Je Stadt: pro Bewertungsmonat G, Gewichte, Bias — alles aus Tagen VOR dem
    Monat. Liefert (rows, monatsinfo).

    rows       : je Tag (monat, err_ew, err_iv)
    monatsinfo : monat -> (G, bias, w)
    """
    rows, info = [], {}
    for monat in sorted(set(monate)):
        vor = dates < np.datetime64(f"{monat}-01")
        if vor.sum() < WARMUP:
            continue
        hist = R[vor]
        bias = hist.mean(axis=0)
        E = hist - bias
        sd = E.std(axis=0, ddof=1)
        sd_ens = E.mean(axis=1).std(ddof=1)
        G = float(sd_ens / sd.min()) if sd.min() > 0 else np.nan
        w = invkov_gewichte(E, sd)
        info[monat] = (G, bias, w)
        for i in np.where(np.array(monate) == monat)[0]:
            deb = R[i] - bias
            rows.append((monat, float(deb.mean()), float(w @ deb)))
    return rows, info


def gewinn(paare):
    """(MAE-Gewinn %, sd-Gewinn %) aus [(err_ew, err_iv), ...]."""
    if not paare:
        return np.nan, np.nan
    ew = np.array([p[0] for p in paare])
    iv = np.array([p[1] for p in paare])
    mae_g = (np.abs(ew).mean() - np.abs(iv).mean()) / np.abs(ew).mean() * 100
    sd_g = (ew.std(ddof=1) - iv.std(ddof=1)) / ew.std(ddof=1) * 100
    return float(mae_g), float(sd_g)


def stadt_t(werte):
    d = np.array([w for w in werte if np.isfinite(w)], float)
    if len(d) < 3 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def lade_ladder():
    """Ladder-Log, max-Bretter, Lead 1, gesettelt. Ein Stadttag = eine Zeile."""
    import pymssql
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT city, target_date, MAX(settle_k)
        FROM bb_WeatherLadders
        WHERE settle_k IS NOT NULL AND var = 'max'
          AND DATEDIFF(day, CAST(snapshot_utc AS DATE), target_date) = 1
        GROUP BY city, target_date""")
    rows = cur.fetchall()
    conn.close()
    return {(c, str(d)): int(s) for c, d, s in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ohne-g4", action="store_true",
                    help="G4 ueberspringen (kein DB-Zugriff)")
    args = ap.parse_args()

    df = pd.read_csv(DUMP, parse_dates=["date"])
    df["resid"] = df.forecast - df.actual
    cities = sorted(df.city.unique())
    print("=" * 96)
    print("STADT-KONDITIONALE MODELLGEWICHTUNG — Auswahl = oberstes Quartil nach G")
    print("=" * 96)
    print(f"Dump: {len(df)} Zeilen, {len(cities)} Staedte, "
          f"{df.date.min().date()} .. {df.date.max().date()}")
    print(f"Anlauf {WARMUP} Serien-Tage, Refit monatlich, Auswahl "
          f"{round(len(cities)/QUARTIL_TEILER)} von {len(cities)} Staedten je Monat")

    # ---------------------------------------------- Walk-forward je Stadt
    tage = {}       # city -> [(monat, err_ew, err_iv)]
    G_monat = {}    # city -> {monat: G}
    gewichte = {}   # city -> {monat: (bias, w)}
    for city in cities:
        dates, R, idx = stadt_matrix(df, city)
        if dates is None or len(dates) < WARMUP + 30:
            print(f"  uebersprungen (zu kurz): {city}")
            continue
        monate = [str(d)[:7] for d in idx.strftime("%Y-%m-%d")]
        rows, info = walkforward(dates, R, monate)
        if not rows:
            continue
        tage[city] = rows
        G_monat[city] = {m: v[0] for m, v in info.items()}
        gewichte[city] = {m: (v[1], v[2]) for m, v in info.items()}
    print(f"Staedte mit Bewertungstagen: {len(tage)}")

    alle_monate = sorted({m for c in tage for m, *_ in tage[c]})
    print(f"Bewertungsmonate: {alle_monate[0]} .. {alle_monate[-1]} "
          f"({len(alle_monate)})")

    # ---------------------------------------------- monatliche Auswahl
    auswahl = {}    # monat -> set(city)
    for monat in alle_monate:
        kand = [(c, G_monat[c][monat]) for c in tage
                if monat in G_monat[c] and np.isfinite(G_monat[c][monat])]
        if not kand:
            continue
        k = max(1, int(round(len(kand) / QUARTIL_TEILER)))
        kand.sort(key=lambda x: -x[1])
        auswahl[monat] = {c for c, _ in kand[:k]}

    haeufigkeit = {c: sum(1 for m in auswahl if c in auswahl[m]) for c in tage}
    gewaehlt = [c for c in tage if haeufigkeit[c] > 0]
    print(f"\nStaedte, die mindestens einmal ins Quartil kommen: {len(gewaehlt)}")
    print(f"{'Stadt':<15}{'Monate im Quartil':>18}{'mittleres G':>13}   Liste 18.07.")
    print("-" * 96)
    for c in sorted(gewaehlt, key=lambda c: -haeufigkeit[c]):
        gq = np.nanmean(list(G_monat[c].values()))
        mark = "  <- ja" if c in LISTE_1807 else ""
        print(f"{c:<15}{haeufigkeit[c]:>10} / {len(auswahl):<5}{gq:>13.3f}{mark}")
    treffer = sum(1 for c in LISTE_1807 if haeufigkeit.get(c, 0) > 0)
    print(f"\nValidierung: {treffer} von {len(LISTE_1807)} Staedten der 18.07.-Liste "
          f"werden vom Kriterium mindestens einmal gewaehlt.")

    # ---------------------------------------------- Reproduktionsprobe
    alle_paare = [(e, i) for c in tage for _, e, i in tage[c]]
    p_mae, p_sd = gewinn(alle_paare)
    je_stadt_pauschal = {c: gewinn([(e, i) for _, e, i in tage[c]])
                         for c in tage}
    t_pauschal = stadt_t([v[0] for v in je_stadt_pauschal.values()])
    print(f"\nREPRODUKTIONSPROBE (kein Gate) — pauschale Gewichtung ueber alle Staedte:")
    print(f"  MAE-Gewinn gepoolt {p_mae:+.2f} %  Stadt-t {t_pauschal:+.2f}   "
          f"(18.07.: +5,63 % / t 1,91)")
    print(f"  sd-Gewinn gepoolt  {p_sd:+.2f} %")

    # ---------------------------------------------- G1
    print("\n" + "=" * 96)
    print("G1  WIRKUNG IN DER AUSWAHLGRUPPE (nur Monate, in denen die Stadt gewaehlt ist)")
    print("=" * 96)
    kond = {}
    for c in gewaehlt:
        paare = [(e, i) for m, e, i in tage[c] if c in auswahl.get(m, ())]
        kond[c] = (gewinn(paare), len(paare))
    print(f"{'Stadt':<15}{'Tage':>6}{'MAE-Gewinn':>13}{'sd-Gewinn':>12}"
          f"{'pauschal':>11}")
    print("-" * 96)
    for c in sorted(kond, key=lambda c: -kond[c][0][0]):
        (mg, sg), n = kond[c]
        print(f"{c:<15}{n:>6}{mg:>12.1f} %{sg:>11.1f} %"
              f"{je_stadt_pauschal[c][0]:>10.1f} %")
    werte = [kond[c][0][0] for c in kond]
    m1, t1 = float(np.mean(werte)), stadt_t(werte)
    g1 = m1 >= G1_REDUKTION and t1 > G1_T
    print(f"\n  Mittel ueber {len(werte)} Auswahlstaedte: {m1:+.2f} %   Stadt-t {t1:+.2f}")
    print(f"  G1 verlangt >= {G1_REDUKTION:.0f} % UND t > {G1_T:.1f}  ->  "
          f"{'BELEGT' if g1 else 'NICHT BELEGT'}")

    # ---------------------------------------------- G2
    schlechter = [c for c in kond if kond[c][0][0] <= 0]
    anteil = len(schlechter) / len(kond)
    g2 = anteil <= G2_MAX_SCHLECHTER
    print(f"\nG2  FEHLKLASSIFIKATION — {len(schlechter)} von {len(kond)} "
          f"Auswahlstaedten werden schlechter ({anteil:.0%})")
    if schlechter:
        print("     " + ", ".join(f"{c} {kond[c][0][0]:+.1f} %" for c in schlechter))
    print(f"  G2 verlangt <= {G2_MAX_SCHLECHTER:.0%}  ->  "
          f"{'BELEGT' if g2 else 'NICHT BELEGT'}")

    # ---------------------------------------------- G3
    xs, ys = [], []
    for c in tage:
        gq = np.nanmean(list(G_monat[c].values()))
        if np.isfinite(gq) and np.isfinite(je_stadt_pauschal[c][0]):
            xs.append(gq)
            ys.append(je_stadt_pauschal[c][0])
    rho, p = spearmanr(xs, ys)
    g3 = rho > G3_RHO
    print(f"\nG3  TRENNSCHAERFE — Spearman(G, realisierter Gewinn) ueber alle "
          f"{len(xs)} Staedte")
    print(f"  rho = {rho:+.3f}  (p = {p:.4f})")
    print(f"  G3 verlangt rho > {G3_RHO:.1f}  ->  "
          f"{'BELEGT' if g3 else 'NICHT BELEGT'}")

    # ---------------------------------------------- G5
    print(f"\nG5  ROBUSTHEIT — leave-one-city-out ueber die Auswahlgruppe")
    schlimmst, wer = None, None
    for weg in kond:
        rest = [kond[c][0][0] for c in kond if c != weg]
        mm, tt = float(np.mean(rest)), stadt_t(rest)
        if schlimmst is None or mm < schlimmst[0]:
            schlimmst, wer = (mm, tt), weg
    ohne_seoul = [kond[c][0][0] for c in kond if c != "Seoul"]
    if ohne_seoul:
        ms, ts = float(np.mean(ohne_seoul)), stadt_t(ohne_seoul)
        print(f"  ohne Seoul: {ms:+.2f} %  t {ts:+.2f}   (verlangt >= "
              f"{G5_REDUKTION:.0f} % und t > {G5_T:.1f})")
        g5 = ms >= G5_REDUKTION and ts > G5_T
    else:
        ms, ts, g5 = np.nan, np.nan, False
    print(f"  schlechtester Fall: ohne {wer} -> {schlimmst[0]:+.2f} % "
          f"t {schlimmst[1]:+.2f}")
    print(f"  G5  ->  {'BELEGT' if g5 else 'NICHT BELEGT'}")

    # ---------------------------------------------- G4
    g4 = None
    if args.ohne_g4:
        print("\nG4  uebersprungen (--ohne-g4)")
    else:
        print(f"\nG4  BUCHEBENE — bewegt die Gewichtung den ANKER? "
              f"(Ladder-Log, max, Lead 1)")
        try:
            settle = lade_ladder()
        except Exception as exc:
            print(f"  DB nicht erreichbar: {exc}")
            settle = None
        if settle:
            fc = df.pivot_table(index=["city", "date"], columns="model",
                                values="forecast")
            gruppen = {"Auswahl": [], "uebrige": []}
            for (city, datum), reihe in fc.iterrows():
                if city not in gewichte or reihe[BASE5].isna().any():
                    continue
                tag = str(datum)[:10]
                monat = tag[:7]
                if monat not in gewichte[city] or (city, tag) not in settle:
                    continue
                bias, w = gewichte[city][monat]
                f = reihe[BASE5].values.astype(float) - bias
                mu_ew, mu_iv = float(f.mean()), float(w @ f)
                s = settle[(city, tag)]
                grp = ("Auswahl" if city in auswahl.get(monat, ()) else "uebrige")
                gruppen[grp].append((city,
                                     abs(favorit_k(mu_ew, city) - s),
                                     abs(favorit_k(mu_iv, city) - s)))
            for name, rows in gruppen.items():
                if not rows:
                    print(f"  {name}: keine Ueberlappung von Dump und Ladder-Log")
                    continue
                a = np.mean([r[1] for r in rows])
                b = np.mean([r[2] for r in rows])
                print(f"  {name:<9} n={len(rows):>4} Stadttage, "
                      f"{len(set(r[0] for r in rows)):>2} Staedte   "
                      f"MAE gleich {a:.2f} -> gewichtet {b:.2f} Bucket "
                      f"({a-b:+.2f})")
            aus, ueb = gruppen["Auswahl"], gruppen["uebrige"]
            if aus:
                d_aus = (np.mean([r[1] for r in aus])
                         - np.mean([r[2] for r in aus]))
                d_ueb = ((np.mean([r[1] for r in ueb])
                          - np.mean([r[2] for r in ueb])) if ueb else 0.0)
                g4 = d_aus >= G4_BUCKET and d_ueb >= -0.01
                print(f"  G4 verlangt Auswahl >= {G4_BUCKET:.1f} Bucket besser UND "
                      f"uebrige nicht schlechter  ->  "
                      f"{'BELEGT' if g4 else 'NICHT BELEGT'}")

    # ---------------------------------------------- Bilanz
    print("\n" + "=" * 96)
    mark = lambda ok: "GRUEN" if ok else ("ROT" if ok is not None else "OFFEN")
    print(f"  G1 {mark(g1)} · G2 {mark(g2)} · G3 {mark(g3)} · G4 {mark(g4)} · "
          f"G5 {mark(g5)}")
    print("Ein PASS erlaubt KEINE Live-Aenderung — es folgt ein Forward-Fenster.")
    print("Bis 02.09.2026 laufen drei vorregistrierte Tests; eine Aenderung an mu")
    print("mitten im Fenster macht deren zweite Haelfte zu einer anderen Stichprobe.")
    print("=" * 96)


if __name__ == "__main__":
    main()
