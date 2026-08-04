#!/usr/bin/env python3
"""weather_minus1_ursache_eval.py — Auswertung zu
`preregs/weather_minus1_ursache_2026_08_02.md`.

FRAGE: Warum verliert die -1-Klasse als Ganzes? Liegt es am Anker (mu_ens
verschoben), an der Breite (sigma_ens zu klein) — oder schlicht daran, dass eine
fair bepreiste Leiter nach Gebuehr keinen Lay traegt? Und traegt die -2-Klasse?

DIE ZENTRALE GROESSE ist der realisierte Offset  d = settle_k - k0  mit
k0 = favorit_k(mu_ens, city) = k - offset_fav. Aus seiner Verteilung folgen alle
Trefferquoten: P(d = -1) IST die Verliererquote der -1-Klasse.

DREI WAHRSCHEINLICHKEITEN werden je Offset gegeneinander gestellt:
  P_ist    beobachtete Haeufigkeit von d
  P_modell aus mu_ens, sigma_ens, bucket_grenzen(k, city) — je Stadt-Tag einzeln
           integriert, NIE aus einer Durchschnittsposition (Designfalle 2)
  P_markt  Mitte aus buy_yes und 1 - buy_no
Daraus: P_ist vs P_markt = verdient der Handel; P_modell vs P_ist = ist unser
Modell kalibriert; P_modell vs P_markt = haben wir ueberhaupt eine Meinung.

BREAK-EVEN steht vor der Messung fest: bei NO-Preis p verliert eine Lay-Klasse
zwangslaeufig, wenn sie haeufiger trifft als q = 1 - p (ohne Gebuehr). Fuer die
Medianpreise: -1 bei 0,770 -> 21,4 % mit Fee 0,07; -2 bei 0,960 -> 3,7 %.

LEAD 1 ist fix (Kaufzeitpunkt des Autobuy, 14:45). Wer ueber Leads mittelt,
mischt Prognosen verschiedener Frische und verwischt genau die gesuchte
Verschiebung. Lead 2 laeuft nur als Robustheitsprobe.

SHENZHEN: bis zum Fix am 02.08. gegen "Lau Fau Shan" in Hong Kong kalibriert —
alle historischen Zeilen tragen ein verzerrtes mu_ens und damit ein falsches k0,
also genau die Groesse, die hier gemessen wird. Der Kontrolllauf ohne Shenzhen
ist vorregistriert; bei Abweichung gilt er.

SIGNIFIKANZ: t ueber TAGES-Mittel, nie ueber Stadt-Tage. Dreissig Staedte eines
Zieltags haengen an derselben Grosswetterlage.

Aufruf:
  python weather_minus1_ursache_eval.py
  python weather_minus1_ursache_eval.py --lead 2          # Robustheitsprobe
  python weather_minus1_ursache_eval.py --ohne Shenzhen   # Kontrolllauf
"""

import argparse
import math
import statistics
import sys
from collections import defaultdict

import pymssql

from weather_ladder_logger import DB_CONFIG
from weather_stations import bucket_grenzen

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

USD = 5.0
FEE_HAUPT = 0.07
FEE_ALT = 0.04
HAELFTEN = (("2026-07-11", "2026-07-21"), ("2026-07-22", "2026-08-01"))
OFFSETS = (-3, -2, -1, 0, 1, 2, 3)


def haelften(saetze):
    """Die vorregistrierte Trennung 11.-21.07. / ab 22.07. — die zweite Haelfte
    reicht bis zum Ende des tatsaechlichen Fensters.

    Am 02.08. endete der Bestand am 01.08.; dann ist das identisch zu HAELFTEN.
    Kommen spaeter Zieltage dazu (Lead-2-Nachholung), duerfen sie nicht aus der
    Vorzeichenprobe herausfallen — die vorregistrierte GRENZE bleibt unangetastet,
    nur das offene Ende waechst mit."""
    ende = max(s["tag"] for s in saetze) if saetze else HAELFTEN[1][1]
    return (HAELFTEN[0], (HAELFTEN[1][0], max(ende, HAELFTEN[1][1])))


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def p_bucket(mu, sigma, k, city):
    """Modell-Wahrscheinlichkeit des Buckets k — ueber die ECHTEN Grenzen."""
    if not sigma or sigma <= 0:
        return None
    unten, oben = bucket_grenzen(k, city)
    return phi((oben - mu) / sigma) - phi((unten - mu) / sigma)


def pnl_lay(no, verloren, fee_rate):
    """Identisch zu weather_band_breite_eval / _minus1_shadow, damit vergleichbar."""
    n = USD / no
    fee = fee_rate * n * min(no, 1 - no)
    return (-USD - fee) if verloren else (n - USD - fee)


def breakeven_q(no, fee_rate):
    """Trefferquote, bei der die Klasse zu diesem Preis gerade nichts mehr verdient."""
    n = USD / no
    fee = fee_rate * n * min(no, 1 - no)
    gewinn, verlust = n - USD - fee, USD + fee
    return gewinn / (gewinn + verlust)


def ein_stichproben_t(werte):
    """t gegen 0 ueber Tageswerte."""
    xs = [x for x in werte if x is not None and not math.isnan(x)]
    if len(xs) < 3:
        return float("nan"), len(xs), float("nan")
    m = statistics.mean(xs)
    sd = statistics.stdev(xs)
    if sd == 0:
        return float("inf"), len(xs), m
    return m / (sd / math.sqrt(len(xs))), len(xs), m


def lade(lead, ohne, bis=None, nur=None):
    """Ein Datensatz je Stadt-Tag: Anker, Settlement, Leiter aus dem neuesten
    Snapshot des gewuenschten Leads.

    `bis`  begrenzt das Zieltagsfenster (Reproduktion eines aelteren Laufs).
    `nur`  beschraenkt auf eine Menge von (Zieltag, Stadt)-Schluesseln — dafuer,
           Lead 1 und Lead 2 GEPAART auf denselben Stadt-Tagen zu rechnen. Ohne
           das vergleicht man 30 Staedte gegen 25, und weil die Achsen-
           verschiebung je Stadt sitzt (sd = 0,699 Bucket), waere ein Teil des
           Unterschieds reine Staedteauswahl statt Vorlauf."""
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor(as_dict=True)
    cur.execute(
        "SELECT snapshot_utc, target_date, city, k, buy_no, buy_yes, mu_ens, "
        "sigma_ens, offset_fav, market_fav_k, settle_k, settle_result "
        "FROM bb_WeatherLadders WHERE var='max' AND kind='eq'")
    roh = cur.fetchall()
    conn.close()

    def lead_of(r):
        return (r["target_date"] - r["snapshot_utc"].date()).days

    neueste = {}
    for r in roh:
        if lead_of(r) != lead:
            continue
        key = (str(r["target_date"]), r["city"])
        if key not in neueste or r["snapshot_utc"] > neueste[key]:
            neueste[key] = r["snapshot_utc"]

    je = {}
    for r in roh:
        key = (str(r["target_date"]), r["city"])
        if lead_of(r) != lead or r["snapshot_utc"] != neueste.get(key):
            continue
        if r["city"] in ohne:
            continue
        if bis is not None and key[0] > bis:
            continue
        if nur is not None and key not in nur:
            continue
        d = je.setdefault(key, {"tag": key[0], "city": key[1], "k0": None,
                                "mu": None, "sigma": None, "settle_k": None,
                                "market_fav_k": None, "leiter": {}})
        if r["mu_ens"] is not None:
            d["mu"] = float(r["mu_ens"])
        if r["sigma_ens"] is not None:
            d["sigma"] = float(r["sigma_ens"])
        if r["settle_k"] is not None:
            d["settle_k"] = int(r["settle_k"])
        if r["market_fav_k"] is not None:
            d["market_fav_k"] = int(r["market_fav_k"])
        if r["offset_fav"] is not None:
            d["k0"] = int(r["k"]) - int(r["offset_fav"])
            d["leiter"][int(r["offset_fav"])] = {
                "k": int(r["k"]),
                "no": float(r["buy_no"]) if r["buy_no"] else None,
                "yes": float(r["buy_yes"]) if r["buy_yes"] else None,
                "res": None if r["settle_result"] is None else bool(r["settle_result"]),
            }

    voll = [d for d in je.values()
            if d["k0"] is not None and d["settle_k"] is not None and d["mu"] is not None]
    for d in voll:
        d["d"] = d["settle_k"] - d["k0"]
        d["d_markt"] = (d["settle_k"] - d["market_fav_k"]
                        if d["market_fav_k"] is not None else None)
    return sorted(voll, key=lambda x: (x["tag"], x["city"]))


def je_tag(saetze, f):
    """Tagesmittel einer Groesse f(datensatz) — die Einheit fuer jedes t."""
    eimer = defaultdict(list)
    for s in saetze:
        v = f(s)
        if v is not None:
            eimer[s["tag"]].append(v)
    return {t: statistics.mean(v) for t, v in eimer.items() if v}


def block_wahrscheinlichkeiten(saetze):
    """P_ist, P_modell, P_markt je Offset."""
    n = len(saetze)
    zeilen = []
    for o in OFFSETS:
        ist = sum(1 for s in saetze if s["d"] == o) / n
        mods = [p_bucket(s["mu"], s["sigma"], s["k0"] + o, s["city"]) for s in saetze]
        mods = [m for m in mods if m is not None]
        mkt = []
        for s in saetze:
            b = s["leiter"].get(o)
            if b and b["no"] and b["yes"]:
                mkt.append(((1 - b["no"]) + b["yes"]) / 2)
        zeilen.append({
            "off": o, "ist": ist,
            "modell": statistics.mean(mods) if mods else float("nan"),
            "markt": statistics.mean(mkt) if mkt else float("nan"),
            "n_markt": len(mkt),
        })
    return zeilen


def klassen_oekonomie(saetze, o, fee_rate):
    """ROI der Lay-Klasse mit Offset o zum jeweiligen Marktpreis."""
    posten = []
    for s in saetze:
        b = s["leiter"].get(o)
        if not b or not b["no"]:
            continue
        verloren = (s["d"] == o)          # Wahrheit ist settle_k, nicht settle_result
        posten.append({"tag": s["tag"], "no": b["no"], "verloren": verloren,
                       "pnl": pnl_lay(b["no"], verloren, fee_rate)})
    if not posten:
        return None
    p = sum(x["pnl"] for x in posten)
    v = sum(1 for x in posten if x["verloren"])
    return {"n": len(posten), "pnl": p, "roi": 100 * p / (USD * len(posten)),
            "quote": 100 * v / len(posten),
            "no": statistics.mean(x["no"] for x in posten),
            "be": statistics.mean(breakeven_q(x["no"], fee_rate) for x in posten) * 100,
            "posten": posten}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--ohne", nargs="*", default=[], metavar="STADT")
    ap.add_argument("--bis", default=None, metavar="YYYY-MM-DD",
                    help="Zieltagsfenster begrenzen (Reproduktion aelterer Laeufe)")
    ap.add_argument("--gepaart-mit-lead", type=int, default=None, dest="gepaart",
                    help="nur Stadt-Tage, die auch in diesem Lead vorliegen")
    a = ap.parse_args()
    ohne = set(a.ohne)

    nur = None
    if a.gepaart is not None:
        nur = {(s["tag"], s["city"]) for s in lade(a.gepaart, ohne, bis=a.bis)}

    saetze = lade(a.lead, ohne, bis=a.bis, nur=nur)
    tage = sorted({s["tag"] for s in saetze})
    staedte = sorted({s["city"] for s in saetze})
    hlf = haelften(saetze)

    print("=" * 78)
    print(f"URSACHE DER -1-KLASSE   Lead {a.lead}"
          + (f"   gepaart mit Lead {a.gepaart}" if a.gepaart is not None else "")
          + (f"   bis {a.bis}" if a.bis else "")
          + (f"   ohne {', '.join(sorted(ohne))}" if ohne else ""))
    print("=" * 78)
    print(f"Stadt-Tage {len(saetze)} | Zieltage {len(tage)} ({tage[0]} .. {tage[-1]}) "
          f"| Staedte {len(staedte)}")
    print(f"Haelften: {hlf[0][0]} .. {hlf[0][1]}  /  {hlf[1][0]} .. {hlf[1][1]}")

    # ---------------------------------------------------------------- G0
    wid = ges = 0
    for s in saetze:
        for o, b in s["leiter"].items():
            if b["res"] is not None:
                ges += 1
                if b["res"] != (s["d"] == o):
                    wid += 1
    quote = 100 * wid / ges if ges else 0.0
    print(f"\nG0  DATENBASIS")
    print(f"  settle_result gegen settle_k: {wid}/{ges} Widersprueche ({quote:.2f} %)")
    g0 = (len(saetze) >= 250 and len(tage) >= 18 and len(staedte) >= 20 and quote < 3.0)
    print(f"  Verlangt: >= 250 Stadt-Tage, >= 18 Zieltage, >= 20 Staedte, "
          f"Widerspruch < 3 %  ->  {'BESTANDEN' if g0 else 'GERISSEN'}")
    if not g0:
        print("\n  Abbruchregel: bei gerissenem G0 wird nicht gerechnet, sondern")
        print("  erst die Datenlage geklaert.")
        return

    # ------------------------------------------------- Verteilung von d
    print(f"\nDIE VERTEILUNG DES REALISIERTEN OFFSETS  d = settle_k - k0")
    zeilen = block_wahrscheinlichkeiten(saetze)
    print("  Offset   P_ist     P_modell   P_markt    Break-even (Fee 0,07)")
    for z in zeilen:
        be = klassen_oekonomie(saetze, z["off"], FEE_HAUPT)
        be_s = f"{be['be']:5.1f} %" if be else "    —"
        print(f"   {z['off']:+2d}    {100*z['ist']:5.1f} %   {100*z['modell']:5.1f} %"
              f"    {100*z['markt']:5.1f} %      {be_s}")
    ausserhalb = 1 - sum(z["ist"] for z in zeilen)
    print(f"  |d| > 3: {100*ausserhalb:.1f} %")

    # ---------------------------------------------------------------- G1
    print(f"\nG1  ANKER — ist unsere Bucket-Achse verschoben?")
    d_tage = je_tag(saetze, lambda s: s["d"])
    t1, n1, m1 = ein_stichproben_t(list(d_tage.values()))
    print(f"  E[d] = {m1:+.3f} Bucket   (t = {t1:+.2f} ueber {n1} Zieltage)")
    vorz = []
    for von, bis in hlf:
        h = [s for s in saetze if von <= s["tag"] <= bis]
        if h:
            mh = statistics.mean(statistics.mean([x["d"] for x in g])
                                 for g in _nach_tag(h).values())
            vorz.append(mh)
            print(f"    {von} .. {bis}:  E[d] = {mh:+.3f}  (n = {len(h)})")
    gleich = len(vorz) == 2 and (vorz[0] > 0) == (vorz[1] > 0)
    g1 = abs(m1) >= 0.25 and abs(t1) > 2.0 and gleich
    print(f"  Verlangt: |E[d]| >= 0,25 UND |t| > 2,0 UND gleiches Vorzeichen "
          f"in beiden Haelften  ->  {'BELEGT' if g1 else 'NICHT BELEGT'}")

    # Die Streuung von d selbst — die Groesse, an der sich die Erwartung
    # "laengerer Vorlauf verbreitert die Verteilung" ueberhaupt erst pruefen
    # laesst. Kein Gate, aber ohne sie ist die Lead-2-Probe nicht lesbar.
    alle_d = [s["d"] for s in saetze]
    print(f"  Streuung des Offsets selbst: sd(d) = {statistics.pstdev(alle_d):.3f} "
          f"Bucket   |d| >= 2: {100*sum(1 for x in alle_d if abs(x) >= 2)/len(alle_d):.1f} %")

    print("  Diagnostisch (kein Gate) — Verschiebung je Stadt:")
    je_stadt = defaultdict(list)
    for s in saetze:
        je_stadt[s["city"]].append(s["d"])
    mittel = {c: statistics.mean(v) for c, v in je_stadt.items() if len(v) >= 5}
    if mittel:
        srt = sorted(mittel.items(), key=lambda x: x[1])
        print(f"    Streuung der Stadt-Mittelwerte: sd = "
              f"{statistics.pstdev(mittel.values()):.3f} Bucket ueber {len(mittel)} Staedte")
        print(f"    kaeltester Ausreisser {srt[0][0]} {srt[0][1]:+.2f} | "
              f"waermster {srt[-1][0]} {srt[-1][1]:+.2f}")

    # ---------------------------------------------------------------- G2
    print(f"\nG2  BREITE — ist sigma_ens zu klein?")
    diff_tage = je_tag(saetze, lambda s: (1.0 if s["d"] == -1 else 0.0) -
                       (p_bucket(s["mu"], s["sigma"], s["k0"] - 1, s["city"]) or 0.0))
    t2, n2, m2 = ein_stichproben_t(list(diff_tage.values()))
    z_minus1 = next(z for z in zeilen if z["off"] == -1)
    print(f"  P_ist(-1) {100*z_minus1['ist']:.1f} %  gegen  P_modell(-1) "
          f"{100*z_minus1['modell']:.1f} %   ->  {100*m2:+.1f} pp  (t = {t2:+.2f})")
    h_ok = []
    for von, bis in hlf:
        h = [s for s in saetze if von <= s["tag"] <= bis]
        if h:
            zh = block_wahrscheinlichkeiten(h)
            zm = next(z for z in zh if z["off"] == -1)
            h_ok.append(zm["ist"] - zm["modell"])
            print(f"    {von} .. {bis}:  {100*(zm['ist']-zm['modell']):+.1f} pp")
    gleich2 = len(h_ok) == 2 and (h_ok[0] > 0) == (h_ok[1] > 0)
    g2 = (100 * m2) >= 4.0 and t2 > 2.0 and gleich2
    print(f"  Verlangt: >= 4 pp UND t > 2,0 UND gleichgerichtet  ->  "
          f"{'BELEGT' if g2 else 'NICHT BELEGT'}")

    # ---------------------------------------------------------------- G3
    print(f"\nG3  FAIRNESS — liegt das Ist auf dem Marktpreis?")
    abstand = 100 * (z_minus1["ist"] - z_minus1["markt"])
    print(f"  P_ist(-1) {100*z_minus1['ist']:.1f} %  gegen  P_markt(-1) "
          f"{100*z_minus1['markt']:.1f} %   ->  {abstand:+.1f} pp")
    g3 = abs(abstand) < 3.0 and not g1 and not g2
    print(f"  Verlangt: |Abstand| < 3 pp UND G1 und G2 nicht belegt  ->  "
          f"{'BELEGT' if g3 else 'NICHT BELEGT'}")

    # ---------------------------------------------------------------- G4
    print(f"\nG4  SYMMETRIE — verliert die +1-Klasse genauso? (diagnostisch)")
    for o in (-1, 1):
        e = klassen_oekonomie(saetze, o, FEE_HAUPT)
        if e:
            print(f"  Offset {o:+d}:  {e['n']:3d} Lays  Ø NO {e['no']:.3f}  "
                  f"Treffer {e['quote']:5.1f} %  (Break-even {e['be']:.1f} %)  "
                  f"ROI {e['roi']:+7.2f} %")
    sym = je_tag(saetze, lambda s: (1.0 if s["d"] == -1 else 0.0) -
                 (1.0 if s["d"] == +1 else 0.0))
    t4, n4, m4 = ein_stichproben_t(list(sym.values()))
    print(f"  P_ist(-1) - P_ist(+1) = {100*m4:+.1f} pp  (t = {t4:+.2f} ueber {n4} Tage)")

    # ---------------------------------------------------------------- G5
    print(f"\nG5  DIE -2-KLASSE — traegt sie nach Gebuehr?")
    z_minus2 = next(z for z in zeilen if z["off"] == -2)
    e2 = klassen_oekonomie(saetze, -2, FEE_HAUPT)
    e2_alt = klassen_oekonomie(saetze, -2, FEE_ALT)
    if e2:
        print(f"  volle Menge:  {e2['n']:3d} Lays  Ø NO {e2['no']:.3f}  "
              f"Treffer {e2['quote']:.1f} %  gegen Break-even {e2['be']:.1f} %")
        print(f"    ROI Fee 0,07 {e2['roi']:+.2f} %   |   Fee 0,04 "
              f"{e2_alt['roi']:+.2f} %")
        roi_h = []
        for von, bis in hlf:
            h = [s for s in saetze if von <= s["tag"] <= bis]
            eh = klassen_oekonomie(h, -2, FEE_HAUPT) if h else None
            if eh:
                roi_h.append(eh["roi"])
                print(f"    {von} .. {bis}:  ROI {eh['roi']:+.2f} % (n = {eh['n']})")
        gleich5 = len(roi_h) == 2 and (roi_h[0] > 0) == (roi_h[1] > 0)
        g5 = z_minus2["ist"] < 0.037 and e2["roi"] > 0 and gleich5
        print(f"  Verlangt: P_ist(-2) < 3,7 % UND ROI > 0 UND beide Haelften "
              f"gleich  ->  {'TRAEGT' if g5 else 'TRAEGT NICHT'}")
        # Designfalle 3: die Band-Teilmenge ist eine Selektion, kein Test
        band = [s for s in saetze
                if s["leiter"].get(-2) and s["leiter"][-2]["no"]
                and 0.70 <= s["leiter"][-2]["no"] < 0.90]
        eb = klassen_oekonomie(band, -2, FEE_HAUPT) if band else None
        if eb:
            print(f"  nur Preisband 0,70-0,90 (DIAGNOSTISCH, kein Gate — der Markt "
                  f"haelt\n    diese Lagen fuer breit): {eb['n']} Lays, "
                  f"Treffer {eb['quote']:.1f} %, ROI {eb['roi']:+.2f} %")

    # ---------------------------------------------------------------- H6
    print(f"\nH6  ANKER-VERGLEICH — waere der Markt-Favorit der bessere Anker?")
    mit = [s for s in saetze if s["d_markt"] is not None]
    if mit:
        eig = sum(1 for s in mit if s["d"] == -1) / len(mit)
        mkt = sum(1 for s in mit if s["d_markt"] == -1) / len(mit)
        tref_e = sum(1 for s in mit if s["d"] == 0) / len(mit)
        tref_m = sum(1 for s in mit if s["d_markt"] == 0) / len(mit)
        print(f"  Bucket unter dem Anker trifft:  eigener {100*eig:.1f} %   "
              f"Markt {100*mkt:.1f} %   (n = {len(mit)})")
        print(f"  Anker selbst trifft:            eigener {100*tref_e:.1f} %   "
              f"Markt {100*tref_m:.1f} %")
        print("  Kein Gate: hieraus folgt keine Codeaenderung, hoechstens eine")
        print("  eigene Forward-Pre-Reg.")

    # ------------------------------------------------------------ Fazit
    print("\n" + "=" * 78)
    diag = ("Anker verschoben" if g1 and not g2 else
            "Anker UND Breite" if g1 and g2 else
            "sigma zu klein" if g2 else
            "fair bepreist (H3)" if g3 else "keine der drei Ursachen belegt")
    print(f"DIAGNOSE laut Entscheidungstabelle der Pre-Reg: {diag}")
    print("Aus dieser Messung folgt KEINE Aenderung am laufenden Autobuy.")


def _nach_tag(saetze):
    eimer = defaultdict(list)
    for s in saetze:
        eimer[s["tag"]].append(s)
    return eimer


if __name__ == "__main__":
    main()
