#!/usr/bin/env python3
"""weather_pruefstunde_v2_eval.py — Auswertung zu
`preregs/weather_pruefstunde_v2_2026_08_02.md`.

FRAGE: Traegt die stadtspezifische Pruefstunde? Also: verbessert sie die
TRENNSCHAERFE des Waechter-Signals gegenueber der starren 16:20-Pruefung?

WAS SICH GEGENUEBER v1 AENDERT: nur die Kontrolle. q = 0,12, Raster 10:20-20:20,
Monotonie-Bedingung und Signaldefinition sind unveraendert aus
weather_pruefstunde_eval uebernommen und werden von dort importiert — waere ein
Parameter angefasst worden, waere das Schwellensuche.

DER FEHLER, DEN v1 GERISSEN HAT: G3 verlangte fuer London, Paris und Madrid je
17:20-18:20. Die 12 %, aus denen q stammt, sind aber ein DURCHSCHNITT ueber fuenf
Staedte, nicht die Restwahrscheinlichkeit einer Einzelstadt (Madrid liegt um
16:20 bei 71,4 %). Eine Aggregatzahl wurde zur Erwartung an Einzelfaelle.

ZWEI KONSEQUENZEN AUS DER GEGENRECHNUNG, beide vorab gezogen:
 1. Gerechnet wird NUR auf Staedten mit GEAENDERTER Pruefstunde. Global gerechnet
    misst man Verduennung: auf die fuenf stark korrigierten Staedte entfallen nur
    rund 17 % der Kandidaten, der Rest liefert definitionsgemaess identische
    Zahlen.
 2. Die Reproduktionsprobe gegen die alte Kurve laeuft DIAGNOSTISCH ohne Gate.
    Sie mischt Jahre (Juli 2026 gegen 2024/25) und Quellen (WU gegen ISD) und
    kann eine Regel deshalb nicht erledigen.

TRENNSCHAERFE = LIFT = P(Lay verliert | Signal) - P(Lay verliert | kein Signal),
paarweise auf DENSELBEN Kandidaten. Nicht die Trefferquote: eine zu fruehe
Pruefung erzeugt Fehlalarme, eine zu spaete kommt zu spaet zum Handeln — beide
senken den Lift, waehrend die Trefferquote je nach Mischung in beide Richtungen
wandern kann.

Aufruf:
  python weather_pruefstunde_v2_eval.py
"""

import sys
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import weather_daily_max_timing_isd as isd
from weather_pruefstunde_eval import (ALT, MIN_TAGE_JE_SOMMER, Q, RASTER, S1, S2,
                                      LADDER_VON, LADDER_BIS, basisraten_je_stadt,
                                      hh, iem_reihe, lade_kandidaten, pruefstunde,
                                      signal)
from weather_stations import station_info

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ERSTMESS = ("Helsinki", "Munich", "Paris", "Madrid", "London")
ALT_KURVE = {(13, 20): 91, (14, 20): 87, (15, 20): 76, (16, 20): 41, (17, 20): 12}
MIN_STAEDTE, MIN_KAND, MIN_ABDECKUNG = 8, 60, 0.80


def lift(paare):
    """P(verloren|Signal) - P(verloren|kein Signal), in Prozentpunkten."""
    mit = [v for s, v in paare if s]
    ohne = [v for s, v in paare if not s]
    if not mit or not ohne:
        return None, len(mit), len(ohne)
    p_mit = 100 * sum(mit) / len(mit)
    p_ohne = 100 * sum(ohne) / len(ohne)
    return p_mit - p_ohne, len(mit), len(ohne)


def main():
    print("=" * 78)
    print(f"PRUEFSTUNDE v2 — traegt sie?   q = {Q:.2f} (unveraendert)")
    print("=" * 78)
    print(f"Basisraten aus ISD, Sommer {S1} und {S2}:")
    raten = basisraten_je_stadt([S1, S2])

    brauchbar = {c: v for c, v in raten.items()
                 if all(v.get("n", {}).get(j, 0) >= MIN_TAGE_JE_SOMMER
                        for j in (S1, S2))}
    ps = {}
    for c, v in brauchbar.items():
        zus = {s: [sum(v[j][s][0] for j in (S1, S2) if j in v),
                   sum(v[j][s][1] for j in (S1, S2) if j in v)] for s in RASTER}
        p = pruefstunde(zus)
        if p:
            ps[c] = p

    geaendert = {c: p for c, p in ps.items() if p != ALT}
    print(f"\nStaedte mit Pruefstunde: {len(ps)}   davon geaendert gegen "
          f"{hh(ALT)}: {len(geaendert)}")

    # ------------------------------------------- diagnostisch: Reproduktion
    print(f"\nREPRODUKTIONSPROBE (DIAGNOSTISCH, KEIN GATE)")
    print("  gepoolte Kurve der fuenf Erstmess-Staedte gegen Juli 2026 / WU:")
    for s in sorted(ALT_KURVE):
        z, n = 0, 0
        for c in ERSTMESS:
            if c in brauchbar:
                for j in (S1, S2):
                    if j in brauchbar[c] and s in brauchbar[c][j]:
                        z += brauchbar[c][j][s][0]
                        n += brauchbar[c][j][s][1]
        neu = 100 * z / n if n else float("nan")
        print(f"    {hh(s)}   ISD 2024/25 {neu:5.1f} %   gegen  WU Juli 2026 "
              f"{ALT_KURVE[s]:3d} %")
    print("  Abweichungen sind erwartet — andere Jahre, andere Quelle. Diese")
    print("  Probe kann die Regel nicht erledigen, deshalb ohne Gate.")

    # ---------------------------------------------------------------- G0
    print(f"\nG0  BASIS")
    kand_alle = lade_kandidaten()
    kand = {k: v for k, v in kand_alle.items() if v["city"] in geaendert}
    print(f"  Kandidaten gesamt {len(kand_alle)}, auf geaenderten Staedten "
          f"{len(kand)}")
    reihen = {}
    for c in sorted({v["city"] for v in kand.values()}):
        icao = next((v["icao"] for v in kand.values() if v["city"] == c), None)
        tz = (station_info(icao) or {}).get("tz") if icao else None
        if not tz:
            continue
        try:
            reihen[c] = iem_reihe(icao, tz,
                                  datetime.strptime(LADDER_VON, "%Y-%m-%d"),
                                  datetime.strptime(LADDER_BIS, "%Y-%m-%d"))
        except Exception as ex:
            print(f"  {c}: IEM-Abruf fehlgeschlagen ({str(ex)[:50]})")
    nutzbar = {k: v for k, v in kand.items()
               if v["city"] in reihen and reihen[v["city"]].get(v["tag"])}
    abdeckung = len(nutzbar) / len(kand) if kand else 0
    print(f"  IEM-Reihen fuer {len(reihen)} Staedte, nutzbare Stadt-Tage "
          f"{len(nutzbar)} ({100*abdeckung:.0f} %)")
    g0 = (len(geaendert) >= MIN_STAEDTE and len(nutzbar) >= MIN_KAND
          and abdeckung >= MIN_ABDECKUNG)
    print(f"  Verlangt: >= {MIN_STAEDTE} Staedte, >= {MIN_KAND} Kandidaten, "
          f">= {100*MIN_ABDECKUNG:.0f} % Abdeckung  ->  "
          f"{'BESTANDEN' if g0 else 'GERISSEN'}")
    if not g0:
        print("  Vor G0 wird nicht gerechnet; der Test wird auf ein groesseres")
        print("  Kandidatenfenster vertagt, ohne die Schwellen anzufassen.")
        return

    # ---------------------------------------------------------------- G1
    print(f"\nG1  TRENNSCHAERFE — Lift stadtspezifisch gegen starr {hh(ALT)}")
    paare = {"stadt": [], "starr": []}
    je_stadt = defaultdict(lambda: {"stadt": [], "starr": []})
    for v in nutzbar.values():
        reihe = reihen[v["city"]][v["tag"]]
        for lbl, stunde in (("stadt", geaendert[v["city"]]), ("starr", ALT)):
            sig = signal(reihe, stunde, v["k"], v["city"])
            if sig is None:
                continue
            paare[lbl].append((sig, 1 if v["verloren"] else 0))
            je_stadt[v["city"]][lbl].append((sig, 1 if v["verloren"] else 0))

    werte = {}
    for lbl, name in (("starr", f"starr {hh(ALT)}"), ("stadt", "stadtspezifisch")):
        l, nm, no = lift(paare[lbl])
        werte[lbl] = (l, nm)
        if l is None:
            print(f"  {name:<18} zu wenig Faelle (Signal {nm}, kein Signal {no})")
        else:
            pm = 100 * sum(v for _, v in paare[lbl] if _) / nm if nm else float("nan")
            print(f"  {name:<18} Signal {nm:3d}x -> Verlierer {pm:5.1f} %   "
                  f"Lift {l:+6.1f} pp")
    if werte["stadt"][0] is not None and werte["starr"][0] is not None:
        besser = werte["stadt"][0] > werte["starr"][0]
        genug = werte["stadt"][1] >= werte["starr"][1] / 2
        g1 = besser and genug
        print(f"  Differenz {werte['stadt'][0]-werte['starr'][0]:+.1f} pp   "
              f"Signale {werte['stadt'][1]} gegen {werte['starr'][1]}")
        print(f"  Verlangt: Lift groesser UND Signale nicht unter der Haelfte"
              f"  ->  {'BELEGT' if g1 else 'NICHT BELEGT'}")

        # ------------------------------------------------------------ G2
        print(f"\nG2  RICHTUNGSPROBE — haengt der Vorteil an einer Stadt?")
        beitrag = {}
        for c in je_stadt:
            rest = {lbl: [p for cc in je_stadt if cc != c
                          for p in je_stadt[cc][lbl]] for lbl in ("stadt", "starr")}
            ls, _, _ = lift(rest["stadt"])
            lt, _, _ = lift(rest["starr"])
            if ls is not None and lt is not None:
                beitrag[c] = ls - lt
        if beitrag:
            schwach = min(beitrag, key=beitrag.get)
            print(f"  ohne die staerkste Stadt ({schwach}): Differenz "
                  f"{beitrag[schwach]:+.1f} pp")
            g2 = beitrag[schwach] > 0
            print(f"  Verlangt: Vorzeichen dreht nicht  ->  "
                  f"{'BESTANDEN' if g2 else 'GERISSEN'}")

    # -------------------------------------------------- Berliner Zeiten
    print(f"\nDIAGNOSTISCH — Pruefstunde in Berliner Zeit (manuelle Nutzbarkeit)")
    for c in sorted(geaendert):
        tz = (station_info(isd.STAEDTE.get(c, "")) or {}).get("tz")
        b = "—"
        if tz:
            try:
                lok = datetime(2026, 7, 15, geaendert[c][0], geaendert[c][1],
                               tzinfo=ZoneInfo(tz))
                b = lok.astimezone(ZoneInfo("Europe/Berlin")).strftime("%H:%M")
            except Exception:
                pass
        nacht = "  (nachts, manuell unbrauchbar)" if b != "—" and (
            int(b[:2]) >= 22 or int(b[:2]) < 7) else ""
        print(f"  {c:<16}{hh(geaendert[c])} lokal  ->  {b} Berlin{nacht}")

    print("\n" + "=" * 78)
    print("Der Waechter wird in KEINEM Ausgang eingeschaltet. Und diese Pre-Reg")
    print("bekommt keinen dritten Anlauf: reisst sie, ist das Thema Pruefstunde")
    print("abgeschlossen — mit der Tabelle als Ergebnis und ohne Anspruch auf mehr.")


if __name__ == "__main__":
    main()
