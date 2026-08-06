#!/usr/bin/env python3
"""
weather_turnaround_eval.py — "Stand mal bei 30 %, fiel unter 10 %, traf dann doch."

FRAGE DES BETREIBERS (06.08.2026): "Ich habe ein paar Mal beobachtet, dass eine
Quote (z. B. heute Tokio) schon bei 5 % Wahrscheinlichkeit war und dann doch
gewonnen hat. In wie vielen Faellen kam so etwas vor? Sagen wir: stand bei >30,
dann bei <10, und hat dann doch gewonnen — gegen verloren?"

Das ist die erste Frage dieser Serie, die unmittelbar handelbar waere: faellt
ein Bucket von ueber 30 % unter 10 % und trifft dann ueberdurchschnittlich oft,
kauft man ihn genau dort billig. Der Gegentest ist entscheidend und wird
mitgerechnet — die Basisrate aller Buckets, die unter 10 % fallen, OHNE vorher
ueber 30 % gestanden zu haben. Nur wenn die Turnaround-Gruppe besser abschneidet
als diese Vergleichsgruppe, steckt in der Vorgeschichte Information.

DATENQUELLE: Polymarket, oeffentlich und historisch, ohne Auth.
  Events   GET gamma-api.polymarket.com/events?slug=highest-temperature-in-{stadt}-on-{monat}-{tag}-{jahr}
  Preise   GET clob.polymarket.com/prices-history?market={yesTokenId}&startTs=&endTs=&fidelity=10
Settlement kommt aus `outcomePrices` des Marktes ("1" auf Yes = Bucket
getroffen) — keine Wetterquelle noetig, und damit auch keine METAR/WU-Falle.

DEFINITION Turnaround: es gibt Zeitpunkte t1 < t2 im Handelsverlauf mit
  p_yes(t1) >= HOCH (0,30)  und  p_yes(t2) <= TIEF (0,10).
Die Reihenfolge zaehlt: erst hoch, DANN tief. Ein Bucket, der nur billig war,
ist kein Turnaround.

FALLEN, bewusst behandelt:
  * fidelity=10 (10-Minuten-Punkte) statt 1 — fuer einen Preisverfall von 30 auf
    10 Prozentpunkte reicht das und spart das Zehnfache an Daten.
  * Nur Buecher mit echtem Handel: Buckets ohne Preishistorie werden gezaehlt,
    aber nicht gewertet. Ein Preis, den nie jemand gehandelt hat, ist kein Preis.
  * Der letzte Punkt der Reihe wird fuer den TIEF-Test ausgeschlossen, wenn er
    nach dem Settlement liegt — sonst zaehlt man das Aufloesen selbst als Verfall.

Aufruf:
    python weather_turnaround_eval.py --tage 30           # sammeln + rechnen
    python weather_turnaround_eval.py --tage 30 --cache-only   # nur rechnen
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import requests

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

GAMMA = "https://gamma-api.polymarket.com/events"
CLOB = "https://clob.polymarket.com/prices-history"
HOCH, TIEF = 0.30, 0.10
FEE = 0.07
USD = 5.0
CACHE = os.environ.get("TURNAROUND_CACHE", "turnaround_cache.jsonl")

STAEDTE = [
    "Amsterdam", "Ankara", "Beijing", "Buenos Aires", "Busan", "Cape Town",
    "Chengdu", "Chongqing", "Guangzhou", "Helsinki", "Hong Kong", "Istanbul",
    "Jeddah", "Karachi", "Kuala Lumpur", "London", "Lucknow", "Madrid",
    "Manila", "Mexico City", "Milan", "Moscow", "Munich", "Panama City",
    "Paris", "Qingdao", "Sao Paulo", "Seoul", "Shanghai", "Shenzhen",
    "Singapore", "Taipei", "Tel Aviv", "Tokyo", "Toronto", "Warsaw",
    "Wellington", "Wuhan",
]

S = requests.Session()
S.headers["User-Agent"] = "boersenbot-research/1.0"


def hole(url, params, tries=4):
    for versuch, pause in enumerate(((3, 8, 20, 0)), start=1):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(pause or 3)
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            if not pause:
                return None
            time.sleep(pause)
    return None


def slug(city, d):
    return (f"highest-temperature-in-{city.lower().replace(' ', '-')}"
            f"-on-{d.strftime('%B').lower()}-{d.day}-{d.year}")


def event_buckets(city, d):
    """Alle Buckets eines Stadt-Tages mit Token, Startzeit und Settlement."""
    j = hole(GAMMA, {"slug": slug(city, d)})
    if not j:
        return []
    out = []
    for m in (j[0].get("markets") or []):
        try:
            toks = json.loads(m.get("clobTokenIds") or "[]")
            preise = json.loads(m.get("outcomePrices") or "[]")
        except (json.JSONDecodeError, TypeError):
            continue
        if len(toks) < 2 or len(preise) < 2:
            continue
        if not m.get("closed"):
            continue                      # noch offen -> kein Settlement
        ts = m.get("acceptingOrdersTimestamp") or m.get("startDate")
        if not ts:
            continue
        out.append({
            "city": city, "target": d.isoformat(), "frage": m.get("question", "")[:90],
            "yes_token": toks[0], "start": ts,
            "getroffen": preise[0] == "1",
            "volumen": float(m.get("volumeNum") or 0),
        })
    return out


def preisreihe(b):
    """10-Minuten-Preisreihe des YES-Tokens ueber die ganze Laufzeit."""
    try:
        st = datetime.strptime(b["start"][:19], "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc)
    except ValueError:
        return None
    j = hole(CLOB, {"market": b["yes_token"], "fidelity": 10,
                    "startTs": int(st.timestamp()),
                    "endTs": int(st.timestamp()) + 4 * 24 * 3600})
    if not j:
        return None
    return [(p["t"], float(p["p"])) for p in (j.get("history") or [])]


def sammeln(tage, workers):
    """Cache fuellen. Eine JSONL-Zeile je Bucket."""
    gesehen = set()
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            for z in f:
                try:
                    r = json.loads(z)
                    gesehen.add((r["city"], r["target"], r["frage"]))
                except json.JSONDecodeError:
                    pass
    print(f"Cache: {len(gesehen)} Buckets bereits geholt.")

    heute = date.today()
    aufgaben = [(c, heute - timedelta(days=i))
                for i in range(1, tage + 1) for c in STAEDTE]
    print(f"{len(aufgaben)} Stadt-Tage zu pruefen ({tage} Tage x {len(STAEDTE)} Staedte).")

    fh = open(CACHE, "a", encoding="utf-8")
    n_neu = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, buckets in enumerate(ex.map(lambda a: event_buckets(*a), aufgaben)):
            if i % 60 == 0:
                print(f"  ... {i}/{len(aufgaben)} Stadt-Tage, {n_neu} Buckets neu")
            for b in buckets or []:
                if (b["city"], b["target"], b["frage"]) in gesehen:
                    continue
                reihe = preisreihe(b)
                b["reihe"] = reihe
                fh.write(json.dumps(b) + "\n")
                fh.flush()
                n_neu += 1
    fh.close()
    print(f"Fertig: {n_neu} Buckets neu im Cache.")


ENTSCHIEDEN_UNTEN, ENTSCHIEDEN_OBEN = 0.02, 0.98


def handelsreihe(reihe):
    """Preisreihe bis zu dem Punkt, an dem der Markt faktisch entschieden ist.

    ⚠️ OHNE DIESEN SCHNITT IST DIE GANZE AUSWERTUNG FALSCH. Die Rohreihe laeuft
    ueber das Settlement hinaus: nach der Aufloesung notiert der Verlierer bei
    0,000 und der Gewinner bei 1,000. Wer den Tiefstpreis ueber die ganze Reihe
    sucht, findet also immer die Aufloesung — im ersten Lauf ergab das fuer die
    Vergleichsgruppe einen "mittleren Kaufpreis" von 0,001 und +88 % ROI, also
    den Kauf eines bereits gewonnenen Marktes fuer einen Zehntelcent.

    Geschnitten wird beim ERSTEN Beruehren von <= 2 % oder >= 98 %. Das ist
    konservativ: es verwirft auch echte Handelspunkte in diesen Zonen, aber die
    Frage des Betreibers zielt auf 5-10 %, nicht auf 0,5 %.
    """
    preise = []
    for _, p in reihe:
        if p <= ENTSCHIEDEN_UNTEN or p >= ENTSCHIEDEN_OBEN:
            break
        preise.append(p)
    return preise


def auswerten():
    rows = []
    with open(CACHE, encoding="utf-8") as f:
        for z in f:
            try:
                rows.append(json.loads(z))
            except json.JSONDecodeError:
                pass
    print(f"\nCache: {len(rows)} Buckets.")
    mit_reihe = [r for r in rows if r.get("reihe")]
    print(f"davon mit Preisreihe: {len(mit_reihe)} "
          f"({100*len(mit_reihe)/max(len(rows),1):.0f} %) — "
          f"ohne Handel kein Preis, die uebrigen sind nicht wertbar.\n")

    turn, nur_tief, rest = [], [], []
    for r in mit_reihe:
        preise = handelsreihe(r["reihe"])
        if not preise:
            continue
        # erst HOCH, DANN TIEF — Reihenfolge zaehlt
        idx_hoch = next((i for i, p in enumerate(preise) if p >= HOCH), None)
        tief_nach_hoch = (idx_hoch is not None
                          and any(p <= TIEF for p in preise[idx_hoch + 1:]))
        je_tief = any(p <= TIEF for p in preise)
        if tief_nach_hoch:
            # Preis am ersten Tiefpunkt NACH dem Hoch = wo man kaufen wuerde
            kauf = next(p for p in preise[idx_hoch + 1:] if p <= TIEF)
            turn.append((r, kauf))
        elif je_tief:
            nur_tief.append((r, min(preise)))
        else:
            rest.append((r, min(preise)))

    def block(name, gruppe):
        n = len(gruppe)
        if not n:
            print(f"{name}: keine Faelle")
            return None
        tr = sum(1 for r, _ in gruppe if r["getroffen"])
        preis = sum(p for _, p in gruppe) / n
        print(f"{name}")
        print(f"   {n:5d} Buckets, davon GETROFFEN {tr} = {100*tr/n:.2f} %  "
              f"(mittlerer Kaufpreis {preis:.3f})")
        return n, tr, preis

    print("=" * 78)
    a = block(f"TURNAROUND — stand ueber {HOCH:.0%}, fiel dann unter {TIEF:.0%}:", turn)
    b = block(f"VERGLEICHSGRUPPE — fiel unter {TIEF:.0%}, war aber NIE ueber {HOCH:.0%}:",
              nur_tief)
    block(f"REST — nie unter {TIEF:.0%}:", rest)
    print("=" * 78)

    if a and b:
        na, tra, pa = a
        nb, trb, pb = b
        qa, qb = tra / na, trb / nb
        print(f"\nTrifft die Turnaround-Gruppe oefter? {100*qa:.2f} % gegen "
              f"{100*qb:.2f} % = {100*(qa-qb):+.2f} pp")
        # Zweistichproben-z auf Anteile
        import math
        p = (tra + trb) / (na + nb)
        se = math.sqrt(p * (1 - p) * (1 / na + 1 / nb))
        z = (qa - qb) / se if se else 0.0
        print(f"   z = {z:+.2f}  ({'signifikant' if abs(z) > 1.96 else 'NICHT signifikant'})")

        # Was haette ein YES-Kauf am Tiefpunkt gebracht?
        for name, gruppe in (("Turnaround", turn), ("Vergleich", nur_tief)):
            ges = 0.0
            for r, kauf in gruppe:
                if kauf <= 0 or kauf >= 1:
                    continue
                n_k = USD / kauf
                ges += (n_k - USD - FEE * n_k * min(kauf, 1 - kauf)
                        if r["getroffen"] else -USD)
            einsatz = len(gruppe) * USD
            print(f"   YES-Kauf am Tiefpunkt, {name:10s}: {ges:+9.2f} $ auf "
                  f"{einsatz:8.0f} $ = {100*ges/einsatz:+7.2f} % ROI")

    # Die Beobachtung des Betreibers nachschlagen
    print("\nDie extremsten Turnarounds (tiefster Kaufpreis, trotzdem getroffen):")
    tref = sorted([t for t in turn if t[0]["getroffen"]], key=lambda x: x[1])[:10]
    for r, kauf in tref:
        print(f"   {r['target']}  {r['city']:14s} {r['frage'][:52]:52s} "
              f"tiefster Preis {kauf:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tage", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-only", action="store_true")
    args = ap.parse_args()
    if not args.cache_only:
        sammeln(args.tage, args.workers)
    auswerten()
    return 0


if __name__ == "__main__":
    sys.exit(main())
