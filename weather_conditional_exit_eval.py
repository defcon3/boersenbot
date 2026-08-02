# -*- coding: utf-8 -*-
"""weather_conditional_exit_eval.py — Auswertung zur Pre-Reg vom 01.08.2026.

Gehoert zu `preregs/weather_conditional_exit_2026_08_01.md`. Die Regel, exakt
wie dort vorregistriert:

    Verkaufe die volle NO-Position, sobald die settelnde Quelle fuer den Zieltag
    einen Wert meldet, dessen Rundung dem gelayten Bucket k entspricht.

    Variante A: ohne Zeitfilter, feuert bei der ersten Beobachtung.
    Variante B: feuert fruehestens ab 16:20 Ortszeit.

Getestet wird der **informierte** Ausstieg. Der uninformierte (Preis laeuft weg,
These noch offen) ist bereits gemessen und kostet -8,00 % — das ist NICHT der
Gegenstand hier.

ZWEI KORREKTUREN GEGENUEBER DER SONDIERUNG
------------------------------------------
1. **Der Kandidatenbegriff.** Die Pre-Reg nennt n = 330 Kandidaten / 61
   Verlierer. Das ist die Zahl der ZEILEN in bb_WeatherLadders — die Leiter wird
   je Zieltag mehrfach geschrieben (Lead 0/1/2 und am 26.07. vier Testlaeufe am
   selben Tag). Eine Position ist aber ein (Stadt, Zieltag, k)-Tripel; derselbe
   Bucket am selben Tag ist EINE Wette, nicht drei. Entduplizert bleiben
   **223 Kandidaten / 47 Verlierer** (Stand 01.08., wachsend).
   Folge: die Mengenbedingung von G1 (n >= 250) ist heute nicht erfuellbar. Das
   wird unten ausgewiesen und NICHT stillschweigend abgesenkt.
2. **Einstieg = fruehester qualifizierender Snapshot.** buy_no und offset_fav
   aendern sich zwischen den Snapshots; genommen wird der erste, in dem der
   Kandidat die Bandbedingung erfuellt — so, wie der Autobuy ihn gesehen haette.
   Die Regel kann folglich nur NACH diesem Zeitpunkt feuern.

LATENZ — die zentrale Designfalle
---------------------------------
Ein Backtest, der auf `valid_time_gmt` ausloest, unterstellt Wissen, das zu
diesem Zeitpunkt niemand hatte. Der Sichtzeitpunkt ist deshalb

    Sichtzeitpunkt := valid_time_gmt + stationsspezifischer WU-Latenzaufschlag

mit dem **oberen Ende** der Sondenschaetzung (P90 aus `weather_source_latency.csv`,
`--latency-quantile` aenderbar) und pauschal 1800 s, wo keine Stationsschaetzung
vorliegt. Bewusst pessimistisch: ein Effekt, der nur bei optimistischer Latenz
ueberlebt, ist keiner.

`--source noaa` rechnet denselben Lauf mit den NOAA-Aufschlaegen — das ist die
Kopplung der Screen-Frage (WU oder NOAA) an Geld. Settlement bleibt in beiden
Faellen WU.

Aufruf:
    python weather_conditional_exit_eval.py                 # G1, Variante A+B
    python weather_conditional_exit_eval.py --source noaa   # Gegenrechnung
    python weather_conditional_exit_eval.py --detail        # Zeile je Ausloesung
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pymssql

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weather_foreknowledge_eval import (condition_id, trades,  # noqa: E402
                                        wu_observations)
from weather_stations import favorit_k, station_info  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

DB_CONFIG = {
    "server": "158.181.48.77",
    "database": "dbdata",
    "user": "326773",
    "password": "Extaler11!",
}

LATENZ_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "weather_source_latency.csv")
LATENZ_PAUSCHAL = 1800.0   # s, wo keine Stationsschaetzung vorliegt (Pre-Reg)
SPREAD = 0.01              # ct zu unseren Ungunsten (gemessener eff. Spread)
FEE = 0.036                # zweite Gebuehr, auf min(p, 1-p)
G4_FENSTER = 30 * 60       # s, in denen ein Tape-Trade existieren muss
B_STUNDE, B_MINUTE = 16, 20  # Variante B, Ortszeit


# ---------------------------------------------------------------- Latenz ----

def latenz_tabelle(quelle, quantil):
    """({ICAO: Aufschlag in s}, Pauschalwert) aus der Sonde.

    Erste Runde raus (Startartefakt: beim Start ist jede Beobachtung 'neu', die
    Sichtung also kuenstlich spaet).

    Der Pauschalwert fuer Stationen ohne eigene Schaetzung ist quellenabhaengig:
    fuer WU gilt der in der Pre-Reg fixierte Wert von 1800 s, weil WU je Station
    extrem streut (Median 262 s an KMIA gegen 2102 s an EGLC). NOAA und IEM sind
    einheitliche Feeds mit enger Streuung — dort ist der gepoolte P90 die
    ehrlichere Schaetzung als ein aus WU geliehener Pauschalwert.

    'null' rechnet ohne Aufschlag. Das ist KEIN handelbares Szenario, sondern
    die Diagnose: bleibt der Effekt auch bei perfekter Information aus, liegt es
    nicht an der Quelle."""
    if quelle == "null":
        return {}, 0.0
    if not os.path.exists(LATENZ_CSV):
        print(f"WARNUNG: {LATENZ_CSV} fehlt — alles auf {LATENZ_PAUSCHAL:.0f} s")
        return {}, LATENZ_PAUSCHAL
    rows = list(csv.DictReader(open(LATENZ_CSV, encoding="utf-8")))
    if not rows:
        return {}, LATENZ_PAUSCHAL
    erste = min(r["seen_utc"] for r in rows)
    per = defaultdict(list)
    for r in rows:
        if r["seen_utc"] == erste or r["source"].lower() != quelle.lower():
            continue
        per[r["station"]].append(float(r["lag_s"]))
    tab = {st: float(np.percentile(v, quantil)) for st, v in per.items()
           if len(v) >= 5}
    if quelle == "wu":
        return tab, LATENZ_PAUSCHAL
    alle = [x for v in per.values() for x in v]
    return tab, (float(np.percentile(alle, quantil)) if alle else LATENZ_PAUSCHAL)


# ------------------------------------------------------------ Kandidaten ----

def lade_kandidaten():
    """Ein Kandidat je (city, target_date, k) — fruehester Snapshot im Band."""
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT city, icao, target_date, k, buy_no, snapshot_utc, settle_k,
               wu_settle_k, market_id
        FROM bb_WeatherLadders
        WHERE var='max' AND kind='eq' AND offset_fav=-1
          AND buy_no >= 0.70 AND buy_no < 0.90
          AND settle_k IS NOT NULL AND market_id IS NOT NULL
        ORDER BY snapshot_utc""")
    best = {}
    for city, icao, td, k, buy_no, snap, settle_k, wu_k, mid in cur.fetchall():
        schluessel = (city, td, k)
        if schluessel in best:            # nach snapshot_utc sortiert -> erster gilt
            continue
        best[schluessel] = {
            "city": city, "icao": icao, "target_date": td, "k": k,
            "buy_no": float(buy_no),
            "einstieg": snap.replace(tzinfo=timezone.utc).timestamp(),
            "settle_k": settle_k, "wu_settle_k": wu_k, "market_id": mid,
        }
    conn.close()
    return sorted(best.values(), key=lambda x: (x["target_date"], x["city"], x["k"]))


def erste_beobachtung(obs, city, k, ausloeser):
    """(valid_time_gmt, temp) der ersten Zeile, die den Ausloeser erfuellt.

    Gerundet wird ueber favorit_k, nie von Hand: Hong Kong hat floor-Buckets,
    und Pythons round() ist banker's rounding (round(22.5) == 22).

    ZWEI LESARTEN — die Pre-Reg ist hier unterspezifiziert
    ------------------------------------------------------
    Woertlich steht dort "round(temp) == k in einer Tabellenzeile". Das trifft
    aber auch die Zeilen auf dem WEG nach oben: bei k = 25 und einem Tagesgang
    24 -> 25 -> 26 -> 27 feuert die Regel um 10 Uhr morgens, obwohl die Position
    danach sicher gewinnt. Gemessen (Lauf 01.08.): so feuert sie bei 90 % aller
    Kandidaten, und 155 der 199 Ausloesungen sind spaeter doch Gewinner.

    Gemeint ist erkennbar das LAUFENDE TAGESMAXIMUM:
      * H1 der Pre-Reg lautet "P(Endmax > k | k wurde beobachtet)" — eine Frage,
        die nur Sinn hat, wenn k das bisherige Hoch ist;
      * der Anlassfall Chengdu ist ein monotoner Anstieg 25 -> 26 -> 27, dort
        fallen beide Lesarten zusammen;
      * die Guardrail-Messung vom 24.07. definiert das Signal bereits so
        ("gerundetes laufendes Tagesmaximum sitzt exakt auf dem gelayten Bucket").

    Fuer ein Tagesmaximum-Brett ist der Unterschied entscheidend:
      laufendes Max <  k  -> Bucket noch nicht erreicht, Position gesund
      laufendes Max == k  -> Gefahr, hier kann das Tagesmax stehenbleiben
      laufendes Max >  k  -> Bucket uebersprungen, Position sicher gewonnen

    Default ist deshalb 'max'. 'momentan' bleibt als Kontrolle abrufbar und
    zaehlt NICHT als eigener Gate-Kandidat (sonst waeren es vier Tests statt
    der zwei vorregistrierten).
    """
    laufend = None
    for o in sorted(obs, key=lambda x: x["valid_time_gmt"]):
        t = int(o["valid_time_gmt"])
        bucket = favorit_k(float(o["temp"]), city)
        if ausloeser == "momentan":
            if bucket == k:
                return t, float(o["temp"])
            continue
        if laufend is None or bucket > laufend:
            laufend = bucket
            if laufend == k:
                return t, float(o["temp"])
            if laufend > k:
                return None, None      # uebersprungen — die Regel feuert nie
    return None, None


def lokale_sperre(target_date, tzname):
    """Unix-Zeit von 16:20 Ortszeit am Zieltag (Variante B)."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tzname)
    except Exception:
        return None
    return datetime(target_date.year, target_date.month, target_date.day,
                    B_STUNDE, B_MINUTE, tzinfo=tz).timestamp()


def sichtbares_max(obs, city, ts, aufschlag):
    """Laufendes Tagesmaximum, wie es zum Zeitpunkt ts SICHTBAR war.

    Sichtbar ist eine Beobachtung erst valid_time + Latenzaufschlag spaeter —
    ohne diesen Filter unterstellt der Test Wissen, das es nicht gab."""
    werte = [favorit_k(float(o["temp"]), city) for o in obs
             if int(o["valid_time_gmt"]) + aufschlag <= ts]
    return max(werte) if werte else None


def no_preis(t):
    """NO-Preis eines Tape-Trades (das Tape notiert die gehandelte Seite)."""
    p = float(t["price"])
    return 1.0 - p if t["outcome"] == "Yes" else p


def naechster_trade(tr, ab_ts):
    """Erster Trade ab ab_ts (G4: muss innerhalb G4_FENSTER liegen)."""
    spaeter = [t for t in tr if int(t["timestamp"]) >= ab_ts]
    if not spaeter:
        return None
    t = min(spaeter, key=lambda x: int(x["timestamp"]))
    return t if int(t["timestamp"]) - ab_ts <= G4_FENSTER else None


# ------------------------------------------------------------- Statistik ----

def one_sample_t(a):
    a = np.asarray(a, float)
    if len(a) < 2 or a.std(ddof=1) == 0:
        return float("nan")
    return a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))


def buch_roi(faelle, mit_ausstieg):
    """ROI je eingesetztem Dollar bei gleicher Dollargroesse je Position.

    Kontrakte = E / buy_no, Rueckfluss = 1/0 beim Halten bzw. exit-fee beim
    Ausstieg. Mit E = 1 normiert ist der ROI das Mittel von
    (rueckfluss - buy_no) / buy_no."""
    r = []
    for f in faelle:
        p_ein = f["buy_no"]
        if mit_ausstieg and f.get("exit_netto") is not None:
            rueck = f["exit_netto"]
        else:
            rueck = 1.0 if f["gewinnt"] else 0.0
        r.append((rueck - p_ein) / p_ein)
    return np.array(r, float)


# ----------------------------------------------------------------- Lauf -----

def auswerten(a):
    lat, pauschal = latenz_tabelle(a.source, a.latency_quantile)
    kand = lade_kandidaten()
    print(f"Quelle fuer den Sichtzeitpunkt: {a.source.upper()}  "
          f"(P{a.latency_quantile:.0f} der Sonde, sonst {pauschal:.0f} s)")
    if lat:
        print("  Stationsschaetzungen: " +
              ", ".join(f"{s} {v:.0f}s" for s, v in sorted(lat.items())))
    print(f"\nKandidaten (entdupliziert): {len(kand)}   "
          f"Verlierer: {sum(1 for x in kand if x['settle_k'] == x['k'])}")
    print(f"Zeitraum: {kand[0]['target_date']} bis {kand[-1]['target_date']}, "
          f"{len({x['city'] for x in kand})} Staedte\n")

    faelle, verworfen = [], defaultdict(int)
    for i, c in enumerate(kand, 1):
        if i % 25 == 0:
            print(f"  ... {i}/{len(kand)}", flush=True)
        st = station_info(c["icao"]) or {}
        obs = wu_observations(c["icao"], c["target_date"])
        if not obs:
            verworfen["keine WU-Reihe (u. a. Hong Kong/HKO)"] += 1
            continue
        if not st.get("tz"):
            verworfen["keine Zeitzone"] += 1
            continue

        f = dict(c)
        f["gewinnt"] = c["settle_k"] != c["k"]
        f["auswertbar"] = True

        aufschlag = lat.get(c["icao"], pauschal)
        f["aufschlag"] = aufschlag

        if a.variante_b:
            # Variante B ist eine PRUEFUNG zur Ortsstunde, keine verzoegerte
            # Reaktion auf ein Ereignis von morgens: sitzt das bis 16:20
            # sichtbare Tagesmaximum noch auf k? Steht es hoeher, ist die
            # Position sicher und es gibt nichts zu verkaufen. So ist auch der
            # Waechter vom 24.07. definiert, auf den die Pre-Reg verweist.
            pruef = lokale_sperre(c["target_date"], st["tz"])
            if pruef is None:
                verworfen["Zeitzone unbekannt"] += 1
                continue
            m = sichtbares_max(obs, c["city"], pruef, aufschlag)
            if m != c["k"]:
                f["feuert"] = False
                faelle.append(f)
                continue
            t_valid, temp, sicht = pruef - aufschlag, None, max(pruef, c["einstieg"])
        else:
            t_valid, temp = erste_beobachtung(obs, c["city"], c["k"], a.trigger)
            if t_valid is None:
                # Bucket nie als laufendes Maximum beobachtet (oder
                # uebersprungen) -> die Position laeuft bis Settlement.
                f["feuert"] = False
                faelle.append(f)
                continue
            # Verkauft werden kann fruehestens nach dem Kauf.
            sicht = max(t_valid + aufschlag, c["einstieg"])

        cid = condition_id(c["market_id"])
        if not cid:
            verworfen["keine conditionId"] += 1
            continue
        tr = trades(cid)
        if not tr:
            verworfen["kein Tape"] += 1
            continue

        f.update(feuert=True, t_valid=t_valid, temp=temp, sicht=sicht)
        t = naechster_trade(tr, sicht)
        if t is None:
            # G4: kein Handel im Fenster -> gilt als NICHT ausgestiegen,
            # nie als Ausstieg zum letzten bekannten Preis.
            f.update(handelbar=False, exit_roh=None, exit_netto=None)
        else:
            roh = no_preis(t)
            netto = max(0.0, roh - SPREAD)
            netto -= FEE * min(netto, 1.0 - netto)
            f.update(handelbar=True, exit_roh=roh, exit_netto=netto,
                     exit_ts=int(t["timestamp"]))
        faelle.append(f)

    if verworfen:
        print("\nVerworfen:")
        for g, n in sorted(verworfen.items(), key=lambda x: -x[1]):
            print(f"  {g:<38}{n}")

    return faelle


def bericht(faelle, a):
    n = len(faelle)
    feuert = [f for f in faelle if f["feuert"]]
    handelbar = [f for f in feuert if f.get("handelbar")]
    variante = "B (ab 16:20 Ortszeit)" if a.variante_b else "A (ohne Zeitfilter)"
    trig = ("laufendes Max erreicht k" if a.trigger == "max"
            else "KONTROLLE: jede Zeile mit Wert k")

    print(f"\n{'='*78}")
    print(f"VARIANTE {variante}   —   Quelle {a.source.upper()}   —   {trig}")
    print(f"{'='*78}")
    print(f"auswertbare Kandidaten      {n}")
    print(f"Regel feuert                {len(feuert)}  ({len(feuert)/n*100:.1f} %)")
    print(f"davon handelbar (G4)        {len(handelbar)}  "
          f"({len(handelbar)/len(feuert)*100:.1f} % der Ausloesungen)"
          if feuert else "")

    if not handelbar:
        print("\nKeine handelbare Ausloesung — nicht auswertbar.")
        return None

    # --- Falle 3 der Pre-Reg: die Regel feuert auch bei Gewinnern ---
    gew = sum(1 for f in feuert if f["gewinnt"])
    print(f"  darunter spaeter doch gewonnen (Bucket ueberschritten): {gew}"
          f"  von {len(feuert)}")

    # --- H1: eingepreiste vs. realisierte Erholungsrate ---
    # NO gewinnt nach der Beobachtung nur noch, wenn das Maximum weiter steigt.
    # Der NO-Preis zum Ausstiegszeitpunkt IST also die eingepreiste Rate.
    p = np.array([f["exit_roh"] for f in handelbar], float)
    y = np.array([1.0 if f["gewinnt"] else 0.0 for f in handelbar], float)
    d = p - y
    t1 = one_sample_t(d)
    print(f"\nH1  eingepreiste Erholungsrate   {p.mean():.3f}")
    print(f"    realisierte Erholungsrate    {y.mean():.3f}")
    print(f"    Differenz                    {d.mean():+.3f}  "
          f"(Median {np.median(d):+.3f})   t = {t1:+.2f}")

    # --- Wer wird gerettet, wer gekappt? Die Rechnung hinter dem Urteil ---
    gerettet = [f for f in handelbar if not f["gewinnt"]]
    gekappt = [f for f in handelbar if f["gewinnt"]]
    v_ret = sum(f["exit_netto"] / f["buy_no"] for f in gerettet)
    v_kap = sum((f["exit_netto"] - 1.0) / f["buy_no"] for f in gekappt)
    print(f"\n    Verlierer gerettet   {len(gerettet):>3}  -> {v_ret:+.2f} Einsaetze")
    print(f"    Gewinner gekappt     {len(gekappt):>3}  -> {v_kap:+.2f} Einsaetze")
    print(f"    Saldo                     {v_ret + v_kap:+.2f} Einsaetze")

    # --- H2/G3: Geld, netto nach der zweiten Gebuehr ---
    roi_h = buch_roi(faelle, False)
    roi_e = buch_roi(faelle, True)
    print(f"\nH2  ROI Halte-Buch               {roi_h.mean()*100:+.2f} %")
    print(f"    ROI mit konditionalem Ausstieg {roi_e.mean()*100:+.2f} %  "
          f"(netto nach {FEE*100:.1f} % Gebuehr + {SPREAD*100:.0f} ct Spread)")
    diff = roi_e - roi_h
    aktiv = diff[diff != 0]
    print(f"    Differenz                    {diff.mean()*100:+.2f} pp   "
          f"t = {one_sample_t(aktiv):+.2f} (n={len(aktiv)} veraenderte Positionen)")

    # --- G5: Robustheit ---
    print("\nG5  Robustheit")
    je_stadt = defaultdict(list)
    for f, x in zip(handelbar, d):
        je_stadt[f["city"]].append(x)
    groesste = max(je_stadt, key=lambda c: len(je_stadt[c]))
    ohne = np.array([x for f, x in zip(handelbar, d) if f["city"] != groesste], float)
    print(f"    ohne '{groesste}' ({len(je_stadt[groesste])} Faelle): "
          f"{ohne.mean():+.3f}  t = {one_sample_t(ohne):+.2f}")
    je_tag = defaultdict(float)
    for f, x in zip(handelbar, d):
        je_tag[str(f["target_date"])] += x
    # Bei einem Effekt nahe null ist ein Anteil am Gesamteffekt bedeutungslos
    # (Division durch fast null) — dann gar nicht erst ausweisen.
    if abs(d.sum()) > 0.5:
        top_tag = max(je_tag, key=lambda t: abs(je_tag[t]))
        anteil = je_tag[top_tag] / d.sum() * 100
        print(f"    groesster Zieltag {top_tag}: {anteil:.0f} % des Effekts "
              f"({'PASS' if abs(anteil) <= 30 else 'FAIL'}, Gate 30 %)")
    else:
        print("    Zieltag-Konzentration: Effekt zu klein zum Aufteilen")

    # --- Gates ---
    print(f"\n{'-'*78}")
    print("GATES (Bonferroni: zwei Varianten -> t > 2.5)")
    menge_ok = n >= 250
    print(f"  G1a Menge n >= 250              n={n}  "
          f"{'PASS' if menge_ok else 'FAIL — Universum ist heute kleiner'}")
    print(f"  G1b Differenz >= 5 pp, t > 2.5  {d.mean()*100:+.1f} pp, t={t1:+.2f}  "
          f"{'PASS' if (d.mean() >= 0.05 and t1 > 2.5) else 'FAIL'}")
    print(f"  G3  netto positiv               {diff.mean()*100:+.2f} pp  "
          f"{'PASS' if diff.mean() > 0 else 'FAIL'}")
    q = len(handelbar) / len(feuert)
    print(f"  G4  >= 80 % handelbar           {q*100:.0f} %  "
          f"{'PASS' if q >= 0.80 else 'FAIL'}")
    h3 = len(feuert) / n
    print(f"  H3  feuert bei < 25 %           {h3*100:.0f} %  "
          f"{'PASS' if h3 < 0.25 else 'FAIL'}")
    print("  G2  Forward — offen (ab 02.08., siehe Pre-Reg)")

    if a.detail:
        print(f"\n{'Datum':<12}{'Stadt':<15}{'k':>3}{'Ein':>6}{'Exit':>7}"
              f"{'Aussch':>8}{'Ausgang':>9}{'Vorteil':>9}")
        for f in sorted(handelbar, key=lambda x: (x["target_date"], x["city"])):
            v = f["exit_netto"] - (1.0 if f["gewinnt"] else 0.0)
            print(f"{str(f['target_date']):<12}{f['city'][:14]:<15}{f['k']:>3}"
                  f"{f['buy_no']:>6.2f}{f['exit_roh']:>7.3f}"
                  f"{f['aufschlag']:>7.0f}s"
                  f"{'gewinnt' if f['gewinnt'] else 'verliert':>9}{v:>+9.3f}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="wu", choices=["wu", "noaa", "iem", "null"],
                    help="Quelle, aus der der Latenzaufschlag stammt "
                         "(Settlement bleibt immer WU). 'null' = ohne Aufschlag, "
                         "Diagnose statt handelbares Szenario")
    ap.add_argument("--latency-quantile", type=float, default=90.0,
                    help="Quantil der Sondenlatenz (Pre-Reg: oberes Ende, nicht Median)")
    ap.add_argument("--variante", default="beide", choices=["a", "b", "beide"])
    ap.add_argument("--trigger", default="max", choices=["max", "momentan"],
                    help="'max': laufendes Tagesmaximum erreicht k (Default, die "
                         "gemeinte Lesart). 'momentan': jede Tabellenzeile mit "
                         "Wert k — Kontrolle, kein Gate-Kandidat")
    ap.add_argument("--detail", action="store_true", help="Zeile je Ausloesung")
    a = ap.parse_args()

    for v in (["a", "b"] if a.variante == "beide" else [a.variante]):
        a.variante_b = (v == "b")
        faelle = auswerten(a)
        if faelle:
            bericht(faelle, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
