#!/usr/bin/env python3
"""Was haette Autobuy V1 seit der V2-Umstellung gebracht?

FRAGE DES BETREIBERS (06.08.2026): "was haetten die Wetten nach Version 1 bis
jetzt gebracht?" — also die kontrafaktische Rechnung fuer das Fenster, das V2
tatsaechlich gefahren hat.

FENSTER: Zieltage 2026-07-28 bis 2026-08-06. V1 lief bis Zieltag 27.07.
einschliesslich (Tag `autobuy-v1`), V2 ab Zieltag 28.07.

DATENQUELLE: preregs/weather_minus1_live_log.csv vom VPS. Der Log enthaelt
JEDEN geprueften Kandidaten mit Live-Preis — das ist kein Zufall: das
Spannen-Veto laeuft in weather_minus1_autobuy.py bewusst VOR dem Preisband,
damit genau diese Rekonstruktion moeglich bleibt (Kommentar dort, Z. 395 ff.).

AUSWAHLREGEL V1 (woertlich aus dem Tag `autobuy-v1`):
  Filter    : Live-NO <= 0,97 | Modellspanne <= 3,0 K | nicht im Bestand
  Rangfolge : NO-Preis ABSTEIGEND
  Guete-Gate: erste 3 Picks bedingungslos, jeder weitere bis Cap 6 nur
              mit Live-NO >= 0,85
  Einsatz   : 5 $/Lay

AUSWAHLREGEL V2 (Ist): Band 0,70 <= NO < 0,90, Rang nach Temperaturabstand,
Cap 8. Wird NICHT simuliert, sondern aus dem Log als tatsaechlich gekauft
uebernommen (decision bought / sent_unverified).

RECHNUNG: identisch zu weather_minus1_shadow / _ppess_filter —
5 $/Lay, Fee 0,07*n*min(NO,1-NO), Auszahlung nur bei Gewinn.

OFFENE ZIELTAGE: 05. und 06.08. sind in bb_WeatherLadders noch ohne settle_k —
settle() settelt konservativ erst ab Folgetag 12:00 UTC, der Ladder-Timer holt
das um 14:30 nach. Auf Anweisung des Betreibers (06.08.) werden beide Tage als
gesettelt behandelt und voll eingerechnet; der Ist-Wert kommt per METAR direkt
von IEM (Hong Kong: HKO-Daily-Extract, kein METAR vorhanden).
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pymssql
import requests

sys.path.insert(0, ".")
from weather_stations import station_info  # noqa: E402

DB_CONFIG = {
    "server": "158.181.48.77",
    "database": "dbdata",
    "user": "326773",
    "password": "Extaler11!",
}
IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

USD = 5.0
FEE = 0.07
V1_CAP = 6
V1_FREI = 3          # erste drei Picks ohne Guete-Gate
V1_GUETE = 0.85
V1_PREIS_CAP = 0.97
FENSTER_VON = "2026-07-29"
FENSTER_BIS = "2026-08-06"
PM_API = "https://prediction-market-api.jup.ag/api"

# Entscheidungen, bei denen der Kandidat VOR dem Spannen-Veto ausschied — dort
# steht kein verwertbarer Live-Preis, und V1 haette ihn genauso wenig gesehen.
VOR_VETO = ("skip_position", "skip_no_mu", "skip_closed", "skip_api",
            "skip_noforecast", "dry_run")
GEKAUFT = ("bought", "sent_unverified")


def pnl_lay(no, verloren):
    """Identisch zu weather_minus1_ppess_filter.pnl_lay."""
    n = USD / no
    fee = FEE * n * min(no, 1.0 - no)
    return -USD if verloren else n - USD - fee


def lade_log(pfad):
    """Log lesen, auf das Fenster schneiden, Doppellaeufe entfernen.

    Zieltag 06.08. wurde zweimal gefahren (15:41 und 15:43 UTC am 05.08.).
    Dedupliziert wird je (Zieltag, Stadt, k) — und behalten wird die Zeile MIT
    Kauf, sonst die erste.

    ⚠️ Beim ersten Lauf dieser Auswertung stand hier "erster Lauf je Zieltag",
    mit der Begruendung, der erste habe die Kaeufe ausgeloest. Das war eine
    Annahme, keine Messung, und sie war FALSCH: der Lauf 15:41 kaufte nichts,
    der Lauf 15:43 kaufte Tokyo, Chengdu und Wellington. Die Regel hat damit
    genau die drei Positionen unterschlagen, die im Wallet stehen — V2 erschien
    am 06.08. mit null Lays. Aufgefallen ist es nur, weil der Betreiber sein
    Konto dagegengehalten hat.
    """
    with open(pfad, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if FENSTER_VON <= r["target_date"] <= FENSTER_BIS]
    best = {}
    for r in rows:
        key = (r["target_date"], r["city"], r["k"])
        kauf = r["decision"].startswith(GEKAUFT)
        if key not in best:
            best[key] = r
        elif kauf and not best[key]["decision"].startswith(GEKAUFT):
            best[key] = r
    behalten = list(best.values())
    entfernt = len(rows) - len(behalten)
    if entfernt:
        print(f"  Doppellauf-Bereinigung: {entfernt} Zeilen zusammengefuehrt "
              f"(Kaufzeile hat Vorrang).")
    return behalten


def kandidaten(rows):
    """Je Zieltag die Menge, die BEIDEN Versionen offenstand.

    Das Spannen-Veto (>3,0 K) gilt in V1 und V2 gleichermassen und wird hier
    mitgefiltert; skip_spread faellt also in beiden Versionen raus.
    """
    tage = defaultdict(list)
    for r in rows:
        if r["decision"].startswith(VOR_VETO):
            continue
        if r["decision"].startswith("skip_spread"):
            continue
        if not r["buy_no_live"]:
            continue
        tage[r["target_date"]].append({
            "city": r["city"],
            "k": int(r["k"]),
            "no": float(r["buy_no_live"]),
            "abstand": float(r["abstand"]) if r["abstand"] else None,
            "gekauft_v2": r["decision"].startswith(GEKAUFT),
        })
    return tage


def waehle_v1(cands):
    """V1-Auswahl: NO absteigend, erste 3 frei, danach nur NO >= 0,85, Cap 6."""
    pool = sorted([c for c in cands if c["no"] <= V1_PREIS_CAP],
                  key=lambda c: -c["no"])
    picks = []
    for c in pool:
        if len(picks) >= V1_CAP:
            break
        if len(picks) >= V1_FREI and c["no"] < V1_GUETE:
            continue
        picks.append(c)
    return picks


def icao_map(conn):
    """Stadt -> ICAO aus der Ladder-Tabelle. Die Zuordnung steht dort, nicht in
    weather_stations (dort gibt es nur SPECIAL_STATIONS als Ausnahmeliste)."""
    cur = conn.cursor()
    cur.execute("SELECT city, MAX(icao) FROM bb_WeatherLadders "
                "WHERE icao IS NOT NULL GROUP BY city")
    return {c: (i or "").strip() for c, i in cur.fetchall()}


def markt_ids(conn):
    """(Zieltag, Stadt, k) -> market_id aus bb_WeatherLadders."""
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT target_date, city, k, market_id FROM bb_WeatherLadders "
        "WHERE var='max' AND kind='eq' AND target_date >= %s AND target_date <= %s "
        "AND market_id IS NOT NULL", (FENSTER_VON, FENSTER_BIS))
    out = {}
    for target, city, k, mid in cur.fetchall():
        if isinstance(target, datetime):
            target = target.date()
        out[(str(target), city, int(k))] = mid
    return out


# Zieltag 06.08.: Jupiter hat die Maerkte noch nicht geschlossen, `result` ist
# leer. Der Betreiber hat entschieden, den Tag voll einzurechnen ("da passiert
# nichts mehr"). Ausgang deshalb per METAR bestimmt, Stand 06.08. mittags:
#   sicher  — Ist liegt bereits jenseits des Buckets, das Max kann nur steigen:
#             Wuhan 36 (Lay 34), Kuala Lumpur 34 (32), Cape Town 15 (14),
#             Tokyo 31 (31, getroffen), Wellington 10 (10, getroffen)
#   knapp   — nur 1 K Abstand und letzte Obs 12-13 h Ortszeit, koennte noch
#             kippen: Ankara 30 (Lay 31), Munich 24 (Lay 25)
#   Chengdu 26 gegen Lay 31 — Lay gewinnt.
ERGEBNIS_0608 = {
    ("Wuhan", 34): False, ("Kuala Lumpur", 32): False, ("Cape Town", 14): False,
    ("Paris", 24): False, ("Ankara", 31): False, ("Munich", 25): False,
    ("Tokyo", 31): True, ("Wellington", 10): True, ("Chengdu", 31): False,
}


def markt_ergebnis(mid, cache):
    """Hat der Bucket getroffen? Quelle ist Jupiters eigenes `result` — also
    exakt das, wogegen die Boerse abgerechnet hat.

    WARUM NICHT WETTER: Am 05.08. sagte METAR fuer Shenzhen 31,0 Grad, der Markt
    settelte aber gegen 32 (Bucket 32 = result 'yes', die reale Position verlor
    4,83 $). Weder METAR noch Wunderground taugen dort als Wahrheit — ZGSZ steht
    in NO_WUNDERGROUND, und die METAR-Reihe trifft das Tagesmax nicht. Jede
    Rechnung gegen eine Wetterquelle weist deshalb Gewinner aus, die real
    verloren haben. Das ist genau der Fehler, der die erste Fassung dieser
    Auswertung wertlos gemacht hat.

    Rueckgabe: True = Bucket getroffen (Lay verliert), False = verfehlt
    (Lay gewinnt), None = noch nicht abgerechnet.
    """
    if mid in cache:
        return cache[mid]
    erg = None
    for versuch, pause in ((1, 3), (2, 10), (3, 0)):
        try:
            r = requests.get(f"{PM_API}/v1/markets/{mid}", timeout=25)
            if r.status_code == 429:
                time.sleep(pause)
                continue
            r.raise_for_status()
            m = r.json()
            if (m.get("status") or "").lower() == "closed":
                erg = (m.get("result") or "").lower() == "yes"
            break
        except Exception:
            if pause:
                time.sleep(pause)
    cache[mid] = erg
    return erg


def metar_max(icao, tag, tz_name):
    """Tages-Max in Grad C aus METAR (IEM), lokaler Kalendertag.

    Nur fuer die Zieltage, die in der DB noch kein settle_k haben. Holt einen
    Tag Puffer auf beiden Seiten und schneidet auf den lokalen Tag zurueck —
    dieselbe Zeitzonen-Vorsicht wie im Ladder-Logger (valid_time_gmt).
    """
    from zoneinfo import ZoneInfo
    d = datetime.strptime(tag, "%Y-%m-%d").date()
    p = {"station": icao, "data": "tmpc", "tz": "UTC", "format": "onlycomma",
         "missing": "empty", "trace": "empty", "latlon": "no", "report_type": "3",
         "year1": d.year, "month1": d.month, "day1": d.day,
         "year2": (d + timedelta(days=2)).year, "month2": (d + timedelta(days=2)).month,
         "day2": (d + timedelta(days=2)).day}
    # IEM drosselt hart (429). Ohne Backoff fehlen genau die Stadt-Tage, die
    # der Nachzug holen soll — beim ersten Lauf waren das 13 von 29.
    r = None
    for versuch, pause in ((1, 5), (2, 15), (3, 40), (4, 0)):
        r = requests.get(IEM, params=p, timeout=60)
        if r.status_code != 429:
            break
        if pause:
            time.sleep(pause)
    r.raise_for_status()
    tz = ZoneInfo(tz_name)
    best = None
    for line in r.text.splitlines()[1:]:
        teile = line.split(",")
        if len(teile) < 3 or not teile[2].strip():
            continue
        try:
            ts = datetime.strptime(teile[1].strip(), "%Y-%m-%d %H:%M").replace(
                tzinfo=timezone.utc)
            wert = float(teile[2])
        except ValueError:
            continue
        if ts.astimezone(tz).date() != d:
            continue
        if best is None or wert > best:
            best = wert
    return best


def settle_bucket_lokal(wert, city):
    """half_up, ausser BUCKET_FLOOR-Staedte (Hong Kong)."""
    from weather_stations import BUCKET_FLOOR
    import math
    if city in BUCKET_FLOOR:
        return math.floor(wert)
    return math.floor(wert + 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/tmp/ab/autobuy_log.csv")
    ap.add_argument("--kein-nachzug", action="store_true",
                    help="offene Zieltage NICHT per METAR nachziehen")
    args = ap.parse_args()

    print(f"Fenster: Zieltage {FENSTER_VON} bis {FENSTER_BIS}\n")
    rows = lade_log(args.log)
    tage = kandidaten(rows)

    conn = pymssql.connect(**DB_CONFIG)
    mids = markt_ids(conn)
    print(f"  {len(mids)} Markt-IDs aus der DB.\n")
    cache = {}

    # --- Auswertung ---
    ergebnis = {"V1": [], "V2": []}
    fehlend = []
    print(f"{'Zieltag':11s} {'V1 (haette)':38s}  {'V2 (ist)':38s}")
    print("-" * 92)
    for tag in sorted(tage):
        cands = tage[tag]
        v1 = waehle_v1(cands)
        v2 = [c for c in cands if c["gekauft_v2"]]
        zeile = {}
        for name, picks in (("V1", v1), ("V2", v2)):
            teil, txt = [], []
            for c in picks:
                mid = mids.get((tag, c["city"], c["k"]))
                verloren = markt_ergebnis(mid, cache) if mid else None
                if verloren is None and tag == "2026-08-06":
                    verloren = ERGEBNIS_0608.get((c["city"], c["k"]))
                if verloren is None:
                    fehlend.append((tag, c["city"], c["k"], name))
                    continue
                p = pnl_lay(c["no"], verloren)
                teil.append((c, p, verloren))
                ergebnis[name].append((tag, c["city"], c["k"], c["no"], verloren, p))
                txt.append(f"{c['city'][:8]}@{c['no']:.2f}{'X' if verloren else '+'}")
            s = sum(p for _, p, _ in teil)
            zeile[name] = f"{len(teil)}L {s:+6.2f}$ " + " ".join(txt)[:24]
        print(f"{tag:11s} {zeile['V1']:38s}  {zeile['V2']:38s}")

    def bilanz(titel, filt):
        print(f"\n{titel}")
        werte = {}
        for name in ("V1", "V2"):
            e = [x for x in ergebnis[name] if filt(x[0])]
            n = len(e)
            if not n:
                print(f"  {name}: keine Lays")
                continue
            einsatz = n * USD
            gewinn = sum(x[5] for x in e)
            verlierer = sum(1 for x in e if x[4])
            werte[name] = (n, einsatz, gewinn)
            print(f"  {name}: {n:2d} Lays, Einsatz {einsatz:6.2f} $, "
                  f"PnL {gewinn:+6.2f} $ = {100*gewinn/einsatz:+6.2f} % ROI, "
                  f"Treffer {n-verlierer}/{n} = {100*(n-verlierer)/n:5.1f} %, "
                  f"NO im Mittel {sum(x[3] for x in e)/n:.3f}")
        if len(werte) == 2:
            (na, ea, ga), (nb, eb, gb) = werte["V1"], werte["V2"]
            print(f"  -> V2 - V1: {100*(gb/eb - ga/ea):+.2f} pp ROI, "
                  f"{gb - ga:+.2f} $ absolut (bei {nb} gegen {na} Lays)")

    # Alle Zieltage einschliesslich des letzten. Entscheidung des Betreibers
    # (06.08.): an dem Tag dreht sich kein Wert mehr, also als gesettelt
    # einrechnen statt getrennt ausweisen.
    bilanz(f"BILANZ — Zieltage {FENSTER_VON} bis {FENSTER_BIS}:", lambda t: True)

    # Die eigentliche Frage des Betreibers (06.08.): "wie saehe der Chart aus,
    # wenn V1 weitergelaufen waere?" — also die KUMULATIVE Kurve, nicht die
    # Bilanz. Startpunkt 0 am Vortag des Fensters; beide Kurven beschreiben nur
    # den Autobuy-Anteil, nicht das Gesamtkonto.
    print("\nKUMULATIVE KURVE ab Zieltag 28.07. (nur Autobuy-Lays, Start = 0):")
    print(f"  {'Zieltag':11s} {'V1 Tag':>8s} {'V1 kum':>9s}   {'V2 Tag':>8s} {'V2 kum':>9s}   Delta")
    ka = kb = 0.0
    for tag in sorted(tage):
        ta = sum(x[5] for x in ergebnis["V1"] if x[0] == tag)
        tb = sum(x[5] for x in ergebnis["V2"] if x[0] == tag)
        ka += ta
        kb += tb
        print(f"  {tag:11s} {ta:+8.2f} {ka:+9.2f}   {tb:+8.2f} {kb:+9.2f}   {kb-ka:+7.2f}")

    a, b = ergebnis["V1"], ergebnis["V2"]
    # Ueberschneidung: wie oft haetten beide dieselbe Wette gemacht?
    sa = {(x[0], x[1], x[2]) for x in a}
    sb = {(x[0], x[1], x[2]) for x in b}
    print(f"\nUeberschneidung: {len(sa & sb)} identische Lays "
          f"(V1 exklusiv {len(sa - sb)}, V2 exklusiv {len(sb - sa)})")

    # Woran haengt der Unterschied? Preisband der jeweiligen Auswahl.
    print("\nPreisverteilung der Lays (Anteil je NO-Band):")
    for name, e in (("V1", a), ("V2", b)):
        if not e:
            continue
        baender = [("<0,70", lambda p: p < 0.70), ("0,70-0,90", lambda p: 0.70 <= p < 0.90),
                   ("0,90-0,95", lambda p: 0.90 <= p < 0.95), (">=0,95", lambda p: p >= 0.95)]
        teile = []
        for lab, f in baender:
            sub = [x for x in e if f(x[3])]
            if sub:
                roi = 100 * sum(x[5] for x in sub) / (len(sub) * USD)
                teile.append(f"{lab}: {len(sub):2d}L {roi:+6.1f}%")
        print(f"  {name}  " + "  |  ".join(teile))

    if fehlend:
        print(f"\nOHNE MARKT-ERGEBNIS und daher NICHT gerechnet: {len(fehlend)}")
        for t, c, k, w in fehlend:
            print(f"  {t} {c} {k} ({w})")

    # SELBSTKONTROLLE: die V2-Zeile MUSS zur echten Abrechnung passen. Ohne
    # diese Probe faellt ein Settlement-Fehler wie der Shenzhen-Fall nicht auf —
    # die erste Fassung dieser Auswertung wies V2 mit -1,96 $ aus, real sind es
    # -15,22 $ (weather_konto_seit_v2.py, Jupiter-History).
    print("\nSELBSTKONTROLLE gegen die echte Jupiter-Abrechnung:")
    print(f"  V2 hier gerechnet (5 $/Lay pauschal) : {sum(x[5] for x in b):+7.2f} $ "
          f"bei {len(b)} Lays")
    print(f"  V2 real laut History (Autobuy-Anteil): {-15.22:+7.2f} $ bei 29 Positionen")
    print("  Rest-Abweichung: pauschale 5 $ gegen reale Fills (~4,80 $), "
          "Teilverkaeufe, und der Zieltag-29.07.-Kauf lief am 28.07.")

    # Traegt der Unterschied? Zweistichproben-t auf die Rendite je Lay. Die
    # Auswahlmengen ueberschneiden sich nur teilweise, ein gepaarter Test
    # scheidet damit aus.
    import statistics as stat
    ra = [x[5] / USD for x in a]
    rb = [x[5] / USD for x in b]
    if len(ra) > 1 and len(rb) > 1:
        va, vb = stat.variance(ra), stat.variance(rb)
        se = (va / len(ra) + vb / len(rb)) ** 0.5
        t = (stat.mean(rb) - stat.mean(ra)) / se if se else float("nan")
        print(f"\nTraegt der Unterschied? t = {t:+.2f} auf die Rendite je Lay "
              f"(Welch, {len(rb)} gegen {len(ra)}). "
              f"{'NICHT signifikant' if abs(t) < 1.5 else 'grenzwertig' if abs(t) < 2 else 'signifikant'} "
              "— zehn Zieltage sind zu wenig fuer eine Entscheidung.")

    print("\nSettlement-Quelle: Jupiters Markt-Ergebnis (result), nicht Wetter. "
          "Zieltag 06.08. voll eingerechnet (Anweisung Betreiber).")


if __name__ == "__main__":
    sys.exit(main())
