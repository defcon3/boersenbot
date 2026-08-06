#!/usr/bin/env python3
"""
weather_minus1_autobuy.py — Autonomer Live-Test der −1-Lay-Klasse.

Pre-Reg: preregs/weather_minus1_live_2026_07_20.md (Klasse-B-Basis: −1-Klasse
+3,35 % netto, t 7,9 — preregs/weather_classb_lay_2026_07_18.md).

=== VERSION 2 (27.07.2026) — PREISBAND STATT PREIS-RANKING ===

V1 (Tag `autobuy-v1`) nahm "die konservativsten zuerst", also die höchsten
NO-Preise. Die Messung über 116 Kandidaten / 7 Zieltage
(`weather_minus1_ppess_filter.py`) zeigt, dass genau das das ertragsschwächste
Ende ist:

    NO 0,95-1,00   +2,05 %   <- V1 kaufte hier
    NO 0,85-0,90  +14,01 %
    NO 0,75-0,85   +9,85 %
    NO 0,70-0,75  +16,83 %
    NO unter 0,70 -17,64 %   <- Klippe, dort ist der Markt fair

Der Edge der −1-Klasse ist NICHT "sichere Wetten sammeln", sondern die
systematische Fehlbepreisung des Marktes im Bereich 10–30 % eingepreiste
Bucket-Chance (er sagt 25,7 %, eingetreten sind 9,1 %). Über NO 0,90 zahlt er
zu wenig dafür, unter 0,70 liegt er richtig.

WICHTIG zur Doktrin: Das Band ist KEIN "dem Markt folgen". Der Bucket kommt
weiter allein aus unserem mu (k = Modell-Favorit − 1); der Preis sagt nur, wo
der Markt am weitesten danebenliegt. Die Rangfolge INNERHALB des Bandes läuft
deshalb über den Temperaturabstand — die eigene Prognose entscheidet, wer
zuerst drankommt, der Preis nur, wer überhaupt in Frage kommt.

Der Betreiber hat den vorgeschlagenen Forward-Test bewusst übersprungen; V2
geht ohne Pre-Reg live. Deklarierte Abweichung von der Projektmethodik.
Vorbehalt, der damit offen bleibt: 7 Tage, in-sample, ~10 Varianten probiert.

=== NACHTRAG 06.08.2026 — DER BEFUND OBEN IST OOS GEFALLEN ===

Die Staffel darüber war in-sample. Auf den neun Zieltagen 29.07.–06.08., die
es am 27.07. noch nicht gab, sieht sie so aus (weather_preisband_oos.py, alle
110 Kandidaten, Settlement aus Jupiters Markt-`result`):

    NO 0,70-0,75  -54,61 %   <- in-sample war +16,83 %, Vorzeichen gedreht
    NO 0,85-0,90   -0,79 %   <- in-sample war +14,01 %
    NO 0,95-1,00   -1,12 %
    Band 0,70-0,90 gesamt: -7,57 % gegen +13,71 % in-sample, t -0,66 (war +4,75)

Entscheidend ist nicht der ROI, sondern der Abstand zwischen Trefferquote und
Break-even: er liegt in fünf von sieben Bändern innerhalb ±1,6 pp. **Der Markt
ist über die ganze Preisskala fair bepreist** — es gibt keinen Preisband-Edge,
weder oben noch unten. Das ist dieselbe Aussage wie am 02.08. für die
−1-Klasse; der Ertrag kann nur aus der Selektion kommen, nicht aus dem Preis.

Was daraus folgt und was NICHT: Die Begründung für V2 trägt nicht mehr. Daraus
folgt aber kein Beleg für V1 — die reale V2-Ära (−15,22 $ über 29 Positionen
gegen +14,91 $ in der V1-Woche) hängt an wenigen Tagen, t −0,65. Die Anhebung
von BAND_LO auf 0,75 ist eine Entscheidung unter Unsicherheit, kein Ergebnis.
Belege: weather_preisband_oos.py, weather_autobuy_v1_gegenrechnung.py,
weather_konto_seit_v2.py.

Regel (täglich, VPS-Timer 12:45 UTC, direkt nach dem 12:30-Ladder-Snapshot):
  1. Kandidaten = heutiger bb_WeatherLadders-Snapshot mit var='max', kind='eq',
     offset_fav=-1, status='open', target_date=morgen (Lead 1, identische
     µ-Definition wie die Klasse-B-Messung).
  2. Live-Preis-Recheck je Markt; handelbar nur im BAND
     BAND_LO (0,75 seit 06.08., davor 0,70) <= buyNo < BAND_HI (0,90). Märkte
     mit bestehender Position werden übersprungen (Idempotenz + keine Kollision
     mit manuellen Wetten). Zur Anhebung siehe den Kommentar an BAND_LO —
     sie ist eine Entscheidung des Betreibers auf unbelegter Datenlage.
  2b. Spannen-Veto (25.07.): rohe Modellspanne der 5 Modelle > MAX_SPREAD
     (3 °C) → kein Kauf. Schwelle und Abfrage kommen per Import aus
     weather_outlier_screen, damit es nur einen Codepfad gibt. Ist die
     Prognose gar nicht abrufbar, wird ebenfalls nicht gekauft.
  3. TEMPERATURABSTAND: mu minus Oberkante des gelayten Buckets, also wie weit
     die eigene Prognose über der Bucket-Grenze sitzt (konstruktionsbedingt in
     [0,1) K). Dient doppelt:
     (a) als VETO — seit 06.08.2026 ist ABSTAND_MIN = 0,50 K scharf, s. dort.
         Vorher aus; die V2-Ära war damit der OOS-Test der Schwelle.
     (b) als RANGFOLGE innerhalb der verbleibenden Kandidaten, absteigend.
     Gemessen trennt er die Ausgänge (Gewinner +0,48 K, Verlierer +0,38 K).
     Er hebt den ROI nicht — er begrenzt den Verlustschwanz, und darauf kommt
     es an, weil unter der Woche niemand eingreift. Warschau 21 (28.07.) hatte
     0,37 K, Wellington 12 (07.08.) hatte 0,33 K.
  4. Cap 8 (V1: 6). Das Band liefert im Mittel 7,4 Kandidaten/Tag (4..10) —
     der Cap greift also selten. Er ist eine OBERGRENZE, kein Ziel: gibt das
     Band nur vier her, werden vier gesetzt und NICHT mit schwächeren
     aufgefüllt. Kein Güte-Gate mehr (V1: erste 3 frei, dann NO >= 0,85) —
     seine Schwelle drängte in genau die Ecke, die kaum etwas trägt.
  5. Guthaben-Check VOR dem Kaufloop: reicht das freie JupUSD nicht für alle
     Picks, wird die Liste gekürzt statt in fail_send zu laufen (am 27.07.
     rutschte Mexico City still durch, weil der Bot blind sendete).
  6. Kauf 5 $ NO je Markt (Jupiter-Minimum), Limit = Live-Ask + 0,005.
     Max 2 Sendeversuche, KEIN Nachrücker bei Fehlschlag.
  7. Halten bis Settlement — kein TP (TP-Lehre 14.07.), Claims macht der
     bestehende VPS-Autopilot.
  8. Log: preregs/weather_minus1_live_log.csv (führende Datei auf dem VPS).

Guards: Zeitfenster 12:30–14:30 UTC (kein Nachhol-Lauf zu anderem Lead),
Tages-Idempotenz übers Log, harter Fehler wenn heute kein Ladder-Snapshot da.
"""

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pymssql
import requests

from jupiter_buy import place
from jupiter_sell import API as JUP_API, load_keypair
from weather_ladder_logger import DB_CONFIG
# Bucket-Grenzen NICHT nachbauen: half_up fuer die meisten Staedte, [k, k+1) fuer
# die BUCKET_FLOOR-Staedte (Hong Kong). Wer hier danebengreift, verschiebt den
# Temperaturabstand um ein halbes Grad.
from weather_stations import bucket_grenzen
# Spannen-Veto aus dem Screen IMPORTIERT, nicht kopiert (Kopier-Lehre aus dem
# Beijing-33-Verlust 14.07.: ein Fix, der nur in einer Kopie landet, ist keiner).
from weather_outlier_screen import MAX_SPREAD, fetch_raw_models, model_spread

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

PM_API = "https://prediction-market-api.jup.ag/api"
LOG_CSV = Path("preregs/weather_minus1_live_log.csv")
# "abstand" haengt bewusst am ENDE: die bestehenden Zeilen auf dem VPS haben 12
# Felder. Eine Spalte in der Mitte wuerde beim Lesen alles dahinter verschieben —
# angehaengt liefert DictReader fuer Altzeilen schlicht None.
LOG_FIELDS = ["run_utc", "target_date", "city", "k", "mu_ens", "buy_no_snap",
              "buy_no_live", "decision", "usd", "contracts", "avg_price",
              "signature", "abstand"]
# Preisband statt Preis-Ranking (V2, 27.07.). Halboffen [LO, HI): oberhalb zahlt
# der Markt zu wenig fuer das Risiko (+2 % im Band 0,95-1,00), unterhalb ist er
# fair und es kippt (-17,6 %). Belegt in weather_minus1_ppess_filter.py.
# 06.08.2026: BAND_LO von 0,70 auf 0,75 angehoben. Anweisung des Betreibers,
# nach der ersten OOS-Messung des Bandbefunds (weather_preisband_oos.py, 103
# Kandidaten, Zieltage 29.07.-06.08. — Daten, die es am 27.07. noch nicht gab).
#
#   Schwelle    n   Treffer   Break-even   Delta      ROI       t
#     0,70     71    87,3 %     89,2 %    -1,8 pp   -3,47 %   -0,77
#     0,75     65    92,3 %     90,8 %    +1,5 pp   +1,25 %   +0,33
#     0,80     57    94,7 %     92,8 %    +2,0 pp   +1,98 %   +0,60
#
# ⚠️ NICHT BELEGT. t = 0,33 an der gewaehlten Schwelle, und die Tabelle stammt
# aus denselben Daten, die die Frage ausgeloest haben — also in-sample. Sechs
# Schwellen geprueft, Bonferroni verlangte t >= 2,64. Fuer t = 2,0 braeuchte es
# rund 633 Lays (~5 Monate). Die Schwelle 0,80 saehe besser aus, haengt aber
# komplett an EINEM Band (0,80-0,85, 7 von 7 Treffern; bei fairer Bepreisung
# passiert das in 24,3 % der Faelle rein zufaellig) — ohne dieses Band faellt
# sie von +1,98 % auf -0,66 %. Deshalb 0,75 und nicht 0,80.
#
# Zwei unabhaengige Gruende sprechen dafuer, dass die RICHTUNG stimmt: das
# Delta dreht bei 0,75 ins Positive, und die Gebuehr faellt monoton mit dem
# Preis (erwartete Fee je Lay: 1,96 % bei NO 0,72 gegen 0,52 % bei 0,93) — bei
# einem fair bepreisten Markt ist sie der einzige Term, der nicht ausmittelt.
# Beleg ist das nicht. Vorherige Fassung: git show <dieser Commit>^.
BAND_LO = 0.75
BAND_HI = 0.90
CAP_DEFAULT = 8         # OBERGRENZE, kein Ziel — das Band liefert im Mittel 7,4
                        #   Kandidaten/Tag; nie mit schwaecheren auffuellen.
# 06.08.2026 SCHARF GESCHALTET (vorher 0.0, also aus). Anweisung des Betreibers.
#
# Die Schwelle 0,50 K stammt vom 27.07.2026 (weather_minus1_ppess_filter.py,
# Schattenbuch: Verliererquote 20,7 % -> 12,2 %) und wurde damals bewusst NICHT
# aktiviert. Damit ist die gesamte V2-Aera ab Zieltag 29.07. ein echter
# Out-of-Sample-Test dieser Schwelle — und sie haelt:
#
#   Abstand < 0,50 K : 10 Positionen, -17,34 $   (5 Totalverluste)
#   Abstand >= 0,50 K: 19 Positionen,  +2,12 $   (4 Verlierer von 19)
#   Verliererquote 50 % unterhalb gegen 21 % oberhalb der Schwelle.
#
# Der GESAMTE V2-Verlust steckt in den zehn Positionen unterhalb. Mit Filter
# stuende das Konto bei -16,02 $ statt bei -33,36 $.
#
# ⚠️ WIE ER WIRKT, ehrlich: Er trennt Gewinner und Verlierer NICHT sauber — die
# Abstaende ueberlappen (Verlierer 0,01..0,47 K, Gewinner 0,11..0,42 K). Er
# wirkt ueber die ASYMMETRIE: ein Verlust kostet die vollen ~4,83 $, ein Gewinn
# bringt nur 1,00..1,85 $. Wer bei halb-halber Trennung die teure Seite
# mitnimmt, gewinnt trotzdem.
# ⚠️ Und er macht den Bot NICHT profitabel: die verbleibenden 19 Positionen
# liegen bei +2,12 $ mit t = +0,25. Das ist Schadensbegrenzung, kein Edge.
# Preis: rund ein Drittel weniger Lays (19 statt 29 in der V2-Aera).
ABSTAND_MIN = 0.50      # Veto auf den Temperaturabstand (--abstand-min 0 schaltet aus)
USD_DEFAULT = 5.0       # Jupiter-Minimum
CASH_PUFFER = 1.02      # Fee kommt zum Einsatz dazu (0,07*n*min(p,1-p))
LIMIT_CAP = 0.97        # harter Deckel fuer den Limitpreis (Tick = ganze Cents).
                        #   Bindend ist BAND_HI; das hier faengt nur den Fall ab,
                        #   dass Ask+1ct ueber eine sinnvolle Rendite hinausliefe.


def get_json(url, params=None, tries=4):
    r = None
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=20)
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"GET {url} scheiterte ({r.status_code if r is not None else 'conn'})")


def load_candidates(target_day):
    """Heutiger 12:30-Snapshot: alle offenen −1-Fenster (max/eq) des Zieltags."""
    today_1200 = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0,
                                                    microsecond=0, tzinfo=None)
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor(as_dict=True)
    cur.execute("SELECT MAX(snapshot_utc) AS snap FROM bb_WeatherLadders WHERE snapshot_utc >= %s",
                (today_1200,))
    snap = cur.fetchone()["snap"]
    if snap is None:
        conn.close()
        raise RuntimeError("Kein Ladder-Snapshot von heute >=12:00 UTC — Logger-Lauf pruefen!")
    cur.execute(
        "SELECT city, k, market_id, buy_yes, buy_no, mu_ens FROM bb_WeatherLadders "
        "WHERE snapshot_utc = %s AND target_date = %s AND var = 'max' AND kind = 'eq' "
        "AND offset_fav = -1 AND status = 'open' ORDER BY city",
        (snap, target_day))
    rows = cur.fetchall()
    conn.close()
    print(f"Snapshot {snap} UTC: {len(rows)} −1-Kandidaten fuer {target_day}.")
    return rows


def owned_market_ids(owner):
    j = get_json(f"{JUP_API}/positions", {"ownerPubkey": owner})
    return {p.get("marketId") for p in j.get("data", [])
            if str(p.get("contracts", "0")) not in ("0", "")}


def live_market(mid):
    """(status, buyNo, eventId). Die eventId ist die Bretts-ID fuer den UI-Link —
    jup.ag/prediction/<eventId> zeigt die ganze Leiter der Stadt, die
    marketId-Variante loest dort nicht sauber auf."""
    m = get_json(f"{PM_API}/v1/markets/{mid}")
    pr = m.get("pricing") or {}
    return m.get("status"), (pr.get("buyNoPriceUsd") or 0) / 1e6, m.get("eventId")


def verify_fill(owner, mid, tries=3):
    """Fill-Zwischenstand im Kaufloop — rein kosmetisch, darf NIE werfen.

    Am 25.07. lief /positions nach dem vierten Kauf in ein 429; get_json() gab
    danach auf und die Exception riss den ganzen Lauf mit, NACHDEM alle vier
    Kaeufe schon abgesendet waren — also ohne CSV-Zeile und ohne Mail. Die
    echten Zahlen holt ohnehin final_fills() am Ende des Laufs nach, ein
    Fehlschlag hier kostet nur die Zwischenausgabe."""
    for _ in range(tries):
        time.sleep(5)
        try:
            j = get_json(f"{JUP_API}/positions", {"ownerPubkey": owner})
        except Exception as e:
            print(f"  (Fill-Zwischencheck fehlgeschlagen, egal: {e})")
            return None, None
        for p in j.get("data", []):
            if p.get("marketId") == mid and str(p.get("contracts", "0")) not in ("0", ""):
                return p.get("contractsDecimal"), int(p.get("avgPriceUsd", 0)) / 1e6
    return None, None


def final_fills(owner, mids):
    """Fills der gekauften Maerkte final abfragen (marketId -> (contracts, avg, kosten)).

    verify_fill() im Kaufloop wartet nur ~15 s und lief deshalb an drei Tagen in
    Folge in 'sent_unverified', obwohl alle Kaeufe echt gefuellt waren — Jupiter
    materialisiert die Position traege. Hier, am Ende des Laufs, sind laengst
    Minuten vergangen, die Zahlen stimmen also fuer die Mail."""
    out = {}
    try:
        # Grosszuegig retrien: am 27.07. lief genau dieser Aufruf in ein 429 und
        # die Mail zeigte daraufhin "0.00 @ 0.000" statt der Quoten. Kurz warten,
        # dann bis zu 6 Versuche — der Lauf hat es nicht eilig.
        time.sleep(5)
        j = get_json(f"{JUP_API}/positions", {"ownerPubkey": owner}, tries=6)
        for p in j.get("data", []):
            if p.get("marketId") in mids:
                out[p["marketId"]] = (float(p.get("contractsDecimal") or 0),
                                      int(p.get("avgPriceUsd", 0)) / 1e6,
                                      int(p.get("totalCostUsd", 0)) / 1e6)
    except Exception as e:
        print(f"  (Fill-Nachfrage fuer die Mail fehlgeschlagen: {e})")
    return out


def send_summary_mail(run_utc, target, cap, n_cands, bought, failed):
    """Mail nach dem Lauf: was wurde gesetzt. Fehler brechen den Lauf NICHT ab."""
    try:
        from autopilot import notify
    except Exception as e:
        print(f"  (Mail-Import fehlgeschlagen: {e})")
        return
    if bought:
        zeilen = []
        for b in bought:
            link = (f"https://jup.ag/prediction/{b['event_id']}"
                    if b.get("event_id") else None)
            # Quote = Preis je Kontrakt. Der Lay gewinnt (1 − Quote) je Kontrakt,
            # deshalb steht die Gegenwahrscheinlichkeit gleich daneben: sie ist
            # das, wogegen die eigene Prognose zu pruefen ist.
            quote = f"{b['av']:.3f}{'≈' if b['geschaetzt'] else ''}"
            impl = f"{(1 - b['av']) * 100:.0f} %" if b["av"] else "—"
            stadt = (f'<a href="{link}" style="color:#1565c0;text-decoration:none;">'
                     f'<b>{b["city"]}</b> {b["k"]}°C NO ↗</a>' if link
                     else f'<b>{b["city"]}</b> {b["k"]}°C NO')
            zeilen.append(
                f'<tr><td style="padding:7px 10px;border-bottom:1px solid #eee;">{stadt}'
                f'<br><span style="color:#888;font-size:12px;">Bucket-Chance laut Markt '
                f'{impl}</span></td>'
                f'<td style="padding:7px 10px;border-bottom:1px solid #eee;text-align:right;'
                f'white-space:nowrap;">{b["ct"]:.2f} @ <b>{quote}</b></td>'
                f'<td style="padding:7px 10px;border-bottom:1px solid #eee;text-align:right;">'
                f'{b["ko"]:.2f} $</td></tr>')
        rows = "".join(zeilen)
        einsatz = sum(b["ko"] for b in bought)
        gewinn = sum(b["ct"] - b["ko"] for b in bought)
        rendite = f" (+{gewinn / einsatz * 100:.1f} %)" if einsatz else ""
        unsicher = any(b["geschaetzt"] for b in bought)
        hinweis = ('<br><span style="color:#888;font-size:12px;">≈ = Fill war noch '
                   'nicht abrufbar, Quote ist der Ask zum Kaufzeitpunkt.</span>'
                   if unsicher else "")
        kopf = f"🤖 −1-Autobuy: {len(bought)} Lay{'s' if len(bought) != 1 else ''} gesetzt"
        body = (f'<table style="width:100%;border-collapse:collapse;font-size:14px;">{rows}</table>'
                f'<br>Einsatz <b>{einsatz:.2f} $</b> · bei vollem Durchlauf '
                f'<b>+{gewinn:.2f} $</b>{rendite} · Settlement am {target}.{hinweis}')
        text = ("".join(
            f"  {b['city']} {b['k']}°C NO — {b['ct']:.2f} Kontr. @ "
            f"{b['av']:.3f}{'~' if b['geschaetzt'] else ''} = {b['ko']:.2f} $"
            f"  (Markt: {(1 - b['av']) * 100:.0f} % Bucket-Chance)\n"
            + (f"      https://jup.ag/prediction/{b['event_id']}\n"
               if b.get("event_id") else "")
            for b in bought)
            + f"\nEinsatz {einsatz:.2f} $ | bei vollem Durchlauf +{gewinn:.2f} $"
              f"{rendite} | Settlement am {target}."
            + ("\n(~ = Fill noch nicht abrufbar, Quote ist der Ask zum Kaufzeitpunkt.)"
               if unsicher else ""))
    else:
        kopf = "🤖 −1-Autobuy: nichts gesetzt"
        body = ("Es wurde <b>kein</b> Markt gekauft — kein Kandidat hat die Kriterien "
                "erfuellt oder alle waren schon im Bestand.")
        text = "Es wurde KEIN Markt gekauft (kein Kandidat qualifiziert / schon im Bestand)."
    fehl = (f'<br><br><span style="color:#b71c1c;">Fehlgeschlagen: {", ".join(failed)}</span>'
            if failed else "")
    html = (f'<div style="font-family:Segoe UI,Arial;max-width:520px;margin:auto;">'
            f'<div style="background:#1565c0;color:#fff;padding:18px;text-align:center;'
            f'border-radius:10px 10px 0 0;font-size:18px;font-weight:800;">{kopf}</div>'
            f'<div style="padding:18px;background:#fff;color:#333;font-size:15px;line-height:1.6;">'
            f'Lauf <b>{run_utc} UTC</b> · Zieltag <b>{target}</b> · '
            f'{n_cands} Kandidaten im Preisband, Cap {cap}.<br><br>{body}{fehl}</div></div>')
    notify(f"{kopf} — Zieltag {target}", html,
           f"−1-AUTOBUY {run_utc} UTC | Zieltag {target}\n"
           f"{n_cands} Kandidaten im Preisband, Cap {cap}.\n\n{text}"
           + (f"\nFehlgeschlagen: {', '.join(failed)}" if failed else ""))


def append_log(rows):
    new = not LOG_CSV.exists()
    LOG_CSV.parent.mkdir(exist_ok=True)
    with LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def bought_today():
    if not LOG_CSV.exists():
        return 0
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with LOG_CSV.open(encoding="utf-8") as f:
        return sum(1 for r in csv.DictReader(f)
                   if r["run_utc"].startswith(today) and r["decision"] == "bought")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="alles ausser /execute")
    ap.add_argument("--target", help="Zieltag YYYY-MM-DD (Default: morgen UTC)")
    ap.add_argument("--cap", type=int, default=CAP_DEFAULT,
                    help="Obergrenze der Lays pro Lauf (wird NIE aufgefuellt)")
    ap.add_argument("--band-lo", type=float, default=BAND_LO,
                    help="untere Bandgrenze des Live-NO (darunter ist der Markt fair)")
    ap.add_argument("--band-hi", type=float, default=BAND_HI,
                    help="obere Bandgrenze des Live-NO (darueber zahlt er zu wenig)")
    ap.add_argument("--abstand-min", type=float, default=ABSTAND_MIN,
                    help="Mindest-Temperaturabstand mu ueber Bucket-Oberkante in K "
                         "(0.50 senkt die Verliererquote auf ~12 %%)")
    ap.add_argument("--usd", type=float, default=USD_DEFAULT)
    ap.add_argument("--force-window", action="store_true",
                    help="Zeitfenster-Guard uebergehen (nur fuer Tests)")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    target = args.target or (now + timedelta(days=1)).strftime("%Y-%m-%d")
    run_utc = now.strftime("%Y-%m-%d %H:%M")
    print(f"=== −1-Autobuy V2 {run_utc} UTC | Zieltag {target} | "
          f"Band [{args.band_lo:.2f}, {args.band_hi:.2f}) | cap {args.cap} | "
          f"abstand >= {args.abstand_min:.2f} K | {args.usd:.0f} $/Lay | "
          f"{'DRY-RUN' if args.dry_run else 'ECHT'} ===")

    if not args.force_window and not (12 * 60 + 30 <= now.hour * 60 + now.minute <= 14 * 60 + 30):
        print("Ausserhalb des 12:30–14:30-UTC-Fensters — Skip (Guard gegen Nachhol-Laeufe).")
        return 0

    n_prev = bought_today()
    if n_prev:
        print(f"Heute bereits {n_prev} Kaeufe im Log — Skip (Doppellauf-Schutz).")
        return 0

    kp = load_keypair()
    owner = str(kp.pubkey())
    cands = load_candidates(target)
    owned = owned_market_ids(owner)

    log_rows, tradeable = [], []
    for c in cands:
        # Temperaturabstand: wie weit sitzt die eigene Prognose ueber der
        # Oberkante des gelayten Buckets? Bei offset_fav=-1 liegt mu immer im
        # Bucket darueber, der Wert also in [0,1) K. Klein = mu klebt an der
        # Kante = gefaehrlich.
        abstand = None
        if c["mu_ens"] is not None:
            abstand = c["mu_ens"] - bucket_grenzen(c["k"], c["city"])[1]
        base = {"run_utc": run_utc, "target_date": target, "city": c["city"], "k": c["k"],
                "mu_ens": round(c["mu_ens"], 2) if c["mu_ens"] is not None else "",
                "buy_no_snap": c["buy_no"],
                "abstand": round(abstand, 2) if abstand is not None else "",
                "usd": "", "contracts": "", "avg_price": "", "signature": ""}
        if c["market_id"] in owned:
            log_rows.append({**base, "buy_no_live": "", "decision": "skip_position"})
            continue
        if abstand is None:
            # Ohne mu kein Abstand — und ohne Abstand keine Rangfolge. Kann nur
            # auftreten, wenn der Logger die Stadt ohne mu geschrieben hat.
            log_rows.append({**base, "buy_no_live": "", "decision": "skip_no_mu"})
            continue
        try:
            status, no_live, event_id = live_market(c["market_id"])
        except RuntimeError as e:
            log_rows.append({**base, "buy_no_live": "", "decision": f"skip_api_{e}"})
            continue
        base["buy_no_live"] = round(no_live, 3)
        if status != "open" or no_live <= 0:
            log_rows.append({**base, "decision": "skip_closed"})
            time.sleep(0.4)
            continue

        # REIHENFOLGE: Spannen-Veto VOR dem Preisband, obwohl es teurer ist (ein
        # Modellabruf je Stadt). Grund ist das V1-Schattenbuch: nur wenn fuer
        # JEDEN Kandidaten feststeht, ob die Spanne haelt, laesst sich aus dem
        # Log rekonstruieren, was V1 gekauft haette — V1 kannte das Preisband
        # nicht und haette auch ueber 0,90 zugegriffen. Stuende das Band vorher,
        # blieben genau jene Kandidaten ungeprueft und der Vergleich haette
        # Loecher. (Spannen-Veto 25.07.: >3 K Streuung im Ensemble traegt kein mu.)
        raw, grund = fetch_raw_models(c["city"], target, "max")
        if raw is None:
            # Ohne Prognose keine Qualitaetspruefung -> NICHT kaufen. Faellt
            # Open-Meteo ganz aus, setzt der Bot an dem Tag nichts; das ist
            # der harmlosere Ausfall gegenueber ungeprueften Lays.
            log_rows.append({**base, "decision": f"skip_noforecast_{grund}"})
        elif model_spread(raw) > MAX_SPREAD:
            log_rows.append({**base, "decision": f"skip_spread_{model_spread(raw):.1f}"})
        elif no_live >= args.band_hi:
            # Zu teuer eingekauft: der Markt haelt den Bucket fuer so
            # unwahrscheinlich, dass kaum Rendite bleibt (+2,05 % gemessen).
            log_rows.append({**base, "decision": f"skip_band_teuer_{no_live:.2f}"})
        elif no_live < args.band_lo:
            # Unterhalb des Bandes ist der Markt fair kalibriert — dort verliert
            # die Klasse Geld (-17,64 % gemessen), egal was die Prognose sagt.
            log_rows.append({**base, "decision": f"skip_band_billig_{no_live:.2f}"})
        elif abstand < args.abstand_min:
            log_rows.append({**base, "decision": f"skip_abstand_{abstand:.2f}"})
        else:
            c["event_id"] = event_id      # fuer den Jupiter-Link in der Mail
            tradeable.append((abstand, no_live, c, base))
        time.sleep(0.4)

    # Rangfolge = eigene Prognose, nicht Preis: wer am weitesten ueber der
    # Bucket-Kante sitzt, kommt zuerst. Der Preis hat seine Arbeit schon getan
    # (Band-Filter oben) und darf hier nicht noch einmal mitreden — sonst waere
    # es wieder das V1-Muster "der Markt bestimmt die Auswahl".
    tradeable.sort(key=lambda x: -x[0])
    picks = []
    for abstand, no_live, c, base in tradeable:
        # Cap ist eine Obergrenze, KEIN Ziel — es wird nie aufgefuellt, weil
        # ausserhalb des Bandes gar nichts mehr in dieser Liste steht.
        if len(picks) >= args.cap:
            log_rows.append({**base, "decision": "skip_cap"})
        else:
            picks.append((abstand, no_live, c, base))

    # Guthaben-Check VOR dem Senden: am 27.07. lief Mexico City in fail_send,
    # weil der Bot blind sendete. Lieber weniger Lays als stille Luecken in der
    # Messreihe — eine gekuerzte Liste ist dokumentiert, ein Sendefehler nicht.
    if picks and not args.dry_run:
        cash = None
        try:
            from autopilot import wallet_cash
            r = wallet_cash()
            cash = r[0] if r else None
        except Exception as e:
            print(f"  (Guthaben nicht abrufbar, fahre ungeprueft fort: {e})")
        if cash is not None:
            passt = int(cash / (args.usd * CASH_PUFFER))
            print(f"Guthaben {cash:.2f} JupUSD — reicht fuer {passt} Lays a {args.usd} $.")
            if passt < len(picks):
                for abstand, no_live, c, base in picks[passt:]:
                    log_rows.append({**base, "decision": f"skip_cash_{cash:.2f}"})
                picks = picks[:passt]

    picks_txt = ", ".join(f"{c['city']} {c['k']}° NO@{no:.3f} d{a:+.2f}"
                          for a, no, c, _ in picks) or "—"
    print(f"{len(tradeable)} im Band [{args.band_lo:.2f}, {args.band_hi:.2f}), "
          f"{len(picks)} werden gesetzt (Rang nach Temperaturabstand): {picks_txt}")

    n_ok = 0
    gekauft, fehlgeschlagen = [], []   # fuer die Abschluss-Mail
    for i, (abstand, no_live, c, base) in enumerate(picks):
        # Atempause zwischen den Kaeufen. Am 27.07. lief der DRITTE Kauf (Mexico
        # City) in zwei 429er und fiel aus — der Bot feuerte /execute ohne Pause
        # hintereinander. Mit Cap 8 statt 6 waere das oefter passiert.
        if i:
            time.sleep(4)
        # Tick-Size 1 Cent: Limit = naechster ganzer Cent ueber dem Ask (Fill-Puffer)
        limit = min(round((int(no_live * 100) + 1) / 100, 2), LIMIT_CAP)
        print(f"\nKauf {c['city']} {c['k']}°C NO {args.usd}$ @ limit {limit} ({c['market_id']}):")
        result = {"ok": False}
        # Drei Versuche mit wachsender Pause. 10 s reichten gegen das Rate-Limit
        # nicht: Versuch 2 kam am 27.07. sofort wieder als 429 zurueck.
        for attempt, pause in ((1, 15), (2, 45), (3, 0)):
            try:
                result = place(owner, c["market_id"], False, args.usd, limit, kp,
                               send=not args.dry_run)
            except Exception as e:
                txt = str(e)
                print(f"  Versuch {attempt} Exception: {txt}")
                result = {"ok": False}
                # 429 heisst "zu schnell", nicht "geht nicht" — laenger warten.
                if "429" in txt and pause:
                    pause = max(pause, 30)
            if result.get("ok"):
                break
            if pause:
                print(f"  ... {pause} s warten, dann neuer Versuch")
                time.sleep(pause)
        if args.dry_run:
            log_rows.append({**base, "decision": "dry_run", "usd": args.usd})
            continue
        if not result.get("ok"):
            log_rows.append({**base, "decision": "fail_send", "usd": args.usd})
            fehlgeschlagen.append(f"{c['city']} {c['k']}°C")
            continue
        sig = (result.get("resp") or {}).get("signature", "")
        contracts, avg = verify_fill(owner, c["market_id"])
        decision = "bought" if contracts else "sent_unverified"
        n_ok += 1 if contracts else 0
        row = {**base, "decision": decision, "usd": args.usd,
               "contracts": contracts or "", "avg_price": avg or "", "signature": sig}
        log_rows.append(row)
        gekauft.append((c, row))   # Zeile mitfuehren -> unten mit finalem Fill patchen
        print(f"  -> {decision}: {contracts} Kontr. @ {avg}")

    # Fills final abfragen, BEVOR das CSV geschrieben wird: verify_fill() oben wartet
    # nur ~15 s und meldet echte Kaeufe faelschlich als 'sent_unverified' mit leeren
    # Zahlen — die Auswertung filtert aber auf 'bought' und haette sie uebersehen.
    bought = []
    if not args.dry_run and gekauft:
        fills = final_fills(owner, {c["market_id"] for c, _ in gekauft})
        for c, row in gekauft:
            ct, av, ko = fills.get(c["market_id"], (0.0, 0.0, 0.0))
            if ct and row["decision"] == "sent_unverified":
                row.update(decision="bought", contracts=ct, avg_price=f"{av:.4f}")
                n_ok += 1
                print(f"  nachgetragen: {c['city']} {c['k']}° -> bought "
                      f"{ct} Kontr. @ {av:.4f}")
            # Ist der Fill trotz Retry nicht abrufbar, NICHT mit Nullen mailen:
            # der Kauf lief zum Limit, der Ask ist die beste bekannte Naeherung.
            # Als Schaetzung kenntlich gemacht — das CSV bleibt davon unberuehrt.
            geschaetzt = not ct
            if geschaetzt:
                ask = float(row.get("buy_no_live") or 0) or None
                if ask:
                    av, ct, ko = ask, args.usd / ask, args.usd
            bought.append(dict(city=c["city"], k=c["k"], ct=ct, av=av, ko=ko,
                               event_id=c.get("event_id"), geschaetzt=geschaetzt))

    append_log(log_rows)
    print(f"\nFertig: {n_ok} Kaeufe bestaetigt, {len(log_rows)} Zeilen geloggt -> {LOG_CSV}")

    if not args.dry_run and (bought or fehlgeschlagen):
        send_summary_mail(run_utc, target, args.cap, len(tradeable), bought, fehlgeschlagen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
