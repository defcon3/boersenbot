#!/usr/bin/env python3
"""
weather_lay_guardrail_eval.py — Lohnt ein automatischer Waechter fuer die
-1-Lay-Klasse, der aussteigt, wenn das laufende Tageshoch auf dem gelayten
Bucket sitzt?

ANLASS (24.07.2026)
  Der Autobuy (`weather_minus1_autobuy.py`) layt den Bucket EINE Klasse unter
  dem Markt-Favoriten und haelt bis zum Settlement — kein Ausstieg, kein TP.
  Verloren wird genau dann, wenn das Tageshoch **exakt auf dem gelayten Bucket
  stehenbleibt**. Da das laufende Maximum monoton steigt, laeuft es auf diese
  Gefahr zu und muss darueber hinaus, damit die Wette aufgeht.

  Frage: Erkennt man das rechtzeitig? Konkret — wenn das laufende Maximum zur
  Stunde T bereits auf dem gelayten Bucket sitzt und laut Basisrate kaum noch
  etwas kommt, ist die Wette so gut wie verloren. Ein Waechter koennte dann
  aussteigen, statt auf null zu laufen. Das braucht keinen Menschen (der Nutzer
  ist unter der Woche nicht da) und ist genau das, was der beobachteten
  Gross-Wallet fehlt (446 Kaeufe : 7 Verkaeufe).

BASISRATE (gemessen 24.07., 125 Stadt-Tage, s. Memory `weather-daily-max-timing`)
  Anteil Tage, an denen das Tageshoch nach der Uhrzeit noch steigt:
  13:20 -> 91 % | 14:20 -> 87 % | 15:20 -> 76 % | 16:20 -> 41 % | 17:20 -> 12 %

DIESES SKRIPT — Stufe 1 (wetterseitig, ohne Preise)
  Fuer jeden abgerechneten -1-Kandidaten aus `bb_WeatherLadders`:
    - Intraday-METAR (WU) der Zielstadt/-datum -> laufendes Maximum je Ortsstunde
    - Signal(T) := gerundetes laufendes Maximum == gelayter Bucket k zur Stunde T
    - Ausgang := `settle_result` (True = gelayter Bucket getroffen = Lay VERLOREN)
  Ausgegeben wird die Vierfeldertafel Signal x Ausgang je Stunde, damit sichtbar
  wird, ob der Waechter Verlierer trifft (Trefferquote) und wie viele Gewinner er
  faelschlich abbrechen wuerde (Fehlalarm).

  Stufe 2 (--stage2) haengt die Ausstiegspreise an (Polymarket `prices-history`,
  NO-Token, s. `POLYMARKET_DATA_API.md`) und vergleicht Halten gegen Waechter.

ERGEBNIS 24.07.2026 (Lead 1, 13 Zieltage 11.-23.07., Details in
preregs/weather_lay_guardrail_2026_07_24.md)
  - Signal trennt scharf: 16:20 -> 67 % Treffer gegen 2 % Basisrate, 17:20 -> 88 %.
  - Markt preist es nicht ein: Ausstiegspreis 0,545 gegen echte Gewinnquote 0,293
    (16:20). Tages-geclustert 13/13 Tage positiv, Mittel +0,294, t = 5,46.
  - ABER der Nutzen haengt an der Breite der Auswahl:
        alle Kandidaten (144 Pos., 21,5 % Verlierer): +30,60 $ -> +79,75 $
        Live-Auswahl   ( 38 Pos.,  7,5 % Verlierer): +10,52 $ ->  +6,38 $
    Im engen Buch kappt der Waechter mehr Gewinner als er Verlierer rettet.
    Er ist also kein Zusatz zum heutigen Bot, sondern die Voraussetzung fuer Breite.
  IN-SAMPLE, 13 Tage, nur Hochsommer -> Pre-Reg + Forward-Test noetig, kein Deploy.

Aufruf:
  python weather_lay_guardrail_eval.py                      # Stufe 1, Lead 1, alle
  python weather_lay_guardrail_eval.py --stage2             # + P&L-Vergleich
  python weather_lay_guardrail_eval.py --picked-only --stage2   # Live-Auswahl
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pymssql
import requests

from weather_ladder_logger import DB_CONFIG
from weather_latency_logger import CITIES

WU_KEY = "e1f10a1e78da46f5b10a1e78da96f525"
CACHE = Path(".wu_intraday_cache")
HOURS = [13, 14, 15, 16, 17, 18]

# Laenderkuerzel fuer den WU-Standortschluessel ICAO:9:LAND
CC = {"EF": "FI", "ED": "DE", "LF": "FR", "LE": "ES", "LI": "IT", "EG": "GB",
      "EH": "NL", "EP": "PL", "FA": "ZA", "MM": "MX", "SA": "AR", "RJ": "JP",
      "RK": "KR", "Z": "CN", "VH": "HK", "WM": "MY", "OP": "PK", "OE": "SA",
      "LT": "TR", "NZ": "NZ"}

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def country(icao):
    for pre in (icao[:2], icao[:1]):
        if pre in CC:
            return CC[pre]
    return None


def wu_day(icao, date):
    """Intraday-Beobachtungen -> [(utc_datetime, temp)] (mit Plattencache)."""
    CACHE.mkdir(exist_ok=True)
    cf = CACHE / f"{icao}_{date:%Y%m%d}.json"
    if cf.exists():
        raw = json.loads(cf.read_text())
    else:
        cc = country(icao)
        if not cc:
            return []
        try:
            r = requests.get(
                f"https://api.weather.com/v1/location/{icao}:9:{cc}/observations/historical.json",
                params={"apiKey": WU_KEY, "units": "m", "startDate": f"{date:%Y%m%d}"},
                timeout=25)
            raw = r.json().get("observations") or [] if r.ok else []
        except requests.RequestException:
            raw = []
        cf.write_text(json.dumps([{"t": o.get("temp"), "v": o.get("valid_time_gmt")}
                                  for o in raw]))
        raw = json.loads(cf.read_text())
        time.sleep(0.1)
    out = []
    for o in raw:
        t, v = o.get("t"), o.get("v")
        if t is not None and v is not None:
            out.append((datetime.utcfromtimestamp(v), float(t)))
    out.sort()
    return out


def running_max_at(obs, off, hour):
    """Gerundetes laufendes Tageshoch bis <hour>:20 Ortszeit (None = keine Daten)."""
    vals = [t for u, t in obs
            if (u + timedelta(hours=off)).hour < hour
            or ((u + timedelta(hours=off)).hour == hour and (u + timedelta(hours=off)).minute <= 20)]
    return round(max(vals)) if vals else None


def load_candidates(picked_only, lead=1):
    c = pymssql.connect(**DB_CONFIG)
    cur = c.cursor(as_dict=True)
    cur.execute(
        "SELECT target_date, city, icao, k, buy_no, settle_result, settle_k, "
        "       wu_settle_k, market_fav_k, snapshot_utc, market_id "
        "FROM bb_WeatherLadders "
        "WHERE var='max' AND kind='eq' AND offset_fav=-1 AND status='open' "
        "  AND settle_result IS NOT NULL AND buy_no IS NOT NULL "
        "  AND buy_no BETWEEN 0.50 AND 0.97 "
        "  AND DATEDIFF(day, CAST(snapshot_utc AS date), target_date) = %d "
        "ORDER BY target_date, buy_no DESC", (lead,))
    rows = cur.fetchall()
    c.close()
    if not picked_only:
        return rows
    # Live-Auswahl nachbilden: buy_no <= 0.97, konservativste zuerst,
    # erste 3 bedingungslos, danach nur buy_no >= 0.85, Cap 6.
    byday = defaultdict(list)
    for r in rows:
        if r["buy_no"] <= 0.97:
            byday[r["target_date"]].append(r)
    picked = []
    for day, rs in byday.items():
        rs.sort(key=lambda x: -x["buy_no"])
        for i, r in enumerate(rs[:6]):
            if i < 3 or r["buy_no"] >= 0.85:
                picked.append(r)
    return picked


PCACHE = Path(".pm_price_cache")


def no_price_series(market_id):
    """NO-Preisreihe eines Marktes -> [(utc_ts, preis)] (mit Plattencache)."""
    mid = str(market_id).replace("POLY-", "")
    PCACHE.mkdir(exist_ok=True)
    cf = PCACHE / f"{mid}.json"
    if cf.exists():
        return [(x["t"], x["p"]) for x in json.loads(cf.read_text())]
    hist = []
    try:
        m = requests.get(f"https://gamma-api.polymarket.com/markets/{mid}", timeout=25)
        if m.ok:
            toks = json.loads(m.json().get("clobTokenIds") or "[]")
            if len(toks) > 1:                      # Token 1 = "No"
                r = requests.get("https://clob.polymarket.com/prices-history",
                                 params={"market": toks[1], "interval": "max", "fidelity": 10},
                                 timeout=25)
                if r.ok:
                    hist = r.json().get("history", [])
    except requests.RequestException:
        hist = []
    cf.write_text(json.dumps(hist))
    time.sleep(0.12)
    return [(x["t"], x["p"]) for x in hist]


def price_at(series, ts, tol=1800):
    """Letzter Preis bis ts (max. tol Sekunden alt)."""
    best = None
    for t, p in series:
        if t <= ts:
            best = (t, p)
        else:
            break
    if best and ts - best[0] <= tol:
        return best[1]
    return None


def stage2(rows, hour):
    """P&L-Vergleich Halten vs. Waechter-Ausstieg zur Ortsstunde `hour`."""
    FEE = 0.036
    USD = 5.0
    hold = exitp = 0.0
    n = fired = saved_lost = cut_won = 0
    detail = []
    for r in rows:
        off = CITIES.get(r["city"])
        if off is None or not r["buy_no"]:
            continue
        obs = wu_day(r["icao"], r["target_date"])
        if len(obs) < 10:
            continue
        rm = running_max_at(obs, off, hour)
        if rm is None:
            continue
        contracts = USD / r["buy_no"]
        lost = bool(r["settle_result"])
        pnl_hold = (0.0 if lost else contracts * 1.0) - USD
        hold += pnl_hold
        n += 1
        if rm != r["k"]:                            # kein Signal -> halten
            exitp += pnl_hold
            continue
        # Signal: zur Stunde hour:20 Ortszeit aussteigen
        ts = int((datetime(r["target_date"].year, r["target_date"].month,
                          r["target_date"].day, hour, 20) - timedelta(hours=off)).timestamp())
        px = price_at(no_price_series(r["market_id"]), ts)
        if px is None:
            exitp += pnl_hold                       # kein Preis -> nicht handelbar
            continue
        fired += 1
        net_px = px - FEE * min(px, 1 - px)         # Verkaufsgebuehr
        pnl_exit = contracts * net_px - USD
        exitp += pnl_exit
        if lost:
            saved_lost += 1
        else:
            cut_won += 1
        detail.append((r["city"], r["target_date"], r["k"], r["buy_no"], px, lost))
    return dict(n=n, hold=hold, exit=exitp, fired=fired,
                saved=saved_lost, cut=cut_won, detail=detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picked-only", action="store_true",
                    help="nur die Maerkte, die der Live-Bot tatsaechlich gekauft haette")
    ap.add_argument("--stage2", action="store_true",
                    help="P&L-Vergleich Halten vs. Waechter (holt Marktpreise)")
    ap.add_argument("--lead", type=int, default=1,
                    help="Vorlauf in Tagen; der Live-Bot handelt Lead 1 (heute fuer morgen)")
    a = ap.parse_args()

    rows = load_candidates(a.picked_only, a.lead)
    print(f"Lead {a.lead}  |  Kandidaten (abgerechnet): {len(rows)}"
          f"{'  [Live-Auswahl nachgebildet]' if a.picked_only else '  [alle]'}")
    days = sorted({r["target_date"] for r in rows})
    print(f"Zieltage: {len(days)}  ({days[0]} bis {days[-1]})")
    losers = sum(1 for r in rows if r["settle_result"])
    print(f"Verlierer (gelayter Bucket getroffen): {losers} = {losers/len(rows)*100:.1f} %\n")

    # Vierfeldertafel je Pruefstunde
    tab = {h: defaultdict(int) for h in HOURS}
    missing = 0
    for r in rows:
        off = CITIES.get(r["city"])
        if off is None:
            continue
        obs = wu_day(r["icao"], r["target_date"])
        if len(obs) < 10:
            missing += 1
            continue
        lost = bool(r["settle_result"])
        for h in HOURS:
            rm = running_max_at(obs, off, h)
            if rm is None:
                continue
            sig = (rm == r["k"])
            tab[h][("sig" if sig else "nosig", "lost" if lost else "won")] += 1

    if missing:
        print(f"(ohne Intraday-Daten übersprungen: {missing})\n")

    print("Signal := laufendes Tageshoch sitzt zur Stunde GENAU auf dem gelayten Bucket\n")
    print(f"{'Ortszeit':<11}{'Signal':>8}{'davon Verl.':>13}{'Treffer':>10}"
          f"{'  |  ':^7}{'kein Sig.':>10}{'davon Verl.':>13}{'Basisrate':>11}")
    print("-" * 86)
    for h in HOURS:
        t = tab[h]
        sig = t[("sig", "lost")] + t[("sig", "won")]
        nos = t[("nosig", "lost")] + t[("nosig", "won")]
        if not sig or not nos:
            continue
        prec = t[("sig", "lost")] / sig * 100
        base = t[("nosig", "lost")] / nos * 100
        print(f"  {h}:20{'':<4}{sig:>8}{t[('sig','lost')]:>13}{prec:>9.0f} %"
              f"{'  |  ':^7}{nos:>10}{t[('nosig','lost')]:>13}{base:>10.0f} %")
    print("-" * 86)
    print("  'Treffer' = Anteil Verlierer unter den Signal-Tagen (Praezision des Wächters)")
    print("  'Basisrate' = Anteil Verlierer ohne Signal — der Vergleichsmaßstab")

    if not a.stage2:
        return
    print("\n\n=== STUFE 2: Was bringt der Ausstieg wirklich? (5 $ je Position) ===")
    print("  Ausstieg zum NO-Marktpreis der Stunde, abzüglich Verkaufsgebühr 3,6 %\n")
    print(f"{'Ortszeit':<11}{'Pos.':>6}{'Halten':>11}{'Wächter':>11}{'Differenz':>12}"
          f"{'ausgelöst':>11}{'Verl. raus':>12}{'Gew. gekappt':>14}")
    print("-" * 88)
    for h in HOURS:
        s2 = stage2(rows, h)
        if not s2["n"]:
            continue
        print(f"  {h}:20{'':<4}{s2['n']:>6}{s2['hold']:>10.2f}${s2['exit']:>10.2f}$"
              f"{s2['exit']-s2['hold']:>+11.2f}${s2['fired']:>11}{s2['saved']:>12}{s2['cut']:>14}")
    print("-" * 88)


if __name__ == "__main__":
    main()
