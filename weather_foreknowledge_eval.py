#!/usr/bin/env python3
"""
weather_foreknowledge_eval.py — Weiss der Markt vom Kipp, bevor er sichtbar ist?

Die Frage des Nutzers (25.07.2026, woertlich): "ich will eigentlich nur wissen
woher der markt weiss was passiert wenn noch kein wert zu sehen ist."

Beleg damals (Helsinki, 19 vs 20): der Kauffluss lief der 20er-Meldung ~30 min
voraus. Offen war, ob das reproduzierbar ist oder ein Einzelfall.

METHODISCHE FALLE, die diesen Test definiert
--------------------------------------------
Ein naiver Vorlauf-Test misst nur Trend-Extrapolation. Beispiel Madrid 29.07.:
der 39er-Bucket wurde ab 12:00 UTC stark gekauft, die 39 erschien erst 15:00 UTC
— drei Stunden "Vorlauf". Erklaerung ist aber trivial: die Tabelle stieg mit
1 K je 30 min (35 °C um 14:00 lokal), jeder konnte 39 hochrechnen.

Interessant ist nur das PLATEAU: die Tabelle steht seit >= PLATEAU_MIN Minuten
auf k, der Anstieg stockt, und der Ausgang (kommt k+1 noch?) ist offen. Genau
dort — und nur dort — misst dieser Test den Fluss.

DESIGN
------
Ereignis je Stadt-Tag (var='max'):
  m(t) = laufendes Maximum der settelnden WU-History-Tabelle
  Plateau = m steht seit >= PLATEAU_MIN min auf k
  KIPP    : m springt danach auf k+1 bei t0        -> Fall
  BLEIBER : m bleibt bis Tagesende auf k           -> Kontrolle

Gemessen wird der Netto-Taker-Fluss im (k+1)-eq-Bucket im Fenster
[t0 - WINDOW_MIN, t0), normalisiert auf YES-Aequivalent:
  BUY Yes / SELL No  -> bullish (der Bucket kommt)
  SELL Yes / BUY No  -> bearish
Bei Bleibern gibt es kein t0; dort wird das Fenster an das Plateau-Ende gelegt
(letzte Beobachtung des lokalen Tagesmaximums + WINDOW_MIN), damit beide Gruppen
dieselbe Phase des Tages vergleichen.

Normiert wird auf das Tagesvolumen des Buckets, damit grosse und kleine Maerkte
vergleichbar sind: flow_share = netto_kontrakte / gehandelte_kontrakte_am_tag.

GATES (vorregistriert, vor dem ersten Lauf festgelegt)
  G1  Kipper: mittlerer flow_share im Fenster > 0, t > 2.0
  G2  Kipper vs Bleiber: Differenz t > 2.0 (der Fluss muss den Kipp TRENNEN,
      nicht bloss "Buckets werden immer gekauft")
  G3  n >= 20 Kipper mit auswertbarem Fenster
  G4  Der Befund ueberlebt den Ausschluss der letzten 15 min vor t0
      (sonst ist es Latenz gegen METAR-Bots, und die ist tot —
      s. weather-tile-vs-table-latency)

Aufruf:
    python weather_foreknowledge_eval.py --plateau 60 --window 60
    python weather_foreknowledge_eval.py --limit 40        # schneller Probelauf
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pymssql
import requests

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weather_stations import station_info, has_wunderground  # noqa: E402

WU_KEY = "e1f10a1e78da46f5b10a1e78da96f525"  # oeffentlicher Web-Key, wie im Ladder-Logger
WU_URL = "https://api.weather.com/v1/location/{icao}:9:{cc}/observations/historical.json"
GAMMA = "https://gamma-api.polymarket.com/markets/{mid}"
TAPE = "https://data-api.polymarket.com/trades"

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".foreknowledge_cache")
os.makedirs(CACHE, exist_ok=True)


def get_json(url, params=None, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=25)
            if r.status_code == 429:
                time.sleep(4 + 3 * i)
                continue
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            time.sleep(2 + 2 * i)
    return None


def cached(name, fn):
    p = os.path.join(CACHE, name)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    v = fn()
    if v is not None:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(v, f)
    return v


def wu_observations(icao, datum):
    """WU-History-Tabelle (die settelnde Quelle) fuer einen Tag. valid_time_gmt ist UTC."""
    if not has_wunderground(icao):
        return None          # z. B. HKO — Wunderground fuehrt diese Station nicht
    cc = (station_info(icao) or {}).get("country")
    if not cc:
        return None
    ds = datum.strftime("%Y%m%d")

    def fetch():
        d = get_json(WU_URL.format(icao=icao, cc=cc),
                     {"apiKey": WU_KEY, "units": "m", "startDate": ds})
        return (d or {}).get("observations")

    obs = cached(f"wu_{icao}_{ds}.json", fetch)
    if not obs:
        return None
    return [o for o in obs if o.get("temp") is not None]


def condition_id(market_id):
    """POLY-3124879 -> Polymarket-conditionId (fuer den Trade-Tape)."""
    mid = str(market_id).replace("POLY-", "")

    def fetch():
        d = get_json(GAMMA.format(mid=mid))
        return {"conditionId": (d or {}).get("conditionId")} if d else None

    v = cached(f"cid_{mid}.json", fetch)
    return (v or {}).get("conditionId")


def trades(cid):
    def fetch():
        out, off = [], 0
        while off < 4000:
            b = get_json(TAPE, {"market": cid, "limit": 500, "offset": off})
            if b is None:
                break
            if not b:
                break
            out += b
            off += 500
            if len(b) < 500:
                break
            time.sleep(0.25)
        return out

    return cached(f"tape_{cid[:16]}.json", fetch) or []


def running_max_events(obs):
    """[(t_utc, laufendes_max)] aus den Beobachtungen, nur bei Aenderung."""
    ev, m = [], None
    for o in sorted(obs, key=lambda x: x["valid_time_gmt"]):
        t = int(o["valid_time_gmt"])
        v = int(round(float(o["temp"])))
        if m is None or v > m:
            m = v
            ev.append((t, m))
    return ev


def netto_flow(tr, t_von, t_bis):
    """Netto-Taker-Fluss (Kontrakte, YES-aequivalent) im Fenster + Tagesvolumen."""
    netto = 0.0
    for t in tr:
        ts = int(t["timestamp"])
        if not (t_von <= ts < t_bis):
            continue
        bull = (t["side"] == "BUY" and t["outcome"] == "Yes") or \
               (t["side"] == "SELL" and t["outcome"] == "No")
        netto += float(t["size"]) * (1 if bull else -1)
    return netto


def tagesvolumen(tr, t_von, t_bis):
    return sum(float(t["size"]) for t in tr if t_von <= int(t["timestamp"]) < t_bis) or 0.0


def welch_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return (a.mean() - b.mean()) / np.sqrt(va + vb) if va + vb > 0 else float("nan")


def one_sample_t(a):
    a = np.asarray(a, float)
    if len(a) < 2 or a.std(ddof=1) == 0:
        return float("nan")
    return a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))


def lade_tage(limit=0):
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT city, icao, target_date, MAX(settle_k) AS settle_k
        FROM bb_WeatherLadders
        WHERE var='max' AND settle_k IS NOT NULL AND icao IS NOT NULL
        GROUP BY city, icao, target_date
        ORDER BY target_date, city""")
    tage = cur.fetchall()
    cur.execute("""
        SELECT DISTINCT city, target_date, k, market_id
        FROM bb_WeatherLadders
        WHERE var='max' AND kind='eq' AND market_id IS NOT NULL""")
    mkt = {(c, d, k): m for c, d, k, m in cur.fetchall()}
    conn.close()
    return (tage[-limit:] if limit else tage), mkt


def fixed_hour_mode(a):
    """Der Confound-freie Test.

    Beide Gruppen werden am SELBEN relativen Objekt gemessen — dem Bucket eine
    Stufe ueber dem laufenden Tabellen-Maximum — und zur SELBEN lokalen Uhrzeit T,
    die nicht vom Ausgang abhaengt. Unterschieden wird nur, ob dieser Bucket
    spaeter noch eintrat.

      KAM      : settle_k >  m(T)   -> die Stufe kam noch
      KAM NICHT: settle_k == m(T)   -> das Plateau hielt bis Tagesende

    Bedingung fuer beide: das Plateau laeuft bei T bereits >= --plateau Minuten,
    der Ausgang ist also offen und der Anstieg gestockt.
    """
    tage, mkt = lade_tage(a.limit)
    print(f"Stadt-Tage: {len(tage)} | feste lokale Uhrzeit T={a.fixed_hour:.2f} h, "
          f"Fenster {a.window} min, Plateau >= {a.plateau} min, skip-last {a.skip_last}\n")

    kam, kam_nicht, verworfen = [], [], defaultdict(int)
    zeilen = []

    for city, icao, datum, settle_k in tage:
        st = station_info(icao) or {}
        tzname = st.get("tz")
        obs = wu_observations(icao, datum)
        if not obs or not tzname:
            verworfen["keine WU-Daten/tz"] += 1
            continue
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(tzname)
        except Exception:
            verworfen["tz unbekannt"] += 1
            continue

        # T = feste lokale Uhrzeit am Zieltag, in UTC umgerechnet
        stunde = int(a.fixed_hour)
        minute = int(round((a.fixed_hour - stunde) * 60))
        T = int(datetime(datum.year, datum.month, datum.day, stunde, minute,
                         tzinfo=tz).timestamp())

        ev = running_max_events(obs)
        vor_T = [(t, m) for t, m in ev if t <= T]
        if not vor_T:
            verworfen["kein Wert vor T"] += 1
            continue
        t_m, m_T = vor_T[-1]                      # laufendes Max bei T + seit wann
        if (T - t_m) / 60.0 < a.plateau:
            verworfen[f"Plateau < {a.plateau} min"] += 1
            continue
        if settle_k < m_T:
            verworfen["settle_k < Max(T) — inkonsistent"] += 1
            continue

        ziel = m_T + 1                            # IMMER eine Stufe ueber dem Max
        mid = mkt.get((city, datum, ziel))
        if not mid:
            verworfen["kein eq-Markt fuer Max+1"] += 1
            continue
        cid = condition_id(mid)
        if not cid:
            verworfen["keine conditionId"] += 1
            continue
        tr = trades(cid)
        if not tr:
            verworfen["kein Tape"] += 1
            continue

        t_bis = T - a.skip_last * 60
        t_von = t_bis - a.window * 60
        vol = tagesvolumen(tr, T - 24 * 3600, T)
        if vol < 50:
            verworfen["Bucket zu illiquide"] += 1
            continue
        share = netto_flow(tr, t_von, t_bis) / vol

        if settle_k >= ziel:
            kam.append(share)
        else:
            kam_nicht.append(share)
        zeilen.append((str(datum), city, m_T, ziel, settle_k,
                       settle_k >= ziel, share, vol))
        time.sleep(0.1)

    print("Verworfen:")
    for g, n in sorted(verworfen.items(), key=lambda x: -x[1]):
        print(f"  {g:<34} {n}")

    if not kam or not kam_nicht:
        print("\nEine Gruppe leer — nicht auswertbar.")
        return

    k_arr, n_arr = np.array(kam, float), np.array(kam_nicht, float)
    print(f"\n{'='*76}")
    print(f"Bucket Max+1 KAM       n={len(k_arr):>3}  Ø flow_share {k_arr.mean():+.4f}  "
          f"Median {np.median(k_arr):+.4f}  t={one_sample_t(k_arr):+.2f}")
    print(f"Bucket Max+1 KAM NICHT n={len(n_arr):>3}  Ø flow_share {n_arr.mean():+.4f}  "
          f"Median {np.median(n_arr):+.4f}  t={one_sample_t(n_arr):+.2f}")
    tdiff = welch_t(k_arr, n_arr)
    print(f"TRENNUNG               Welch-t = {tdiff:+.2f}")
    print(f"{'='*76}")
    print("\nGATE (der einzige, der hier zaehlt):")
    print(f"  Fluss trennt KAM von KAM-NICHT, t > 2.0 : t={tdiff:+.2f}  "
          f"{'PASS' if tdiff > 2.0 else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plateau", type=int, default=60,
                    help="Minuten, die die Tabelle auf k stehen muss (Default 60)")
    ap.add_argument("--window", type=int, default=60,
                    help="Messfenster vor t0 in Minuten (Default 60)")
    ap.add_argument("--skip-last", type=int, default=0,
                    help="G4: die letzten N Minuten vor t0 ausblenden")
    ap.add_argument("--limit", type=int, default=0, help="nur N Stadt-Tage (Probelauf)")
    ap.add_argument("--fixed-hour", type=float, default=None,
                    help="SAUBERER MODUS: feste lokale Uhrzeit T statt t0-Fenster. "
                         "Misst den Fluss in [T-window, T) im Bucket (laufendes Max + 1) "
                         "und gruppiert danach, ob dieser Bucket spaeter noch kam. "
                         "Kein Niveau-Confound, kein Ergebnis-Leck im Fensterrand.")
    ap.add_argument("--profit", type=int, default=None,
                    help="Verwertbarkeits-Test bei lokaler Uhrzeit N: Fluss gegen Preis")
    a = ap.parse_args()
    if a.profit is not None:
        a.profit_hour = a.profit
        return profit_mode(a)
    if a.fixed_hour is not None:
        return fixed_hour_mode(a)

    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT city, icao, target_date, MAX(settle_k) AS settle_k
        FROM bb_WeatherLadders
        WHERE var='max' AND settle_k IS NOT NULL AND icao IS NOT NULL
        GROUP BY city, icao, target_date
        ORDER BY target_date, city""")
    tage = cur.fetchall()

    cur.execute("""
        SELECT DISTINCT city, target_date, k, market_id
        FROM bb_WeatherLadders
        WHERE var='max' AND kind='eq' AND market_id IS NOT NULL""")
    mkt = {(c, d, k): m for c, d, k, m in cur.fetchall()}
    conn.close()

    if a.limit:
        tage = tage[-a.limit:]
    print(f"Stadt-Tage in der Auswahl: {len(tage)}  "
          f"(Plateau {a.plateau} min, Fenster {a.window} min, skip-last {a.skip_last})\n")

    kipper, bleiber, verworfen = [], [], defaultdict(int)
    details = []

    for city, icao, datum, settle_k in tage:
        obs = wu_observations(icao, datum)
        if not obs:
            verworfen["keine WU-Daten"] += 1
            continue
        ev = running_max_events(obs)
        if len(ev) < 2:
            verworfen["kein Verlauf"] += 1
            continue

        letzte_zeit = max(int(o["valid_time_gmt"]) for o in obs)

        # Kandidat: das letzte Plateau vor dem finalen Wert
        # ev[-1] = (t0, settle_k_erreicht); ev[-2] = (t_k, k) davor
        t_final, m_final = ev[-1]
        if m_final != settle_k:
            verworfen["Tagesmax != settle_k"] += 1
            continue

        t_k, k = ev[-2]
        plateau_min = (t_final - t_k) / 60.0
        if plateau_min < a.plateau:
            verworfen[f"Plateau < {a.plateau} min"] += 1
            continue

        # KIPP-Fall: der Bucket k+1 == settle_k ist erst bei t_final sichtbar geworden
        mid = mkt.get((city, datum, settle_k))
        if not mid:
            verworfen["kein eq-Markt fuer settle_k"] += 1
            continue
        cid = condition_id(mid)
        if not cid:
            verworfen["keine conditionId"] += 1
            continue
        tr = trades(cid)
        if not tr:
            verworfen["kein Tape"] += 1
            continue

        t_bis = t_final - a.skip_last * 60
        t_von = t_bis - a.window * 60
        if t_von < t_k:                       # Fenster muss im Plateau liegen
            t_von = t_k
        if t_bis - t_von < 15 * 60:
            verworfen["Fenster zu kurz"] += 1
            continue

        tag_von = t_final - 24 * 3600
        vol = tagesvolumen(tr, tag_von, t_final)
        if vol < 50:
            verworfen["Bucket zu illiquide"] += 1
            continue
        share = netto_flow(tr, t_von, t_bis) / vol
        kipper.append(share)
        details.append((str(datum), city, k, settle_k, plateau_min, share, vol))

        # KONTROLLE aus demselben Tag: der Bucket settle_k+1, der NICHT kam.
        # Gleiches Fenster, gleiche Marktphase — nur ist hier der Ausgang negativ.
        mid2 = mkt.get((city, datum, settle_k + 1))
        if mid2:
            cid2 = condition_id(mid2)
            if cid2:
                tr2 = trades(cid2)
                vol2 = tagesvolumen(tr2, tag_von, t_final)
                if vol2 >= 50:
                    bleiber.append(netto_flow(tr2, t_von, t_bis) / vol2)
        time.sleep(0.15)

    print("Verworfen:")
    for g, n in sorted(verworfen.items(), key=lambda x: -x[1]):
        print(f"  {g:<28} {n}")
    print()

    if not kipper:
        print("Keine auswertbaren Faelle.")
        return

    print(f"{'Datum':<12}{'Stadt':<16}{'k->settle':<12}{'Plateau':>9}{'flow_share':>12}{'Vol':>10}")
    for d, c, k, s, pl, sh, v in sorted(details):
        print(f"{d:<12}{c[:15]:<16}{f'{k} -> {s}':<12}{pl:>7.0f}m{sh:>+12.3f}{v:>10.0f}")

    k_arr = np.array(kipper, float)
    print(f"\n{'='*74}")
    print(f"KIPPER   n={len(k_arr):>3}  mittlerer flow_share {k_arr.mean():+.4f}  "
          f"Median {np.median(k_arr):+.4f}  t={one_sample_t(k_arr):+.2f}")
    if bleiber:
        b_arr = np.array(bleiber, float)
        print(f"KONTROLLE n={len(b_arr):>3}  mittlerer flow_share {b_arr.mean():+.4f}  "
              f"Median {np.median(b_arr):+.4f}  t={one_sample_t(b_arr):+.2f}")
        print(f"DIFFERENZ  Welch-t = {welch_t(k_arr, b_arr):+.2f}")
    print(f"{'='*74}")
    print("\nGATES:")
    t1 = one_sample_t(k_arr)
    print(f"  G1 Kipper-Fluss > 0, t > 2.0 : t={t1:+.2f}  "
          f"{'PASS' if t1 > 2.0 else 'FAIL'}")
    if bleiber:
        t2 = welch_t(k_arr, np.array(bleiber, float))
        print(f"  G2 trennt vs Kontrolle, t>2.0: t={t2:+.2f}  "
              f"{'PASS' if t2 > 2.0 else 'FAIL'}")
    print(f"  G3 n >= 20                   : n={len(k_arr)}  "
          f"{'PASS' if len(k_arr) >= 20 else 'FAIL'}")
    print("  G4 siehe Lauf mit --skip-last 15")



# ---------------------------------------------------------------------------
# Verwertbarkeits-Test (29.07.2026, auf Nachfrage "kann man daraus Profit
# schlagen?"): Traegt der FLUSS Information UEBER DEN PREIS HINAUS?
#
# Der Befund oben sagt nur, dass der Markt informiert ist — das spricht erstmal
# GEGEN Verwertbarkeit. Nutzbar waere er nur, wenn der Netto-Fluss im
# (Max+1)-Bucket den Ausgang besser trennt als der dort bereits gestellte Preis.
# Ist der Preis der bessere Praediktor, steckt alles schon drin und es bleibt
# nichts zu holen.
#
# Aufruf:  python weather_foreknowledge_eval.py --profit 15
# ---------------------------------------------------------------------------

def preis_bei(tr, T):
    """Letzter gehandelter YES-Preis des Buckets vor T (aus dem Tape)."""
    vor = [t for t in tr if int(t["timestamp"]) < T]
    if not vor:
        return None
    t = max(vor, key=lambda x: int(x["timestamp"]))
    p = float(t["price"])
    return p if t["outcome"] == "Yes" else 1.0 - p


def profit_mode(a):
    from math import sqrt
    tage, mkt = lade_tage(a.limit)
    T_h = a.profit_hour
    print(f"Verwertbarkeit: T={T_h}:00 lokal, Fenster {a.window} min, "
          f"Plateau >= {a.plateau} min\n")

    rows = []
    for city, icao, datum, settle_k in tage:
        st = station_info(icao) or {}
        obs = wu_observations(icao, datum)
        if not obs or not st.get("tz"):
            continue
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(st["tz"])
        except Exception:
            continue
        T = int(datetime(datum.year, datum.month, datum.day, T_h, 0,
                         tzinfo=tz).timestamp())
        ev = running_max_events(obs)
        vor_T = [(t, m) for t, m in ev if t <= T]
        if not vor_T:
            continue
        t_m, m_T = vor_T[-1]
        if (T - t_m) / 60.0 < a.plateau or settle_k < m_T:
            continue
        mid = mkt.get((city, datum, m_T + 1))
        if not mid:
            continue
        cid = condition_id(mid)
        if not cid:
            continue
        tr = trades(cid)
        if not tr:
            continue
        vol = tagesvolumen(tr, T - 24 * 3600, T)
        if vol < 50:
            continue
        p = preis_bei(tr, T)
        if p is None:
            continue
        share = netto_flow(tr, T - a.window * 60, T) / vol
        rows.append((share, p, 1 if settle_k >= m_T + 1 else 0))

    if len(rows) < 40:
        print(f"zu wenige Faelle ({len(rows)})")
        return
    fl = np.array([r[0] for r in rows], float)
    pr = np.array([r[1] for r in rows], float)
    y = np.array([r[2] for r in rows], float)
    print(f"n = {len(rows)}   davon kam: {int(y.sum())} ({y.mean()*100:.0f} %)\n")

    def auc(x):
        """Rang-AUC: Wahrscheinlichkeit, dass ein KAM-Fall hoeher liegt."""
        o = np.argsort(x)
        r = np.empty(len(x), float)
        r[o] = np.arange(1, len(x) + 1)
        n1, n0 = y.sum(), (1 - y).sum()
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    print(f"{'Praediktor':<24}{'AUC':>8}{'Welch-t':>10}")
    for name, x in [("Netto-Fluss (share)", fl), ("Preis des Buckets", pr)]:
        print(f"{name:<24}{auc(x):>8.3f}{welch_t(x[y==1], x[y==0]):>10.2f}")

    # Traegt der Fluss NACH Kontrolle fuer den Preis noch? Residualisieren.
    A = np.vstack([np.ones(len(pr)), pr]).T
    beta, *_ = np.linalg.lstsq(A, fl, rcond=None)
    resid = fl - A @ beta
    print(f"{'Fluss | Preis (Residuum)':<24}{auc(resid):>8.3f}"
          f"{welch_t(resid[y==1], resid[y==0]):>10.2f}")
    print("\nLesart: AUC 0,50 = wertlos. Traegt das Residuum nicht mehr,")
    print("steckt die Information vollstaendig im Preis — nichts zu holen.")


if __name__ == "__main__":
    main()
