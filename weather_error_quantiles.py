# -*- coding: utf-8 -*-
"""weather_error_quantiles.py — Empirische Offset-Verteilung des Ist-Extremums
relativ zum bias-korrigierten ENS-Favoriten (Fensterklassen ... -2, -1, 0, +1, +2 ...).

Entstanden 11.07.2026 aus der Lay-Frage des Betreibers („mich interessieren
hauptsächlich Lay-Wetten"): Wie oft landet das gerundete Ist wirklich im
Favoriten-Fenster / 1 Grad daneben / 2 Grad daneben? Das ist die Groesse, die
entscheidet, welche Fensterklassen am Markt ueber-/unterbepreist sind — und
beantwortet zugleich die offene Frage der Flanken-Pre-Reg
(preregs/weather_low_yes_flanks_2026_07_11.md): Normal-Annahme vs Realitaet.

Methodik:
  - Datenbasis identisch zu weather_source_compare.py (Open-Meteo Previous-Runs
    previous_day1 = echter 24h-Lead-Forecast, IEM-METAR als Ist), ~700 Tage.
  - mu = roh-ENS(5 Modelle) − LOO-Bias (Leave-one-out, damit die Korrektur den
    bewerteten Tag nicht enthaelt); Favorit k0 = round_half_up(mu).
  - Offset = round_half_up(Ist) − k0; Klassen-Haeufigkeiten vs. dem, was die
    Normal-Annahme (Sigma aus denselben Daten) je Klasse behaupten wuerde.

Kernbefund (Lauf 11.07.2026, min: Paris/London/Shanghai/Seoul n=2423,
max: +Mexico City/Tokyo n=3475):
  - Zentrum ist SCHAERFER als die Normal-Annahme behauptet (P(0) emp. 37–38 %
    vs Normal 33–35 %) -> stuetzt „Markt-Sigma korrekt, Kalibrier-Sigma zu
    breit" (Gegenhypothese der YES-Flanken-Pre-Reg).
  - Minima sind warm-schief: P(+1)=26,0 % > P(−1)=20,1 % (Ist haeufiger waermer
    als der Favorit als kaelter); bei Maxima nahezu symmetrisch.
  - Weite Fenster sind duenn: P(±2)≈6–7,5 %, P(±3)≈1–2,3 % -> empirische Basis
    des Outlier-Lay-Programms (weather_outlier_screen*.py).
  - Achtung Seoul-Tageshoch: sigma≈2,1 und fette Tails (≥+4: 5,9 %) — einzelne
    Staedte koennen deutlich schlechter prognostizierbar sein; Pooling prueft
    man mit --city.

Aufruf:
  python weather_error_quantiles.py                       # min: 4 Kern-Staedte
  python weather_error_quantiles.py --var max --city "Mexico City,Shanghai,Seoul,London,Paris,Tokyo"
  python weather_error_quantiles.py --var max --screen-file <output von weather_outlier_screen.py>
      # ergaenzt eine Markt-Querschnitts-Tabelle: Ø YES/NO je Offset-Klasse
      # + Lay-ROI (emp. Klassen-P vs NO-Ask + Fee 0.07*min(p,1-p)).
      # VORSICHT Interpretation: Offsets sind relativ zu UNSEREM Favoriten;
      # weicht der Markt-Favorit ab (11.07.: 14/19 Staedten!), misst die Tabelle
      # auch Modell-vs-Markt-Zentrumsstreit, nicht nur Fenster-Fehlpreisung.
"""
import argparse
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import requests

from weather_source_compare import MODELS, PREVRUN, IEM, _airports, STATIONS

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

FEE = 0.07
DEFAULT_MIN = "Paris,London,Shanghai,Seoul"
DEFAULT_MAX = "Mexico City,Shanghai,Seoul,London,Paris,Tokyo"


def half_up(x):
    return math.floor(x + 0.5)


def ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fetch_city_both(icao, days):
    """Ein Previous-Runs-Fetch -> taegliche ENS-Min/Max (nur Tage mit allen 5
    Modellen) + ein IEM-Fetch -> Ist-(Min, Max) je lokalem Tag."""
    st = _airports[icao]
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=days)

    j = {}
    for attempt in range(4):
        r = requests.get(PREVRUN, params={
            "latitude": st["lat"], "longitude": st["lon"],
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "hourly": "temperature_2m_previous_day1",
            "models": ",".join(MODELS), "timezone": "auto",
        }, timeout=60)
        r.raise_for_status()
        j = r.json()
        hourly = j.get("hourly", {})
        if hourly.get("time") and any(v is not None for k in hourly if k != "time" for v in hourly[k]):
            break
        time.sleep(2 * (attempt + 1))
    tz_name = j.get("timezone", "UTC")
    times = hourly.get("time", [])
    per_model = {m: {} for m in MODELS}
    for m in MODELS:
        vals = hourly.get(f"temperature_2m_previous_day1_{m}", [])
        for t, v in zip(times, vals):
            if v is not None:
                per_model[m].setdefault(t[:10], []).append(v)
    days_all = None
    for m in MODELS:
        ks = set(per_model[m].keys())
        days_all = ks if days_all is None else (days_all & ks)
    ens_min = {d: sum(min(per_model[m][d]) for m in MODELS) / len(MODELS) for d in days_all or set()}
    ens_max = {d: sum(max(per_model[m][d]) for m in MODELS) / len(MODELS) for d in days_all or set()}

    r = requests.get(IEM, params={
        "station": icao, "data": "tmpc",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "tz": tz_name, "format": "onlycomma", "latlon": "no", "elev": "no",
        "missing": "M", "trace": "T", "direct": "no",
        # 3+4 = Routine + SPECI (Fix 12.07., analog weather_source_compare/ladder_logger)
        "report_type": [3, 4],
    }, timeout=120)
    r.raise_for_status()
    act = {}
    for line in r.text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 3 or parts[2] == "M":
            continue
        try:
            t = float(parts[2])
        except ValueError:
            continue
        d = parts[1][:10]
        lo, hi = act.get(d, (t, t))
        act[d] = (min(lo, t), max(hi, t))
    return ens_min, ens_max, act


def offsets_for(ens, act, which):
    pairs = [(ens[d], act[d][0] if which == "min" else act[d][1])
             for d in sorted(set(ens) & set(act))]
    n = len(pairs)
    if n < 50:
        return None
    diffs = [fc - a for fc, a in pairs]
    s = sum(diffs)
    mean_all = s / n
    sigma = (sum((x - mean_all) ** 2 for x in diffs) / (n - 1)) ** 0.5
    offs, model_p = [], {}
    for i, (fc, a) in enumerate(pairs):
        mu = fc - (s - diffs[i]) / (n - 1)
        k0 = half_up(mu)
        offs.append(half_up(a) - k0)
        for jj in range(-3, 4):
            p = ncdf((k0 + jj + 0.5 - mu) / sigma) - ncdf((k0 + jj - 0.5 - mu) / sigma)
            model_p[jj] = model_p.get(jj, 0.0) + p
    return offs, sigma, {k: v / n for k, v in model_p.items()}, n


def dist_table(offs):
    n = len(offs)
    c = Counter(offs)
    out = {jj: c.get(jj, 0) / n for jj in range(-3, 4)}
    out["<=-4"] = sum(v for k, v in c.items() if k <= -4) / n
    out[">=+4"] = sum(v for k, v in c.items() if k >= 4) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", choices=["min", "max"], default="min")
    ap.add_argument("--days", type=int, default=700)
    ap.add_argument("--city", default=None, help="Kommaliste; Default je --var")
    ap.add_argument("--screen-file", default=None,
                    help="Output-Datei von weather_outlier_screen.py -> Markt-Querschnitt je Klasse")
    args = ap.parse_args()
    cities = (args.city or (DEFAULT_MIN if args.var == "min" else DEFAULT_MAX)).split(",")
    cities = [c.strip() for c in cities if c.strip() in STATIONS]

    pooled, pooled_model, pooled_n = [], {}, 0
    print(f"Empirische Offset-Verteilung — Tages{'tief' if args.var == 'min' else 'hoch'} "
          f"({args.days}d, LOO-Bias, ENS 5 Modelle)")
    print("Klasse:      " + "".join(f"{jj:>7}" for jj in range(-3, 4)) + "   <=-4   >=+4      n  sigma")
    data = {}
    for city in cities:
        icao = STATIONS[city]
        data[city] = fetch_city_both(icao, args.days)
        time.sleep(1)
        ens_min, ens_max, act = data[city]
        res = offsets_for(ens_min if args.var == "min" else ens_max, act, args.var)
        if not res:
            print(f"{city:12s} zu wenig Daten")
            continue
        offs, sigma, model_p, n = res
        d = dist_table(offs)
        print(f"{city:12s} " + "".join(f"{d[jj]*100:6.1f}%" for jj in range(-3, 4)) +
              f" {d['<=-4']*100:5.1f}% {d['>=+4']*100:5.1f}% {n:6d}  {sigma:.2f}")
        pooled += offs
        pooled_n += n
        for k, v in model_p.items():
            pooled_model[k] = pooled_model.get(k, 0.0) + v * n
    if not pooled:
        sys.exit("keine Daten")
    d = dist_table(pooled)
    print(f"{'GEPOOLT':12s} " + "".join(f"{d[jj]*100:6.1f}%" for jj in range(-3, 4)) +
          f" {d['<=-4']*100:5.1f}% {d['>=+4']*100:5.1f}% {len(pooled):6d}")
    print(f"{'Normal-Mod.':12s} " + "".join(f"{pooled_model.get(jj,0)/pooled_n*100:6.1f}%" for jj in range(-3, 4)) +
          "   (was die Normal/Sigma-Annahme je Klasse behauptet)")

    if args.screen_file:
        txt = open(args.screen_file, encoding="utf-8").read()
        city_re = re.compile(r"^--- (.+?) \(\w{4}\)  korr\. Forecasts:.*\| ENS ([\d.]+)±", re.M)
        win_re = re.compile(r"^\s+(-?\d+)°C  YES ([\d.]+)  NO ([\d.]+)  P_ens", re.M)
        blocks = list(city_re.finditer(txt))
        per_class, fav_match, fav_tot = {}, 0, 0
        for i, m in enumerate(blocks):
            mu = float(m.group(2))
            seg = txt[m.end(): blocks[i + 1].start() if i + 1 < len(blocks) else len(txt)]
            wins = [(int(w.group(1)), float(w.group(2)), float(w.group(3))) for w in win_re.finditer(seg)]
            if not wins:
                continue
            k0 = half_up(mu)
            fav_tot += 1
            fav_match += (max(wins, key=lambda x: x[1])[0] == k0)
            for k, yes, no in wins:
                off = k - k0
                if -3 <= off <= 3 and 0 < no <= 1:
                    per_class.setdefault(off, []).append((yes, no))
        print(f"\nMarkt-Querschnitt je Klasse ({fav_tot} Staedte; Markt-Fav == Modell-Fav: {fav_match}/{fav_tot})")
        print("Offset |  n | Ø YES | Ø NO-Ask | emp.P(Kl.) | Kosten NO+Fee | Lay-ROI netto")
        for off in sorted(per_class):
            rows = per_class[off]
            n = len(rows)
            ay = sum(r[0] for r in rows) / n
            ano = sum(r[1] for r in rows) / n
            p = d[off]
            cost = ano + FEE * min(ano, 1 - ano)
            roi = (1 - p) / cost - 1
            print(f"  {off:+d}   | {n:2d} | {ay:5.3f} |  {ano:5.3f}   |   {p*100:5.1f} %  |     {cost:5.3f}     |    {roi*100:+6.2f} %")


if __name__ == "__main__":
    main()
