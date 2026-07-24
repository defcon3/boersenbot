#!/usr/bin/env python3
"""
weather_mm_spread_test.py — Falsifikation der Market-Making-These auf den
Wetter-Maerkten (Pre-Reg `preregs/weather_market_making_2026_07_24_FAIL.md`).

FRAGE
  Lohnt es sich, auf den taeglichen Temperatur-Maerkten selbst Liquiditaet zu
  stellen, statt nur zu layen? Ein Market Maker verdient den Spread; er zahlt
  dafuer die Gebuehr auf jeden Fill. Die These lebt also genau dann, wenn gilt

      halber effektiver Spread  >  Gebuehr je Kontrakt

MESSUNG
  Datenquelle ist Polymarkets oeffentlicher Trade-Tape (kein Auth, kein Geld,
  historisch — s. `POLYMARKET_DATA_API.md`). Jupiters Maerkte SIND
  Polymarket-Maerkte, die IDs sind durchgereicht (`POLY-3025209` -> 3025209).

  Der effektive Spread wird aus den Handelsseiten rekonstruiert. Jeder Trade
  wird auf Yes-Aequivalent normalisiert:

      BUY  Yes @ p   -> Taker hebt den Yes-Brief bei p      ("ask")
      SELL No  @ r   -> dasselbe, bei 1-r                   ("ask")
      SELL Yes @ p   -> Taker trifft das Yes-Geld bei p     ("bid")
      BUY  No  @ r   -> dasselbe, bei 1-r                   ("bid")

  Je Zeitfenster (Default 10 min) und Bucket:
      eff. Spread = mean(ask-Preise) - mean(bid-Preise)

  Das misst den Spread GENAU DANN, wenn tatsaechlich gehandelt wird — also
  unter der Bedingung, unter der ein Market Maker ueberhaupt verdient. Bewegt
  sich der Preis innerhalb des Fensters, verzerrt das nach OBEN; die Schaetzung
  ist damit konservativ zugunsten der These.

WARUM NICHT bb_WeatherLatency
  Die dort geloggte Spalte `all_prices` ist fuer Mikrostruktur unbrauchbar: Die
  Bucket-Preise eines Zeitpunkts summieren sich auf Median 1,08 und bis 1,54
  statt auf 1,00 — es sind veraltete Einzelnotierungen aus verschiedenen
  Momenten. Eine Autokorrelations-Auswertung darauf zeigt scheinbare Mean
  Reversion (r=-0,11), die reines Artefakt ist. Nicht wiederholen.

ERGEBNIS 24.07.2026 (Helsinki, 11 Buckets, 18.-24.07., 13.611 Trades)
  eff. Spread Median 1,0 ct. Weitgehend FAIL, ein schmales Restband ueberlebt:
    Preismitte 0,25-0,60  -> Gebuehr frisst den halben Spread (netto -0,6..-1,1 ct)
    Rand 0,03-0,10        -> netto +0,5 ct, aber Fluss 73 % einseitig (Verkaeufe)
                             => man sammelt nur den sterbenden Bucket ein
    Band 0,10-0,25        -> netto +0,27 ct bei zweiseitigem Fluss: ueberlebt,
                             aber duenn. Offen bleibt die Warteschlangen-Position
                             gegen den vorhandenen Quoter (aus dem Tape nicht messbar).
  Details: preregs/weather_market_making_2026_07_24.md

Aufruf:
  python weather_mm_spread_test.py                          # Helsinki, 18.-24.07.
  python weather_mm_spread_test.py --city madrid --days 20-24
  python weather_mm_spread_test.py --no-cache               # Cache umgehen
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"
CACHE = Path(".mm_tape_cache")

# Jupiter-Gebuehr, empirisch aus eigenen Fills vom 24.07.2026 (Helsinki 21C):
#   NO  @0,750  6,25 Kontr. -> 0,05862 $  = 0,94 ct/Kontr. bei min(p,1-p)=0,250
#   NO  @0,730 12,83 Kontr. -> 0,12650 $  = 0,99 ct/Kontr. bei min(p,1-p)=0,270
#   YES @0,2755 15,57 Kontr. -> 0,15536 $ = 1,00 ct/Kontr. bei min(p,1-p)=0,2755
#   NO  @0,900  5,34 Kontr. -> 0,02403 $  = 0,45 ct/Kontr. bei min(p,1-p)=0,100
# => rund 3,6-4,5 % von min(p, 1-p). Konservativ (guenstigste Lesart) 3,6 %.
FEE_RATE = 0.036

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get(url, params, tries=5):
    """GET mit 429-Backoff. Gibt JSON oder None."""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(3 + 2 * i)
                continue
            if r.ok:
                return r.json()
        except requests.RequestException:
            time.sleep(1 + i)
    return None


def event_markets(city, month, day, year):
    """Alle Bucket-Maerkte eines Stadt-Tages -> [(bucket, conditionId), ...]."""
    slug = f"highest-temperature-in-{city}-on-{month}-{day}-{year}"
    ev = get(f"{GAMMA}/events", {"slug": slug})
    if not ev:
        return []
    out = []
    for m in ev[0].get("markets", []):
        cond = m.get("conditionId")
        if cond:
            out.append((m.get("groupItemTitle") or m.get("question"), cond))
    return out


def tape(cond, cap=6000):
    """Kompletter Trade-Tape eines Marktes (paginiert)."""
    tr, off = [], 0
    while off < cap:
        d = get(f"{DATA}/trades", {"market": cond, "limit": 500, "offset": off})
        if not d:
            break
        tr.extend(d)
        off += len(d)
        if len(d) < 500:
            break
        time.sleep(0.2)
    return tr


def normalize(t):
    """Trade -> (seite, yes_aequivalenter_preis, size) oder None."""
    try:
        p, sz = float(t["price"]), float(t["size"])
    except (KeyError, TypeError, ValueError):
        return None
    side = (t.get("side") or "").upper()
    if side not in ("BUY", "SELL"):
        return None
    if (t.get("outcome") or "").lower().startswith("y"):
        yes = p
    else:
        yes = 1.0 - p
        side = "SELL" if side == "BUY" else "BUY"
    return ("ask" if side == "BUY" else "bid"), yes, sz


def collect(city, days, month, year, use_cache=True):
    """{(tag, bucket): [trades]} — mit Plattencache je Stadt-Tag."""
    CACHE.mkdir(exist_ok=True)
    tapes = {}
    for d in days:
        cf = CACHE / f"{city}_{month}_{d}_{year}.json"
        if use_cache and cf.exists():
            tapes.update({tuple(k.split("|", 1)): v
                          for k, v in json.loads(cf.read_text()).items()})
            print(f"  {month} {d}: aus Cache")
            continue
        mk = event_markets(city, month, d, year)
        if not mk:
            print(f"  {month} {d}: kein Event gefunden")
            continue
        day_tapes = {}
        for bucket, cond in mk:
            day_tapes[f"{d}|{bucket}"] = tape(cond)
            time.sleep(0.15)
        cf.write_text(json.dumps(day_tapes))
        tapes.update({tuple(k.split("|", 1)): v for k, v in day_tapes.items()})
        print(f"  {month} {d}: {len(mk)} Buckets, {sum(len(v) for v in day_tapes.values())} Trades")
    return tapes


def analyse(tapes, window):
    """-> (perday, perbucket, alle_spreads)"""
    perday = defaultdict(lambda: {"n": 0, "notional": 0.0, "spreads": [], "ts": []})
    perbucket = defaultdict(lambda: {"n": 0, "notional": 0.0, "spreads": []})
    all_sp = []
    flow = []          # (seite, yes_preis) fuer JEDEN Trade — Basis der Fluss-Balance
    for (day, bucket), tr in tapes.items():
        if not tr:
            continue
        win = defaultdict(lambda: {"ask": [], "bid": []})
        for t in tr:
            nz = normalize(t)
            if not nz:
                continue
            side, yes, sz = nz
            ts = int(t["timestamp"])
            perday[day]["n"] += 1
            perday[day]["notional"] += yes * sz
            perday[day]["ts"].append(ts)
            perbucket[bucket]["n"] += 1
            perbucket[bucket]["notional"] += yes * sz
            flow.append((side, yes))
            win[ts // window][side].append(yes)
        for d in win.values():
            if d["ask"] and d["bid"]:
                sp = statistics.mean(d["ask"]) - statistics.mean(d["bid"])
                lvl = statistics.mean(d["ask"] + d["bid"])   # Preislage des Fensters
                if -0.5 < sp < 1.0:          # grobe Ausreisser-Klammer
                    perday[day]["spreads"].append(sp)
                    perbucket[bucket]["spreads"].append(sp)
                    all_sp.append((sp, lvl))
    return perday, perbucket, all_sp, flow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="helsinki")
    ap.add_argument("--days", default="18-24", help="z. B. 18-24 oder 20,22,24")
    ap.add_argument("--month", default="july")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--window", type=int, default=600, help="Spread-Fenster in Sekunden")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    if "-" in a.days:
        lo, hi = a.days.split("-")
        days = list(range(int(lo), int(hi) + 1))
    else:
        days = [int(x) for x in a.days.split(",")]

    print(f"Lade {a.city.title()}, {a.month.title()} {days[0]}-{days[-1]} {a.year} …")
    tapes = collect(a.city, days, a.month, a.year, use_cache=not a.no_cache)
    if not tapes:
        print("Keine Daten."); return

    perday, perbucket, all_sp, flow = analyse(tapes, a.window)
    total_n = sum(v["n"] for v in perday.values())
    total_no = sum(v["notional"] for v in perday.values())

    print(f"\n=== {a.city.title()}, alle Buckets, je Markttag ===\n")
    print(f"{'Tag':<10}{'Trades':>8}{'Notional':>12}{'Ø Größe':>10}{'eff.Spread':>13}{'Median-Lücke':>14}")
    print("-" * 67)
    for day in sorted(perday, key=int):
        v = perday[day]
        ts = sorted(set(v["ts"]))
        gaps = [b - a_ for a_, b in zip(ts, ts[1:])] or [0]
        sp = statistics.median(v["spreads"]) * 100 if v["spreads"] else float("nan")
        print(f"{a.month[:3].title()} {day:<5}{v['n']:>8}{v['notional']:>11,.0f}$"
              f"{v['notional']/max(v['n'],1):>9.0f}${sp:>11.1f}ct{statistics.median(gaps):>11.0f}s")
    print("-" * 67)
    print(f"{'GESAMT':<10}{total_n:>8}{total_no:>11,.0f}$")

    if not all_sp:
        print("\nZu wenige beidseitige Fenster für eine Spread-Schätzung."); return

    sps = [s for s, _ in all_sp]
    q = statistics.quantiles(sps, n=10)
    med = statistics.median(sps)
    print(f"\n=== effektiver Spread (n={len(all_sp)} Fenster à {a.window//60} min) ===")
    print(f"  Median {med*100:.1f} ct | 25 % {q[1]*100:.1f} | 75 % {q[6]*100:.1f} | 90 % {q[8]*100:.1f} ct")

    print(f"\n=== je Bucket ===")
    print(f"{'Bucket':<18}{'Trades':>8}{'Notional':>12}{'eff.Spread':>13}")
    for b, v in sorted(perbucket.items(), key=lambda x: -x[1]["notional"])[:15]:
        sp = statistics.median(v["spreads"]) * 100 if v["spreads"] else float("nan")
        print(f"{b:<18}{v['n']:>8}{v['notional']:>11,.0f}${sp:>11.1f}ct")

    # ---- Gate: halber Spread gegen Gebuehr, JE PREISLAGE ----
    # Entscheidend: der Spread wird IN der jeweiligen Preisecke gemessen, nicht
    # global unterstellt. Die Gebuehr faellt zum Rand hin (min(p,1-p)) — wenn der
    # Spread dort nicht mindestens genauso schnell faellt, lebt die These am Rand.
    bands = [(0.40, 0.60, "Mitte  0,40–0,60"), (0.25, 0.40, "0,25–0,40"),
             (0.10, 0.25, "0,10–0,25"), (0.03, 0.10, "Rand   0,03–0,10"),
             (0.00, 0.03, "Tail   < 0,03")]
    print(f"\n=== GATE: halber Spread vs. Gebühr ({FEE_RATE*100:.1f} % von min(p,1−p)) ===")
    print(f"{'Preislage':<20}{'n':>6}{'Spread':>10}{'halber':>9}{'Gebühr':>9}{'netto':>9}{'Kauf-Anteil':>13}")
    print("-" * 76)
    verdict_ok = False
    for lo, hi, name in bands:
        band = [s for s, l in all_sp if lo <= min(l, 1 - l) < hi]
        fl = [sd for sd, pz in flow if lo <= min(pz, 1 - pz) < hi]
        if len(band) < 20:
            print(f"  {name:<18}{len(band):>6}   — zu wenige Fenster")
            continue
        m = statistics.median(band)
        bal = fl.count("ask") / len(fl) if fl else float("nan")
        p_mid = (lo + hi) / 2
        fee = FEE_RATE * p_mid
        net = m / 2 - fee
        # Ein positives Netto zaehlt nur bei zweiseitigem Fluss: wer 3 von 4 Fills
        # auf derselben Seite bekommt, stellt keinen Markt, sondern sammelt Bestand
        # in genau dem Bucket ein, den die anderen gerade als tot verkaufen.
        two_sided = 0.40 <= bal <= 0.60
        if net > 0 and two_sided:
            verdict_ok = True
        print(f"  {name:<18}{len(band):>6}{m*100:>8.1f}ct{m/2*100:>7.1f}ct"
              f"{fee*100:>7.2f}ct{net*100:>7.2f}ct{bal*100:>9.0f} %"
              f"{'  ok' if two_sided else '  EINSEITIG'}")
    print("-" * 76)
    if verdict_ok:
        print("  -> Teilbereich mit positivem Netto UND zweiseitigem Fluss:")
        print("     Market Making nicht ausgeschlossen — weiter prüfen.")
    else:
        print("  -> FAIL auf beiden Wegen:")
        print("     In der Preismitte frisst die Gebühr den halben Spread.")
        print("     Am Rand trägt der Spread zwar die Gebühr, aber der Fluss ist")
        print("     einseitig (3 von 4 Trades Verkäufe) — man sammelt dort nur den")
        print("     Bestand ein, den andere als tot abstoßen. Adverse Selection.")


if __name__ == "__main__":
    main()
