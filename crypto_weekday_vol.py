#!/usr/bin/env python3
"""
crypto_weekday_vol.py — Wochentags-Effekt in der Krypto-Schwankungsbreite.

BEOBACHTUNG (Nutzer, 2026-07-26): "Die Volatilitaet der Coins ist sonntags
deutlich geringer als sonst" — zwei Sonntage in Folge aufgefallen (19.07., 26.07.).

FRAGE: Ist das ein systematischer Wochentags-Effekt oder waren das zwei ruhige
Tage? Gemessen ueber ALLE Assets, die Jupiter im Up/Down-Handel fuehrt.

MESSGROESSEN (beide, weil "Schwankungsbreite" zweierlei heissen kann):
  RV   — realisierte Vol eines Tages: sqrt(sum(r_i^2)) ueber 5-Min-Log-Returns,
         in bps. Das klassische Vol-Mass.
  RANGE— mittlere (high-low)/open je 5-Min-Kerze in bps. Naeher an dem, was den
         Flip-Markt entscheidet: wie weit laeuft der Preis INNERHALB eines
         kurzen Fensters von der Null-Linie weg.

METHODIK — zwei Fallen, die hier bewusst behandelt werden:
  1. Vol-Regime-Clustering. Vol ist stark autokorreliert; ein ruhiger Monat mit
     vielen Sonntagen wuerde einen Wochentags-Effekt vortaeuschen. Deshalb wird
     jeder Tag durch das Mittel SEINER Kalenderwoche geteilt (rel = Tag/Woche).
     Der Test laeuft auf diesen wochen-relativen Werten — ein Regime hebt sich
     damit weg, weil jede Woche ihren eigenen Massstab bekommt.
  2. Schiefe. RV ist rechtsschief/lognormal -> gerechnet wird auf log(RV),
     Mittelwerte werden als geometrische zurueckverwandelt.

Der Test ist zweiseitig gegen "Sonntag verhaelt sich wie die uebrigen Tage".
Zusaetzlich als Kontrolle: alle sieben Wochentage einzeln, damit sichtbar wird,
ob der Effekt am Sonntag haengt oder am ganzen Wochenende.

ZEITZONE: Tagesgrenzen in UTC. Zusaetzlich wird der Befund gegen ET-Grenzen
geprueft (--tz), weil "Wochenende" ueber die TradFi-Oeffnungszeiten definiert
ist und Jupiter/Polymarket ihre Fenster in ET benennen.

Aufruf:
  python crypto_weekday_vol.py --days 365
  python crypto_weekday_vol.py --days 365 --tz America/New_York
  python crypto_weekday_vol.py --days 720 --assets BTC,ETH
"""

import argparse
import math
import statistics as st
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

BINANCE = "https://api.binance.com/api/v3/klines"

# Assets, die Jupiter im Up/Down-Handel fuehrt (Scan 26.07.2026) -> Binance-Symbol.
# Hyperliquid ist dort neu; falls Binance das Paar nicht fuehrt, faellt es raus.
ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
    "BNB": "BNBUSDT", "DOGE": "DOGEUSDT", "HYPE": "HYPEUSDT",
}
WD = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]


def klines(symbol, start_ms, end_ms, interval="5m"):
    """Paginiert Binance-Klines. Gibt [(open_ms, open, high, low, close)] zurueck."""
    out, cur = [], start_ms
    while cur < end_ms:
        for attempt in range(4):
            try:
                r = requests.get(BINANCE, params={
                    "symbol": symbol, "interval": interval,
                    "startTime": cur, "endTime": end_ms, "limit": 1000}, timeout=30)
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 400:
                return None          # Symbol existiert nicht
            r.raise_for_status()
            break
        else:
            return out
        rows = r.json()
        if not rows:
            break
        for k in rows:
            out.append((int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])))
        nxt = int(rows[-1][0]) + 1
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.12)
    return out


def tagesmasse(rows, tz):
    """-> {datum: (rv_bps, range_bps, n_kerzen)} mit Tagesgrenzen in tz."""
    per_day = defaultdict(list)
    for ts, o, h, l, c in rows:
        d = datetime.fromtimestamp(ts / 1000, tz).date()
        per_day[d].append((o, h, l, c))
    out = {}
    for d, ks in per_day.items():
        if len(ks) < 200:            # angebrochene Tage raus (voll = 288 Kerzen)
            continue
        rets = []
        for i in range(1, len(ks)):
            p0, p1 = ks[i - 1][3], ks[i][3]
            if p0 > 0 and p1 > 0:
                rets.append(math.log(p1 / p0))
        if len(rets) < 100:
            continue
        rv = math.sqrt(sum(r * r for r in rets)) * 1e4
        rng = st.mean((h - l) / o * 1e4 for o, h, l, c in ks if o > 0)
        out[d] = (rv, rng, len(ks))
    return out


def wochen_relativ(tage):
    """Jeden Tag durch das geometrische Mittel SEINER ISO-Woche teilen.
    Neutralisiert Vol-Regime; nur Wochen mit >= 5 Tagen zaehlen."""
    wochen = defaultdict(list)
    for d, v in tage.items():
        wochen[d.isocalendar()[:2]].append((d, v))
    rel = {}
    for _, tagesliste in wochen.items():
        if len(tagesliste) < 5:
            continue
        for idx, name in ((0, "rv"), (1, "range")):
            logs = [math.log(v[idx]) for _, v in tagesliste if v[idx] > 0]
            if not logs:
                continue
            mittel = math.exp(st.mean(logs))
            for d, v in tagesliste:
                if v[idx] > 0:
                    rel.setdefault(d, {})[name] = v[idx] / mittel
    return rel


def welch(a, b):
    """Welch-t fuer zwei Stichproben (ungleiche Varianz)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return float("nan"), float("nan")
    return (ma - mb) / se, ma - mb


def auswerten(name, rel, key):
    """Wochentags-Tabelle + Test Sonntag vs. Rest, auf log(rel)."""
    per_wd = defaultdict(list)
    for d, v in rel.items():
        if key in v:
            per_wd[d.weekday()].append(math.log(v[key]))
    if not per_wd:
        print(f"   {name}: keine Daten")
        return None
    print(f"\n   {name} — wochen-relativ (1,00 = Mittel der eigenen Woche)")
    print(f"      {'Tag':<11} {'n':>4} {'rel':>7} {'Abw.':>8}")
    for wd in range(7):
        v = per_wd.get(wd, [])
        if not v:
            continue
        g = math.exp(st.mean(v))
        print(f"      {WD[wd]:<11} {len(v):4d} {g:7.3f} {(g - 1) * 100:+7.1f} %")
    so = per_wd.get(6, [])
    rest = [x for wd, v in per_wd.items() if wd != 6 for x in v]
    t, diff = welch(so, rest)
    g_so, g_rest = math.exp(st.mean(so)), math.exp(st.mean(rest))
    print(f"      -> Sonntag {g_so:.3f} vs. uebrige Tage {g_rest:.3f} "
          f"= {(g_so / g_rest - 1) * 100:+.1f} %  |  t = {t:.2f}  (n={len(so)} Sonntage)")
    we = [x for wd, v in per_wd.items() if wd in (5, 6) for x in v]
    wo = [x for wd, v in per_wd.items() if wd < 5 for x in v]
    t2, _ = welch(we, wo)
    print(f"      -> Wochenende {math.exp(st.mean(we)):.3f} vs. Wochentage "
          f"{math.exp(st.mean(wo)):.3f} = "
          f"{(math.exp(st.mean(we)) / math.exp(st.mean(wo)) - 1) * 100:+.1f} %  |  t = {t2:.2f}")
    return {"t_sonntag": t, "rel_sonntag": g_so / g_rest, "n_sonntage": len(so)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365, help="Historie in Tagen")
    ap.add_argument("--assets", default=None, help="Komma-Liste, sonst alle")
    ap.add_argument("--tz", default="UTC", help="Zeitzone der Tagesgrenzen")
    ap.add_argument("--interval", default="5m", help="Binance-Kerzen")
    args = ap.parse_args()

    tz = ZoneInfo(args.tz)
    wanted = ([a.strip().upper() for a in args.assets.split(",")]
              if args.assets else list(ASSETS))
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    print(f"Wochentags-Effekt in der Krypto-Schwankungsbreite")
    print(f"Zeitraum {start.date()} bis {end.date()} | Kerzen {args.interval} "
          f"| Tagesgrenzen {args.tz}")

    alle_rel = defaultdict(dict)
    ergebnisse = {}
    for a in wanted:
        sym = ASSETS.get(a)
        if not sym:
            print(f"\n{a}: unbekannt, uebersprungen")
            continue
        rows = klines(sym, int(start.timestamp() * 1000), int(end.timestamp() * 1000),
                      args.interval)
        if not rows:
            print(f"\n{a} ({sym}): keine Binance-Daten — uebersprungen")
            continue
        tage = tagesmasse(rows, tz)
        rel = wochen_relativ(tage)
        print(f"\n{'=' * 72}\n{a} ({sym}) — {len(rows):,} Kerzen, {len(tage)} volle Tage, "
              f"{len(rel)} in vollen Wochen")
        ergebnisse[a] = {k: auswerten(lbl, rel, k)
                         for k, lbl in (("rv", "RV (realisierte Vol)"),
                                        ("range", "RANGE (5-Min-Spanne)"))}
        for d, v in rel.items():
            for k, val in v.items():
                alle_rel[d].setdefault(k, []).append(val)

    # Pool ueber alle Assets: je Tag das Mittel der Assets (Krypto-Beta -> die
    # Assets sind KEINE unabhaengigen Stichproben, deshalb gepoolt und nicht addiert)
    if len(ergebnisse) > 1:
        pool = {d: {k: st.mean(v) for k, v in kv.items()} for d, kv in alle_rel.items()}
        print(f"\n{'=' * 72}\nGEPOOLT ueber {len(ergebnisse)} Assets "
              f"(je Tag das Asset-Mittel — die Coins laufen synchron, "
              f"zaehlen also nicht als unabhaengige Stichproben)")
        for k, lbl in (("rv", "RV (realisierte Vol)"), ("range", "RANGE (5-Min-Spanne)")):
            auswerten(lbl, pool, k)

    print(f"\n{'=' * 72}\nLesehilfe: rel < 1 heisst ruhiger als die eigene Woche. "
          f"|t| > 2 = auf 5 % signifikant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
