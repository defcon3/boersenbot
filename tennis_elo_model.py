#!/usr/bin/env python3
"""
tennis_elo_model.py — Faires Match-Wahrscheinlichkeitsmodell aus Tennis-Abstract-Elo.

Zweck (vgl. JUPITER_TENNIS_BETTING_PLAN.md, Abschnitt 4 Schritt 1):
  Eine FUNDAMENTAL hergeleitete faire Siegwahrscheinlichkeit pro Spieler liefern —
  belagsgewichtetes Elo gegen den Marktpreis halten, NICHT Fremdquoten nachlaufen.

Datenquelle (Stand 2026-06-24):
  Jeff Sackmanns `tennis_atp`/`tennis_wta` GitHub-Repos sind NICHT MEHR ÖFFENTLICH.
  Stattdessen: die Live-Elo-Reports auf tennisabstract.com — pro Spieler bereits
  fertiges Overall- + Hard/Clay/Grass-Elo. Abdeckung: ~531 ATP-Spieler (runter bis
  ATP-Rang ~531). Das DECKT Grand-Slam-Quali und Haupt-Challenger-Felder (Ränge
  ~100–400) ab — genau die Zielmärkte — aber NICHT ITF-Futures/Spieler jenseits ~531.
  -> Für solche Spieler liefert das Modell KEINE Schätzung (ehrlicher None statt Raten).

Modell:
  Belags-Elo = blend * Overall + (1-blend) * Belag-spezifisch  (blend default 0.5).
  P(A schlägt B) = 1 / (1 + 10^((Elo_B - Elo_A)/400))   (logistische Elo-Erwartung).

Aufruf:
  python tennis_elo_model.py "Sinner" "Alcaraz" --surface clay
  python tennis_elo_model.py "Mochizuki" "Onclin" --tour atp --surface grass
  python tennis_elo_model.py --refresh                 # Cache neu ziehen
  python tennis_elo_model.py --list 20                 # Top-20 anzeigen (Sanity)

Importierbar (für den Forward-Paper-Test-Logger):
  from tennis_elo_model import fair_prob, load_elo, resolve
"""

import argparse
import csv
import html as _html
import os
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher

import requests

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ELO_URL = {
    "atp": "https://tennisabstract.com/reports/atp_elo_ratings.html",
    "wta": "https://tennisabstract.com/reports/wta_elo_ratings.html",
}
CACHE_FMT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tennis_elo_{tour}.csv")
SURFACES = ("hard", "clay", "grass")
UA = {"User-Agent": "Mozilla/5.0 (boersenbot tennis_elo_model)"}

# Spalten im Tennis-Abstract-Elo-Report (echte <td>-Reihenfolge, 17 Zellen mit
# drei leeren Spacer-Spalten bei 4/11/14):
#   0 EloRank 1 Player 2 Age 3 Elo  [4 _]  5 hRank 6 hElo 7 cRank 8 cElo
#   9 gRank 10 gElo  [11 _]  12 Peak 13 PeakMonth  [14 _]  15 ATPRank 16 LogDiff
COL = dict(elo_rank=0, age=2, overall=3, hard=6, clay=8, grass=10, peak=12, atp_rank=15)


# ---------------------------------------------------------------- Parsing / Cache

def _clean(cell):
    """HTML-Zelle -> reiner Text (Tags raus, &nbsp;/Entities aufgelöst)."""
    t = re.sub(r"<[^>]+>", "", cell)
    t = _html.unescape(t).replace("\xa0", " ").strip()
    return t


def _slug_of(cell):
    """Spieler-Slug aus dem <a ...p=JannikSinner>-Link (kanonische ID, falls vorhanden)."""
    m = re.search(r"[?&]p=([A-Za-z0-9]+)", cell)
    return m.group(1) if m else ""


def _parse_report(htmltext):
    """Roh-HTML des Reports -> Liste von Spieler-Dicts."""
    body = re.search(r"<tbody>(.*?)</tbody>", htmltext, re.S)
    if not body:
        return []
    rows = []
    for tr in re.findall(r"<tr>(.*?)</tr>", body.group(1), re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        if len(cells) < 16:
            continue
        name = _clean(cells[1])

        def num(idx):
            v = _clean(cells[idx])
            try:
                return float(v)
            except ValueError:
                return None
        rows.append({
            "name": name,
            "slug": _slug_of(cells[1]),
            "elo_rank": num(COL["elo_rank"]),
            "age": num(COL["age"]),
            "overall": num(COL["overall"]),
            "hard": num(COL["hard"]),
            "clay": num(COL["clay"]),
            "grass": num(COL["grass"]),
            "peak": num(COL["peak"]),
            "atp_rank": num(COL["atp_rank"]),
        })
    return rows


def _cache_path(tour):
    return CACHE_FMT.format(tour=tour)


def _write_cache(tour, rows):
    path = _cache_path(tour)
    fields = ["name", "slug", "elo_rank", "age", "overall", "hard", "clay", "grass", "peak", "atp_rank"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def _read_cache(tour):
    path = _cache_path(tour)
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("elo_rank", "age", "overall", "hard", "clay", "grass", "peak", "atp_rank"):
                r[k] = float(r[k]) if r[k] not in ("", None) else None
            rows.append(r)
    return rows


def load_elo(tour="atp", max_age_h=12, force=False, quiet=False):
    """Elo-Tabelle laden (Cache, sonst von tennisabstract.com ziehen).

    Rückgabe: Liste von Spieler-Dicts (s. _parse_report). Cache wird neu gezogen,
    wenn er fehlt, älter als max_age_h ist oder force=True.
    """
    tour = tour.lower()
    if tour not in ELO_URL:
        raise ValueError(f"tour muss in {list(ELO_URL)} sein")
    path = _cache_path(tour)
    fresh = os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_h * 3600
    if fresh and not force:
        cached = _read_cache(tour)
        if cached:
            return cached
    try:
        r = requests.get(ELO_URL[tour], headers=UA, timeout=30)
        r.raise_for_status()
        rows = _parse_report(r.text)
        if not rows:
            raise RuntimeError("Report geparst, aber 0 Zeilen — HTML-Struktur geändert?")
        _write_cache(tour, rows)
        if not quiet:
            print(f"  Elo-Report {tour.upper()} frisch gezogen: {len(rows)} Spieler -> {os.path.basename(path)}")
        return rows
    except Exception as e:
        cached = _read_cache(tour)
        if cached:
            if not quiet:
                print(f"  ⚠ Netzfehler ({e}); nutze veralteten Cache ({len(cached)} Spieler).")
            return cached
        raise


# ---------------------------------------------------------------- Name-Matching

def _norm(s):
    """Akzente weg, lower, nur a-z0-9 — für robusten Namensvergleich."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def resolve(query, rows, min_score=0.0):
    """Freitext-Spielername -> bester Tabellen-Treffer (NACHNAMEN-verankert).

    Markt-Namen sind 'Vorname Nachname'; der Nachname ist das verlässliche Signal.
    Score = 0.7·Nachnamen-Ähnlichkeit + 0.3·max(Vollname, Slug); exakter Nachname → ~0.97.
    Das verhindert Fehl-Matches wie 'Aziz Dougaz' → 'Zizou Bergs' (gleiche Buchstaben,
    anderer Nachname). Gibt (spieler_dict, score, kandidaten) zurück; None < min_score.
    """
    q_full = _norm(query)
    if not q_full:
        return None, 0.0, []
    q_last = _norm(query.split()[-1])
    scored = []
    for r in rows:
        name_n = _norm(r["name"])
        slug_n = _norm(r["slug"])
        c_last = _norm(r["name"].split()[-1]) if r["name"] else ""
        last_sim = SequenceMatcher(None, q_last, c_last).ratio() if q_last and c_last else 0.0
        full_sim = max(SequenceMatcher(None, q_full, name_n).ratio(),
                       SequenceMatcher(None, q_full, slug_n).ratio())
        score = 0.7 * last_sim + 0.3 * full_sim
        if q_last and q_last == c_last:                 # exakter Nachname dominiert
            score = max(score, 0.95 + 0.05 * full_sim)
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]
    if best_score < min_score:
        return None, best_score, [r for _, r in scored[:4]]
    return best, best_score, [r for _, r in scored[1:4]]


# ---------------------------------------------------------------- Modell

def surface_elo(player, surface, blend=0.5):
    """Belagsgewichtetes Elo: blend*Overall + (1-blend)*Belag-Elo.

    Fehlt das Belag-Elo (Spieler auf dem Belag ungewertet) -> reines Overall.
    surface='all'/None -> reines Overall.
    """
    ov = player.get("overall")
    if ov is None:
        return None
    if not surface or surface == "all":
        return ov
    surf = player.get(surface)
    if surf is None:
        return ov
    return blend * ov + (1.0 - blend) * surf


def elo_win_prob(elo_a, elo_b):
    """Logistische Elo-Siegwahrscheinlichkeit von A gegen B."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def fair_prob(name_a, name_b, surface="hard", tour="atp", blend=0.5,
              rows=None, min_score=0.85):
    """Faire P(A schlägt B) auf gegebenem Belag.

    Rückgabe-Dict mit aufgelösten Spielern, Belags-Elo, p_a/p_b und Match-Konfidenz.
    'ok' ist False, wenn ein Spieler nicht (sicher genug) gefunden wurde oder kein
    Elo hat — dann KEINE faire Wahrscheinlichkeit (lieber nichts als Raten).
    """
    surface = (surface or "all").lower()
    if surface not in SURFACES and surface != "all":
        raise ValueError(f"surface muss in {SURFACES} oder 'all' sein")
    if rows is None:
        rows = load_elo(tour, quiet=True)

    pa, sa, alt_a = resolve(name_a, rows, min_score)
    pb, sb, alt_b = resolve(name_b, rows, min_score)

    res = {
        "surface": surface, "tour": tour, "blend": blend,
        "a": {"query": name_a, "match": pa["name"] if pa else None, "score": round(sa, 3),
              "alts": [c["name"] for c in (alt_a or [])]},
        "b": {"query": name_b, "match": pb["name"] if pb else None, "score": round(sb, 3),
              "alts": [c["name"] for c in (alt_b or [])]},
        "ok": False, "p_a": None, "p_b": None, "warn": [],
    }
    if pa is None:
        res["warn"].append(f"'{name_a}' nicht sicher gefunden (bester Score {sa:.2f}) — außerhalb Top-{len(rows)}?")
    if pb is None:
        res["warn"].append(f"'{name_b}' nicht sicher gefunden (bester Score {sb:.2f}) — außerhalb Top-{len(rows)}?")
    if pa is None or pb is None:
        return res

    ea = surface_elo(pa, surface, blend)
    eb = surface_elo(pb, surface, blend)
    res["a"].update(elo=round(ea, 1), overall=pa["overall"], surf_raw=pa.get(surface if surface != "all" else "overall"))
    res["b"].update(elo=round(eb, 1), overall=pb["overall"], surf_raw=pb.get(surface if surface != "all" else "overall"))
    p = elo_win_prob(ea, eb)
    res["p_a"], res["p_b"], res["ok"] = round(p, 4), round(1 - p, 4), True
    if min(sa, sb) < 0.8:
        res["warn"].append(f"Namens-Match unsicher (min Score {min(sa, sb):.2f}) — Kandidaten prüfen.")
    return res


# ---------------------------------------------------------------- CLI

def _print_match(res):
    a, b = res["a"], res["b"]
    print("=" * 66)
    print(f"FAIRES MODELL  ·  {res['tour'].upper()}  ·  Belag: {res['surface']}  ·  blend={res['blend']}")
    print("=" * 66)
    for tag, side in (("A", a), ("B", b)):
        line = f"{tag}: {side['query']!r:24s} -> {side['match'] or '— kein Treffer —'}"
        if side.get("elo") is not None:
            line += f"  Elo {side['elo']} (Overall {side['overall']}, Belag-roh {side['surf_raw']})"
        line += f"   [Match {side['score']:.2f}]"
        print(line)
        if side.get("alts"):
            print(f"     andere Kandidaten: {', '.join(side['alts'])}")
    print("-" * 66)
    if res["ok"]:
        print(f"  Faire Siegwahrscheinlichkeit:  {a['match']}  {res['p_a']:.3f}   |   {b['match']}  {res['p_b']:.3f}")
        print(f"  Faire 'Preise' (= Wahrsch.):   YES_A ≈ {res['p_a']:.2f}   YES_B ≈ {res['p_b']:.2f}")
        print(f"  -> Markt-Limit nur, wenn Preis DEUTLICH unter dem fairen Wert liegt (Schwelle separat).")
    else:
        print("  KEINE faire Wahrscheinlichkeit ableitbar (s. Warnungen).")
    for w in res["warn"]:
        print(f"  ⚠ {w}")
    print("=" * 66)


def main():
    ap = argparse.ArgumentParser(description="Faires Tennis-Match-Modell aus Tennis-Abstract-Elo.")
    ap.add_argument("player_a", nargs="?", help="Spieler A (Frei-Text, z. B. 'Sinner')")
    ap.add_argument("player_b", nargs="?", help="Spieler B")
    ap.add_argument("--surface", default="hard", choices=[*SURFACES, "all"], help="Belag (default hard)")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"], help="Tour (default atp)")
    ap.add_argument("--blend", type=float, default=0.5, help="Gewicht Overall vs. Belag-Elo (0.5=halb/halb)")
    ap.add_argument("--refresh", action="store_true", help="Elo-Cache zwingend neu ziehen und beenden")
    ap.add_argument("--list", type=int, metavar="N", help="Top-N Spieler der Tabelle anzeigen (Sanity-Check)")
    args = ap.parse_args()

    if args.refresh:
        for t in ("atp", "wta"):
            try:
                rows = load_elo(t, force=True)
                print(f"  {t.upper()}: {len(rows)} Spieler im Cache.")
            except Exception as e:
                print(f"  {t.upper()}: Fehler — {e}")
        return

    rows = load_elo(args.tour)
    if args.list:
        print(f"{'#':>3}  {'Spieler':28s} {'Elo':>7} {'Hard':>7} {'Clay':>7} {'Grass':>7}  ATP")
        for r in rows[:args.list]:
            print(f"{int(r['elo_rank'] or 0):>3}  {r['name']:28s} "
                  f"{r['overall'] or 0:>7.1f} {r['hard'] or 0:>7.1f} {r['clay'] or 0:>7.1f} "
                  f"{r['grass'] or 0:>7.1f}  {int(r['atp_rank'] or 0)}")
        return

    if not args.player_a or not args.player_b:
        ap.error("Zwei Spielernamen angeben — oder --refresh / --list N nutzen.")
    res = fair_prob(args.player_a, args.player_b, args.surface, args.tour, args.blend, rows=rows)
    _print_match(res)


if __name__ == "__main__":
    main()
