#!/usr/bin/env python3
"""
zero_zero_empirical.py — Empirischer Markt-Pfad vs. Poisson-Baseline in 0:0-Spielen.

Holt für vergangene Favorit/Außenseiter-Spiele, die 0:0 endeten, den ECHTEN minütlichen
Polymarket-Pfad des Favoriten (P(Favorit gewinnt), /prices-history) und legt ihn gegen die
analytische Baseline (zero_zero_baseline). Aggregiert über alle Spiele und zeigt, WO der
Markt systematisch steiler/flacher rutscht als das effiziente-Markt-Modell.

Datenfluss:
  1) Gamma closed&soccer paginieren -> 3-Wege-Match-Events (Team1/Draw/Team2).
  2) Vorfilter: Draw-Markt resolved=Yes (nur Remis können 0:0 sein) -> spart Calls.
  3) Begleitevent <slug>-more-markets: Gesamt-"O/U 0.5"=Under  => echtes 0:0.
  4) prices-history je Token: Anpfiff-Preise (Favorit bestimmen + λ lösen) + Favoriten-Pfad.
  5) Pfad auf 0..90-min-Raster (step-hold), pro Spiel emp. vs Baseline, dann Mittel über Spiele.

Aufruf:
  python zero_zero_empirical.py                      # Default-Caps
  python zero_zero_empirical.py --max-pages 6 --max-matches 80 --min-fav 0.50
"""

import argparse
import json
import sys
import time
import datetime as dt

import numpy as np
import requests

from zero_zero_baseline import hda, solve_lambdas, cond_probs
from backfill_football_results import fetch_event_by_slug, market_winner, derive_result

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

GAMMA = "https://gamma-api.polymarket.com/events"
CLOB = "https://clob.polymarket.com"
GRID = list(range(0, 91, 5))      # ECHTE Spielminuten 0..90 (nicht Wall-Clock)
HALFTIME_BREAK = 15               # min Pause -> 2. HZ ist in Wall-Clock um 15 min verschoben


def wall_offset_sec(match_min):
    """Echte Spielminute -> Wall-Clock-Sekunden seit Anpfiff (Halbzeitpause eingerechnet)."""
    w = match_min if match_min <= 45 else match_min + HALFTIME_BREAK
    return w * 60


def _gst_epoch(event):
    for m in event.get("markets") or []:
        g = m.get("gameStartTime")
        if g:
            return dt.datetime.fromisoformat(g.replace(" ", "T").replace("+00", "+00:00")).timestamp()
    return None


def list_match_events(max_pages):
    """Closed Soccer-Events paginieren -> 3-Wege-Match-Events (mit Team-/Draw-Märkten)."""
    out = []
    for page in range(max_pages):
        try:
            r = requests.get(GAMMA, params={"closed": "true", "limit": 500, "offset": page * 500,
                                            "tag_slug": "soccer", "order": "endDate", "ascending": "false"}, timeout=30)
            ev = r.json()
        except Exception as e:
            print(f"  Gamma-Seite {page} Fehler: {e}")
            break
        if not isinstance(ev, list) or not ev:
            break
        for e in ev:
            if not isinstance(e, dict):
                continue
            mks = e.get("markets") or []
            teams = [m for m in mks if m.get("groupItemTitle") and "draw" not in (m.get("groupItemTitle") or "").lower()]
            draws = [m for m in mks if "draw" in (m.get("groupItemTitle") or "").lower()]
            if len(teams) == 2 and len(draws) == 1 and e.get("slug"):
                out.append(e)
        time.sleep(0.15)
    return out


def total_ou05_is_under(slug):
    """0:0-Test: Gesamt-'O/U 0.5' im Begleitevent resolved 'Under'."""
    mm = fetch_event_by_slug(f"{slug}-more-markets")
    if not mm:
        return None
    for m in mm.get("markets") or []:
        if (m.get("groupItemTitle") or "").strip() == "O/U 0.5":
            w = market_winner(m)
            if w is None:
                return None
            return w == "Under"
    return None


def prices_history(token):
    """[(epoch, price), ...] eines Tokens, minütliche Auflösung."""
    try:
        h = requests.get(f"{CLOB}/prices-history",
                         params={"market": token, "interval": "max", "fidelity": 1}, timeout=25).json().get("history", [])
        return [(p["t"], float(p["p"])) for p in h]
    except Exception:
        return []


def price_at(hist, target_epoch):
    """Step-hold: letzter Preis mit t <= target; sonst frühester."""
    prev = None
    for t, p in hist:
        if t <= target_epoch:
            prev = p
        else:
            break
    if prev is not None:
        return prev
    return hist[0][1] if hist else None


def grid_path(hist, t0):
    """Favoriten-Preis auf dem ECHTEN-Spielminuten-Raster (Halbzeitpause korrigiert)."""
    return [price_at(hist, t0 + wall_offset_sec(m)) for m in GRID]


def analyse(args):
    print(f"Scanne bis {args.max_pages*500} closed Soccer-Events …")
    events = list_match_events(args.max_pages)
    print(f"  {len(events)} 3-Wege-Match-Events gefunden.")

    # Vorfilter: nur Remis-Endergebnisse (Draw-Markt resolved Yes)
    draws = []
    for e in events:
        dmk = next((m for m in e.get("markets") or [] if "draw" in (m.get("groupItemTitle") or "").lower()), None)
        if dmk and market_winner(dmk) == "Yes":
            draws.append(e)
    print(f"  davon {len(draws)} mit Endstand REMIS — prüfe auf echtes 0:0 …")

    matches = []  # (title, kickoff_fav_prob, emp_path[grid], base_path[grid])
    for e in draws:
        if len(matches) >= args.max_matches:
            break
        slug = e["slug"]
        mks = e["markets"]
        teams = [m for m in mks if m.get("groupItemTitle") and "draw" not in (m.get("groupItemTitle") or "").lower()]
        dmk = next(m for m in mks if "draw" in (m.get("groupItemTitle") or "").lower())

        # 0:0-Erkennung (breit): Pro-Team-O/U-Leitern / Gesamt-O/U 0.5 / BTTS+Draw
        _, _, ended00, _, _ = derive_result(slug, teams[0]["groupItemTitle"], teams[1]["groupItemTitle"])
        if ended00 is not True:
            continue
        t0 = _gst_epoch(e)
        if t0 is None:
            continue
        try:
            tok1 = json.loads(teams[0]["clobTokenIds"])[0]
            tok2 = json.loads(teams[1]["clobTokenIds"])[0]
            tokd = json.loads(dmk["clobTokenIds"])[0]
        except Exception:
            continue

        h1, h2, hd = prices_history(tok1), prices_history(tok2), prices_history(tokd)
        time.sleep(0.1)
        if not h1 or not h2 or not hd:
            continue

        # Anpfiff-Wahrscheinlichkeiten (Preis nahe t0)
        k1, k2, kd = price_at(h1, t0), price_at(h2, t0), price_at(hd, t0)
        if None in (k1, k2, kd):
            continue
        s = k1 + k2 + kd
        if s <= 0:
            continue
        k1, k2, kd = k1 / s, k2 / s, kd / s

        fav_is_1 = k1 >= k2
        p_fav0 = max(k1, k2)  # rohe 3-Wege-Siegwahrsch. (für Anzeige/Pfad)
        two_way0 = p_fav0 / (k1 + k2) if (k1 + k2) > 0 else 0.0  # Remis herausgerechnet
        if two_way0 < args.min_fav:
            continue  # kein klarer Favorit (Zwei-Wege-Definition)

        # In-Match-Coverage prüfen (genug Punkte im Spielfenster?)
        favhist = h1 if fav_is_1 else h2
        in_win = [1 for t, _ in favhist if t0 <= t <= t0 + 115 * 60]
        if len(in_win) < args.min_points:
            continue

        # Empirischer Favoriten-Pfad
        emp = grid_path(favhist, t0)

        # Baseline für dieses Spiel (λ aus Anpfiff)
        lh, la = solve_lambdas(k1, kd)  # λ aus (P_team1=k1, P_draw=kd); Team2 ergibt sich
        base = []
        for m in GRID:
            H, D, A = cond_probs(lh, la, m)
            base.append(H if fav_is_1 else A)

        matches.append((e.get("title", slug), p_fav0, np.array(emp, float), np.array(base, float)))
        print(f"    0:0 ✔ {e.get('title','')[:40]:40s} Fav-Anpfiff {p_fav0:.2f}  Pkt {len(in_win)}")

    n = len(matches)
    if n == 0:
        print("\nKeine auswertbaren 0:0-Spiele gefunden. Mehr Seiten (--max-pages) oder min-fav senken.")
        return

    # Aggregation: Mittel über Spiele je Rasterpunkt (auf eigenen Anpfiff zentriert)
    EMP = np.vstack([m[2] for m in matches])
    BASE = np.vstack([m[3] for m in matches])
    emp_mean = np.nanmean(EMP, axis=0)
    base_mean = np.nanmean(BASE, axis=0)
    # Green-up = Anpfiff-Preis − Preis(t)  (gemittelt; pro Spiel auf eigenen Start bezogen)
    emp_gu = np.nanmean(EMP[:, 0:1] - EMP, axis=0)
    base_gu = np.nanmean(BASE[:, 0:1] - BASE, axis=0)
    dev = emp_mean - base_mean  # <0: Markt preist Favorit tiefer als Poisson => steiler gerutscht

    emp_gu_net = emp_gu - args.spread  # Lay-Green-up nach angenommenen Kosten
    emp_gu_net[0] = 0.0

    print("\n" + "=" * 80)
    print(f"AGGREGAT über {n} echte 0:0-Favorit-Spiele  —  Markt vs. Poisson-Baseline")
    print("Spieluhr-korrigiert (Halbzeitpause), Achse = ECHTE Spielminute")
    print("=" * 80)
    print(f"{'Min':>4} | {'Markt P(Fav)':>12} {'Baseline':>9} {'Δ(M−B)':>8} | "
          f"{'GU Markt':>9} {'GU netto':>9} {'GU Base':>9}")
    print("-" * 80)
    for i, m in enumerate(GRID):
        end_tag = "  ⚠settle" if m >= 85 else ""
        print(f"{m:>4} | {emp_mean[i]:>12.3f} {base_mean[i]:>9.3f} {dev[i]:>+8.3f} | "
              f"{emp_gu[i]:>+9.3f} {emp_gu_net[i]:>+9.3f} {base_gu[i]:>+9.3f}{end_tag}")
    print("-" * 80)

    # Nur In-Play-Phase werten (5..80 min): Settlement-Ende (>=85) ausklammern
    ip = [i for i, m in enumerate(GRID) if 5 <= m <= 80]
    dev_ip = dev[ip]
    iext = ip[int(np.nanargmax(np.abs(dev_ip)))]
    sign = "STEILER (Lay-Edge-Richtung)" if dev[iext] < 0 else "FLACHER als Poisson"
    print(f"In-Play (5–80'): stärkste Abweichung bei Min {GRID[iext]}: Δ={dev[iext]:+.3f} → Markt rutscht {sign}")
    # Bestes Netto-Green-up im In-Play-Fenster (Lay-Favorit zum Anpfiff, Rückkauf bei t)
    gunet_ip = emp_gu_net[ip]
    ibest = ip[int(np.nanargmax(gunet_ip))]
    print(f"Bestes NETTO-Green-up (Lay-Favorit) im Fenster: +{emp_gu_net[ibest]:.3f}/Kontrakt bei Min {GRID[ibest]} "
          f"(brutto +{emp_gu[ibest]:.3f}, Spread −{args.spread:.2f})")
    print("=" * 80)
    print("Lies-Hilfe: Δ<0 = Markt preist Favorit bei noch-0:0 NIEDRIGER als Poisson → realer Rutsch")
    print("schneller → Lay-Edge-Richtung. Δ>0 = Markt hält Favorit länger hoch (kein mechanischer Edge).")
    print(f"⚠ Min ≥85 settlement-kontaminiert (Token fällt erst bei Oracle-Auflösung). "
          f"N={n} = explorativ, KEIN Pre-Reg-Test.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=20, help="Max. Gamma-Seiten à 500 (stoppt früher, wenn erschöpft)")
    ap.add_argument("--max-matches", type=int, default=200, help="Max. 0:0-Spiele auswerten")
    ap.add_argument("--min-fav", type=float, default=0.625,
                    help="Min. ZWEI-WEGE-Siegwahrsch. des Favoriten, Remis herausgerechnet (0.625 = Duell-Quote <1.6)")
    ap.add_argument("--min-points", type=int, default=8, help="Min. prices-history-Punkte im Spielfenster")
    ap.add_argument("--spread", type=float, default=0.02, help="Angenommener Round-Trip-Spread/Kosten, vom Green-up abgezogen (default 0.02)")
    analyse(ap.parse_args())


if __name__ == "__main__":
    main()
