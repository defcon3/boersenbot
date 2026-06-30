#!/usr/bin/env python3
"""
backfill_football_results.py — Trägt Endstände für gesammelte Fußball-Matches nach.

Ergänzt zu poll_football_odds.py: der Collector speichert die minütlichen Quoten, weiß
aber NICHT, wie ein Spiel ausging. Dieses Skript füllt in bb_FootballMatches die Felder
final_team1, final_team2, ended_0_0, result_filled — die Grundlage, um in der Analyse die
Spiele zu isolieren, die **0:0** endeten (Lay-the-Draw-Kontext).

Quelle = Polymarket (wie der Collector). Polymarket löst zwar nur W/D/U auf (keinen Endstand
direkt), ABER das Begleitevent "<slug>-more-markets" enthält aufgelöste Over/Under-Märkte:
  - Gesamt-Markt  "O/U 0.5"  -> Sieger "Under"  => 0 Tore gesamt => 0:0 (eindeutig).
  - Pro-Team-Leiter "<Team> O/U n.5" -> Anzahl der mit "Over" aufgelösten Linien = Tore des Teams
    => exakter Endstand (robust für niedrige Ergebnisse; bei Torfestivals ggf. gedeckelt,
       für die 0:0-Frage aber irrelevant).
Fallback ohne More-Markets: Haupt-3-Wege-Resolution -> kein Draw ⇒ sicher NICHT 0:0.

Aufruf:
  python backfill_football_results.py                 # ein Durchlauf (schreibt in DB)
  python backfill_football_results.py --dry           # nur zeigen, nicht schreiben
  python backfill_football_results.py --probe-slug fifwc-prt-uzb-2026-06-23   # Logik testen, ohne DB
  python backfill_football_results.py --loop --interval 3600 --grace-hours 3
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone

import requests

try:
    import pymssql
except ImportError:
    pymssql = None

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
GAMMA = "https://gamma-api.polymarket.com/events"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger("football_backfill")


def get_conn():
    if pymssql is None:
        raise RuntimeError("pymssql nicht installiert")
    return pymssql.connect(**DB_CONFIG, autocommit=True)


# ---------------------------------------------------------------- Polymarket

def fetch_event_by_slug(slug):
    """Holt genau ein Event per Slug (Gamma ?slug=...). None, wenn nicht vorhanden."""
    try:
        r = requests.get(GAMMA, params={"slug": slug}, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        return data[0] if data else None
    except Exception as e:
        log.warning(f"Gamma-Fetch {slug} fehlgeschlagen: {e}")
        return None


def market_winner(m):
    """Aufgelöster Sieger-Outcome eines Marktes (z. B. 'Over'/'Under'/'Yes'/'No') oder None."""
    if (m.get("umaResolutionStatus") or "") != "resolved":
        return None
    try:
        outs = json.loads(m.get("outcomes") or "[]")
        prices = json.loads(m.get("outcomePrices") or "[]")
    except Exception:
        return None
    for o, p in zip(outs, prices):
        try:
            if float(p) > 0.99:
                return o
        except (TypeError, ValueError):
            continue
    return None


def team_goals_from_ladder(mm_markets, team):
    """Tore eines Teams = Anzahl der mit 'Over' aufgelösten '<Team> O/U n.5'-Linien
    (Vollzeit, NICHT '1st Half'). None, wenn keine solche Linie aufgelöst ist."""
    pat = re.compile(rf"^{re.escape(team)} O/U \d+\.5$")
    overs = unders = 0
    for m in mm_markets:
        gi = (m.get("groupItemTitle") or "").strip()
        if "1st Half" in gi or not pat.match(gi):
            continue
        w = market_winner(m)
        if w == "Over":
            overs += 1
        elif w == "Under":
            unders += 1
    if overs == 0 and unders == 0:
        return None
    return overs  # Anzahl der überschrittenen .5-Linien = Tore


def derive_result(slug, team1, team2):
    """Leitet (final_t1, final_t2, ended_0_0, method, resolved) aus Polymarket ab."""
    mm = fetch_event_by_slug(f"{slug}-more-markets")
    mm_markets = (mm or {}).get("markets") or []

    # 1) Bevorzugt: exakter Endstand aus den Pro-Team-O/U-Leitern
    g1 = team_goals_from_ladder(mm_markets, team1)
    g2 = team_goals_from_ladder(mm_markets, team2)
    total_ou05 = next((market_winner(m) for m in mm_markets
                       if (m.get("groupItemTitle") or "").strip() == "O/U 0.5"), None)
    btts = next((market_winner(m) for m in mm_markets
                 if (m.get("groupItemTitle") or "").strip() == "Both Teams to Score"), None)

    if g1 is not None and g2 is not None:
        return g1, g2, (g1 == 0 and g2 == 0), "per_team_ou_ladder", True

    # 2) Gesamt-O/U 0.5 (Under => 0:0) — exakter Score evtl. unbekannt
    if total_ou05 == "Under":
        return 0, 0, True, "total_ou05_under", True
    if total_ou05 == "Over":
        # mind. 1 Tor -> nicht 0:0; exakter Stand unbekannt
        return None, None, False, "total_ou05_over", True

    # 3) BTTS=No + Draw => 0:0 (braucht Haupt-Resolution)
    main = fetch_event_by_slug(slug)
    draw_w = None
    if main:
        for m in (main.get("markets") or []):
            if (m.get("groupItemTitle") or "").strip().lower() == "draw" or \
               (m.get("groupItemTitle") or "").strip().lower().startswith("draw "):
                draw_w = market_winner(m)
                break
    if draw_w is None:
        return None, None, None, "unresolved", False  # noch nicht aufgelöst -> später erneut
    is_draw = (draw_w == "Yes")
    if not is_draw:
        return None, None, False, "main_no_draw", True       # kein Draw => sicher nicht 0:0
    if btts == "No":
        return 0, 0, True, "btts_no_draw", True               # Draw ohne beide-treffen => 0:0
    # Draw, aber Endstand nicht ableitbar (z. B. 1:1)
    return None, None, None, "draw_score_unknown", True


# ---------------------------------------------------------------- DB-Lauf

def fetch_unfilled(conn, grace_hours):
    cur = conn.cursor()
    cur.execute(
        """SELECT event_id, slug, match_title, team1, team2, game_start_utc
           FROM bb_FootballMatches
           WHERE result_filled = 0
             AND game_start_utc IS NOT NULL
             AND game_start_utc < DATEADD(hour, -%d, GETUTCDATE())
           ORDER BY game_start_utc""",
        (int(grace_hours),),
    )
    return cur.fetchall()


def write_result(conn, event_id, g1, g2, ended00):
    cur = conn.cursor()
    cur.execute(
        """UPDATE bb_FootballMatches
           SET final_team1=%s, final_team2=%s, ended_0_0=%s, result_filled=1, updated_utc=GETUTCDATE()
           WHERE event_id=%s""",
        (g1, g2, (None if ended00 is None else (1 if ended00 else 0)), event_id),
    )


def run_once(args, conn):
    rows = fetch_unfilled(conn, args.grace_hours)
    if not rows:
        log.info("Keine offenen Matches zum Nachtragen.")
        return 0
    log.info(f"{len(rows)} Match(es) zu prüfen.")
    filled = 0
    for r in rows:
        slug = r["slug"] or (r["match_title"] or "")
        if not slug:
            continue
        g1, g2, ended00, method, resolved = derive_result(slug, r["team1"], r["team2"])
        score = f"{g1}:{g2}" if (g1 is not None and g2 is not None) else "?:?"
        tag = "0:0 ✅" if ended00 else ("nicht 0:0" if ended00 is False else "0:0 unklar")
        if not resolved:
            log.info(f"  …noch nicht aufgelöst: {r['match_title']} ({method})")
            continue
        log.info(f"  {r['match_title']}: {score}  [{tag}]  via {method}")
        if not args.dry:
            write_result(conn, r["event_id"], g1, g2, ended00)
        filled += 1
    return filled


def run(args):
    if args.probe_slug:
        # Nur Logik testen (ohne DB): braucht team1/team2 -> aus Haupt-Event ziehen
        main = fetch_event_by_slug(args.probe_slug)
        if not main:
            log.error(f"Event {args.probe_slug} nicht gefunden.")
            return
        teams = [ (m.get("groupItemTitle") or "").strip()
                  for m in (main.get("markets") or [])
                  if "draw" not in (m.get("groupItemTitle") or "").lower() and m.get("groupItemTitle") ]
        if len(teams) < 2:
            log.error(f"Keine zwei Teams in {args.probe_slug} gefunden (Markets: {teams}).")
            return
        t1, t2 = teams[0], teams[1]
        g1, g2, ended00, method, resolved = derive_result(args.probe_slug, t1, t2)
        log.info(f"PROBE {args.probe_slug}: {t1} {g1} : {g2} {t2}  | ended_0_0={ended00} "
                 f"| method={method} | resolved={resolved}")
        return

    conn = get_conn()
    while True:
        try:
            n = run_once(args, conn)
            log.info(f"Durchlauf fertig: {n} Match(es) {'(dry) ' if args.dry else ''}nachgetragen.")
        except Exception as e:
            log.warning(f"Durchlauf-Fehler: {e}")
            try:
                conn = get_conn()
            except Exception:
                pass
        if not args.loop:
            break
        time.sleep(args.interval)


def main():
    ap = argparse.ArgumentParser(description="Endstand-Backfill für gesammelte Fußball-Matches (Polymarket)")
    ap.add_argument("--dry", action="store_true", help="Nur zeigen, nicht in DB schreiben")
    ap.add_argument("--loop", action="store_true", help="Dauerlauf (sonst ein Durchlauf)")
    ap.add_argument("--interval", type=int, default=3600, help="Sekunden zwischen Durchläufen im --loop (default 3600)")
    ap.add_argument("--grace-hours", type=float, default=3.0,
                    help="Erst Spiele nachtragen, deren Anpfiff > grace-hours her ist (default 3)")
    ap.add_argument("--probe-slug", help="Debug: einen Slug auswerten, ohne DB (testet die Ableitungslogik)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
