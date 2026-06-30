#!/usr/bin/env python3
"""
poll_football_odds.py — Minütlicher Quoten-Logger für Fußball-Match-Märkte (Polymarket).

Idee (mit Claude besprochen, 2026-06-23): Für Spiele mit einem klaren FAVORITEN und
einem AUSSENSEITER minütlich die Live-Quoten (Heim / Unentschieden / Auswärts) mitschneiden
und persistent speichern — zur SPÄTEREN Analyse, insbesondere der Spiele, die 0:0 endeten
(Lay-the-Draw-Kontext: wie driftet die Draw-Wahrscheinlichkeit, während es 0:0 bleibt?).

WICHTIG: "endete 0:0" lässt sich live nicht vorfiltern — das weiß man erst nach Abpfiff.
Darum sammelt dieser Collector ALLE Match-Quoten im Live-Fenster und taggt den Favoriten;
die 0:0-Filterung passiert nachgelagert in der Analyse (Endstand wird später nachgetragen).

Quelle:
  - Discovery:  https://gamma-api.polymarket.com/events?tag_slug=soccer  (Match-Events: 3 Märkte
                Heim/Draw/Auswärts, je Yes/No; mit gameStartTime + clobTokenIds)
  - Live-Preis: https://clob.polymarket.com/midpoint?token_id=...        (Mid der Yes-Seite = p)

Speicher: Centron SQL Server (dbdata), Tabellen bb_FootballOdds_1min + bb_FootballMatches.

Aufruf:
  python poll_football_odds.py                 # Live (schreibt in die DB)
  python poll_football_odds.py --dry           # Dry-Run (loggt, schreibt NICHT)
  python poll_football_odds.py --once          # genau ein Snapshot-Durchlauf, dann Ende
  python poll_football_odds.py --pre-min 15 --post-min 135 --interval 60
"""

import argparse
import json
import logging
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

# Hardcodierte Centron-Creds — bewusste Projektentscheidung (siehe CLAUDE.md), Muster aus app.py.
DB_CONFIG = {
    "server": "158.181.48.77",
    "database": "dbdata",
    "user": "326773",
    "password": "Extaler11!",
}

GAMMA = "https://gamma-api.polymarket.com/events"
CLOB = "https://clob.polymarket.com"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("football_odds")


# ---------------------------------------------------------------- DB

def get_conn():
    if pymssql is None:
        raise RuntimeError("pymssql nicht installiert (pip install pymssql)")
    return pymssql.connect(**DB_CONFIG, autocommit=True)


DDL = [
    """
    IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name='bb_FootballMatches')
    CREATE TABLE bb_FootballMatches (
        event_id        NVARCHAR(64)  NOT NULL PRIMARY KEY,
        slug            NVARCHAR(256) NULL,
        match_title     NVARCHAR(256) NULL,
        team1           NVARCHAR(128) NULL,
        team2           NVARCHAR(128) NULL,
        game_start_utc  DATETIME      NULL,
        fav_team        NVARCHAR(128) NULL,
        fav_prob_pre    FLOAT         NULL,
        dog_prob_pre    FLOAT         NULL,
        final_team1     INT           NULL,
        final_team2     INT           NULL,
        ended_0_0       BIT           NULL,
        result_filled   BIT           NOT NULL DEFAULT 0,
        created_utc     DATETIME      NOT NULL DEFAULT GETUTCDATE(),
        updated_utc     DATETIME      NOT NULL DEFAULT GETUTCDATE()
    )
    """,
    """
    IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name='bb_FootballOdds_1min')
    CREATE TABLE bb_FootballOdds_1min (
        id              BIGINT IDENTITY(1,1) PRIMARY KEY,
        ts_utc          DATETIME      NOT NULL,
        event_id        NVARCHAR(64)  NOT NULL,
        match_title     NVARCHAR(256) NULL,
        game_start_utc  DATETIME      NULL,
        minute_rel      INT           NULL,   -- Minuten seit Anpfiff (negativ = vor Anpfiff)
        p_team1         FLOAT         NULL,
        p_draw          FLOAT         NULL,
        p_team2         FLOAT         NULL,
        favorite        NVARCHAR(16)  NULL,   -- 'team1' | 'team2' | 'even'
        fav_prob        FLOAT         NULL,
        dog_prob        FLOAT         NULL,
        source          NVARCHAR(32)  NOT NULL DEFAULT 'polymarket_clob'
    )
    """,
    """
    IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name='IX_bb_FootballOdds_event_ts')
    CREATE INDEX IX_bb_FootballOdds_event_ts ON bb_FootballOdds_1min (event_id, ts_utc)
    """,
]


def ensure_tables(conn):
    cur = conn.cursor()
    for stmt in DDL:
        cur.execute(stmt)
    log.info("Tabellen bb_FootballMatches / bb_FootballOdds_1min sichergestellt.")


# ---------------------------------------------------------------- Polymarket

def _parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace(" ", "T").replace("+00", "+00:00").replace("Z", "+00:00"))
    except Exception:
        return None


def fetch_match_events():
    """Holt aktive Soccer-Match-Events (3-Wege: Team1 / Draw / Team2) von der Gamma-API.
    Liefert Liste dicts mit Token-IDs für die Yes-Seite jeder Outcome."""
    r = requests.get(GAMMA, params={"closed": "false", "limit": 500, "tag_slug": "soccer"}, timeout=30)
    r.raise_for_status()
    out = []
    for e in r.json():
        markets = e.get("markets") or []
        draw_tok = team_markets = None
        teams = []
        for m in markets:
            gi = (m.get("groupItemTitle") or "").strip()
            if not gi:
                continue
            try:
                toks = json.loads(m.get("clobTokenIds") or "[]")
            except Exception:
                toks = []
            if not toks:
                continue
            yes_tok = toks[0]  # outcomes = ["Yes","No"] -> Yes-Token = Wahrscheinlichkeit
            if "draw" in gi.lower():
                draw_tok = yes_tok
            else:
                teams.append((gi, yes_tok))
        # Nur echte 3-Wege-Match-Events (genau 2 Team-Märkte + 1 Draw)
        if draw_tok is None or len(teams) != 2:
            continue
        gst = None
        for m in markets:
            gst = _parse_dt(m.get("gameStartTime"))
            if gst:
                break
        title = e.get("title", "")
        out.append({
            "event_id": str(e.get("id")),
            "slug": e.get("slug"),
            "title": title,
            "team1": teams[0][0], "tok1": teams[0][1],
            "team2": teams[1][0], "tok2": teams[1][1],
            "draw_tok": draw_tok,
            "game_start_utc": gst,
        })
    return out


def clob_midpoint(token_id):
    """Live-Mittelkurs der Yes-Seite (= Wahrscheinlichkeit) oder None bei fehlender Liquidität."""
    try:
        r = requests.get(f"{CLOB}/midpoint", params={"token_id": token_id}, timeout=10)
        if r.status_code != 200:
            return None
        mid = r.json().get("mid")
        return float(mid) if mid is not None else None
    except Exception:
        return None


# ---------------------------------------------------------------- Snapshot

def in_window(gst, now, pre_min, post_min):
    """Liegt der Anpfiff so, dass das Spiel jetzt 'live-fensterrelevant' ist?"""
    if gst is None:
        return False
    delta_min = (now - gst).total_seconds() / 60.0  # Minuten seit Anpfiff
    return -pre_min <= delta_min <= post_min


def upsert_match(conn, ev, fav, fav_p, dog_p):
    cur = conn.cursor()
    cur.execute(
        """
        IF EXISTS (SELECT 1 FROM bb_FootballMatches WHERE event_id=%s)
            UPDATE bb_FootballMatches SET updated_utc=GETUTCDATE() WHERE event_id=%s
        ELSE
            INSERT INTO bb_FootballMatches
              (event_id, slug, match_title, team1, team2, game_start_utc, fav_team, fav_prob_pre, dog_prob_pre)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (ev["event_id"], ev["event_id"],
         ev["event_id"], ev["slug"], ev["title"], ev["team1"], ev["team2"],
         ev["game_start_utc"],
         ev["team1"] if fav == "team1" else (ev["team2"] if fav == "team2" else None),
         fav_p, dog_p),
    )


def snapshot(conn, args):
    """Ein Durchlauf: alle Live-Fenster-Matches abfragen und eine Zeile je Match schreiben."""
    now = datetime.now(timezone.utc)
    try:
        events = fetch_match_events()
    except Exception as e:
        log.warning(f"Gamma-Discovery fehlgeschlagen: {e}")
        return 0

    live = [ev for ev in events if in_window(ev["game_start_utc"], now, args.pre_min, args.post_min)]
    if not live:
        return 0

    written = 0
    for ev in live:
        p1 = clob_midpoint(ev["tok1"])
        pd = clob_midpoint(ev["draw_tok"])
        p2 = clob_midpoint(ev["tok2"])
        if p1 is None or p2 is None:
            log.info(f"  überspringe {ev['title']}: keine Live-Preise (illiquide).")
            continue

        # Favorit korrekt für den 3-Wege-Markt: das Remis wird HERAUSGERECHNET, die
        # <fav_max_odds-Schwelle gilt auf den ZWEI-WEGE-Preis (Favorit vs. Außenseiter
        # im Direktduell). 1.6 -> Favorit muss das Duell zu >62,5% gewinnen. Rohwerte
        # p1/p2/p_draw bleiben gespeichert -> Schwelle jederzeit nachträglich änderbar.
        fav_threshold = 1.0 / args.fav_max_odds
        hi, lo = max(p1, p2), min(p1, p2)
        two_way_fav = hi / (hi + lo) if (hi + lo) > 0 else 0.0
        if two_way_fav >= fav_threshold:
            fav = "team1" if p1 >= p2 else "team2"
        else:
            fav = "even"
        fav_p, dog_p = hi, lo

        minute_rel = int((now - ev["game_start_utc"]).total_seconds() // 60) if ev["game_start_utc"] else None
        log.info(f"  {ev['title']}  t={minute_rel:+}min  "
                 f"p1={p1:.3f} draw={pd if pd is None else round(pd,3)} p2={p2:.3f}  fav={fav}({fav_p:.3f})")

        if args.dry:
            written += 1
            continue

        cur = conn.cursor()
        cur.execute(
            """INSERT INTO bb_FootballOdds_1min
               (ts_utc, event_id, match_title, game_start_utc, minute_rel,
                p_team1, p_draw, p_team2, favorite, fav_prob, dog_prob)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (now, ev["event_id"], ev["title"], ev["game_start_utc"], minute_rel,
             p1, pd, p2, fav, fav_p, dog_p),
        )
        upsert_match(conn, ev, fav, fav_p, dog_p)
        written += 1

    return written


def run(args):
    conn = None
    if not args.dry:
        conn = get_conn()
        ensure_tables(conn)

    log.info("=" * 64)
    log.info(f"FOOTBALL-ODDS-COLLECTOR  |  {'DRY-RUN' if args.dry else 'LIVE (schreibt in DB)'}")
    log.info(f"Fenster: {args.pre_min}min vor … {args.post_min}min nach Anpfiff  |  "
             f"Intervall {args.interval}s  |  min-gap {args.min_gap}")
    log.info("=" * 64)

    while True:
        t0 = time.time()
        try:
            n = snapshot(conn, args)
            if n:
                log.info(f"Snapshot: {n} Match(es) erfasst.")
        except Exception as e:
            log.warning(f"Snapshot-Fehler: {e}")
            if conn is None and not args.dry:
                pass
            else:
                # DB-Verbindung evtl. tot -> neu aufbauen
                try:
                    conn = get_conn()
                except Exception as e2:
                    log.error(f"DB-Reconnect fehlgeschlagen: {e2}")

        if args.once:
            break
        time.sleep(max(1, args.interval - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser(description="Minütlicher Fußball-Quoten-Logger (Polymarket -> Centron)")
    ap.add_argument("--interval", type=int, default=60, help="Poll-Intervall in Sekunden (default 60)")
    ap.add_argument("--pre-min", type=int, default=15, help="Minuten VOR Anpfiff schon loggen (default 15)")
    ap.add_argument("--post-min", type=int, default=135, help="Minuten NACH Anpfiff loggen (default 135 = 90+HZ+Nachspiel)")
    ap.add_argument("--fav-max-odds", type=float, default=1.6,
                    help="Favorit = ZWEI-WEGE-Siegquote (Remis herausgerechnet) < diesem Wert. "
                         "1.6 -> Favorit gewinnt das Direktduell zu >62,5%. Sonst 'even'. "
                         "Alle Spiele werden gespeichert, nur die Favorit-Markierung hängt daran.")
    ap.add_argument("--min-gap", type=float, default=0.0, help="(veraltet, ungenutzt — durch --fav-max-odds ersetzt)")
    ap.add_argument("--dry", action="store_true", help="Dry-Run: loggt, schreibt NICHT in die DB")
    ap.add_argument("--once", action="store_true", help="Nur ein Snapshot-Durchlauf, dann beenden")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
