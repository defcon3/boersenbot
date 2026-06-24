#!/usr/bin/env python3
"""
tennis_paper_logger.py — Forward-Paper-Test für den Jupiter-Tennis-Edge (KEIN Echtgeld).

Zweck (JUPITER_TENNIS_BETTING_PLAN.md §4 Schritt 2): Für jedes kommende ATP/WTA-
Singles-Match auf Jupiter den PRE-MATCH-Marktpreis gegen die faire Elo-Wahrscheinlichkeit
(tennis_elo_model.py) halten und einen HYPOTHETISCHEN Einsatz protokollieren — ohne Geld.
Nach ~50–100 Matches lässt sich auswerten, ob unterbepreiste Seiten real positiven EV
hatten oder ob Adverse Selection sie auffrisst (vgl. §3 des Plans).

Methodik wie beim Football-Collector: ROHWERTE speichern (fair + Preis je Seite), damit
Schwelle/Blend JEDERZEIT nachträglich in SQL variierbar sind — kein Vorab-Filtern.

Datenfluss:
  1) Discovery: Jupiter GET /api/v1/events?subcategory={atp,wta}  -> 2-Wege-Singles.
  2) Pro Match: faire P(Spieler) aus Elo (Belag aus Turniername geschätzt) + Marktpreis.
  3) Nur Matches, deren BEIDE Spieler im Elo-Report sind (sonst out-of-coverage -> skip).
  4) PRE-MATCH-Snapshot in bb_TennisPaperBets (Upsert: aktualisiert bis Anpfiff = Closing-Linie).
  5) --settle: nach Anpfiff den Sieger aus Jupiters market.result nachtragen.

Speicher: Centron SQL Server (dbdata), Tabelle bb_TennisPaperBets.

Aufruf:
  python tennis_paper_logger.py --once            # ein Discovery-/Log-Durchlauf
  python tennis_paper_logger.py --dry --once      # nur zeigen, nicht schreiben
  python tennis_paper_logger.py --settle --once   # Ergebnisse beendeter Matches nachtragen
  python tennis_paper_logger.py --loop --interval 1800
"""

import argparse
import logging
import sys
from datetime import datetime, timezone

import requests

try:
    import pymssql
except ImportError:
    pymssql = None

from tennis_elo_model import fair_prob, load_elo

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

JUP = "https://prediction-market-api.jup.ag"
EVENTS = f"{JUP}/api/v1/events"
SUBCATS = ("atp", "wta")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger("tennis_paper")

# Belag-Schätzung aus dem Turniernamen (best-effort; Turnier wird mitgespeichert,
# damit sich der Belag jederzeit in SQL korrigieren lässt). Reihenfolge: grass/clay vor hard.
SURFACE_KW = {
    "grass": ("wimbledon", "eastbourne", "halle", "queen", "hertogenbosch", "'s-hertog",
              "mallorca", "stuttgart", "nottingham", "birmingham", "bad homburg", "berlin",
              "newport", "ilkley", "surbiton"),
    "clay": ("roland garros", "french open", "monte", "madrid", "rome", "roma", "barcelona",
             "hamburg", "kitzbuhel", "kitzbühel", "gstaad", "bastad", "båstad", "umag",
             "bucharest", "estoril", "munich", "münchen", "houston", "marrakech", "santiago",
             "cordoba", "córdoba", "buenos aires", "rio", "piracicaba", "sao paulo", "são paulo",
             "lima", "bogota", "bogotá", "perugia", "parma", "cagliari", "tunis", "antalya clay"),
    "hard": ("australian open", "us open", "indian wells", "miami", "cincinnati", "shanghai",
             "dubai", "doha", "acapulco", "tokyo", "beijing", "toronto", "montreal", "canada",
             "winston", "atlanta", "washington", "metz", "vienna", "basel", "paris masters",
             "rotterdam", "marseille", "delray", "dallas", "eastbourne hard"),
}


def guess_surface(tournament):
    t = (tournament or "").lower()
    for surf in ("grass", "clay", "hard"):
        if any(kw in t for kw in SURFACE_KW[surf]):
            return surf
    return "hard"   # Default + bewusst protokolliert (Turnier gespeichert -> später korrigierbar)


# ---------------------------------------------------------------- DB

def get_conn():
    if pymssql is None:
        raise RuntimeError("pymssql nicht installiert (pip install pymssql)")
    return pymssql.connect(**DB_CONFIG, autocommit=True)


DDL = [
    """
    IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name='bb_TennisPaperBets')
    CREATE TABLE bb_TennisPaperBets (
        id              BIGINT IDENTITY(1,1) PRIMARY KEY,
        event_id        NVARCHAR(64)  NOT NULL UNIQUE,
        slug            NVARCHAR(256) NULL,
        tour            NVARCHAR(8)   NULL,        -- atp | wta
        tournament      NVARCHAR(256) NULL,
        surface         NVARCHAR(16)  NULL,        -- grass|clay|hard (geschätzt)
        begin_utc       DATETIME      NULL,        -- Anpfiff
        player_a        NVARCHAR(128) NULL,
        player_b        NVARCHAR(128) NULL,
        elo_a           FLOAT         NULL,
        elo_b           FLOAT         NULL,
        fair_a          FLOAT         NULL,        -- Modell P(A gewinnt)
        fair_b          FLOAT         NULL,
        price_a         FLOAT         NULL,        -- Markt-Ask A (buyYes) zum letzten Pre-Match-Snapshot
        price_b         FLOAT         NULL,
        match_score_a   FLOAT         NULL,        -- Namens-Match-Konfidenz
        match_score_b   FLOAT         NULL,
        blend           FLOAT         NULL,
        value_side      NVARCHAR(8)   NULL,        -- A|B|none: Seite mit positivem Edge (fair-price)
        edge            FLOAT         NULL,        -- max(fair_a-price_a, fair_b-price_b)
        actual_winner   NVARCHAR(8)   NULL,        -- A|B (nach Auflösung)
        settled         BIT           NOT NULL DEFAULT 0,
        logged_utc      DATETIME      NOT NULL DEFAULT GETUTCDATE(),
        updated_utc     DATETIME      NOT NULL DEFAULT GETUTCDATE()
    )
    """,
]


def ensure_tables(conn):
    cur = conn.cursor()
    for stmt in DDL:
        cur.execute(stmt)
    log.info("Tabelle bb_TennisPaperBets sichergestellt.")


# ---------------------------------------------------------------- Jupiter-Discovery

def _epoch_to_dt(e):
    try:
        return datetime.fromtimestamp(int(e), timezone.utc)
    except Exception:
        return None


def fetch_singles(subcat, max_events=200):
    """Jupiter-Events einer Subkategorie paginieren -> 2-Wege-Singles-Matches.

    Liefert dicts mit Spielern, Ask-Preisen (buyYes je Outcome), Turnier, Anpfiff,
    eventId, isLive und den marketResult-Feldern (für spätere Auflösung).
    """
    out, start = [], 0
    while start < max_events:
        try:
            r = requests.get(EVENTS, params={"subcategory": subcat, "start": start, "end": start + 10}, timeout=20)
            data = r.json().get("data", [])
        except Exception as e:
            log.warning(f"  {subcat} Discovery-Fehler bei start={start}: {e}")
            break
        if not data:
            break
        for e in data:
            md = e.get("metadata") or {}
            mks = e.get("markets") or []
            title = md.get("title", "") or ""
            slug = md.get("slug", "") or ""
            if len(mks) != 2:
                continue
            # Doppel ausschließen (Teamnamen mit '/'), Futures/Winner-Märkte sind keine 2-Markt-Events
            if "/" in title or "doubles" in slug.lower():
                continue
            ma, mb = mks
            pa = (ma.get("title") or "").strip()
            pb = (mb.get("title") or "").strip()
            if not pa or not pb:
                continue
            # Turniername = Teil vor dem ersten ':' im Event-Titel ("Wimbledon, Qualification ATP: A vs B")
            tournament = title.split(":")[0].strip() if ":" in title else title

            def ask(m):
                p = m.get("pricing") or {}
                v = p.get("buyYesPriceUsd")
                return round(v / 1e6, 4) if v else None
            out.append({
                "event_id": str(e.get("eventId")),
                "slug": slug, "tour": subcat, "tournament": tournament,
                "title": title, "is_live": bool(e.get("isLive")),
                "begin_utc": _epoch_to_dt(e.get("beginAt")),
                "player_a": pa, "player_b": pb,
                "price_a": ask(ma), "price_b": ask(mb),
                "result_a": ma.get("result"), "result_b": mb.get("result"),
                "status_a": ma.get("status"), "status_b": mb.get("status"),
            })
        start += 10
    return out


def fetch_inventory_index():
    """Komplettes ATP+WTA-Singles-Inventory EINMAL ziehen, nach event_id indiziert.
    Beendete Matches erscheinen als status='closed' mit result 'yes'/'no' und bleiben
    eine Weile gelistet — Settle muss daher zeitnah (innerhalb ~1–2 Tagen) laufen,
    sonst fällt das Event aus der Liste und das Ergebnis ist nicht mehr abrufbar."""
    idx = {}
    for sub in SUBCATS:
        for ev in fetch_singles(sub):
            idx[ev["event_id"]] = ev
    return idx


# ---------------------------------------------------------------- Logging

def evaluate(ev, blend, min_score):
    """Faire Wahrscheinlichkeit + Edge für ein Match berechnen. None, wenn Spieler
    nicht (sicher) im Elo-Report oder Preise fehlen."""
    if ev["price_a"] is None or ev["price_b"] is None:
        return None, "keine Preise"
    surface = guess_surface(ev["tournament"])
    fp = fair_prob(ev["player_a"], ev["player_b"], surface=surface, tour=ev["tour"],
                   blend=blend, min_score=min_score)
    if not fp["ok"]:
        return None, "; ".join(fp["warn"]) or "Spieler nicht im Report"
    fair_a, fair_b = fp["p_a"], fp["p_b"]
    edge_a = fair_a - ev["price_a"]
    edge_b = fair_b - ev["price_b"]
    side, edge = ("A", edge_a) if edge_a >= edge_b else ("B", edge_b)
    if edge <= 0:
        side = "none"
    return {
        "surface": surface, "fair_a": fair_a, "fair_b": fair_b,
        "elo_a": fp["a"]["elo"], "elo_b": fp["b"]["elo"],
        "score_a": fp["a"]["score"], "score_b": fp["b"]["score"],
        "value_side": side, "edge": round(edge, 4),
    }, None


def upsert_bet(conn, ev, ev_eval, blend):
    """PRE-MATCH-Upsert: legt die Zeile an oder aktualisiert Preise/Fair, SOLANGE
    das Match noch nicht angepfiffen ist (letzter Pre-Match-Stand = Closing-Linie).
    Nach Anpfiff wird NICHT mehr überschrieben (verhindert In-Play-Kontamination)."""
    cur = conn.cursor()
    cur.execute(
        """
        IF EXISTS (SELECT 1 FROM bb_TennisPaperBets WHERE event_id=%s)
            UPDATE bb_TennisPaperBets
               SET price_a=%s, price_b=%s, fair_a=%s, fair_b=%s, elo_a=%s, elo_b=%s,
                   value_side=%s, edge=%s, surface=%s, begin_utc=%s, updated_utc=GETUTCDATE()
             WHERE event_id=%s AND settled=0
               AND (begin_utc IS NULL OR begin_utc > GETUTCDATE())   -- nur PRE-MATCH aktualisieren
        ELSE
            INSERT INTO bb_TennisPaperBets
              (event_id, slug, tour, tournament, surface, begin_utc, player_a, player_b,
               elo_a, elo_b, fair_a, fair_b, price_a, price_b, match_score_a, match_score_b,
               blend, value_side, edge)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            ev["event_id"],
            ev["price_a"], ev["price_b"], ev_eval["fair_a"], ev_eval["fair_b"],
            ev_eval["elo_a"], ev_eval["elo_b"], ev_eval["value_side"], ev_eval["edge"],
            ev_eval["surface"], ev["begin_utc"], ev["event_id"],
            ev["event_id"], ev["slug"], ev["tour"], ev["tournament"], ev_eval["surface"],
            ev["begin_utc"], ev["player_a"], ev["player_b"], ev_eval["elo_a"], ev_eval["elo_b"],
            ev_eval["fair_a"], ev_eval["fair_b"], ev["price_a"], ev["price_b"],
            ev_eval["score_a"], ev_eval["score_b"], blend, ev_eval["value_side"], ev_eval["edge"],
        ),
    )


def run_log(conn, args):
    now = datetime.now(timezone.utc)
    matches = []
    for sub in SUBCATS:
        matches += fetch_singles(sub)
    log.info(f"Discovery: {len(matches)} Singles-Matches (ATP+WTA).")

    n_log = n_skip = n_live = 0
    for ev in matches:
        # Nur PRE-MATCH protokollieren (In-Play-Preise sind score-getrieben -> kontaminiert)
        if ev["begin_utc"] is not None and ev["begin_utc"] <= now:
            n_live += 1
            continue
        ev_eval, why = evaluate(ev, args.blend, args.min_score)
        if ev_eval is None:
            n_skip += 1
            if args.verbose:
                log.info(f"  skip {ev['player_a']} vs {ev['player_b']} ({ev['tournament']}): {why}")
            continue
        tag = f"{ev['tour'].upper()} {ev['tournament'][:22]:22s} {ev['player_a']} vs {ev['player_b']}"
        log.info(f"  {tag}  [{ev_eval['surface']}]  fair {ev_eval['fair_a']:.2f}/{ev_eval['fair_b']:.2f} "
                 f"Markt {ev['price_a']:.2f}/{ev['price_b']:.2f}  -> Value {ev_eval['value_side']} "
                 f"(Edge {ev_eval['edge']:+.3f})")
        if not args.dry:
            upsert_bet(conn, ev, ev_eval, args.blend)
        n_log += 1
    log.info(f"Fertig: {n_log} geloggt/aktualisiert, {n_skip} out-of-coverage, {n_live} bereits angepfiffen.")


# ---------------------------------------------------------------- Settle

def run_settle(conn, args):
    """Sieger beendeter Matches aus Jupiters market.result nachtragen."""
    cur = conn.cursor()
    cur.execute("SELECT event_id, player_a, player_b, begin_utc FROM bb_TennisPaperBets WHERE settled=0")
    pending = cur.fetchall()
    log.info(f"Settle: {len(pending)} offene Matches.")
    now = datetime.now(timezone.utc)
    idx = None
    n = 0
    for event_id, pa, pb, begin in pending:
        if begin is not None and begin.replace(tzinfo=timezone.utc) > now:
            continue   # noch nicht gespielt
        if idx is None:                       # Inventory erst ziehen, wenn wirklich nötig
            idx = fetch_inventory_index()
        ev = idx.get(event_id)
        if ev is None:
            continue
        winner = _winner_side(ev)
        if winner is None:
            continue
        if not args.dry:
            cur.execute("UPDATE bb_TennisPaperBets SET actual_winner=%s, settled=1, updated_utc=GETUTCDATE() "
                        "WHERE event_id=%s", (winner, event_id))
        log.info(f"  aufgelöst {pa} vs {pb}: Sieger {winner}")
        n += 1
    log.info(f"Settle fertig: {n} Matches aufgelöst.")


def _winner_side(ev):
    """Sieger 'A'/'B' aus market.result (oder extremem Schlusspreis) ableiten; sonst None."""
    ra = (str(ev.get("result_a") or "")).lower()
    rb = (str(ev.get("result_b") or "")).lower()
    # Jupiter setzt result auf das gewinnende Outcome ('Yes'/'No' bzw. Spielername)
    if ra in ("yes", "won", "win", "1") or ra == ev["player_a"].lower():
        return "A"
    if rb in ("yes", "won", "win", "1") or rb == ev["player_b"].lower():
        return "B"
    if ra in ("no", "lost", "0") and rb in ("yes", "won", "win", "1"):
        return "B"
    # Fallback: aufgelöster Markt -> Preis ~1.0 = Sieger
    if ev.get("status_a") not in ("open", None) and ev["price_a"] is not None and ev["price_b"] is not None:
        if ev["price_a"] >= 0.95:
            return "A"
        if ev["price_b"] >= 0.95:
            return "B"
    return None


# ---------------------------------------------------------------- CLI

def run(args):
    conn = None
    if not args.dry or args.settle:
        conn = get_conn()
        ensure_tables(conn)
    load_elo("atp", quiet=True)
    if "wta" in SUBCATS:
        load_elo("wta", quiet=True)

    log.info("=" * 70)
    log.info(f"TENNIS-PAPER-LOGGER  |  {'SETTLE' if args.settle else ('DRY' if args.dry else 'LIVE')}  "
             f"|  blend={args.blend}  min-score={args.min_score}")
    log.info("=" * 70)

    import time
    while True:
        try:
            if args.settle:
                run_settle(conn, args)
            else:
                run_log(conn, args)
        except Exception as e:
            log.warning(f"Durchlauf-Fehler: {e}")
            try:
                conn = get_conn()
            except Exception:
                pass
        if args.once:
            break
        time.sleep(args.interval)


def main():
    ap = argparse.ArgumentParser(description="Forward-Paper-Test-Logger für Jupiter-Tennis (-> Centron).")
    ap.add_argument("--blend", type=float, default=0.5, help="Overall/Belag-Elo-Gewicht (default 0.5)")
    ap.add_argument("--min-score", type=float, default=0.85, help="Min. Namens-Match-Konfidenz (default 0.85)")
    ap.add_argument("--interval", type=int, default=1800, help="Loop-Intervall Sek. (default 1800)")
    ap.add_argument("--settle", action="store_true", help="Statt loggen: Ergebnisse beendeter Matches nachtragen")
    ap.add_argument("--dry", action="store_true", help="Nicht in die DB schreiben, nur zeigen")
    ap.add_argument("--once", action="store_true", help="Genau ein Durchlauf, dann Ende")
    ap.add_argument("--loop", dest="once", action="store_false", help="Endlos (Default ist --once)")
    ap.add_argument("--verbose", action="store_true", help="Auch übersprungene (out-of-coverage) Matches zeigen")
    ap.set_defaults(once=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
