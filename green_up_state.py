#!/usr/bin/env python3
"""
green_up_state.py — geteilter Centron-State zwischen green_up_daemon.py und
autopilot.py: welche Märkte stehen gerade unter Green-up-Verwaltung und sind
deshalb für den Autopilot-Verkauf TABU (Skip-Liste)?

Design-Entscheidung (2026-07-03, Nutzerfrage unbeantwortet -> konservative
Default-Wahl): EIN Wallet, Skip-Liste statt zweitem Wallet — kleinste Änderung
an der laufenden autopilot.py, kein Workflow-Bruch (Leg 1 weiter normal kaufen).
Nur EXPLIZIT per --add benannte Märkte werden verwaltet, kein Auto-Hedge auf
jede neue Position (sonst würde z. B. der Langläufer POLY-2591183 unerwünscht
gehedgt). Siehe preregs/... nein — das hier ist Infra, kein Backtest; Details
in Memory green-up-daemon-todo.

Tabelle bb_GreenUpHedges (Centron, geteilt mit Fußballbot — bb_-Präfix):
  market_id        PK, Leg-1-Markt
  profit_target    gewünschter gesicherter Gewinn/Kontrakt (USD)
  status           'watching' (noch keine Hedge-Order) | 'placed' (Order ruht)
                    | 'locked' (Hedge gefüllt, beide Seiten gehalten)
                    | 'done' (Markt aufgelöst+geclaimt) | 'cancelled' (--remove)
  hedge_is_yes     Seite der Hedge-Order (Gegenseite von Leg 1), NULL bis platziert
  hedge_order_pk   orderPubkey der Hedge-Order (für Cancel), NULL bis platziert
  added_utc / updated_utc
  note             Freitext (letzter Zustand/Fehler)

WICHTIG: get_active_markets() ist die Funktion, die autopilot.py aufruft, um
Verkäufe zu unterdrücken. Sie MUSS fail-open sein (leeres Set bei DB-Fehler) —
ein Centron-Ausfall darf den Autopilot nicht lahmlegen, sondern höchstens auf
altes Verhalten (keine Skip-Liste) zurückfallen.
"""

import logging

try:
    import pymssql
except ImportError:
    pymssql = None

log = logging.getLogger("green_up_state")

# Hardcodierte Centron-Creds — bewusste Projektentscheidung (siehe CLAUDE.md).
DB_CONFIG = {
    "server": "158.181.48.77",
    "database": "dbdata",
    "user": "326773",
    "password": "Extaler11!",
}

ACTIVE_STATUSES = ("watching", "placed", "locked")

DDL = """
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name='bb_GreenUpHedges')
CREATE TABLE bb_GreenUpHedges (
    market_id       NVARCHAR(64)  NOT NULL PRIMARY KEY,
    profit_target   FLOAT         NOT NULL,
    status          NVARCHAR(16)  NOT NULL DEFAULT 'watching',
    hedge_is_yes    BIT           NULL,
    hedge_order_pk  NVARCHAR(64)  NULL,
    added_utc       DATETIME      NOT NULL DEFAULT GETUTCDATE(),
    updated_utc     DATETIME      NOT NULL DEFAULT GETUTCDATE(),
    note            NVARCHAR(256) NULL
)
"""


def get_conn():
    if pymssql is None:
        raise RuntimeError("pymssql nicht installiert (pip install pymssql)")
    return pymssql.connect(**DB_CONFIG, autocommit=True)


def ensure_table(conn=None):
    own = conn is None
    conn = conn or get_conn()
    try:
        conn.cursor().execute(DDL)
    finally:
        if own:
            conn.close()


def get_active_markets() -> set:
    """Market-IDs, die der Autopilot NICHT verkaufen darf. Fail-open: DB-Fehler
    -> leeres Set + Warn-Log, NIEMALS Exception nach außen (autopilot.py darf
    dadurch nicht crashen)."""
    try:
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT market_id FROM bb_GreenUpHedges WHERE status IN (%s,%s,%s)",
                       ACTIVE_STATUSES)
            return {r[0] for r in cur.fetchall()}
        finally:
            conn.close()
    except Exception as e:
        log.warning(f"green_up_state.get_active_markets() DB-Fehler ({e}) — fail-open, kein Skip.")
        return set()


def add_hedge(market_id, profit_target):
    conn = get_conn()
    try:
        ensure_table(conn)
        cur = conn.cursor()
        cur.execute(
            """MERGE bb_GreenUpHedges AS t
               USING (SELECT %s AS market_id) AS s ON t.market_id = s.market_id
               WHEN MATCHED THEN UPDATE SET profit_target=%s, status='watching',
                    hedge_is_yes=NULL, hedge_order_pk=NULL, updated_utc=GETUTCDATE(), note=NULL
               WHEN NOT MATCHED THEN INSERT (market_id, profit_target, status)
                    VALUES (%s, %s, 'watching');""",
            (market_id, profit_target, market_id, profit_target),
        )
    finally:
        conn.close()


def remove_hedge(market_id, note="manuell entfernt"):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """UPDATE bb_GreenUpHedges SET status='cancelled', updated_utc=GETUTCDATE(), note=%s
               WHERE market_id=%s""",
            (note, market_id),
        )
        return cur.rowcount
    finally:
        conn.close()


def update_hedge(market_id, **fields):
    """Setzt beliebige Spalten (status, hedge_is_yes, hedge_order_pk, note)."""
    if not fields:
        return
    cols = ", ".join(f"{k}=%s" for k in fields)
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(f"UPDATE bb_GreenUpHedges SET {cols}, updated_utc=GETUTCDATE() "
                   f"WHERE market_id=%s", (*fields.values(), market_id))
    finally:
        conn.close()


def list_hedges():
    conn = get_conn()
    try:
        ensure_table(conn)
        cur = conn.cursor(as_dict=True)
        cur.execute("SELECT * FROM bb_GreenUpHedges ORDER BY added_utc DESC")
        return cur.fetchall()
    finally:
        conn.close()


def get_hedge(market_id):
    conn = get_conn()
    try:
        cur = conn.cursor(as_dict=True)
        cur.execute("SELECT * FROM bb_GreenUpHedges WHERE market_id=%s", (market_id,))
        return cur.fetchone()
    finally:
        conn.close()
