#!/usr/bin/env python3
"""
btc_flip_logger.py — Digital-Option-Studie auf "Bitcoin Up or Down" (Polymarket/Jupiter).

HYPOTHESE (Nutzer, 2026-07-22): Der 5-/15-Min-BTC-Up/Down-Flip ist eine binaere
(digitale) Option — Strike = BTC-Preis bei Fenster-Open ("Null-Linie"), Expiry =
Fenster-Ende. Die Zeit treibt den Preis gegen 0/1 (Gamma-Explosion nahe Expiry),
auch bei kleinem Abstand zur Null-Linie. Frage: ist die Markt-IMPLIED an einem
gegebenen (Abstand, Restzeit) verschieden von der REALISIERTEN Flip-Haeufigkeit?
Der Penny-Edge des gescouteten Wallets war die extreme Ecke davon (kleiner Abstand
+ kurz vor Schluss: realisiert 16,7 %, impliziert 1 % — Markt hielt Fast-Ties fuer
entschiedener als sie waren). Siehe [[btc-coinflip-degen-edges-dead]].

Erwartung: heute effizient in den handelbaren Ecken (April-Mispricing wegkonkurriert).
Der Logger BEWEIST das oder findet eine gruene Zelle.

Was pro aktivem Fenster alle INTERVAL s geloggt wird:
  - BTC-Spot (Binance BTCUSDT), Strike K (Binance-Open bei Fenster-Open),
    distance_bps = (spot-K)/K*1e4
  - Restzeit in s, Fensterdauer
  - Markt-IMPLIED = Polymarket-CLOB-Midpoint des 'Up'-Tokens (+ best bid/ask)
  - nach Schluss backfill: outcome_up (Markt-Resolution; Binance nur Fallback)

Nur kurze Fenster (Dauer <= MAX_WINDOW_S) — dort sitzt die Konvergenz.
Volumen-Schutz fuer die geteilte Centron-DB: harter ROW_CAP.

Aufruf:
  python btc_flip_logger.py --dry --once
  python btc_flip_logger.py --once
  python btc_flip_logger.py --loop --interval 15
"""
import argparse
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

# Hardcodierte Centron-Creds — bewusste Projektentscheidung (siehe CLAUDE.md).
DB_CONFIG = {
    "server": "158.181.48.77", "database": "dbdata",
    "user": "326773", "password": "Extaler11!",
}

JUP = "https://api.jup.ag/prediction/v1"
CLOB = "https://clob.polymarket.com"
BINANCE = "https://api.binance.com/api/v3"

MAX_WINDOW_S = 15 * 60      # nur Fenster <= 15 Min (5-/10-/15-Min-Bretter)
ROW_CAP = 400_000           # harter Deckel -> schuetzt die geteilte 400-MB-Centron-DB
DEFAULT_INTERVAL = 15

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger("btcflip")

S = requests.Session()
S.headers["User-Agent"] = "boersenbot-btcflip/1.0"

DDL = """
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name='bb_BtcFlip')
CREATE TABLE bb_BtcFlip (
    id            BIGINT IDENTITY(1,1) PRIMARY KEY,
    ts_utc        DATETIME     NOT NULL,
    market_id     NVARCHAR(40) NOT NULL,   -- POLY-...
    title         NVARCHAR(80) NULL,
    win_open_utc  DATETIME     NULL,
    win_close_utc DATETIME     NULL,
    win_secs      INT          NULL,       -- Fensterdauer (close-open)
    secs_left     INT          NULL,       -- Restzeit bis close
    btc_spot      FLOAT        NULL,       -- Binance BTCUSDT
    strike_k      FLOAT        NULL,       -- Binance-Preis bei Fenster-Open (Null-Linie)
    distance_bps  FLOAT        NULL,       -- (spot-K)/K*1e4
    clob_mid      FLOAT        NULL,       -- Polymarket-CLOB-Midpoint 'Up' = IMPLIED P(up)
    clob_bid      FLOAT        NULL,
    clob_ask      FLOAT        NULL,
    outcome_up    INT          NULL,       -- backfill: 1=Up, 0=Down
    outcome_src   NVARCHAR(10) NULL,       -- 'market' | 'binance'
    logged_utc    DATETIME     NOT NULL DEFAULT GETUTCDATE()
);
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name='ix_btcflip_mkt')
CREATE INDEX ix_btcflip_mkt ON bb_BtcFlip(market_id, ts_utc);
"""


def get_json(url, params=None, tries=3, timeout=15):
    r = None
    for a in range(tries):
        try:
            r = S.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            time.sleep(2 * (a + 1)); r = None; continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(3 * (a + 1)); continue
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except ValueError:
            return None
    return None


def binance_spot():
    d = get_json(f"{BINANCE}/ticker/price", {"symbol": "BTCUSDT"})
    return float(d["price"]) if d and d.get("price") else None


def binance_open_at(ts):
    """Binance 1m-Bar-Open bei Sekunden-Timestamp ts (= Strike-Naeherung)."""
    st = int(ts) * 1000
    d = get_json(f"{BINANCE}/klines",
                 {"symbol": "BTCUSDT", "interval": "1m", "startTime": st, "endTime": st + 60000, "limit": 1})
    if d and len(d) >= 1:
        return float(d[0][1])  # open der Bar
    return None


# Titel-Zeitspanne, z. B. "... - July 23, 1:00PM-1:15PM ET"
_TIME_RANGE = re.compile(r"(\d{1,2}):(\d{2})\s*([AP]M)\s*-\s*(\d{1,2}):(\d{2})\s*([AP]M)", re.I)


def _minute_of_day(h, m, ap):
    h = h % 12
    if ap.upper() == "PM":
        h += 12
    return h * 60 + m


def window_dur_from_title(title):
    """Fensterdauer in Sekunden aus der Titel-Zeitspanne, sonst None.
    'openTime' der API ist der HANDELS-Open (~24 h), NICHT der Fenster-Start —
    daher leiten wir die Dauer aus dem Titel ab und open = close - dur."""
    mm = _TIME_RANGE.search(title)
    if not mm:
        return None
    a = _minute_of_day(int(mm.group(1)), int(mm.group(2)), mm.group(3))
    b = _minute_of_day(int(mm.group(4)), int(mm.group(5)), mm.group(6))
    d = (b - a) % (24 * 60)  # Tageswechsel (z. B. 11:55PM-12:00AM)
    return d * 60 if d > 0 else None


def active_windows(now_ts):
    """Offene BTC-Up/Down-Fenster mit Dauer <= MAX_WINDOW_S, dedupliziert.
    -> [{market_id, up_token, title, open_ts, close_ts, dur}]"""
    seen, out = set(), []
    for s in range(0, 60, 10):
        page = get_json(f"{JUP}/events", {"category": "crypto", "start": s, "end": s + 10}, timeout=20)
        if page is None:
            break
        data = page.get("data", [])
        for e in data:
            t = (e.get("metadata", {}) or {}).get("title", "")
            if "Bitcoin Up or Down" not in t:
                continue
            dur = window_dur_from_title(t)
            if not dur or dur > MAX_WINDOW_S:
                continue
            for mk in e.get("markets", []):
                if mk.get("status") != "open":
                    continue
                ct = mk.get("closeTime")
                toks = mk.get("clobTokenIds") or []
                if not (ct and toks) or ct <= now_ts:
                    continue
                ot = ct - dur  # Fenster-Start aus zuverlaessigem close abgeleitet
                key = (t, ot, ct)
                if key in seen:
                    continue
                seen.add(key)
                base = (mk.get("marketId") or "").rsplit("-", 1)[0] or mk.get("marketId")
                out.append({"market_id": base, "up_token": toks[0], "title": t[:80],
                            "open_ts": ot, "close_ts": ct, "dur": dur})
        if len(data) < 10:
            break
    return out


def clob_quote(token):
    """(mid, best_bid, best_ask) fuer den Up-Token aus dem Polymarket-CLOB-Book."""
    b = get_json(f"{CLOB}/book", {"token_id": token})
    if not b:
        m = get_json(f"{CLOB}/midpoint", {"token_id": token})
        mid = float(m["mid"]) if m and m.get("mid") else None
        return mid, None, None
    bids = b.get("bids") or []
    asks = b.get("asks") or []
    # Polymarket-Book: Preise als Strings; bestes Gebot = hoechster bid, bester Ask = niedrigster ask
    bb = max((float(x["price"]) for x in bids), default=None)
    ba = min((float(x["price"]) for x in asks), default=None)
    if bb is not None and ba is not None:
        mid = (bb + ba) / 2
    else:
        m = get_json(f"{CLOB}/midpoint", {"token_id": token})
        mid = float(m["mid"]) if m and m.get("mid") else (bb if bb is not None else ba)
    return mid, bb, ba


def market_outcome(market_id, up_token):
    """Resolution des Fensters. Bevorzugt Markt-Resolution (Chainlink), sonst None.
    Rueckgabe (outcome_up 1/0/None, src)."""
    # Jupiter/PM-Markt erneut abfragen — nach Schluss traegt 'result' den Ausgang.
    for mid_try in (f"{market_id}-0", market_id):
        d = get_json(f"https://prediction-market-api.jup.ag/api/v1/markets/{mid_try}")
        if not d:
            continue
        res = d.get("result")
        if res in ("Up", "up", "YES", "Yes"):
            return 1, "market"
        if res in ("Down", "down", "NO", "No"):
            return 0, "market"
        # manche geben ergebnis ueber pricing (aufgeloest -> 1.0/0.0)
        pr = d.get("pricing") or {}
        by = (pr.get("buyYesPriceUsd") or 0) / 1e6
        if d.get("status") in ("resolved", "closed") and by:
            return (1 if by >= 0.5 else 0), "market"
    return None, None


def insert_rows(conn, rows):
    cur = conn.cursor()
    cols = ("ts_utc", "market_id", "title", "win_open_utc", "win_close_utc", "win_secs",
            "secs_left", "btc_spot", "strike_k", "distance_bps", "clob_mid", "clob_bid",
            "clob_ask")
    ph = ",".join(["%s"] * len(cols))
    cur.executemany(f"INSERT INTO bb_BtcFlip ({','.join(cols)}) VALUES ({ph})",
                    [tuple(r[c] for c in cols) for r in rows])
    conn.commit()


def backfill(conn, now_ts):
    """Fenster, die >120 s geschlossen sind und noch kein outcome haben -> Resolution."""
    cur = conn.cursor(as_dict=True)
    cur.execute("""SELECT DISTINCT market_id, win_close_utc FROM bb_BtcFlip
                   WHERE outcome_up IS NULL AND win_close_utc < DATEADD(second,-120,GETUTCDATE())""")
    todo = cur.fetchall()
    for r in todo[:40]:
        mid = r["market_id"]
        # up_token brauchen wir fuer den pricing-Fallback nicht zwingend
        oc, src = market_outcome(mid, None)
        if oc is None:
            continue
        cur2 = conn.cursor()
        cur2.execute("UPDATE bb_BtcFlip SET outcome_up=%s, outcome_src=%s WHERE market_id=%s AND outcome_up IS NULL",
                     (oc, src, mid))
        conn.commit()
        log.info(f"  backfill {mid}: outcome_up={oc} ({src})")


def row_count(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM bb_BtcFlip")
    return cur.fetchone()[0]


def poll_once(conn, strike_cache, dry):
    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()
    wins = active_windows(now_ts)
    spot = binance_spot()
    if spot is None:
        log.warning("kein Binance-Spot — Poll uebersprungen"); return
    rows = []
    for w in wins:
        mid = w["market_id"]
        if mid not in strike_cache:
            k = binance_open_at(w["open_ts"])
            strike_cache[mid] = k
        k = strike_cache.get(mid)
        cmid, cbid, cask = clob_quote(w["up_token"])
        dist = (spot - k) / k * 1e4 if k else None
        rows.append({
            "ts_utc": now, "market_id": mid, "title": w["title"],
            "win_open_utc": datetime.fromtimestamp(w["open_ts"], timezone.utc),
            "win_close_utc": datetime.fromtimestamp(w["close_ts"], timezone.utc),
            "win_secs": int(w["dur"]), "secs_left": int(w["close_ts"] - now_ts),
            "btc_spot": spot, "strike_k": k, "distance_bps": dist,
            "clob_mid": cmid, "clob_bid": cbid, "clob_ask": cask,
        })
    log.info(f"{len(wins)} Fenster | spot {spot:.0f} | " +
             " | ".join(f"{w['title'][-18:]} left{r['secs_left']}s mid={r['clob_mid']} d={r['distance_bps']:.0f}bps"
                        for w, r in zip(wins, rows)))
    if dry:
        return
    if rows:
        insert_rows(conn, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="nicht schreiben")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    args = ap.parse_args()

    if pymssql is None and not args.dry:
        log.error("pymssql fehlt"); return 1
    conn = None
    if not args.dry:
        conn = pymssql.connect(**DB_CONFIG)
        cur = conn.cursor()
        for stmt in DDL.strip().split(";\n"):
            if stmt.strip():
                cur.execute(stmt)
        conn.commit()

    strike_cache = {}
    def tick():
        if conn is not None:
            n = row_count(conn)
            if n >= ROW_CAP:
                log.error(f"ROW_CAP {ROW_CAP} erreicht ({n}) — STOPP (schuetzt geteilte DB). Tabelle auswerten/droppen.")
                return False
            backfill(conn, datetime.now(timezone.utc).timestamp())
        poll_once(conn, strike_cache, args.dry)
        return True

    if args.loop:
        log.info(f"btc_flip_logger LOOP alle {args.interval}s (dry={args.dry})")
        while True:
            try:
                if not tick():
                    break
            except Exception as e:
                log.warning(f"tick-Fehler: {e}")
            time.sleep(args.interval)
    else:
        tick()
    if conn:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
