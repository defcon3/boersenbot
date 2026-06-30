#!/usr/bin/env python3
"""
crypto_updown_logger.py — Tick-Logger für Jupiters 15-Minuten-"Up or Down"-Krypto-Märkte.

Assets: BTC, ETH, SOL, DOGE, BNB (alle mit slug-Muster {asset}-updown-15m-{startEpoch},
Range = [epoch, epoch+900], Auflösung "Up" wenn Asset-Preis am Range-Ende >= am Range-Start
lt. Chainlink). ETH/XRP-Hinweis: ETH HAT 15m-Märkte (nur tiefer in der Liste); XRP/ADA hatten
beim Scan keine — die ASSETS-Liste unten ist die belastbare Auswahl.

Zweck (HYPOTHESE, NICHT verifiziert): Betreiber-Beobachtung — der jeweils hintenliegende
(Underdog-)Kontrakt pendelt im Band ~0,20–0,40 (Sinuskurve) statt linear gegen 0/1. FRAGE:
Mean-Reversion-Edge ausnutzbar, oder nur Abbildung des Spot-Random-Walks, der binär zu 0/1
auflöst? Methodik wie Football-Collector / Tennis-Paper-Logger: ROHE Zeitreihe, KEIN
Vorab-Filter — Strategie wird SPÄTER per SQL/Backtest geprüft. Kein Echtgeld in diesem Skript.

Speicher: Centron SQL Server (dbdata), Tabelle bb_CryptoUpDown15m (asset-Spalte).

Aufruf:
  python crypto_updown_logger.py --once
  python crypto_updown_logger.py --dry --once
  python crypto_updown_logger.py --loop --interval 30
  python crypto_updown_logger.py --assets btc,eth --loop      # Teilmenge
"""

import argparse
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

JUP = "https://prediction-market-api.jup.ag/api/v1"
EVENTS = f"{JUP}/events"
ASSETS = ["btc", "eth", "sol", "doge", "bnb"]

# Discovery-Fenster: ein Markt ist "aktiv" zu loggen, sobald seine 15-Min-Range
# bald startet/läuft, bis kurz nach Schluss (um result einzufangen).
HORIZON_MIN = 22       # nächste Range IM VORAUS tracken (frische Range steht erst spät in der Liste)
LOOKBACK_MIN = 4
DISCOVER_PAGES = 8     # tief genug paginieren, damit auch die volumenarme nächste Range gefunden wird

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger("crypto_updown")


# ---------------------------------------------------------------- DB

def get_conn():
    if pymssql is None:
        raise RuntimeError("pymssql nicht installiert (pip install pymssql)")
    return pymssql.connect(**DB_CONFIG, autocommit=True)


DDL = [
    """
    IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name='bb_CryptoUpDown15m')
    CREATE TABLE bb_CryptoUpDown15m (
        id              BIGINT IDENTITY(1,1) PRIMARY KEY,
        asset           NVARCHAR(8)   NOT NULL,      -- btc|eth|sol|doge|bnb
        event_id        NVARCHAR(64)  NOT NULL,
        slug            NVARCHAR(128) NULL,
        range_start_utc DATETIME      NULL,          -- Beginn der 15-Min-Range (Referenzpreis fixiert)
        range_end_utc   DATETIME      NULL,          -- Ende/Auflösung (= closeTime)
        ts_utc          DATETIME      NOT NULL,       -- Messzeitpunkt dieses Ticks
        secs_to_close   INT           NULL,           -- Restsekunden bis range_end
        up_buy          FLOAT         NULL,           -- Ask 'Up'  (buyYesPriceUsd / 1e6)
        up_sell         FLOAT         NULL,           -- Bid 'Up'  (sellYesPriceUsd / 1e6)
        down_buy        FLOAT         NULL,           -- Ask 'Down'
        down_sell       FLOAT         NULL,           -- Bid 'Down'
        result          NVARCHAR(8)   NULL,           -- Up | Down (nach Auflösung nachgetragen)
        settled         BIT           NOT NULL DEFAULT 0,
        logged_utc      DATETIME      NOT NULL DEFAULT GETUTCDATE()
    )
    """,
    """
    IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name='ix_cryptoud_event')
    CREATE INDEX ix_cryptoud_event ON bb_CryptoUpDown15m(event_id, ts_utc)
    """,
    """
    IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name='ix_cryptoud_asset')
    CREATE INDEX ix_cryptoud_asset ON bb_CryptoUpDown15m(asset, range_end_utc)
    """,
]


def ensure_tables(conn):
    cur = conn.cursor()
    for stmt in DDL:
        cur.execute(stmt)
    log.info("Tabelle bb_CryptoUpDown15m sichergestellt.")


# ---------------------------------------------------------------- Jupiter-Discovery

def _dt(epoch):
    try:
        return datetime.fromtimestamp(int(epoch), timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _market(ev, title):
    for m in ev.get("markets", []):
        if m.get("title") == title:
            return m
    return {}


def get_events_page(subcat, start, end, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(EVENTS, params={"subcategory": subcat, "start": start, "end": end},
                             timeout=15)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", "") or 3 * (attempt + 1))
                log.warning(f"429 {subcat} Seite {start} — warte {wait:.0f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json().get("data", [])
        except requests.RequestException as e:
            log.warning(f"{subcat} Seite {start} Fehler ({e}), Versuch {attempt + 1}")
            time.sleep(2)
    return []


def discover(now, assets, tracked):
    """Paginiert je Asset, fügt neue 15m-Events (closeTime im Fenster) zu `tracked`.
    Teuer (Pagination) -> selten aufrufen; das eigentliche Tick-Polling holt die
    Events danach gezielt per fetch_event()."""
    for asset in assets:
        for s in range(0, DISCOVER_PAGES * 10, 10):
            page = get_events_page(asset, s, s + 10)
            if not page:
                break
            for e in page:
                if "15m" not in e.get("tags", []):
                    continue
                ct = _market(e, "Up").get("closeTime") or 0
                if not ct:
                    continue
                # Tracken sobald die Range bald startet bzw. läuft (bis kurz nach Schluss).
                if now - LOOKBACK_MIN * 60 <= ct <= now + HORIZON_MIN * 60:
                    eid = e.get("eventId")
                    if eid not in tracked:
                        tracked[eid] = {"asset": asset, "slug": e.get("metadata", {}).get("slug"),
                                        "end": ct}
    return tracked


def fetch_event(event_id, retries=3):
    """Einzelnes Event gezielt per eventId holen (billig, 1 Request) -> ev-dict oder None.

    /events/{id} liefert das Event-Objekt DIREKT zurück (nicht in 'data'), Struktur
    wie in der Pagination (metadata, markets[].pricing/result)."""
    for attempt in range(retries):
        try:
            r = requests.get(f"{JUP}/events/{event_id}", timeout=12)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", "") or 3 * (attempt + 1))
                log.warning(f"429 fetch {event_id} — warte {wait:.0f}s")
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            log.warning(f"fetch {event_id} Fehler ({e}), Versuch {attempt + 1}")
            time.sleep(1.5)
    return None


def snapshot(asset, e, now):
    up, down = _market(e, "Up"), _market(e, "Down")
    pu, pd = up.get("pricing", {}), down.get("pricing", {})
    ct = up.get("closeTime") or 0
    md = e.get("metadata", {})
    rs = ct - 900 if ct else None
    return {
        "asset": asset,
        "event_id": e.get("eventId"),
        "slug": md.get("slug"),
        "range_start_utc": _dt(rs),
        "range_end_utc": _dt(ct),
        "secs_to_close": int(ct - now) if ct else None,
        "up_buy": (pu.get("buyYesPriceUsd") or 0) / 1e6,
        "up_sell": (pu.get("sellYesPriceUsd") or 0) / 1e6,
        "down_buy": (pd.get("buyYesPriceUsd") or 0) / 1e6,
        "down_sell": (pd.get("sellYesPriceUsd") or 0) / 1e6,
        "result": up.get("result"),
    }


# ---------------------------------------------------------------- Schreiben

def insert_tick(conn, snap):
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO bb_CryptoUpDown15m
           (asset, event_id, slug, range_start_utc, range_end_utc, ts_utc, secs_to_close,
            up_buy, up_sell, down_buy, down_sell)
           VALUES (%s,%s,%s,%s,%s,GETUTCDATE(),%s,%s,%s,%s,%s)""",
        (snap["asset"], snap["event_id"], snap["slug"], snap["range_start_utc"],
         snap["range_end_utc"], snap["secs_to_close"],
         snap["up_buy"], snap["up_sell"], snap["down_buy"], snap["down_sell"]),
    )


def settle_event(conn, event_id, result):
    cur = conn.cursor()
    cur.execute("UPDATE bb_CryptoUpDown15m SET result=%s, settled=1 WHERE event_id=%s",
                (result, event_id))
    return cur.rowcount


# ---------------------------------------------------------------- Lauf

def poll_tick(conn, dry, tracked, settled_cache):
    """Häufiger, billiger Poll: jedes getrackte Event dessen Range läuft gezielt holen,
    Tick schreiben bzw. nach Auflösung settlen. Räumt erledigte/abgelaufene Events auf."""
    now = int(time.time())
    ticks, settles = 0, 0
    per_asset = {}
    for eid in list(tracked.keys()):
        meta = tracked[eid]
        ct = meta["end"]
        # Aufräumen: lange nach Schluss + nicht mehr brauchbar -> verwerfen
        if now > ct + LOOKBACK_MIN * 60 + 30:
            tracked.pop(eid, None)
            continue
        # Range noch nicht gestartet (Referenzpreis nicht fixiert) -> noch nicht pollen
        if now < ct - 900 - 15:
            continue
        ev = fetch_event(eid)
        if not ev:
            continue
        snap = snapshot(meta["asset"], ev, now)
        if snap["result"] in ("Up", "Down"):
            if eid not in settled_cache:
                if not dry:
                    settle_event(conn, eid, snap["result"])
                settled_cache.add(eid)
                settles += 1
            tracked.pop(eid, None)
            continue
        # Range vorbei, aber result noch nicht da -> Preise degenerieren -> kein Tick
        if snap["secs_to_close"] is not None and snap["secs_to_close"] <= 0:
            continue
        if dry:
            log.info(f"[dry] {meta['asset']:4} {eid} t={snap['secs_to_close']:>4}s  "
                     f"Up {snap['up_buy']:.3f}/{snap['up_sell']:.3f}  "
                     f"Down {snap['down_buy']:.3f}/{snap['down_sell']:.3f}")
        else:
            insert_tick(conn, snap)
        ticks += 1
        per_asset[meta["asset"]] = per_asset.get(meta["asset"], 0) + 1
    log.info(f"Poll: {ticks} Ticks {dict(per_asset)}, {settles} Settles, "
             f"{len(tracked)} getrackt.")


def main():
    ap = argparse.ArgumentParser(description="Multi-Asset Tick-Logger für Jupiter 15m Up/Down.")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=10, help="Poll-Intervall in s (Loop, gezieltes fetch)")
    ap.add_argument("--discover-interval", type=int, default=180,
                    help="Discovery-Intervall in s (teure Pagination, findet neue Events)")
    ap.add_argument("--assets", default=",".join(ASSETS), help="Komma-Liste, default alle")
    ap.add_argument("--dry", action="store_true", help="nur zeigen, nicht schreiben")
    args = ap.parse_args()

    assets = [a.strip().lower() for a in args.assets.split(",") if a.strip()]

    conn = None
    if not args.dry:
        conn = get_conn()
        ensure_tables(conn)

    tracked, settled_cache = {}, set()
    if args.loop:
        log.info(f"Loop-Start (Poll {args.interval}s, Discovery {args.discover_interval}s, "
                 f"Assets {assets}). Strg+C zum Beenden.")
        last_discover = 0.0
        while True:
            t0 = time.time()
            try:
                if time.time() - last_discover >= args.discover_interval:
                    discover(int(time.time()), assets, tracked)
                    last_discover = time.time()
                poll_tick(conn, args.dry, tracked, settled_cache)
            except Exception as e:
                log.error(f"Loop-Fehler: {e}")
            if len(settled_cache) > 1000:
                settled_cache = set(list(settled_cache)[-300:])
            # Fetch-/Discovery-Dauer vom Intervall abziehen -> echter ~interval-Zyklus
            time.sleep(max(1.0, args.interval - (time.time() - t0)))
    else:
        # --once: einmal entdecken + einmal pollen (für Test/Cron)
        discover(int(time.time()), assets, tracked)
        poll_tick(conn, args.dry, tracked, settled_cache)


if __name__ == "__main__":
    main()
