#!/usr/bin/env python3
"""
Inkrementeller Daily-OHLCV-Refresher fuer bb_StockPrices.

Warum noetig: ingest_sp500_fixed.py hat end="2026-05-15" hart codiert
und ueberspringt bereits vorhandene Symbole -> haengt nie neue Tage an.
Dieses Skript holt fuer ALLE bereits in bb_StockPrices vorhandenen
Symbole nur die fehlenden juengsten Tage (ab MAX(Date)+1 bis heute)
und macht ein idempotentes Upsert. Fuer den taeglichen Cron gedacht.
"""
import sys
import logging
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, date, timedelta

import yfinance as yf
import pymssql
from config import DB_CONFIG

logging.basicConfig(
    filename='logs/daily_refresh.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

UPSERT = """
IF EXISTS (SELECT 1 FROM bb_StockPrices WHERE Symbol=%s AND Date=%s)
    UPDATE bb_StockPrices
    SET OpenPrice=%s, HighPrice=%s, LowPrice=%s, ClosePrice=%s,
        Volume=%s, UpdatedAt=GETDATE()
    WHERE Symbol=%s AND Date=%s
ELSE
    INSERT INTO bb_StockPrices
        (Symbol, Date, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, UpdatedAt)
    VALUES (%s, %s, %s, %s, %s, %s, %s, GETDATE())
"""


def main():
    log.info("=== Daily-Refresh Start ===")
    conn = pymssql.connect(**DB_CONFIG, autocommit=True)
    cur = conn.cursor()

    cur.execute("SELECT Symbol FROM bb_StockPrices GROUP BY Symbol")
    symbols = sorted(r[0] for r in cur.fetchall())
    cur.execute("SELECT MAX([Date]) FROM bb_StockPrices")
    max_date = cur.fetchone()[0]
    if max_date is None:
        log.error("bb_StockPrices ist leer - Abbruch (kein Initial-Import)")
        return 1

    start = max_date + timedelta(days=1)
    end_excl = date.today() + timedelta(days=1)  # yfinance end ist exklusiv
    if start >= end_excl:
        log.info(f"Bereits aktuell (MAX={max_date}) - nichts zu tun.")
        return 0

    log.info(f"{len(symbols)} Symbole, lade {start} .. {date.today()}")
    BATCH = 60
    total = 0
    for i in range(0, len(symbols), BATCH):
        chunk = symbols[i:i + BATCH]
        try:
            data = yf.download(chunk, start=start, end=end_excl, interval="1d",
                               progress=False, group_by="ticker", threads=True)
        except Exception as ex:
            log.error(f"Batch {i} DL-Fehler: {str(ex)[:120]}")
            continue
        for sym in chunk:
            try:
                sub = data[sym] if len(chunk) > 1 else data
                sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna()
                for idx, r in sub.iterrows():
                    d = idx.date()
                    o, h, l, c = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"])
                    v = int(r["Volume"])
                    cur.execute(UPSERT, (sym, d, o, h, l, c, v, sym, d,
                                         sym, d, o, h, l, c, v))
                    total += 1
            except Exception as ex:
                log.debug(f"{sym}: {str(ex)[:80]}")
        log.info(f"  {min(i + BATCH, len(symbols))}/{len(symbols)} Symbole, {total} Zeilen")

    cur.close()
    conn.close()
    log.info(f"=== FERTIG: {total} Zeilen ge-upserted ===")
    return 0


if __name__ == '__main__':
    sys.exit(main())
