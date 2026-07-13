# -*- coding: utf-8 -*-
"""archive_crypto_to_nas.py — bb_CryptoUpDown15m (Centron) -> Synology-MariaDB (Archiv).

Anlass 13.07.2026: Centron `dbdata` war voll (400/400 MB Quota), der
Ladder-Logger konnte nicht mehr schreiben. bb_CryptoUpDown15m (417k Zeilen,
145 MB) ist seit dem RED-Befund vom 11.07. (Commit 78be0c44) toter Ballast —
wird hier verlustfrei auf die Synology-MariaDB archiviert und erst nach
Zeilen- und Checksummen-Gleichstand auf Centron gedroppt.

Archiv-Ziel: MariaDB 10.3 auf Synology (192.168.178.32:3306),
DB `boersenbot`, Tabelle `crypto_updown_15m` — per SQL abfragbar.

  python archive_crypto_to_nas.py            # kopieren + verifizieren
  python archive_crypto_to_nas.py --drop     # nach erfolgreicher Verify: DROP auf Centron
"""
import sys
import pymssql
import pymysql

MSSQL = dict(server="158.181.48.77", user="326773", password="Extaler11!",
             database="dbdata", timeout=120, login_timeout=30)
MARIA = dict(host="192.168.178.32", port=3306, user="boersenbot",
             password="bb!Ladder2026mdb", database="boersenbot",
             connect_timeout=15, charset="utf8mb4")

COLS = ["id", "asset", "event_id", "slug", "range_start_utc", "range_end_utc",
        "ts_utc", "secs_to_close", "up_buy", "up_sell", "down_buy", "down_sell",
        "result", "settled", "logged_utc", "spot", "price_to_beat"]

DDL = """
CREATE TABLE IF NOT EXISTS crypto_updown_15m (
  id BIGINT NOT NULL PRIMARY KEY,
  asset VARCHAR(8), event_id VARCHAR(64), slug VARCHAR(128),
  range_start_utc DATETIME, range_end_utc DATETIME, ts_utc DATETIME,
  secs_to_close INT, up_buy DOUBLE, up_sell DOUBLE,
  down_buy DOUBLE, down_sell DOUBLE,
  result VARCHAR(8), settled TINYINT(1), logged_utc DATETIME,
  spot DOUBLE, price_to_beat DOUBLE,
  KEY idx_asset_ts (asset, ts_utc)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

VERIFY_SQL = ("SELECT COUNT(*), MIN(id), MAX(id), SUM(CAST(settled AS INT)), "
              "ROUND(SUM(up_buy), 2) FROM {}")


def verify(cur, table):
    cur.execute(VERIFY_SQL.format(table))
    row = cur.fetchone()
    return tuple(float(x) if x is not None else None for x in row)


def main():
    src = pymssql.connect(**MSSQL)
    dst = pymysql.connect(**MARIA)
    sc, dc = src.cursor(), dst.cursor()

    dc.execute(DDL)
    dst.commit()

    dc.execute("SELECT COALESCE(MAX(id), 0) FROM crypto_updown_15m")
    start_id = dc.fetchone()[0]
    print(f"Archiv-Stand: bis id {start_id}")

    sc.execute(f"SELECT {', '.join(COLS)} FROM bb_CryptoUpDown15m "
               f"WHERE id > %s ORDER BY id", (start_id,))
    ins = (f"INSERT IGNORE INTO crypto_updown_15m ({', '.join(COLS)}) "
           f"VALUES ({', '.join(['%s'] * len(COLS))})")
    total = 0
    while True:
        rows = sc.fetchmany(5000)
        if not rows:
            break
        rows = [tuple(int(v) if isinstance(v, bool) else v for v in r) for r in rows]
        dc.executemany(ins, rows)
        dst.commit()
        total += len(rows)
        if total % 50000 < 5000:
            print(f"  {total:,} Zeilen kopiert ...", flush=True)
    print(f"Kopiert: {total:,} neue Zeilen")

    v_src, v_dst = verify(sc, "bb_CryptoUpDown15m"), verify(dc, "crypto_updown_15m")
    print(f"Centron : {v_src}")
    print(f"NAS     : {v_dst}")
    ok = v_src == v_dst
    print("VERIFY:", "OK — identisch" if ok else "FEHLER — Abweichung!")

    if ok and "--drop" in sys.argv:
        sc.execute("DROP TABLE bb_CryptoUpDown15m")
        src.commit()
        print("DROP bb_CryptoUpDown15m auf Centron ausgefuehrt.")
    elif "--drop" in sys.argv:
        print("DROP verweigert (Verify nicht bestanden).")

    src.close()
    dst.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
