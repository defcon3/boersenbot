#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MSSQL (Centron `dbdata`) -> PostgreSQL (Contabo) — Erstladung und Wiederholabgleich.

Phase P1 aus BUNDESLIGA_MIGRATION_PLAN.md. **Centron bleibt Master**: dieses
Skript liest dort ausschliesslich und schreibt nur nach Postgres. Es ist
jederzeit wiederholbar — jeder Lauf verwirft den Zielbestand und laedt neu.

    python mssql_to_pg_sync.py --schema           # nur Tabellen/PK/Indizes anlegen
    python mssql_to_pg_sync.py --full             # Schema + alle Daten neu laden
    python mssql_to_pg_sync.py --data             # nur Daten neu laden  <- der Cron
    python mssql_to_pg_sync.py --verify           # Zeilenzahlen beidseitig vergleichen
    python mssql_to_pg_sync.py --data --tables Odds,Matches   # Auswahl statt aller

⚠ Der naechtliche Abgleich laeuft mit **--data**, nicht mit --full: `--schema`
legt jede Tabelle per DROP TABLE ... CASCADE neu an und nimmt dabei die Views
aus `pg_views.sql` mit. Nach einem bewussten --full-Lauf deshalb immer die
Views neu einspielen:
    psql -h 127.0.0.1 -U boersenbot -d dbdata -f deploy/pg_views.sql

Zwei Konventionen, die den ganzen Nachbau tragen:

1. **Alle Bezeichner werden kleingeschrieben.** PostgreSQL faltet unquotete
   Namen ohnehin nach lowercase; damit findet bestehender Code jede
   Schreibweise (`bb_StockPrices`, `BB_STOCKPRICES`) dieselbe Tabelle. Der
   Preis steht in P3: wer Ergebniszeilen als dict ueber den Spaltennamen liest
   (`row['ClosePrice']`), muss auf lowercase umstellen.
2. **Die MSSQL-Schemata werden eingeebnet.** Quelle hat `dbo` und `[326773]`,
   Ziel hat nur `public`. Bei Namenskollision bricht das Skript ab.

Fremdschluessel werden bewusst **nicht** uebernommen — der Nachbau ist bis P4
rein lesend, und ohne FKs bleibt jede Tabelle einzeln neu ladbar. Sie kommen
dazu, wenn die Schreiber umziehen.
"""

import argparse
import sys
import time
from datetime import datetime

import pymssql
import psycopg

MSSQL = dict(server='158.181.48.77', database='dbdata',
             user='326773', password='Extaler11!')
PG = dict(host='127.0.0.1', port=5432, dbname='dbdata',
          user='boersenbot', password='Extaler11!')

# MS-interne Diagrammtabelle, kein Nutzdatenbestand
SKIP_TABLES = {'sysdiagrams'}

BATCH = 5000


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


# ---------------------------------------------------------------- Typabbildung

def pg_type(dt, maxlen, prec, scale):
    """MSSQL-Datentyp -> PostgreSQL-Datentyp. Deckt die 13 real vorkommenden ab,
    kennt aber auch die uebrigen gaengigen, falls die Quelle waechst."""
    dt = dt.lower()
    if dt == 'bigint':
        return 'bigint'
    if dt == 'bit':
        return 'boolean'
    if dt in ('tinyint', 'smallint'):
        return 'smallint'
    if dt == 'int':
        return 'integer'
    if dt == 'float':
        return 'double precision'
    if dt == 'real':
        return 'real'
    if dt in ('decimal', 'numeric'):
        return f'numeric({prec},{scale})'
    if dt == 'money':
        return 'numeric(19,4)'
    if dt == 'date':
        return 'date'
    if dt in ('datetime', 'datetime2', 'smalldatetime'):
        return 'timestamp'
    if dt == 'datetimeoffset':
        return 'timestamptz'
    if dt == 'time':
        return 'time'
    if dt == 'uniqueidentifier':
        return 'uuid'
    if dt in ('varbinary', 'binary', 'image'):
        return 'bytea'
    if dt in ('char', 'nchar'):
        return f'char({maxlen})' if maxlen and maxlen > 0 else 'text'
    if dt in ('varchar', 'nvarchar'):
        # CHARACTER_MAXIMUM_LENGTH ist -1 bei (max)
        return 'text' if not maxlen or maxlen < 0 else f'varchar({maxlen})'
    if dt in ('text', 'ntext', 'xml'):
        return 'text'
    raise ValueError(f'unbekannter MSSQL-Typ: {dt}')


def converter(dt):
    """Wert-Umwandlung dort, wo die Treiber sich nicht einig sind."""
    if dt.lower() == 'bit':
        return lambda v: None if v is None else bool(v)
    return None


# ------------------------------------------------------------ Quelle einlesen

def read_source_schema(cur):
    """Liefert {tabelle: {'schema':…, 'cols':[…], 'identity':[…]}} aus der Quelle."""
    cur.execute("""
        SELECT c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION, c.COLUMN_NAME,
               c.DATA_TYPE, c.CHARACTER_MAXIMUM_LENGTH,
               c.NUMERIC_PRECISION, c.NUMERIC_SCALE, c.IS_NULLABLE,
               COLUMNPROPERTY(OBJECT_ID(QUOTENAME(c.TABLE_SCHEMA) + '.'
                              + QUOTENAME(c.TABLE_NAME)),
                              c.COLUMN_NAME, 'IsIdentity') AS is_identity
        FROM INFORMATION_SCHEMA.COLUMNS c
        JOIN INFORMATION_SCHEMA.TABLES t
          ON t.TABLE_SCHEMA = c.TABLE_SCHEMA
         AND t.TABLE_NAME = c.TABLE_NAME
         AND t.TABLE_TYPE = 'BASE TABLE'
        ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION
    """)
    tables = {}
    seen = {}
    for sch, tbl, _pos, col, dt, clen, prec, scale, nullable, ident in cur.fetchall():
        key = tbl.lower()
        if key in SKIP_TABLES:
            continue
        if key in seen and seen[key] != sch:
            sys.exit(f"ABBRUCH: Tabellenname '{tbl}' existiert in den Schemata "
                     f"'{seen[key]}' und '{sch}' — das Einebnen nach public "
                     f"waere verlustbehaftet.")
        seen[key] = sch
        t = tables.setdefault(key, {'schema': sch, 'src_name': tbl, 'cols': []})
        t['cols'].append({
            'name': col.lower(), 'src_name': col,
            'type': pg_type(dt, clen, prec, scale),
            'raw_type': dt,
            'nullable': nullable == 'YES',
            'identity': bool(ident),
            'conv': converter(dt),
        })
    return tables


def read_source_indexes(cur):
    """PK und Sekundaerindizes der Quelle, nach Zieltabelle gruppiert."""
    cur.execute("""
        SELECT s.name AS sch, t.name AS tbl, i.name AS idx, c.name AS col,
               i.is_primary_key, i.is_unique, ic.key_ordinal
        FROM sys.indexes i
        JOIN sys.tables t   ON t.object_id = i.object_id
        JOIN sys.schemas s  ON s.schema_id = t.schema_id
        JOIN sys.index_columns ic
          ON ic.object_id = i.object_id AND ic.index_id = i.index_id
        JOIN sys.columns c
          ON c.object_id = ic.object_id AND c.column_id = ic.column_id
        WHERE i.type_desc <> 'HEAP' AND ic.is_included_column = 0
        ORDER BY s.name, t.name, i.index_id, ic.key_ordinal
    """)
    idx = {}
    for sch, tbl, name, col, is_pk, is_uq, _ord in cur.fetchall():
        key = tbl.lower()
        if key in SKIP_TABLES:
            continue
        entry = idx.setdefault(key, {})
        e = entry.setdefault(name, {'pk': bool(is_pk), 'unique': bool(is_uq),
                                    'cols': []})
        e['cols'].append(col.lower())
    return idx


# ------------------------------------------------------------- Ziel aufbauen

def create_schema(pg, tables, indexes, only=None):
    created = 0
    with pg.cursor() as cur:
        for tname, t in sorted(tables.items()):
            if only and tname not in only:
                continue
            cols_sql = []
            for c in t['cols']:
                line = f'    {c["name"]} {c["type"]}'
                if c['identity']:
                    line += ' GENERATED BY DEFAULT AS IDENTITY'
                if not c['nullable']:
                    line += ' NOT NULL'
                cols_sql.append(line)
            cur.execute(f'DROP TABLE IF EXISTS {tname} CASCADE')
            cur.execute(f'CREATE TABLE {tname} (\n' + ',\n'.join(cols_sql) + '\n)')
            created += 1

            for name, e in (indexes.get(tname) or {}).items():
                cols = ', '.join(e['cols'])
                # Indexnamen sind in MSSQL je Tabelle eindeutig, in PG global
                iname = f'{tname}_{name}'.lower()[:63]
                if e['pk']:
                    cur.execute(f'ALTER TABLE {tname} ADD CONSTRAINT {iname} '
                                f'PRIMARY KEY ({cols})')
                else:
                    uq = 'UNIQUE ' if e['unique'] else ''
                    cur.execute(f'CREATE {uq}INDEX {iname} ON {tname} ({cols})')
    pg.commit()
    log(f'Schema: {created} Tabellen angelegt (inkl. PK/Indizes)')


# ------------------------------------------------------------- Daten kopieren

def copy_table(src_cur, pg, tname, t):
    cols = t['cols']
    src_cols = ', '.join(f'[{c["src_name"]}]' for c in cols)
    dst_cols = ', '.join(c['name'] for c in cols)
    convs = [c['conv'] for c in cols]
    has_conv = any(convs)

    src_cur.execute(f'SELECT {src_cols} FROM [{t["schema"]}].[{t["src_name"]}]')

    n = 0
    t0 = time.time()
    with pg.cursor() as cur:
        cur.execute(f'TRUNCATE TABLE {tname}')
        with cur.copy(f'COPY {tname} ({dst_cols}) FROM STDIN') as cp:
            while True:
                rows = src_cur.fetchmany(BATCH)
                if not rows:
                    break
                for row in rows:
                    if has_conv:
                        row = tuple(f(v) if f else v for f, v in zip(convs, row))
                    cp.write_row(row)
                n += len(rows)

        # Identity-Sequenzen hinter den geladenen Bestand setzen
        for c in cols:
            if c['identity']:
                cur.execute(
                    f"SELECT setval(pg_get_serial_sequence('{tname}', '{c['name']}'), "
                    f"COALESCE((SELECT MAX({c['name']}) FROM {tname}), 0) + 1, false)")
    pg.commit()
    log(f'  {tname:38s} {n:>10,} Zeilen  ({time.time() - t0:5.1f}s)')
    return n


def load_all(src_cur, pg, tables, only=None):
    loaded = {}
    t0 = time.time()
    for tname, t in sorted(tables.items()):
        if only and tname not in only:
            continue
        loaded[tname] = copy_table(src_cur, pg, tname, t)
    log(f'Ladung fertig: {sum(loaded.values()):,} Zeilen in {time.time() - t0:.0f}s')
    return loaded


# ------------------------------------------------------------------ Abnahme

def verify(src_cur, pg, tables, only=None, loaded=None):
    """Echte COUNT(*) auf beiden Seiten — sys.partitions.rows zaehlt bei
    nvarchar(max) mehrfach und taugt fuer die Abnahme nicht.

    Die Quelle steht waehrenddessen nicht still: der Tile-Logger schreibt im
    Sekundentakt weiter (gemessen 08.08.2026 an `bb_WeatherTileLatency`, 10
    Zeilen Zuwachs zwischen Kopie und Zaehlung). Ein blosser Zahlenvergleich
    meldet das faelschlich als Fehler. Richtig ist:

      * PG hat mehr als MSSQL          -> Fehler, immer
      * PG weicht von dem ab, was      -> Fehler, die Kopie ist unvollstaendig
        dieser Lauf kopiert hat
      * PG hat weniger, aber genau     -> nachgewachsen, kein Fehler
        das Kopierte
    """
    bad, grown = [], []
    loaded = loaded or {}
    with pg.cursor() as cur:
        for tname, t in sorted(tables.items()):
            if only and tname not in only:
                continue
            src_cur.execute(f'SELECT COUNT(*) FROM [{t["schema"]}].[{t["src_name"]}]')
            a = src_cur.fetchone()[0]
            cur.execute(f'SELECT COUNT(*) FROM {tname}')
            b = cur.fetchone()[0]

            if b > a:
                flag, note = 'FEHLER', '  Ziel hat MEHR als die Quelle'
                bad.append(tname)
            elif tname in loaded and b != loaded[tname]:
                flag, note = 'FEHLER', f'  kopiert wurden {loaded[tname]:,}'
                bad.append(tname)
            elif b < a:
                flag, note = 'NACHGEW.', f'  +{a - b} seit der Kopie'
                grown.append(tname)
            else:
                flag, note = 'OK', ''
            log(f'  {flag:10s} {tname:38s} MSSQL {a:>10,}   PG {b:>10,}{note}')

    if grown:
        log(f'{len(grown)} Tabelle(n) in der Quelle nachgewachsen (unkritisch): '
            + ', '.join(grown))
    if bad:
        log(f'FEHLER: {len(bad)} Tabelle(n) fehlerhaft: ' + ', '.join(bad))
        return False
    log('Abnahme bestanden.')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--schema', action='store_true', help='Tabellen/PK/Indizes anlegen')
    ap.add_argument('--full', action='store_true', help='Schema + alle Daten neu laden')
    ap.add_argument('--data', action='store_true', help='nur Daten neu laden')
    ap.add_argument('--verify', action='store_true', help='Zeilenzahlen vergleichen')
    ap.add_argument('--tables', help='Kommaliste statt aller Tabellen')
    args = ap.parse_args()

    if not any([args.schema, args.full, args.data, args.verify]):
        ap.error('nichts zu tun — --schema, --full, --data oder --verify waehlen')

    only = None
    if args.tables:
        only = {t.strip().lower() for t in args.tables.split(',') if t.strip()}

    src = pymssql.connect(**MSSQL)
    src_cur = src.cursor()
    tables = read_source_schema(src_cur)
    indexes = read_source_indexes(src_cur)
    log(f'Quelle: {len(tables)} Tabellen, '
        f'{sum(len(t["cols"]) for t in tables.values())} Spalten')

    if only:
        unknown = only - set(tables)
        if unknown:
            sys.exit(f'unbekannte Tabellen: {", ".join(sorted(unknown))}')

    pg = psycopg.connect(**PG)

    ok = True
    loaded = None
    if args.schema or args.full:
        create_schema(pg, tables, indexes, only)
    if args.full or args.data:
        loaded = load_all(src_cur, pg, tables, only)
    if args.verify or args.full or args.data:
        ok = verify(src_cur, pg, tables, only, loaded)

    pg.close()
    src.close()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
