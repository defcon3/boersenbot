#!/bin/bash
# Naechtlicher logischer Dump der Postgres-`dbdata` (P1 aus BUNDESLIGA_MIGRATION_PLAN.md).
#
# Warum ein Dump und nicht einfach das Datenverzeichnis: der NAS-Pull um 04:00
# braucht fuer /home/veit rund 31 Minuten, in denen die Datenbank weiterschreibt.
# Ein mitkopiertes PGDATA waere zerrissen und im Ernstfall nicht startfaehig.
# Der Dump ist eine konsistente Datei, fertig VOR dem Backup-Fenster — die
# bestehende Kette (NAS -> CloudSync -> Infomaniak) traegt sie danach ohne eine
# Zeile Aenderung mit.
#
# Ruecksicherung:  gunzip -c dbdata_JJJJ-MM-TT.sql.gz | psql -h 127.0.0.1 -U boersenbot -d dbdata

set -euo pipefail

DUMPDIR=/home/veit/db_dumps
mkdir -p "$DUMPDIR"

OUT="$DUMPDIR/dbdata_$(date +%F).sql.gz"
export PGPASSWORD='Extaler11!'

pg_dump -h 127.0.0.1 -U boersenbot -d dbdata --no-owner --no-privileges \
    | gzip -9 > "$OUT.tmp"
mv "$OUT.tmp" "$OUT"

echo "$(date '+%F %T')  Dump $OUT ($(du -h "$OUT" | cut -f1))"

# Auf dem VPS genuegen sieben Tage: der NAS-rsync laeuft ohne --delete, auf NAS
# und Infomaniak bleibt jeder Stand liegen.
find "$DUMPDIR" -name 'dbdata_*.sql.gz' -mtime +7 -delete
