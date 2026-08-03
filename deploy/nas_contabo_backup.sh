#!/bin/bash
# nas_contabo_backup.sh — laeuft auf dem Synology-NAS (192.168.178.32), taeglich per Cron.
#
# Sichert den Contabo-VPS (144.91.98.234) auf das NAS. Das NAS ZIEHT, weil es im
# LAN steht und der VPS nicht hineinreicht.
#
# Ziel liegt im NAS-Home und damit in der CloudSync-Session 10
# (homes/benutzername -> Infomaniak kDrive /DiskStation_backup/home).
#
# Semantik bewusst ADDITIV: kein --delete. Was auf dem VPS geloescht wird, bleibt
# auf NAS und in der Cloud liegen.
#
# Der verwendete SSH-Key ist auf dem VPS per authorized_keys an
# `rrsync -ro /home/veit` gebunden: nur Lesen, keine Shell.
#
# Installiert nach: /var/services/homes/benutzername/boersenbot/nas_contabo_backup.sh

set -u

SRC_HOST=veit@144.91.98.234
SSH_KEY=/var/services/homes/benutzername/.ssh/vps_backup
DEST=/var/services/homes/benutzername/contabo_backup
LOG=/var/services/homes/benutzername/boersenbot/contabo_backup.log
LOCK=/tmp/contabo_backup.lock

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# --- Nur ein Lauf gleichzeitig (erster Vollzug dauert lange) -----------------
exec 9>"$LOCK"
if ! flock -n 9; then
    log "SKIP: vorheriger Lauf laeuft noch"
    exit 0
fi

mkdir -p "$DEST/home"

log "=== Start ==="
START=$(date +%s)

# rrsync sperrt uns auf /home/veit ein -> Quellpfad ist dort "/"
rsync -rlpt \
      --no-owner --no-group \
      --partial \
      --human-readable \
      --stats \
      --timeout=1800 \
      --exclude='.gunicorn/' \
      -e "ssh -i $SSH_KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -c chacha20-poly1305@openssh.com" \
      "$SRC_HOST:/" "$DEST/home/" >> "$LOG" 2>&1
RC=$?

DUR=$(( $(date +%s) - START ))

if [ $RC -eq 0 ] || [ $RC -eq 24 ]; then
    # 24 = "vanished source files" — normal, waehrend der VPS weiterarbeitet
    date -Iseconds > "$DEST/LAST_SUCCESS.txt"
    du -sh "$DEST/home" 2>/dev/null >> "$DEST/LAST_SUCCESS.txt"
    log "OK (rc=$RC) nach ${DUR}s"
else
    log "FEHLER rc=$RC nach ${DUR}s"
fi

# --- Logdatei begrenzen ------------------------------------------------------
if [ "$(wc -c < "$LOG")" -gt 5000000 ]; then
    tail -n 2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi

exit $RC
