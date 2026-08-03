#!/bin/bash
# collect_system_config.sh — laeuft auf dem VPS (144.91.98.234), taeglich per Cron.
#
# Zweck: alles, was NICHT unter /home/veit liegt, aber bei einem Totalverlust des
# VPS fehlen wuerde, in /home/veit/_backup_meta/ spiegeln — damit der NAS-Job
# (der nur /home/veit lesen darf) es mitzieht.
#
# Laeuft bewusst OHNE sudo: systemd-Units, nginx-Configs und die eigene Crontab
# sind fuer veit lesbar. Was root-only ist, fehlt hier absichtlich.

set -u

META=/home/veit/_backup_meta
mkdir -p "$META"/{systemd,nginx,cron,system}

# --- systemd: Units + Aktivierungsstatus -------------------------------------
rm -f "$META"/systemd/*.service "$META"/systemd/*.timer
cp -p /etc/systemd/system/boersenbot_*.service "$META"/systemd/ 2>/dev/null
cp -p /etc/systemd/system/boersenbot_*.timer   "$META"/systemd/ 2>/dev/null
cp -p /etc/systemd/system/intraday*.service    "$META"/systemd/ 2>/dev/null
cp -p /etc/systemd/system/intraday*.timer      "$META"/systemd/ 2>/dev/null
systemctl list-unit-files --no-pager --no-legend > "$META"/systemd/unit_files.txt 2>/dev/null
systemctl list-units --type=service --state=running --no-pager --no-legend \
    > "$META"/systemd/running_services.txt 2>/dev/null

# --- nginx: komplette Konfiguration ------------------------------------------
rm -rf "$META"/nginx
mkdir -p "$META"/nginx
cp -rp /etc/nginx/nginx.conf /etc/nginx/sites-available /etc/nginx/conf.d "$META"/nginx/ 2>/dev/null
ls -l /etc/nginx/sites-enabled > "$META"/nginx/sites-enabled.txt 2>/dev/null
# Zertifikate selbst kommen nicht mit (root-only) — Certbot holt sie neu.
ls -l /etc/letsencrypt/live 2>/dev/null > "$META"/nginx/letsencrypt_live.txt

# --- Cron ---------------------------------------------------------------------
crontab -l > "$META"/cron/crontab_veit.txt 2>/dev/null
ls -l /etc/cron.d /etc/cron.daily > "$META"/cron/cron_dirs.txt 2>/dev/null

# --- System-Inventar fuer den Wiederaufbau ------------------------------------
dpkg --get-selections            > "$META"/system/dpkg_selections.txt 2>/dev/null
lsb_release -a                   > "$META"/system/os_release.txt 2>&1
uname -a                        >> "$META"/system/os_release.txt 2>&1
ip -brief addr                   > "$META"/system/ip_addr.txt 2>/dev/null
df -h                            > "$META"/system/disk_usage.txt 2>/dev/null
for v in /home/veit/boersenbot/venv /home/veit/intraday-bot/venv; do
    [ -x "$v/bin/pip" ] && "$v/bin/pip" freeze \
        > "$META"/system/pip_freeze_$(basename "$(dirname "$v")").txt 2>/dev/null
done

date -Iseconds > "$META"/COLLECTED_AT.txt
echo "collect_system_config.sh fertig: $(date -Iseconds)"
