#!/usr/bin/env python3
"""
weather_minus1_autobuy.py — Autonomer Live-Test der −1-Lay-Klasse.

Pre-Reg: preregs/weather_minus1_live_2026_07_20.md (Klasse-B-Basis: −1-Klasse
+3,35 % netto, t 7,9 — preregs/weather_classb_lay_2026_07_18.md).

Regel (täglich, VPS-Timer 12:45 UTC, direkt nach dem 12:30-Ladder-Snapshot):
  1. Kandidaten = heutiger bb_WeatherLadders-Snapshot mit var='max', kind='eq',
     offset_fav=-1, status='open', target_date=morgen (Lead 1, identische
     µ-Definition wie die Klasse-B-Messung).
  2. Live-Preis-Recheck je Markt; handelbar wenn open und buyNo <= 0.975
     (Mindestrendite 2,5 %). Märkte mit bestehender Position werden
     übersprungen (Idempotenz + keine Kollision mit manuellen Wetten).
  3. Auswahl: die 3 KONSERVATIVSTEN (höchster Live-NO-Preis) — Nutzer-Entscheid
     19.07., bewusste Abweichung vom klassenreinen "alle"-Test.
  4. Kauf 5 $ NO je Markt (Jupiter-Minimum), Limit = Live-Ask + 0,005
     (Cap 0.975). Max 2 Sendeversuche, KEIN Nachrücker bei Fehlschlag.
  5. Halten bis Settlement — kein TP (TP-Lehre 14.07.), Claims macht der
     bestehende VPS-Autopilot.
  6. Log: preregs/weather_minus1_live_log.csv (führende Datei auf dem VPS).

Guards: Zeitfenster 12:30–14:30 UTC (kein Nachhol-Lauf zu anderem Lead),
Tages-Idempotenz übers Log, harter Fehler wenn heute kein Ladder-Snapshot da.
"""

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pymssql
import requests

from jupiter_buy import place
from jupiter_sell import API as JUP_API, load_keypair
from weather_ladder_logger import DB_CONFIG

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

PM_API = "https://prediction-market-api.jup.ag/api"
LOG_CSV = Path("preregs/weather_minus1_live_log.csv")
LOG_FIELDS = ["run_utc", "target_date", "city", "k", "mu_ens", "buy_no_snap",
              "buy_no_live", "decision", "usd", "contracts", "avg_price", "signature"]
MAX_NO = 0.97           # Mindestrendite ~3 % (Tick-Size der Maerkte: ganze Cents)
CAP_DEFAULT = 6         # die N konservativsten (Nutzer-Entscheid 20.07.: 3 -> 6;
                        #   Cap-Wechsel gilt ab Zieltag 22.07., Review 27.07. beachten)
USD_DEFAULT = 5.0       # Jupiter-Minimum


def get_json(url, params=None, tries=4):
    r = None
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=20)
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"GET {url} scheiterte ({r.status_code if r is not None else 'conn'})")


def load_candidates(target_day):
    """Heutiger 12:30-Snapshot: alle offenen −1-Fenster (max/eq) des Zieltags."""
    today_1200 = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0,
                                                    microsecond=0, tzinfo=None)
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor(as_dict=True)
    cur.execute("SELECT MAX(snapshot_utc) AS snap FROM bb_WeatherLadders WHERE snapshot_utc >= %s",
                (today_1200,))
    snap = cur.fetchone()["snap"]
    if snap is None:
        conn.close()
        raise RuntimeError("Kein Ladder-Snapshot von heute >=12:00 UTC — Logger-Lauf pruefen!")
    cur.execute(
        "SELECT city, k, market_id, buy_yes, buy_no, mu_ens FROM bb_WeatherLadders "
        "WHERE snapshot_utc = %s AND target_date = %s AND var = 'max' AND kind = 'eq' "
        "AND offset_fav = -1 AND status = 'open' ORDER BY city",
        (snap, target_day))
    rows = cur.fetchall()
    conn.close()
    print(f"Snapshot {snap} UTC: {len(rows)} −1-Kandidaten fuer {target_day}.")
    return rows


def owned_market_ids(owner):
    j = get_json(f"{JUP_API}/positions", {"ownerPubkey": owner})
    return {p.get("marketId") for p in j.get("data", [])
            if str(p.get("contracts", "0")) not in ("0", "")}


def live_market(mid):
    m = get_json(f"{PM_API}/v1/markets/{mid}")
    pr = m.get("pricing") or {}
    return m.get("status"), (pr.get("buyNoPriceUsd") or 0) / 1e6


def verify_fill(owner, mid, tries=3):
    for _ in range(tries):
        time.sleep(5)
        j = get_json(f"{JUP_API}/positions", {"ownerPubkey": owner})
        for p in j.get("data", []):
            if p.get("marketId") == mid and str(p.get("contracts", "0")) not in ("0", ""):
                return p.get("contractsDecimal"), int(p.get("avgPriceUsd", 0)) / 1e6
    return None, None


def final_fills(owner, mids):
    """Fills der gekauften Maerkte final abfragen (marketId -> (contracts, avg, kosten)).

    verify_fill() im Kaufloop wartet nur ~15 s und lief deshalb an drei Tagen in
    Folge in 'sent_unverified', obwohl alle Kaeufe echt gefuellt waren — Jupiter
    materialisiert die Position traege. Hier, am Ende des Laufs, sind laengst
    Minuten vergangen, die Zahlen stimmen also fuer die Mail."""
    out = {}
    try:
        j = get_json(f"{JUP_API}/positions", {"ownerPubkey": owner})
        for p in j.get("data", []):
            if p.get("marketId") in mids:
                out[p["marketId"]] = (float(p.get("contractsDecimal") or 0),
                                      int(p.get("avgPriceUsd", 0)) / 1e6,
                                      int(p.get("totalCostUsd", 0)) / 1e6)
    except Exception as e:
        print(f"  (Fill-Nachfrage fuer die Mail fehlgeschlagen: {e})")
    return out


def send_summary_mail(run_utc, target, cap, n_cands, bought, failed):
    """Mail nach dem Lauf: was wurde gesetzt. Fehler brechen den Lauf NICHT ab."""
    try:
        from autopilot import notify
    except Exception as e:
        print(f"  (Mail-Import fehlgeschlagen: {e})")
        return
    if bought:
        rows = "".join(
            f'<tr><td style="padding:7px 10px;border-bottom:1px solid #eee;">'
            f'<b>{c}</b> {k}°C NO</td>'
            f'<td style="padding:7px 10px;border-bottom:1px solid #eee;text-align:right;">'
            f'{ct:.2f} @ {av:.3f}</td>'
            f'<td style="padding:7px 10px;border-bottom:1px solid #eee;text-align:right;">'
            f'{ko:.2f} $</td></tr>'
            for c, k, ct, av, ko in bought)
        einsatz = sum(b[4] for b in bought)
        gewinn = sum(b[2] - b[4] for b in bought)
        rendite = f" (+{gewinn / einsatz * 100:.1f} %)" if einsatz else ""
        kopf = f"🤖 −1-Autobuy: {len(bought)} Lay{'s' if len(bought) != 1 else ''} gesetzt"
        body = (f'<table style="width:100%;border-collapse:collapse;font-size:14px;">{rows}</table>'
                f'<br>Einsatz <b>{einsatz:.2f} $</b> · bei vollem Durchlauf '
                f'<b>+{gewinn:.2f} $</b>{rendite} · Settlement am {target}.')
        text = ("".join(f"  {c} {k}°C NO — {ct:.2f} Kontr. @ {av:.3f} = {ko:.2f} $\n"
                        for c, k, ct, av, ko in bought)
                + f"\nEinsatz {einsatz:.2f} $ | bei vollem Durchlauf +{gewinn:.2f} $"
                  f"{rendite} | Settlement am {target}.")
    else:
        kopf = "🤖 −1-Autobuy: nichts gesetzt"
        body = ("Es wurde <b>kein</b> Markt gekauft — kein Kandidat hat die Kriterien "
                "erfuellt oder alle waren schon im Bestand.")
        text = "Es wurde KEIN Markt gekauft (kein Kandidat qualifiziert / schon im Bestand)."
    fehl = (f'<br><br><span style="color:#b71c1c;">Fehlgeschlagen: {", ".join(failed)}</span>'
            if failed else "")
    html = (f'<div style="font-family:Segoe UI,Arial;max-width:520px;margin:auto;">'
            f'<div style="background:#1565c0;color:#fff;padding:18px;text-align:center;'
            f'border-radius:10px 10px 0 0;font-size:18px;font-weight:800;">{kopf}</div>'
            f'<div style="padding:18px;background:#fff;color:#333;font-size:15px;line-height:1.6;">'
            f'Lauf <b>{run_utc} UTC</b> · Zieltag <b>{target}</b> · '
            f'{n_cands} Kandidaten handelbar, Cap {cap}.<br><br>{body}{fehl}</div></div>')
    notify(f"{kopf} — Zieltag {target}", html,
           f"−1-AUTOBUY {run_utc} UTC | Zieltag {target}\n"
           f"{n_cands} Kandidaten handelbar, Cap {cap}.\n\n{text}"
           + (f"\nFehlgeschlagen: {', '.join(failed)}" if failed else ""))


def append_log(rows):
    new = not LOG_CSV.exists()
    LOG_CSV.parent.mkdir(exist_ok=True)
    with LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def bought_today():
    if not LOG_CSV.exists():
        return 0
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with LOG_CSV.open(encoding="utf-8") as f:
        return sum(1 for r in csv.DictReader(f)
                   if r["run_utc"].startswith(today) and r["decision"] == "bought")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="alles ausser /execute")
    ap.add_argument("--target", help="Zieltag YYYY-MM-DD (Default: morgen UTC)")
    ap.add_argument("--cap", type=int, default=CAP_DEFAULT)
    ap.add_argument("--usd", type=float, default=USD_DEFAULT)
    ap.add_argument("--force-window", action="store_true",
                    help="Zeitfenster-Guard uebergehen (nur fuer Tests)")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    target = args.target or (now + timedelta(days=1)).strftime("%Y-%m-%d")
    run_utc = now.strftime("%Y-%m-%d %H:%M")
    print(f"=== −1-Autobuy {run_utc} UTC | Zieltag {target} | cap {args.cap} | "
          f"{'DRY-RUN' if args.dry_run else 'ECHT'} ===")

    if not args.force_window and not (12 * 60 + 30 <= now.hour * 60 + now.minute <= 14 * 60 + 30):
        print("Ausserhalb des 12:30–14:30-UTC-Fensters — Skip (Guard gegen Nachhol-Laeufe).")
        return 0

    n_prev = bought_today()
    if n_prev:
        print(f"Heute bereits {n_prev} Kaeufe im Log — Skip (Doppellauf-Schutz).")
        return 0

    kp = load_keypair()
    owner = str(kp.pubkey())
    cands = load_candidates(target)
    owned = owned_market_ids(owner)

    log_rows, tradeable = [], []
    for c in cands:
        base = {"run_utc": run_utc, "target_date": target, "city": c["city"], "k": c["k"],
                "mu_ens": round(c["mu_ens"], 2) if c["mu_ens"] is not None else "",
                "buy_no_snap": c["buy_no"], "usd": "", "contracts": "", "avg_price": "",
                "signature": ""}
        if c["market_id"] in owned:
            log_rows.append({**base, "buy_no_live": "", "decision": "skip_position"})
            continue
        try:
            status, no_live = live_market(c["market_id"])
        except RuntimeError as e:
            log_rows.append({**base, "buy_no_live": "", "decision": f"skip_api_{e}"})
            continue
        base["buy_no_live"] = round(no_live, 3)
        if status != "open" or no_live <= 0:
            log_rows.append({**base, "decision": "skip_closed"})
        elif no_live > MAX_NO:
            log_rows.append({**base, "decision": "skip_price"})
        else:
            tradeable.append((no_live, c, base))
        time.sleep(0.4)

    tradeable.sort(key=lambda x: -x[0])  # konservativste = hoechster NO-Preis
    picks, rest = tradeable[:args.cap], tradeable[args.cap:]
    log_rows += [{**b, "decision": "skip_cap"} for _, _, b in rest]
    picks_txt = ", ".join(f"{c['city']} {c['k']}° NO@{no:.3f}" for no, c, _ in picks) or "—"
    print(f"{len(tradeable)} handelbar, {len(picks)} werden gesetzt ({picks_txt})")

    n_ok = 0
    gekauft, fehlgeschlagen = [], []   # fuer die Abschluss-Mail
    for no_live, c, base in picks:
        # Tick-Size 1 Cent: Limit = naechster ganzer Cent ueber dem Ask (Fill-Puffer)
        limit = min(round((int(no_live * 100) + 1) / 100, 2), MAX_NO)
        print(f"\nKauf {c['city']} {c['k']}°C NO {args.usd}$ @ limit {limit} ({c['market_id']}):")
        result = {"ok": False}
        for attempt in (1, 2):
            try:
                result = place(owner, c["market_id"], False, args.usd, limit, kp,
                               send=not args.dry_run)
            except Exception as e:
                print(f"  Versuch {attempt} Exception: {e}")
                result = {"ok": False}
            if result.get("ok"):
                break
            if attempt == 1:
                time.sleep(10)
        if args.dry_run:
            log_rows.append({**base, "decision": "dry_run", "usd": args.usd})
            continue
        if not result.get("ok"):
            log_rows.append({**base, "decision": "fail_send", "usd": args.usd})
            fehlgeschlagen.append(f"{c['city']} {c['k']}°C")
            continue
        gekauft.append(c)
        sig = (result.get("resp") or {}).get("signature", "")
        contracts, avg = verify_fill(owner, c["market_id"])
        decision = "bought" if contracts else "sent_unverified"
        n_ok += 1 if contracts else 0
        log_rows.append({**base, "decision": decision, "usd": args.usd,
                         "contracts": contracts or "", "avg_price": avg or "",
                         "signature": sig})
        print(f"  -> {decision}: {contracts} Kontr. @ {avg}")

    append_log(log_rows)
    print(f"\nFertig: {n_ok} Kaeufe bestaetigt, {len(log_rows)} Zeilen geloggt -> {LOG_CSV}")

    if not args.dry_run and (gekauft or fehlgeschlagen):
        fills = final_fills(owner, {c["market_id"] for c in gekauft})
        bought = []
        for c in gekauft:
            ct, av, ko = fills.get(c["market_id"], (0.0, 0.0, args.usd))
            bought.append((c["city"], c["k"], ct, av, ko))
        send_summary_mail(run_utc, target, args.cap, len(tradeable), bought, fehlgeschlagen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
