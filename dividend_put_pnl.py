"""
EHRLICHER FORWARD-TRACKER — Put-Optionsschein-Bilanz rund um Ex-Dividenden-Tage.
================================================================================
Liest die vom dividend_tracker.py gesammelten Snapshots (dividend_tracker.db) und
rechnet fuer JEDEN Put eine realistische P&L ueber sein Ex-Fenster ab — ALLE,
nicht nur die optisch schoenen. Antwort auf die Chart-Illusion "die Puts haben
doch gewonnen": hier steht die volle Bilanz inkl. Verlierer + Spread.

Regel (eingefroren, fuer alle gleich):
  ENTRY = letzter Snapshot VOR dem Ex-Tag, gekauft zum BRIEF (ask)
  EXIT  = Snapshot an mehreren Horizonten (Ex-Schluss / +1 / +3 / letzter), zum GELD (bid)
  -> realistische, spread-inkl. Rendite. Zusaetzlich mid->mid als reibungsfreie Obergrenze.
Kein Cherry-Picking: abrechenbar = hat >=1 Snap vor UND >=1 ab Ex-Tag. Rest = "offen".

Aufruf:  python3 dividend_put_pnl.py [--mail]
"""
import argparse
import datetime as dt
import sqlite3
import statistics as st
from pathlib import Path

DB = Path(__file__).parent / "dividend_tracker.db"
HORIZONS = [("Ex-Schluss", 0), ("T+1", 1), ("T+3", 3)]  # Index in post-ex-Snapshotliste


def _fill(snap, side):
    """side='buy' -> ask (sonst mid); side='sell' -> bid (sonst mid)."""
    bid, ask, mid = snap["bid"], snap["ask"], snap["mid"]
    if side == "buy":
        return ask or mid or bid
    return bid or mid or ask


def collect():
    con = sqlite3.connect(str(DB)); con.row_factory = sqlite3.Row
    cur = con.cursor()
    warrants = cur.execute("""
        SELECT w.warrant_id, w.wkn, w.issuer, w.strike, w.event_id,
               e.ticker, e.name, e.ex_date, e.gross_div
        FROM warrants w JOIN events e ON e.event_id = w.event_id
    """).fetchall()
    rows = []
    for w in warrants:
        ex = dt.date.fromisoformat(w["ex_date"])
        snaps = [dict(s) for s in cur.execute("""
            SELECT snapshot_date, bid, ask, mid FROM warrant_snapshots
            WHERE warrant_id=? ORDER BY snapshot_date""", (w["warrant_id"],)).fetchall()]
        pre = [s for s in snaps if dt.date.fromisoformat(s["snapshot_date"]) < ex]
        post = [s for s in snaps if dt.date.fromisoformat(s["snapshot_date"]) >= ex]
        # Aktien-Bewegung ueber dasselbe Fenster (entlarvt den Confounder)
        st_snaps = [dict(s) for s in cur.execute("""
            SELECT snapshot_date, stock_close FROM stock_snapshots
            WHERE event_id=? ORDER BY snapshot_date""", (w["event_id"],)).fetchall()]
        sp = [s for s in st_snaps if dt.date.fromisoformat(s["snapshot_date"]) < ex]
        sq = [s for s in st_snaps if dt.date.fromisoformat(s["snapshot_date"]) >= ex]
        stock_ret = None
        if sp and sq and sp[-1]["stock_close"]:
            stock_ret = (sq[-1]["stock_close"] - sp[-1]["stock_close"]) / sp[-1]["stock_close"] * 100.0
        rows.append(dict(w=w, ex=ex, pre=pre, post=post, stock_ret=stock_ret))
    con.close()
    return rows


def pnl(entry_snap, exit_snap, kind):
    if kind == "real":
        e = _fill(entry_snap, "buy"); x = _fill(exit_snap, "sell")
    else:  # mid->mid
        e = entry_snap["mid"] or (((entry_snap["bid"] or 0)+(entry_snap["ask"] or 0))/2 or None)
        x = exit_snap["mid"] or (((exit_snap["bid"] or 0)+(exit_snap["ask"] or 0))/2 or None)
    if not e or not x or e <= 0:
        return None
    return (x - e) / e * 100.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mail", action="store_true")
    args = ap.parse_args()
    rows = collect()
    abrechenbar = [r for r in rows if r["pre"] and r["post"]]
    offen = [r for r in rows if not (r["pre"] and r["post"])]

    lines = []
    lines.append(f"EHRLICHE PUT-BILANZ (Stand {dt.date.today()})")
    lines.append(f"  Puts gesamt {len(rows)} | abrechenbar (Snap vor+nach Ex) {len(abrechenbar)} | "
                 f"noch offen {len(offen)}")
    lines.append("")
    # Tabelle: realistische P&L zum letzten verfuegbaren Exit + Aktien-Bewegung
    header = f"  {'Ticker':9s} {'WKN':8s} {'ExTag':10s} | {'Aktie%':>7s} {'Put real%':>9s} {'mid%':>7s}"
    lines.append(header); lines.append("  " + "-"*len(header))
    agg = {"real": [], "mid": []}; kein_entry = 0
    for r in sorted(abrechenbar, key=lambda r: r["ex"]):
        entry = r["pre"][-1]; exit_ = r["post"][-1]
        pr = pnl(entry, exit_, "real"); pm = pnl(entry, exit_, "mid")
        if pr is None:
            kein_entry += 1
            continue   # kein gueltiger Einstiegskurs -> ehrlich raus, nicht als 0 zaehlen
        agg["real"].append(pr)
        if pm is not None: agg["mid"].append(pm)
        sr = r["stock_ret"]
        lines.append(f"  {r['w']['ticker']:9s} {str(r['w']['wkn']):8s} {r['ex'].isoformat():10s} | "
                     f"{(sr if sr is not None else float('nan')):7.1f} "
                     f"{pr:9.1f} {(pm if pm is not None else float('nan')):7.1f}")

    def summ(name, xs):
        if not xs:
            return f"  {name}: keine Daten"
        win = sum(1 for v in xs if v > 0)
        return (f"  {name}: n={len(xs)}  Mittel {st.mean(xs):+.1f}%  Median {st.median(xs):+.1f}%  "
                f"Trefferquote {win}/{len(xs)} = {win/len(xs)*100:.0f}%  "
                f"best {max(xs):+.0f}% / schlecht {min(xs):+.0f}%")
    lines.append("")
    lines.append("  === GESAMTBILANZ (Kauf Brief -> Verkauf Geld, letzter Exit) ===")
    lines.append(summ("REALISTISCH (mit Spread)", agg["real"]))
    lines.append(summ("mid->mid (reibungsfrei) ", agg["mid"]))
    if agg["real"] and agg["mid"]:
        lines.append(f"  -> Spread frisst im Schnitt {st.mean(agg['mid'])-st.mean(agg['real']):.1f} Prozentpunkte.")
    lines.append("")
    lines.append(f"  Hinweis: {len(offen)} Puts noch offen; {kein_entry} ohne gueltigen Einstiegskurs (raus).")
    lines.append("  LIES DIE 'Aktie%'-SPALTE: Put gewinnt ~nur, wenn die Aktie faellt -- das ist der")
    lines.append("  Kursverlust der Aktie (oft idiosynkratisch), NICHT ein Dividenden-Edge. n noch winzig.")
    lines.append("  Theorie-Erwartung (veitluther.de/done #12): ~-2%/Trade nach Spread+Theta+Vol-Crush.")
    out = "\n".join(lines)
    print(out)

    if args.mail and abrechenbar:
        try:
            from dividend_tracker import send_mail, setup_logging
            html = "<pre style='font-family:monospace;font-size:13px'>" + out.replace("<","&lt;") + "</pre>"
            send_mail(f"Ehrliche Put-Bilanz ({len(abrechenbar)} abgerechnet)", html, setup_logging())
        except Exception as e:
            print(f"[mail err] {e}")


if __name__ == "__main__":
    main()
