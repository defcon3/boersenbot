#!/usr/bin/env python3
"""
weather_minus1_review.py — Review der −1-Autobuy-Woche, nach REGIMES getrennt.

Termin laut Pre-Reg `preregs/weather_minus1_live_2026_07_20.md`: 27.07.2026.
Kauf-Laeufe 20.–26.07. (Zieltage 21.–27.07.), erwartetes N ~21 — Indikation,
kein Gate-Anspruch.

WARUM REGIMES: Die Reihe ist NICHT homogen. Waehrend der Woche wurde die
Auswahlregel zweimal geaendert, jede Aenderung macht den Bot weniger
konservativ oder wieder strenger. Zusammengemischt vergleicht man zwei
verschiedene Bots und haelt das Ergebnis fuer Rauschen (oder fuer Edge).

  R1  Zieltag 21.07.        Cap 3                      (Ausgangsregel)
  R2  Zieltag 22.07.        Cap 6, ungefiltert         (ffb1ddbb)
  R3  Zieltag 23.–26.07.    Cap 6 + Guete-Gate 0,85    (b4c066f)
  R4  ab Zieltag 27.07.     + Spannen-Veto             (eb2dbeff) — noch offen

DATENQUELLEN
  preregs/weather_minus1_live_log.csv  — fuehrend auf dem VPS. Liefert WELCHE
      Kandidaten der Bot gekauft hat (decision + Signatur). Lokal per scp holen,
      Pfad ueber --log.
  Jupiter /v2/history                  — liefert den REALISIERTEN PnL inkl.
      Gebuehren je Position. Besser als Nachrechnen: keine Annahme ueber Fill,
      Fee-Satz oder Payout noetig.
  bb_WeatherLadders                    — Settlement (wu_settle_k als Wahrheit,
      settle_k/METAR nur als Fallback) und der Snapshot-Preis fuer die Slippage.

JOIN: ueber die Transaktions-Signatur aus dem CSV gegen events[].signature der
History. Das ist eindeutig und noetig, weil in derselben Wallet auch MANUELLE
Wetten liegen — ein Join ueber Stadt/Datum wuerde die mit einsammeln.

MERKE: `sent_unverified` heisst NICHT „nicht gekauft" — verify_fill() wartet nur
~15 s, Jupiter materialisiert die Position traeger. Diese Zeilen zaehlen als
Kauf; die History belegt es je Signatur.

Aufruf:
  python weather_minus1_review.py --log <pfad/live_log.csv>
  python weather_minus1_review.py --log ... --bis 2026-07-27   # wenn der 27. gesettelt ist
"""

import argparse
import csv
import re
import sys
from collections import defaultdict

import pymssql

from jupiter_wallet_scout import hole_history
from weather_ladder_logger import DB_CONFIG

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

OWNER = "4XxStoKPzoiEJ6hUGEESfE54dCRo97LcCGk2UFieKjSi"
KAUF = ("bought", "sent_unverified")

# KORREKTUR 27.07. abends: R3 begann NICHT am 23.07. Das Guete-Gate (b4c066f)
# wurde am 22.07. erst NACH dem 12:45-Lauf deployt — der Kauf fuer Zieltag 23.07.
# lief also noch ungefiltert. Beleg im Log: an dem Tag gingen Helsinki 0,83,
# Ankara 0,81 und Cape Town 0,71 als Rang 4-6 durch, die das Gate alle
# abgelehnt haette. Ab Zieltag 24.07. steht dort sauber skip_quality.
# Der Trockentest vom 22.07. ("haette 3 statt 6 gesetzt") lief gegen genau diese
# Kandidatenliste — er beschrieb die Zukunft, nicht den gefahrenen Lauf.
REGIMES = [
    ("R1  Cap 3",                      "2026-07-21", "2026-07-21"),
    ("R2  Cap 6, ungefiltert",         "2026-07-22", "2026-07-23"),
    ("R3  Cap 6 + Guete-Gate 0,85",    "2026-07-24", "2026-07-26"),
    ("R4  + Spannen-Veto",             "2026-07-27", "2026-07-27"),
    ("R5  V2 Preisband 0,70-0,90",     "2026-07-28", "2026-12-31"),
]


def lade_log(pfad):
    with open(pfad, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["decision"] in KAUF]


def history_nach_signatur():
    """signature -> Positions-Eintrag. Alle Events einer Position zeigen auf sie."""
    idx = {}
    for pos in hole_history(OWNER, 300):
        for ev in pos.get("events") or []:
            if ev.get("signature"):
                idx[ev["signature"]] = pos
    return idx


def lade_settlement(von, bis):
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT target_date, city, k, settle_k, wu_settle_k, settle_result "
        "FROM bb_WeatherLadders WHERE var='max' AND kind='eq' "
        "AND target_date BETWEEN %s AND %s", (von, bis))
    out = {}
    for td, city, k, sk, wu, res in cur.fetchall():
        out[(str(td), city, int(k))] = (sk, wu, res)
    conn.close()
    return out


def usd(mikro):
    return None if mikro in (None, "") else int(mikro) / 1e6


def kauf_fee(pos):
    """Nur die KAUF-Gebuehr. feesPaidUsd enthaelt bei verkauften Positionen auch
    die Verkaufs-Fee — fuer die Halte-Rechnung waere das zuviel abgezogen."""
    f = 0.0
    for ev in pos.get("events") or []:
        if ev.get("eventType") == "order_filled" and ev.get("isBuy"):
            f += usd(ev.get("feeUsd")) or 0.0
    return f


def auswerten_pos(r, pos):
    """Eine Position zu einem Datensatz verdichten.

    getroffen: hat der gelayte Bucket eingetroffen? Quelle ist Jupiters eigenes
      `result` — also exakt das, wogegen die Position abgerechnet wurde.
    pnl_ist:   realisiert, so wie es gelaufen ist (inkl. manueller Cuts).
    pnl_hold:  kontrafaktisch bis Settlement gehalten = die Pre-Reg-Regel.
      Beide sind identisch, solange niemand eingegriffen hat.
    """
    res = ((pos.get("marketMetadata") or {}).get("result") or "").lower()
    if res not in ("yes", "no"):
        return None, f"Markt noch nicht aufgeloest (result={res or 'None'})"
    getroffen = (res == "yes")

    entry = usd(pos.get("entryPriceUsd"))
    ktr = float(pos.get("totalContractsDecimal") or 0)
    kosten = entry * ktr
    fee_b = kauf_fee(pos)
    pnl_hold = (0.0 if getroffen else ktr) - kosten - fee_b

    rp = usd(pos.get("realizedPnlUsd"))
    verkauft = pos.get("status") == "sold"
    if rp is None:
        if verkauft:
            return None, "verkauft, aber realizedPnlUsd fehlt"
        rp = pnl_hold

    snap = float(r["buy_no_snap"] or 0) or None
    return dict(entry=entry, ktr=ktr, kosten=kosten, getroffen=getroffen,
                pnl_ist=rp, pnl_hold=pnl_hold, verkauft=verkauft,
                fee=usd(pos.get("feesPaidUsd")) or 0.0,
                slip=(entry - snap) if snap else None), None


def bewerte(zeilen, hist):
    """Eine Regime-Menge auswerten. Gibt (kennzahlen, unklar) zurueck."""
    d = []
    unklar = []
    for r in zeilen:
        pos = hist.get(r.get("signature") or "")
        if pos is None:
            unklar.append((r, "keine History-Position zur Signatur"))
            continue
        rec, grund = auswerten_pos(r, pos)
        if rec is None:
            unklar.append((r, grund))
        else:
            d.append(rec)
    if not d:
        return None, unklar

    n = len(d)
    einsatz = sum(x["kosten"] for x in d)
    slips = [x["slip"] for x in d if x["slip"] is not None]
    kz = dict(
        n=n,
        einsatz=einsatz,
        pnl_ist=sum(x["pnl_ist"] for x in d),
        pnl_hold=sum(x["pnl_hold"] for x in d),
        fees=sum(x["fee"] for x in d),
        treffer=sum(1 for x in d if x["getroffen"]),
        verkauft=sum(1 for x in d if x["verkauft"]),
        impl_treffer=sum(1.0 - x["entry"] for x in d) / n,
        slip=sum(slips) / len(slips) if slips else 0.0,
        slip_max=max(slips) if slips else 0.0,
    )
    kz["roi_ist"] = kz["pnl_ist"] / einsatz if einsatz else 0.0
    kz["roi_hold"] = kz["pnl_hold"] / einsatz if einsatz else 0.0
    kz["ist_treffer"] = kz["treffer"] / n
    return kz, unklar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="weather_minus1_live_log.csv (fuehrend auf dem VPS)")
    ap.add_argument("--von", default="2026-07-21")
    ap.add_argument("--bis", default="2026-07-26", help="letzter GESETTELTER Zieltag")
    a = ap.parse_args()

    log = lade_log(a.log)
    hist = history_nach_signatur()
    settle = lade_settlement(a.von, a.bis)
    print(f"Kauf-Zeilen im Log: {len(log)} | History-Positionen indiziert: {len(set(id(v) for v in hist.values()))}")
    print(f"Auswertungsfenster: Zieltage {a.von} .. {a.bis}\n")

    nach_tag = defaultdict(list)
    for r in log:
        nach_tag[r["target_date"]].append(r)

    ergebnisse = []
    alle_unklar = []
    for name, von, bis in REGIMES:
        zeilen = [r for t, rs in nach_tag.items() if von <= t <= bis and t <= a.bis for r in rs]
        if not zeilen:
            ergebnisse.append((name, None))
            continue
        kz, unklar = bewerte(zeilen, hist)
        alle_unklar += unklar
        ergebnisse.append((name, kz))

    print("=" * 84)
    print("REGIME-VERGLEICH   ist = wie gelaufen | hold = Pre-Reg-Regel (bis Settlement halten)")
    print("=" * 84)
    print(f"{'Regime':30s} {'N':>3} {'Einsatz':>8} {'PnL ist':>9} {'ROI ist':>8} "
          f"{'PnL hold':>9} {'ROI hold':>9} {'Slip':>8}")
    ges = defaultdict(float)
    gn = gt = gv = 0
    for name, kz in ergebnisse:
        if kz is None:
            print(f"{name:30s}   —  (kein gesetteltes Fenster)")
            continue
        print(f"{name:30s} {kz['n']:3d} {kz['einsatz']:8.2f} {kz['pnl_ist']:+9.2f} "
              f"{kz['roi_ist']:+7.2%} {kz['pnl_hold']:+9.2f} {kz['roi_hold']:+8.2%} "
              f"{kz['slip']:+8.4f}")
        for k in ("einsatz", "pnl_ist", "pnl_hold", "fees"):
            ges[k] += kz[k]
        gn += kz["n"]; gt += kz["treffer"]; gv += kz["verkauft"]
    print("-" * 84)
    if gn:
        print(f"{'GESAMT':30s} {gn:3d} {ges['einsatz']:8.2f} {ges['pnl_ist']:+9.2f} "
              f"{ges['pnl_ist']/ges['einsatz']:+7.2%} {ges['pnl_hold']:+9.2f} "
              f"{ges['pnl_hold']/ges['einsatz']:+8.2%}")
        print(f"  Kauf-Gebuehren gesamt {ges['fees']:.2f} $ | "
              f"Bucket getroffen (= Lay verloren): {gt}/{gn} | "
              f"vorzeitig verkauft: {gv}")
        if gv:
            print(f"  Kosten der Eingriffe: {ges['pnl_ist']-ges['pnl_hold']:+.2f} $ "
                  f"gegenueber regelkonformem Halten")

    print("\n" + "=" * 84)
    print("KALIBRIERUNG je Regime: preist der Markt die Trefferquote richtig?")
    print("(implizit = 1 − Fill-Preis NO. Positiv = Bucket traf OEFTER als bezahlt = Lay zu teuer)")
    print("=" * 84)
    print(f"{'Regime':30s} {'implizit P(Bucket)':>19} {'ist':>8} {'Differenz':>12}")
    for name, kz in ergebnisse:
        if kz is None:
            continue
        print(f"{name:30s} {kz['impl_treffer']:18.1%} {kz['ist_treffer']:8.1%} "
              f"{100*(kz['ist_treffer']-kz['impl_treffer']):+10.1f} pp")

    # ---- Belastbarkeit -------------------------------------------------
    # Der Regime-Vergleich oben sieht nach klarer Rangfolge aus. Er ist aber fast
    # vollstaendig davon getrieben, in WELCHES Regime der eine Verlierer fiel:
    # ein Lay auf NO~0,88 gewinnt ~+0,55 $ und verliert ~−4,8 $. Ein einziger
    # Treffer mehr oder weniger verschiebt jedes Regime-ROI um zweistellige
    # Prozentpunkte. Diese Rechnung macht das explizit, statt es zu verschweigen.
    alle = []
    for name, von, bis in REGIMES:
        for t, rs in nach_tag.items():
            if von <= t <= bis and t <= a.bis:
                for r in rs:
                    pos = hist.get(r.get("signature") or "")
                    if pos is None:
                        continue
                    rec, _ = auswerten_pos(r, pos)
                    if rec:
                        alle.append((t, name, rec))
    if alle:
        print("\n" + "=" * 84)
        print("BELASTBARKEIT")
        print("=" * 84)
        n = len(alle)
        einsatz = sum(x[2]["kosten"] for x in alle)
        treffer = sum(1 for x in alle if x[2]["getroffen"])
        impl = sum(1.0 - x[2]["entry"] for x in alle) / n
        gew = [x[2]["pnl_hold"] for x in alle if not x[2]["getroffen"]]
        ver = [x[2]["pnl_hold"] for x in alle if x[2]["getroffen"]]
        mgew = sum(gew) / len(gew) if gew else 0.0
        mver = sum(ver) / len(ver) if ver else -(einsatz / n)
        # Break-even je Position aus dem PREIS, nicht aus dem realisierten Mittel:
        # bei nur einem Verlierer waere `mver` das Ergebnis einer einzigen Position
        # zu einem untypischen Preis (Seoul @0,94) und damit kein Mittelwert.
        bes = []
        for _, _, rec in alle:
            w = rec["ktr"] - rec["kosten"]   # Gewinn, wenn der Bucket ausbleibt
            v = rec["kosten"]                # Verlust, wenn er trifft (Einsatz weg)
            bes.append(w / (w + v))
        be = sum(bes) / len(bes)
        pnl0 = sum(x[2]["pnl_hold"] for x in alle)
        print(f"  N = {n} Positionen an {len(set(x[0] for x in alle))} Zieltagen, "
              f"{treffer} Verlierer ({treffer/n:.1%})")
        print(f"  Ø Gewinn {mgew:+.3f} $ | Ø Verlust {mver:+.3f} $  "
              f"→ Asymmetrie 1 : {abs(mver/mgew):.1f}")
        print(f"  Break-even-Verlustquote: {be:.1%}  (darueber ist die Serie negativ)")
        print(f"  Vom Markt eingepreist:   {impl:.1%}    Ist: {treffer/n:.1%}")
        print(f"  → Der Edge ist die Luecke {impl:.1%} − {treffer/n:.1%}. "
              f"Bei n={n} steht dahinter {treffer} Ereignis, kein Beleg.")
        print(f"\n  Sensitivitaet (PnL hold gesamt {pnl0:+.2f} $ = {pnl0/einsatz:+.2%}):")
        for zusatz in (1, 2, 3):
            alt = pnl0 + zusatz * (mver - mgew)   # ein Gewinner wird zum Verlierer
            print(f"    {zusatz} Verlierer mehr → {alt:+7.2f} $ = {alt/einsatz:+7.2%}"
                  f"{'   ← Serie kippt' if alt < 0 else ''}")
        print("\n  Je Zieltag (die Streuung ist die eigentliche Aussage):")
        tage = defaultdict(float)
        for t, _, rec in alle:
            tage[t] += rec["pnl_hold"]
        for t in sorted(tage):
            print(f"    {t}: {tage[t]:+7.2f} $")

    print("\n" + "=" * 84)
    print("EINZELPOSITIONEN")
    print("=" * 78)
    print(f"{'Zieltag':11s} {'Stadt':13s} {'k':>3} {'snap':>5} {'fill':>5} "
          f"{'Kontr':>6} {'WU':>3} {'PnL ist':>8} {'PnL hold':>9}  Ergebnis")
    for t in sorted(nach_tag):
        if t > a.bis:
            continue
        for r in sorted(nach_tag[t], key=lambda x: x["city"]):
            pos = hist.get(r.get("signature") or "")
            st = settle.get((t, r["city"], int(r["k"])), (None, None, None))
            if pos is None:
                print(f"{t:11s} {r['city']:13s} {r['k']:>3}  —  keine History-Position")
                continue
            rec, grund = auswerten_pos(r, pos)
            if rec is None:
                print(f"{t:11s} {r['city']:13s} {r['k']:>3}  —  {grund}")
                continue
            erg = "Bucket TRAF (Lay verloren)" if rec["getroffen"] else "Lay gewonnen"
            if rec["verkauft"]:
                erg += "  << vorzeitig VERKAUFT"
            print(f"{t:11s} {r['city']:13s} {r['k']:>3} {float(r['buy_no_snap'] or 0):5.2f} "
                  f"{rec['entry']:5.2f} {rec['ktr']:6.2f} {str(st[1]):>3} "
                  f"{rec['pnl_ist']:+8.3f} {rec['pnl_hold']:+9.3f}  {erg}")

    if alle_unklar:
        print("\nUNKLAR / NICHT BEWERTET:")
        for r, grund in alle_unklar:
            print(f"  {r['target_date']} {r['city']} k={r['k']}: {grund}")


if __name__ == "__main__":
    main()
