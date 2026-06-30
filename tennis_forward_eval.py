#!/usr/bin/env python3
"""
tennis_forward_eval.py — Auswertung des Elo-Forward-Paper-Tests (bb_TennisPaperBets).

Hintergrund: tennis_paper_logger.py loggte für kommende ATP/WTA-Singles einen
Pre-Match-Snapshot (Modell-Fair via Elo vs. Jupiter-Markt-Ask) und markierte die
Seite mit positivem Edge (fair − price > 0) als value_side. backfill_tennis_results.py
trägt den Sieger nach (settled=1). Dieses Skript misst, ob die Modell-„Value"-Picks
den Markt tatsächlich schlagen.

Pick-Universum: settled=1, source='elo_auto', edge>0, value_side ∈ {A,B}.
Hypothetischer Trade: 1 USD auf value_side @ Markt-Ask -> Payout 1/p bei Sieg,
0 bei Niederlage; ROI = ΣP&L / N. Zusätzlich Trefferquote, Ø Modell-Fair der
Pick-Seite (erwartete Trefferquote bei perfekter Kalibrierung) und Brier-Score.

================================================================================
BEFUND 2026-06-30 (N=80, ~Wimbledon-Vorlauf, vorläufig — FALSIFIZIERT):
  ALLES:           35.0 % Treffer, ROI −19.9 %, Brier 0.231
  MAIN-TOUR-only:  37.8 % Treffer, ROI −17.5 %, Brier 0.219  (N=45)
  nur Quali:       31.4 % Treffer, ROI −22.9 %                (N=35)

Der Main-Tour-only-Cut rettet die Edge NICHT — er verbessert nur marginal.
Kernproblem: Das Modell behauptet auf seinen Picks ~54 % Siegchance, real gewinnen
nur ~38 % (16-Punkte-Lücke). Wo das Modell am stärksten vom Markt abweicht (= Value
sieht), hat der Markt recht -> Ranking-Lag-Trap. Brier ~0.22 ≈ Münzwurf (0.25).
Konsequenz: Elo-Auto-Picker auf dem VPS abgeschaltet (boersenbot_tennis_paper.timer
disabled). OOS schlägt IS: kein deploybarer Edge.
================================================================================

Aufruf:  python tennis_forward_eval.py
"""
from collections import Counter

import pymssql

# Hardcodierte Centron-Creds — bewusste Projektentscheidung (siehe CLAUDE.md).
DB_CONFIG = {
    "server": "158.181.48.77",
    "database": "dbdata",
    "user": "326773",
    "password": "Extaler11!",
}


def classify(tournament):
    """Grobe Turnier-Stufe aus dem Namen (Jupiter liefert keine Level-Spalte)."""
    t = (tournament or "").lower()
    if "qualif" in t or "qualifying" in t:
        return "quali"
    if "challenger" in t:
        return "challenger"
    if "itf" in t:
        return "itf"
    return "main"


def analyze(rows, label):
    n = len(rows)
    if n == 0:
        print(f"\n### {label}: keine Picks")
        return
    hits = pnl = fair_sum = brier = 0.0
    used = skipped_price = 0
    for r in rows:
        side = r["value_side"]
        won = (r["actual_winner"] == side)
        hits += 1 if won else 0
        fair = r["fair_a"] if side == "A" else r["fair_b"]
        if fair is not None:
            fair_sum += fair
            brier += (1.0 - fair) ** 2 if won else fair ** 2
        price = r["price_a"] if side == "A" else r["price_b"]
        if price and price > 0:
            pnl += (1.0 / price - 1.0) if won else -1.0
            used += 1
        else:
            skipped_price += 1
    print(f"\n### {label}")
    print(f"  Picks (settled, edge>0):      {n}")
    print(f"  Trefferquote (value_side):    {int(hits)}/{n} = {hits/n*100:.1f}%")
    print(f"  Ø Modell-Fair der Pick-Seite: {fair_sum/n*100:.1f}%  (erwartete Trefferquote bei Kalibrierung)")
    if used:
        print(f"  Hypoth. P&L @ Markt (1$/Pick, N={used}): {pnl:+.2f} USD   ROI {pnl/used*100:+.1f}%")
    print(f"  Brier-Score (Pick-Seite):     {brier/n:.4f}  (niedriger = besser; 0.25 = Münzwurf)")
    if skipped_price:
        print(f"  (ohne Preis verworfen: {skipped_price})")


def main():
    conn = pymssql.connect(**DB_CONFIG, autocommit=True)
    cur = conn.cursor(as_dict=True)
    cur.execute("""
        SELECT tournament, tour, value_side, actual_winner,
               fair_a, fair_b, price_a, price_b, edge
        FROM bb_TennisPaperBets
        WHERE settled=1 AND source='elo_auto' AND edge>0
          AND value_side IN ('A','B') AND actual_winner IN ('A','B')
    """)
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        r["cat"] = classify(r["tournament"])

    print("=" * 64)
    print(f"ELO-FORWARD-TEST — settled elo_auto value-picks: N={len(rows)}")
    print("=" * 64)
    print("\nKategorien:", dict(Counter(r["cat"] for r in rows)))

    analyze(rows, "ALLES (inkl. Quali/Challenger/ITF)")
    analyze([r for r in rows if r["cat"] == "main"], "MAIN-TOUR-ONLY (ATP/WTA Hauptfeld)")
    analyze([r for r in rows if r["cat"] == "quali"], "nur QUALIFIKATION")
    analyze([r for r in rows if r["cat"] in ("challenger", "itf")], "nur CHALLENGER/ITF")
    analyze([r for r in rows if r["cat"] == "main" and r["tour"] == "atp"], "MAIN ATP")
    analyze([r for r in rows if r["cat"] == "main" and r["tour"] == "wta"], "MAIN WTA")


if __name__ == "__main__":
    main()
