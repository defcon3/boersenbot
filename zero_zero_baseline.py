#!/usr/bin/env python3
"""
zero_zero_baseline.py — Analytisches Baseline-Modell: Quoten-Drift in 0:0-Spielen.

Forschungsfrage (Lay-the-Draw-Kontext, vgl. Jupiter-Bot): Man layt zum Anpfiff (0:0)
EINE Mannschaft — hier den FAVORITEN — und hedged später, solange es 0:0 bleibt.
  (1) Wie verläuft P(Sieg | noch 0:0) je Minute — linear oder „erst langsam, dann schnell"?
  (2) Wie groß ist der Green-up (Lay-Gewinn) beim Favoriten vs. Außenseiter?
  (3) Wo beschleunigt der Rutsch?

Methode (kein Tick-Daten-Bedarf — rein analytisch als ERWARTUNGS-Baseline):
  - Tore = unabhängige Poisson-Prozesse mit Raten λ_fav, λ_dog (über 90 min).
  - λ aus den Anpfiff-Wahrscheinlichkeiten (Heim/Draw/Auswärts) gelöst (Skellam).
  - Bei noch 0:0 nach Minute t: Restspiel = frisches Spiel mit Raten λ·(1 − t/90).
    => P(Sieg | 0:0 bei t) = P(Team erzielt in der Restzeit mehr Tore als der Gegner).
  - Green-up beim Lay zum Anpfiff (Preis p0) und Rückkauf bei p_t:  Gewinn/Kontrakt = p0 − p_t.

Das ist die ERWARTUNG des effizienten Marktes. Der spätere empirische Vergleich
(Polymarket /prices-history) zeigt, WO der echte Markt schneller/langsamer rutscht.

Aufruf:
  python zero_zero_baseline.py                          # repräsentativer Favorit (Default)
  python zero_zero_baseline.py --ph 0.62 --pd 0.24 --pa 0.14
  python zero_zero_baseline.py --slug fifwc-prt-uzb-2026-06-23   # Anpfiff-Quoten real ziehen
"""

import argparse
import sys

import numpy as np
from scipy.stats import skellam
from scipy.optimize import fsolve

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

MINUTES = 90


def hda(lh, la):
    """P(Heimsieg), P(Remis), P(Auswärtssieg) bei Poisson-Raten lh, la (Skellam für die Differenz)."""
    lh, la = max(lh, 1e-9), max(la, 1e-9)
    p_draw = float(skellam.pmf(0, lh, la))
    p_home = float(1 - skellam.cdf(0, lh, la))   # Tordiff >= 1
    p_away = float(skellam.cdf(-1, lh, la))       # Tordiff <= -1
    return p_home, p_draw, p_away


def solve_lambdas(p_home, p_draw):
    """Löst (λ_heim, λ_auswärts) so, dass das Poisson-Modell die Anpfiff-Quoten trifft."""
    def eqs(L):
        H, D, _ = hda(L[0], L[1])
        return [H - p_home, D - p_draw]
    sol = fsolve(eqs, [1.4, 1.0], full_output=False)
    return float(sol[0]), float(sol[1])


def cond_probs(lh, la, t):
    """P(Heim/Remis/Auswärts | noch 0:0 nach Minute t) — Restspiel mit skalierten Raten."""
    r = max(0.0, (MINUTES - t) / MINUTES)
    return hda(lh * r, la * r)


def fetch_kickoff_probs(slug):
    """Anpfiff-Wahrscheinlichkeiten real aus Polymarket ziehen (Preis je Token nahe gameStartTime)."""
    import json
    import requests
    ev = requests.get("https://gamma-api.polymarket.com/events", params={"slug": slug}, timeout=20).json()[0]
    teams, draw_tok, gst = [], None, None
    for m in ev.get("markets") or []:
        gi = (m.get("groupItemTitle") or "").strip()
        toks = json.loads(m.get("clobTokenIds") or "[]")
        if not gi or not toks:
            continue
        gst = gst or m.get("gameStartTime")
        if "draw" in gi.lower():
            draw_tok = toks[0]
        else:
            teams.append((gi, toks[0]))
    import datetime as dt
    t0 = dt.datetime.fromisoformat(gst.replace(" ", "T").replace("+00", "+00:00")).timestamp()

    def price_at(tok):
        h = requests.get("https://clob.polymarket.com/prices-history",
                         params={"market": tok, "interval": "max", "fidelity": 1}, timeout=20).json().get("history", [])
        if not h:
            return None
        # nächster Punkt zum Anpfiff
        return min(h, key=lambda p: abs(p["t"] - t0))["p"]

    p1, pd, p2 = price_at(teams[0][1]), price_at(draw_tok), price_at(teams[1][1])
    return (teams[0][0], p1), (teams[1][0], p2), pd, ev.get("title", slug)


def bar(v, vmax, width=28):
    n = int(round(width * v / vmax)) if vmax > 0 else 0
    return "█" * n + "·" * (width - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ph", type=float, default=0.58, help="Anpfiff P(Heim/Team1-Sieg)")
    ap.add_argument("--pd", type=float, default=0.26, help="Anpfiff P(Remis)")
    ap.add_argument("--pa", type=float, default=0.16, help="Anpfiff P(Auswärts/Team2-Sieg)")
    ap.add_argument("--slug", help="Anpfiff-Quoten real aus Polymarket-Event ziehen")
    ap.add_argument("--t1", default="Team1")
    ap.add_argument("--t2", default="Team2")
    args = ap.parse_args()

    t1, t2, title = args.t1, args.t2, "Beispiel-Favorit"
    ph, pd, pa = args.ph, args.pd, args.pa
    if args.slug:
        (t1, ph), (t2, pa), pd, title = fetch_kickoff_probs(args.slug)

    # Overround entfernen (auf Summe 1 normieren)
    s = ph + pd + pa
    ph, pd, pa = ph / s, pd / s, pa / s

    lh, la = solve_lambdas(ph, pd)
    Hc, Dc, Ac = hda(lh, la)  # Kontroll-Rückrechnung

    # Favorit = höhere Anpfiff-Siegwahrscheinlichkeit
    fav_is_home = ph >= pa
    fav_name, dog_name = (t1, t2) if fav_is_home else (t2, t1)
    p_fav0 = ph if fav_is_home else pa
    p_dog0 = pa if fav_is_home else ph

    print("=" * 70)
    print(f"0:0-BASELINE  |  {title}")
    print("=" * 70)
    print(f"Anpfiff (normiert):  {t1} {ph:.3f}  |  Remis {pd:.3f}  |  {t2} {pa:.3f}")
    print(f"Implizite Torraten:  λ({t1})={lh:.3f}   λ({t2})={la:.3f}   "
          f"(erw. Tore {lh+la:.2f})")
    print(f"Modell-Rückrechnung: Heim {Hc:.3f} / Remis {Dc:.3f} / Ausw. {Ac:.3f}  (sollte oben matchen)")
    print(f"FAVORIT = {fav_name}  (Anpfiff {p_fav0:.3f})   ·   Außenseiter = {dog_name}  ({p_dog0:.3f})")
    print("-" * 70)
    print(f"{'Min':>4} | {'P(Fav|0:0)':>10} {'P(Dog|0:0)':>10} {'P(Remis)':>9} | "
          f"{'GreenUp Lay-Fav':>15} {'Lay-Dog':>8}")
    print("-" * 70)

    rows = []
    for t in range(0, 91, 5):
        H, D, A = cond_probs(lh, la, t)
        p_fav = H if fav_is_home else A
        p_dog = A if fav_is_home else H
        gu_fav = p_fav0 - p_fav  # Lay zum Anpfiff, Rückkauf bei t
        gu_dog = p_dog0 - p_dog
        rows.append((t, p_fav, p_dog, D, gu_fav, gu_dog))
        print(f"{t:>4} | {p_fav:>10.3f} {p_dog:>10.3f} {D:>9.3f} | "
              f"{gu_fav:>+15.3f} {gu_dog:>+8.3f}")

    # Nicht-Linearität: Rutsch der Favoriten-Siegwahrscheinlichkeit je 15-min-Block
    print("-" * 70)
    print("NICHT-LINEARITÄT — Abfall P(Fav|0:0) je 15-min-Block:")
    blocks = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90)]
    drops = []
    for a, b in blocks:
        fa = (cond_probs(lh, la, a)[0] if fav_is_home else cond_probs(lh, la, a)[2])
        fb = (cond_probs(lh, la, b)[0] if fav_is_home else cond_probs(lh, la, b)[2])
        drops.append(fa - fb)
    dmax = max(drops)
    for (a, b), d in zip(blocks, drops):
        print(f"  {a:>2}–{b:<2} min:  −{d:.3f}  {bar(d, dmax)}")
    print(f"  => letzter Block fällt {drops[-1]/drops[0]:.1f}× so stark wie der erste "
          f"(erst langsam, dann schnell ✔)" if drops[0] > 0 else "")

    # Fazit Lay-Favorit vs Lay-Außenseiter
    gf80 = rows[16][4] if len(rows) > 16 else rows[-1][4]   # t=80
    gd80 = rows[16][5] if len(rows) > 16 else rows[-1][5]
    print("-" * 70)
    print("FAZIT (Modell-Erwartung):")
    print(f"  • Lay-Favorit greent bei Minute 80 (noch 0:0) theoretisch +{gf80:.3f}/Kontrakt,")
    print(f"    Lay-Außenseiter nur +{gd80:.3f}  → Favorit liefert den {gf80/max(gd80,1e-9):.1f}× größeren Rutsch.")
    print(f"  • Preis dafür: der Favorit BRICHT das 0:0 am ehesten (λ höher) = dein Verlustfall.")
    print(f"  • Empirisch zu prüfen: wo läuft der echte Polymarket-Pfad STEILER als diese Baseline?")
    print("=" * 70)


if __name__ == "__main__":
    main()
