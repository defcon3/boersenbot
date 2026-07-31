#!/usr/bin/env python3
"""
weather_open_taker_pnl.py — Nachtrag zur Pre-Reg
`preregs/weather_open_convergence_2026_07_31.md` (Commit 476134e).

## Warum es diese Datei gibt

Die Pre-Reg testet, ob man als *Taker* den Eroeffnungs-Fehlpreis abgreifen kann
(H1/H2/H3 — alle gefallen, s. `weather_open_convergence_eval.py`). Die
eigentliche Idee war aber eine andere: nicht als Erster kaufen, sondern als
Erster **quoten** — eine Order ins junge Buch legen, die nur ein
uninformierter Zufallskaeufer nimmt. Kurz: den Dummen finden.

Ob die *eigene* Order gefuellt worden waere, ist historisch nicht pruefbar —
Polymarket gibt den Trade-Tape her, aber keine Buchsnapshots, und die
Warteschlangenposition einer nie gestellten Order ist kontrafaktisch.

Die entscheidende Vorfrage ist es sehr wohl: **Verliert der Zugreifende Geld?**
Der Bruttoverlust des Takers ist exakt die Einnahme des Quotenstellers. Ist er
positiv, ist der fruehe Fluss informiert und der Maker waere der Ausgenommene.
Das braucht keine Annahme ueber Fills und ist damit die obere Schranke der
Beute.

Achtung bei der Interpretation: gerechnet wird **vor Gebuehr**. Der Taker zahlt
zusaetzlich 5 % von min(p, 1-p) und liegt netto oft im Minus — aber diese
Gebuehr geht an die Venue, **nicht** an den Maker. Fuer die Maker-Rechnung
zaehlt nur das Vorzeichen des Brutto-Ertrags, mit umgekehrtem Vorzeichen.

Datenbasis ist derselbe Sammler wie fuer die Pre-Reg (`collect_open.py`,
data/<stadt>.jsonl). Aufruf:

    python weather_open_taker_pnl.py --data <verzeichnis>
"""
import sys, json, glob, math, os, argparse
from collections import defaultdict

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

WINDOWS   = [(0, 5), (5, 15), (15, 60), (60, 240)]   # Minuten seit Eroeffnung
SIZE_BINS = [(0, 5), (5, 25), (25, 100), (100, float("inf"))]      # Einsatz $
PX_BINS   = [(0, 0.10), (0.10, 0.30), (0.30, 0.70), (0.70, 0.90), (0.90, 1.0)]
EARLY_MIN = 15            # Fenster fuer die beiden Aufschluesselungen


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def tstat(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m / math.sqrt(var / n) if var > 0 else 0.0


def taker_leg(px_yes, side_yes, settled):
    """Preis und Ausgang aus Sicht des ZUGREIFENDEN.

    side_yes == +1: der Taker nahm die YES-Ask-Seite, kaufte also YES zu px_yes.
    side_yes == -1: er nahm die Bid-Seite, kaufte also NO zu 1 - px_yes.
    Rueckgabe (preis_der_gekauften_seite, gewonnen?).
    """
    if side_yes == 1:
        return px_yes, settled == 1
    return 1.0 - px_yes, settled == 0


def iter_trades(data_dir):
    """(minuten_seit_open, preis_gekaufte_seite, gewonnen, einsatz_usd)"""
    for path in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                s = r.get("settled")
                if s is None:
                    continue
                for d, px, side, sz in (r.get("tr") or []):
                    if d < 0 or sz <= 0:
                        continue
                    q, won = taker_leg(px, side, s)
                    if q <= 0.001 or q >= 0.999:
                        continue          # entschiedene Buckets tragen nichts bei
                    yield d / 60.0, q, won, sz * q


class Bin:
    __slots__ = ("pnl", "vol", "n", "rets")
    def __init__(self):
        self.pnl = 0.0; self.vol = 0.0; self.n = 0; self.rets = []
    def add(self, ret, stake):
        self.pnl += ret * stake; self.vol += stake; self.n += 1; self.rets.append(ret)


def table(title, bins, order, fmt):
    print(f"\n{title}")
    print(f"  {'Klasse':>12s} {'Trades':>8s} {'Umsatz $':>11s} {'PnL $':>10s}"
          f" {'% Umsatz':>10s} {'t':>7s}")
    for k in order:
        b = bins.get(k)
        if not b or not b.n:
            continue
        print(f"  {fmt(k):>12s} {b.n:8d} {b.vol:11.0f} {b.pnl:10.0f}"
              f" {b.pnl / b.vol * 100:9.2f} % {tstat(b.rets):7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Verzeichnis mit <stadt>.jsonl")
    args = ap.parse_args()

    by_win = defaultdict(Bin)
    by_sz  = defaultdict(Bin)
    by_px  = defaultdict(Bin)
    total_early = 0.0

    for m, q, won, stake in iter_trades(args.data):
        ret = ((1.0 if won else 0.0) - q) / q
        for w in WINDOWS:
            if w[0] <= m < w[1]:
                by_win[w].add(ret, stake); break
        if m < EARLY_MIN:
            total_early += stake
            for b in SIZE_BINS:
                if b[0] <= stake < b[1]:
                    by_sz[b].add(ret, stake); break
            for b in PX_BINS:
                if b[0] <= q < b[1]:
                    by_px[b].add(ret, stake); break

    print("PnL des ZUGREIFENDEN (Taker) je Dollar Einsatz, VOR Gebuehr.")
    print("negativ = der Zugreifende verliert = Beute fuer den Quotensteller.")

    table("nach Zeitfenster seit Eroeffnung:", by_win, WINDOWS,
          lambda k: f"{k[0]}-{k[1]} min")
    table(f"nach Einsatzgroesse, Minute 0-{EARLY_MIN} ('der Dumme setzt klein'):",
          by_sz, SIZE_BINS,
          lambda k: f"{k[0]:.0f}-{k[1]:.0f}" if k[1] != float("inf") else f">{k[0]:.0f}")
    table(f"nach Preis der gekauften Seite, Minute 0-{EARLY_MIN} (Rand vs Mitte):",
          by_px, PX_BINS, lambda k: f"{k[0]:.2f}-{k[1]:.2f}")

    print(f"\nGesamtumsatz Minute 0-{EARLY_MIN} ueber alle Staedte und Bretter: "
          f"{total_early:,.0f} $")
    print("Das ist die harte Obergrenze: die Beute des Quotenstellers kann nicht")
    print("groesser sein als der Umsatz, der ueberhaupt stattfindet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
