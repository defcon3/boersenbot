#!/usr/bin/env python3
"""
weather_open_convergence_eval.py — Auswertung zur Pre-Registrierung
`preregs/weather_open_convergence_2026_07_31.md` (committet 31.07.2026, VOR
jeder Kennzahl).

Liest die Rohdaten des Sammlers (`collect_open.py`, eine JSONL-Zeile je Bucket)
und rechnet ausschliesslich, was dort vorregistriert ist:

  H1  Formresiduum gegen die Leave-one-out-Glocke  -> Gates G1..G5
  H2  Longshot-Bias  (Buckets < 0,10 in Minute T)
  H3  Niveau-Arbitrage (Summe der Bucketpreise weicht von 1,00 ab)

Aufruf:
    python weather_open_convergence_eval.py --data <verzeichnis>

Die Sensitivitaetszellen (T x theta) werden berichtet, duerfen aber laut
Pre-Reg kein Gate erfuellen.
"""
import os, sys, json, glob, math, argparse
from collections import defaultdict

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

# ---------------------------------------------------------------- Parameter
T_PRIMARY     = 30          # Minuten seit acceptingOrdersTimestamp
THETA_PRIMARY = 0.05
MIN_BUCKETS   = 6           # Mindestbelegung je Brett
T_SENS        = (15, 30, 60)
THETA_SENS    = (0.03, 0.05, 0.10)
EXIT2_MIN     = 60          # Exit E2: Verkauf in Minute 60
FEE_RATE      = 0.05        # 5 % von min(p, 1-p), ohne rebateRate (konservativ)

IS_MONTHS  = ("2026-04", "2026-05", "2026-06")
OOS_MONTHS = ("2026-07",)


# ---------------------------------------------------------------- Statistik
def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def tstat(xs):
    """t gegen 0, mit Stichproben-Standardabweichung."""
    n = len(xs)
    if n < 2:
        return float("nan")
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    if var <= 0:
        return float("inf") if m > 0 else (float("-inf") if m < 0 else 0.0)
    return m / math.sqrt(var / n)

def median(xs):
    if not xs:
        return float("nan")
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# ---------------------------------------------------------------- Bucket-Achse
def bucket_value(label):
    """Temperaturachse aus dem Bucket-Label.

    Die Bretter mischen Rand- und Innenbuckets ("below 18", "18", "19", ...,
    "above 27"). Fuer die Formreferenz wird nur eine monotone, aequidistante
    Achse gebraucht - die Raender bekommen daher den nachsten ganzzahligen
    Schritt und keine Extrapolation.
    """
    if label is None:
        return None
    s = str(label).strip().lower().replace("°", " ").replace("c", " ")
    s = s.replace("º", " ")
    neg = False
    if any(w in s for w in ("below", "under", "less", "<", "or lower", "oder weniger")):
        neg = True
    hi = any(w in s for w in ("above", "over", "more", ">", "or higher", "oder mehr"))
    num = []
    cur = ""
    for ch in s:
        if ch.isdigit() or (ch == "-" and not cur):
            cur += ch
        else:
            if cur:
                num.append(cur); cur = ""
    if cur:
        num.append(cur)
    if not num:
        return None
    try:
        v = float(num[0])
    except ValueError:
        return None
    if neg:
        v -= 1.0
    elif hi:
        v += 1.0
    return v


# ---------------------------------------------------------------- Preise aus dem Tape
def prices_at(trades, t_min):
    """Letzter YES-normalisierter Preis bis Minute t_min, getrennt nach Seite.

    Rueckgabe (last, ask, bid):
      last = letzter Preis egal welche Seite  -> Signalbildung
      ask  = letzter Preis mit side_yes == +1 -> so teuer war YES zu KAUFEN
      bid  = letzter Preis mit side_yes == -1 -> so teuer war NO  zu kaufen
    Ohne Trade bis t_min: (None, None, None).
    """
    cut = t_min * 60
    last = ask = bid = None
    for d, px, side, _sz in trades:      # aufsteigend sortiert
        if d > cut:
            break
        last = px
        if side == 1:
            ask = px
        else:
            bid = px
    return last, ask, bid

def notional(trades, t_min):
    cut = t_min * 60
    return sum(sz for d, _px, _s, sz in trades if d <= cut)


def fit_normal_loo(xs, ys, i):
    """Normalverteilung an alle Punkte AUSSER i, ausgewertet an x_i.

    Kleinste Quadrate ueber mu und sigma per Gitter - der Fit hat nur zwei
    Parameter und die Achse ist grob, ein Gitter ist hier robuster als ein
    Gradientenverfahren, das an flachen Verteilungen wegdriftet.
    Amplitude wird analytisch als optimaler Skalar mitgeloest.
    """
    ox = [xs[j] for j in range(len(xs)) if j != i]
    oy = [ys[j] for j in range(len(ys)) if j != i]
    if len(ox) < 3:
        return None
    lo, hi = min(ox), max(ox)
    span = max(hi - lo, 1.0)
    best = None
    mu = lo - 0.5 * span
    while mu <= hi + 0.5 * span + 1e-9:
        sig = 0.4
        while sig <= 2.5 * span + 1e-9:
            g = [math.exp(-0.5 * ((x - mu) / sig) ** 2) for x in ox]
            gg = sum(v * v for v in g)
            if gg > 1e-12:
                a = sum(v * y for v, y in zip(g, oy)) / gg      # optimale Amplitude
                if a > 0:
                    sse = sum((a * v - y) ** 2 for v, y in zip(g, oy))
                    if best is None or sse < best[0]:
                        best = (sse, mu, sig, a)
            sig += 0.1 * span
        mu += 0.1 * span
    if best is None:
        return None
    _sse, mu, sig, a = best
    return a * math.exp(-0.5 * ((xs[i] - mu) / sig) ** 2)


# ---------------------------------------------------------------- Ein-/Ausstieg
def fee(p):
    return FEE_RATE * min(p, 1.0 - p)

def trade_pnl(side, entry, settled_yes, exit_px=None):
    """Netto-Ertrag je 1 Einheit Einsatz.

    side: "YES" oder "NO"; entry ist der Preis der gekauften Seite.
    exit_px is None  -> E1, halten bis Settlement (Auszahlung 1 oder 0)
    exit_px gesetzt  -> E2, Verkauf zu exit_px (Preis derselben Seite)
    Gebuehr faellt auf beiden Seiten des Handels an.
    """
    cost = entry + fee(entry)
    if exit_px is None:
        won = (settled_yes == 1) if side == "YES" else (settled_yes == 0)
        payout = 1.0 if won else 0.0
    else:
        payout = exit_px - fee(exit_px)
    return (payout - cost) / cost


# ---------------------------------------------------------------- Laden
def load(data_dir):
    boards = defaultdict(list)
    n_rows = 0
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
                if r.get("settled") is None or not r.get("slug"):
                    continue
                r["_x"] = bucket_value(r.get("bucket"))
                if r["_x"] is None:
                    continue
                boards[(r["city"], r["slug"])].append(r)
                n_rows += 1
    return boards, n_rows

def board_month(rows):
    c = str(rows[0].get("created") or "")[:7]
    return c


# ---------------------------------------------------------------- H1
def h1_signals(boards, t_min, theta):
    """Alle Signale des Primaersignals bei (t_min, theta)."""
    out = []
    for (city, slug), rows in boards.items():
        obs = []
        for r in rows:
            last, ask, bid = prices_at(r.get("tr") or [], t_min)
            if last is None:
                continue
            obs.append((r, last, ask, bid))
        if len(obs) < MIN_BUCKETS:
            continue
        tot = sum(o[1] for o in obs)
        if tot <= 0:
            continue
        xs = [o[0]["_x"] for o in obs]
        qs = [o[1] / tot for o in obs]
        for i, (r, last, ask, bid) in enumerate(obs):
            g = fit_normal_loo(xs, qs, i)
            if g is None:
                continue
            resid = qs[i] - g
            if resid > theta:
                side, entry = "NO", (1.0 - bid) if bid is not None else None
            elif resid < -theta:
                side, entry = "YES", ask
            else:
                continue
            if entry is None or not (0.01 <= entry <= 0.99):
                continue          # benoetigte Seite bis T nicht gehandelt
            _l2, ask2, bid2 = prices_at(r.get("tr") or [], EXIT2_MIN)
            if side == "YES":
                exit2 = bid2
            else:
                exit2 = (1.0 - ask2) if ask2 is not None else None
            out.append({
                "city": city, "slug": slug, "month": board_month(rows),
                "bucket": r.get("bucket"), "side": side, "entry": entry,
                "resid": resid, "settled": r["settled"],
                "exit2": exit2, "notional": notional(r.get("tr") or [], t_min),
                "e1": trade_pnl(side, entry, r["settled"]),
                "e2": (trade_pnl(side, entry, r["settled"], exit2)
                       if exit2 is not None and 0.01 <= exit2 <= 0.99 else None),
            })
    return out

def split(sigs):
    return ([s for s in sigs if s["month"] in IS_MONTHS],
            [s for s in sigs if s["month"] in OOS_MONTHS])


def report_leg(name, sigs, key="e1"):
    xs = [s[key] for s in sigs if s[key] is not None]
    if not xs:
        print(f"  {name:22s} keine Signale")
        return None, None
    m, t = mean(xs), tstat(xs)
    print(f"  {name:22s} n={len(xs):5d}  ROI={m*100:+7.2f} %  t={t:+6.2f}")
    return m, t


def gates(sigs_is, sigs_oos, all_sigs):
    print("\n--- Gates (Primaerparameter T=%d, theta=%.2f) ---" % (T_PRIMARY, THETA_PRIMARY))
    m1, t1 = report_leg("G1 In-Sample (E1)", sigs_is)
    m2, t2 = report_leg("G2 OOS (E1)", sigs_oos)
    report_leg("   OOS Exit E2", sigs_oos, "e2")

    g1 = (m1 is not None and m1 > 0 and t1 > 2.0)
    g2 = (g1 and m2 is not None and m2 > 0 and t2 > 1.5)
    g3 = (g2 and m2 > 0)          # Kosten stecken bereits in e1

    # G4: >= 1 Signal pro Tag im OOS
    days = {s["slug"] for s in sigs_oos}
    n_days_oos = 31               # Juli 2026
    g4 = (len(sigs_oos) / n_days_oos) >= 1.0

    # G5: Median ueber Staedte > 0, robust gegen beste Stadt, kein Tag > 30 %
    per_city = defaultdict(list)
    for s in sigs_oos:
        if s["e1"] is not None:
            per_city[s["city"]].append(s["e1"])
    city_rois = {c: mean(v) for c, v in per_city.items()}
    med_city = median(list(city_rois.values()))
    g5a = med_city > 0
    if city_rois:
        best = max(city_rois, key=lambda c: city_rois[c])
        rest = [s["e1"] for s in sigs_oos if s["city"] != best and s["e1"] is not None]
        g5b = bool(rest) and mean(rest) > 0
    else:
        best, g5b = None, False
    per_day = defaultdict(float)
    tot = sum(s["e1"] for s in sigs_oos if s["e1"] is not None)
    for s in sigs_oos:
        if s["e1"] is not None:
            per_day[s["slug"]] += s["e1"]
    if tot > 0 and per_day:
        g5c = (max(per_day.values()) / tot) <= 0.30
    else:
        g5c = False
    g5 = g1 and g5a and g5b and g5c

    def mark(ok):
        return "GRUEN" if ok else "ROT"
    print()
    print(f"  G1 IS t>2,0 ................. {mark(g1)}")
    print(f"  G2 OOS t>1,5 gleiches VZ .... {mark(g2)}")
    print(f"  G3 netto ROI>0 im OOS ....... {mark(g3)}")
    print(f"  G4 >=1 Signal/Tag (OOS) ..... {mark(g4)}   ({len(sigs_oos)} Signale / {n_days_oos} Tage, {len(days)} Bretter)")
    print(f"  G5 Robustheit ............... {mark(g5)}   (Median Stadt {med_city*100:+.2f} %, "
          f"ohne beste Stadt {'ok' if g5b else 'faellt'}"
          + (f" [{best}]" if best else "") + f", Tageskonzentration {'ok' if g5c else 'zu hoch'})")

    print("\n  Buchtiefe (Median Notional je Signal-Fenster, OOS): "
          f"{median([s['notional'] for s in sigs_oos]):.0f} $" if sigs_oos else "")
    return g1 and g2 and g3 and g4 and g5


def sensitivity(boards):
    print("\n--- Sensitivitaet (berichtet, KEIN Gate) ---")
    print("     T   theta |    IS n   IS ROI    IS t |   OOS n  OOS ROI   OOS t")
    for t_min in T_SENS:
        for th in THETA_SENS:
            sigs = h1_signals(boards, t_min, th)
            a, b = split(sigs)
            xa = [s["e1"] for s in a]; xb = [s["e1"] for s in b]
            fa = f"{mean(xa)*100:+7.2f} {tstat(xa):+6.2f}" if xa else "      -      -"
            fb = f"{mean(xb)*100:+7.2f} {tstat(xb):+6.2f}" if xb else "      -      -"
            star = " *" if (t_min == T_PRIMARY and th == THETA_PRIMARY) else "  "
            print(f"  {t_min:4d}   {th:.2f}{star}| {len(a):6d} {fa} | {len(b):6d} {fb}")


# ---------------------------------------------------------------- H2
def h2(boards, t_min=T_PRIMARY):
    print("\n=== H2 Longshot-Bias (NO auf alle Buckets < 0,10 in Minute T) ===")
    rows = []
    for (_city, _slug), brd in boards.items():
        for r in brd:
            last, ask, bid = prices_at(r.get("tr") or [], t_min)
            if last is None or last >= 0.10:
                continue
            entry = (1.0 - bid) if bid is not None else None
            if entry is None or not (0.01 <= entry <= 0.99):
                continue
            rows.append({"month": board_month(brd), "px": last,
                         "settled": r["settled"],
                         "e1": trade_pnl("NO", entry, r["settled"])})
    a = [r for r in rows if r["month"] in IS_MONTHS]
    b = [r for r in rows if r["month"] in OOS_MONTHS]
    for nm, grp in (("IS ", a), ("OOS", b)):
        if not grp:
            print(f"  {nm}: keine Signale"); continue
        impl = mean([r["px"] for r in grp])
        real = mean([1.0 if r["settled"] == 1 else 0.0 for r in grp])
        xs = [r["e1"] for r in grp]
        print(f"  {nm}: n={len(grp):5d}  impliziert {impl*100:5.2f} %  realisiert {real*100:5.2f} %"
              f"  ->  NO-ROI {mean(xs)*100:+6.2f} %  t={tstat(xs):+6.2f}   (Schwelle t>2,5)")


# ---------------------------------------------------------------- H3
def h3(boards, t_min=T_PRIMARY):
    print("\n=== H3 Niveau-Arbitrage (Summe der Bucketpreise in Minute T) ===")
    sums, arb_ask, arb_bid, n = [], 0, 0, 0
    n_partial = 0
    for (_city, _slug), brd in boards.items():
        last_l, ask_l, bid_l = [], [], []
        for r in brd:
            l, a, bd = prices_at(r.get("tr") or [], t_min)
            if l is None:
                continue
            last_l.append(l)
            if a is not None: ask_l.append(a)
            if bd is not None: bid_l.append(bd)
        if len(last_l) < MIN_BUCKETS:
            continue
        # ZWINGEND: nur VOLLSTAENDIGE Bretter. Sonst misst die Summe die
        # Handelsdichte (nur ~60 % der Buckets handeln binnen 30 min) und nicht
        # das Preisniveau - ein unvollstaendiges Brett summiert zwangslaeufig
        # unter 1,00 und taeuscht eine Ask-Arbitrage vor, die es nicht gibt.
        if len(last_l) < len(brd):
            n_partial += 1
            continue
        n += 1
        s = sum(last_l)
        sums.append(s)
        # Den Korb ausrechnen statt die Summe mit 1,00 zu vergleichen: die
        # Gebuehr faellt auf JEDES Bein an und muss summiert werden, nicht
        # gemittelt. Korb A: alle Buckets YES kaufen -> Kosten sum(ask),
        # Auszahlung genau 1. Korb B: alle NO kaufen -> Kosten N - sum(bid),
        # Auszahlung N - 1 (alle ausser dem Gewinner).
        if len(ask_l) == len(last_l):
            f = sum(FEE_RATE * min(p, 1 - p) for p in ask_l)
            if 1.0 - sum(ask_l) - f > 0:
                arb_ask += 1
        if len(bid_l) == len(last_l):
            N = len(bid_l)
            f = sum(FEE_RATE * min(p, 1 - p) for p in bid_l)
            if (N - 1) - (N - sum(bid_l)) - f > 0:
                arb_bid += 1
    if not sums:
        print("  keine auswertbaren Bretter"); return
    sums.sort()
    print(f"  {n_partial} unvollstaendige Bretter verworfen (nicht alle Buckets bis T gehandelt)")
    print(f"  Bretter: {n}   Summe: Median {median(sums):.3f}  Mittel {mean(sums):.3f}"
          f"  P10 {sums[int(0.10*len(sums))]:.3f}  P90 {sums[int(0.90*len(sums))]:.3f}")
    print(f"  Anteil mit risikoloser Position nach Gebuehr: "
          f"Ask-Seite {arb_ask/n*100:.1f} %   Bid-Seite {arb_bid/n*100:.1f} %")
    print("  (Buchtiefe nicht beruecksichtigt - der Tape zeigt nicht, was ausfuehrbar war.)")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Verzeichnis mit <stadt>.jsonl")
    ap.add_argument("--skip-sens", action="store_true")
    args = ap.parse_args()

    boards, n_rows = load(args.data)
    if not boards:
        print("Keine Daten gefunden."); return 1
    months = defaultdict(int)
    for _k, brd in boards.items():
        months[board_month(brd)] += 1
    print(f"Geladen: {len(boards)} Bretter / {n_rows} Buckets aus "
          f"{len({c for c, _ in boards})} Staedten")
    print("  je Monat: " + "  ".join(f"{m}: {c}" for m, c in sorted(months.items())))

    sigs = h1_signals(boards, T_PRIMARY, THETA_PRIMARY)
    a, b = split(sigs)
    print(f"\n=== H1 Formresiduum ===")
    ok = gates(a, b, sigs)
    print(f"\n  ERGEBNIS H1: {'ALLE GATES GRUEN' if ok else 'GEFALLEN'}")
    if not ok:
        print("  Abbruchregel der Pre-Reg: keine Umparametrisierung.")

    if not args.skip_sens:
        sensitivity(boards)
    h2(boards)
    h3(boards)
    return 0


if __name__ == "__main__":
    sys.exit(main())
