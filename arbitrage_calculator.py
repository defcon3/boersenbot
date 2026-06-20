#!/usr/bin/env python3
"""
Arbitrage Calculator (v2) — korrekter Vergleich Jupiter vs Buchmacher.

Kernfix gegenüber v1:
  - Jupiter-Preise (0..1) SIND bereits implizite Wahrscheinlichkeiten.
  - Buchmacher liefert DEZIMALQUOTEN -> Wahrscheinlichkeit = 1 / quote.
  - Beide werden vor dem Vergleich ins gleiche Format (Wahrscheinlichkeit)
    gebracht. Erst dann ist ein Vergleich sinnvoll.

Echte Cross-Platform-Arbitrage besteht nur, wenn man pro Outcome die jeweils
BESTE Quote (= niedrigste implizite Wahrscheinlichkeit) wählt und die Summe
dieser besten Wahrscheinlichkeiten < 1.0 ist.
"""

import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Windows-Konsole auf UTF-8 (sonst Crash bei Σ, ö etc.)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "arbitrage.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Config
JUPITER_CSV = Path("C:/Users/defco/Desktop/wm_matches.csv")
ODDS_SNAPSHOT = Path("odds_api_raw")
OUTPUT_DIR = Path("arbitrage_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Pinnacle gilt als "sharpster" Buchmacher -> guter fairer Referenzpreis.
REFERENCE_BOOKMAKER = "pinnacle"

# Arbitrage gilt erst ab diesem Netto-Profit als meldenswert (in %).
ARB_THRESHOLD_PCT = 0.5


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------------
# Umrechnungen
# ----------------------------------------------------------------------------
def jupiter_to_prob(price):
    """Jupiter-Preis (0..1) IST die implizite Wahrscheinlichkeit."""
    if price is None or price <= 0:
        return None
    return float(price)


def decimal_to_prob(odds):
    """Dezimalquote -> implizite Wahrscheinlichkeit."""
    if odds is None or odds <= 0:
        return None
    return 1.0 / float(odds)


def prob_to_decimal(prob):
    """Wahrscheinlichkeit -> faire Dezimalquote."""
    if prob is None or prob <= 0:
        return None
    return 1.0 / prob


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return None
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


# Quoten, deren implizite Wahrscheinlichkeit um mehr als diesen Faktor vom
# Median aller Quellen abweicht, gelten als Datenfehler (z.B. vertauschtes
# Mapping bei einem einzelnen Buchmacher) und werden verworfen.
OUTLIER_FACTOR = 1.5


def robust_best(quotes):
    """
    quotes: Liste von (decimal, prob, source).
    Filtert Ausreißer per Median-Konsens und gibt die beste (höchste) Quote
    unter den verbleibenden zurück.
    """
    valid = [q for q in quotes if q[0] and q[1]]
    if not valid:
        return None
    med = median([q[1] for q in valid])
    if med and med > 0:
        kept = [q for q in valid if med / OUTLIER_FACTOR <= q[1] <= med * OUTLIER_FACTOR]
        valid = kept or valid
    return max(valid, key=lambda q: q[0])  # höchste Dezimalquote


# ----------------------------------------------------------------------------
# Laden
# ----------------------------------------------------------------------------
def load_jupiter_matches():
    """Jupiter-Spiele aus CSV. Format: 'home - away', Preise = Wahrscheinlichkeiten."""
    matches = {}
    try:
        with open(JUPITER_CSV) as f:
            for row in csv.DictReader(f):
                event = row.get("event_name", "").strip().lower()
                if not event:
                    continue
                matches[event] = {
                    "home_prob": jupiter_to_prob(float(row["yes_price"])) if row.get("yes_price") else None,
                    "away_prob": jupiter_to_prob(float(row["no_price"])) if row.get("no_price") else None,
                    "draw_prob": jupiter_to_prob(float(row["draw_price"])) if row.get("draw_price") else None,
                }
        logger.info(f"Loaded {len(matches)} Jupiter-Spiele")
        return matches
    except Exception as e:
        logger.error(f"Fehler beim Laden der Jupiter-CSV: {e}")
        return {}


def load_latest_snapshot():
    """Neuesten Odds-API-Snapshot laden."""
    if not ODDS_SNAPSHOT.exists():
        return None
    snaps = sorted(ODDS_SNAPSHOT.glob("*.json"), reverse=True)
    if not snaps:
        return None
    try:
        with open(snaps[0]) as f:
            logger.info(f"Snapshot: {snaps[0].name}")
            return json.load(f)
    except Exception as e:
        logger.error(f"Fehler beim Laden des Snapshots: {e}")
        return None


# ----------------------------------------------------------------------------
# Hilfen
# ----------------------------------------------------------------------------
def split_teams(event_name, sep):
    parts = [p.strip().lower() for p in event_name.split(sep)]
    return parts if len(parts) == 2 else None


def orientation_matches(jupiter_event, api_event_str):
    """
    Prüft, ob Jupiters team1 (=home) auch beim API-Event home ist.
    api_event_str hat Form 'Home vs Away'. Gibt True/False/None zurück.
    """
    jt = split_teams(jupiter_event, "-")
    at = split_teams(api_event_str, " vs ")
    if not jt or not at:
        return None
    jhome = jt[0]
    ahome = at[0]
    # Token-Überlappung: ist ein Wort aus Jupiter-home im API-home?
    jtokens = set(jhome.split())
    return any(tok and tok in ahome for tok in jtokens)


# ----------------------------------------------------------------------------
# Kernanalyse
# ----------------------------------------------------------------------------
def analyze(jupiter_matches, snapshot):
    comparisons = []   # pro Outcome: Jupiter-Wahrsch. vs bester Buchmacher
    arbitrages = []    # echte cross-platform Arbitrage pro Spiel

    bm_data = snapshot.get("data", {}).get("bookmakers", []) if snapshot else []

    for entry in bm_data:
        event_name = entry.get("jupiter_event", "").strip().lower()
        api_event_str = entry.get("api_event", "")

        jupiter = None
        for jev, jdata in jupiter_matches.items():
            if jev in event_name or event_name in jev:
                jupiter = jdata
                break
        if not jupiter:
            continue

        # Reihenfolge home/away abgleichen; ggf. spiegeln.
        swap = orientation_matches(event_name, api_event_str) is False
        if swap:
            logger.warning(f"  Home/Away gespiegelt für '{event_name}' (API: '{api_event_str}')")

        bookmakers = entry.get("bookmakers", {})
        if not bookmakers:
            continue

        # Pro Outcome ALLE Quoten (decimal, prob, source) sammeln.
        # Ausreißer werden später per robust_best() gefiltert.
        all_quotes = {"home": [], "away": [], "draw": []}

        def collect(outcome, decimal, source):
            if decimal and decimal > 0:
                all_quotes[outcome].append((decimal, decimal_to_prob(decimal), source))

        # 1) Jupiter als Quelle (Wahrscheinlichkeit -> faire Quote)
        jup_outcomes = {
            "home": jupiter.get("home_prob"),
            "away": jupiter.get("away_prob"),
            "draw": jupiter.get("draw_prob"),
        }
        for oc, prob in jup_outcomes.items():
            collect(oc, prob_to_decimal(prob), "jupiter")

        # 2) Buchmacher (Dezimalquoten); ggf. home/away spiegeln
        ref_probs = None
        for bm_name, bm_odds in bookmakers.items():
            if not bm_odds:
                continue
            home = bm_odds.get("home")
            away = bm_odds.get("away")
            draw = bm_odds.get("draw")
            if swap:
                home, away = away, home

            collect("home", home, bm_name)
            collect("away", away, bm_name)
            collect("draw", draw, bm_name)

            if bm_name == REFERENCE_BOOKMAKER:
                ref_probs = {
                    "home": decimal_to_prob(home),
                    "away": decimal_to_prob(away),
                    "draw": decimal_to_prob(draw),
                }

        # Beste Quote pro Outcome nach Ausreißer-Filter.
        best = {}
        for oc in ("home", "away", "draw"):
            rb = robust_best(all_quotes[oc])
            best[oc] = (
                {"decimal": rb[0], "prob": rb[1], "source": rb[2]}
                if rb else {"decimal": None, "prob": None, "source": None}
            )

        # Fallback-Referenz: Durchschnitt aller Buchmacher, falls Pinnacle fehlt.
        if ref_probs is None:
            acc = {"home": [], "away": [], "draw": []}
            for bm_name, bm_odds in bookmakers.items():
                if not bm_odds:
                    continue
                home, away, draw = bm_odds.get("home"), bm_odds.get("away"), bm_odds.get("draw")
                if swap:
                    home, away = away, home
                for oc, val in (("home", home), ("away", away), ("draw", draw)):
                    p = decimal_to_prob(val)
                    if p is not None:
                        acc[oc].append(p)
            ref_probs = {oc: (sum(v) / len(v) if v else None) for oc, v in acc.items()}
            ref_label = "markt-Ø"
        else:
            ref_label = REFERENCE_BOOKMAKER

        # --- Outcome-Vergleich: Jupiter-Wahrsch. vs Referenz-Buchmacher ---
        for oc, label in (("home", "HOME"), ("away", "AWAY"), ("draw", "DRAW")):
            jp = jup_outcomes.get(oc)
            rp = ref_probs.get(oc)
            if jp is None or rp is None:
                continue
            edge_pp = (jp - rp) * 100  # Differenz in Prozentpunkten
            comparisons.append({
                "event": event_name,
                "side": label,
                "jupiter_prob_pct": round(jp * 100, 1),
                "reference": ref_label,
                "reference_prob_pct": round(rp * 100, 1),
                "edge_pp": round(edge_pp, 2),  # >0: Jupiter hält Outcome für wahrscheinlicher
                "abs_edge_pp": round(abs(edge_pp), 2),
            })

        # --- Echte Cross-Platform-Arbitrage ---
        # Nur wenn alle drei (oder bei 2-Way zwei) Outcomes eine beste Quote haben.
        legs = {oc: best[oc] for oc in ("home", "away", "draw") if best[oc]["prob"] is not None}
        if len(legs) >= 2:
            total_prob = sum(leg["prob"] for leg in legs.values())
            if total_prob < 1.0 - (ARB_THRESHOLD_PCT / 100.0):
                profit_pct = (1.0 / total_prob - 1.0) * 100
                arbitrages.append({
                    "event": event_name,
                    "type": f"{len(legs)}-way",
                    "total_implied_prob": round(total_prob, 4),
                    "profit_pct": round(profit_pct, 2),
                    "legs": {
                        oc: {
                            "decimal": round(leg["decimal"], 3),
                            "implied_prob_pct": round(leg["prob"] * 100, 1),
                            "source": leg["source"],
                        }
                        for oc, leg in legs.items()
                    },
                })

    return comparisons, arbitrages


# ----------------------------------------------------------------------------
# Speichern
# ----------------------------------------------------------------------------
def save_results(comparisons, arbitrages):
    comparisons = sorted(comparisons, key=lambda x: x["abs_edge_pp"], reverse=True)
    arbitrages = sorted(arbitrages, key=lambda x: x["profit_pct"], reverse=True)

    with open(OUTPUT_DIR / "quote_comparisons.json", "w") as f:
        json.dump({"timestamp": now_iso(), "total": len(comparisons),
                   "comparisons": comparisons}, f, indent=2)

    with open(OUTPUT_DIR / "arbitrage_opportunities.json", "w") as f:
        json.dump({"timestamp": now_iso(), "total": len(arbitrages),
                   "opportunities": arbitrages}, f, indent=2)

    if comparisons:
        with open(OUTPUT_DIR / "quote_comparisons.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(comparisons[0].keys()))
            w.writeheader()
            w.writerows(comparisons)

    logger.info(f"Gespeichert: {len(comparisons)} Vergleiche, {len(arbitrages)} Arbitrage-Chancen")


# ----------------------------------------------------------------------------
def main():
    logger.info("=== Arbitrage Calculator v2 Start ===")

    jupiter_matches = load_jupiter_matches()
    snapshot = load_latest_snapshot()
    if not jupiter_matches or not snapshot:
        logger.error("Fehlende Daten, Abbruch.")
        return

    comparisons, arbitrages = analyze(jupiter_matches, snapshot)

    logger.info("\n=== Ergebnis ===")
    logger.info(f"Outcome-Vergleiche: {len(comparisons)}")
    logger.info(f"Echte Arbitrage-Chancen (> {ARB_THRESHOLD_PCT}% Profit): {len(arbitrages)}")

    if comparisons:
        logger.info(f"\n=== Größte Bewertungs-Unterschiede (Jupiter vs Referenz, Prozentpunkte) ===")
        for c in sorted(comparisons, key=lambda x: x["abs_edge_pp"], reverse=True)[:10]:
            richtung = "Jupiter höher" if c["edge_pp"] > 0 else "Buchmacher höher"
            logger.info(
                f"  {c['event']:24s} {c['side']:5s}: "
                f"Jupiter {c['jupiter_prob_pct']:5.1f}%  vs  "
                f"{c['reference']} {c['reference_prob_pct']:5.1f}%  "
                f"=> {c['edge_pp']:+.1f} pp ({richtung})"
            )

    if arbitrages:
        logger.info(f"\n=== Echte Arbitrage ===")
        for a in arbitrages:
            logger.info(f"  {a['event']} [{a['type']}]: +{a['profit_pct']:.2f}% "
                        f"(Σ impl. Wahrsch. = {a['total_implied_prob']:.3f})")
            for oc, leg in a["legs"].items():
                logger.info(f"      {oc:5s}: {leg['decimal']:.2f} @ {leg['source']} "
                            f"({leg['implied_prob_pct']:.1f}%)")
    else:
        logger.info("\n  Keine echte Arbitrage — wie erwartet bei effizienten Märkten.")

    save_results(comparisons, arbitrages)
    logger.info("\n=== Fertig ===")


if __name__ == "__main__":
    main()
