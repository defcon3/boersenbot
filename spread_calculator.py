#!/usr/bin/env python3
"""
Spread Calculator — Analyzes Polymarket vs Bookmaker odds for arbitrage
Matches events, calculates spreads & friction, outputs arbitrage report
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import difflib


# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "spread_calculator.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Config
POLYMARKET_DIR = Path("polymarket_raw")
BOOKMAKER_DIR = Path("bookmaker_raw")
OUTPUT_DIR = Path("arbitrage_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Friction costs (in basis points)
POLYMARKET_TAKER_FEE = 20  # 0.2%
WETTSTEUER_DE = 500  # 5.0% (German betting tax)
BETFAIR_COMMISSION = 50  # 0.5–2% commission on net (conservative: 0.5%)
PINNACLE_MARGIN = 20  # Pinnacle is tight, ~0.2%

# Output CSV fields
CSV_HEADERS = [
    "timestamp",
    "event_name",
    "poly_market_id",
    "poly_yes_price",
    "poly_no_price",
    "bookie_source",
    "bookie_yes_price",
    "bookie_no_price",
    "spread_yes_pct",
    "spread_no_pct",
    "friction_yes_pct",
    "friction_no_pct",
    "net_arb_yes_pct",
    "net_arb_no_pct",
    "best_arb_pct",
    "best_side",
    "arbitrageable",
]


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Calculate string similarity (0.0 to 1.0)"""
    matcher = difflib.SequenceMatcher(None, s1.lower(), s2.lower())
    return matcher.ratio()


def load_latest_snapshot(directory: Path) -> Optional[dict]:
    """Load the most recent snapshot from a directory"""
    snapshots = sorted(directory.glob("snapshot_*.json"), reverse=True)
    if not snapshots:
        logger.warning(f"No snapshots found in {directory}")
        return None

    try:
        with open(snapshots[0]) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading snapshot {snapshots[0]}: {e}")
        return None


def odds_to_implied_prob(odds: float) -> float:
    """Convert decimal odds to implied probability (0–1)"""
    if odds <= 0:
        return 0.0
    return 1.0 / odds


def implied_prob_to_odds(prob: float) -> float:
    """Convert implied probability to decimal odds"""
    if prob <= 0 or prob >= 1:
        return 0.0
    return 1.0 / prob


def match_events(poly_markets: list, bookie_markets: list) -> list:
    """
    Match Polymarket events with bookmaker markets using fuzzy string matching
    Returns: list of (poly_market, bookie_market, match_score) tuples
    """
    matches = []
    for poly in poly_markets:
        poly_title = poly.get("title", "").lower()
        best_bookie = None
        best_score = 0.0

        for bookie in bookie_markets:
            bookie_name = bookie.get("event_name", "") or bookie.get("match_name", "")
            bookie_name = bookie_name.lower()

            score = levenshtein_ratio(poly_title, bookie_name)
            if score > best_score:
                best_score = score
                best_bookie = bookie

        if best_score > 0.5:  # Threshold for reasonable match
            matches.append((poly, best_bookie, best_score))

    return matches


def calculate_spread(poly_yes: float, poly_no: float, bookie_yes: float, bookie_no: float) -> dict:
    """
    Calculate spread between Polymarket and bookmaker odds
    Returns: dict with spread percentages and friction
    """
    if not all([poly_yes, poly_no, bookie_yes, bookie_no]) or any(x <= 0 for x in [poly_yes, poly_no, bookie_yes, bookie_no]):
        return {}

    # Implied probabilities
    poly_yes_prob = odds_to_implied_prob(poly_yes)
    poly_no_prob = odds_to_implied_prob(poly_no)
    bookie_yes_prob = odds_to_implied_prob(bookie_yes)
    bookie_no_prob = odds_to_implied_prob(bookie_no)

    # Raw spreads (in bps)
    spread_yes_bps = abs(bookie_yes_prob - poly_yes_prob) * 10000
    spread_no_bps = abs(bookie_no_prob - poly_no_prob) * 10000

    # Friction (in bps)
    friction_yes_bps = POLYMARKET_TAKER_FEE + WETTSTEUER_DE  # Both sides: Poly fee + Bookie tax
    friction_no_bps = POLYMARKET_TAKER_FEE + WETTSTEUER_DE

    # Net arbitrage opportunity (if bookmaker is > Polymarket)
    net_arb_yes_bps = max(0, spread_yes_bps - friction_yes_bps)
    net_arb_no_bps = max(0, spread_no_bps - friction_no_bps)

    return {
        "spread_yes_pct": spread_yes_bps / 100,
        "spread_no_pct": spread_no_bps / 100,
        "friction_yes_pct": friction_yes_bps / 100,
        "friction_no_pct": friction_no_bps / 100,
        "net_arb_yes_pct": net_arb_yes_bps / 100,
        "net_arb_no_pct": net_arb_no_bps / 100,
        "best_arb_pct": max(net_arb_yes_bps, net_arb_no_bps) / 100,
        "best_side": "YES" if net_arb_yes_bps > net_arb_no_bps else "NO",
    }


def analyze_snapshot(poly_snapshot: dict, bookie_snapshot: dict) -> list:
    """
    Main analysis: match events, calculate spreads, generate report rows
    """
    rows = []
    timestamp = poly_snapshot.get("timestamp", datetime.utcnow().isoformat())

    # Extract markets
    poly_markets = poly_snapshot.get("markets", [])
    bookie_markets = (
        bookie_snapshot.get("betfair", []) + bookie_snapshot.get("pinnacle", [])
    )

    logger.info(f"Analyzing {len(poly_markets)} Poly markets vs {len(bookie_markets)} bookie markets")

    # Match events
    matches = match_events(poly_markets, bookie_markets)
    logger.info(f"Found {len(matches)} potential matches")

    # Calculate spreads for each match
    for poly, bookie, match_score in matches:
        poly_id = poly.get("market_id", "")
        poly_title = poly.get("title", "")
        poly_yes = poly.get("yes_price")
        poly_no = poly.get("no_price")

        bookie_source = "Betfair" if "runners" in bookie else "Pinnacle"
        event_name = bookie.get("event_name", "") or bookie.get("match_name", "")

        # Extract bookie odds (simplified; adapt based on structure)
        bookie_yes = None
        bookie_no = None
        if "runners" in bookie:  # Betfair format
            runners = bookie.get("runners", [])
            if len(runners) >= 2:
                bookie_yes = runners[0].get("back_odds")
                bookie_no = runners[1].get("back_odds")
        elif "periods" in bookie:  # Pinnacle format
            periods = bookie.get("periods", [])
            if periods and "moneyline" in periods[0]:
                ml = periods[0].get("moneyline", {})
                bookie_yes = ml.get("home")
                bookie_no = ml.get("away")

        # Calculate spread
        spread = calculate_spread(poly_yes, poly_no, bookie_yes, bookie_no)
        if not spread:
            continue

        # Create row
        row = {
            "timestamp": timestamp,
            "event_name": event_name or poly_title,
            "poly_market_id": poly_id,
            "poly_yes_price": poly_yes,
            "poly_no_price": poly_no,
            "bookie_source": bookie_source,
            "bookie_yes_price": bookie_yes,
            "bookie_no_price": bookie_no,
            "spread_yes_pct": f"{spread['spread_yes_pct']:.2f}",
            "spread_no_pct": f"{spread['spread_no_pct']:.2f}",
            "friction_yes_pct": f"{spread['friction_yes_pct']:.2f}",
            "friction_no_pct": f"{spread['friction_no_pct']:.2f}",
            "net_arb_yes_pct": f"{spread['net_arb_yes_pct']:.2f}",
            "net_arb_no_pct": f"{spread['net_arb_no_pct']:.2f}",
            "best_arb_pct": f"{spread['best_arb_pct']:.2f}",
            "best_side": spread["best_side"],
            "arbitrageable": "YES" if spread["best_arb_pct"] > 0.5 else "NO",  # > 0.5% = noteworthy
        }
        rows.append(row)

    return rows


def save_csv(rows: list, filename: str = "arbitrage_spreads.csv"):
    """Save rows to CSV"""
    filepath = OUTPUT_DIR / filename
    try:
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved {len(rows)} rows to {filepath}")
    except Exception as e:
        logger.error(f"Error saving CSV: {e}")


def main():
    """Load latest snapshots, analyze, save results"""
    logger.info("=== Spread Calculator Start ===")

    poly_snap = load_latest_snapshot(POLYMARKET_DIR)
    bookie_snap = load_latest_snapshot(BOOKMAKER_DIR)

    if not poly_snap or not bookie_snap:
        logger.error("Missing snapshots — cannot analyze")
        return

    logger.info(f"Poly snapshot: {poly_snap.get('timestamp')}")
    logger.info(f"Bookie snapshot: {bookie_snap.get('timestamp')}")

    rows = analyze_snapshot(poly_snap, bookie_snap)
    logger.info(f"Generated {len(rows)} analysis rows")

    # Filter for noteworthy arbs
    arbs = [r for r in rows if r["arbitrageable"] == "YES"]
    logger.info(f"Found {len(arbs)} arbitrage opportunities (> 0.5% net)")

    if arbs:
        logger.info("\n=== Top Arbitrage Opportunities ===")
        for arb in sorted(arbs, key=lambda x: float(x["best_arb_pct"]), reverse=True)[:5]:
            logger.info(
                f"{arb['event_name']}: {arb['best_arb_pct']}% ({arb['best_side']}) via {arb['bookie_source']}"
            )

    save_csv(rows)
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
