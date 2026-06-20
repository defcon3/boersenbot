#!/usr/bin/env python3
"""
Live Arbitrage Dashboard (v2) — angepasst an wahrscheinlichkeits-basierte Felder.

Liest:
  arbitrage_results/quote_comparisons.json   (Felder: event, side,
      jupiter_prob_pct, reference, reference_prob_pct, edge_pp, abs_edge_pp)
  arbitrage_results/arbitrage_opportunities.json (Felder: event, type,
      total_implied_prob, profit_pct, legs)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, render_template, jsonify

# Windows-Konsole auf UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "dashboard.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")

RESULTS_DIR = Path("arbitrage_results")


def _load(filename, key):
    path = RESULTS_DIR / filename
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get(key, [])
    except Exception as e:
        logger.error(f"Fehler beim Laden von {filename}: {e}")
        return []


def load_comparisons():
    return _load("quote_comparisons.json", "comparisons")


def load_arbitrage():
    return _load("arbitrage_opportunities.json", "opportunities")


@app.route("/")
def index():
    return render_template("arbitrage_dashboard.html")


@app.route("/api/stats")
def get_stats():
    comparisons = load_comparisons()
    arbitrage = load_arbitrage()

    if not comparisons:
        return jsonify({
            "total_comparisons": 0,
            "total_arbitrage": 0,
            "best_edge_pp": 0,
            "best_edge_event": "",
            "events": 0,
            "message": "Keine Daten. Erst arbitrage_calculator.py laufen lassen.",
        })

    best = max(comparisons, key=lambda x: x.get("abs_edge_pp", 0))

    return jsonify({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_comparisons": len(comparisons),
        "total_arbitrage": len(arbitrage),
        "best_edge_pp": best.get("edge_pp", 0),
        "best_edge_event": best.get("event", ""),
        "best_edge_side": best.get("side", ""),
        "best_edge_reference": best.get("reference", ""),
        "events": len(set(c.get("event", "") for c in comparisons)),
    })


@app.route("/api/comparisons")
def get_comparisons():
    comparisons = load_comparisons()

    by_event = {}
    for comp in comparisons:
        by_event.setdefault(comp.get("event", "Unknown"), []).append(comp)

    return jsonify({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(comparisons),
        "by_event": by_event,
        "top": sorted(comparisons, key=lambda x: x.get("abs_edge_pp", 0), reverse=True),
    })


@app.route("/api/arbitrage")
def get_arbitrage():
    arbitrage = load_arbitrage()
    return jsonify({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(arbitrage),
        "opportunities": sorted(arbitrage, key=lambda x: x.get("profit_pct", 0), reverse=True),
    })


if __name__ == "__main__":
    logger.info("Arbitrage Dashboard v2 auf http://localhost:5003")
    app.run(debug=True, port=5003, use_reloader=False)
