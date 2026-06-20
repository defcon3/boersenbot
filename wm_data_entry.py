#!/usr/bin/env python3
"""
WM Data Entry Form — Web interface for entering Jupiter Prediction Market quotes
Flask app with simple form to log WM matches and quotes
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "wm_data_entry.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Config
DATA_DIR = Path("wm_matches_data")
DATA_DIR.mkdir(exist_ok=True)
DATA_FILE = DATA_DIR / "wm_matches.json"

# Flask app
app = Flask(__name__, template_folder="templates")
app.config["JSON_SORT_KEYS"] = False


def load_matches():
    """Load matches from JSON file"""
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE) as f:
                data = json.load(f)
                return data.get("matches", [])
        except Exception as e:
            logger.error(f"Error loading matches: {e}")
    return []


def save_matches(matches):
    """Save matches to JSON file"""
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_matches": len(matches),
        "matches": matches,
    }
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(matches)} matches")
        return True
    except Exception as e:
        logger.error(f"Error saving matches: {e}")
        return False


@app.route("/")
def index():
    """Main form page"""
    matches = load_matches()
    return render_template("wm_entry_form.html", matches=matches)


@app.route("/api/matches", methods=["GET"])
def get_matches():
    """Get all matches (JSON)"""
    matches = load_matches()
    return jsonify(matches)


@app.route("/api/matches", methods=["POST"])
def add_match():
    """Add a new match"""
    try:
        data = request.get_json()

        # Validate
        required_fields = ["event_name", "yes_price", "no_price"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        # Create match entry
        match = {
            "id": len(load_matches()) + 1,
            "event_name": data.get("event_name", "").strip(),
            "yes_price": float(data.get("yes_price", 0)),
            "no_price": float(data.get("no_price", 0)),
            "draw_price": float(data.get("draw_price", 0)) if data.get("draw_price") else None,
            "timestamp": datetime.utcnow().isoformat(),
            "notes": data.get("notes", ""),
        }

        # Validate prices sum to ~1.0 (allow some rounding error)
        prices = [match["yes_price"], match["no_price"]]
        if match["draw_price"]:
            prices.append(match["draw_price"])

        total = sum(prices)
        if total < 0.95 or total > 1.05:
            logger.warning(f"Prices don't sum to 1.0: {total} (match: {match['event_name']})")

        # Save
        matches = load_matches()
        matches.append(match)
        save_matches(matches)

        logger.info(f"Added match: {match['event_name']}")
        return jsonify(match), 201

    except Exception as e:
        logger.error(f"Error adding match: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/matches/<int:match_id>", methods=["DELETE"])
def delete_match(match_id):
    """Delete a match"""
    try:
        matches = load_matches()
        matches = [m for m in matches if m.get("id") != match_id]
        save_matches(matches)
        logger.info(f"Deleted match ID {match_id}")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error deleting match: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/export", methods=["GET"])
def export_data():
    """Export data as CSV"""
    import csv
    import io

    matches = load_matches()

    # Create CSV
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["event_name", "yes_price", "no_price", "draw_price", "notes"],
    )
    writer.writeheader()

    for match in matches:
        writer.writerow(
            {
                "event_name": match.get("event_name"),
                "yes_price": match.get("yes_price"),
                "no_price": match.get("no_price"),
                "draw_price": match.get("draw_price", ""),
                "notes": match.get("notes", ""),
            }
        )

    return output.getvalue(), 200, {"Content-Disposition": "attachment; filename=wm_matches.csv"}


if __name__ == "__main__":
    logger.info("WM Data Entry Server starting on http://localhost:5002")
    logger.info("Open http://localhost:5002 in your browser")
    app.run(debug=True, port=5002, use_reloader=False)
