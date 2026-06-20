#!/usr/bin/env python3
"""
The Odds API Logger — Fetch sports betting odds for arbitrage analysis
Pulls live odds from 40+ bookmakers (Bet365, Tipico, Pinnacle, etc.)
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import requests


# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "odds_api.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Config
API_KEY = "139312810e0e8728f6df65e3a0e00aad"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DATA_DIR = Path("odds_api_raw")
DATA_DIR.mkdir(exist_ok=True)

# Load Jupiter matches from CSV
JUPITER_CSV = Path("C:/Users/defco/Desktop/wm_matches.csv")

# Interesting bookmakers (German/Europe)
BOOKMAKERS = [
    "bet365",
    "tipico",
    "pinnacle",
    "betfair_exchange",
    "bwin",
    "unibet",
]


class OddsAPILogger:
    """Fetches odds from The Odds API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def fetch_sports(self):
        """Fetch available sports"""
        try:
            url = f"{ODDS_API_BASE}/sports"
            params = {"apiKey": self.api_key}
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Error fetching sports: {e}")
            return []

    def fetch_odds(self, sport_key: str, region: str = "eu"):
        """
        Fetch live odds for a sport
        sport_key: e.g. "soccer_fifa_world_cup"
        """
        try:
            url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
            params = {
                "apiKey": self.api_key,
                "regions": region,
                "markets": "h2h",  # Only head-to-head for now
            }
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Error fetching odds for {sport_key}: {e}")
            logger.error(f"  Params: {params}")
            return None

    def find_matching_games(self, jupiter_matches: list, api_odds: dict) -> list:
        """
        Match Jupiter games with API odds
        Returns list of (jupiter_match, api_match) tuples
        """
        matches = []

        if not api_odds or "events" not in api_odds:
            return matches

        api_events = api_odds.get("events", [])

        for jupiter in jupiter_matches:
            event_name = jupiter.get("event_name", "").lower()

            # Parse teams from event_name (format: "Team1 - Team2")
            teams = [t.strip() for t in event_name.split("-")]
            if len(teams) != 2:
                logger.warning(f"Could not parse teams from: {event_name}")
                continue

            team1, team2 = teams

            # Try to find matching API event
            for api_event in api_events:
                api_name = api_event.get("home_team", "") + " vs " + api_event.get("away_team", "")
                api_name_lower = api_name.lower()

                # Simple fuzzy match: check if both teams are in API event name
                if team1 in api_name_lower and team2 in api_name_lower:
                    matches.append((jupiter, api_event))
                    logger.info(f"Matched: {event_name} <-> {api_name}")
                    break

        return matches

    def extract_best_odds(self, api_event: dict) -> dict:
        """
        Extract best odds from API event for each bookmaker.

        WICHTIG: Die Odds API garantiert KEINE feste Reihenfolge der outcomes.
        Wir ordnen daher per outcome["name"] gegen home_team/away_team zu,
        statt blind outcomes[0]=home anzunehmen.

        Returns: {
            "pinnacle": {"home": 1.51, "away": 6.6, "draw": 4.8},
            ...
        }
        """
        bookmaker_odds = {}

        home_team = (api_event.get("home_team") or "").strip().lower()
        away_team = (api_event.get("away_team") or "").strip().lower()

        # Bookmakers can be a list or dict
        bookmakers = api_event.get("bookmakers", [])
        if isinstance(bookmakers, dict):
            bookmakers = bookmakers.items()
        elif isinstance(bookmakers, list):
            bookmakers = [(bm.get("key"), bm) for bm in bookmakers]

        for bookmaker_key, bookmaker_data in bookmakers:
            if isinstance(bookmaker_data, str):
                continue  # Skip if it's just a key string

            odds_dict = {}

            for market in bookmaker_data.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                for outcome in market.get("outcomes", []):
                    name = (outcome.get("name") or "").strip().lower()
                    price = outcome.get("price")

                    if name == "draw":
                        odds_dict["draw"] = price
                    elif home_team and name == home_team:
                        odds_dict["home"] = price
                    elif away_team and name == away_team:
                        odds_dict["away"] = price

            if odds_dict:
                bookmaker_odds[bookmaker_key] = odds_dict

        return bookmaker_odds

    def save_snapshot(self, jupiter_matches: list, bookmaker_odds: list):
        """Save combined Jupiter + Bookmaker data"""
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "jupiter_count": len(jupiter_matches),
            "bookmaker_count": len(bookmaker_odds),
            "data": {
                "jupiter": jupiter_matches,
                "bookmakers": bookmaker_odds,
            },
        }

        filename = DATA_DIR / f"snapshot_{datetime.utcnow().isoformat().replace(':', '-')}.json"
        try:
            with open(filename, "w") as f:
                json.dump(snapshot, f, indent=2)
            logger.info(f"Saved snapshot: {filename}")
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")

    def run(self):
        """Main workflow"""
        logger.info("=== Odds API Logger Start ===")

        # Load Jupiter matches
        if not JUPITER_CSV.exists():
            logger.error(f"Jupiter CSV not found: {JUPITER_CSV}")
            return

        jupiter_matches = []
        try:
            with open(JUPITER_CSV) as f:
                reader = csv.DictReader(f)
                jupiter_matches = list(reader)
            logger.info(f"Loaded {len(jupiter_matches)} Jupiter matches")
        except Exception as e:
            logger.error(f"Error loading Jupiter CSV: {e}")
            return

        # Fetch sports from API
        logger.info("Fetching available sports...")
        sports = self.fetch_sports()
        logger.info(f"Found {len(sports)} sports")

        # Find soccer/FIFA World Cup
        fifa_sport = None
        for sport in sports:
            key = sport.get("key", "").lower()
            # Look for soccer + world_cup
            if "soccer" in key and "world_cup" in key:
                fifa_sport = sport
                logger.info(f"Found FIFA World Cup sport: {sport.get('key')}")
                break

        if not fifa_sport:
            # Try common keys
            for test_key in ["soccer_fifa_world_cup", "soccer_world_cup", "soccer"]:
                logger.warning(f"Trying fallback: {test_key}")
                fifa_sport = {"key": test_key}
                break

        # Fetch odds for FIFA
        logger.info(f"Fetching odds for {fifa_sport.get('key')}...")
        api_odds = self.fetch_odds(fifa_sport.get("key"))

        if not api_odds:
            logger.error("Failed to fetch odds from API")
            return

        # Handle both dict and list response formats
        if isinstance(api_odds, dict):
            events = api_odds.get("events", [])
        elif isinstance(api_odds, list):
            events = api_odds
        else:
            logger.error(f"Unexpected API response format: {type(api_odds)}")
            return

        logger.info(f"Got {len(events)} events from API")

        # Wrap in dict format for consistency
        api_odds = {"events": events}

        # Match games
        matched = self.find_matching_games(jupiter_matches, api_odds)
        logger.info(f"Matched {len(matched)} games between Jupiter and Bookmakers")

        # Extract bookmaker odds and save
        bookmaker_odds_list = []
        for jupiter, api_event in matched:
            bm_odds = self.extract_best_odds(api_event)
            if bm_odds:
                bookmaker_odds_list.append({
                    "jupiter_event": jupiter.get("event_name"),
                    "api_event": f"{api_event.get('home_team')} vs {api_event.get('away_team')}",
                    "bookmakers": bm_odds,
                })

        self.save_snapshot(jupiter_matches, bookmaker_odds_list)
        logger.info("=== Done ===")

        return matched, bookmaker_odds_list


def main():
    logger.info("The Odds API Logger starting...")
    logger.info(f"API Key: {API_KEY[:10]}...")

    logger_obj = OddsAPILogger(API_KEY)
    matched, bm_odds = logger_obj.run()

    if bm_odds:
        logger.info("\n=== Summary ===")
        for item in bm_odds:
            logger.info(f"\n{item['jupiter_event']}")
            for bm, odds in item.get("bookmakers", {}).items():
                logger.info(f"  {bm}: {odds}")


if __name__ == "__main__":
    main()
