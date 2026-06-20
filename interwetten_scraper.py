#!/usr/bin/env python3
"""
Interwetten Odds Scraper — Fetch tennis odds from Interwetten website
Uses browser automation (Selenium) to get live odds for arbitrage analysis
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup


# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "interwetten.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Config
DATA_DIR = Path("bookmaker_raw")
DATA_DIR.mkdir(exist_ok=True)

POLL_INTERVAL = 300  # 5 minutes

# Interwetten URLs (Tennis betting)
INTERWETTEN_TENNIS_URL = "https://www.interwetten.com/de/sports/1/8"  # Tennis market


class InterwettenScraper:
    """Scrapes Interwetten odds for tennis matches"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

    def fetch_tennis_odds(self):
        """
        Fetch tennis odds from Interwetten
        Returns: list of match dicts with odds
        """
        try:
            resp = self.session.get(INTERWETTEN_TENNIS_URL, timeout=15)
            resp.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(resp.text, "html.parser")

            # TODO: Adjust selectors based on actual Interwetten HTML structure
            # This is a placeholder — you'll need to inspect the page and find the right CSS selectors
            matches = []

            # Example: find all match containers
            match_containers = soup.find_all("div", class_="event-row")  # Adjust selector

            for container in match_containers:
                try:
                    # Extract match names
                    match_name_elem = container.find("span", class_="event-name")
                    if not match_name_elem:
                        continue

                    match_name = match_name_elem.get_text(strip=True)

                    # Extract odds (1 = home/player1, 2 = away/player2)
                    odds_1 = container.find("span", class_="odds-1")
                    odds_2 = container.find("span", class_="odds-2")

                    if odds_1 and odds_2:
                        match = {
                            "match_name": match_name,
                            "player1_odds": float(odds_1.get_text(strip=True)),
                            "player2_odds": float(odds_2.get_text(strip=True)),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                        matches.append(match)
                except Exception as e:
                    logger.warning(f"Error parsing match: {e}")
                    continue

            logger.info(f"Fetched {len(matches)} tennis matches from Interwetten")
            return matches

        except requests.Timeout:
            logger.error("Interwetten request timeout")
            return []
        except requests.RequestException as e:
            logger.error(f"Interwetten request error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error scraping Interwetten: {e}")
            return []

    def poll_once(self):
        """Single poll cycle"""
        logger.info("=== Interwetten poll cycle start ===")
        timestamp = datetime.utcnow().isoformat()

        matches = self.fetch_tennis_odds()

        snapshot = {
            "timestamp": timestamp,
            "source": "Interwetten",
            "matches": matches,
        }

        # Save snapshot
        filename = DATA_DIR / f"interwetten_snapshot_{timestamp.replace(':', '-').replace('.', '_')}.json"
        try:
            with open(filename, "w") as f:
                json.dump(snapshot, f, indent=2)
            logger.info(f"Saved {len(matches)} matches to {filename}")
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")

        logger.info("=== Interwetten poll cycle done ===")

    def run_forever(self):
        """Main loop"""
        import time

        try:
            while True:
                try:
                    self.poll_once()
                except Exception as e:
                    logger.error(f"Poll failed: {e}")

                logger.info(f"Sleeping {POLL_INTERVAL}s...")
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Shutting down...")


def main():
    logger.info("Interwetten Scraper starting...")
    logger.info(f"Target: {INTERWETTEN_TENNIS_URL}")
    logger.warning("⚠️ NOTE: Selectors need to be adjusted based on actual Interwetten page structure")
    logger.warning("Run browser inspection (F12) to find correct CSS selectors for matches & odds")

    scraper = InterwettenScraper()
    scraper.run_forever()


if __name__ == "__main__":
    main()
