#!/usr/bin/env python3
"""
Interwetten Odds Scraper — Selenium-based (bypasses blocking)
Uses Selenium with Chrome to load and scrape live tennis odds
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options


# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "interwetten_selenium.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Config
DATA_DIR = Path("bookmaker_raw")
DATA_DIR.mkdir(exist_ok=True)

POLL_INTERVAL = 300  # 5 minutes

# Interwetten URLs
INTERWETTEN_TENNIS_URL = "https://www.interwetten.com/de/sports/1/8"  # Tennis


class InterwettenSeleniumScraper:
    """Scrapes Interwetten odds using Selenium + Chrome"""

    def __init__(self):
        """Initialize Chrome WebDriver"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Headless mode for faster loading
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )

        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Chrome WebDriver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            self.driver = None

    def fetch_tennis_odds(self):
        """
        Load Interwetten tennis page and extract match odds
        Returns: list of match dicts
        """
        if not self.driver:
            logger.error("WebDriver not initialized")
            return []

        try:
            logger.info(f"Loading {INTERWETTEN_TENNIS_URL}...")
            self.driver.get(INTERWETTEN_TENNIS_URL)

            # Wait for page to load (wait for matches to appear)
            wait = WebDriverWait(self.driver, 15)

            # Try to find match elements (adjust selectors as needed)
            # Interwetten typically uses divs with data-attributes for matches
            try:
                wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "event")))
            except Exception:
                logger.warning("Timeout waiting for events to load")

            # Get all match elements
            matches = []

            # Try different selector patterns (Interwetten changes structure frequently)
            selectors_to_try = [
                ("//div[contains(@class, 'event')]", "class='event'"),
                ("//div[contains(@class, 'match')]", "class='match'"),
                ("//tr[contains(@class, 'event-row')]", "class='event-row'"),
            ]

            match_elements = None
            for xpath, desc in selectors_to_try:
                try:
                    match_elements = self.driver.find_elements(By.XPATH, xpath)
                    if match_elements:
                        logger.info(f"Found {len(match_elements)} matches using {desc}")
                        break
                except Exception:
                    continue

            if not match_elements:
                logger.warning("No match elements found with any selector")
                # Dump page source for debugging
                page_source = self.driver.page_source
                debug_file = Path("interwetten_debug.html")
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(page_source)
                logger.warning(f"Dumped page source to {debug_file} for inspection")

                if "Tenis" in page_source or "Tennis" in page_source:
                    logger.info("Page contains tennis content, but extraction failed")
                return []

            # Extract odds from each match
            for idx, elem in enumerate(match_elements[:20]):  # Limit to 20 matches
                try:
                    # Try to get match name
                    match_name = None
                    name_selectors = [
                        "div[class*='team']",
                        "div[class*='participant']",
                        "span[class*='name']",
                    ]

                    for sel in name_selectors:
                        try:
                            name_elem = elem.find_element(By.CSS_SELECTOR, sel)
                            match_name = name_elem.text
                            if match_name:
                                break
                        except Exception:
                            continue

                    # Try to get odds
                    odds_elements = elem.find_elements(By.CSS_SELECTOR, "[class*='odd']")

                    if len(odds_elements) >= 2:
                        try:
                            player1_odds = float(odds_elements[0].text)
                            player2_odds = float(odds_elements[1].text)

                            match = {
                                "match_name": match_name or f"Match {idx+1}",
                                "player1_odds": player1_odds,
                                "player2_odds": player2_odds,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                            matches.append(match)
                            logger.info(
                                f"  {match['match_name']}: {player1_odds} vs {player2_odds}"
                            )
                        except ValueError:
                            # Couldn't parse as float
                            pass

                except Exception as e:
                    logger.debug(f"Error extracting match {idx}: {e}")
                    continue

            logger.info(f"Successfully extracted {len(matches)} matches")
            return matches

        except Exception as e:
            logger.error(f"Error fetching odds: {e}")
            return []

    def poll_once(self):
        """Single poll cycle"""
        logger.info("=== Interwetten poll cycle start ===")
        timestamp = datetime.utcnow().isoformat()

        matches = self.fetch_tennis_odds()

        snapshot = {
            "timestamp": timestamp,
            "source": "Interwetten",
            "sport": "Tennis",
            "matches": matches,
        }

        # Save snapshot
        filename = DATA_DIR / f"interwetten_snapshot_{timestamp.replace(':', '-').replace('.', '_')}.json"
        try:
            with open(filename, "w") as f:
                json.dump(snapshot, f, indent=2)
            logger.info(f"Saved snapshot: {filename} ({len(matches)} matches)")
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")

        logger.info("=== Poll cycle done ===")
        return matches

    def run_forever(self):
        """Main loop"""
        try:
            while True:
                try:
                    self.poll_once()
                except Exception as e:
                    logger.error(f"Poll failed: {e}")

                logger.info(f"Sleeping {POLL_INTERVAL}s until next poll...")
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            if self.driver:
                self.driver.quit()
                logger.info("WebDriver closed")

    def test_once(self):
        """Test mode: single poll and exit (for debugging)"""
        logger.info("=== TEST MODE ===")
        matches = self.poll_once()
        logger.info(f"\nTest result: {len(matches)} matches extracted")
        if matches:
            logger.info("Sample matches:")
            for m in matches[:3]:
                logger.info(f"  - {m['match_name']}: {m['player1_odds']} vs {m['player2_odds']}")
        if self.driver:
            self.driver.quit()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (single poll, then exit)",
    )
    args = parser.parse_args()

    logger.info("Interwetten Selenium Scraper starting...")
    logger.info(f"Target: {INTERWETTEN_TENNIS_URL}")

    scraper = InterwettenSeleniumScraper()

    if args.test:
        scraper.test_once()
    else:
        scraper.run_forever()


if __name__ == "__main__":
    main()
