#!/usr/bin/env python3
"""
Jupiter Prediction Markets Scraper — Selenium-based
Loads jup.ag/prediction and extracts market data
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
        logging.FileHandler(log_dir / "jupag_selenium.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Config
DATA_DIR = Path("jupag_raw")
DATA_DIR.mkdir(exist_ok=True)

POLL_INTERVAL = 300  # 5 minutes

# Jupiter URLs
JUP_PREDICTION_URL = "https://jup.ag/prediction"


class JupiterSeleniumScraper:
    """Scrapes Jupiter prediction markets using Selenium + Chrome"""

    def __init__(self):
        """Initialize Chrome WebDriver"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
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

    def fetch_prediction_markets(self):
        """
        Load Jupiter prediction page and extract markets
        Returns: list of market dicts with quotes
        """
        if not self.driver:
            logger.error("WebDriver not initialized")
            return []

        try:
            logger.info(f"Loading {JUP_PREDICTION_URL}...")
            self.driver.get(JUP_PREDICTION_URL)

            # Wait for page to load
            wait = WebDriverWait(self.driver, 20)

            try:
                # Scroll down to trigger lazy loading
                for _ in range(3):
                    self.driver.execute_script("window.scrollBy(0, 500)")
                    time.sleep(2)

                # Try to find market containers (adjust selectors as needed)
                wait.until(EC.presence_of_all_elements_located((By.XPATH, "//*[contains(@class, 'market')]")))
            except Exception:
                logger.warning("Timeout waiting for markets to load")
                # Try even longer wait
                logger.info("Trying extended wait...")
                time.sleep(10)

            markets = []

            # Try different selector patterns for Jupiter's market cards
            selectors_to_try = [
                ("//div[contains(@class, 'market-card')]", "market-card"),
                ("//div[contains(@class, 'market')]", "market"),
                ("//div[@role='button'][contains(@class, 'card')]", "card with role=button"),
                ("//article[contains(@class, 'market')]", "article.market"),
            ]

            market_elements = None
            for xpath, desc in selectors_to_try:
                try:
                    market_elements = self.driver.find_elements(By.XPATH, xpath)
                    if market_elements:
                        logger.info(f"Found {len(market_elements)} markets using selector: {desc}")
                        break
                except Exception:
                    continue

            if not market_elements:
                logger.warning("No market elements found with any selector")
                # Dump page for debugging
                page_source = self.driver.page_source
                debug_file = Path("jupag_debug.html")
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(page_source)
                logger.warning(f"Dumped page source to {debug_file}")
                return []

            # Extract data from each market
            import re

            for idx, elem in enumerate(market_elements[:100]):
                try:
                    # Get all text from the element
                    elem_text = elem.text

                    # Extract title (usually the first line or h2/h3)
                    title = None
                    lines = elem_text.split("\n")
                    for line in lines[:3]:
                        if line.strip() and len(line.strip()) > 3 and not "¢" in line:
                            title = line.strip()
                            break

                    # Extract prices (look for "¢" symbol, format: "OPTION 66¢" or "66¢")
                    prices = {}
                    for line in lines:
                        if "¢" in line:
                            # Extract option name and price
                            # Format: "Germany 66¢" or "66¢" or "DRAW 21¢"
                            match = re.search(r"([A-Za-z\s]+)?(\d+)¢", line)
                            if match:
                                option = match.group(1).strip() if match.group(1) else f"option_{len(prices)}"
                                price_cents = int(match.group(2))
                                price_decimal = price_cents / 100.0  # Convert cents to decimal
                                prices[option] = price_decimal
                                logger.debug(f"    Found {option}: {price_decimal}")

                    # Extract yes/no or binary prices
                    yes_price = None
                    no_price = None

                    # Jupiter often has binary (YES/NO) or multiple options (team1, draw, team2)
                    if prices:
                        price_list = list(prices.values())
                        if len(price_list) >= 2:
                            # Use first two prices as yes/no
                            yes_price = price_list[0]
                            no_price = price_list[1]

                    # Only add if we have at least a title and some prices
                    if title and prices:
                        market = {
                            "title": title,
                            "yes_price": yes_price,
                            "no_price": no_price,
                            "all_prices": prices,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                        markets.append(market)
                        logger.info(f"  {title}: Prices={prices}")

                except Exception as e:
                    logger.debug(f"Error extracting market {idx}: {e}")
                    continue

            logger.info(f"Successfully extracted {len(markets)} markets")
            return markets

        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []

    def poll_once(self):
        """Single poll cycle"""
        logger.info("=== Jupiter poll cycle start ===")
        timestamp = datetime.utcnow().isoformat()

        markets = self.fetch_prediction_markets()

        snapshot = {
            "timestamp": timestamp,
            "source": "Jupiter Prediction",
            "markets": markets,
        }

        # Save snapshot
        filename = DATA_DIR / f"snapshot_{timestamp.replace(':', '-').replace('.', '_')}.json"
        try:
            with open(filename, "w") as f:
                json.dump(snapshot, f, indent=2)
            logger.info(f"Saved snapshot: {filename} ({len(markets)} markets)")
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")

        logger.info("=== Poll cycle done ===")
        return markets

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
        """Test mode: single poll and exit"""
        logger.info("=== TEST MODE ===")
        markets = self.poll_once()
        logger.info(f"\nTest result: {len(markets)} markets extracted")
        if markets:
            logger.info("Sample markets:")
            for m in markets[:3]:
                logger.info(f"  - {m['title']}: YES={m['yes_price']}, NO={m['no_price']}")
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

    logger.info("Jupiter Selenium Scraper starting...")
    logger.info(f"Target: {JUP_PREDICTION_URL}")

    scraper = JupiterSeleniumScraper()

    if args.test:
        scraper.test_once()
    else:
        scraper.run_forever()


if __name__ == "__main__":
    main()
