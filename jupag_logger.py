#!/usr/bin/env python3
"""
Jupiter Prediction Markets Logger (jup.ag/prediction)
Fetches prediction market data and quotes for arbitrage analysis
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp


# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "jupag.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Config
DATA_DIR = Path("jupag_raw")
DATA_DIR.mkdir(exist_ok=True)

POLL_INTERVAL = 300  # 5 minutes

# Jupiter API endpoints
JUP_BASE_API = "https://api.jup.ag"
JUP_PREDICTION_API = "https://prediction.jup.ag/api"


class JupiterLogger:
    """Fetches prediction market data from Jupiter"""

    def __init__(self):
        self.session = None

    async def init_session(self):
        """Initialize aiohttp session"""
        self.session = aiohttp.ClientSession()

    async def close_session(self):
        """Clean up session"""
        if self.session:
            await self.session.close()

    async def fetch_markets(self) -> list:
        """
        Fetch all active prediction markets from Jupiter
        Returns: list of market dicts
        """
        try:
            # Try Jupiter Prediction API first
            url = f"{JUP_PREDICTION_API}/markets"
            async with self.session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"Fetched markets via {url}")
                    # Handle different response formats
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and "markets" in data:
                        return data.get("markets", [])
                    else:
                        return []
                else:
                    logger.warning(f"API returned {resp.status}, trying alternative endpoint")

            # Fallback: try alternative endpoint
            url = f"{JUP_BASE_API}/prediction/markets"
            async with self.session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"Fetched markets via fallback endpoint")
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and "data" in data:
                        return data.get("data", [])

            return []

        except asyncio.TimeoutError:
            logger.error("Jupiter API timeout")
            return []
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []

    async def fetch_market_details(self, market_id: str) -> Optional[dict]:
        """
        Fetch detailed info for a specific market (quotes, TVL, etc.)
        """
        try:
            url = f"{JUP_PREDICTION_API}/markets/{market_id}"
            async with self.session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return None
        except Exception as e:
            logger.warning(f"Error fetching market {market_id}: {e}")
            return None

    async def fetch_market_quotes(self, market_id: str) -> Optional[dict]:
        """
        Fetch current quotes (odds) for a market
        Usually endpoints: /markets/{id}/quotes or similar
        """
        try:
            # Try different quote endpoints
            endpoints = [
                f"{JUP_PREDICTION_API}/markets/{market_id}/quotes",
                f"{JUP_PREDICTION_API}/markets/{market_id}/prices",
                f"{JUP_BASE_API}/prediction/{market_id}/quotes",
            ]

            for url in endpoints:
                try:
                    async with self.session.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            return await resp.json()
                except Exception:
                    continue

            return None
        except Exception as e:
            logger.warning(f"Error fetching quotes for {market_id}: {e}")
            return None

    async def poll_once(self):
        """Single poll cycle"""
        logger.info("=== Jupiter poll cycle start ===")
        timestamp = datetime.utcnow().isoformat()

        markets = await self.fetch_markets()
        logger.info(f"Fetched {len(markets)} markets")

        if not markets:
            logger.warning("No markets returned from API")
            return

        snapshot = {
            "timestamp": timestamp,
            "source": "Jupiter Prediction",
            "markets": [],
        }

        # Fetch detailed data for each market
        for market in markets[:50]:  # Limit to 50 markets per poll
            try:
                market_id = market.get("id") or market.get("marketId")
                title = market.get("title") or market.get("question") or "Unknown"

                # Try to get quotes
                quotes = await self.fetch_market_quotes(market_id)

                market_snapshot = {
                    "market_id": market_id,
                    "title": title,
                    "description": market.get("description", ""),
                    "status": market.get("status", ""),
                    "yes_price": None,
                    "no_price": None,
                    "yes_tvl": None,
                    "no_tvl": None,
                    "end_time": market.get("endTime") or market.get("deadline"),
                }

                # Extract quote data
                if quotes:
                    if isinstance(quotes, dict):
                        market_snapshot["yes_price"] = quotes.get("yes") or quotes.get("yesPrice")
                        market_snapshot["no_price"] = quotes.get("no") or quotes.get("noPrice")
                        market_snapshot["yes_tvl"] = quotes.get("yesTVL") or quotes.get("yes_liquidity")
                        market_snapshot["no_tvl"] = quotes.get("noTVL") or quotes.get("no_liquidity")

                # Add raw market data as fallback
                if "price" in market:
                    market_snapshot["yes_price"] = market["price"].get("yes")
                    market_snapshot["no_price"] = market["price"].get("no")

                snapshot["markets"].append(market_snapshot)

            except Exception as e:
                logger.warning(f"Error processing market: {e}")
                continue

        logger.info(f"Processed {len(snapshot['markets'])} markets")

        # Save snapshot
        filename = DATA_DIR / f"snapshot_{timestamp.replace(':', '-').replace('.', '_')}.json"
        try:
            with open(filename, "w") as f:
                json.dump(snapshot, f, indent=2)
            logger.info(f"Saved snapshot: {filename}")
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")

        logger.info("=== Poll cycle done ===")

    async def run_forever(self):
        """Main loop"""
        await self.init_session()
        try:
            while True:
                try:
                    await self.poll_once()
                except Exception as e:
                    logger.error(f"Poll failed: {e}")

                logger.info(f"Sleeping {POLL_INTERVAL}s until next poll...")
                await asyncio.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await self.close_session()


async def main():
    logger.info("Jupiter Prediction Logger starting...")
    logger.info(f"Poll interval: {POLL_INTERVAL}s")
    logger.info(f"Data directory: {DATA_DIR}")

    jup_logger = JupiterLogger()
    await jup_logger.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
