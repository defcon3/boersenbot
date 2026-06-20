#!/usr/bin/env python3
"""
Polymarket CLOB Logger — Read-only data collection
Polls Gamma API every 5 minutes, logs market snapshots
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import aiohttp


# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "polymarket.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Config
GAMMA_API = "https://gamma-api.polymarket.com"
POLL_INTERVAL = 300  # 5 minutes
DATA_DIR = Path("polymarket_raw")
DATA_DIR.mkdir(exist_ok=True)


class PolymarketLogger:
    def __init__(self):
        self.session = None
        self.markets_cache = {}

    async def init_session(self):
        """Initialize aiohttp session"""
        self.session = aiohttp.ClientSession()

    async def close_session(self):
        """Clean up session"""
        if self.session:
            await self.session.close()

    async def fetch_markets(self):
        """
        Fetch active markets from Gamma API
        Returns: list of market dicts with (id, title, status, etc.)
        """
        try:
            url = f"{GAMMA_API}/markets"
            params = {
                "limit": 100,
                "offset": 0,
                "status": "active",
            }
            async with self.session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # API returns list directly or wrapped in 'data' key
                    if isinstance(data, list):
                        return data
                    else:
                        return data.get("data", [])
                else:
                    logger.error(f"Gamma API error: {resp.status}")
                    return []
        except asyncio.TimeoutError:
            logger.error("Gamma API timeout")
            return []
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []

    async def fetch_orderbook(self, market_id):
        """
        Fetch order book (prices, depth) for a single market
        Returns: {yes_price, no_price, yes_depth, no_depth, ...}
        """
        try:
            url = f"{GAMMA_API}/orderbook/{market_id}"
            async with self.session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return None
        except Exception as e:
            logger.warning(f"Error fetching orderbook for {market_id}: {e}")
            return None

    async def fetch_trades(self, market_id, limit=10):
        """
        Fetch recent trades for sentiment/volume
        """
        try:
            url = f"{GAMMA_API}/trades"
            params = {"market_id": market_id, "limit": limit}
            async with self.session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return None
        except Exception:
            return None

    async def poll_once(self):
        """
        Single poll cycle: fetch markets, orderbooks, aggregate snapshot
        """
        logger.info("=== Poll cycle start ===")
        timestamp = datetime.utcnow().isoformat()

        markets = await self.fetch_markets()
        logger.info(f"Fetched {len(markets)} active markets")

        if not markets:
            logger.warning("No markets returned")
            return

        snapshot = {
            "timestamp": timestamp,
            "markets": [],
        }

        for market in markets[:20]:  # Limit to top 20 for now
            market_id = market.get("id")
            title = market.get("title", "")

            # Fetch orderbook
            orderbook = await self.fetch_orderbook(market_id)
            if not orderbook:
                continue

            # Extract prices (Polymarket has yes/no tokens)
            yes_price = orderbook.get("yes", {}).get("price")
            no_price = orderbook.get("no", {}).get("price")

            market_snapshot = {
                "market_id": market_id,
                "title": title,
                "yes_price": yes_price,
                "no_price": no_price,
                "yes_depth": orderbook.get("yes", {}).get("depth"),
                "no_depth": orderbook.get("no", {}).get("depth"),
                "volume_24h": market.get("volume_24h"),
                "liquidity": market.get("liquidity"),
            }

            snapshot["markets"].append(market_snapshot)

        # Save snapshot
        filename = DATA_DIR / f"snapshot_{timestamp.replace(':', '-').replace('.', '_')}.json"
        try:
            with open(filename, "w") as f:
                json.dump(snapshot, f, indent=2)
            logger.info(f"Saved snapshot: {filename}")
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")

        logger.info(f"=== Poll cycle done ({len(snapshot['markets'])} markets) ===")

    async def run_forever(self):
        """Main loop: poll every POLL_INTERVAL seconds"""
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
    logger.info("Polymarket Logger starting...")
    logger.info(f"Poll interval: {POLL_INTERVAL}s")
    logger.info(f"Data directory: {DATA_DIR}")

    pm_logger = PolymarketLogger()
    await pm_logger.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
