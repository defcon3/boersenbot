#!/usr/bin/env python3
"""
Bookmaker Logger — Betfair Exchange + Pinnacle Sports
Polls odds for sports markets, logs snapshots for arbitrage analysis
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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "bookmaker.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Config
DATA_DIR = Path("bookmaker_raw")
DATA_DIR.mkdir(exist_ok=True)

POLL_INTERVAL = 300  # 5 minutes

# APIs
BETFAIR_API = "https://api.betfair.com/exchange/betting/json-rpc/v1"
PINNACLE_API = "https://api.pinnacle.com/v3"

# TODO: Store credentials in environment vars or config file, NOT hardcoded
BETFAIR_CONFIG = {
    "app_key": "",  # Set via env var
    "session_token": "",  # Login required
    "cert_path": "",  # Path to .crt/.key files
}

PINNACLE_CONFIG = {
    "api_key": "",  # Set via env var
    "username": "",  # Optional
    "password": "",  # Optional
}


class BetfairLogger:
    """Betfair Exchange API client (read-only, looking for odds)"""

    def __init__(self, app_key: str, session_token: str):
        self.app_key = app_key
        self.session_token = session_token
        self.session = None

    async def init_session(self):
        self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()

    def _headers(self):
        """RPC headers for Betfair API"""
        return {
            "X-Application": self.app_key,
            "X-Authentication": self.session_token,
            "Content-Type": "application/json",
        }

    async def list_events(self):
        """
        List active sports events (Tennis, Football, etc.)
        """
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "SportsAPING/v1.0/listEventTypes",
                "params": {},
                "id": 1,
            }
            async with self.session.post(
                BETFAIR_API, json=payload, headers=self._headers(), timeout=10
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("result", [])
                else:
                    logger.error(f"Betfair error: {resp.status}")
                    return []
        except Exception as e:
            logger.error(f"Error listing events: {e}")
            return []

    async def list_market_catalogue(self, event_type_id: str, limit: int = 50):
        """
        List markets for an event type (e.g., Tennis match odds)
        """
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "SportsAPING/v1.0/listMarketCatalogue",
                "params": {
                    "eventTypeIds": [event_type_id],
                    "marketProjection": ["RUNNER_METADATA"],
                    "sort": "FIRST_TO_START",
                    "maxResults": limit,
                },
                "id": 1,
            }
            async with self.session.post(
                BETFAIR_API, json=payload, headers=self._headers(), timeout=10
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("result", [])
                else:
                    return []
        except Exception as e:
            logger.error(f"Error listing markets: {e}")
            return []

    async def get_market_book(self, market_ids: list):
        """
        Get live odds (prices & matched bets) for markets
        """
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "SportsAPING/v1.0/listMarketBook",
                "params": {
                    "marketIds": market_ids,
                    "priceProjection": {
                        "priceData": ["EX_BEST_OFFERS", "EX_TRADED"],
                    },
                },
                "id": 1,
            }
            async with self.session.post(
                BETFAIR_API, json=payload, headers=self._headers(), timeout=10
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("result", [])
                else:
                    return []
        except Exception as e:
            logger.error(f"Error getting market book: {e}")
            return []


class PinnacleLogger:
    """Pinnacle Sports API client (tolerates profitable bettors)"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None

    async def init_session(self):
        self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()

    def _headers(self):
        """Headers for Pinnacle API"""
        return {
            "Authorization": f"Basic {self.api_key}",
            "Content-Type": "application/json",
        }

    async def list_sports(self):
        """
        List available sports
        """
        try:
            url = f"{PINNACLE_API}/sports"
            async with self.session.get(url, headers=self._headers(), timeout=10) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.error(f"Pinnacle error: {resp.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching sports: {e}")
            return {}

    async def list_fixtures(self, sport_id: int):
        """
        List upcoming fixtures for a sport
        """
        try:
            url = f"{PINNACLE_API}/fixtures"
            params = {"sportId": sport_id, "leagueIds": ""}
            async with self.session.get(
                url, params=params, headers=self._headers(), timeout=10
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return {}
        except Exception as e:
            logger.error(f"Error fetching fixtures: {e}")
            return {}

    async def list_odds(self, sport_id: int):
        """
        List current odds for all fixtures in a sport
        """
        try:
            url = f"{PINNACLE_API}/odds"
            params = {"sportId": sport_id}
            async with self.session.get(
                url, params=params, headers=self._headers(), timeout=10
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return {}
        except Exception as e:
            logger.error(f"Error fetching odds: {e}")
            return {}


class BookmakerLogger:
    """Combined logger for Betfair + Pinnacle"""

    def __init__(self, betfair_token: Optional[str] = None, pinnacle_key: Optional[str] = None):
        self.betfair = BetfairLogger(BETFAIR_CONFIG["app_key"], betfair_token or "") if betfair_token else None
        self.pinnacle = PinnacleLogger(pinnacle_key or "") if pinnacle_key else None

    async def init_sessions(self):
        if self.betfair:
            await self.betfair.init_session()
        if self.pinnacle:
            await self.pinnacle.init_session()

    async def close_sessions(self):
        if self.betfair:
            await self.betfair.close_session()
        if self.pinnacle:
            await self.pinnacle.close_session()

    async def poll_once(self):
        """Single poll cycle"""
        timestamp = datetime.utcnow().isoformat()
        logger.info(f"=== Bookmaker poll cycle start ({timestamp}) ===")

        snapshot = {
            "timestamp": timestamp,
            "betfair": [],
            "pinnacle": [],
        }

        # Betfair: Tennis odds
        if self.betfair:
            try:
                # Get Tennis event type
                events = await self.betfair.list_events()
                tennis_event = next((e for e in events if "Tennis" in str(e)), None)
                if tennis_event:
                    event_id = tennis_event.get("id")
                    markets = await self.betfair.list_market_catalogue(event_id, limit=20)
                    market_ids = [m.get("marketId") for m in markets[:10]]
                    if market_ids:
                        books = await self.betfair.get_market_book(market_ids)
                        for book in books:
                            market_snapshot = {
                                "market_id": book.get("marketId"),
                                "event_name": book.get("description", {}).get("eventName"),
                                "runners": [],
                            }
                            for runner in book.get("runners", [])[:2]:  # Back/Lay for binary
                                ex = runner.get("ex", {})
                                market_snapshot["runners"].append(
                                    {
                                        "runner_id": runner.get("selectionId"),
                                        "back_odds": ex.get("availableToBack", [{}])[0].get("price"),
                                        "lay_odds": ex.get("availableToLay", [{}])[0].get("price"),
                                    }
                                )
                            snapshot["betfair"].append(market_snapshot)
                    logger.info(f"Betfair: captured {len(snapshot['betfair'])} markets")
            except Exception as e:
                logger.error(f"Betfair poll error: {e}")

        # Pinnacle: Tennis odds
        if self.pinnacle:
            try:
                sports = await self.pinnacle.list_sports()
                tennis_sport = next(
                    (s for s in sports.get("sports", []) if "Tennis" in s.get("name", "")), None
                )
                if tennis_sport:
                    sport_id = tennis_sport.get("id")
                    odds = await self.pinnacle.list_odds(sport_id)
                    for fixture in odds.get("fixtures", [])[:10]:
                        fixture_snapshot = {
                            "fixture_id": fixture.get("id"),
                            "match_name": f"{fixture.get('home', 'N/A')} vs {fixture.get('away', 'N/A')}",
                            "periods": [],
                        }
                        for period in fixture.get("periods", []):
                            fixture_snapshot["periods"].append(
                                {
                                    "period_id": period.get("id"),
                                    "spread": period.get("spread"),
                                    "moneyline": period.get("moneyline"),
                                }
                            )
                        snapshot["pinnacle"].append(fixture_snapshot)
                    logger.info(f"Pinnacle: captured {len(snapshot['pinnacle'])} fixtures")
            except Exception as e:
                logger.error(f"Pinnacle poll error: {e}")

        # Save snapshot
        filename = DATA_DIR / f"snapshot_{timestamp.replace(':', '-').replace('.', '_')}.json"
        try:
            with open(filename, "w") as f:
                json.dump(snapshot, f, indent=2)
            logger.info(f"Saved snapshot: {filename}")
        except Exception as e:
            logger.error(f"Error saving: {e}")

        logger.info(f"=== Poll done ===")

    async def run_forever(self):
        """Main loop"""
        await self.init_sessions()
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
            await self.close_sessions()


async def main():
    """
    TODO: Before running:
    1. Set BETFAIR_CONFIG["app_key"] and get session_token via Betfair API login
    2. Set PINNACLE_CONFIG["api_key"] via env var
    3. Ensure certs for Betfair are in place (if needed)
    """
    logger.info("Bookmaker Logger starting...")
    logger.info(f"Poll interval: {POLL_INTERVAL}s")
    logger.warning("⚠️ API credentials not configured — logger will run but collect no data")
    logger.warning("Set BETFAIR_SESSION_TOKEN and PINNACLE_API_KEY environment variables")

    logger_obj = BookmakerLogger()
    await logger_obj.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
