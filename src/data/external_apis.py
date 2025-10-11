"""
External APIs Module
Integrates with external data sources for enhanced market intelligence
Supports CryptoCompare, CoinGecko, News APIs, and social sentiment
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import time
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

from config.settings import config
from src.utils.logger import setup_logger
from src.utils.helpers import retry_decorator, timing_decorator

logger = setup_logger(__name__)

@dataclass

class ExternalDataConfig:
    """Configuration for external data sources"""
    cryptocompare_api_key: str = ""
    coingecko_api_key: str = ""
    news_api_key: str = ""
    twitter_bearer_token: str = ""
    rate_limit_delay: float = 1.0
    max_retries: int = 3
    timeout: int = 30

class ExternalDataManager:
    """Manager for external data sources"""

    def __init__(self):
        # Load configuration from environment
        import os
        self.config = ExternalDataConfig(
            cryptocompare_api_key=os.getenv('CRYPTOCOMPARE_API_KEY', ''),
            coingecko_api_key=os.getenv('COINGECKO_API_KEY', ''),
            news_api_key=os.getenv('NEWS_API_KEY', ''),
            twitter_bearer_token=os.getenv('TWITTER_BEARER_TOKEN', ''),
            rate_limit_delay=float(os.getenv('API_RATE_LIMIT_DELAY', '1.0')),
            max_retries=int(os.getenv('API_MAX_RETRIES', '3')),
            timeout=int(os.getenv('API_TIMEOUT', '30'))
        )

        # API endpoints
        self.api_endpoints = {
            'cryptocompare': {
                'base_url': 'https://min-api.cryptocompare.com/data',
                'ohlcv': '/v2/histohour',
                'price': '/price',
                'social': '/social/coin/general'
            },
            'coingecko': {
                'base_url': 'https://api.coingecko.com/api/v3',
                'price': '/simple/price',
                'ohlcv': '/coins/{id}/market_chart',
                'trending': '/search/trending'
            },
            'newsapi': {
                'base_url': 'https://newsapi.org/v2',
                'everything': '/everything',
                'headlines': '/top-headlines'
            },
            'feargreed': {
                'base_url': 'https://api.alternative.me',
                'fng': '/fng/?limit={limit}&format=json'
            }
        }

        # Rate limiting
        self.last_request_times = {}
        self.request_counts = {}

        # Data cache
        self.cache = {}
        self.cache_ttl = {}

        logger.info("External data manager initialized")

    async def initialize(self):
        """Initialize external data connections"""
        try:
            # Test connections to available APIs
            available_apis = []

            # Test CryptoCompare
            if self.config.cryptocompare_api_key:
                if await self._test_cryptocompare_connection():
                    available_apis.append('cryptocompare')

            # Test CoinGecko
            if await self._test_coingecko_connection():
                available_apis.append('coingecko')

            # Test News API
            if self.config.news_api_key:
                if await self._test_newsapi_connection():
                    available_apis.append('newsapi')

            # Test Fear & Greed Index (no API key required)
            if await self._test_feargreed_connection():
                available_apis.append('feargreed')

            logger.info(f"Initialized external APIs: {available_apis}")
            return len(available_apis) > 0

        except Exception as e:
            logger.error(f"External API initialization failed: {e}")
            return False

    async def _test_cryptocompare_connection(self) -> bool:
        """Test CryptoCompare API connection"""
        try:
            url = f"{self.api_endpoints['cryptocompare']['base_url']}/price"
            params = {
                'fsym': 'BTC',
                'tsyms': 'USD',
                'api_key': self.config.cryptocompare_api_key
            }

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        logger.debug("CryptoCompare API connection successful")
                        return True
            return False

        except Exception as e:
            logger.debug(f"CryptoCompare connection test failed: {e}")
            return False

    async def _test_coingecko_connection(self) -> bool:
        """Test CoinGecko API connection"""
        try:
            url = f"{self.api_endpoints['coingecko']['base_url']}/ping"

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        logger.debug("CoinGecko API connection successful")
                        return True
            return False

        except Exception as e:
            logger.debug(f"CoinGecko connection test failed: {e}")
            return False

    async def _test_newsapi_connection(self) -> bool:
        """Test News API connection"""
        try:
            url = f"{self.api_endpoints['newsapi']['base_url']}/sources"
            headers = {'X-API-Key': self.config.news_api_key}

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        logger.debug("News API connection successful")
                        return True
            return False

        except Exception as e:
            logger.debug(f"News API connection test failed: {e}")
            return False

    async def _test_feargreed_connection(self) -> bool:
        """Test Fear & Greed Index API connection"""
        try:
            url = f"{self.api_endpoints['feargreed']['base_url']}/fng/?limit=1&format=json"

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        logger.debug("Fear & Greed API connection successful")
                        return True
            return False

        except Exception as e:
            logger.debug(f"Fear & Greed API connection test failed: {e}")
            return False

    @retry_decorator(max_retries=3, delay=2.0)

    async def get_market_data(self, symbol: str,
                            source: str = 'coingecko') -> Dict[str, Any]:
        """Get current market data for a symbol"""
        try:
            cache_key = f"market_data_{symbol}_{source}"

            # Check cache (5 minute TTL)
            if self._is_cache_valid(cache_key, ttl_minutes=5):
                return self.cache[cache_key]

            data = {}

            if source == 'coingecko':
                data = await self._get_coingecko_market_data(symbol)
            elif source == 'cryptocompare':
                data = await self._get_cryptocompare_market_data(symbol)

            # Cache the result
            if data:
                self.cache[cache_key] = data
                self.cache_ttl[cache_key] = datetime.now() + timedelta(minutes=5)

            return data

        except Exception as e:
            logger.error(f"Market data retrieval failed for {symbol}: {e}")
            return {}

    async def _get_coingecko_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data from CoinGecko"""
        try:
            # Convert symbol to CoinGecko format
            coin_id = self._symbol_to_coingecko_id(symbol)

            await self._rate_limit_check('coingecko')

            url = f"{self.api_endpoints['coingecko']['base_url']}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }

            if self.config.coingecko_api_key:
                params['x_cg_demo_api_key'] = self.config.coingecko_api_key

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        if coin_id in data:
                            coin_data = data[coin_id]
                            return {
                                'symbol': symbol,
                                'price_usd': coin_data.get('usd', 0),
                                'market_cap_usd': coin_data.get('usd_market_cap', 0),
                                'volume_24h_usd': coin_data.get('usd_24h_vol', 0),
                                'change_24h': coin_data.get('usd_24h_change', 0),
                                'source': 'coingecko',
                                'timestamp': datetime.now()
                            }

            return {}

        except Exception as e:
            logger.error(f"CoinGecko market data error: {e}")
            return {}

    async def _get_cryptocompare_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data from CryptoCompare"""
        try:
            if not self.config.cryptocompare_api_key:
                return {}

            await self._rate_limit_check('cryptocompare')

            url = f"{self.api_endpoints['cryptocompare']['base_url']}/price"
            params = {
                'fsym': symbol.replace('USDT', '').replace('USD', ''),
                'tsyms': 'USD',
                'api_key': self.config.cryptocompare_api_key
            }

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        return {
                            'symbol': symbol,
                            'price_usd': data.get('USD', 0),
                            'source': 'cryptocompare',
                            'timestamp': datetime.now()
                        }

            return {}

        except Exception as e:
            logger.error(f"CryptoCompare market data error: {e}")
            return {}

    @retry_decorator(max_retries=3, delay=2.0)

    async def get_news_sentiment(self, symbol: str,
                               days_back: int = 1) -> Dict[str, Any]:
        """Get news sentiment for a cryptocurrency"""
        try:
            if not self.config.news_api_key:
                return {}

            cache_key = f"news_sentiment_{symbol}_{days_back}"

            # Check cache (30 minute TTL)
            if self._is_cache_valid(cache_key, ttl_minutes=30):
                return self.cache[cache_key]

            await self._rate_limit_check('newsapi')

            # Search for news articles
            url = f"{self.api_endpoints['newsapi']['base_url']}/everything"

            # Create search query
            crypto_name = self._get_crypto_name(symbol)
            search_query = f"{crypto_name} OR {symbol}"

            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

            params = {
                'q': search_query,
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': 50
            }

            headers = {'X-API-Key': self.config.news_api_key}

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Analyze sentiment
                        sentiment_analysis = self._analyze_news_sentiment(data.get('articles', []))

                        result = {
                            'symbol': symbol,
                            'sentiment_score': sentiment_analysis['sentiment_score'],
                            'article_count': len(data.get('articles', [])),
                            'positive_articles': sentiment_analysis['positive_count'],
                            'negative_articles': sentiment_analysis['negative_count'],
                            'neutral_articles': sentiment_analysis['neutral_count'],
                            'source': 'newsapi',
                            'timestamp': datetime.now()
                        }

                        # Cache result
                        self.cache[cache_key] = result
                        self.cache_ttl[cache_key] = datetime.now() + timedelta(minutes=30)

                        return result

            return {}

        except Exception as e:
            logger.error(f"News sentiment retrieval failed for {symbol}: {e}")
            return {}

    def _analyze_news_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment of news articles"""
        try:
            if not articles:
                return {
                    'sentiment_score': 0.0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0
                }

            # Simple keyword-based sentiment analysis
            positive_keywords = [
                'bullish', 'surge', 'rally', 'gains', 'rise', 'increase', 'up',
                'growth', 'adoption', 'positive', 'breakthrough', 'success',
                'milestone', 'partnership', 'upgrade', 'launch'
            ]

            negative_keywords = [
                'bearish', 'crash', 'fall', 'drop', 'decline', 'down',
                'loss', 'selloff', 'dump', 'negative', 'concern', 'warning',
                'ban', 'regulation', 'hack', 'scam', 'fraud'
            ]

            sentiment_scores = []
            positive_count = 0
            negative_count = 0
            neutral_count = 0

            for article in articles:
                title = (article.get('title', '') + ' ' + article.get('description', '')).lower()

                positive_score = sum(1 for keyword in positive_keywords if keyword in title)
                negative_score = sum(1 for keyword in negative_keywords if keyword in title)

                if positive_score > negative_score:
                    sentiment_scores.append(1)
                    positive_count += 1
                elif negative_score > positive_score:
                    sentiment_scores.append(-1)
                    negative_count += 1
                else:
                    sentiment_scores.append(0)
                    neutral_count += 1

            # Calculate overall sentiment score (-1 to 1)
            overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0

            return {
                'sentiment_score': float(overall_sentiment),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count
            }

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                'sentiment_score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }

    @retry_decorator(max_retries=3, delay=1.0)

    async def get_fear_greed_index(self, days: int = 30) -> Dict[str, Any]:
        """Get Fear & Greed Index data"""
        try:
            cache_key = f"fear_greed_{days}"

            # Check cache (1 hour TTL)
            if self._is_cache_valid(cache_key, ttl_minutes=60):
                return self.cache[cache_key]

            await self._rate_limit_check('feargreed')

            url = f"{self.api_endpoints['feargreed']['base_url']}/fng/"
            params = {'limit': days, 'format': 'json'}

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        if 'data' in data:
                            fng_data = data['data']

                            # Calculate statistics
                            values = [int(item['value']) for item in fng_data]

                            result = {
                                'current_value': values[0] if values else 50,
                                'current_classification': fng_data[0]['value_classification'] if fng_data else 'Neutral',
                                'average_value': np.mean(values),
                                'trend': 'increasing' if len(values) > 1 and values[0] > values[-1] else 'decreasing',
                                'historical_data': fng_data[:10],  # Last 10 days
                                'source': 'alternative.me',
                                'timestamp': datetime.now()
                            }

                            # Cache result
                            self.cache[cache_key] = result
                            self.cache_ttl[cache_key] = datetime.now() + timedelta(hours=1)

                            return result

            return {}

        except Exception as e:
            logger.error(f"Fear & Greed Index retrieval failed: {e}")
            return {}

    async def get_trending_coins(self) -> List[Dict[str, Any]]:
        """Get trending cryptocurrencies from CoinGecko"""
        try:
            cache_key = "trending_coins"

            # Check cache (15 minute TTL)
            if self._is_cache_valid(cache_key, ttl_minutes=15):
                return self.cache[cache_key]

            await self._rate_limit_check('coingecko')

            url = f"{self.api_endpoints['coingecko']['base_url']}/search/trending"

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()

                        trending = []
                        for coin in data.get('coins', []):
                            coin_info = coin.get('item', {})
                            trending.append({
                                'id': coin_info.get('id'),
                                'name': coin_info.get('name'),
                                'symbol': coin_info.get('symbol'),
                                'market_cap_rank': coin_info.get('market_cap_rank'),
                                'score': coin_info.get('score')
                            })

                        # Cache result
                        self.cache[cache_key] = trending
                        self.cache_ttl[cache_key] = datetime.now() + timedelta(minutes=15)

                        return trending

            return []

        except Exception as e:
            logger.error(f"Trending coins retrieval failed: {e}")
            return []

    async def get_on_chain_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get on-chain metrics (placeholder for future implementation)"""
        try:
            # This would integrate with services like:
            # - Glassnode
            # - IntoTheBlock
            # - CryptoQuant
            # - Messari

            # For now, return placeholder data
            return {
                'symbol': symbol,
                'network_value': 0,
                'active_addresses': 0,
                'transaction_count': 0,
                'hash_rate': 0,
                'source': 'placeholder',
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"On-chain metrics retrieval failed for {symbol}: {e}")
            return {}

    async def _rate_limit_check(self, api_name: str):
        """Check and enforce rate limits"""
        try:
            current_time = time.time()

            # Initialize if first request
            if api_name not in self.last_request_times:
                self.last_request_times[api_name] = 0
                self.request_counts[api_name] = 0

            # Check if we need to wait
            time_since_last = current_time - self.last_request_times[api_name]

            if time_since_last < self.config.rate_limit_delay:
                wait_time = self.config.rate_limit_delay - time_since_last
                await asyncio.sleep(wait_time)

            # Update counters
            self.last_request_times[api_name] = time.time()
            self.request_counts[api_name] += 1

        except Exception as e:
            logger.debug(f"Rate limit check failed for {api_name}: {e}")

    def _is_cache_valid(self, cache_key: str, ttl_minutes: int = 5) -> bool:
        """Check if cached data is still valid"""
        try:
            if cache_key not in self.cache:
                return False

            if cache_key not in self.cache_ttl:
                return False

            return datetime.now() < self.cache_ttl[cache_key]

        except Exception:
            return False

    def _symbol_to_coingecko_id(self, symbol: str) -> str:
        """Convert trading symbol to CoinGecko ID"""
        # Mapping of common symbols to CoinGecko IDs
        symbol_mapping = {
            'BTCUSDT': 'bitcoin',
            'BTC': 'bitcoin',
            'ETHUSDT': 'ethereum',
            'ETH': 'ethereum',
            'ADAUSDT': 'cardano',
            'ADA': 'cardano',
            'BNBUSDT': 'binancecoin',
            'BNB': 'binancecoin',
            'SOLUSDT': 'solana',
            'SOL': 'solana',
            'XRPUSDT': 'ripple',
            'XRP': 'ripple',
            'DOTUSDT': 'polkadot',
            'DOT': 'polkadot',
            'DOGEUSDT': 'dogecoin',
            'DOGE': 'dogecoin',
            'AVAXUSDT': 'avalanche-2',
            'AVAX': 'avalanche-2',
            'MATICUSDT': 'matic-network',
            'MATIC': 'matic-network'
        }

        return symbol_mapping.get(symbol, symbol.lower().replace('usdt', ''))

    def _get_crypto_name(self, symbol: str) -> str:
        """Get full cryptocurrency name from symbol"""
        name_mapping = {
            'BTCUSDT': 'Bitcoin',
            'BTC': 'Bitcoin',
            'ETHUSDT': 'Ethereum',
            'ETH': 'Ethereum',
            'ADAUSDT': 'Cardano',
            'ADA': 'Cardano',
            'BNBUSDT': 'Binance Coin',
            'BNB': 'Binance Coin',
            'SOLUSDT': 'Solana',
            'SOL': 'Solana'
        }

        return name_mapping.get(symbol, symbol)

    def clear_cache(self):
        """Clear all cached data"""
        try:
            self.cache.clear()
            self.cache_ttl.clear()
            logger.info("External data cache cleared")

        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")

    def get_api_statistics(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        try:
            stats = {
                'request_counts': self.request_counts.copy(),
                'cache_entries': len(self.cache),
                'api_status': {},
                'timestamp': datetime.now()
            }

            # Check API status based on recent requests
            for api_name in self.request_counts:
                last_request_time = self.last_request_times.get(api_name, 0)
                time_since_last = time.time() - last_request_time

                if time_since_last < 3600:  # Active within last hour
                    stats['api_status'][api_name] = 'active'
                else:
                    stats['api_status'][api_name] = 'inactive'

            return stats

        except Exception as e:
            logger.error(f"API statistics generation failed: {e}")
            return {}

# Convenience functions

async def get_crypto_market_data(symbol: str) -> Dict[str, Any]:
    """Get market data for a cryptocurrency"""
    manager = ExternalDataManager()
    await manager.initialize()
    return await manager.get_market_data(symbol)

async def get_crypto_sentiment(symbol: str) -> Dict[str, Any]:
    """Get sentiment data for a cryptocurrency"""
    manager = ExternalDataManager()
    await manager.initialize()
    return await manager.get_news_sentiment(symbol)

async def get_market_fear_greed() -> Dict[str, Any]:
    """Get Fear & Greed Index"""
    manager = ExternalDataManager()
    await manager.initialize()
    return await manager.get_fear_greed_index()

# Usage example
if __name__ == "__main__":

    async def test_external_apis():
        # Create external data manager
        manager = ExternalDataManager()

        # Initialize connections
        initialized = await manager.initialize()
        print(f"APIs initialized: {initialized}")

        # Test market data
        print("Testing market data...")
        market_data = await manager.get_market_data('BTCUSDT')
        print(f"BTC market data: {market_data}")

        # Test Fear & Greed Index
        print("Testing Fear & Greed Index...")
        fng_data = await manager.get_fear_greed_index(7)
        print(f"Fear & Greed: {fng_data.get('current_value', 'N/A')} - {fng_data.get('current_classification', 'N/A')}")

        # Test trending coins
        print("Testing trending coins...")
        trending = await manager.get_trending_coins()
        print(f"Trending coins: {len(trending)} found")

        if trending:
            for coin in trending[:3]:
                print(f"  - {coin['name']} ({coin['symbol']}) - Rank: {coin.get('market_cap_rank', 'N/A')}")

        # Test news sentiment (if API key available)
        print("Testing news sentiment...")
        sentiment = await manager.get_news_sentiment('BTCUSDT')
        if sentiment:
            print(f"BTC sentiment: {sentiment['sentiment_score']:.2f} ({sentiment['article_count']} articles)")
        else:
            print("News sentiment: API key required")

        # Get API statistics
        stats = manager.get_api_statistics()
        print(f"API requests made: {stats['request_counts']}")
        print(f"Cache entries: {stats['cache_entries']}")

        print("External APIs test completed!")

    # Run test
    asyncio.run(test_external_apis())
