"""
Data Collection Module
Handles data acquisition from various sources including Binance, external APIs
Supports both historical data download and real-time data streaming
"""

import asyncio
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Union
import ccxt
import requests

from config.settings import config
from src.utils.logger import setup_logger
from src.utils.database_manager import DatabaseManager

logger = setup_logger(__name__)

class BinanceDataCollector:
    """Binance data collection handler"""

    def __init__(self):
        self.exchange = self._setup_exchange()
        self.db_manager = DatabaseManager()

    def _setup_exchange(self) -> ccxt.Exchange:
        """Initialize Binance exchange connection with optimized rate limiting"""
        try:
            exchange = ccxt.binance({
                'apiKey': config.trading.api_key,
                'secret': config.trading.api_secret,
                'sandbox': config.trading.testnet,
                'rateLimit': 200,  # 增加到200ms，減少到每秒5次請求
                'enableRateLimit': True,
                'options': {
                    'recvWindow': 10000,  # Increased recv window for better reliability
                    'defaultType': 'future',  # 🔧 修改為期貨合約數據
                }
            })

            # Test connection
            exchange.load_markets()
            market_type = exchange.options.get('defaultType', 'spot')
            logger.info(f"Connected to Binance {'Testnet' if config.trading.testnet else 'Mainnet'} ({market_type.upper()} market)")
            return exchange

        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            raise

    async def download_historical_data(self, symbol: str, timeframe: str,
                                     start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Download historical OHLCV data from Binance with progress display

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Downloading {symbol} {timeframe} data from {start_date} to {end_date}")

            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date or datetime.now().strftime('%Y-%m-%d'),
                                        '%Y-%m-%d').timestamp() * 1000)

            # Calculate total time range for progress tracking
            total_duration = end_ts - start_ts

            all_data = []
            current_ts = start_ts
            limit = 1000  # 降低到1000條記錄以減少負載
            batch_count = 0
            consecutive_errors = 0
            max_concurrent = 2  # 降低到2個並發請求以防止速率限制

            # Progress tracking
            print(f"📊 開始下載 {symbol} {timeframe} 數據...")
            print(f"   時間範圍: {start_date} 到 {end_date or datetime.now().strftime('%Y-%m-%d')}")
            print(f"   🚀 優化模式：每批次{limit}條，最多{max_concurrent}個並發請求")

            # Use semaphore to control concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)

            async def fetch_batch_with_semaphore(symbol, timeframe, timestamp, limit):
                async with semaphore:
                    return await self._fetch_ohlcv_batch_optimized(symbol, timeframe, timestamp, limit)

            while current_ts < end_ts:
                try:
                    # Create tasks for concurrent requests
                    tasks = []
                    batch_timestamps = []

                    # Prepare multiple requests for concurrent execution
                    temp_ts = current_ts
                    for _ in range(min(max_concurrent, 1)):  # 更保守的並發，一次只處理1個批次
                        if temp_ts >= end_ts:
                            break
                        tasks.append(fetch_batch_with_semaphore(symbol, timeframe, temp_ts, limit))
                        batch_timestamps.append(temp_ts)
                        temp_ts += limit * 60 * 1000  # Estimate next timestamp (1m = 60*1000ms)

                    # Execute concurrent requests
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        # Process results
                        valid_results = []
                        for i, result in enumerate(results):
                            if isinstance(result, Exception):
                                logger.warning(f"Batch {i} failed: {result}")
                                consecutive_errors += 1
                            elif result:
                                valid_results.extend(result)
                                consecutive_errors = 0  # Reset error counter

                        if valid_results:
                            all_data.extend(valid_results)
                            # Update current timestamp to the latest data point
                            current_ts = max(item[0] for item in valid_results) + 1
                            batch_count += len(tasks)

                            # Progress display
                            progress = min(100.0, ((current_ts - start_ts) / total_duration) * 100)
                            if batch_count % 5 == 0 or progress >= 100:  # Show progress every 5 batches
                                print(f"   📈 進度: {progress:.1f}% | 已下載: {len(all_data):,} 條記錄 | 批次: {batch_count} | 並發: {len(tasks)}")
                        else:
                            # No data received, advance timestamp manually
                            current_ts += limit * 60 * 1000

                    # Adaptive rate limiting based on errors
                    if consecutive_errors > 3:
                        await asyncio.sleep(1)  # Slow down if too many errors
                        max_concurrent = max(1, max_concurrent - 1)  # Reduce concurrency
                        print(f"   ⚠️ 檢測到連續錯誤，降低並發數到 {max_concurrent}")
                    elif consecutive_errors == 0 and batch_count % 20 == 0:
                        max_concurrent = min(5, max_concurrent + 1)  # Gradually increase
                    else:
                        await asyncio.sleep(0.02)  # Very short sleep for high throughput

                    logger.debug(f"Downloaded batch, current timestamp: {current_ts}, errors: {consecutive_errors}")

                except Exception as e:
                    logger.error(f"Error in batch download: {e}")
                    consecutive_errors += 1
                    await asyncio.sleep(min(5, consecutive_errors))  # Progressive backoff
                    continue

            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Convert to numeric types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)

            logger.info(f"Downloaded {len(df)} candles for {symbol} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"Failed to download historical data: {e}")
            raise

    async def _fetch_ohlcv_batch(self, symbol: str, timeframe: str,
                                since: int, limit: int) -> List[List]:
        """Fetch a batch of OHLCV data"""
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        except Exception as e:
            logger.error(f"Error fetching OHLCV batch: {e}")
            return []

    async def _fetch_ohlcv_batch_optimized(self, symbol: str, timeframe: str,
                                         since: int, limit: int) -> List[List]:
        """Optimized batch fetch with retry logic and better error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use asyncio to make the synchronous call non-blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                )
                return result
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    # Rate limit hit, wait longer
                    wait_time = (2 ** attempt) * 0.5  # Exponential backoff
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt+1}/{max_retries}")
                    await asyncio.sleep(wait_time)
                elif attempt == max_retries - 1:
                    logger.error(f"Error fetching OHLCV batch after {max_retries} attempts: {e}")
                    return []
                else:
                    # Other error, short wait before retry
                    await asyncio.sleep(0.5)
        return []

    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for a symbol"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return 0.0

    def get_futures_data(self, symbol: str) -> Dict[str, float]:
        """Get futures market data (open interest, funding rate)"""
        try:
            # Fetch open interest
            oi_response = self.exchange.fapiPublicGetOpenInterest({'symbol': symbol})
            open_interest = float(oi_response['openInterest'])

            # Fetch funding rate
            funding_response = self.exchange.fapiPublicGetFundingRate({'symbol': symbol})
            funding_rate = float(funding_response[0]['fundingRate'])

            return {
                'open_interest': open_interest,
                'funding_rate': funding_rate,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error getting futures data for {symbol}: {e}")
            return {'open_interest': 0.0, 'funding_rate': 0.0, 'timestamp': datetime.now()}

class ExternalDataCollector:
    """Collector for external data sources (on-chain, sentiment, macro)"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'CryptoQuantBot/1.0'})

    def get_fear_greed_index(self) -> Dict[str, Union[int, str]]:
        """Get Fear & Greed Index from Alternative.me"""
        try:
            url = "https://api.alternative.me/fng/"
            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()
            return {
                'value': int(data['data'][0]['value']),
                'classification': data['data'][0]['value_classification'],
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {e}")
            return {'value': 50, 'classification': 'Neutral', 'timestamp': datetime.now()}

    def get_onchain_metrics(self, symbol: str = 'BTC') -> Dict[str, float]:
        """Get on-chain metrics from free APIs"""
        try:
            # Using CryptoCompare free tier
            url = "https://min-api.cryptocompare.com/data/blockchain/histo/day"
            params = {
                'fsym': symbol,
                'limit': 1,
                'api_key': os.getenv('CRYPTOCOMPARE_API_KEY', '')
            }

            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            if data['Response'] == 'Success':
                latest = data['Data']['Data'][-1]
                return {
                    'active_addresses': latest.get('active_addresses', 0),
                    'transaction_count': latest.get('transaction_count', 0),
                    'large_transaction_count': latest.get('large_transaction_count', 0),
                    'timestamp': datetime.now()
                }

        except Exception as e:
            logger.error(f"Error fetching on-chain metrics: {e}")

        # Return default values if API fails
        return {
            'active_addresses': 0,
            'transaction_count': 0,
            'large_transaction_count': 0,
            'timestamp': datetime.now()
        }

    def get_social_sentiment(self, symbol: str = 'bitcoin') -> Dict[str, float]:
        """Get social sentiment data"""
        try:
            # Twitter API v2 (requires bearer token)
            if os.getenv('TWITTER_BEARER_TOKEN'):
                return self._get_twitter_sentiment(symbol)

            # Fallback to Reddit sentiment
            return self._get_reddit_sentiment(symbol)

        except Exception as e:
            logger.error(f"Error fetching social sentiment: {e}")
            return {
                'tweet_count': 0,
                'sentiment_score': 0.5,
                'timestamp': datetime.now()
            }

    def _get_twitter_sentiment(self, symbol: str) -> Dict[str, float]:
        """Get Twitter sentiment using API v2"""
        try:
            bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
            url = "https://api.twitter.com/2/tweets/search/recent"

            headers = {'Authorization': f'Bearer {bearer_token}'}
            params = {
                'query': f'{symbol} -is:retweet lang:en',
                'max_results': 100,
                'tweet.fields': 'created_at,public_metrics'
            }

            response = self.session.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
            tweet_count = len(data.get('data', []))

            # Simple sentiment analysis (can be improved with NLP libraries)
            sentiment_score = 0.5  # Neutral baseline

            return {
                'tweet_count': tweet_count,
                'sentiment_score': sentiment_score,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error fetching Twitter sentiment: {e}")
            return {'tweet_count': 0, 'sentiment_score': 0.5, 'timestamp': datetime.now()}

    def _get_reddit_sentiment(self, symbol: str) -> Dict[str, float]:
        """Get Reddit sentiment using PRAW"""
        try:
            # This would require Reddit API credentials
            # Placeholder implementation
            return {
                'reddit_posts': 0,
                'sentiment_score': 0.5,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error fetching Reddit sentiment: {e}")
            return {'reddit_posts': 0, 'sentiment_score': 0.5, 'timestamp': datetime.now()}

class DataCollectionOrchestrator:
    """Main orchestrator for all data collection activities"""

    def __init__(self):
        self.binance_collector = BinanceDataCollector()
        self.external_collector = ExternalDataCollector()
        self.db_manager = DatabaseManager()

    async def collect_historical_data(self, symbol: str, start_date: str,
                                    end_date: str = None, save_to_db: bool = True) -> Dict[str, pd.DataFrame]:
        """Collect complete historical dataset for a symbol"""
        logger.info(f"Starting historical data collection for {symbol}")

        datasets = {}

        try:
            # Collect 1-minute data (base timeframe)
            df_1m = await self.binance_collector.download_historical_data(
                symbol, '1m', start_date, end_date
            )
            datasets['1m'] = df_1m

            # Resample to other timeframes
            timeframes = ['5m', '15m', '1h', '4h', '1d']
            for tf in timeframes:
                datasets[tf] = self._resample_data(df_1m, tf)

            # Save to database if requested
            if save_to_db:
                await self._save_datasets_to_db(symbol, datasets)

            # Save to parquet files in data/raw/{symbol}/ directory  
            await self._save_datasets_to_parquet(symbol, datasets)

            logger.info(f"Historical data collection completed for {symbol}")
            return datasets

        except Exception as e:
            logger.error(f"Error in historical data collection: {e}")
            raise

    async def collect_realtime_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Collect real-time data for multiple symbols"""
        realtime_data = {}

        for symbol in symbols:
            try:
                # Price data
                price = self.binance_collector.get_latest_price(symbol)

                # Futures data
                futures_data = self.binance_collector.get_futures_data(symbol)

                # External data (less frequent updates)
                fear_greed = self.external_collector.get_fear_greed_index()
                onchain = self.external_collector.get_onchain_metrics(symbol[:3])  # BTC, ETH
                sentiment = self.external_collector.get_social_sentiment(symbol[:3].lower())

                realtime_data[symbol] = {
                    'price': price,
                    'open_interest': futures_data['open_interest'],
                    'funding_rate': futures_data['funding_rate'],
                    'fear_greed_index': fear_greed['value'],
                    'active_addresses': onchain['active_addresses'],
                    'transaction_count': onchain['transaction_count'],
                    'tweet_count': sentiment['tweet_count'],
                    'sentiment_score': sentiment['sentiment_score'],
                    'timestamp': datetime.now()
                }

            except Exception as e:
                logger.error(f"Error collecting real-time data for {symbol}: {e}")
                continue

        return realtime_data

    def _resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample 1-minute data to other timeframes with proper data cleaning"""
        try:
            # Map timeframes to pandas resample rules (updated)
            tf_map = {'5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h', '1d': '1D'}
            rule = tf_map.get(timeframe, '1H')

            base = df.copy()
            # Ensure DateTimeIndex in UTC and deduplicate index
            try:
                if not isinstance(base.index, pd.DatetimeIndex):
                    base.index = pd.to_datetime(base.index, utc=True, errors='coerce')
                elif base.index.tz is None:
                    base.index = pd.to_datetime(base.index, utc=True, errors='coerce')
                if base.index.hasnans:
                    base = base[~base.index.isna()]
                if base.index.duplicated().any():
                    base = base[~base.index.duplicated(keep='last')]
                if not base.index.is_monotonic_increasing:
                    base = base.sort_index()
            except Exception:
                pass

            # Aggregate to target timeframe
            resampled = base.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })

            # Replace inf with NaN for consistent handling
            resampled = resampled.replace([np.inf, -np.inf], np.nan)

            # Reindex to a full grid
            try:
                full_index = pd.date_range(resampled.index.min(), resampled.index.max(), freq=rule)
                reindexed = resampled.reindex(full_index)
            except Exception:
                reindexed = resampled

            # Fill only short gaps (<= max_gap) by creating flat synthetic bars
            max_gap = 3
            price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in reindexed.columns]
            if price_cols:
                missing_mask = reindexed[price_cols].isna().any(axis=1)

                # Fill close with limited forward fill for short gaps
                reindexed['close'] = reindexed['close'].ffill(limit=max_gap)

                # For rows that were missing, set open to previous close
                prev_close = reindexed['close'].shift(1)
                to_fill = missing_mask.copy()
                reindexed.loc[to_fill, 'open'] = prev_close.loc[to_fill]

                # High/Low flatten to open/close bounds
                reindexed.loc[to_fill, 'high'] = np.maximum(reindexed.loc[to_fill, 'open'], reindexed.loc[to_fill, 'close'])
                reindexed.loc[to_fill, 'low'] = np.minimum(reindexed.loc[to_fill, 'open'], reindexed.loc[to_fill, 'close'])

                # Volume: 0 for inserted bars
                if 'volume' in reindexed.columns:
                    reindexed.loc[to_fill, 'volume'] = 0

                # Drop rows that still have NaN in price columns (long gaps)
                reindexed = reindexed.dropna(subset=price_cols, how='any')

            # Enforce OHLC validity
            if all(col in reindexed.columns for col in ['open', 'high', 'low', 'close']):
                reindexed['high'] = reindexed[['high', 'open', 'close']].max(axis=1)
                reindexed['low'] = reindexed[['low', 'open', 'close']].min(axis=1)

            # Ensure positive prices and non-negative volume
            if price_cols:
                for col in price_cols:
                    reindexed = reindexed[reindexed[col] > 0]
            if 'volume' in reindexed.columns:
                reindexed = reindexed[reindexed['volume'] >= 0]

            logger.info(f"Resampled {timeframe} data: {len(reindexed)} records after cleaning")
            return reindexed

        except Exception as e:
            logger.error(f"Error resampling data to {timeframe}: {e}")
            return pd.DataFrame()

    async def _save_datasets_to_db(self, symbol: str, datasets: Dict[str, pd.DataFrame]):
        """Save datasets to database"""
        try:
            for timeframe, df in datasets.items():
                await self.db_manager.save_ohlcv_data(symbol, timeframe, df)

            logger.info(f"Saved {symbol} datasets to database")

        except Exception as e:
            logger.error(f"Error saving datasets to database: {e}")

    async def _save_datasets_to_parquet(self, symbol: str, datasets: Dict[str, pd.DataFrame]):
        """Save datasets to parquet files in data/raw/{symbol}/ directory"""
        try:
            symbol_dir = config.get_symbol_data_path(symbol)
            symbol_dir.mkdir(parents=True, exist_ok=True)

            for timeframe, df in datasets.items():
                # 保存為parquet格式，符合項目標準
                parquet_path = symbol_dir / f"{symbol}_{timeframe}_ohlcv.parquet"
                df.to_parquet(parquet_path)
                logger.info(f"Saved {symbol} {timeframe}: {len(df)} records → {parquet_path}")

            logger.info(f"Saved {symbol} datasets to parquet files")

        except Exception as e:
            logger.error(f"Error saving datasets to parquet: {e}")

    async def update_realtime_data(self, symbols: List[str], interval_seconds: int = 60):
        """Continuously update real-time data"""
        logger.info(f"Starting real-time data updates for {symbols} every {interval_seconds} seconds")

        while True:
            try:
                # Collect real-time data
                realtime_data = await self.collect_realtime_data(symbols)

                # Save to database
                for symbol, data in realtime_data.items():
                    await self.db_manager.save_realtime_data(symbol, data)

                logger.debug(f"Updated real-time data for {len(realtime_data)} symbols")

                # Wait for next update
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in real-time data update: {e}")
                await asyncio.sleep(60)  # Wait before retry

# Usage example functions

async def download_symbol_data(symbol: str, days: int = 365):
    """Download historical data for a symbol"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    collector = DataCollectionOrchestrator()
    return await collector.collect_historical_data(symbol, start_date, end_date)

async def start_realtime_collection(symbols: List[str] = None):
    """Start real-time data collection"""
    if symbols is None:
        symbols = config.trading_pairs

    collector = DataCollectionOrchestrator()
    await collector.update_realtime_data(symbols, interval_seconds=60)
