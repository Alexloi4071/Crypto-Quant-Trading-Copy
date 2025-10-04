"""
Helper Utilities Module
Common utility functions used across the trading system
Includes data processing, file operations, and mathematical utilities
"""

import os
import json
import yaml
import pickle
import gzip
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Callable
import functools
import time
from decimal import Decimal, ROUND_HALF_UP
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ==================== File Operations ====================

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def safe_save_json(data: Any, filepath: Union[str, Path], backup: bool = True) -> bool:
    """Safely save JSON with backup"""
    try:
        filepath = Path(filepath)
        ensure_dir(filepath.parent)

        # Create backup if file exists
        if backup and filepath.exists():
            backup_path = filepath.with_suffix(f'.backup.{int(time.time())}.json')
            filepath.rename(backup_path)
            logger.debug(f"Created backup: {backup_path}")

        # Save new file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.debug(f"Saved JSON to: {filepath}")
        return True

    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        return False

def safe_load_json(filepath: Union[str, Path], default: Any = None) -> Any:
    """Safely load JSON with default fallback"""
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            return default

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {e}")
        return default

def safe_save_yaml(data: Any, filepath: Union[str, Path]) -> bool:
    """Safely save YAML file"""
    try:
        filepath = Path(filepath)
        ensure_dir(filepath.parent)

        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        logger.debug(f"Saved YAML to: {filepath}")
        return True

    except Exception as e:
        logger.error(f"Failed to save YAML to {filepath}: {e}")
        return False

def safe_load_yaml(filepath: Union[str, Path], default: Any = None) -> Any:
    """Safely load YAML with default fallback"""
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            return default

        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    except Exception as e:
        logger.error(f"Failed to load YAML from {filepath}: {e}")
        return default

def compress_pickle(data: Any, filepath: Union[str, Path]) -> bool:
    """Save compressed pickle file"""
    try:
        filepath = Path(filepath)
        ensure_dir(filepath.parent)

        with gzip.open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.debug(f"Saved compressed pickle to: {filepath}")
        return True

    except Exception as e:
        logger.error(f"Failed to save compressed pickle to {filepath}: {e}")
        return False

def load_compressed_pickle(filepath: Union[str, Path], default: Any = None) -> Any:
    """Load compressed pickle file"""
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            return default

        with gzip.open(filepath, 'rb') as f:
            return pickle.load(f)

    except Exception as e:
        logger.error(f"Failed to load compressed pickle from {filepath}: {e}")
        return default

def calculate_file_hash(filepath: Union[str, Path], algorithm: str = 'md5') -> str:
    """Calculate file hash"""
    try:
        hash_func = getattr(hashlib, algorithm)()

        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    except Exception as e:
        logger.error(f"Failed to calculate hash for {filepath}: {e}")
        return ""

# ==================== Data Processing ====================

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to different timeframe"""
    try:
        # Timeframe mapping (updated for pandas compatibility)
        tf_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '12h': '12h',
            '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
        }

        rule = tf_map.get(timeframe, '1H')

        base = df.copy()
        # Ensure DateTimeIndex in UTC and clean index
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

        resampled = base.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        resampled = resampled.replace([np.inf, -np.inf], np.nan)

        # Reindex to full grid
        try:
            full_index = pd.date_range(resampled.index.min(), resampled.index.max(), freq=rule)
            reindexed = resampled.reindex(full_index)
        except Exception:
            reindexed = resampled

        # Only fill short gaps by creating flat bars, keep volume=0
        max_gap = 3
        price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in reindexed.columns]
        if price_cols:
            missing_mask = reindexed[price_cols].isna().any(axis=1)
            reindexed['close'] = reindexed['close'].ffill(limit=max_gap)
            prev_close = reindexed['close'].shift(1)
            to_fill = missing_mask.copy()
            reindexed.loc[to_fill, 'open'] = prev_close.loc[to_fill]
            reindexed.loc[to_fill, 'high'] = np.maximum(reindexed.loc[to_fill, 'open'], reindexed.loc[to_fill, 'close'])
            reindexed.loc[to_fill, 'low'] = np.minimum(reindexed.loc[to_fill, 'open'], reindexed.loc[to_fill, 'close'])
            if 'volume' in reindexed.columns:
                reindexed.loc[to_fill, 'volume'] = 0
            reindexed = reindexed.dropna(subset=price_cols, how='any')

        # OHLC validity
        if all(col in reindexed.columns for col in ['open', 'high', 'low', 'close']):
            reindexed['high'] = reindexed[['high', 'open', 'close']].max(axis=1)
            reindexed['low'] = reindexed[['low', 'open', 'close']].min(axis=1)

        # Ensure positive prices and non-negative volume
        if price_cols:
            for col in price_cols:
                reindexed = reindexed[reindexed[col] > 0]
        if 'volume' in reindexed.columns:
            reindexed = reindexed[reindexed['volume'] >= 0]

        return reindexed

    except Exception as e:
        logger.error(f"Failed to resample OHLCV data: {e}")
        return pd.DataFrame()

def clean_ohlcv_data(df: pd.DataFrame, remove_outliers: bool = True) -> pd.DataFrame:
    """Clean OHLCV data"""
    try:
        df_clean = df.copy()

        # Remove invalid data
        df_clean = df_clean[
            (df_clean['high'] >= df_clean['low']) &
            (df_clean['high'] >= df_clean['open']) &
            (df_clean['high'] >= df_clean['close']) &
            (df_clean['low'] <= df_clean['open']) &
            (df_clean['low'] <= df_clean['close']) &
            (df_clean['volume'] >= 0)
        ]

        # Remove outliers if requested
        if remove_outliers:
            for col in ['open', 'high', 'low', 'close']:
                Q1 = df_clean[col].quantile(0.01)
                Q3 = df_clean[col].quantile(0.99)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

        logger.info(f"Cleaned data: {len(df)} -> {len(df_clean)} rows")
        return df_clean

    except Exception as e:
        logger.error(f"Failed to clean OHLCV data: {e}")
        return df

def normalize_symbol(symbol: str) -> str:
    """Normalize trading pair symbol"""
    return symbol.upper().replace('/', '').replace('-', '').replace('_', '')

def parse_timeframe(timeframe: str) -> int:
    """Parse timeframe to minutes"""
    timeframe = timeframe.lower()

    if timeframe.endswith('m'):
        return int(timeframe[:-1])
    elif timeframe.endswith('h'):
        return int(timeframe[:-1]) * 60
    elif timeframe.endswith('d'):
        return int(timeframe[:-1]) * 1440
    elif timeframe.endswith('w'):
        return int(timeframe[:-1]) * 10080
    else:
        return 60  # Default to 1 hour

def format_number(number: float, decimals: int = 2, use_comma: bool = True) -> str:
    """Format number with specified decimals"""
    try:
        # Round to specified decimals
        rounded = Decimal(str(number)).quantize(
            Decimal('0.' + '0' * decimals),
            rounding=ROUND_HALF_UP
        )

        # Format with commas if requested
        if use_comma:
            return f"{rounded:,}"
        else:
            return str(rounded)

    except Exception as e:
        logger.error(f"Failed to format number {number}: {e}")
        return str(number)

def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate returns for given periods"""
    return prices.pct_change(periods)

def calculate_log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate log returns"""
    return np.log(prices / prices.shift(periods))

def calculate_volatility(returns: pd.Series, window: int = 20,
                        annualize: bool = True) -> pd.Series:
    """Calculate rolling volatility"""
    vol = returns.rolling(window).std()

    if annualize:
        # Assume 365 days, 24 hours, 60 minutes per year
        periods_per_year = 365 * 24 * 60 / parse_timeframe('1h')  # Adjust based on timeframe
        vol = vol * np.sqrt(periods_per_year)

    return vol

# ==================== Mathematical Utilities ====================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default fallback"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio"""
    try:
        excess_returns = returns - risk_free_rate
        return excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0.0
    except:
        return 0.0

def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, datetime, datetime]:
    """Calculate maximum drawdown"""
    try:
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak

        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()

        # Find peak before max drawdown
        peak_date = equity_curve[:max_dd_date].idxmax()

        return max_dd, peak_date, max_dd_date

    except Exception as e:
        logger.error(f"Failed to calculate max drawdown: {e}")
        return 0.0, None, None

def calculate_win_rate(returns: pd.Series) -> float:
    """Calculate win rate"""
    try:
        winning_trades = (returns > 0).sum()
        total_trades = len(returns[returns != 0])
        return winning_trades / total_trades if total_trades > 0 else 0.0
    except:
        return 0.0

def calculate_profit_factor(returns: pd.Series) -> float:
    """Calculate profit factor"""
    try:
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
    except:
        return 0.0

# ==================== Time Utilities ====================

def utc_now() -> datetime:
    """Get current UTC time"""
    return datetime.now(timezone.utc)

def timestamp_to_datetime(timestamp: Union[int, float], unit: str = 's') -> datetime:
    """Convert timestamp to datetime"""
    try:
        if unit == 'ms':
            timestamp = timestamp / 1000
        return datetime.fromtimestamp(timestamp, timezone.utc)
    except:
        return utc_now()

def datetime_to_timestamp(dt: datetime, unit: str = 's') -> Union[int, float]:
    """Convert datetime to timestamp"""
    try:
        timestamp = dt.timestamp()
        if unit == 'ms':
            timestamp = timestamp * 1000
        return int(timestamp) if unit != 'ms' or timestamp == int(timestamp) else timestamp
    except:
        return 0

def format_timedelta(td: timedelta) -> str:
    """Format timedelta to human readable string"""
    try:
        total_seconds = int(td.total_seconds())

        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 or not parts:
            parts.append(f"{seconds}s")

        return " ".join(parts)

    except:
        return str(td)

def market_is_open(symbol: str = 'BTCUSDT') -> bool:
    """Check if market is open (crypto markets are always open)"""
    if 'USDT' in symbol or 'USD' in symbol:
        return True  # Crypto markets never close

    # For traditional markets, implement market hours check
    now = utc_now()
    weekday = now.weekday()

    # Simple check for traditional markets (Monday-Friday)
    if weekday < 5:  # Monday = 0, Friday = 4
        return True

    return False

# ==================== Performance Utilities ====================

def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @functools.wraps(func)

    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f} seconds: {e}")
            raise
    return wrapper

def memory_usage_decorator(func: Callable) -> Callable:
    """Decorator to measure memory usage"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import psutil

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        try:
            result = func(*args, **kwargs)
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = memory_after - memory_before

            logger.debug(f"{func.__name__} memory usage: {memory_diff:+.2f} MB")
            return result

        except Exception as e:
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = memory_after - memory_before

            logger.error(f"{func.__name__} failed with memory usage: {memory_diff:+.2f} MB, error: {e}")
            raise

    return wrapper

def retry_decorator(max_retries: int = 3, delay: float = 1.0,
                   backoff: float = 2.0, exceptions: Tuple = (Exception,)) -> Callable:
    """Decorator to retry function on failure"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)

        def wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} retries: {e}")
                        raise

                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}, retrying in {current_delay}s")
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper
    return decorator

# ==================== Network Utilities ====================

@retry_decorator(max_retries=3, delay=1.0)

def fetch_url(url: str, timeout: int = 30, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Fetch URL with retry and error handling"""
    try:
        default_headers = {'User-Agent': 'CryptoQuantBot/1.0'}
        if headers:
            default_headers.update(headers)

        response = requests.get(url, timeout=timeout, headers=default_headers)
        response.raise_for_status()

        return {
            'success': True,
            'data': response.json(),
            'status_code': response.status_code
        }

    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return {
            'success': False,
            'error': str(e),
            'data': None
        }

async def fetch_url_async(session: aiohttp.ClientSession, url: str,
                         timeout: int = 30, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Async version of fetch_url"""
    try:
        default_headers = {'User-Agent': 'CryptoQuantBot/1.0'}
        if headers:
            default_headers.update(headers)

        async with session.get(url, timeout=timeout, headers=default_headers) as response:
            response.raise_for_status()
            data = await response.json()

            return {
                'success': True,
                'data': data,
                'status_code': response.status
            }

    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return {
            'success': False,
            'error': str(e),
            'data': None
        }

def parallel_fetch_urls(urls: List[str], max_workers: int = 5) -> List[Dict[str, Any]]:
    """Fetch multiple URLs in parallel"""
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_url, url): url for url in urls}

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                result['url'] = url
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {e}")
                results.append({
                    'url': url,
                    'success': False,
                    'error': str(e),
                    'data': None
                })

    return results

# ==================== Validation Utilities ====================

def validate_ohlcv_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate OHLCV data format and content"""
    errors = []

    # Check required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")

    if errors:
        return False, errors

    # Check data consistency
    invalid_rows = df[
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close']) |
        (df['volume'] < 0)
    ]

    if len(invalid_rows) > 0:
        errors.append(f"Found {len(invalid_rows)} invalid OHLCV rows")

    # Check for missing data
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        errors.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

    return len(errors) == 0, errors

def validate_symbol_format(symbol: str) -> bool:
    """Validate trading symbol format"""
    try:
        normalized = normalize_symbol(symbol)
        return len(normalized) >= 6 and normalized.isalnum()
    except:
        return False

def validate_timeframe(timeframe: str) -> bool:
    """Validate timeframe format"""
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '3d', '1w', '1M']
    return timeframe in valid_timeframes

# ==================== Configuration Utilities ====================

def load_config_with_fallback(config_path: str, fallback_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load configuration with fallback to default"""
    try:
        if Path(config_path).exists():
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_data = safe_load_yaml(config_path, {})
            else:
                config_data = safe_load_json(config_path, {})

            # Merge with fallback
            merged_config = fallback_config.copy()
            merged_config.update(config_data)
            return merged_config
        else:
            logger.warning(f"Config file not found: {config_path}, using fallback")
            return fallback_config

    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return fallback_config

def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result

# Usage examples
if __name__ == "__main__":
    # Test file operations
    test_data = {"test": "data", "timestamp": utc_now()}
    safe_save_json(test_data, "test_output.json")
    loaded_data = safe_load_json("test_output.json")
    print(f"Loaded data: {loaded_data}")

    # Test data processing
    sample_df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [99, 100, 101],
        'close': [104, 105, 106],
        'volume': [1000, 1100, 1200]
    }, index=pd.date_range('2024-01-01', periods=3, freq='1H'))

    print(f"Sample data valid: {validate_ohlcv_data(sample_df)}")

    # Test math utilities
    returns = calculate_returns(sample_df['close'])
    sharpe = calculate_sharpe_ratio(returns)
    print(f"Sharpe ratio: {sharpe:.4f}")

    print("Helper utilities test completed successfully!")
