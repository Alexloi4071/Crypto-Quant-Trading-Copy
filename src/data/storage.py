"""
Data Storage Module
Handles data storage, retrieval, and management for the trading system
Supports multiple data sources and efficient caching
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import pickle
import asyncio
from pathlib import Path
import sqlite3
from sqlalchemy import create_engine, text
import warnings

warnings.filterwarnings('ignore')

from config.settings import config
from src.utils.logger import setup_logger
from src.utils.helpers import timing_decorator
from src.utils.database_manager import DatabaseManager

logger = setup_logger(__name__)

class DataStorage:
    """Centralized data storage management"""

    def __init__(self):
        self.db_manager = DatabaseManager()
        self.cache = {}
        self.cache_ttl = {}  # Time-to-live for cached data

        # Storage paths
        self.data_dir = Path(config.project_root) / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.cache_dir = self.data_dir / "cache"

        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache configuration
        self.cache_ttl_seconds = {
            'ohlcv': 300,        # 5 minutes for price data
            'features': 600,     # 10 minutes for features
            'signals': 60,       # 1 minute for signals
            'models': 3600,      # 1 hour for models
            'metadata': 1800     # 30 minutes for metadata
        }

        logger.info("Data storage initialized")

    async def initialize(self) -> bool:
        """Initialize data storage connections"""
        try:
            # Connect to database
            await self.db_manager.initialize()
            logger.info("Database connection initialized")

            logger.info("Data storage connections initialized")
            return True

        except Exception as e:
            logger.error(f"Data storage initialization failed: {e}")
            return False

    @timing_decorator

    async def store_ohlcv_data(self, symbol: str, timeframe: str,
                             df: pd.DataFrame, source: str = "exchange") -> bool:
        """Store OHLCV data"""
        try:
            if df.empty:
                logger.warning(f"Empty DataFrame provided for {symbol}_{timeframe}")
                return False

            # Store in database
            await self.db_manager.save_ohlcv_data(symbol, timeframe, df)
            success = True

            if success:
                # Also store as file backup
                await self._store_ohlcv_file(symbol, timeframe, df)

                # Update cache
                cache_key = f"ohlcv_{symbol}_{timeframe}"
                self.cache[cache_key] = df.copy()
                self.cache_ttl[cache_key] = datetime.now() + timedelta(seconds=self.cache_ttl_seconds['ohlcv'])

                logger.debug(f"Stored {len(df)} OHLCV records for {symbol}_{timeframe}")

            return success

        except Exception as e:
            logger.error(f"Failed to store OHLCV data: {e}")
            return False

    async def _store_ohlcv_file(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Store OHLCV data as file backup"""
        try:
            file_path = self.raw_data_dir / f"{symbol}_{timeframe}_ohlcv.parquet"

            # Load existing data if file exists
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)

                # Merge with new data
                combined_df = pd.concat([existing_df, df]).drop_duplicates().sort_index()
            else:
                combined_df = df.copy()

            # Save to parquet for efficiency
            combined_df.to_parquet(file_path)

        except Exception as e:
            logger.debug(f"Failed to store OHLCV file backup: {e}")

    @timing_decorator

    async def load_ohlcv_data(self, symbol: str, timeframe: str,
                            start_date: datetime = None, end_date: datetime = None,
                            limit: int = None) -> pd.DataFrame:
        """Load OHLCV data with caching"""
        try:
            cache_key = f"ohlcv_{symbol}_{timeframe}"

            # Check cache first
            if self._is_cache_valid(cache_key):
                cached_df = self.cache[cache_key]

                # Apply filters if specified
                if start_date or end_date or limit:
                    return self._filter_dataframe(cached_df, start_date, end_date, limit)
                else:
                    return cached_df.copy()

            # Load from database
            df = await self.db_manager.load_ohlcv_data(symbol, timeframe, start_date, end_date, limit)

            if df.empty:
                # Try loading from file backup
                df = await self._load_ohlcv_file(symbol, timeframe, start_date, end_date, limit)

            # Update cache
            if not df.empty:
                self.cache[cache_key] = df.copy()
                self.cache_ttl[cache_key] = datetime.now() + timedelta(seconds=self.cache_ttl_seconds['ohlcv'])

            return df

        except Exception as e:
            logger.error(f"Failed to load OHLCV data: {e}")
            return pd.DataFrame()

    async def _load_ohlcv_file(self, symbol: str, timeframe: str,
                             start_date: datetime = None, end_date: datetime = None,
                             limit: int = None) -> pd.DataFrame:
        """Load OHLCV data from file backup"""
        try:
            file_path = self.raw_data_dir / f"{symbol}_{timeframe}_ohlcv.parquet"

            if not file_path.exists():
                return pd.DataFrame()

            df = pd.read_parquet(file_path)

            # Apply filters
            df = self._filter_dataframe(df, start_date, end_date, limit)

            logger.debug(f"Loaded {len(df)} OHLCV records from file for {symbol}_{timeframe}")
            return df

        except Exception as e:
            logger.debug(f"Failed to load OHLCV file backup: {e}")
            return pd.DataFrame()

    def _filter_dataframe(self, df: pd.DataFrame, start_date: datetime = None,
                         end_date: datetime = None, limit: int = None) -> pd.DataFrame:
        """Apply date and limit filters to DataFrame"""
        try:
            if df.empty:
                return df

            filtered_df = df.copy()

            # Apply date filters
            if start_date:
                filtered_df = filtered_df[filtered_df.index >= start_date]

            if end_date:
                filtered_df = filtered_df[filtered_df.index <= end_date]

            # Apply limit
            if limit and len(filtered_df) > limit:
                filtered_df = filtered_df.tail(limit)

            return filtered_df

        except Exception as e:
            logger.error(f"DataFrame filtering failed: {e}")
            return df

    @timing_decorator

    async def store_feature_data(self, symbol: str, timeframe: str, version: str,
                               features_df: pd.DataFrame, feature_metadata: Dict[str, Any] = None) -> bool:
        """Store processed feature data"""
        try:
            if features_df.empty:
                logger.warning(f"Empty features DataFrame for {symbol}_{timeframe}_{version}")
                return False

            # Store in database
            success = await self.db_manager.store_feature_data(symbol, timeframe, version, features_df, feature_metadata)

            if success:
                # Store as file backup
                await self._store_feature_file(symbol, timeframe, version, features_df, feature_metadata)

                # Update cache
                cache_key = f"features_{symbol}_{timeframe}_{version}"
                self.cache[cache_key] = features_df.copy()
                self.cache_ttl[cache_key] = datetime.now() + timedelta(seconds=self.cache_ttl_seconds['features'])

                logger.debug(f"Stored {len(features_df)} feature records for {symbol}_{timeframe}_{version}")

            return success

        except Exception as e:
            logger.error(f"Failed to store feature data: {e}")
            return False

    async def _store_feature_file(self, symbol: str, timeframe: str, version: str,
                                features_df: pd.DataFrame, metadata: Dict[str, Any] = None):
        """Store feature data as file backup"""
        try:
            version_dir = self.processed_data_dir / symbol / timeframe / version
            version_dir.mkdir(parents=True, exist_ok=True)

            # Store features
            features_file = version_dir / "features.parquet"
            features_df.to_parquet(features_file)

            # Store metadata
            if metadata:
                metadata_file = version_dir / "feature_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

        except Exception as e:
            logger.debug(f"Failed to store feature file backup: {e}")

    async def load_feature_data(self, symbol: str, timeframe: str, version: str,
                              start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Load processed feature data"""
        try:
            cache_key = f"features_{symbol}_{timeframe}_{version}"

            # Check cache
            if self._is_cache_valid(cache_key):
                cached_df = self.cache[cache_key]
                return self._filter_dataframe(cached_df, start_date, end_date)

            # Load from database
            df = await self.db_manager.load_feature_data(symbol, timeframe, version, start_date, end_date)

            if df.empty:
                # Try loading from file
                df = await self._load_feature_file(symbol, timeframe, version, start_date, end_date)

            # Update cache
            if not df.empty:
                self.cache[cache_key] = df.copy()
                self.cache_ttl[cache_key] = datetime.now() + timedelta(seconds=self.cache_ttl_seconds['features'])

            return df

        except Exception as e:
            logger.error(f"Failed to load feature data: {e}")
            return pd.DataFrame()

    async def _load_feature_file(self, symbol: str, timeframe: str, version: str,
                               start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Load feature data from file backup"""
        try:
            version_dir = self.processed_data_dir / symbol / timeframe / version
            features_file = version_dir / "features.parquet"

            if not features_file.exists():
                return pd.DataFrame()

            df = pd.read_parquet(features_file)
            df = self._filter_dataframe(df, start_date, end_date)

            logger.debug(f"Loaded {len(df)} feature records from file for {symbol}_{timeframe}_{version}")
            return df

        except Exception as e:
            logger.debug(f"Failed to load feature file backup: {e}")
            return pd.DataFrame()

    async def store_model_data(self, symbol: str, timeframe: str, version: str,
                             model_data: bytes, model_metadata: Dict[str, Any]) -> bool:
        """Store trained model data"""
        try:
            # Store in database
            success = await self.db_manager.store_model_metadata(symbol, timeframe, version, model_metadata)

            if success:
                # Store model file
                await self._store_model_file(symbol, timeframe, version, model_data, model_metadata)

                # Update cache
                cache_key = f"model_metadata_{symbol}_{timeframe}_{version}"
                self.cache[cache_key] = model_metadata.copy()
                self.cache_ttl[cache_key] = datetime.now() + timedelta(seconds=self.cache_ttl_seconds['models'])

                logger.debug(f"Stored model data for {symbol}_{timeframe}_{version}")

            return success

        except Exception as e:
            logger.error(f"Failed to store model data: {e}")
            return False

    async def _store_model_file(self, symbol: str, timeframe: str, version: str,
                              model_data: bytes, metadata: Dict[str, Any]):
        """Store model as file backup"""
        try:
            version_dir = self.processed_data_dir / symbol / timeframe / version
            version_dir.mkdir(parents=True, exist_ok=True)

            # Store model
            model_file = version_dir / "model.pkl"
            with open(model_file, 'wb') as f:
                f.write(model_data)

            # Store metadata
            metadata_file = version_dir / "model_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

        except Exception as e:
            logger.debug(f"Failed to store model file backup: {e}")

    async def load_model_data(self, symbol: str, timeframe: str, version: str) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """Load trained model data"""
        try:
            cache_key = f"model_metadata_{symbol}_{timeframe}_{version}"

            # Load metadata from cache or database
            if self._is_cache_valid(cache_key):
                metadata = self.cache[cache_key]
            else:
                metadata = await self.db_manager.load_model_metadata(symbol, timeframe, version)
                if metadata:
                    self.cache[cache_key] = metadata.copy()
                    self.cache_ttl[cache_key] = datetime.now() + timedelta(seconds=self.cache_ttl_seconds['models'])

            if not metadata:
                return None, {}

            # Load model data from file
            model_data = await self._load_model_file(symbol, timeframe, version)

            return model_data, metadata

        except Exception as e:
            logger.error(f"Failed to load model data: {e}")
            return None, {}

    async def _load_model_file(self, symbol: str, timeframe: str, version: str) -> Optional[bytes]:
        """Load model data from file"""
        try:
            version_dir = self.processed_data_dir / symbol / timeframe / version
            model_file = version_dir / "model.pkl"

            if not model_file.exists():
                return None

            with open(model_file, 'rb') as f:
                model_data = f.read()

            return model_data

        except Exception as e:
            logger.debug(f"Failed to load model file: {e}")
            return None

    async def store_trading_signals(self, signals: List[Dict[str, Any]]) -> bool:
        """Store trading signals"""
        try:
            if not signals:
                return True

            success = await self.db_manager.store_trading_signals(signals)

            if success:
                # Store recent signals in cache
                cache_key = "recent_signals"
                if cache_key not in self.cache:
                    self.cache[cache_key] = []

                self.cache[cache_key].extend(signals)

                # Keep only recent signals (last 1000)
                if len(self.cache[cache_key]) > 1000:
                    self.cache[cache_key] = self.cache[cache_key][-1000:]

                self.cache_ttl[cache_key] = datetime.now() + timedelta(seconds=self.cache_ttl_seconds['signals'])

                logger.debug(f"Stored {len(signals)} trading signals")

            return success

        except Exception as e:
            logger.error(f"Failed to store trading signals: {e}")
            return False

    async def load_trading_signals(self, symbol: str = None, start_date: datetime = None,
                                 end_date: datetime = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Load trading signals"""
        try:
            # Check cache for recent signals
            cache_key = "recent_signals"
            if self._is_cache_valid(cache_key) and not start_date and not end_date:
                cached_signals = self.cache[cache_key]

                # Filter by symbol if specified
                if symbol:
                    cached_signals = [s for s in cached_signals if s.get('symbol') == symbol]

                return cached_signals[-limit:] if limit else cached_signals

            # Load from database
            signals = await self.db_manager.load_trading_signals(symbol, start_date, end_date, limit)

            return signals

        except Exception as e:
            logger.error(f"Failed to load trading signals: {e}")
            return []

    async def store_portfolio_state(self, portfolio_data: Dict[str, Any]) -> bool:
        """Store portfolio state"""
        try:
            success = await self.db_manager.store_portfolio_state(portfolio_data)

            if success:
                # Update cache
                cache_key = "portfolio_state"
                self.cache[cache_key] = portfolio_data.copy()
                self.cache_ttl[cache_key] = datetime.now() + timedelta(seconds=300)  # 5 minutes

            return success

        except Exception as e:
            logger.error(f"Failed to store portfolio state: {e}")
            return False

    async def load_portfolio_state(self) -> Dict[str, Any]:
        """Load latest portfolio state"""
        try:
            cache_key = "portfolio_state"

            # Check cache
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]

            # Load from database
            portfolio_data = await self.db_manager.load_portfolio_state()

            # Update cache
            if portfolio_data:
                self.cache[cache_key] = portfolio_data.copy()
                self.cache_ttl[cache_key] = datetime.now() + timedelta(seconds=300)

            return portfolio_data or {}

        except Exception as e:
            logger.error(f"Failed to load portfolio state: {e}")
            return {}

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        try:
            if cache_key not in self.cache:
                return False

            if cache_key not in self.cache_ttl:
                return False

            return datetime.now() < self.cache_ttl[cache_key]

        except Exception:
            return False

    def clear_cache(self, cache_type: str = None):
        """Clear cache data"""
        try:
            if cache_type:
                # Clear specific type of cache
                keys_to_remove = [k for k in self.cache.keys() if cache_type in k]
                for key in keys_to_remove:
                    if key in self.cache:
                        del self.cache[key]
                    if key in self.cache_ttl:
                        del self.cache_ttl[key]

                logger.info(f"Cleared {cache_type} cache ({len(keys_to_remove)} entries)")
            else:
                # Clear all cache
                cache_size = len(self.cache)
                self.cache.clear()
                self.cache_ttl.clear()

                logger.info(f"Cleared all cache ({cache_size} entries)")

        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            stats = {
                'total_entries': len(self.cache),
                'cache_types': {},
                'cache_sizes': {},
                'expired_entries': 0
            }

            # Analyze cache contents
            for key in self.cache.keys():
                cache_type = key.split('_')[0]

                if cache_type not in stats['cache_types']:
                    stats['cache_types'][cache_type] = 0
                stats['cache_types'][cache_type] += 1

                # Calculate cache size (approximate)
                try:
                    import sys
                    size = sys.getsizeof(self.cache[key])
                    stats['cache_sizes'][key] = size
                except:
                    pass

                # Check if expired
                if not self._is_cache_valid(key):
                    stats['expired_entries'] += 1

            return stats

        except Exception as e:
            logger.error(f"Cache stats generation failed: {e}")
            return {}

    async def backup_data(self, backup_path: Path) -> bool:
        """Create data backup"""
        try:
            backup_path.mkdir(parents=True, exist_ok=True)

            # Backup database
            if self.db_manager:
                db_backup = backup_path / "database_backup.sql"
                await self.db_manager.backup_database(str(db_backup))

            # Backup file data
            import shutil

            if self.raw_data_dir.exists():
                shutil.copytree(self.raw_data_dir, backup_path / "raw_data", dirs_exist_ok=True)

            if self.processed_data_dir.exists():
                shutil.copytree(self.processed_data_dir, backup_path / "processed_data", dirs_exist_ok=True)

            logger.info(f"Data backup created at {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Data backup failed: {e}")
            return False

    async def cleanup_old_data(self, days_to_keep: int = 30) -> bool:
        """Clean up old data files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            # Clean up old cache files
            cache_files = list(self.cache_dir.glob("*"))
            cleaned_count = 0

            for file_path in cache_files:
                try:
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_date:
                            file_path.unlink()
                            cleaned_count += 1
                except Exception:
                    continue

            # Clean up database old records
            if self.db_manager:
                await self.db_manager.cleanup_old_records(cutoff_date)

            logger.info(f"Cleaned up {cleaned_count} old files")
            return True

        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            return False

    async def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data storage summary"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'storage_paths': {
                    'data_dir': str(self.data_dir),
                    'raw_data_dir': str(self.raw_data_dir),
                    'processed_data_dir': str(self.processed_data_dir),
                    'cache_dir': str(self.cache_dir)
                },
                'cache_stats': self.get_cache_stats(),
                'database_connected': self.db_manager.is_connected() if self.db_manager else False
            }

            # File system statistics
            try:
                import os
                total_size = 0
                file_count = 0

                for root, dirs, files in os.walk(self.data_dir):
                    for file in files:
                        file_path = Path(root) / file
                        total_size += file_path.stat().st_size
                        file_count += 1

                summary['filesystem'] = {
                    'total_size_mb': total_size / (1024 * 1024),
                    'total_files': file_count
                }
            except Exception:
                summary['filesystem'] = {'error': 'Unable to calculate filesystem stats'}

            return summary

        except Exception as e:
            logger.error(f"Data summary generation failed: {e}")
            return {'error': str(e)}

# Convenience functions

async def create_data_storage() -> DataStorage:
    """Create and initialize data storage"""
    storage = DataStorage()
    await storage.initialize()
    return storage

# Usage example
if __name__ == "__main__":

    async def test_data_storage():
        # Create data storage
        storage = DataStorage()
        await storage.initialize()

        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        sample_df = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 45000,
            'high': np.random.randn(100).cumsum() + 45100,
            'low': np.random.randn(100).cumsum() + 44900,
            'close': np.random.randn(100).cumsum() + 45000,
            'volume': np.random.randint(1000, 5000, 100)
        }, index=dates)

        # Test OHLCV storage and retrieval
        print("Testing OHLCV data storage...")
        success = await storage.store_ohlcv_data('BTCUSDT', '1h', sample_df)
        print(f"Storage success: {success}")

        loaded_df = await storage.load_ohlcv_data('BTCUSDT', '1h')
        print(f"Loaded {len(loaded_df)} records")

        # Test caching
        print("Testing cache...")
        cache_stats = storage.get_cache_stats()
        print(f"Cache entries: {cache_stats['total_entries']}")

        # Test data summary
        summary = await storage.get_data_summary()
        print(f"Database connected: {summary['database_connected']}")
        print(f"Cache types: {summary['cache_stats']['cache_types']}")

        print("Data storage test completed!")

    # Run test
    asyncio.run(test_data_storage())
