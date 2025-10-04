"""
Database Manager Module
Handles all database operations for the crypto quant trading system
Supports both SQLite (development) and PostgreSQL+TimescaleDB (production)
"""

import asyncio
import asyncpg
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import yaml
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from config.settings import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

Base = declarative_base()

class OHLCVData(Base):
    """OHLCV data table schema"""
    __tablename__ = 'ohlcv_data'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    # External data fields
    active_addresses = Column(Integer)
    tx_count = Column(Integer)
    whale_volume = Column(Float)
    open_interest = Column(Float)
    funding_rate = Column(Float)
    tweet_count = Column(Integer)
    sentiment_score = Column(Float)
    fear_greed_index = Column(Integer)
    dxy_index = Column(Float)
    vix_index = Column(Float)

    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
        Index('idx_timestamp_desc', 'timestamp', postgresql_using='btree'),
    )

class FeatureData(Base):
    """Feature data table schema"""
    __tablename__ = 'feature_data'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    version = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    features = Column(JSON, nullable=False)
    label = Column(Integer)  # -1, 0, 1

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index('idx_feature_symbol_timeframe_version', 'symbol', 'timeframe', 'version'),
    )

class ModelMetadata(Base):
    """Model metadata table schema"""
    __tablename__ = 'model_metadata'

    id = Column(Integer, primary_key=True)
    model_id = Column(String(50), nullable=False, unique=True, index=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    version = Column(String(10), nullable=False)
    model_type = Column(String(20), nullable=False)

    # Model configuration
    hyperparameters = Column(JSON)
    feature_list = Column(JSON)
    feature_importance = Column(JSON)

    # Performance metrics
    training_accuracy = Column(Float)
    validation_accuracy = Column(Float)
    cross_validation_score = Column(Float)
    backtest_performance = Column(JSON)

    # Status and timestamps
    status = Column(String(20), default='trained')  # trained, deployed, retired
    training_date = Column(DateTime, default=func.now())
    deployment_date = Column(DateTime)
    retirement_date = Column(DateTime)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_model_symbol_timeframe_version', 'symbol', 'timeframe', 'version'),
    )

class TradingSignals(Base):
    """Trading signals table schema"""
    __tablename__ = 'trading_signals'

    id = Column(Integer, primary_key=True)
    signal_id = Column(String(50), nullable=False, unique=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    model_version = Column(String(10), nullable=False)

    # Signal data
    signal_type = Column(Integer, nullable=False)  # -1: sell, 0: hold, 1: buy
    confidence = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    features_snapshot = Column(JSON)

    # Execution data
    executed = Column(Boolean, default=False)
    execution_price = Column(Float)
    execution_time = Column(DateTime)
    order_id = Column(String(50))

    # Results
    result_pnl = Column(Float)
    result_pnl_pct = Column(Float)
    closed_at = Column(DateTime)

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index('idx_signals_symbol_created', 'symbol', 'created_at'),
    )

class DatabaseManager:
    """Main database manager supporting both SQLite and PostgreSQL"""

    def __init__(self):
        self.db_url = config.database.current_url
        self.is_postgres = self.db_url.startswith('postgresql')
        self.engine = None
        self.session_maker = None
        self.pool = None

    async def initialize(self):
        """Initialize database connection and create tables"""
        try:
            if self.is_postgres:
                await self._initialize_postgres()
            else:
                await self._initialize_sqlite()

            logger.info(f"Database initialized: {'PostgreSQL' if self.is_postgres else 'SQLite'}")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    async def _initialize_postgres(self):
        """Initialize PostgreSQL connection"""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        # Create engine
        self.engine = create_engine(
            self.db_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False
        )

        # Create session maker
        self.session_maker = sessionmaker(bind=self.engine)

        # Create connection pool for async queries
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create TimescaleDB hypertables
        await self._create_hypertables()

    async def _initialize_sqlite(self):
        """Initialize SQLite connection"""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        # Ensure directory exists
        db_path = Path(self.db_url.replace('sqlite:///', ''))
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine
        self.engine = create_engine(
            self.db_url,
            echo=False,
            connect_args={'check_same_thread': False}
        )

        # Create session maker
        self.session_maker = sessionmaker(bind=self.engine)

        # Create tables
        Base.metadata.create_all(self.engine)

    async def _create_hypertables(self):
        """Create TimescaleDB hypertables for time-series data"""
        try:
            async with self.pool.acquire() as conn:
                # Create hypertable for OHLCV data
                await conn.execute("""
                    SELECT create_hypertable('ohlcv_data', 'timestamp',
                                           chunk_time_interval => INTERVAL '1 day',
                                           if_not_exists => TRUE);
                """)

                # Create hypertable for feature data
                await conn.execute("""
                    SELECT create_hypertable('feature_data', 'timestamp',
                                           chunk_time_interval => INTERVAL '1 day',
                                           if_not_exists => TRUE);
                """)

                # Create hypertable for trading signals
                await conn.execute("""
                    SELECT create_hypertable('trading_signals', 'created_at',
                                           chunk_time_interval => INTERVAL '1 day',
                                           if_not_exists => TRUE);
                """)

                logger.info("TimescaleDB hypertables created successfully")

        except Exception as e:
            logger.warning(f"Failed to create hypertables (may already exist): {e}")

    async def save_ohlcv_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Save OHLCV data to database"""
        try:
            if df.empty:
                logger.warning(f"Empty DataFrame for {symbol}_{timeframe}")
                return

            # Prepare data
            df_copy = df.copy()
            df_copy.reset_index(inplace=True)

            # Add symbol and timeframe columns
            df_copy['symbol'] = symbol
            df_copy['timeframe'] = timeframe

            # Ensure timestamp column
            if 'timestamp' not in df_copy.columns:
                df_copy['timestamp'] = df_copy.index

            # Convert timestamp to string format for SQLite compatibility
            if 'timestamp' in df_copy.columns:
                df_copy['timestamp'] = df_copy['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # Convert to records
            records = df_copy.to_dict('records')

            if self.is_postgres:
                await self._save_records_postgres('ohlcv_data', records)
            else:
                await self._save_records_sqlite('ohlcv_data', records)

            logger.info(f"Saved {len(records)} OHLCV records for {symbol}_{timeframe}")

        except Exception as e:
            logger.error(f"Failed to save OHLCV data for {symbol}_{timeframe}: {e}")
            raise

    async def load_ohlcv_data(self, symbol: str, timeframe: str,
                            start_date: datetime = None, end_date: datetime = None,
                            limit: int = None) -> pd.DataFrame:
        """Load OHLCV data from database"""
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume,
                       active_addresses, tx_count, whale_volume,
                       open_interest, funding_rate, tweet_count,
                       sentiment_score, fear_greed_index, dxy_index, vix_index
                FROM ohlcv_data
                WHERE symbol = $1 AND timeframe = $2
            """

            params = [symbol, timeframe]

            # Add date filters
            if start_date:
                query += f" AND timestamp >= ${len(params) + 1}"
                params.append(start_date)

            if end_date:
                query += f" AND timestamp <= ${len(params) + 1}"
                params.append(end_date)

            query += " ORDER BY timestamp ASC"

            if limit:
                query += f" LIMIT ${len(params) + 1}"
                params.append(limit)

            if self.is_postgres:
                async with self.pool.acquire() as conn:
                    rows = await conn.fetch(query, *params)

                    if not rows:
                        return pd.DataFrame()

                    df = pd.DataFrame([dict(row) for row in rows])
                    df.set_index('timestamp', inplace=True)
                    return df
            else:
                # SQLite version
                query_sqlite = query.replace('$', '?')
                for i in range(len(params)):
                    query_sqlite = query_sqlite.replace(f'?{i+1}', '?')

                with sqlite3.connect(self.db_url.replace('sqlite:///', '')) as conn:
                    df = pd.read_sql_query(query_sqlite, conn, params=params, index_col='timestamp', parse_dates=['timestamp'])
                    return df

        except Exception as e:
            logger.error(f"Failed to load OHLCV data for {symbol}_{timeframe}: {e}")
            return pd.DataFrame()

    async def save_feature_data(self, symbol: str, timeframe: str, version: str,
                              df: pd.DataFrame, labels: pd.Series = None):
        """Save feature data to database"""
        try:
            if df.empty:
                logger.warning(f"Empty feature DataFrame for {symbol}_{timeframe}_{version}")
                return

            records = []
            for idx, row in df.iterrows():
                record = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'version': version,
                    'timestamp': idx,
                    'features': row.to_dict(),
                    'label': labels.loc[idx] if labels is not None and idx in labels.index else None
                }
                records.append(record)

            if self.is_postgres:
                await self._save_records_postgres('feature_data', records)
            else:
                await self._save_records_sqlite('feature_data', records)

            logger.info(f"Saved {len(records)} feature records for {symbol}_{timeframe}_{version}")

        except Exception as e:
            logger.error(f"Failed to save feature data: {e}")
            raise

    async def load_feature_data(self, symbol: str, timeframe: str, version: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load feature data from database"""
        try:
            query = """
                SELECT timestamp, features, label
                FROM feature_data
                WHERE symbol = $1 AND timeframe = $2 AND version = $3
                ORDER BY timestamp ASC
            """

            if self.is_postgres:
                async with self.pool.acquire() as conn:
                    rows = await conn.fetch(query, symbol, timeframe, version)

                    if not rows:
                        return pd.DataFrame(), pd.Series()

                    # Extract features and labels
                    timestamps = [row['timestamp'] for row in rows]
                    features_list = [row['features'] for row in rows]
                    labels_list = [row['label'] for row in rows if row['label'] is not None]

                    # Create DataFrames
                    df_features = pd.DataFrame(features_list, index=timestamps)
                    series_labels = pd.Series(labels_list,
                                            index=[timestamps[i] for i, row in enumerate(rows) if row['label'] is not None])

                    return df_features, series_labels
            else:
                # SQLite implementation
                query_sqlite = query.replace('$1', '?').replace('$2', '?').replace('$3', '?')

                with sqlite3.connect(self.db_url.replace('sqlite:///', '')) as conn:
                    cursor = conn.execute(query_sqlite, (symbol, timeframe, version))
                    rows = cursor.fetchall()

                    if not rows:
                        return pd.DataFrame(), pd.Series()

                    timestamps = [row[0] for row in rows]
                    features_list = [json.loads(row[1]) for row in rows]
                    labels_list = [row[2] for row in rows if row[2] is not None]

                    df_features = pd.DataFrame(features_list, index=timestamps)
                    series_labels = pd.Series(labels_list,
                                            index=[timestamps[i] for i, row in enumerate(rows) if row[2] is not None])

                    return df_features, series_labels

        except Exception as e:
            logger.error(f"Failed to load feature data: {e}")
            return pd.DataFrame(), pd.Series()

    async def save_model_metadata(self, model_data: Dict[str, Any]):
        """Save model metadata to database"""
        try:
            record = {
                'model_id': f"{model_data['symbol']}_{model_data['timeframe']}_{model_data['version']}",
                'symbol': model_data['symbol'],
                'timeframe': model_data['timeframe'],
                'version': model_data['version'],
                'model_type': model_data['model_type'],
                'hyperparameters': model_data.get('hyperparameters', {}),
                'feature_list': model_data.get('feature_list', []),
                'feature_importance': model_data.get('feature_importance', {}),
                'training_accuracy': model_data.get('training_accuracy'),
                'validation_accuracy': model_data.get('validation_accuracy'),
                'cross_validation_score': model_data.get('cross_validation_score'),
                'backtest_performance': model_data.get('backtest_performance', {}),
                'status': model_data.get('status', 'trained')
            }

            if self.is_postgres:
                await self._save_records_postgres('model_metadata', [record])
            else:
                await self._save_records_sqlite('model_metadata', [record])

            logger.info(f"Saved model metadata: {record['model_id']}")

        except Exception as e:
            logger.error(f"Failed to save model metadata: {e}")
            raise

    async def save_trading_signal(self, signal_data: Dict[str, Any]):
        """Save trading signal to database"""
        try:
            record = {
                'signal_id': str(uuid.uuid4()),
                'symbol': signal_data['symbol'],
                'timeframe': signal_data['timeframe'],
                'model_version': signal_data['model_version'],
                'signal_type': signal_data['signal_type'],
                'confidence': signal_data['confidence'],
                'price': signal_data['price'],
                'features_snapshot': signal_data.get('features_snapshot', {})
            }

            if self.is_postgres:
                await self._save_records_postgres('trading_signals', [record])
            else:
                await self._save_records_sqlite('trading_signals', [record])

            logger.info(f"Saved trading signal: {record['signal_id']}")
            return record['signal_id']

        except Exception as e:
            logger.error(f"Failed to save trading signal: {e}")
            raise

    async def _save_records_postgres(self, table_name: str, records: List[Dict]):
        """Save records to PostgreSQL using bulk insert"""
        if not records:
            return

        try:
            async with self.pool.acquire() as conn:
                # Build insert query
                columns = list(records[0].keys())
                placeholders = ', '.join([f'${i+1}' for i in range(len(columns))])
                query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

                # Prepare values
                values_list = []
                for record in records:
                    values = []
                    for col in columns:
                        value = record.get(col)
                        if isinstance(value, dict):
                            value = json.dumps(value)
                        values.append(value)
                    values_list.append(values)

                # Execute bulk insert
                await conn.executemany(query, values_list)

        except Exception as e:
            logger.error(f"Failed to save records to {table_name}: {e}")
            raise

    async def _save_records_sqlite(self, table_name: str, records: List[Dict]):
        """Save records to SQLite with performance optimization"""
        if not records:
            return

        try:
            db_path = self.db_url.replace('sqlite:///', '')

            with sqlite3.connect(db_path) as conn:
                # Enhanced SQLite performance optimizations
                conn.execute("PRAGMA synchronous = OFF")       # Fastest, disable sync
                conn.execute("PRAGMA cache_size = -200000")    # 200MB cache (negative = KB)
                conn.execute("PRAGMA temp_store = MEMORY")     # Use memory for temp
                conn.execute("PRAGMA journal_mode = WAL")      # Write-ahead logging
                conn.execute("PRAGMA wal_autocheckpoint = 10000")  # WAL checkpoint every 10K pages
                conn.execute("PRAGMA page_size = 32768")       # Larger page size for bulk inserts

                # Build insert query with IGNORE to handle duplicates
                columns = list(records[0].keys())
                placeholders = ', '.join(['?' for _ in columns])
                query = f"INSERT OR IGNORE INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

                # Optimized batch size for faster inserts
                batch_size = 25000  # Increased from 10K to 25K records per batch
                total_processed = 0

                print(f"💾 開始寫入數據到 {table_name}...")

                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]

                    # Prepare values for this batch
                    values_list = []
                    for record in batch:
                        values = []
                        for col in columns:
                            value = record.get(col)
                            if isinstance(value, dict):
                                value = json.dumps(value)
                            values.append(value)
                        values_list.append(values)

                    # Execute batch insert within transaction for better performance
                    conn.execute("BEGIN IMMEDIATE TRANSACTION")  # Use IMMEDIATE for faster writes
                    conn.executemany(query, values_list)
                    conn.execute("COMMIT")

                    total_processed += len(batch)

                    # Enhanced progress display
                    progress_percent = (total_processed / len(records)) * 100
                    if i % batch_size == 0:  # Show progress after each batch
                        print(f"   💾 寫入進度: {progress_percent:.1f}% | 已處理: {total_processed:,}/{len(records):,} 條記錄")

                    # Log less frequently to avoid spam
                    if i % (batch_size * 4) == 0:  # Log every 4 batches (100K records)
                        logger.info(f"Processed {total_processed}/{len(records)} records...")

                print(f"   ✅ 數據寫入完成: {total_processed:,} 條記錄已保存到 {table_name}")
                logger.info(f"Successfully saved {total_processed} records to {table_name}")

        except Exception as e:
            logger.error(f"Failed to save records to {table_name}: {e}")
            raise

    async def save_realtime_data(self, symbol: str, data: Dict[str, Any]):
        """Save real-time market data"""
        try:
            # Convert to OHLCV-like format for storage
            record = {
                'symbol': symbol,
                'timeframe': '1m',  # Real-time data as 1-minute bars
                'timestamp': data['timestamp'],
                'open': data['price'],
                'high': data['price'],
                'low': data['price'],
                'close': data['price'],
                'volume': 0,  # No volume for real-time ticks
                'open_interest': data.get('open_interest', 0),
                'funding_rate': data.get('funding_rate', 0),
                'active_addresses': data.get('active_addresses', 0),
                'tx_count': data.get('transaction_count', 0),
                'tweet_count': data.get('tweet_count', 0),
                'sentiment_score': data.get('sentiment_score', 0.5),
                'fear_greed_index': data.get('fear_greed_index', 50)
            }

            if self.is_postgres:
                await self._save_records_postgres('ohlcv_data', [record])
            else:
                await self._save_records_sqlite('ohlcv_data', [record])

        except Exception as e:
            logger.error(f"Failed to save real-time data for {symbol}: {e}")

    async def get_latest_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Get latest data for a symbol/timeframe"""
        return await self.load_ohlcv_data(symbol, timeframe, limit=limit)

    async def test_connection(self) -> bool:
        """Test database connection"""
        try:
            if self.is_postgres:
                async with self.pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
            else:
                with sqlite3.connect(self.db_url.replace('sqlite:///', '')) as conn:
                    conn.execute("SELECT 1").fetchone()

            logger.info("Database connection test successful")
            return True

        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to save space"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            # Clean up old real-time data (keep only daily and above)
            query = """
                DELETE FROM ohlcv_data
                WHERE timeframe = '1m' AND timestamp < $1
            """

            if self.is_postgres:
                async with self.pool.acquire() as conn:
                    result = await conn.execute(query, cutoff_date)
                    logger.info(f"Cleaned up old real-time data: {result}")
            else:
                with sqlite3.connect(self.db_url.replace('sqlite:///', '')) as conn:
                    cursor = conn.execute(query.replace('$1', '?'), (cutoff_date,))
                    conn.commit()
                    logger.info(f"Cleaned up old real-time data: {cursor.rowcount} rows")

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

    async def close(self):
        """Close database connections"""
        try:
            if self.pool:
                await self.pool.close()

            if self.engine:
                self.engine.dispose()

            logger.info("Database connections closed")

        except Exception as e:
            logger.error(f"Failed to close database connections: {e}")

# Usage examples and utility functions

async def create_test_data():
    """Create test data for development"""
    db = DatabaseManager()
    await db.initialize()

    # Test OHLCV data
    test_df = pd.DataFrame({
        'open': [45000, 45100, 45200],
        'high': [45500, 45600, 45700],
        'low': [44800, 44900, 45000],
        'close': [45100, 45200, 45300],
        'volume': [1000, 1100, 1200]
    }, index=pd.date_range('2024-01-01', periods=3, freq='1H'))

    await db.save_ohlcv_data('BTCUSDT', '1h', test_df)

    # Load and verify
    loaded_df = await db.load_ohlcv_data('BTCUSDT', '1h')
    print(f"Loaded {len(loaded_df)} records")
    print(loaded_df.head())

    await db.close()

if __name__ == "__main__":
    # Run test
    asyncio.run(create_test_data())
