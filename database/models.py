"""
Database Schema and Configuration Module
Defines database tables, relationships, and configuration for the trading system
Supports both SQLite (development) and PostgreSQL+TimescaleDB (production)
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from sqlalchemy.types import TypeDecorator, String as SQLString
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

from config.settings import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

Base = declarative_base()

class JSONEncodedDict(TypeDecorator):
    """Custom JSON type that works with both SQLite and PostgreSQL"""
    impl = SQLString

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value

# ==================== Core Data Tables ====================

class TradingPair(Base):
    """Trading pairs configuration"""
    __tablename__ = 'trading_pairs'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    base_currency = Column(String(10), nullable=False)
    quote_currency = Column(String(10), nullable=False)

    # Trading configuration
    is_active = Column(Boolean, default=True)
    min_quantity = Column(Float, default=0.0)
    max_quantity = Column(Float, default=1000000.0)
    price_precision = Column(Integer, default=8)
    quantity_precision = Column(Integer, default=8)

    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class OHLCVData(Base):
    """OHLCV market data with external indicators"""
    __tablename__ = 'ohlcv_data'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Core OHLCV data
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

    # Technical indicators (pre-calculated for performance)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    ema_20 = Column(Float)
    ema_50 = Column(Float)
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    bb_upper = Column(Float)
    bb_lower = Column(Float)
    atr_14 = Column(Float)

    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
        Index('idx_timestamp_desc', 'timestamp', postgresql_using='btree'),
    )

class FeatureData(Base):
    """Processed feature data for ML models"""
    __tablename__ = 'feature_data'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    version = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Feature data (stored as JSON)
    features = Column(JSONEncodedDict)

    # Label information
    label = Column(Integer)  # -1: SHORT, 0: NEUTRAL, 1: LONG
    future_return = Column(Float)

    # Feature metadata
    feature_count = Column(Integer)
    feature_hash = Column(String(64))  # Hash of feature names for consistency

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index('idx_feature_symbol_timeframe_version', 'symbol', 'timeframe', 'version'),
        Index('idx_feature_timestamp', 'timestamp'),
    )

# ==================== Model Management Tables ====================

class ModelMetadata(Base):
    """Model metadata and configuration"""
    __tablename__ = 'model_metadata'

    id = Column(Integer, primary_key=True)
    model_id = Column(String(50), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    version = Column(String(10), nullable=False)
    model_type = Column(String(20), nullable=False)  # LightGBM, RandomForest, LSTM

    # Model configuration
    hyperparameters = Column(JSONEncodedDict)
    feature_list = Column(JSONEncodedDict)  # List of feature names
    feature_importance = Column(JSONEncodedDict)

    # Training information
    training_samples = Column(Integer)
    training_duration_seconds = Column(Float)
    training_start_date = Column(DateTime)
    training_end_date = Column(DateTime)

    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_score = Column(Float)
    cross_validation_score = Column(Float)

    # Backtest performance
    backtest_return = Column(Float)
    backtest_sharpe = Column(Float)
    backtest_max_drawdown = Column(Float)
    backtest_win_rate = Column(Float)

    # Model lifecycle
    status = Column(String(20), default='trained')  # trained, deployed, retired
    deployment_date = Column(DateTime)
    retirement_date = Column(DateTime)

    # File paths
    model_file_path = Column(String(255))
    config_file_path = Column(String(255))

    # Metadata
    description = Column(Text)
    tags = Column(JSONEncodedDict)  # List of tags
    parent_model_id = Column(String(50))  # For model versioning

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_model_symbol_timeframe_version', 'symbol', 'timeframe', 'version'),
        Index('idx_model_status', 'status'),
    )

class OptimizationRun(Base):
    """Optuna optimization run metadata"""
    __tablename__ = 'optimization_runs'

    id = Column(Integer, primary_key=True)
    study_name = Column(String(100), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    optimization_type = Column(String(20), nullable=False)  # feature, model, combined

    # Optimization configuration
    n_trials = Column(Integer, nullable=False)
    objective_name = Column(String(50), nullable=False)
    optimization_direction = Column(String(10), nullable=False)  # maximize, minimize

    # Results
    best_score = Column(Float)
    best_params = Column(JSONEncodedDict)
    best_trial_number = Column(Integer)

    # Execution info
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    duration_seconds = Column(Float)
    status = Column(String(20), default='running')  # running, completed, failed

    # Associated model (if created)
    resulting_model_id = Column(String(50))

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index('idx_opt_symbol_timeframe', 'symbol', 'timeframe'),
        Index('idx_opt_study_name', 'study_name'),
    )

class OptimizationTrial(Base):
    """Individual Optuna trial results"""
    __tablename__ = 'optimization_trials'

    id = Column(Integer, primary_key=True)
    optimization_run_id = Column(Integer, ForeignKey('optimization_runs.id'), nullable=False)
    trial_number = Column(Integer, nullable=False)

    # Trial parameters and results
    params = Column(JSONEncodedDict, nullable=False)
    score = Column(Float)
    state = Column(String(20), nullable=False)  # COMPLETE, FAIL, PRUNED

    # Execution info
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    duration_seconds = Column(Float)

    # Additional metrics (if available)
    intermediate_values = Column(JSONEncodedDict)
    user_attrs = Column(JSONEncodedDict)

    created_at = Column(DateTime, default=func.now())

    # Relationship
    optimization_run = relationship("OptimizationRun", backref="trials")

    __table_args__ = (
        Index('idx_trial_run_number', 'optimization_run_id', 'trial_number'),
    )

# ==================== Trading Operations Tables ====================

class TradingSignal(Base):
    """Generated trading signals"""
    __tablename__ = 'trading_signals'

    id = Column(Integer, primary_key=True)
    signal_id = Column(String(50), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    model_id = Column(String(50), nullable=False)

    # Signal data
    signal_type = Column(Integer, nullable=False)  # -1: SELL, 0: HOLD, 1: BUY
    confidence = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    target_price = Column(Float)
    stop_loss_price = Column(Float)

    # Feature snapshot (for analysis)
    features_snapshot = Column(JSONEncodedDict)

    # Execution tracking
    is_executed = Column(Boolean, default=False)
    execution_price = Column(Float)
    execution_time = Column(DateTime)
    execution_order_id = Column(String(50))

    # Results tracking
    is_closed = Column(Boolean, default=False)
    close_price = Column(Float)
    close_time = Column(DateTime)
    pnl = Column(Float)
    pnl_percentage = Column(Float)

    # Signal quality metrics
    signal_strength = Column(Float)  # Additional confidence metric
    market_conditions = Column(JSONEncodedDict)  # Market context

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_signal_symbol_created', 'symbol', 'created_at'),
        Index('idx_signal_execution_status', 'is_executed', 'is_closed'),
    )

class TradingOrder(Base):
    """Trading orders (both simulated and real)"""
    __tablename__ = 'trading_orders'

    id = Column(Integer, primary_key=True)
    order_id = Column(String(50), unique=True, nullable=False, index=True)
    signal_id = Column(String(50), ForeignKey('trading_signals.signal_id'))

    # Order details
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY, SELL
    order_type = Column(String(20), nullable=False)  # MARKET, LIMIT, STOP_LOSS
    quantity = Column(Float, nullable=False)
    price = Column(Float)  # For limit orders

    # Execution details
    status = Column(String(20), nullable=False)  # NEW, FILLED, CANCELED, REJECTED
    filled_quantity = Column(Float, default=0.0)
    average_price = Column(Float)
    commission = Column(Float, default=0.0)
    commission_asset = Column(String(10))

    # Trading mode
    is_simulation = Column(Boolean, default=True)
    exchange_order_id = Column(String(50))  # Real exchange order ID

    # Timestamps
    created_at = Column(DateTime, default=func.now())
    filled_at = Column(DateTime)

    # Relationship
    signal = relationship("TradingSignal", backref="orders")

    __table_args__ = (
        Index('idx_order_symbol_status', 'symbol', 'status'),
        Index('idx_order_created', 'created_at'),
    )

class Portfolio(Base):
    """Portfolio tracking"""
    __tablename__ = 'portfolio'

    id = Column(Integer, primary_key=True)
    account_type = Column(String(20), nullable=False)  # simulation, live

    # Current holdings
    symbol = Column(String(20), nullable=False, index=True)
    quantity = Column(Float, nullable=False, default=0.0)
    average_price = Column(Float, nullable=False, default=0.0)

    # Portfolio metrics
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    total_value = Column(Float, default=0.0)

    # Risk metrics
    position_size_pct = Column(Float, default=0.0)  # % of total portfolio
    risk_exposure = Column(Float, default=0.0)

    # Last update info
    last_price = Column(Float)
    last_updated = Column(DateTime, default=func.now())

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_portfolio_account_symbol', 'account_type', 'symbol'),
    )

# ==================== System Monitoring Tables ====================

class SystemLog(Base):
    """System operation logs"""
    __tablename__ = 'system_logs'

    id = Column(Integer, primary_key=True)
    log_level = Column(String(10), nullable=False, index=True)
    module = Column(String(50), nullable=False, index=True)
    function = Column(String(50))
    message = Column(Text, nullable=False)

    # Context information
    symbol = Column(String(20))
    timeframe = Column(String(10))
    model_id = Column(String(50))

    # Additional data
    extra_data = Column(JSONEncodedDict)

    # Stack trace for errors
    stack_trace = Column(Text)

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index('idx_log_level_created', 'log_level', 'created_at'),
        Index('idx_log_module_created', 'module', 'created_at'),
    )

class PerformanceMetric(Base):
    """System and strategy performance metrics"""
    __tablename__ = 'performance_metrics'

    id = Column(Integer, primary_key=True)
    metric_type = Column(String(50), nullable=False, index=True)  # system, strategy, model
    symbol = Column(String(20))
    timeframe = Column(String(10))
    model_id = Column(String(50))

    # Metric data
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20))

    # Time period
    period_start = Column(DateTime)
    period_end = Column(DateTime)

    # Additional context
    metadata = Column(JSONEncodedDict)

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index('idx_metric_type_name_created', 'metric_type', 'metric_name', 'created_at'),
    )

# ==================== Database Configuration ====================

def get_database_engine(database_url: str = None):
    """Get database engine with appropriate configuration"""
    if database_url is None:
        database_url = config.database.current_url

    if database_url.startswith('postgresql'):
        # PostgreSQL configuration
        engine = create_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
            echo=False
        )
    else:
        # SQLite configuration
        engine = create_engine(
            database_url,
            connect_args={'check_same_thread': False},
            echo=False
        )

    return engine

def create_all_tables(database_url: str = None):
    """Create all database tables"""
    try:
        engine = get_database_engine(database_url)
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        return False

def get_session_maker(database_url: str = None):
    """Get SQLAlchemy session maker"""
    engine = get_database_engine(database_url)
    return sessionmaker(bind=engine)

def drop_all_tables(database_url: str = None, confirm: bool = False):
    """Drop all database tables (use with caution!)"""
    if not confirm:
        raise ValueError("Must set confirm=True to drop all tables")

    try:
        engine = get_database_engine(database_url)
        Base.metadata.drop_all(engine)
        logger.warning("All database tables dropped!")
        return True
    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}")
        return False

# ==================== Database Utilities ====================

def init_default_data(session):
    """Initialize database with default data"""
    try:
        # Add default trading pairs
        default_pairs = [
            ('BTCUSDT', 'BTC', 'USDT'),
            ('ETHUSDT', 'ETH', 'USDT'),
            ('ADAUSDT', 'ADA', 'USDT'),
            ('BNBUSDT', 'BNB', 'USDT'),
            ('SOLUSDT', 'SOL', 'USDT'),
        ]

        for symbol, base, quote in default_pairs:
            existing = session.query(TradingPair).filter_by(symbol=symbol).first()
            if not existing:
                pair = TradingPair(
                    symbol=symbol,
                    base_currency=base,
                    quote_currency=quote,
                    is_active=True
                )
                session.add(pair)

        session.commit()
        logger.info("Default data initialized")

    except Exception as e:
        logger.error(f"Failed to initialize default data: {e}")
        session.rollback()

def backup_database(backup_path: str, database_url: str = None):
    """Create database backup"""
    try:
        # This is a simplified backup - in production, use proper database backup tools
        engine = get_database_engine(database_url)

        # For PostgreSQL, you'd use pg_dump
        # For SQLite, you can copy the file
        if database_url and database_url.startswith('sqlite'):
            import shutil
            db_file = database_url.replace('sqlite:///', '')
            shutil.copy2(db_file, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return True
        else:
            logger.warning("Database backup not implemented for PostgreSQL")
            return False

    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        return False

# Usage example
if __name__ == "__main__":
    # Create tables
    success = create_all_tables()
    print(f"Tables created: {success}")

    # Get session
    Session = get_session_maker()
    session = Session()

    # Initialize default data
    init_default_data(session)

    # Test query
    pairs = session.query(TradingPair).all()
    print(f"Trading pairs: {[p.symbol for p in pairs]}")

    session.close()
    print("Database configuration test completed!")
