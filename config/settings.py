"""
Configuration Management Module
Handles all configuration settings for the crypto quant trading system
"""

import os
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables

load_dotenv()

@dataclass

class DatabaseConfig:
    """Database configuration"""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '5432'))
    name: str = os.getenv('DB_NAME', 'crypto_trading')
    user: str = os.getenv('DB_USER', 'postgres')
    password: str = os.getenv('DB_PASSWORD', '')
    url: str = f"postgresql://{user}:{password}@{host}:{port}/{name}"

    # SQLite for development
    sqlite_url: str = "sqlite:///data/crypto_trading.db"

    # Use SQLite in development, PostgreSQL in production
    current_url: str = sqlite_url if os.getenv('ENVIRONMENT', 'dev') == 'dev' else url

@dataclass

class TradingConfig:
    """Trading configuration"""
    # Exchange settings
    exchange: str = 'binance'
    testnet: bool = os.getenv('TESTNET', 'True').lower() == 'true'

    # API credentials
    api_key: str = os.getenv('BINANCE_API_KEY', '')
    api_secret: str = os.getenv('BINANCE_API_SECRET', '')

    # Trading parameters
    default_position_size: float = 0.02  # 2% of portfolio
    max_position_size: float = 0.1       # 10% max position
    max_total_positions: int = 10

    # Risk management
    stop_loss_pct: float = 0.03          # 3% stop loss
    take_profit_pct: float = 0.06        # 6% take profit
    max_drawdown_pct: float = 0.15       # 15% max drawdown
    daily_loss_limit: float = 0.05       # 5% daily loss limit

    # Signal thresholds
    min_confidence: float = 0.7          # Minimum signal confidence
    signal_cooldown_minutes: int = 60    # Cooldown between signals

@dataclass

class OptimizationConfig:
    """Optimization configuration"""
    n_trials: int = 100
    n_jobs: int = -1
    cv_splits: int = 5

    # Feature selection
    max_features: int = 30
    correlation_threshold: float = 0.95
    pca_variance_threshold: float = 0.95

    # Label thresholds ranges
    pos_threshold_range: tuple = (0.002, 0.02)
    neg_threshold_range: tuple = (-0.02, -0.002)

@dataclass

class MonitoringConfig:
    """Monitoring and alerting configuration"""
    # Telegram
    telegram_token: str = os.getenv('TELEGRAM_TOKEN', '')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '')

    # Alert thresholds
    performance_check_interval: int = 3600  # 1 hour in seconds
    alert_drawdown_threshold: float = 0.1   # 10%
    alert_loss_streak: int = 5              # 5 consecutive losses

    # Logging
    log_level: str = 'INFO'
    log_retention_days: int = 30

class Config:
    """Main configuration class"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results"
        self.logs_dir = self.project_root / "logs"

        # Create directories if they don't exist
        self._create_directories()

        # Load configurations
        self.database = DatabaseConfig()
        self.trading = TradingConfig()
        self.optimization = OptimizationConfig()
        self.monitoring = MonitoringConfig()

        # Load trading pairs and indicators from YAML files
        self.trading_pairs = self._load_trading_pairs()
        self.indicators = self._load_indicators()
        self.timeframes = ['5m', '15m', '1h', '4h', '1d']

    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "backups",
            self.results_dir / "optimization",
            self.results_dir / "backtesting",
            self.results_dir / "models",
            self.results_dir / "reports",
            self.logs_dir / "trading",
            self.logs_dir / "optimization",
            self.logs_dir / "system"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_trading_pairs(self) -> List[str]:
        """Load trading pairs from YAML config"""
        config_path = self.project_root / "config" / "trading_pairs.yaml"

        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data.get('pairs', ['BTCUSDT', 'ETHUSDT'])
        else:
            # Default trading pairs
            default_pairs = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'SOLUSDT']
            self._save_default_trading_pairs(config_path, default_pairs)
            return default_pairs

    def _load_indicators(self) -> Dict[str, Any]:
        """Load indicator configurations from YAML"""
        config_path = self.project_root / "config" / "indicators.yaml"

        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            default_indicators = self._get_default_indicators()
            self._save_default_indicators(config_path, default_indicators)
            return default_indicators

    def _get_default_indicators(self) -> Dict[str, Any]:
        """Get default indicator configurations"""
        return {
            'trend': {
                'MA': {'periods': [5, 10, 20, 50, 100, 200]},
                'EMA': {'periods': [5, 10, 20, 50, 100, 200]},
                'MACD': {'fast': [12], 'slow': [26], 'signal': [9]},
                'ADX': {'periods': [14, 20, 25]},
                'SAR': {'acceleration': [0.02], 'maximum': [0.2]},
                'Donchian': {'periods': [10, 20, 55]}
            },
            'momentum': {
                'RSI': {'periods': [14, 21, 30]},
                'StochRSI': {'periods': [14], 'k': [3], 'd': [3]},
                'KDJ': {'periods': [9, 3, 3]},
                'CCI': {'periods': [14, 20]},
                'WR': {'periods': [14, 21]},
                'ROC': {'periods': [10, 20, 30]},
                'MOM': {'periods': [10, 20, 30]}
            },
            'volatility': {
                'BOLL': {'periods': [20], 'std': [2]},
                'ATR': {'periods': [14, 20]},
                'Keltner': {'periods': [20], 'multiplier': [2]}
            },
            'volume': {
                'OBV': {},
                'PVT': {},
                'MFI': {'periods': [14]},
                'VWAP': {'periods': [20]},
                'Volume_MA': {'periods': [10, 20, 50]}
            }
        }

    def _save_default_trading_pairs(self, path: Path, pairs: List[str]):
        """Save default trading pairs to YAML"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump({'pairs': pairs}, f, default_flow_style=False)

    def _save_default_indicators(self, path: Path, indicators: Dict[str, Any]):
        """Save default indicators to YAML"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(indicators, f, default_flow_style=False)

    def get_symbol_data_path(self, symbol: str) -> Path:
        """Get data directory path for a specific symbol"""
        return self.data_dir / "raw" / symbol

    def get_processed_path(self, symbol: str, timeframe: str, version: str) -> Path:
        """Get processed data path for symbol/timeframe/version"""
        return self.data_dir / "processed" / f"{symbol}_{timeframe}_{version}"

    def get_model_path(self, symbol: str, timeframe: str, version: str) -> Path:
        """Get model save path"""
        return self.results_dir / "models" / f"{symbol}_{timeframe}_{version}"

    def validate_config(self) -> bool:
        """Validate configuration settings"""
        errors = []

        # Check API credentials for live trading
        if not self.trading.testnet:
            if not self.trading.api_key or not self.trading.api_secret:
                errors.append("Live trading requires API credentials")

        # Check Telegram configuration
        if not self.monitoring.telegram_token or not self.monitoring.telegram_chat_id:
            errors.append("Telegram configuration required for monitoring")

        # Check database connection (implement database connection test)
        # This would be implemented in the database manager

        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False

        return True

# Global configuration instance
config = Config()

# Utility functions for common configuration access

def get_database_url() -> str:
    """Get database connection URL"""
    return config.database.current_url

def get_trading_pairs() -> List[str]:
    """Get list of trading pairs"""
    return config.trading_pairs

def get_timeframes() -> List[str]:
    """Get list of timeframes"""
    return config.timeframes

def is_testnet() -> bool:
    """Check if running in testnet mode"""
    return config.trading.testnet

def get_telegram_config() -> tuple:
    """Get Telegram bot configuration"""
    return config.monitoring.telegram_token, config.monitoring.telegram_chat_id


def setup_config():
    """設置配置 - 初始化配置參數"""
    global config
    try:
        # 確保所有必要的配置項都存在
        if not hasattr(config, 'EXCHANGE'):
            config.EXCHANGE = "binance"
        if not hasattr(config, 'ENVIRONMENT'):
            config.ENVIRONMENT = "development"
        
        print("✅ 配置初始化完成")
        return True
    except Exception as e:
        print(f"❌ 配置初始化失敗: {e}")
        return False
