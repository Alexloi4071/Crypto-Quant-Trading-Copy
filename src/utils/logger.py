"""
Logging System Module
Centralized logging configuration for the crypto quant trading system
Supports multiple log levels, file rotation, and structured logging
"""

import logging
import logging.handlers
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
import traceback

class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }

    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            record.levelname = colored_levelname

        return super().format(record)

class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for production logs"""

    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                log_data[key] = value

        return json.dumps(log_data, ensure_ascii=False)

class TradingSystemLogger:
    """Main logger configuration for the trading system"""

    def __init__(self, name: str, log_dir: str = "logs", level: str = "INFO"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.level = getattr(logging, level.upper(), logging.INFO)

        # Create log directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "trading").mkdir(exist_ok=True)
        (self.log_dir / "optimization").mkdir(exist_ok=True)
        (self.log_dir / "system").mkdir(exist_ok=True)

        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logger with multiple handlers"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)

        # Clear existing handlers
        logger.handlers.clear()

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler - general logs
        general_log_file = self.log_dir / f"system/{self.name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            general_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Structured JSON handler for production
        if os.getenv('ENVIRONMENT', 'dev') == 'prod':
            json_log_file = self.log_dir / f"system/{self.name}.json"
            json_handler = logging.handlers.RotatingFileHandler(
                json_log_file,
                maxBytes=50*1024*1024,  # 50MB
                backupCount=10,
                encoding='utf-8'
            )
            json_handler.setLevel(logging.INFO)
            json_handler.setFormatter(StructuredFormatter())
            logger.addHandler(json_handler)

        # Error handler - separate file for errors
        error_log_file = self.log_dir / f"system/{self.name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)

        return logger

    def get_logger(self) -> logging.Logger:
        """Get the configured logger"""
        return self.logger

class TradingLogger:
    """Specialized logger for trading operations"""

    def __init__(self, symbol: str, timeframe: str, version: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.version = version
        self.log_dir = Path("logs/trading")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_trading_logger()

    def _setup_trading_logger(self) -> logging.Logger:
        """Setup specialized trading logger"""
        logger_name = f"trading.{self.symbol}.{self.timeframe}.{self.version}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        logger.handlers.clear()

        # Trading log file
        log_file = self.log_dir / f"{self.symbol}_{self.timeframe}_{self.version}.log"
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=20*1024*1024,  # 20MB
            backupCount=5,
            encoding='utf-8'
        )

        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def log_signal(self, signal_data: Dict[str, Any]):
        """Log trading signal"""
        self.logger.info(f"SIGNAL | {json.dumps(signal_data, ensure_ascii=False)}")

    def log_order(self, order_data: Dict[str, Any]):
        """Log order execution"""
        self.logger.info(f"ORDER | {json.dumps(order_data, ensure_ascii=False)}")

    def log_position(self, position_data: Dict[str, Any]):
        """Log position update"""
        self.logger.info(f"POSITION | {json.dumps(position_data, ensure_ascii=False)}")

    def log_pnl(self, pnl_data: Dict[str, Any]):
        """Log P&L calculation"""
        self.logger.info(f"PNL | {json.dumps(pnl_data, ensure_ascii=False)}")

class OptimizationLogger:
    """Specialized logger for optimization processes"""

    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.log_dir = Path("logs/optimization")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_optimization_logger()

    def _setup_optimization_logger(self) -> logging.Logger:
        """Setup optimization logger"""
        logger_name = f"optimization.{self.symbol}.{self.timeframe}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        logger.handlers.clear()

        # Optimization log file
        log_file = self.log_dir / f"{self.symbol}_{self.timeframe}_optimization.log"
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3,
            encoding='utf-8'
        )

        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def log_trial(self, trial_number: int, params: Dict[str, Any], score: float):
        """Log optimization trial"""
        trial_data = {
            'trial': trial_number,
            'params': params,
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(f"TRIAL | {json.dumps(trial_data, ensure_ascii=False)}")

    def log_best_params(self, best_params: Dict[str, Any], best_score: float):
        """Log best parameters found"""
        best_data = {
            'best_params': best_params,
            'best_score': best_score,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(f"BEST | {json.dumps(best_data, ensure_ascii=False)}")

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Convenient function to setup a logger

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger
    """
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO')

    trading_logger = TradingSystemLogger(name, level=level)
    return trading_logger.get_logger()

def get_trading_logger(symbol: str, timeframe: str, version: str) -> TradingLogger:
    """Get specialized trading logger"""
    return TradingLogger(symbol, timeframe, version)

def get_optimization_logger(symbol: str, timeframe: str) -> OptimizationLogger:
    """Get specialized optimization logger"""
    return OptimizationLogger(symbol, timeframe)

class LoggingMixin:
    """Mixin class to add logging capabilities to any class"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = setup_logger(self.__class__.__name__)

    def log_debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)

    def log_info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)

    def log_warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)

    def log_error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, extra=kwargs)

    def log_critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra=kwargs)

    def log_exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, extra=kwargs)

# Performance logging decorator

def log_performance(func):
    """Decorator to log function performance"""

    def wrapper(*args, **kwargs):
        logger = setup_logger(f"performance.{func.__module__}.{func.__name__}")
        start_time = datetime.now()

        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            logger.info(f"Function executed successfully in {execution_time:.4f} seconds",
                       extra={'function': func.__name__, 'execution_time': execution_time})

            return result

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            logger.error(f"Function failed after {execution_time:.4f} seconds: {str(e)}",
                        extra={'function': func.__name__, 'execution_time': execution_time, 'error': str(e)})
            raise

    return wrapper

# Usage examples for documentation
if __name__ == "__main__":
    # Basic logger
    logger = setup_logger("test_module")
    logger.info("This is a test message")

    # Trading logger
    trading_logger = get_trading_logger("BTCUSDT", "1h", "V1")
    trading_logger.log_signal({
        'symbol': 'BTCUSDT',
        'action': 'BUY',
        'price': 45000,
        'confidence': 0.85
    })

    # Optimization logger
    opt_logger = get_optimization_logger("BTCUSDT", "1h")
    opt_logger.log_trial(1, {'num_leaves': 31, 'learning_rate': 0.05}, 0.75)

    # Using mixin

    class TestClass(LoggingMixin):

        def test_method(self):
            self.log_info("Testing mixin logger")

    test_obj = TestClass()
    test_obj.test_method()

    # Performance decorator
    @log_performance

    def slow_function():
        import time
        time.sleep(1)
        return "Done"

    slow_function()

    print("Logging system demonstration completed. Check logs/ directory for output files.")
