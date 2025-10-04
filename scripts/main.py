#!/usr/bin/env python3
"""
Main Trading System Script
Entry point for the complete trading system
Supports multiple modes: data download, training, backtesting, and live trading
"""
__file__ = r'D:\crypto-quant-trading-copy\scripts\main.py'

import asyncio
import argparse
import sys
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import config
from src.utils.logger import setup_logger, get_trading_logger
from src.data.collector import BinanceDataCollector as DataManager
# DEPRECATED: 舊管線入口，請改用 src/optimization/main_optimizer.py
from src.optimization.main_optimizer import ModularOptunaOptimizer
from src.models.model_trainer import ModelTrainer
from src.trading.trading_system import TradingSystem, TradingMode, TradingSystemConfig
from src.monitoring.system_monitor import SystemMonitor
from src.monitoring.alerting import AlertingSystem
from src.monitoring.notifications import NotificationService
from src.monitoring.health_checker import HealthChecker
from src.analysis.backtester import BacktestRunner
from src.analysis.performance_analyzer import PerformanceAnalyzer

logger = setup_logger(__name__)


class TradingSystemManager:
    """Main trading system manager"""

    def __init__(self):
        self.trading_system = None
        self.system_monitor = None
        self.alerting_system = None
        self.notification_service = None
        self.health_checker = None

        # System state
        self.running = False
        self.shutdown_requested = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Trading System Manager initialized")


    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True


    async def start_system(self, symbols: List[str], mode: str,
                         initial_capital: float = 10000,
                         enable_monitoring: bool = True) -> bool:
        """Start the complete trading system"""
        try:
            logger.info(f"Starting trading system in {mode} mode")

            # Convert mode string to enum
            trading_mode = TradingMode(mode.lower())

            # Initialize monitoring components
            if enable_monitoring:
                await self._initialize_monitoring()

            # Create trading system configuration
            config_obj = TradingSystemConfig(
                symbols=symbols,
                timeframes=['1h', '4h'],
                trading_mode=trading_mode,
                initial_capital=initial_capital,
                update_interval=60,  # 1 minute
                enable_risk_management=True,
                enable_live_trading=(trading_mode == TradingMode.LIVE),
                max_concurrent_positions=len(symbols)
            )

            # Initialize trading system
            self.trading_system = TradingSystem(config_obj)

            if not await self.trading_system.initialize():
                logger.error("Failed to initialize trading system")
                return False

            # Start trading system
            self.running = True
            trading_task = asyncio.create_task(self.trading_system.start())

            # Start monitoring (if enabled)
            monitoring_tasks = []
            if enable_monitoring and self.system_monitor:
                monitoring_tasks.append(asyncio.create_task(self.system_monitor.start_monitoring()))

            if enable_monitoring and self.health_checker:
                monitoring_tasks.append(asyncio.create_task(self.health_checker.start_health_monitoring()))

            # Wait for shutdown signal or system failure
            try:
                while self.running and not self.shutdown_requested:
                    # Check system status
                    if self.trading_system.status.value in ['error', 'emergency_stop']:
                        logger.critical("Trading system encountered critical error")
                        break

                    # Log periodic status
                    await self._log_system_status()

                    await asyncio.sleep(30)  # Check every 30 seconds

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt")

            # Graceful shutdown
            await self._shutdown_system()

            # Wait for tasks to complete
            if not trading_task.done():
                await asyncio.wait_for(trading_task, timeout=10)

            for task in monitoring_tasks:
                if not task.done():
                    task.cancel()

            logger.info("Trading system stopped successfully")
            return True

        except Exception as e:
            logger.critical(f"Trading system startup failed: {e}")
            return False


    async def _initialize_monitoring(self):
        """Initialize monitoring components"""
        try:
            logger.info("Initializing monitoring components")

            # System monitor
            self.system_monitor = SystemMonitor()

            # Alerting system
            self.alerting_system = AlertingSystem()

            # Notification service
            self.notification_service = NotificationService()

            # Health checker
            self.health_checker = HealthChecker()

            # Connect alerting to notifications
            if self.notification_service:
                self.alerting_system.register_notification_handler(
                    'telegram', self.notification_service.send_alert_notification
                )
                self.alerting_system.register_notification_handler(
                    'email', self.notification_service.send_alert_notification
                )
                self.alerting_system.register_notification_handler(
                    'discord', self.notification_service.send_alert_notification
                )

            logger.info("Monitoring components initialized successfully")

        except Exception as e:
            logger.error(f"Monitoring initialization failed: {e}")
            raise


    async def _log_system_status(self):
        """Log periodic system status"""
        try:
            if not self.trading_system:
                return

            status = self.trading_system.get_system_status()

            # Basic status logging
            logger.info(f"System Status: {status['status']}")
            logger.info(f"Cycle Count: {status['cycle_count']}")

            if 'portfolio' in status:
                portfolio = status['portfolio']
                logger.info(f"Portfolio Value: ${portfolio.get('portfolio_value', 0):,.2f}")
                logger.info(f"Open Positions: {portfolio.get('open_positions_count', 0)}")
                logger.info(f"Total P&L: ${portfolio.get('total_pnl', 0):,.2f}")

            # Check for alerts if monitoring enabled
            if self.system_monitor and self.alerting_system:
                metrics = self.system_monitor.get_current_metrics()
                await self.alerting_system.evaluate_metrics(metrics)

        except Exception as e:
            logger.error(f"Status logging failed: {e}")


    async def _shutdown_system(self):
        """Graceful system shutdown"""
        try:
            logger.info("Initiating graceful shutdown")
            self.running = False

            # Stop trading system
            if self.trading_system:
                await self.trading_system.stop()

            # Stop monitoring components
            if self.system_monitor:
                self.system_monitor.stop_monitoring()

            if self.health_checker:
                self.health_checker.stop_health_monitoring()

            # Send shutdown notification
            if self.notification_service:
                try:
                    await self.notification_service.send_notification(
                        "🛑 System Shutdown",
                        "Trading system has been shut down gracefully",
                        self.notification_service.NotificationChannel.TELEGRAM,
                        self.notification_service.NotificationPriority.NORMAL
                    )
                except Exception as e:
                    logger.debug(f"Shutdown notification failed: {e}")

        except Exception as e:
            logger.error(f"Shutdown failed: {e}")


async def run_data_download(symbols: List[str], timeframes: List[str],
                          days_back: int = 30) -> bool:
    """Download historical data for symbols"""
    try:
        logger.info(f"Starting data download for {len(symbols)} symbols")

        data_manager = DataManager()
        await data_manager.initialize()

        success_count = 0

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    start_date = datetime.now() - timedelta(days=days_back)

                    # Download data
                    df = await data_manager.fetch_ohlcv_data(
                        symbol, timeframe,
                        start_date=start_date,
                        limit=None
                    )

                    if not df.empty:
                        # Store data
                        success = await data_manager.store_ohlcv_data(symbol, timeframe, df)
                        if success:
                            success_count += 1
                            logger.info(f"Downloaded {len(df)} records for {symbol}_{timeframe}")
                        else:
                            logger.error(f"Failed to store data for {symbol}_{timeframe}")
                    else:
                        logger.warning(f"No data retrieved for {symbol}_{timeframe}")

                except Exception as e:
                    logger.error(f"Data download failed for {symbol}_{timeframe}: {e}")
                    continue

                # Rate limiting
                await asyncio.sleep(1)

        total_requested = len(symbols) * len(timeframes)
        logger.info(f"Data download completed: {success_count}/{total_requested} successful")

        return success_count > 0

    except Exception as e:
        logger.error(f"Data download process failed: {e}")
        return False


async def run_feature_generation(symbols: List[str], timeframes: List[str],
                               version: str = "latest") -> bool:
    """Generate features for symbols"""
    try:
        logger.info(f"Starting feature generation for {len(symbols)} symbols")

        data_manager = DataManager()
        feature_engineer = FeatureEngineer()

        await data_manager.initialize()

        success_count = 0

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # Load OHLCV data
                    df = await data_manager.load_ohlcv_data(symbol, timeframe)

                    if df.empty:
                        logger.warning(f"No data available for {symbol}_{timeframe}")
                        continue

                    # Generate features
                    features_df, feature_info = feature_engineer.generate_features(df)

                    if not features_df.empty:
                        # Store features
                        success = await data_manager.store_feature_data(
                            symbol, timeframe, version, features_df, feature_info
                        )

                        if success:
                            success_count += 1
                            logger.info(f"Generated {len(features_df.columns)} features for {symbol}_{timeframe}")
                        else:
                            logger.error(f"Failed to store features for {symbol}_{timeframe}")

                except Exception as e:
                    logger.error(f"Feature generation failed for {symbol}_{timeframe}: {e}")
                    continue

        total_requested = len(symbols) * len(timeframes)
        logger.info(f"Feature generation completed: {success_count}/{total_requested} successful")

        return success_count > 0

    except Exception as e:
        logger.error(f"Feature generation process failed: {e}")
        return False


async def run_model_training(symbols: List[str], timeframes: List[str],
                           version: str = "latest") -> bool:
    """Train models for symbols"""
    try:
        logger.info(f"Starting model training for {len(symbols)} symbols")

        data_manager = DataManager()
        feature_engineer = FeatureEngineer()
        label_generator = LabelGenerator()
        model_trainer = ModelTrainer()

        await data_manager.initialize()

        success_count = 0

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # Load features
                    features_df = await data_manager.load_feature_data(symbol, timeframe, version)

                    if features_df.empty:
                        logger.warning(f"No features available for {symbol}_{timeframe}")
                        continue

                    # Generate labels
                    labels_df = label_generator.generate_labels(features_df)

                    if labels_df.empty:
                        logger.warning(f"No labels generated for {symbol}_{timeframe}")
                        continue

                    # Train model
                    training_result = await model_trainer.train_model(
                        symbol, timeframe, version,
                        features_df, labels_df
                    )

                    if training_result.get('success', False):
                        success_count += 1
                        logger.info(f"Model trained for {symbol}_{timeframe} - "
                                  f"Score: {training_result.get('best_score', 0):.4f}")
                    else:
                        logger.error(f"Model training failed for {symbol}_{timeframe}")

                except Exception as e:
                    logger.error(f"Model training failed for {symbol}_{timeframe}: {e}")
                    continue

        total_requested = len(symbols) * len(timeframes)
        logger.info(f"Model training completed: {success_count}/{total_requested} successful")

        return success_count > 0

    except Exception as e:
        logger.error(f"Model training process failed: {e}")
        return False


async def run_backtest(symbols: List[str], strategy: str = "ml_strategy",
                     start_date: str = None, end_date: str = None,
                     initial_capital: float = 10000) -> bool:
    """Run backtest for strategy"""
    try:
        logger.info(f"Starting backtest for strategy: {strategy}")

        # Parse dates
        if start_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_dt = datetime.now() - timedelta(days=90)  # 3 months back

        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_dt = datetime.now()

        backtest_runner = BacktestRunner()
        performance_analyzer = PerformanceAnalyzer()

        # Run backtest
        results = await backtest_runner.run_backtest(
            strategy=strategy,
            symbols=symbols,
            start_date=start_dt,
            end_date=end_dt,
            initial_capital=initial_capital
        )

        if not results:
            logger.error("Backtest failed to produce results")
            return False

        # Analyze results
        analysis = performance_analyzer.analyze_backtest_results(results)

        # Log key metrics
        logger.info("Backtest Results:")
        logger.info(f"Total Return: {analysis.get('total_return', 0):.2%}")
        logger.info(f"Sharpe Ratio: {analysis.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Maximum Drawdown: {analysis.get('max_drawdown', 0):.2%}")
        logger.info(f"Win Rate: {analysis.get('win_rate', 0):.2%}")
        logger.info(f"Total Trades: {analysis.get('total_trades', 0)}")

        # Generate report
        report_path = f"reports/backtest_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        performance_analyzer.generate_report(results, analysis, report_path)

        logger.info(f"Backtest report saved to: {report_path}")

        return True

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Trading System')

    # Main command
    parser.add_argument('command', choices=[
        'download', 'features', 'train', 'backtest', 'trade'
    ], help='Command to execute')

    # Common arguments
    parser.add_argument('--symbols', nargs='+',
                       default=['BTCUSDT', 'ETHUSDT'],
                       help='Trading symbols')

    parser.add_argument('--timeframes', nargs='+',
                       default=['1h', '4h'],
                       help='Timeframes')

    parser.add_argument('--version', default='latest',
                       help='Version tag for features/models')

    # Trading specific arguments
    parser.add_argument('--mode', choices=['simulation', 'paper', 'live'],
                       default='simulation',
                       help='Trading mode')

    parser.add_argument('--capital', type=float, default=10000,
                       help='Initial capital')

    parser.add_argument('--no-monitoring', action='store_true',
                       help='Disable monitoring components')

    # Data download arguments
    parser.add_argument('--days-back', type=int, default=30,
                       help='Days of historical data to download')

    # Backtest arguments
    parser.add_argument('--strategy', default='ml_strategy',
                       help='Backtest strategy')

    parser.add_argument('--start-date',
                       help='Backtest start date (YYYY-MM-DD)')

    parser.add_argument('--end-date',
                       help='Backtest end date (YYYY-MM-DD)')

    # Logging
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Update logging level
    import logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))


    async def run_command():
        """Run the selected command"""
        try:
            if args.command == 'download':
                success = await run_data_download(
                    args.symbols, args.timeframes, args.days_back
                )

            elif args.command == 'features':
                success = await run_feature_generation(
                    args.symbols, args.timeframes, args.version
                )

            elif args.command == 'train':
                success = await run_model_training(
                    args.symbols, args.timeframes, args.version
                )

            elif args.command == 'backtest':
                success = await run_backtest(
                    args.symbols, args.strategy,
                    args.start_date, args.end_date, args.capital
                )

            elif args.command == 'trade':
                manager = TradingSystemManager()
                success = await manager.start_system(
                    args.symbols, args.mode, args.capital,
                    enable_monitoring=not args.no_monitoring
                )

            else:
                logger.error(f"Unknown command: {args.command}")
                return False

            if success:
                logger.info(f"Command '{args.command}' completed successfully")
                return True
            else:
                logger.error(f"Command '{args.command}' failed")
                return False

        except KeyboardInterrupt:
            logger.info("Operation cancelled by user")
            return False
        except Exception as e:
            logger.critical(f"Command execution failed: {e}")
            return False

    # Run the command
    try:
        success = asyncio.run(run_command())
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Program failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
