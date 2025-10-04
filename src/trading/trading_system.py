"""
Trading System Module
Main orchestrator for the complete trading system
Integrates all components: data, signals, positions, risk management
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import json
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from config.settings import config
from src.utils.logger import setup_logger, get_trading_logger
from src.utils.helpers import timing_decorator
from src.utils.database_manager import DatabaseManager
from src.data.external_apis import ExternalDataManager
from src.trading.exchange_manager import ExchangeManager
from src.trading.signal_generator import SignalGenerator, SignalType, TradingSignal
from src.trading.position_manager import PositionManager, PositionSide
from src.trading.risk_manager import RiskManager, RiskLevel

logger = setup_logger(__name__)

class TradingMode(Enum):
    """Trading system modes"""
    SIMULATION = "simulation"
    PAPER = "paper"
    LIVE = "live"

class SystemStatus(Enum):
    """System status types"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class TradingSystemConfig:
    """Trading system configuration"""
    symbols: List[str]
    timeframes: List[str]
    trading_mode: TradingMode
    initial_capital: float
    update_interval: int  # seconds
    enable_risk_management: bool = True
    enable_live_trading: bool = False
    max_concurrent_positions: int = 10
    model_version: str = "latest"

class TradingSystem:
    """Main trading system orchestrator"""
    
    def __init__(self, system_config: TradingSystemConfig):
        self.config = system_config
        self.status = SystemStatus.STOPPED
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.external_data_manager = ExternalDataManager()
        self.exchange_manager = None
        self.signal_generator = SignalGenerator()
        self.position_manager = None
        self.risk_manager = RiskManager()
        
        # System state
        self.running = False
        self.last_update = None
        self.cycle_count = 0
        self.errors = []
        
        # Performance tracking
        self.system_metrics = {
            'start_time': None,
            'uptime_seconds': 0,
            'total_cycles': 0,
            'avg_cycle_time': 0,
            'total_signals_generated': 0,
            'total_trades_executed': 0,
            'system_errors': 0
        }
        
        logger.info(f"Trading system initialized in {system_config.trading_mode.value} mode")
    
    async def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("Initializing trading system components")
            self.status = SystemStatus.STARTING
            
            # Initialize database
            if not await self.db_manager.connect():
                logger.error("Failed to connect to database")
                return False
            
            # Initialize external data manager
            await self.external_data_manager.initialize()
            
            # Initialize exchange manager
            self.exchange_manager = ExchangeManager()
            exchanges = ['binance']  # Can be configured
            
            if not await self.exchange_manager.initialize_exchanges(exchanges):
                logger.error("Failed to initialize exchange connections")
                return False
            
            # Initialize position manager
            self.position_manager = PositionManager(self.exchange_manager)
            
            # Set initial capital
            self.position_manager.portfolio.initial_capital = self.config.initial_capital
            self.position_manager.portfolio.cash = self.config.initial_capital
            self.position_manager.portfolio.total_value = self.config.initial_capital
            
            # Validate configuration
            if not self._validate_configuration():
                logger.error("System configuration validation failed")
                return False
            
            logger.info("Trading system initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.status = SystemStatus.ERROR
            return False
    
    def _validate_configuration(self) -> bool:
        """Validate system configuration"""
        try:
            # Check symbols
            if not self.config.symbols:
                logger.error("No trading symbols configured")
                return False
            
            # Check timeframes
            if not self.config.timeframes:
                logger.error("No timeframes configured")
                return False
            
            # Check trading mode
            if self.config.trading_mode == TradingMode.LIVE and not self.config.enable_live_trading:
                logger.error("Live trading mode requires enable_live_trading=True")
                return False
            
            # Check capital
            if self.config.initial_capital <= 0:
                logger.error("Initial capital must be positive")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the trading system"""
        try:
            if self.status == SystemStatus.RUNNING:
                logger.warning("Trading system is already running")
                return True
            
            if self.status != SystemStatus.STARTING and not await self.initialize():
                return False
            
            logger.info("Starting trading system main loop")
            self.running = True
            self.status = SystemStatus.RUNNING
            self.system_metrics['start_time'] = datetime.now()
            
            # Start main trading loop
            await self._main_trading_loop()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start trading system: {e}")
            self.status = SystemStatus.ERROR
            return False
    
    async def stop(self) -> bool:
        """Stop the trading system"""
        try:
            logger.info("Stopping trading system")
            self.running = False
            self.status = SystemStatus.STOPPED
            
            # Close all positions (optional)
            if self.position_manager and hasattr(self.position_manager, 'close_all_positions'):
                await self.position_manager.close_all_positions("System shutdown")
            
            # Close connections
            if self.exchange_manager:
                await self.exchange_manager.close_all_connections()
            
            if self.db_manager:
                await self.db_manager.disconnect()
            
            logger.info("Trading system stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping trading system: {e}")
            return False
    
    async def pause(self) -> bool:
        """Pause the trading system"""
        try:
            if self.status != SystemStatus.RUNNING:
                logger.warning("System is not running, cannot pause")
                return False
            
            logger.info("Pausing trading system")
            self.status = SystemStatus.PAUSED
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause system: {e}")
            return False
    
    async def resume(self) -> bool:
        """Resume the trading system"""
        try:
            if self.status != SystemStatus.PAUSED:
                logger.warning("System is not paused, cannot resume")
                return False
            
            logger.info("Resuming trading system")
            self.status = SystemStatus.RUNNING
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume system: {e}")
            return False
    
    async def _main_trading_loop(self):
        """Main trading system loop"""
        try:
            while self.running:
                cycle_start_time = datetime.now()
                
                try:
                    # Skip execution if paused
                    if self.status == SystemStatus.PAUSED:
                        await asyncio.sleep(self.config.update_interval)
                        continue
                    
                    # Execute trading cycle
                    await self._execute_trading_cycle()
                    
                    self.cycle_count += 1
                    self.system_metrics['total_cycles'] += 1
                    self.last_update = datetime.now()
                    
                    # Calculate cycle time
                    cycle_time = (datetime.now() - cycle_start_time).total_seconds()
                    
                    # Update average cycle time
                    if self.system_metrics['avg_cycle_time'] == 0:
                        self.system_metrics['avg_cycle_time'] = cycle_time
                    else:
                        self.system_metrics['avg_cycle_time'] = (
                            self.system_metrics['avg_cycle_time'] * 0.9 + cycle_time * 0.1
                        )
                    
                    # Update uptime
                    if self.system_metrics['start_time']:
                        self.system_metrics['uptime_seconds'] = (
                            datetime.now() - self.system_metrics['start_time']
                        ).total_seconds()
                    
                    logger.debug(f"Trading cycle {self.cycle_count} completed in {cycle_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Trading cycle {self.cycle_count} failed: {e}")
                    self.system_metrics['system_errors'] += 1
                    
                    # Add error to history
                    error_info = {
                        'timestamp': datetime.now(),
                        'cycle': self.cycle_count,
                        'error': str(e),
                        'type': type(e).__name__
                    }
                    self.errors.append(error_info)
                    
                    # Keep only recent errors
                    if len(self.errors) > 100:
                        self.errors = self.errors[-100:]
                
                # Sleep until next cycle
                await asyncio.sleep(self.config.update_interval)
                
        except Exception as e:
            logger.critical(f"Main trading loop crashed: {e}")
            self.status = SystemStatus.ERROR
            raise
    
    async def _execute_trading_cycle(self):
        """Execute one complete trading cycle"""
        try:
            # 1. Update market data
            market_data = await self._update_market_data()
            
            # 2. Generate trading signals
            signals = await self._generate_signals(market_data)
            
            # 3. Assess risk
            risk_assessment = await self._assess_risk()
            
            # 4. Execute trading decisions
            if risk_assessment.get('overall_risk_level') != RiskLevel.CRITICAL:
                trading_results = await self._execute_trading_decisions(signals, risk_assessment)
            else:
                logger.warning("Trading suspended due to critical risk level")
                trading_results = {}
            
            # 5. Update positions
            position_updates = await self._update_positions(market_data)
            
            # 6. Log cycle results
            await self._log_cycle_results(signals, trading_results, position_updates, risk_assessment)
            
        except Exception as e:
            logger.error(f"Trading cycle execution failed: {e}")
            raise
    
    async def _update_market_data(self) -> Dict[str, pd.DataFrame]:
        """Update market data for all symbols and timeframes"""
        try:
            market_data = {}
            
            # Get current prices for all symbols
            current_prices = {}
            
            for symbol in self.config.symbols:
                try:
                    ticker = await self.exchange_manager.fetch_ticker(symbol)
                    current_prices[symbol] = ticker.get('price', 0)
                except Exception as e:
                    logger.warning(f"Failed to fetch ticker for {symbol}: {e}")
                    continue
            
            # Get historical data for each symbol/timeframe combination
            for symbol in self.config.symbols:
                for timeframe in self.config.timeframes:
                    try:
                        # Fetch recent OHLCV data
                        df = await self.exchange_manager.fetch_ohlcv(
                            symbol, timeframe, 
                            since=datetime.now() - timedelta(days=30),
                            limit=1000
                        )
                        
                        if not df.empty:
                            # Add external data if available
                            try:
                                external_data = await self.external_data_manager.get_market_data(symbol)
                                if external_data:
                                    # Merge external data (simplified)
                                    df = df.copy()
                                    # External data integration would go here
                            except Exception as e:
                                logger.debug(f"External data fetch failed for {symbol}: {e}")
                            
                            market_data[f"{symbol}_{timeframe}"] = df
                        
                    except Exception as e:
                        logger.warning(f"Failed to fetch OHLCV for {symbol}_{timeframe}: {e}")
                        continue
            
            # Store current prices for position updates
            self.current_prices = current_prices
            
            logger.debug(f"Updated market data for {len(market_data)} symbol/timeframe pairs")
            return market_data
            
        except Exception as e:
            logger.error(f"Market data update failed: {e}")
            return {}
    
    async def _generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, TradingSignal]:
        """Generate trading signals for all symbols/timeframes"""
        try:
            signals = {}
            
            # Generate signals for each timeframe (prioritize longer timeframes)
            for timeframe in sorted(self.config.timeframes, reverse=True):
                timeframe_data = {
                    symbol: df for key, df in market_data.items() 
                    if key.endswith(f"_{timeframe}")
                    for symbol in self.config.symbols
                    if key == f"{symbol}_{timeframe}"
                }
                
                if timeframe_data:
                    timeframe_signals = await self.signal_generator.generate_batch_signals(
                        list(timeframe_data.keys()),
                        timeframe,
                        timeframe_data,
                        self.config.model_version
                    )
                    
                    # Merge signals (longer timeframes override shorter ones)
                    for symbol, signal in timeframe_signals.items():
                        signals[symbol] = signal
            
            self.system_metrics['total_signals_generated'] += len(signals)
            
            logger.debug(f"Generated {len(signals)} trading signals")
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return {}
    
    async def _assess_risk(self) -> Dict[str, Any]:
        """Assess current risk levels"""
        try:
            if not self.config.enable_risk_management:
                return {'overall_risk_level': RiskLevel.LOW}
            
            # Get current portfolio state
            portfolio_summary = self.position_manager.get_portfolio_summary()
            
            # Calculate daily return if possible
            daily_return = None
            if len(self.risk_manager.portfolio_values) > 1:
                current_value = portfolio_summary.get('portfolio_value', self.config.initial_capital)
                previous_value = self.risk_manager.portfolio_values[-1]
                daily_return = (current_value - previous_value) / previous_value
            
            # Assess risk
            risk_assessment = self.risk_manager.assess_portfolio_risk(
                portfolio_summary.get('portfolio_value', self.config.initial_capital),
                portfolio_summary.get('open_positions', {}),
                daily_return
            )
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {'overall_risk_level': RiskLevel.HIGH, 'error': str(e)}
    
    async def _execute_trading_decisions(self, signals: Dict[str, TradingSignal],
                                       risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading decisions based on signals and risk assessment"""
        try:
            trading_results = {
                'positions_opened': 0,
                'positions_closed': 0,
                'orders_placed': [],
                'errors': []
            }
            
            # Check if trading is allowed
            trading_allowed, reason = self.risk_manager.check_trading_allowed()
            
            if not trading_allowed:
                logger.warning(f"Trading not allowed: {reason}")
                return trading_results
            
            # Process signals
            for symbol, signal in signals.items():
                try:
                    await self._process_signal(symbol, signal, trading_results)
                except Exception as e:
                    error_msg = f"Failed to process signal for {symbol}: {e}"
                    logger.error(error_msg)
                    trading_results['errors'].append(error_msg)
            
            return trading_results
            
        except Exception as e:
            logger.error(f"Trading decision execution failed: {e}")
            return {'error': str(e)}
    
    async def _process_signal(self, symbol: str, signal: TradingSignal, 
                            trading_results: Dict[str, Any]):
        """Process an individual trading signal"""
        try:
            # Check if we already have a position
            current_position = self.position_manager.get_position_details(symbol)
            
            # Decision logic based on signal type
            if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
                if not current_position:
                    # Open long position
                    result = await self.position_manager.open_position(
                        symbol=symbol,
                        side=PositionSide.LONG,
                        signal_confidence=signal.confidence,
                        current_price=signal.price,
                        signal_source=signal.source.value
                    )
                    
                    if result['success']:
                        trading_results['positions_opened'] += 1
                        trading_results['orders_placed'].append(result)
                        self.system_metrics['total_trades_executed'] += 1
                        
                        logger.info(f"Opened LONG position for {symbol} based on {signal.signal_type.name} signal")
                    else:
                        trading_results['errors'].append(f"Failed to open position for {symbol}: {result.get('error')}")
            
            elif signal.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
                if current_position and current_position.get('side') == 'long':
                    # Close long position
                    result = await self.position_manager.close_position(
                        symbol, f"{signal.signal_type.name} signal"
                    )
                    
                    if result['success']:
                        trading_results['positions_closed'] += 1
                        trading_results['orders_placed'].append(result)
                        
                        logger.info(f"Closed LONG position for {symbol} based on {signal.signal_type.name} signal")
                    else:
                        trading_results['errors'].append(f"Failed to close position for {symbol}: {result.get('error')}")
                
                elif not current_position and self.config.trading_mode != TradingMode.SIMULATION:
                    # Open short position (if supported)
                    result = await self.position_manager.open_position(
                        symbol=symbol,
                        side=PositionSide.SHORT,
                        signal_confidence=signal.confidence,
                        current_price=signal.price,
                        signal_source=signal.source.value
                    )
                    
                    if result['success']:
                        trading_results['positions_opened'] += 1
                        trading_results['orders_placed'].append(result)
                        self.system_metrics['total_trades_executed'] += 1
                        
                        logger.info(f"Opened SHORT position for {symbol} based on {signal.signal_type.name} signal")
            
            # NEUTRAL signals don't trigger trades but can be logged
            
        except Exception as e:
            logger.error(f"Signal processing failed for {symbol}: {e}")
            raise
    
    async def _update_positions(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Update all positions with current market data"""
        try:
            if not hasattr(self, 'current_prices'):
                return {}
            
            # Update positions with current prices
            update_result = await self.position_manager.update_positions(self.current_prices)
            
            return update_result
            
        except Exception as e:
            logger.error(f"Position update failed: {e}")
            return {'error': str(e)}
    
    async def _log_cycle_results(self, signals: Dict[str, TradingSignal],
                               trading_results: Dict[str, Any],
                               position_updates: Dict[str, Any],
                               risk_assessment: Dict[str, Any]):
        """Log trading cycle results"""
        try:
            cycle_summary = {
                'timestamp': datetime.now().isoformat(),
                'cycle': self.cycle_count,
                'signals_generated': len(signals),
                'positions_opened': trading_results.get('positions_opened', 0),
                'positions_closed': trading_results.get('positions_closed', 0),
                'portfolio_value': position_updates.get('portfolio_value', 0),
                'total_pnl': position_updates.get('total_pnl', 0),
                'risk_level': risk_assessment.get('overall_risk_level', RiskLevel.LOW).value if hasattr(risk_assessment.get('overall_risk_level', RiskLevel.LOW), 'value') else str(risk_assessment.get('overall_risk_level', RiskLevel.LOW)),
                'errors': len(trading_results.get('errors', []))
            }
            
            # Log to trading logger
            system_logger = get_trading_logger("SYSTEM", "main", self.config.model_version)
            system_logger.log_cycle_results(cycle_summary)
            
            # Store in database if available
            if self.db_manager:
                try:
                    await self.db_manager.store_trading_cycle_data(cycle_summary)
                except Exception as e:
                    logger.debug(f"Failed to store cycle data in database: {e}")
            
        except Exception as e:
            logger.error(f"Cycle results logging failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            portfolio_summary = {}
            if self.position_manager:
                portfolio_summary = self.position_manager.get_portfolio_summary()
            
            risk_summary = {}
            if self.risk_manager:
                risk_summary = self.risk_manager.get_risk_summary()
            
            return {
                'status': self.status.value,
                'running': self.running,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'cycle_count': self.cycle_count,
                'configuration': {
                    'symbols': self.config.symbols,
                    'timeframes': self.config.timeframes,
                    'trading_mode': self.config.trading_mode.value,
                    'initial_capital': self.config.initial_capital
                },
                'portfolio': portfolio_summary,
                'risk': risk_summary,
                'metrics': self.system_metrics,
                'recent_errors': self.errors[-10:] if self.errors else []
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    async def emergency_stop(self, reason: str = "Manual emergency stop"):
        """Emergency stop all trading operations"""
        try:
            logger.critical(f"EMERGENCY STOP: {reason}")
            
            self.status = SystemStatus.EMERGENCY_STOP
            self.running = False
            
            # Activate risk manager emergency stop
            if self.risk_manager:
                self.risk_manager.emergency_stop = True
                self.risk_manager.trading_enabled = False
            
            # Close all positions
            if self.position_manager:
                await self.position_manager.close_all_positions(f"Emergency stop: {reason}")
            
            logger.critical("Emergency stop completed")
            
        except Exception as e:
            logger.critical(f"Emergency stop failed: {e}")

# Convenience functions
async def create_trading_system(symbols: List[str], 
                              timeframes: List[str] = None,
                              trading_mode: TradingMode = TradingMode.SIMULATION,
                              initial_capital: float = 10000) -> TradingSystem:
    """Create and initialize trading system"""
    
    if timeframes is None:
        timeframes = ['1h', '4h']
    
    config = TradingSystemConfig(
        symbols=symbols,
        timeframes=timeframes,
        trading_mode=trading_mode,
        initial_capital=initial_capital,
        update_interval=60,  # 1 minute
        enable_risk_management=True,
        enable_live_trading=(trading_mode == TradingMode.LIVE)
    )
    
    system = TradingSystem(config)
    
    if await system.initialize():
        return system
    else:
        raise RuntimeError("Failed to initialize trading system")

# Usage example
if __name__ == "__main__":
    async def test_trading_system():
        # Create trading system
        system = await create_trading_system(
            symbols=['BTCUSDT', 'ETHUSDT'],
            timeframes=['1h'],
            trading_mode=TradingMode.SIMULATION,
            initial_capital=10000
        )
        
        print("Trading system created successfully")
        
        # Get initial status
        status = system.get_system_status()
        print(f"System status: {status['status']}")
        print(f"Portfolio value: ${status['portfolio'].get('portfolio_value', 0):,.2f}")
        
        # Run for a few cycles (in a real scenario, this would run continuously)
        print("Starting trading system...")
        
        # Start system in background
        system_task = asyncio.create_task(system.start())
        
        # Let it run for 30 seconds
        await asyncio.sleep(30)
        
        # Stop system
        await system.stop()
        
        # Wait for system task to complete
        try:
            await asyncio.wait_for(system_task, timeout=5.0)
        except asyncio.TimeoutError:
            pass
        
        # Get final status
        final_status = system.get_system_status()
        print(f"Final cycle count: {final_status['cycle_count']}")
        print(f"Total trades: {final_status['metrics']['total_trades_executed']}")
        print(f"Signals generated: {final_status['metrics']['total_signals_generated']}")
        
        print("Trading system test completed!")
    
    # Run test
    asyncio.run(test_trading_system())