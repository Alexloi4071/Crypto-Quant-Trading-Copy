"""
Test Trading System Module
Comprehensive tests for trading system functionality
Tests signal generation, position management, risk management, and system integration
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.trading_system import TradingSystem, TradingMode, TradingSystemConfig
from src.trading.signal_generator import SignalGenerator, SignalType, TradingSignal, SignalSource
from src.trading.position_manager import PositionManager, PositionSide
from src.trading.risk_manager import RiskManager, RiskLevel
from src.trading.exchange_manager import ExchangeManager


class TestTradingSystem:
    """Test suite for TradingSystem class"""

    @pytest.fixture
    def trading_config(self):
        """Create trading system configuration for testing"""
        return TradingSystemConfig(
            symbols=['BTCUSDT', 'ETHUSDT'],
            timeframes=['1h'],
            trading_mode=TradingMode.SIMULATION,
            initial_capital=10000,
            update_interval=60,
            enable_risk_management=True,
            enable_live_trading=False,
            max_concurrent_positions=5,
            model_version="test"
        )

    @pytest.fixture
    async def trading_system(self, trading_config):
        """Create TradingSystem instance for testing"""
        system = TradingSystem(trading_config)
        yield system
        # Cleanup
        try:
            await system.stop()
        except:
            pass

    @pytest.mark.asyncio
    async def test_system_initialization(self, trading_system):
        """Test trading system initialization"""
        # Mock the dependencies
        with patch.object(trading_system, 'db_manager') as mock_db:
            mock_db.connect = AsyncMock(return_value=True)

            with patch.object(trading_system, 'exchange_manager') as mock_exchange:
                mock_exchange.initialize_exchanges = AsyncMock(return_value=True)

                result = await trading_system.initialize()

                assert result is True
                assert trading_system.position_manager is not None
                assert trading_system.risk_manager is not None

    @pytest.mark.asyncio
    async def test_system_start_stop(self, trading_system):
        """Test trading system start and stop"""
        # Mock initialization
        with patch.object(trading_system, 'initialize', return_value=True):
            with patch.object(trading_system, '_main_trading_loop') as mock_loop:
                mock_loop.return_value = None

                # Test start
                start_task = asyncio.create_task(trading_system.start())

                # Give it a moment to start
                await asyncio.sleep(0.1)

                assert trading_system.running is True
                assert trading_system.status.value == 'running'

                # Test stop
                await trading_system.stop()

                assert trading_system.running is False
                assert trading_system.status.value == 'stopped'

                # Wait for start task to complete
                try:
                    await asyncio.wait_for(start_task, timeout=1.0)
                except asyncio.TimeoutError:
                    start_task.cancel()

    @pytest.mark.asyncio
    async def test_trading_cycle_execution(self, trading_system):
        """Test trading cycle execution"""
        # Mock all dependencies
        with patch.object(trading_system, '_update_market_data') as mock_market:
            mock_market.return_value = {'BTCUSDT_1h': pd.DataFrame()}

            with patch.object(trading_system, '_generate_signals') as mock_signals:
                mock_signals.return_value = {}

                with patch.object(trading_system, '_assess_risk') as mock_risk:
                    mock_risk.return_value = {'overall_risk_level': RiskLevel.LOW}

                    with patch.object(trading_system, '_execute_trading_decisions') as mock_trading:
                        mock_trading.return_value = {'positions_opened': 0}

                        with patch.object(trading_system, '_update_positions') as mock_positions:
                            mock_positions.return_value = {}

                            with patch.object(trading_system, '_log_cycle_results') as mock_log:
                                mock_log.return_value = None

                                # Execute one trading cycle
                                await trading_system._execute_trading_cycle()

                                # Verify all steps were called
                                mock_market.assert_called_once()
                                mock_signals.assert_called_once()
                                mock_risk.assert_called_once()
                                mock_trading.assert_called_once()
                                mock_positions.assert_called_once()
                                mock_log.assert_called_once()


    def test_system_status(self, trading_system):
        """Test system status reporting"""
        status = trading_system.get_system_status()

        assert isinstance(status, dict)
        assert 'status' in status
        assert 'configuration' in status
        assert 'metrics' in status

        # Check configuration details
        config = status['configuration']
        assert config['symbols'] == ['BTCUSDT', 'ETHUSDT']
        assert config['trading_mode'] == 'simulation'
        assert config['initial_capital'] == 10000

    @pytest.mark.asyncio
    async def test_emergency_stop(self, trading_system):
        """Test emergency stop functionality"""
        # Mock components
        trading_system.risk_manager = Mock()
        trading_system.position_manager = Mock()
        trading_system.position_manager.close_all_positions = AsyncMock()

        await trading_system.emergency_stop("Test emergency")

        assert trading_system.status.value == 'emergency_stop'
        assert trading_system.running is False
        assert trading_system.risk_manager.emergency_stop is True
        trading_system.position_manager.close_all_positions.assert_called_once()


class TestSignalGenerator:
    """Test suite for SignalGenerator class"""

    @pytest.fixture
    def signal_generator(self):
        """Create SignalGenerator instance for testing"""
        return SignalGenerator()

    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for signal generation"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')

        np.random.seed(42)
        base_price = 45000
        prices = base_price + np.cumsum(np.random.randn(100) * 50)

        return pd.DataFrame({
            'open': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices + np.random.randn(100) * 20,
            'volume': np.random.randint(1000, 5000, 100)
        }, index=dates)

    @pytest.mark.asyncio
    async def test_generate_signal(self, signal_generator, sample_market_data):
        """Test signal generation for a single symbol"""
        signal = await signal_generator.generate_signal(
            symbol='BTCUSDT',
            timeframe='1h',
            market_data=sample_market_data,
            model_version='test'
        )

        assert isinstance(signal, TradingSignal)
        assert signal.symbol == 'BTCUSDT'
        assert signal.timeframe == '1h'
        assert signal.signal_type in SignalType
        assert 0 <= signal.confidence <= 1
        assert signal.price > 0
        assert isinstance(signal.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_batch_signal_generation(self, signal_generator, sample_market_data):
        """Test batch signal generation"""
        symbols = ['BTCUSDT', 'ETHUSDT']
        market_data = {symbol: sample_market_data for symbol in symbols}

        signals = await signal_generator.generate_batch_signals(
            symbols=symbols,
            timeframe='1h',
            market_data=market_data,
            model_version='test'
        )

        assert isinstance(signals, dict)
        assert len(signals) <= len(symbols)  # May filter out some signals

        for symbol, signal in signals.items():
            assert isinstance(signal, TradingSignal)
            assert signal.symbol == symbol


    def test_signal_aggregation(self, signal_generator):
        """Test signal aggregation from multiple sources"""
        # Create mock signals from different sources
        ml_signal = TradingSignal(
            symbol='BTCUSDT',
            timeframe='1h',
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=45000,
            source=SignalSource.ML_MODEL,
            timestamp=datetime.now()
        )

        ta_signal = TradingSignal(
            symbol='BTCUSDT',
            timeframe='1h',
            signal_type=SignalType.STRONG_BUY,
            confidence=0.7,
            price=45000,
            source=SignalSource.TECHNICAL_ANALYSIS,
            timestamp=datetime.now()
        )

        wyckoff_signal = TradingSignal(
            symbol='BTCUSDT',
            timeframe='1h',
            signal_type=SignalType.NEUTRAL,
            confidence=0.6,
            price=45000,
            source=SignalSource.WYCKOFF_ANALYSIS,
            timestamp=datetime.now()
        )

        signals = [ml_signal, ta_signal, wyckoff_signal]

        aggregated_signal = signal_generator._aggregate_signals(signals)

        assert isinstance(aggregated_signal, TradingSignal)
        assert aggregated_signal.symbol == 'BTCUSDT'
        # Should combine the signals somehow (implementation specific)
        assert 0 <= aggregated_signal.confidence <= 1


    def test_signal_validation(self, signal_generator):
        """Test signal validation"""
        # Valid signal
        valid_signal = TradingSignal(
            symbol='BTCUSDT',
            timeframe='1h',
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=45000,
            source=SignalSource.ML_MODEL,
            timestamp=datetime.now()
        )

        assert signal_generator._validate_signal(valid_signal) is True

        # Invalid signal (negative price)
        invalid_signal = TradingSignal(
            symbol='BTCUSDT',
            timeframe='1h',
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=-1000,  # Invalid negative price
            source=SignalSource.ML_MODEL,
            timestamp=datetime.now()
        )

        assert signal_generator._validate_signal(invalid_signal) is False


class TestPositionManager:
    """Test suite for PositionManager class"""

    @pytest.fixture
    def position_manager(self):
        """Create PositionManager instance for testing"""
        mock_exchange_manager = Mock()
        manager = PositionManager(mock_exchange_manager)
        manager.portfolio.initial_capital = 10000
        manager.portfolio.cash = 10000
        manager.portfolio.total_value = 10000
        return manager

    @pytest.mark.asyncio
    async def test_open_position(self, position_manager):
        """Test opening a position"""
        with patch.object(position_manager, '_calculate_position_size', return_value=0.1) as mock_size:
            with patch.object(position_manager, '_execute_order', return_value={'success': True, 'order_id': '12345'}) as mock_order:

                result = await position_manager.open_position(
                    symbol='BTCUSDT',
                    side=PositionSide.LONG,
                    signal_confidence=0.8,
                    current_price=45000
                )

                assert result['success'] is True
                assert 'order_id' in result
                assert 'BTCUSDT' in position_manager.positions

                mock_size.assert_called_once()
                mock_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_position(self, position_manager):
        """Test closing a position"""
        # First, add a position manually
        position_manager.positions['BTCUSDT'] = {
            'side': 'long',
            'size': 0.1,
            'entry_price': 45000,
            'current_price': 46000,
            'pnl': 100,
            'timestamp': datetime.now()
        }

        with patch.object(position_manager, '_execute_order', return_value={'success': True, 'order_id': '12346'}) as mock_order:

            result = await position_manager.close_position('BTCUSDT', 'Manual close')

            assert result['success'] is True
            assert 'BTCUSDT' not in position_manager.positions
            mock_order.assert_called_once()


    def test_portfolio_summary(self, position_manager):
        """Test portfolio summary generation"""
        # Add some test positions
        position_manager.positions['BTCUSDT'] = {
            'side': 'long',
            'size': 0.1,
            'entry_price': 45000,
            'current_price': 46000,
            'pnl': 100,
            'timestamp': datetime.now()
        }

        position_manager.positions['ETHUSDT'] = {
            'side': 'short',
            'size': 1.0,
            'entry_price': 3000,
            'current_price': 2950,
            'pnl': 50,
            'timestamp': datetime.now()
        }

        summary = position_manager.get_portfolio_summary()

        assert isinstance(summary, dict)
        assert 'portfolio_value' in summary
        assert 'total_pnl' in summary
        assert 'open_positions_count' in summary
        assert summary['open_positions_count'] == 2


    def test_position_size_calculation(self, position_manager):
        """Test position size calculation"""
        # Test with different risk levels
        size_conservative = position_manager._calculate_position_size(
            symbol='BTCUSDT',
            current_price=45000,
            confidence=0.6,
            risk_level='conservative'
        )

        size_aggressive = position_manager._calculate_position_size(
            symbol='BTCUSDT',
            current_price=45000,
            confidence=0.9,
            risk_level='aggressive'
        )

        # Aggressive should be larger than conservative
        assert size_aggressive > size_conservative

        # Both should be positive and reasonable
        assert 0 < size_conservative < 1
        assert 0 < size_aggressive < 1

    @pytest.mark.asyncio
    async def test_update_positions(self, position_manager):
        """Test position updates with current prices"""
        # Add test position
        position_manager.positions['BTCUSDT'] = {
            'side': 'long',
            'size': 0.1,
            'entry_price': 45000,
            'current_price': 45000,
            'pnl': 0,
            'timestamp': datetime.now()
        }

        # Update with new prices
        current_prices = {'BTCUSDT': 46000}

        result = await position_manager.update_positions(current_prices)

        # Position should be updated
        updated_position = position_manager.positions['BTCUSDT']
        assert updated_position['current_price'] == 46000
        assert updated_position['pnl'] > 0  # Should be profitable

        assert 'total_pnl' in result


class TestRiskManager:
    """Test suite for RiskManager class"""

    @pytest.fixture
    def risk_manager(self):
        """Create RiskManager instance for testing"""
        return RiskManager()

    def test_assess_portfolio_risk(self, risk_manager):
        """Test portfolio risk assessment"""
        # Create mock portfolio data
        portfolio_value = 10000
        positions = {
            'BTCUSDT': {'pnl': -200, 'size': 0.1},
            'ETHUSDT': {'pnl': 150, 'size': 0.5}
        }
        daily_return = -0.02  # -2% daily return

        risk_assessment = risk_manager.assess_portfolio_risk(
            portfolio_value, positions, daily_return
        )

        assert isinstance(risk_assessment, dict)
        assert 'overall_risk_level' in risk_assessment
        assert 'risk_metrics' in risk_assessment
        assert isinstance(risk_assessment['overall_risk_level'], RiskLevel)


    def test_position_risk_check(self, risk_manager):
        """Test individual position risk checking"""
        position = {
            'symbol': 'BTCUSDT',
            'side': 'long',
            'size': 0.1,
            'entry_price': 45000,
            'current_price': 42000,  # 6.67% loss
            'pnl': -300
        }

        risk_check = risk_manager.check_position_risk(position)

        assert isinstance(risk_check, dict)
        assert 'risk_level' in risk_check
        assert 'suggested_action' in risk_check

        # Should suggest closing due to large loss
        assert risk_check['suggested_action'] in ['reduce', 'close']


    def test_drawdown_calculation(self, risk_manager):
        """Test drawdown calculation"""
        # Simulate portfolio value history with drawdown
        portfolio_values = [10000, 10500, 11000, 10200, 9800, 9500, 10200, 10800]

        max_drawdown = risk_manager._calculate_max_drawdown(portfolio_values)

        # Should detect the drawdown from 11000 to 9500
        expected_drawdown = (11000 - 9500) / 11000 * 100  # ~13.64%

        assert abs(max_drawdown - expected_drawdown) < 0.1


    def test_risk_limits(self, risk_manager):
        """Test risk limit enforcement"""
        # Test daily loss limit
        portfolio_value = 10000
        daily_pnl = -600  # -6% loss, should trigger limit

        risk_check = risk_manager.check_daily_loss_limit(portfolio_value, daily_pnl)

        assert risk_check['limit_exceeded'] is True
        assert risk_check['recommended_action'] == 'stop_trading'

        # Test position concentration
        positions = {
            'BTCUSDT': {'size': 0.8},  # 80% of portfolio in one position
            'ETHUSDT': {'size': 0.1}
        }

        concentration_check = risk_manager.check_position_concentration(positions)

        assert concentration_check['concentration_risk'] == 'high'


    def test_risk_scoring(self, risk_manager):
        """Test risk scoring system"""
        # Low risk scenario
        low_risk_metrics = {
            'portfolio_volatility': 0.10,
            'max_drawdown': 0.05,
            'var_95': 0.02,
            'position_concentration': 0.15
        }

        low_score = risk_manager._calculate_risk_score(low_risk_metrics)

        # High risk scenario
        high_risk_metrics = {
            'portfolio_volatility': 0.40,
            'max_drawdown': 0.25,
            'var_95': 0.15,
            'position_concentration': 0.80
        }

        high_score = risk_manager._calculate_risk_score(high_risk_metrics)

        # High risk should have higher score
        assert high_score > low_score

        # Both scores should be between 0 and 100
        assert 0 <= low_score <= 100
        assert 0 <= high_score <= 100


class TestExchangeManager:
    """Test suite for ExchangeManager class"""

    @pytest.fixture
    def exchange_manager(self):
        """Create ExchangeManager instance for testing"""
        return ExchangeManager()

    @pytest.mark.asyncio
    async def test_initialize_exchanges(self, exchange_manager):
        """Test exchange initialization"""
        with patch('ccxt.binance') as mock_binance:
            mock_exchange_instance = Mock()
            mock_exchange_instance.load_markets = AsyncMock()
            mock_binance.return_value = mock_exchange_instance

            result = await exchange_manager.initialize_exchanges(['binance'])

            assert result is True
            assert 'binance' in exchange_manager.exchanges
            mock_exchange_instance.load_markets.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_ticker(self, exchange_manager):
        """Test ticker fetching"""
        # Mock exchange
        mock_exchange = Mock()
        mock_exchange.fetch_ticker = AsyncMock(return_value={
            'symbol': 'BTC/USDT',
            'last': 45000,
            'bid': 44990,
            'ask': 45010,
            'volume': 1000
        })

        exchange_manager.exchanges['binance'] = mock_exchange

        ticker = await exchange_manager.fetch_ticker('BTCUSDT', 'binance')

        assert ticker is not None
        assert ticker['symbol'] == 'BTC/USDT'
        assert ticker['last'] == 45000
        mock_exchange.fetch_ticker.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_ohlcv(self, exchange_manager):
        """Test OHLCV data fetching"""
        # Mock OHLCV data
        mock_ohlcv = [
            [1640995200000, 45000, 45100, 44900, 45050, 1000],  # timestamp, o, h, l, c, v
            [1640998800000, 45050, 45200, 45000, 45150, 1200],
            [1641002400000, 45150, 45300, 45100, 45250, 800]
        ]

        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=mock_ohlcv)

        exchange_manager.exchanges['binance'] = mock_exchange

        df = await exchange_manager.fetch_ohlcv(
            symbol='BTCUSDT',
            timeframe='1h',
            exchange='binance'
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert df['open'].iloc[0] == 45000
        mock_exchange.fetch_ohlcv.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_order(self, exchange_manager):
        """Test order placement"""
        mock_order = {
            'id': '12345',
            'symbol': 'BTC/USDT',
            'type': 'market',
            'side': 'buy',
            'amount': 0.1,
            'status': 'closed',
            'filled': 0.1
        }

        mock_exchange = Mock()
        mock_exchange.create_market_buy_order = AsyncMock(return_value=mock_order)

        exchange_manager.exchanges['binance'] = mock_exchange

        order = await exchange_manager.place_order(
            symbol='BTCUSDT',
            side='buy',
            order_type='market',
            amount=0.1,
            exchange='binance'
        )

        assert order is not None
        assert order['id'] == '12345'
        assert order['status'] == 'closed'
        mock_exchange.create_market_buy_order.assert_called_once()

# Integration Tests


class TestTradingIntegration:
    """Integration tests for trading components working together"""

    @pytest.fixture
    async def integrated_trading_system(self):
        """Create integrated trading system for testing"""
        config = TradingSystemConfig(
            symbols=['BTCUSDT'],
            timeframes=['1h'],
            trading_mode=TradingMode.SIMULATION,
            initial_capital=10000,
            update_interval=60
        )

        system = TradingSystem(config)
        yield system

        try:
            await system.stop()
        except:
            pass

    @pytest.mark.asyncio
    async def test_signal_to_position_flow(self, integrated_trading_system):
        """Test complete flow from signal generation to position management"""
        # Mock all external dependencies
        with patch.object(integrated_trading_system, 'initialize', return_value=True):

            # Mock signal generation
            mock_signal = TradingSignal(
                symbol='BTCUSDT',
                timeframe='1h',
                signal_type=SignalType.BUY,
                confidence=0.8,
                price=45000,
                source=SignalSource.ML_MODEL,
                timestamp=datetime.now()
            )

            with patch.object(integrated_trading_system, '_generate_signals', return_value={'BTCUSDT': mock_signal}):
                with patch.object(integrated_trading_system, '_update_market_data', return_value={'BTCUSDT_1h': pd.DataFrame()}):
                    with patch.object(integrated_trading_system, '_assess_risk', return_value={'overall_risk_level': RiskLevel.LOW}):
                        with patch.object(integrated_trading_system.position_manager, 'open_position', return_value={'success': True}):
                            with patch.object(integrated_trading_system, '_update_positions', return_value={}):
                                with patch.object(integrated_trading_system, '_log_cycle_results', return_value=None):

                                    # Execute trading cycle
                                    await integrated_trading_system._execute_trading_cycle()

                                    # Verify position manager was called
                                    integrated_trading_system.position_manager.open_position.assert_called_once()

# Performance Tests


class TestTradingPerformance:
    """Performance tests for trading operations"""

    @pytest.mark.asyncio
    async def test_signal_generation_performance(self):
        """Test signal generation performance"""
        signal_generator = SignalGenerator()

        # Create large market data
        dates = pd.date_range('2020-01-01', periods=1000, freq='1H')
        large_market_data = pd.DataFrame({
            'open': np.random.randn(1000) + 45000,
            'high': np.random.randn(1000) + 45100,
            'low': np.random.randn(1000) + 44900,
            'close': np.random.randn(1000) + 45000,
            'volume': np.random.randint(1000, 5000, 1000)
        }, index=dates)

        import time
        start_time = time.time()

        signal = await signal_generator.generate_signal(
            symbol='BTCUSDT',
            timeframe='1h',
            market_data=large_market_data,
            model_version='test'
        )

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete within reasonable time
        assert processing_time < 5.0  # 5 seconds max
        assert isinstance(signal, TradingSignal)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
