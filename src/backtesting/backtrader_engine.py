"""
Backtrader Backtesting Engine
Professional backtesting framework using Backtrader
Supports multiple strategies, performance analysis, and risk management
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import json
import pickle
import warnings

warnings.filterwarnings('ignore')

# Analysis and visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
# import quantstats as qs  # Optional dependency

from config.settings import config
from src.utils.logger import setup_logger
from src.utils.helpers import timing_decorator, calculate_sharpe_ratio, calculate_max_drawdown
from src.backtesting.wyckoff_strategy import WyckoffAnalyzer

logger = setup_logger(__name__)


class MLSignalStrategy(bt.Strategy):
    """ML-based trading strategy for Backtrader"""

    params = (
        ('model', None),           # ML model for predictions
        ('feature_list', None),    # List of features to use
        ('lookback', 50),          # Lookback period for features
        ('stop_loss_pct', 0.02),   # Stop loss percentage
        ('take_profit_pct', 0.05), # Take profit percentage
        ('position_size', 0.95),   # Position size (% of available cash)
        ('min_confidence', 0.6),   # Minimum confidence for trade
        ('hold_days', 5),          # Maximum holding period
        ('rebalance_freq', 1),     # Rebalancing frequency (days)
    )


    def __init__(self):
        # Data feeds
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume

        # Track positions
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.entry_bar = None

        # ML predictions storage
        self.predictions = {}
        self.confidences = {}

        # Performance tracking
        self.trade_count = 0
        self.winning_trades = 0
        self.total_return = 0.0

        # Initialize indicators for feature generation
        self.sma_20 = bt.indicators.SimpleMovingAverage(self.datas[0], period=20)
        self.sma_50 = bt.indicators.SimpleMovingAverage(self.datas[0], period=50)
        self.rsi = bt.indicators.RelativeStrengthIndex(self.datas[0], period=14)
        self.macd = bt.indicators.MACD(self.datas[0])
        self.bbands = bt.indicators.BollingerBands(self.datas[0], period=20)
        self.atr = bt.indicators.AverageTrueRange(self.datas[0], period=14)

        logger.info(f"ML Strategy initialized with model: {self.params.model is not None}")


    def log(self, txt, dt=None, level='info'):
        """Custom logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        if level == 'error':
            logger.error(f'{dt.isoformat()}, {txt}')
        elif level == 'warning':
            logger.warning(f'{dt.isoformat()}, {txt}')
        else:
            logger.debug(f'{dt.isoformat()}, {txt}')


    def notify_order(self, order):
        """Order notification"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.entry_bar = len(self)
            else:  # Sell
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order {order.getstatusname()}', level='warning')

        self.order = None


    def notify_trade(self, trade):
        """Trade notification"""
        if not trade.isclosed:
            return

        self.trade_count += 1
        pnl = trade.pnl
        pnl_pct = (trade.pnl / trade.value) * 100

        if trade.pnl > 0:
            self.winning_trades += 1

        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, '
                f'NET: {trade.pnlcomm:.2f}, PCT: {pnl_pct:.2f}%')


    def generate_features(self) -> Optional[pd.Series]:
        """Generate features for ML prediction"""
        try:
            if len(self) < self.params.lookback:
                return None

            # Price features
            close = self.dataclose[0]
            high = self.datahigh[0]
            low = self.datalow[0]
            volume = self.datavolume[0]

            # Calculate returns
            returns_1 = (close / self.dataclose[-1] - 1) if len(self) > 1 else 0
            returns_5 = (close / self.dataclose[-5] - 1) if len(self) > 5 else 0
            returns_20 = (close / self.dataclose[-20] - 1) if len(self) > 20 else 0

            # Technical indicators
            sma_20_val = self.sma_20[0] if len(self.sma_20) > 0 else close
            sma_50_val = self.sma_50[0] if len(self.sma_50) > 0 else close
            rsi_val = self.rsi[0] if len(self.rsi) > 0 else 50
            macd_val = self.macd.macd[0] if len(self.macd.macd) > 0 else 0
            macd_signal = self.macd.signal[0] if len(self.macd.signal) > 0 else 0
            bb_upper = self.bbands.top[0] if len(self.bbands.top) > 0 else close * 1.02
            bb_lower = self.bbands.bot[0] if len(self.bbands.bot) > 0 else close * 0.98
            atr_val = self.atr[0] if len(self.atr) > 0 else (high - low)

            # Volatility features
            volatility_20 = np.std([self.dataclose[-i] for i in range(min(20, len(self)))])

            # Volume features
            volume_sma_20 = np.mean([self.datavolume[-i] for i in range(min(20, len(self)))])
            volume_ratio = volume / volume_sma_20 if volume_sma_20 > 0 else 1

            # Price position features
            price_position = (close - low) / (high - low) if (high - low) > 0 else 0.5
            sma_position = (close - sma_20_val) / sma_20_val if sma_20_val > 0 else 0

            # Bollinger Bands position
            bb_position = (close - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

            # Create feature vector
            features = pd.Series({
                'close': close,
                'high': high,
                'low': low,
                'volume': volume,
                'returns_1': returns_1,
                'returns_5': returns_5,
                'returns_20': returns_20,
                'sma_20': sma_20_val,
                'sma_50': sma_50_val,
                'rsi': rsi_val,
                'macd': macd_val,
                'macd_signal': macd_signal,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'atr': atr_val,
                'volatility_20': volatility_20,
                'volume_ratio': volume_ratio,
                'price_position': price_position,
                'sma_position': sma_position,
                'bb_position': bb_position,
                'sma_cross': 1 if sma_20_val > sma_50_val else 0,
                'macd_cross': 1 if macd_val > macd_signal else 0,
            })

            return features

        except Exception as e:
            self.log(f"Feature generation failed: {e}", level='error')
            return None


    def get_ml_prediction(self) -> Tuple[int, float]:
        """Get ML model prediction"""
        try:
            if self.params.model is None:
                return 0, 0.5  # Neutral with no confidence

            features = self.generate_features()
            if features is None:
                return 0, 0.5

            # Select only features used by the model
            if self.params.feature_list:
                available_features = [f for f in self.params.feature_list if f in features.index]
                if len(available_features) == 0:
                    return 0, 0.5
                features_selected = features[available_features]
            else:
                features_selected = features

            # Make prediction
            prediction = self.params.model.predict([features_selected.values])[0]

            # Get prediction probability/confidence if available
            try:
                if hasattr(self.params.model, 'predict_proba'):
                    probabilities = self.params.model.predict_proba([features_selected.values])[0]
                    confidence = max(probabilities)
                else:
                    confidence = 0.6  # Default confidence
            except:
                confidence = 0.6

            return int(prediction), float(confidence)

        except Exception as e:
            self.log(f"ML prediction failed: {e}", level='error')
            return 0, 0.5


    def next(self):
        """Main strategy logic"""
        # Skip if we have a pending order
        if self.order:
            return

        # Get ML prediction
        prediction, confidence = self.get_ml_prediction()

        # Store prediction for analysis
        current_date = self.datas[0].datetime.date(0)
        self.predictions[current_date] = prediction
        self.confidences[current_date] = confidence

        # Current position
        position = self.position.size
        current_price = self.dataclose[0]

        # Exit conditions
        if position != 0:
            # Check stop loss
            if position > 0:  # Long position
                if current_price <= self.buyprice * (1 - self.params.stop_loss_pct):
                    self.log(f'Stop Loss triggered at {current_price:.2f}')
                    self.order = self.sell()
                    return

                # Check take profit
                if current_price >= self.buyprice * (1 + self.params.take_profit_pct):
                    self.log(f'Take Profit triggered at {current_price:.2f}')
                    self.order = self.sell()
                    return

            else:  # Short position
                if current_price >= self.buyprice * (1 + self.params.stop_loss_pct):
                    self.log(f'Stop Loss triggered at {current_price:.2f}')
                    self.order = self.buy()
                    return

                # Check take profit
                if current_price <= self.buyprice * (1 - self.params.take_profit_pct):
                    self.log(f'Take Profit triggered at {current_price:.2f}')
                    self.order = self.buy()
                    return

            # Check maximum holding period
            if self.entry_bar and len(self) - self.entry_bar >= self.params.hold_days:
                self.log(f'Max holding period reached, closing position')
                if position > 0:
                    self.order = self.sell()
                else:
                    self.order = self.buy()
                return

        # Entry conditions
        if confidence >= self.params.min_confidence:
            # Calculate position size
            available_cash = self.broker.getcash()
            position_value = available_cash * self.params.position_size
            size = int(position_value / current_price)

            if prediction == 1 and position <= 0:  # Long signal
                self.log(f'BUY CREATE, Price: {current_price:.2f}, Confidence: {confidence:.2f}')
                if position < 0:  # Close short first
                    self.order = self.buy()
                else:  # Open long
                    self.order = self.buy(size=size)

            elif prediction == -1 and position >= 0:  # Short signal
                self.log(f'SELL CREATE, Price: {current_price:.2f}, Confidence: {confidence:.2f}')
                if position > 0:  # Close long first
                    self.order = self.sell()
                else:  # Open short
                    self.order = self.sell(size=size)


class WyckoffSignalStrategy(bt.Strategy):
    """Wyckoff-based trading strategy for Backtrader"""

    params = (
        ('stop_loss_pct', 0.02),
        ('take_profit_pct', 0.05),
        ('position_size', 0.95),
        ('min_confidence', 0.7),
        ('hold_days', 5),
    )

    def __init__(self):
        # Data feeds
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume

        # Track positions
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.entry_bar = None

        # Wyckoff analyzer
        self.wyckoff_analyzer = WyckoffAnalyzer()
        
        # Initialize indicators
        self.sma_20 = bt.indicators.SimpleMovingAverage(self.datas[0], period=20)
        self.sma_50 = bt.indicators.SimpleMovingAverage(self.datas[0], period=50)
        self.volume_sma = bt.indicators.SimpleMovingAverage(self.datas[0].volume, period=20)

        logger.info("Wyckoff Strategy initialized")

    def log(self, txt, dt=None, level='info'):
        """Custom logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        if level == 'error':
            logger.error(f'{dt.isoformat()}, {txt}')
        elif level == 'warning':
            logger.warning(f'{dt.isoformat()}, {txt}')
        else:
            logger.debug(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        """Order notification"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.entry_bar = len(self)
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order {order.getstatusname()}', level='warning')

        self.order = None

    def get_wyckoff_signal(self):
        """Get Wyckoff analysis signal"""
        try:
            # Prepare data for analysis (last 100 bars)
            lookback = min(100, len(self))
            if lookback < 50:
                return 'NEUTRAL', 0.5
            
            # Create DataFrame from current data
            data_dict = {
                'open': [self.datas[0].open[-i] for i in range(lookback-1, -1, -1)],
                'high': [self.datas[0].high[-i] for i in range(lookback-1, -1, -1)],
                'low': [self.datas[0].low[-i] for i in range(lookback-1, -1, -1)],
                'close': [self.datas[0].close[-i] for i in range(lookback-1, -1, -1)],
                'volume': [self.datas[0].volume[-i] for i in range(lookback-1, -1, -1)]
            }
            
            df = pd.DataFrame(data_dict)
            
            # Run Wyckoff analysis
            analysis = self.wyckoff_analyzer.analyze_wyckoff_structure(df)
            signals = analysis.get('trading_signals', {})
            
            return signals.get('signal', 'NEUTRAL'), signals.get('confidence', 0.5)
            
        except Exception as e:
            self.log(f"Wyckoff signal failed: {e}", level='error')
            return 'NEUTRAL', 0.5

    def next(self):
        """Main strategy logic"""
        if self.order:
            return

        # Get Wyckoff signal
        signal, confidence = self.get_wyckoff_signal()
        
        # Current position and price
        position = self.position.size
        current_price = self.dataclose[0]

        # Exit conditions
        if position != 0:
            if position > 0:  # Long position
                if current_price <= self.buyprice * (1 - self.params.stop_loss_pct):
                    self.log(f'Stop Loss triggered at {current_price:.2f}')
                    self.order = self.sell()
                    return
                elif current_price >= self.buyprice * (1 + self.params.take_profit_pct):
                    self.log(f'Take Profit triggered at {current_price:.2f}')
                    self.order = self.sell()
                    return
            else:  # Short position
                if current_price >= self.buyprice * (1 + self.params.stop_loss_pct):
                    self.log(f'Stop Loss triggered at {current_price:.2f}')
                    self.order = self.buy()
                    return
                elif current_price <= self.buyprice * (1 - self.params.take_profit_pct):
                    self.log(f'Take Profit triggered at {current_price:.2f}')
                    self.order = self.buy()
                    return

            # Maximum holding period
            if self.entry_bar and len(self) - self.entry_bar >= self.params.hold_days:
                self.log('Max holding period reached')
                if position > 0:
                    self.order = self.sell()
                else:
                    self.order = self.buy()
                return

        # Entry conditions based on Wyckoff signals
        if confidence >= self.params.min_confidence:
            available_cash = self.broker.getcash()
            position_value = available_cash * self.params.position_size
            size = int(position_value / current_price)

            if signal == 'BULLISH' and position <= 0:
                self.log(f'WYCKOFF BUY: {signal} (Confidence: {confidence:.2f})')
                if position < 0:
                    self.order = self.buy()
                else:
                    self.order = self.buy(size=size)
            elif signal == 'BEARISH' and position >= 0:
                self.log(f'WYCKOFF SELL: {signal} (Confidence: {confidence:.2f})')
                if position > 0:
                    self.order = self.sell()
                else:
                    self.order = self.sell(size=size)


class BacktestEngine:
    """Main backtesting engine using Backtrader"""

    def __init__(self):
        self.cerebro = None
        self.results = {}
        self.strategies = {
            'ml_signal': MLSignalStrategy,
            'wyckoff': WyckoffSignalStrategy
        }

    @timing_decorator
    def run_backtest(self, symbol: str, timeframe: str,
                    backtest_config: Dict[str, Any],
                    data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Run backtest for a symbol and timeframe

        Args:
            symbol: Trading symbol
            timeframe: Time frame
            backtest_config: Backtest configuration
            data: Optional data DataFrame (if not provided, will load from database)

        Returns:
            Backtest results
        """
        try:
            logger.info(f"Starting backtest for {symbol}_{timeframe}")

            # Initialize Cerebro
            self.cerebro = bt.Cerebro()

            # Set initial capital
            initial_capital = backtest_config.get('initial_capital', 10000)
            self.cerebro.broker.setcash(initial_capital)

            # Set commission
            commission = backtest_config.get('commission', 0.001)
            self.cerebro.broker.setcommission(commission=commission)

            # Load or use provided data
            if data is None:
                data = self._load_backtest_data(
                    symbol, timeframe,
                    backtest_config.get('start_date'),
                    backtest_config.get('end_date')
                )

            if data.empty:
                raise ValueError(f"No data available for {symbol}_{timeframe}")

            # Convert to Backtrader format
            bt_data = self._prepare_backtrader_data(data)
            self.cerebro.adddata(bt_data)

            # Add strategy
            strategy_name = backtest_config.get('strategy_name', 'ml_signal')
            strategy_params = backtest_config.get('strategy_params', {})
            
            # Select strategy class
            strategy_class = self.strategies.get(strategy_name, MLSignalStrategy)
            self.cerebro.addstrategy(strategy_class, **strategy_params)
            
            logger.info(f"Using strategy: {strategy_name} ({strategy_class.__name__})")

            # Add analyzers
            self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')

            # Add observers
            self.cerebro.addobserver(bt.observers.BuySell)
            self.cerebro.addobserver(bt.observers.Value)

            # Record starting value
            start_value = self.cerebro.broker.getvalue()
            logger.info(f'Starting Portfolio Value: {start_value:.2f}')

            # Run backtest
            results = self.cerebro.run()

            # Record ending value
            end_value = self.cerebro.broker.getvalue()
            logger.info(f'Final Portfolio Value: {end_value:.2f}')

            # Extract results
            backtest_results = self._extract_results(results[0], initial_capital, data)

            # Add metadata
            backtest_results.update({
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': data.index[0].strftime('%Y-%m-%d'),
                'end_date': data.index[-1].strftime('%Y-%m-%d'),
                'initial_capital': initial_capital,
                'final_capital': end_value,
                'backtest_config': backtest_config,
                'data_points': len(data)
            })

            logger.info(f"Backtest completed for {symbol}_{timeframe}")
            logger.info(f"Total Return: {backtest_results.get('total_return', 0):.2%}")
            logger.info(f"Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}")
            logger.info(f"Max Drawdown: {backtest_results.get('max_drawdown', 0):.2%}")

            return backtest_results

        except Exception as e:
            logger.error(f"Backtest failed for {symbol}_{timeframe}: {e}")
            return {}


    def _load_backtest_data(self, symbol: str, timeframe: str,
                           start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load data for backtesting"""
        try:
            from src.utils.database_manager import DatabaseManager

            db_manager = DatabaseManager()

            # Convert date strings to datetime
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None

            # Load data
            df = db_manager.load_ohlcv_data(symbol, timeframe, start_dt, end_dt)

            if df.empty:
                logger.warning(f"No data found in database for {symbol}_{timeframe}")
                return pd.DataFrame()

            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            return df[required_cols].dropna()

        except Exception as e:
            logger.error(f"Failed to load backtest data: {e}")
            return pd.DataFrame()


    def _prepare_backtrader_data(self, df: pd.DataFrame) -> bt.feeds.PandasData:
        """Convert DataFrame to Backtrader data format"""
        try:
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Create Backtrader data feed
            data_feed = bt.feeds.PandasData(
                dataname=df,
                datetime=None,  # Use index
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest=-1  # No open interest
            )

            return data_feed

        except Exception as e:
            logger.error(f"Failed to prepare Backtrader data: {e}")
            raise


    def _extract_results(self, strategy_result, initial_capital: float, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract and calculate backtest results"""
        try:
            results = {}

            # Basic performance metrics
            final_value = strategy_result.broker.getvalue()
            total_return = (final_value - initial_capital) / initial_capital

            # Time-based returns
            trading_days = len(data)
            years = trading_days / 252  # Approximate trading days per year
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

            results.update({
                'total_return': total_return,
                'annual_return': annual_return,
                'final_value': final_value,
                'profit_loss': final_value - initial_capital
            })

            # Analyzer results
            analyzers = strategy_result.analyzers

            # Sharpe Ratio
            try:
                sharpe = analyzers.sharpe.get_analysis()
                results['sharpe_ratio'] = sharpe.get('sharperatio', 0) or 0
            except:
                results['sharpe_ratio'] = 0

            # Drawdown
            try:
                drawdown = analyzers.drawdown.get_analysis()
                results['max_drawdown'] = drawdown.get('max', {}).get('drawdown', 0) / 100
                results['max_drawdown_duration'] = drawdown.get('max', {}).get('len', 0)
            except:
                results['max_drawdown'] = 0
                results['max_drawdown_duration'] = 0

            # Trade Analysis
            try:
                trades = analyzers.trades.get_analysis()
                results.update({
                    'total_trades': trades.get('total', {}).get('total', 0),
                    'winning_trades': trades.get('won', {}).get('total', 0),
                    'losing_trades': trades.get('lost', {}).get('total', 0),
                    'win_rate': trades.get('won', {}).get('total', 0) / max(1, trades.get('total', {}).get('total', 1)),
                    'avg_win': trades.get('won', {}).get('pnl', {}).get('average', 0),
                    'avg_loss': trades.get('lost', {}).get('pnl', {}).get('average', 0),
                    'largest_win': trades.get('won', {}).get('pnl', {}).get('max', 0),
                    'largest_loss': trades.get('lost', {}).get('pnl', {}).get('max', 0),
                    'gross_profit': trades.get('won', {}).get('pnl', {}).get('total', 0),
                    'gross_loss': abs(trades.get('lost', {}).get('pnl', {}).get('total', 0))
                })

                # Profit Factor
                gross_profit = results['gross_profit']
                gross_loss = results['gross_loss']
                results['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            except Exception as e:
                logger.warning(f"Failed to extract trade analysis: {e}")
                results.update({
                    'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                    'win_rate': 0, 'profit_factor': 0
                })

            # Time Return Analysis
            try:
                time_returns = analyzers.time_return.get_analysis()
                if time_returns:
                    returns_series = pd.Series(time_returns)

                    # Additional metrics
                    results.update({
                        'volatility': returns_series.std() * np.sqrt(252),  # Annualized
                        'best_month': returns_series.max(),
                        'worst_month': returns_series.min(),
                        'positive_months': (returns_series > 0).sum(),
                        'negative_months': (returns_series < 0).sum()
                    })
            except:
                results.update({
                    'volatility': 0, 'best_month': 0, 'worst_month': 0,
                    'positive_months': 0, 'negative_months': 0
                })

            # Strategy-specific results
            if hasattr(strategy_result, 'trade_count'):
                results['strategy_trade_count'] = strategy_result.trade_count
                results['strategy_winning_trades'] = strategy_result.winning_trades

            return results

        except Exception as e:
            logger.error(f"Failed to extract results: {e}")
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}


    def plot_backtest(self, save_path: str = None, **kwargs):
        """Plot backtest results"""
        try:
            if self.cerebro is None:
                logger.error("No backtest to plot. Run backtest first.")
                return

            # Configure plot style
            plt.style.use('seaborn-v0_8')

            # Plot with Backtrader
            figs = self.cerebro.plot(
                style='candlestick',
                barup='green',
                bardown='red',
                volume=True,
                **kwargs
            )

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Backtest plot saved to {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"Failed to plot backtest: {e}")


    def save_backtest_results(self, symbol: str, timeframe: str, version: str,
                            results: Dict[str, Any]) -> str:
        """Save backtest results to file"""
        try:
            results_path = config.get_processed_path(symbol, timeframe, version)
            results_path.mkdir(parents=True, exist_ok=True)

            # Save detailed results
            results_file = results_path / 'backtest_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # Save summary report
            summary_file = results_path / 'backtest_summary.txt'
            with open(summary_file, 'w') as f:
                f.write(f"Backtest Results for {symbol}_{timeframe}_{version}\n")
                f.write("=" * 50 + "\n\n")

                f.write(f"Period: {results.get('start_date')} to {results.get('end_date')}\n")
                f.write(f"Initial Capital: ${results.get('initial_capital', 0):,.2f}\n")
                f.write(f"Final Capital: ${results.get('final_capital', 0):,.2f}\n")
                f.write(f"Total Return: {results.get('total_return', 0):.2%}\n")
                f.write(f"Annual Return: {results.get('annual_return', 0):.2%}\n")
                f.write(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}\n")
                f.write(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}\n")
                f.write(f"Win Rate: {results.get('win_rate', 0):.2%}\n")
                f.write(f"Total Trades: {results.get('total_trades', 0)}\n")
                f.write(f"Profit Factor: {results.get('profit_factor', 0):.2f}\n")

            logger.info(f"Backtest results saved to {results_path}")
            return str(results_path)

        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")
            return ""

# Convenience functions


def run_simple_backtest(symbol: str, timeframe: str, model,
                       start_date: str, end_date: str,
                       initial_capital: float = 10000,
                       strategy_name: str = 'ml_signal') -> Dict[str, Any]:
    """Run a simple backtest with minimal configuration"""

    engine = BacktestEngine()

    backtest_config = {
        'initial_capital': initial_capital,
        'commission': 0.001,
        'start_date': start_date,
        'end_date': end_date,
        'strategy_name': strategy_name,
        'strategy_params': {
            'model': model,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05
        }
    }

    return engine.run_backtest(symbol, timeframe, backtest_config)


def compare_strategies(symbol: str, timeframe: str, strategies_config: List[Dict],
                      start_date: str, end_date: str) -> Dict[str, Dict[str, Any]]:
    """Compare multiple strategies"""

    engine = BacktestEngine()
    results = {}

    for i, config in enumerate(strategies_config):
        strategy_name = config.get('name', f'Strategy_{i+1}')

        try:
            result = engine.run_backtest(symbol, timeframe, config)
            results[strategy_name] = result

        except Exception as e:
            logger.error(f"Strategy {strategy_name} failed: {e}")
            results[strategy_name] = {}

    return results

# Usage example
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
    np.random.seed(42)

    prices = 45000 + np.cumsum(np.random.randn(len(dates)) * 100)

    sample_data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.randint(1000, 5000, len(dates))
    }, index=dates)

    # Test backtest
    engine = BacktestEngine()

    backtest_config = {
        'initial_capital': 10000,
        'commission': 0.001,
        'start_date': '2024-01-01',
        'end_date': '2024-06-30',
        'strategy_params': {
            'model': None,  # Use simple moving average strategy
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05
        }
    }

    results = engine.run_backtest('BTCUSDT', '1d', backtest_config, sample_data)

    print(f"Backtest Results:")
    print(f"Total Return: {results.get('total_return', 0):.2%}")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
    print(f"Win Rate: {results.get('win_rate', 0):.2%}")

    print("Backtest engine example completed!")
