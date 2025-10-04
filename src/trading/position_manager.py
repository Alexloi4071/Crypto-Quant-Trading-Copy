"""
Position Manager Module
Manages trading positions, portfolio allocation, and position sizing
Supports multi-symbol portfolio management and risk-aware position sizing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
import warnings

warnings.filterwarnings('ignore')

from config.settings import config
from src.utils.logger import setup_logger, get_trading_logger
from src.utils.helpers import timing_decorator
from src.trading.exchange_manager import ExchangeManager

logger = setup_logger(__name__)

class PositionStatus(Enum):
    """Position status types"""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"
    CLOSING = "closing"

class PositionSide(Enum):
    """Position side types"""
    LONG = "long"
    SHORT = "short"

@dataclass

class Position:
    """Individual position data structure"""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN

    # Calculated fields
    current_price: float = field(default=0.0)
    unrealized_pnl: float = field(default=0.0)
    unrealized_pnl_pct: float = field(default=0.0)

    # Execution details
    entry_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    realized_pnl: Optional[float] = None

    # Risk management
    max_loss_pct: float = field(default=0.02)  # 2% max loss
    trailing_stop_pct: Optional[float] = None

    # Metadata
    signal_source: str = field(default="unknown")
    confidence: float = field(default=0.5)
    notes: str = field(default="")

    def update_current_price(self, price: float):
        """Update current price and calculate unrealized PnL"""
        self.current_price = price

        if self.status == PositionStatus.OPEN:
            if self.side == PositionSide.LONG:
                self.unrealized_pnl = (price - self.entry_price) * self.quantity
                self.unrealized_pnl_pct = (price - self.entry_price) / self.entry_price
            else:  # SHORT
                self.unrealized_pnl = (self.entry_price - price) * self.quantity
                self.unrealized_pnl_pct = (self.entry_price - price) / self.entry_price

    def should_close_position(self) -> Tuple[bool, str]:
        """Check if position should be closed based on risk rules"""
        if self.status != PositionStatus.OPEN:
            return False, "Position not open"

        # Stop loss check
        if self.stop_loss:
            if self.side == PositionSide.LONG and self.current_price <= self.stop_loss:
                return True, "Stop loss triggered"
            elif self.side == PositionSide.SHORT and self.current_price >= self.stop_loss:
                return True, "Stop loss triggered"

        # Take profit check
        if self.take_profit:
            if self.side == PositionSide.LONG and self.current_price >= self.take_profit:
                return True, "Take profit triggered"
            elif self.side == PositionSide.SHORT and self.current_price <= self.take_profit:
                return True, "Take profit triggered"

        # Maximum loss percentage check
        if abs(self.unrealized_pnl_pct) >= self.max_loss_pct:
            if self.unrealized_pnl < 0:
                return True, f"Maximum loss reached ({self.unrealized_pnl_pct:.1%})"

        return False, ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'status': self.status.value,
            'signal_source': self.signal_source,
            'confidence': self.confidence,
            'entry_order_id': self.entry_order_id,
            'notes': self.notes
        }

class Portfolio:
    """Portfolio management class"""

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []

        # Portfolio metrics
        self.total_value = initial_capital
        self.total_pnl = 0.0
        self.total_pnl_pct = 0.0
        self.max_drawdown = 0.0
        self.peak_value = initial_capital

        # Risk parameters
        self.max_position_size_pct = config.get('MAX_POSITION_SIZE', 0.05)  # 5% per position
        self.max_total_exposure_pct = config.get('MAX_TOTAL_EXPOSURE', 0.8)  # 80% total exposure
        self.correlation_threshold = 0.7  # Avoid highly correlated positions

        logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")

    def update_portfolio_value(self, prices: Dict[str, float]):
        """Update portfolio value with current prices"""
        try:
            # Update position prices
            position_value = 0.0

            for symbol, position in self.positions.items():
                if symbol in prices:
                    position.update_current_price(prices[symbol])
                    position_value += abs(position.quantity * position.current_price)

            # Calculate total portfolio value
            self.total_value = self.cash + sum(
                pos.quantity * pos.current_price for pos in self.positions.values()
            )

            # Update P&L metrics
            self.total_pnl = self.total_value - self.initial_capital
            self.total_pnl_pct = self.total_pnl / self.initial_capital

            # Update peak and drawdown
            if self.total_value > self.peak_value:
                self.peak_value = self.total_value

            current_drawdown = (self.peak_value - self.total_value) / self.peak_value
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

        except Exception as e:
            logger.error(f"Failed to update portfolio value: {e}")

    def get_position_size(self, symbol: str, price: float,
                         confidence: float = 0.5, volatility: float = None) -> float:
        """Calculate optimal position size based on risk management"""
        try:
            # Base position size as percentage of portfolio
            base_position_pct = self.max_position_size_pct

            # Adjust based on confidence
            confidence_multiplier = min(confidence * 2, 1.0)  # Scale 0.5-1.0 to 0.0-1.0

            # Adjust based on volatility (if provided)
            volatility_multiplier = 1.0
            if volatility is not None:
                # Reduce position size for high volatility
                volatility_multiplier = max(0.5, 1.0 - (volatility - 0.02) * 10)  # Assume 2% base volatility

            # Calculate position size
            adjusted_position_pct = base_position_pct * confidence_multiplier * volatility_multiplier

            # Position value in dollars
            position_value = self.total_value * adjusted_position_pct

            # Convert to quantity
            quantity = position_value / price

            # Check portfolio constraints
            current_exposure = self.get_current_exposure()
            max_additional_exposure = max(0, self.max_total_exposure_pct - current_exposure)

            if adjusted_position_pct > max_additional_exposure:
                adjusted_position_pct = max_additional_exposure
                quantity = (self.total_value * adjusted_position_pct) / price

            logger.debug(f"Position size for {symbol}: {quantity:.6f} "
                        f"({adjusted_position_pct:.1%} of portfolio)")

            return max(0, quantity)

        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0.0

    def get_current_exposure(self) -> float:
        """Get current portfolio exposure as percentage"""
        try:
            if self.total_value == 0:
                return 0.0

            total_position_value = sum(
                abs(pos.quantity * pos.current_price)
                for pos in self.positions.values()
            )

            return total_position_value / self.total_value

        except Exception as e:
            logger.error(f"Exposure calculation failed: {e}")
            return 0.0

    def can_open_position(self, symbol: str, side: PositionSide,
                         quantity: float, price: float) -> Tuple[bool, str]:
        """Check if position can be opened"""
        try:
            # Check if position already exists
            if symbol in self.positions:
                existing_pos = self.positions[symbol]
                if existing_pos.status == PositionStatus.OPEN:
                    return False, f"Position already open for {symbol}"

            # Check position value
            position_value = quantity * price

            # Check if we have enough cash (for long positions)
            if side == PositionSide.LONG and position_value > self.cash:
                return False, f"Insufficient cash: ${self.cash:.2f} < ${position_value:.2f}"

            # Check maximum position size
            position_pct = position_value / self.total_value
            if position_pct > self.max_position_size_pct:
                return False, f"Position too large: {position_pct:.1%} > {self.max_position_size_pct:.1%}"

            # Check total exposure
            new_exposure = self.get_current_exposure() + position_pct
            if new_exposure > self.max_total_exposure_pct:
                return False, f"Total exposure too high: {new_exposure:.1%} > {self.max_total_exposure_pct:.1%}"

            return True, "OK"

        except Exception as e:
            logger.error(f"Position validation failed: {e}")
            return False, str(e)

class PositionManager:
    """Main position management system"""

    def __init__(self, exchange_manager: ExchangeManager = None):
        self.exchange_manager = exchange_manager
        self.portfolio = Portfolio()

        # Position tracking
        self.pending_orders: Dict[str, Dict[str, Any]] = {}

        # Risk management settings
        self.stop_loss_pct = config.get('STOP_LOSS_PCT', 0.02)  # 2%
        self.take_profit_pct = config.get('TAKE_PROFIT_PCT', 0.04)  # 4%
        self.trailing_stop_pct = config.get('TRAILING_STOP_PCT', 0.01)  # 1%

        logger.info("Position manager initialized")

    @timing_decorator

    async def open_position(self, symbol: str, side: PositionSide,
                          signal_confidence: float, current_price: float,
                          signal_source: str = "unknown",
                          custom_quantity: float = None) -> Dict[str, Any]:
        """Open a new position"""
        try:
            logger.info(f"Opening {side.value} position for {symbol}")

            # Calculate position size
            if custom_quantity:
                quantity = custom_quantity
            else:
                quantity = self.portfolio.get_position_size(
                    symbol, current_price, signal_confidence
                )

            if quantity <= 0:
                return {
                    'success': False,
                    'error': 'Invalid position size calculated'
                }

            # Validate position
            can_open, validation_error = self.portfolio.can_open_position(
                symbol, side, quantity, current_price
            )

            if not can_open:
                return {
                    'success': False,
                    'error': validation_error
                }

            # Calculate stop loss and take profit
            if side == PositionSide.LONG:
                stop_loss = current_price * (1 - self.stop_loss_pct)
                take_profit = current_price * (1 + self.take_profit_pct)
            else:  # SHORT
                stop_loss = current_price * (1 + self.stop_loss_pct)
                take_profit = current_price * (1 - self.take_profit_pct)

            # Create position object
            position = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=current_price,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal_source=signal_source,
                confidence=signal_confidence,
                max_loss_pct=self.stop_loss_pct
            )

            # Execute order (if exchange manager available)
            order_result = None
            if self.exchange_manager and config.get('ENABLE_LIVE_TRADING', False):
                order_side = 'buy' if side == PositionSide.LONG else 'sell'

                order_result = await self.exchange_manager.place_order(
                    symbol=symbol,
                    side=order_side,
                    amount=quantity,
                    order_type='market'
                )

                if order_result.get('status') == 'failed':
                    return {
                        'success': False,
                        'error': f"Order execution failed: {order_result.get('error', 'Unknown')}"
                    }

                position.entry_order_id = order_result.get('id')

                # Update entry price with actual execution price
                if order_result.get('price'):
                    position.entry_price = float(order_result['price'])

            # Add position to portfolio
            position.update_current_price(current_price)
            self.portfolio.positions[symbol] = position

            # Update cash (for long positions)
            if side == PositionSide.LONG:
                self.portfolio.cash -= quantity * position.entry_price

            # Log position opening
            trading_logger = get_trading_logger(symbol, "position", "latest")
            trading_logger.log_position_update({
                'action': 'open',
                'position': position.to_dict(),
                'order_result': order_result
            })

            logger.info(f"Position opened: {symbol} {side.value} {quantity:.6f} @ {position.entry_price:.2f}")

            return {
                'success': True,
                'position': position,
                'order_result': order_result
            }

        except Exception as e:
            logger.error(f"Failed to open position for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def close_position(self, symbol: str, reason: str = "Manual close") -> Dict[str, Any]:
        """Close an existing position"""
        try:
            if symbol not in self.portfolio.positions:
                return {
                    'success': False,
                    'error': f'No open position for {symbol}'
                }

            position = self.portfolio.positions[symbol]

            if position.status != PositionStatus.OPEN:
                return {
                    'success': False,
                    'error': f'Position for {symbol} is not open'
                }

            logger.info(f"Closing position for {symbol}: {reason}")

            # Execute close order
            order_result = None
            if self.exchange_manager and config.get('ENABLE_LIVE_TRADING', False):
                # Opposite side for closing
                order_side = 'sell' if position.side == PositionSide.LONG else 'buy'

                order_result = await self.exchange_manager.place_order(
                    symbol=symbol,
                    side=order_side,
                    amount=abs(position.quantity),
                    order_type='market'
                )

                if order_result.get('status') == 'failed':
                    return {
                        'success': False,
                        'error': f"Close order failed: {order_result.get('error', 'Unknown')}"
                    }

                position.exit_order_id = order_result.get('id')
                exit_price = float(order_result.get('price', position.current_price))
            else:
                exit_price = position.current_price

            # Update position with exit details
            position.exit_price = exit_price
            position.exit_time = datetime.now()
            position.status = PositionStatus.CLOSED

            # Calculate realized P&L
            if position.side == PositionSide.LONG:
                position.realized_pnl = (exit_price - position.entry_price) * position.quantity
                cash_return = position.quantity * exit_price
            else:  # SHORT
                position.realized_pnl = (position.entry_price - exit_price) * position.quantity
                cash_return = position.quantity * (2 * position.entry_price - exit_price)

            # Update portfolio
            self.portfolio.cash += cash_return

            # Move to closed positions
            self.portfolio.closed_positions.append(position)
            del self.portfolio.positions[symbol]

            # Log position closing
            trading_logger = get_trading_logger(symbol, "position", "latest")
            trading_logger.log_position_update({
                'action': 'close',
                'reason': reason,
                'position': position.to_dict(),
                'order_result': order_result,
                'realized_pnl': position.realized_pnl
            })

            logger.info(f"Position closed: {symbol} P&L: ${position.realized_pnl:.2f}")

            return {
                'success': True,
                'position': position,
                'realized_pnl': position.realized_pnl,
                'order_result': order_result
            }

        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def update_positions(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """Update all positions with current prices"""
        try:
            # Update portfolio value
            self.portfolio.update_portfolio_value(prices)

            positions_to_close = []

            # Check each position for risk management rules
            for symbol, position in self.portfolio.positions.items():
                if symbol in prices:
                    position.update_current_price(prices[symbol])

                    # Check if position should be closed
                    should_close, close_reason = position.should_close_position()

                    if should_close:
                        positions_to_close.append((symbol, close_reason))

            # Close positions that triggered risk rules
            close_results = []
            for symbol, reason in positions_to_close:
                result = await self.close_position(symbol, reason)
                close_results.append({
                    'symbol': symbol,
                    'reason': reason,
                    'result': result
                })

            return {
                'portfolio_value': self.portfolio.total_value,
                'total_pnl': self.portfolio.total_pnl,
                'positions_closed': len(close_results),
                'close_results': close_results
            }

        except Exception as e:
            logger.error(f"Failed to update positions: {e}")
            return {}

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            # Position summaries
            open_positions = []
            for position in self.portfolio.positions.values():
                open_positions.append({
                    'symbol': position.symbol,
                    'side': position.side.value,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'unrealized_pnl_pct': position.unrealized_pnl_pct,
                    'confidence': position.confidence
                })

            # Recent closed positions
            recent_closed = []
            for position in self.portfolio.closed_positions[-10:]:  # Last 10
                recent_closed.append({
                    'symbol': position.symbol,
                    'side': position.side.value,
                    'realized_pnl': position.realized_pnl,
                    'entry_time': position.entry_time.isoformat(),
                    'exit_time': position.exit_time.isoformat() if position.exit_time else None,
                    'hold_duration': (position.exit_time - position.entry_time).total_seconds() / 3600 if position.exit_time else None  # hours
                })

            # Performance metrics
            total_trades = len(self.portfolio.closed_positions)
            winning_trades = len([p for p in self.portfolio.closed_positions if p.realized_pnl > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            avg_win = np.mean([p.realized_pnl for p in self.portfolio.closed_positions if p.realized_pnl > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([p.realized_pnl for p in self.portfolio.closed_positions if p.realized_pnl < 0]) if (total_trades - winning_trades) > 0 else 0

            return {
                'portfolio_value': self.portfolio.total_value,
                'cash': self.portfolio.cash,
                'total_pnl': self.portfolio.total_pnl,
                'total_pnl_pct': self.portfolio.total_pnl_pct,
                'max_drawdown': self.portfolio.max_drawdown,
                'current_exposure': self.portfolio.get_current_exposure(),
                'open_positions_count': len(self.portfolio.positions),
                'open_positions': open_positions,
                'recent_closed_positions': recent_closed,
                'performance': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                }
            }

        except Exception as e:
            logger.error(f"Failed to generate portfolio summary: {e}")
            return {}

    def get_position_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific position"""
        if symbol not in self.portfolio.positions:
            return None

        position = self.portfolio.positions[symbol]
        return position.to_dict()

    async def close_all_positions(self, reason: str = "Close all") -> Dict[str, Any]:
        """Close all open positions"""
        try:
            symbols = list(self.portfolio.positions.keys())
            results = []

            for symbol in symbols:
                result = await self.close_position(symbol, reason)
                results.append({
                    'symbol': symbol,
                    'result': result
                })

            logger.info(f"Closed {len(symbols)} positions")

            return {
                'positions_closed': len(symbols),
                'results': results
            }

        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return {'error': str(e)}

    def set_risk_parameters(self, stop_loss_pct: float = None,
                          take_profit_pct: float = None,
                          max_position_size_pct: float = None):
        """Update risk management parameters"""
        try:
            if stop_loss_pct is not None:
                self.stop_loss_pct = stop_loss_pct
                logger.info(f"Stop loss updated to {stop_loss_pct:.1%}")

            if take_profit_pct is not None:
                self.take_profit_pct = take_profit_pct
                logger.info(f"Take profit updated to {take_profit_pct:.1%}")

            if max_position_size_pct is not None:
                self.portfolio.max_position_size_pct = max_position_size_pct
                logger.info(f"Max position size updated to {max_position_size_pct:.1%}")

        except Exception as e:
            logger.error(f"Failed to update risk parameters: {e}")

# Convenience functions

async def create_position_manager(exchange_manager: ExchangeManager = None,
                                initial_capital: float = 10000) -> PositionManager:
    """Create position manager with specified parameters"""
    manager = PositionManager(exchange_manager)
    manager.portfolio.initial_capital = initial_capital
    manager.portfolio.cash = initial_capital
    manager.portfolio.total_value = initial_capital
    return manager

# Usage example
if __name__ == "__main__":

    async def test_position_manager():
        # Create position manager
        manager = PositionManager()

        # Open a test position
        result = await manager.open_position(
            symbol='BTCUSDT',
            side=PositionSide.LONG,
            signal_confidence=0.8,
            current_price=45000.0,
            signal_source='test'
        )

        print(f"Position opened: {result['success']}")

        if result['success']:
            # Update with new price
            update_result = await manager.update_positions({'BTCUSDT': 46000.0})
            print(f"Portfolio value: ${update_result.get('portfolio_value', 0):,.2f}")

            # Get portfolio summary
            summary = manager.get_portfolio_summary()
            print(f"Open positions: {summary.get('open_positions_count', 0)}")
            print(f"Total P&L: ${summary.get('total_pnl', 0):,.2f}")

            # Close position
            close_result = await manager.close_position('BTCUSDT', 'Test close')
            print(f"Position closed: {close_result['success']}")

            if close_result['success']:
                print(f"Realized P&L: ${close_result['realized_pnl']:,.2f}")

        print("Position manager test completed!")

    # Run test
    asyncio.run(test_position_manager())
