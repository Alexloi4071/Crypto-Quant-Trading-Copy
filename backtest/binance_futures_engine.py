# -*- coding: utf-8 -*-
"""
Binance合约回测引擎
支持多空双向、杠杆、止损止盈
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging


class PositionSide(Enum):
    """仓位方向"""
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class Trade:
    """交易记录"""
    timestamp: pd.Timestamp
    action: str  # 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT'
    price: float
    quantity: float
    commission: float
    realized_pnl: float = 0.0
    equity: float = 0.0
    note: str = ""


@dataclass
class Position:
    """持仓状态"""
    side: PositionSide = PositionSide.NONE
    entry_price: float = 0.0
    quantity: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    leverage: int = 1
    
    def unrealized_pnl(self, current_price: float) -> float:
        """计算未实现盈亏"""
        if self.side == PositionSide.NONE:
            return 0.0
        elif self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - current_price) * self.quantity


class BinanceFuturesBacktest:
    """Binance合约回测引擎"""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 3,
        commission_rate: float = 0.0004,  # Taker 0.04%
        slippage_rate: float = 0.0005,    # 滑点 0.05%
        min_trade_usdt: float = 10.0
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.min_trade_usdt = min_trade_usdt
        
        # 状态追踪
        self.capital = initial_capital
        self.position = Position(leverage=leverage)
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[pd.Timestamp] = []
        self.max_equity = initial_capital
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(
        self,
        price: float,
        risk_pct: float = 0.02
    ) -> float:
        """
        计算仓位大小（基于风险百分比）
        
        Args:
            price: 当前价格
            risk_pct: 单笔风险百分比（默认2%）
        
        Returns:
            仓位数量
        """
        # 可用资金 = 当前资金 * 杠杆
        available = self.capital * self.leverage
        
        # 风险金额
        risk_amount = self.capital * risk_pct
        
        # 假设止损2%，计算仓位大小
        stop_loss_pct = 0.02
        position_value = risk_amount / stop_loss_pct
        
        # 限制最大仓位
        max_position = available * 0.95
        position_value = min(position_value, max_position)
        
        # 转换为数量
        quantity = position_value / price
        
        return quantity
    
    def execute_signal(
        self,
        timestamp: pd.Timestamp,
        signal: int,  # 0: 卖出, 1: 持有, 2: 买入
        price: float,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04
    ) -> Optional[Trade]:
        """执行交易信号"""
        
        # 记录权益
        current_equity = self.calculate_equity(price)
        self.equity_curve.append(current_equity)
        self.timestamps.append(timestamp)
        
        # 更新最大权益
        self.max_equity = max(self.max_equity, current_equity)
        
        # 检查止损止盈
        if self.position.side != PositionSide.NONE:
            sl_tp_trade = self._check_stop_loss_take_profit(
                timestamp, price, stop_loss_pct, take_profit_pct
            )
            if sl_tp_trade:
                return sl_tp_trade
        
        # 执行信号
        if signal == 2:  # 买入
            return self._open_long(timestamp, price)
        elif signal == 0:  # 卖出
            return self._open_short(timestamp, price)
        else:  # 持有
            return None
    
    def _open_long(self, timestamp: pd.Timestamp, price: float) -> Optional[Trade]:
        """开多仓"""
        # 如果有空仓，先平仓
        if self.position.side == PositionSide.SHORT:
            self._close_position(timestamp, price, "REVERSE_TO_LONG")
        
        # 如果已有多仓，不重复开
        if self.position.side == PositionSide.LONG:
            return None
        
        # 计算仓位
        quantity = self.calculate_position_size(price)
        position_value = quantity * price
        
        if position_value < self.min_trade_usdt:
            return None
        
        # 计算滑点和手续费
        slippage = price * self.slippage_rate
        entry_price = price + slippage
        commission = position_value * self.commission_rate
        
        # 扣除手续费
        self.capital -= commission
        
        # 更新持仓
        self.position = Position(
            side=PositionSide.LONG,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=timestamp,
            leverage=self.leverage
        )
        
        # 记录交易
        trade = Trade(
            timestamp=timestamp,
            action='OPEN_LONG',
            price=entry_price,
            quantity=quantity,
            commission=commission,
            equity=self.capital,
            note=f"开多 {quantity:.4f} @ {entry_price:.2f}"
        )
        
        self.trades.append(trade)
        self.logger.debug(f"开多: {trade.note}")
        
        return trade
    
    def _open_short(self, timestamp: pd.Timestamp, price: float) -> Optional[Trade]:
        """开空仓"""
        # 如果有多仓，先平仓
        if self.position.side == PositionSide.LONG:
            self._close_position(timestamp, price, "REVERSE_TO_SHORT")
        
        # 如果已有空仓，不重复开
        if self.position.side == PositionSide.SHORT:
            return None
        
        # 计算仓位
        quantity = self.calculate_position_size(price)
        position_value = quantity * price
        
        if position_value < self.min_trade_usdt:
            return None
        
        # 计算滑点和手续费
        slippage = price * self.slippage_rate
        entry_price = price - slippage
        commission = position_value * self.commission_rate
        
        # 扣除手续费
        self.capital -= commission
        
        # 更新持仓
        self.position = Position(
            side=PositionSide.SHORT,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=timestamp,
            leverage=self.leverage
        )
        
        # 记录交易
        trade = Trade(
            timestamp=timestamp,
            action='OPEN_SHORT',
            price=entry_price,
            quantity=quantity,
            commission=commission,
            equity=self.capital,
            note=f"开空 {quantity:.4f} @ {entry_price:.2f}"
        )
        
        self.trades.append(trade)
        self.logger.debug(f"开空: {trade.note}")
        
        return trade
    
    def _close_position(
        self,
        timestamp: pd.Timestamp,
        price: float,
        reason: str = "SIGNAL"
    ) -> Optional[Trade]:
        """平仓"""
        if self.position.side == PositionSide.NONE:
            return None
        
        # 计算滑点
        if self.position.side == PositionSide.LONG:
            slippage = price * self.slippage_rate
            exit_price = price - slippage
            pnl = (exit_price - self.position.entry_price) * self.position.quantity
            action = 'CLOSE_LONG'
        else:  # SHORT
            slippage = price * self.slippage_rate
            exit_price = price + slippage
            pnl = (self.position.entry_price - exit_price) * self.position.quantity
            action = 'CLOSE_SHORT'
        
        # 计算手续费
        position_value = self.position.quantity * exit_price
        commission = position_value * self.commission_rate
        
        # 净盈亏
        net_pnl = pnl - commission
        
        # 更新资金
        self.capital += net_pnl
        
        # 记录交易
        trade = Trade(
            timestamp=timestamp,
            action=action,
            price=exit_price,
            quantity=self.position.quantity,
            commission=commission,
            realized_pnl=net_pnl,
            equity=self.capital,
            note=f"平仓({reason}) PnL={net_pnl:.2f}"
        )
        
        self.trades.append(trade)
        self.logger.debug(f"平仓: {trade.note}")
        
        # 重置持仓
        self.position = Position(leverage=self.leverage)
        
        return trade
    
    def _check_stop_loss_take_profit(
        self,
        timestamp: pd.Timestamp,
        price: float,
        stop_loss_pct: float,
        take_profit_pct: float
    ) -> Optional[Trade]:
        """检查止损止盈"""
        if self.position.side == PositionSide.NONE:
            return None
        
        # 计算盈亏比例
        if self.position.side == PositionSide.LONG:
            pnl_pct = (price - self.position.entry_price) / self.position.entry_price
        else:
            pnl_pct = (self.position.entry_price - price) / self.position.entry_price
        
        # 止损
        if pnl_pct <= -stop_loss_pct:
            return self._close_position(timestamp, price, "STOP_LOSS")
        
        # 止盈
        if pnl_pct >= take_profit_pct:
            return self._close_position(timestamp, price, "TAKE_PROFIT")
        
        return None
    
    def calculate_equity(self, current_price: float) -> float:
        """计算当前权益（含未实现盈亏）"""
        unrealized_pnl = self.position.unrealized_pnl(current_price)
        return self.capital + unrealized_pnl
    
    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04
    ) -> Dict:
        """
        运行回测
        
        Args:
            data: OHLCV数据，必须包含'close'列
            signals: 交易信号序列 (0/1/2)
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
        
        Returns:
            回测结果字典
        """
        self.logger.info(f"开始回测: {len(data)}根K线, 初始资金{self.initial_capital:.2f}")
        
        for idx in data.index:
            if idx not in signals.index:
                signal = 1  # 默认持有
            else:
                signal = int(signals.loc[idx])
            
            self.execute_signal(
                timestamp=idx,
                signal=signal,
                price=data.loc[idx, 'close'],
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct
            )
        
        # 最后强制平仓
        if self.position.side != PositionSide.NONE:
            last_price = data.iloc[-1]['close']
            self._close_position(data.index[-1], last_price, "END_OF_BACKTEST")
        
        # 计算绩效指标
        metrics = self._calculate_metrics()
        
        self.logger.info(f"回测完成: 总收益{metrics['total_return']:.2%}, 夏普{metrics['sharpe_ratio']:.2f}")
        
        return metrics
    
    def _calculate_metrics(self) -> Dict:
        """计算绩效指标"""
        if len(self.equity_curve) == 0:
            return self._empty_metrics()
        
        equity_series = pd.Series(self.equity_curve, index=self.timestamps)
        
        # 总收益率
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # 年化收益率
        if len(self.timestamps) > 1:
            days = (self.timestamps[-1] - self.timestamps[0]).days
            if days > 0:
                annual_return = (1 + total_return) ** (365 / days) - 1
            else:
                annual_return = 0
        else:
            annual_return = 0
        
        # 最大回撤
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # 夏普比率（假设无风险利率为0）
        returns = equity_series.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            # 假设15分钟数据，年化因子 = sqrt(365*24*4)
            periods_per_year = 365 * 24 * 4  # 15分钟
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0
        
        # 交易统计
        realized_trades = [t for t in self.trades if t.realized_pnl != 0]
        total_trades = len(realized_trades)
        
        if total_trades > 0:
            winning_trades = [t for t in realized_trades if t.realized_pnl > 0]
            losing_trades = [t for t in realized_trades if t.realized_pnl < 0]
            
            win_rate = len(winning_trades) / total_trades
            
            avg_win = np.mean([t.realized_pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t.realized_pnl) for t in losing_trades]) if losing_trades else 1
            
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
        
        # 总手续费
        total_commission = sum([t.commission for t in self.trades])
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'total_commission': total_commission,
            'equity_curve': equity_series,
            'trades': self.trades
        }
    
    def _empty_metrics(self) -> Dict:
        """空指标（无交易时）"""
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': 0.0,
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'total_commission': 0.0,
            'equity_curve': pd.Series([self.initial_capital]),
            'trades': []
        }
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """生成回测报告"""
        metrics = self._calculate_metrics()
        
        lines = [
            "# Binance合约回测报告\n",
            f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "\n## 📊 绩效摘要\n",
            f"- 初始资金: ${metrics['initial_capital']:,.2f}",
            f"- 最终资金: ${metrics['final_capital']:,.2f}",
            f"- 总收益率: {metrics['total_return']:.2%}",
            f"- 年化收益: {metrics['annual_return']:.2%}",
            f"- 最大回撤: {metrics['max_drawdown']:.2%}",
            f"- 夏普比率: {metrics['sharpe_ratio']:.2f}",
            f"- 胜率: {metrics['win_rate']:.2%}",
            f"- 盈亏比: {metrics['profit_factor']:.2f}",
            f"- 总交易次数: {metrics['total_trades']}",
            f"- 总手续费: ${metrics['total_commission']:.2f}\n",
            "\n## 📈 交易明细\n"
        ]
        
        # 显示最近10笔交易
        for trade in self.trades[-10:]:
            lines.append(
                f"- {trade.timestamp.strftime('%Y-%m-%d %H:%M')}: "
                f"{trade.action} @{trade.price:.2f} "
                f"PnL={trade.realized_pnl:.2f}\n"
            )
        
        report = '\n'.join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"报告已保存: {output_file}")
        
        return report


# 使用示例
if __name__ == "__main__":
    # 模拟数据
    dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
    
    # 生成模拟价格（趋势+噪声）
    trend = np.linspace(45000, 48000, 1000)
    noise = np.random.randn(1000) * 200
    prices = trend + noise
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.rand(1000) * 1000000
    }, index=dates)
    
    # 生成简单信号（移动平均金叉死叉）
    ma_short = data['close'].rolling(20).mean()
    ma_long = data['close'].rolling(50).mean()
    
    signals = pd.Series(1, index=data.index)  # 默认持有
    signals[ma_short > ma_long * 1.01] = 2    # 买入
    signals[ma_short < ma_long * 0.99] = 0    # 卖出
    
    # 运行回测
    backtest = BinanceFuturesBacktest(
        initial_capital=10000,
        leverage=3,
        commission_rate=0.0004
    )
    
    results = backtest.run(data, signals)
    
    # 打印结果
    print("\n回测结果:")
    print(f"总收益率: {results['total_return']:.2%}")
    print(f"年化收益: {results['annual_return']:.2%}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"胜率: {results['win_rate']:.2%}")
    print(f"总交易: {results['total_trades']}次")
    
    # 生成报告
    report = backtest.generate_report('backtest_report.md')
    print("\n回测引擎测试完成！")
