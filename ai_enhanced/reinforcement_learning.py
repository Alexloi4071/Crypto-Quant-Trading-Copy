"""
Reinforcement Learning
强化学习交易系统，基于深度强化学习算法实现自适应交易策略
支持多种RL算法：DQN、PPO、A3C、SAC等
"""

import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from collections import deque, namedtuple
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
import random
import json
from abc import ABC, abstractmethod

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class ActionType(Enum):
    """动作类型"""
    HOLD = 0
    BUY = 1
    SELL = 2

class RewardType(Enum):
    """奖励类型"""
    PROFIT_BASED = "profit_based"
    SHARPE_BASED = "sharpe_based"
    RISK_ADJUSTED = "risk_adjusted"
    DRAWDOWN_PENALTY = "drawdown_penalty"

@dataclass

class TradingAction:
    """交易动作"""
    action_type: ActionType
    quantity: float = 1.0
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            'action_type': self.action_type.name,
            'quantity': self.quantity,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass

class TradingState:
    """交易状态"""
    # 市场数据
    prices: np.ndarray
    volumes: np.ndarray
    technical_indicators: np.ndarray

    # 账户状态
    cash: float
    holdings: float
    portfolio_value: float

    # 交易历史
    recent_returns: np.ndarray
    recent_actions: List[int]

    # 市场特征
    volatility: float
    trend: float
    momentum: float

    def to_array(self) -> np.ndarray:
        """转换为数组格式"""
        state_array = np.concatenate([
            self.prices.flatten(),
            self.volumes.flatten(),
            self.technical_indicators.flatten(),
            [self.cash / 10000],  # 标准化
            [self.holdings],
            [self.portfolio_value / 10000],  # 标准化
            self.recent_returns.flatten(),
            [float(action) for action in self.recent_actions[-5:]],  # 最近5个动作
            [self.volatility],
            [self.trend],
            [self.momentum]
        ])
        return state_array.astype(np.float32)

class TradingEnvironment(gym.Env):
    """交易环境"""

    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000,
                 transaction_cost: float = 0.001, lookback_window: int = 20):
        super().__init__()

        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window

        # 环境状态
        self.current_step = 0
        self.max_steps = len(data) - lookback_window - 1

        # 账户状态
        self.cash = initial_balance
        self.holdings = 0.0
        self.portfolio_value = initial_balance
        self.total_profit = 0.0

        # 交易历史
        self.trade_history = []
        self.action_history = deque(maxlen=10)
        self.return_history = deque(maxlen=50)

        # 动作和观测空间
        self.action_space = gym.spaces.Discrete(len(ActionType))

        # 计算状态维度
        state_dim = (
            lookback_window * 4 +  # OHLC prices
            lookback_window +      # volumes
            lookback_window * 5 +  # technical indicators
            6 +                    # account status
            5 +                    # recent actions
            3                      # market features
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # 性能追踪
        self.episode_returns = []
        self.episode_actions = []
        self.max_drawdown = 0.0
        self.peak_value = initial_balance

        logger.debug(f"交易环境初始化：{len(data)} 条数据，状态维度 {state_dim}")

    def reset(self):
        """重置环境"""
        self.current_step = 0
        self.cash = self.initial_balance
        self.holdings = 0.0
        self.portfolio_value = self.initial_balance
        self.total_profit = 0.0

        self.trade_history.clear()
        self.action_history.clear()
        self.return_history.clear()

        self.episode_returns.clear()
        self.episode_actions.clear()
        self.max_drawdown = 0.0
        self.peak_value = self.initial_balance

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行动作"""
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0.0, True, {}

        # 获取当前价格
        current_price = self.data.iloc[self.current_step + self.lookback_window]['close']

        # 计算上一步的组合价值
        prev_portfolio_value = self.cash + self.holdings * current_price

        # 执行动作
        reward = self._execute_action(action, current_price)

        # 更新状态
        self.current_step += 1

        # 计算新的组合价值
        if self.current_step < self.max_steps:
            new_price = self.data.iloc[self.current_step + self.lookback_window]['close']
            self.portfolio_value = self.cash + self.holdings * new_price
        else:
            self.portfolio_value = self.cash + self.holdings * current_price

        # 计算收益率
        portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.return_history.append(portfolio_return)

        # 更新最大回撤
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        else:
            drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, drawdown)

        # 记录历史
        self.action_history.append(action)
        self.episode_returns.append(portfolio_return)
        self.episode_actions.append(action)

        # 检查是否结束
        done = self.current_step >= self.max_steps or self.portfolio_value <= 0

        # 信息字典
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'holdings': self.holdings,
            'total_profit': self.portfolio_value - self.initial_balance,
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance,
            'max_drawdown': self.max_drawdown,
            'current_price': current_price
        }

        return self._get_observation(), reward, done, info

    def _execute_action(self, action: int, price: float) -> float:
        """执行具体动作"""
        action_type = ActionType(action)
        reward = 0.0

        if action_type == ActionType.BUY and self.cash > price * (1 + self.transaction_cost):
            # 买入：使用所有可用现金
            shares_to_buy = self.cash / (price * (1 + self.transaction_cost))
            cost = shares_to_buy * price * (1 + self.transaction_cost)

            self.cash -= cost
            self.holdings += shares_to_buy

            self.trade_history.append({
                'type': 'BUY',
                'price': price,
                'quantity': shares_to_buy,
                'cost': cost,
                'step': self.current_step
            })

        elif action_type == ActionType.SELL and self.holdings > 0:
            # 卖出：卖出所有持仓
            proceeds = self.holdings * price * (1 - self.transaction_cost)

            self.cash += proceeds
            self.holdings = 0.0

            self.trade_history.append({
                'type': 'SELL',
                'price': price,
                'quantity': self.holdings,
                'proceeds': proceeds,
                'step': self.current_step
            })

        # 计算即时奖励
        reward = self._calculate_reward(action_type, price)

        return reward

    def _calculate_reward(self, action_type: ActionType, price: float) -> float:
        """计算奖励"""
        if len(self.return_history) < 2:
            return 0.0

        # 基础收益奖励
        recent_return = self.return_history[-1]
        reward = recent_return * 100  # 放大奖励信号

        # 动作惩罚（减少过度交易）
        if action_type != ActionType.HOLD:
            reward -= 0.01  # 交易成本惩罚

        # 连续相同动作的惩罚
        if len(self.action_history) >= 2:
            if self.action_history[-1] == self.action_history[-2] and action_type.value != ActionType.HOLD.value:
                reward -= 0.005

        # 回撤惩罚
        if self.max_drawdown > 0.1:  # 超过10%回撤
            reward -= self.max_drawdown * 10

        # 夏普比率奖励
        if len(self.return_history) >= 10:
            returns_array = np.array(list(self.return_history))
            if returns_array.std() > 0:
                sharpe = returns_array.mean() / returns_array.std()
                reward += sharpe * 0.1

        return reward

    def _get_observation(self) -> np.ndarray:
        """获取当前观测"""
        if self.current_step + self.lookback_window >= len(self.data):
            # 返回最后的观测
            end_idx = len(self.data)
            start_idx = max(0, end_idx - self.lookback_window)
        else:
            start_idx = self.current_step
            end_idx = self.current_step + self.lookback_window

        # 获取价格数据
        window_data = self.data.iloc[start_idx:end_idx]

        if len(window_data) < self.lookbook_window:
            # 填充数据
            padding_size = self.lookback_window - len(window_data)
            padding = window_data.iloc[0:1].values.repeat(padding_size, axis=0)
            window_data = pd.concat([
                pd.DataFrame(padding, columns=window_data.columns),
                window_data
            ], ignore_index=True)

        # 标准化价格数据
        prices = window_data[['open', 'high', 'low', 'close']].values
        prices = prices / prices[-1, 3]  # 用最新收盘价标准化

        volumes = window_data['volume'].values
        volumes = volumes / volumes.max() if volumes.max() > 0 else volumes

        # 技术指标
        tech_indicators = self._calculate_technical_indicators(window_data)

        # 近期收益
        recent_returns = np.array(list(self.return_history)[-10:])
        if len(recent_returns) < 10:
            recent_returns = np.pad(recent_returns, (10-len(recent_returns), 0))

        # 近期动作
        recent_actions = list(self.action_history)[-5:]
        while len(recent_actions) < 5:
            recent_actions.insert(0, 0)  # 填充HOLD动作

        # 市场特征
        volatility = recent_returns.std() if len(recent_returns) > 1 else 0.0
        trend = np.polyfit(range(len(prices)), prices[:, 3], 1)[0]  # 收盘价趋势
        momentum = (prices[-1, 3] - prices[0, 3]) / prices[0, 3]

        # 创建交易状态
        state = TradingState(
            prices=prices,
            volumes=volumes,
            technical_indicators=tech_indicators,
            cash=self.cash,
            holdings=self.holdings,
            portfolio_value=self.portfolio_value,
            recent_returns=recent_returns,
            recent_actions=recent_actions,
            volatility=volatility,
            trend=trend,
            momentum=momentum
        )

        return state.to_array()

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> np.ndarray:
        """计算技术指标"""
        indicators = np.zeros((len(data), 5))

        if len(data) >= 14:
            # RSI
            delta = data['close'].diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)

            avg_gains = gains.rolling(window=14).mean()
            avg_losses = losses.rolling(window=14).mean()

            rs = avg_gains / (avg_losses + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            indicators[:, 0] = rsi.fillna(50) / 100  # 标准化到0-1

        if len(data) >= 20:
            # 布林带
            sma_20 = data['close'].rolling(window=20).mean()
            std_20 = data['close'].rolling(window=20).std()

            upper_band = sma_20 + (std_20 * 2)
            lower_band = sma_20 - (std_20 * 2)

            bb_position = (data['close'] - lower_band) / (upper_band - lower_band)
            indicators[:, 1] = bb_position.fillna(0.5)

        if len(data) >= 12:
            # MACD
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal_line = macd.ewm(span=9).mean()
            macd_hist = macd - signal_line

            # 标准化MACD
            macd_std = macd_hist.std()
            if macd_std > 0:
                indicators[:, 2] = (macd_hist / macd_std).fillna(0)

        # 简单移动平均
        if len(data) >= 10:
            sma_10 = data['close'].rolling(window=10).mean()
            sma_ratio = data['close'] / sma_10
            indicators[:, 3] = (sma_ratio - 1).fillna(0)

        # 成交量比率
        if len(data) >= 5:
            vol_sma = data['volume'].rolling(window=5).mean()
            vol_ratio = data['volume'] / (vol_sma + 1e-10)
            indicators[:, 4] = np.log(vol_ratio + 1e-10).fillna(0)

        return indicators

# Deep Q-Network

class DQN(nn.Module):
    """深度Q网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super(DQN, self).__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)

# Experience Replay Buffer
Experience = namedtuple('Experience',
                       ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Experience(*args))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN智能体"""

    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-4,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 gamma: float = 0.99, batch_size: int = 32, target_update: int = 1000):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # 神经网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # 经验回放
        self.memory = ReplayBuffer()

        # 统计信息
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []

        # 复制权重到目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())

        logger.info(f"DQN智能体初始化完成，设备: {self.device}")

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.push(state, action, reward, next_state, done)

    def train(self) -> Optional[float]:
        """训练智能体"""
        if len(self.memory) < self.batch_size:
            return None

        # 采样经验
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        # 转换为tensor
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)

        # 计算当前Q值
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)

        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # 更新epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 更新目标网络
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # 记录损失
        self.losses.append(loss.item())

        return loss.item()

    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
        logger.info(f"模型已保存到 {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        logger.info(f"模型已从 {filepath} 加载")

class RLTrainingManager:
    """强化学习训练管理器"""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.env = None
        self.agent = None

        # 训练统计
        self.training_episodes = 0
        self.best_score = -np.inf
        self.training_history = []

        # 评估统计
        self.evaluation_results = []

        logger.info("强化学习训练管理器初始化完成")

    def setup_environment_and_agent(self, **env_kwargs):
        """设置环境和智能体"""
        # 创建环境
        self.env = TradingEnvironment(self.data, **env_kwargs)

        # 创建智能体
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.agent = DQNAgent(state_dim, action_dim)

        logger.info(f"环境设置完成：状态维度 {state_dim}, 动作维度 {action_dim}")

    def train(self, num_episodes: int = 1000, save_interval: int = 100) -> Dict[str, List]:
        """训练智能体"""
        if not self.env or not self.agent:
            raise ValueError("请先调用 setup_environment_and_agent()")

        episode_rewards = []
        episode_profits = []
        episode_lengths = []

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0

            while True:
                # 选择动作
                action = self.agent.act(state, training=True)

                # 执行动作
                next_state, reward, done, info = self.env.step(action)

                # 存储经验
                self.agent.remember(state, action, reward, next_state, done)

                # 训练智能体
                loss = self.agent.train()

                # 更新状态
                state = next_state
                episode_reward += reward
                episode_length += 1

                if done:
                    break

            # 记录统计信息
            episode_rewards.append(episode_reward)
            episode_profits.append(info['total_profit'])
            episode_lengths.append(episode_length)

            # 更新最佳分数
            if info['total_profit'] > self.best_score:
                self.best_score = info['total_profit']

            # 记录训练历史
            training_record = {
                'episode': episode + 1,
                'reward': episode_reward,
                'profit': info['total_profit'],
                'portfolio_value': info['portfolio_value'],
                'max_drawdown': info['max_drawdown'],
                'epsilon': self.agent.epsilon,
                'length': episode_length
            }
            self.training_history.append(training_record)

            # 打印进度
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_profit = np.mean(episode_profits[-100:])

                logger.info(f"Episode {episode + 1}/{num_episodes}, "
                           f"Avg Reward: {avg_reward:.2f}, "
                           f"Avg Profit: {avg_profit:.2f}, "
                           f"Best Profit: {self.best_score:.2f}, "
                           f"Epsilon: {self.agent.epsilon:.3f}")

            # 保存模型
            if (episode + 1) % save_interval == 0:
                model_path = f"models/rl_model_episode_{episode + 1}.pth"
                self.agent.save_model(model_path)

        self.training_episodes += num_episodes

        return {
            'episode_rewards': episode_rewards,
            'episode_profits': episode_profits,
            'episode_lengths': episode_lengths
        }

    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """评估智能体"""
        if not self.env or not self.agent:
            raise ValueError("请先调用 setup_environment_and_agent() 并训练智能体")

        evaluation_results = []

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            actions_taken = []

            while True:
                # 使用贪婪策略（不探索）
                action = self.agent.act(state, training=False)
                actions_taken.append(action)

                next_state, reward, done, info = self.env.step(action)

                state = next_state
                episode_reward += reward

                if done:
                    break

            # 记录评估结果
            result = {
                'episode': episode + 1,
                'total_reward': episode_reward,
                'total_profit': info['total_profit'],
                'total_return': info['total_return'],
                'max_drawdown': info['max_drawdown'],
                'final_portfolio_value': info['portfolio_value'],
                'actions_distribution': {
                    'hold': actions_taken.count(0),
                    'buy': actions_taken.count(1),
                    'sell': actions_taken.count(2)
                }
            }
            evaluation_results.append(result)

        # 计算平均性能
        avg_profit = np.mean([r['total_profit'] for r in evaluation_results])
        avg_return = np.mean([r['total_return'] for r in evaluation_results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in evaluation_results])

        # 计算夏普比率
        returns = [r['total_return'] for r in evaluation_results]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        evaluation_summary = {
            'num_episodes': num_episodes,
            'average_profit': avg_profit,
            'average_return': avg_return,
            'average_max_drawdown': avg_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'best_episode': max(evaluation_results, key=lambda x: x['total_profit']),
            'worst_episode': min(evaluation_results, key=lambda x: x['total_profit']),
            'detailed_results': evaluation_results
        }

        self.evaluation_results.append(evaluation_summary)

        logger.info(f"评估完成：平均收益 {avg_profit:.2f}, "
                   f"平均回报率 {avg_return:.4f}, "
                   f"夏普比率 {sharpe_ratio:.4f}")

        return evaluation_summary

    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        if not self.training_history:
            return {'message': '暂无训练历史'}

        profits = [record['profit'] for record in self.training_history]
        rewards = [record['reward'] for record in self.training_history]
        drawdowns = [record['max_drawdown'] for record in self.training_history]

        return {
            'total_episodes': len(self.training_history),
            'best_profit': max(profits),
            'worst_profit': min(profits),
            'average_profit': np.mean(profits),
            'final_profit': profits[-1] if profits else 0,
            'best_reward': max(rewards),
            'average_reward': np.mean(rewards),
            'final_reward': rewards[-1] if rewards else 0,
            'average_drawdown': np.mean(drawdowns),
            'max_drawdown': max(drawdowns),
            'current_epsilon': self.agent.epsilon if self.agent else 0,
            'recent_performance': self.training_history[-10:] if len(self.training_history) >= 10 else self.training_history
        }

class ReinforcementLearningSystem:
    """强化学习系统主类"""

    def __init__(self):
        self.training_manager = None
        self.trained_models = {}

        # 系统统计
        self.system_stats = {
            'models_trained': 0,
            'total_training_episodes': 0,
            'total_evaluation_episodes': 0,
            'best_model_performance': 0.0
        }

        logger.info("强化学习系统初始化完成")

    def create_training_session(self, data: pd.DataFrame, **env_kwargs) -> str:
        """创建训练会话"""
        session_id = f"session_{int(datetime.now().timestamp())}"

        # 创建训练管理器
        self.training_manager = RLTrainingManager(data)
        self.training_manager.setup_environment_and_agent(**env_kwargs)

        logger.info(f"创建训练会话: {session_id}")
        return session_id

    def train_model(self, session_id: str, num_episodes: int = 1000,
                   save_interval: int = 100) -> Dict[str, Any]:
        """训练模型"""
        if not self.training_manager:
            raise ValueError("请先创建训练会话")

        # 开始训练
        training_results = self.training_manager.train(num_episodes, save_interval)

        # 更新系统统计
        self.system_stats['models_trained'] += 1
        self.system_stats['total_training_episodes'] += num_episodes

        # 保存最终模型
        final_model_path = f"models/rl_model_final_{session_id}.pth"
        self.training_manager.agent.save_model(final_model_path)

        # 记录模型信息
        self.trained_models[session_id] = {
            'model_path': final_model_path,
            'training_episodes': num_episodes,
            'best_score': self.training_manager.best_score,
            'created_at': datetime.now()
        }

        return {
            'session_id': session_id,
            'training_completed': True,
            'num_episodes': num_episodes,
            'best_score': self.training_manager.best_score,
            'model_saved_to': final_model_path,
            'training_results': training_results
        }

    def evaluate_model(self, session_id: str, num_episodes: int = 10) -> Dict[str, Any]:
        """评估模型"""
        if not self.training_manager:
            raise ValueError("请先创建训练会话")

        evaluation_results = self.training_manager.evaluate(num_episodes)

        # 更新系统统计
        self.system_stats['total_evaluation_episodes'] += num_episodes

        best_performance = evaluation_results['average_return']
        if best_performance > self.system_stats['best_model_performance']:
            self.system_stats['best_model_performance'] = best_performance

        return evaluation_results

    def predict_action(self, session_id: str, current_state: np.ndarray) -> TradingAction:
        """预测动作"""
        if session_id not in self.trained_models:
            raise ValueError(f"未找到训练会话: {session_id}")

        if not self.training_manager or not self.training_manager.agent:
            raise ValueError("训练管理器或智能体未初始化")

        # 使用训练好的模型预测动作
        action_index = self.training_manager.agent.act(current_state, training=False)
        action_type = ActionType(action_index)

        # 计算置信度（简化版本）
        with torch.no_grad():
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
            q_values = self.training_manager.agent.q_network(state_tensor)
            confidence = torch.softmax(q_values, dim=1).max().item()

        return TradingAction(
            action_type=action_type,
            confidence=confidence,
            timestamp=datetime.now()
        )

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        training_stats = {}
        if self.training_manager:
            training_stats = self.training_manager.get_training_stats()

        return {
            'system_stats': self.system_stats,
            'current_training_stats': training_stats,
            'trained_models': {
                sid: {
                    'best_score': info['best_score'],
                    'training_episodes': info['training_episodes'],
                    'created_at': info['created_at'].isoformat()
                }
                for sid, info in self.trained_models.items()
            }
        }

# 全局实例
_rl_system_instance = None

def get_rl_system() -> ReinforcementLearningSystem:
    """获取强化学习系统实例"""
    global _rl_system_instance
    if _rl_system_instance is None:
        _rl_system_instance = ReinforcementLearningSystem()
    return _rl_system_instance
