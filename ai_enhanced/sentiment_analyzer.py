"""
Sentiment Analyzer
情绪分析器，用于分析市场情绪和社交媒体数据
支持多种数据源和情绪计算方法
"""

import re
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
import json
import numpy as np

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class SentimentPolarity(Enum):
    """情绪极性"""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2

class SourceType(Enum):
    """数据源类型"""
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    FINANCIAL_BLOG = "financial_blog"
    FORUM = "forum"
    TELEGRAM = "telegram"
    DISCORD = "discord"

class DataFrequency(Enum):
    """数据频率"""
    REALTIME = "realtime"
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"

@dataclass

class SentimentData:
    """情绪数据"""
    data_id: str
    content: str
    source: SourceType
    author: str
    published_at: datetime

    # 情绪分析结果
    sentiment_score: Optional[float] = None  # -1 到 1
    sentiment_polarity: Optional[SentimentPolarity] = None
    confidence: Optional[float] = None

    # 影响力指标
    likes: int = 0
    shares: int = 0
    replies: int = 0
    views: int = 0
    influence_score: float = 0.0

    # 提取的信息
    mentioned_symbols: List[str] = field(default_factory=list)
    mentioned_keywords: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)

    # 语言特征
    word_count: int = 0
    emoji_count: int = 0
    exclamation_count: int = 0

    def to_dict(self) -> dict:
        return {
            'data_id': self.data_id,
            'content': self.content[:200] + '...' if len(self.content) > 200 else self.content,
            'source': self.source.value,
            'author': self.author,
            'published_at': self.published_at.isoformat(),
            'sentiment_score': self.sentiment_score,
            'sentiment_polarity': self.sentiment_polarity.value if self.sentiment_polarity else None,
            'confidence': self.confidence,
            'likes': self.likes,
            'shares': self.shares,
            'replies': self.replies,
            'views': self.views,
            'influence_score': self.influence_score,
            'mentioned_symbols': self.mentioned_symbols,
            'mentioned_keywords': self.mentioned_keywords,
            'hashtags': self.hashtags,
            'word_count': self.word_count,
            'emoji_count': self.emoji_count,
            'exclamation_count': self.exclamation_count
        }

@dataclass

class AggregatedSentiment:
    """聚合情绪数据"""
    symbol: str
    time_window: datetime
    frequency: DataFrequency

    # 聚合指标
    total_mentions: int = 0
    average_sentiment: float = 0.0
    sentiment_std: float = 0.0
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0

    # 影响力加权指标
    weighted_sentiment: float = 0.0
    total_influence: float = 0.0

    # 来源分布
    source_distribution: Dict[str, int] = field(default_factory=dict)

    # 情绪强度
    sentiment_intensity: float = 0.0  # 情绪波动程度
    sentiment_momentum: float = 0.0   # 情绪变化趋势

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'time_window': self.time_window.isoformat(),
            'frequency': self.frequency.value,
            'total_mentions': self.total_mentions,
            'average_sentiment': self.average_sentiment,
            'sentiment_std': self.sentiment_std,
            'bullish_count': self.bullish_count,
            'bearish_count': self.bearish_count,
            'neutral_count': self.neutral_count,
            'weighted_sentiment': self.weighted_sentiment,
            'total_influence': self.total_influence,
            'source_distribution': self.source_distribution,
            'sentiment_intensity': self.sentiment_intensity,
            'sentiment_momentum': self.sentiment_momentum
        }

class TextPreprocessor:
    """文本预处理器"""

    def __init__(self):
        # 股票符号模式
        self.stock_pattern = re.compile(r'\$([A-Z]{1,5})\b')

        # 情绪词典
        self.positive_words = {
            'excellent', 'amazing', 'fantastic', 'great', 'good', 'positive', 'up', 'rise', 'bull', 'bullish',
            'gain', 'profit', 'win', 'success', 'strong', 'buy', 'long', 'moon', 'rocket', '🚀', '📈',
            'love', 'like', 'awesome', 'incredible', 'outstanding', 'boom', 'surge', 'soar', 'climb'
        }

        self.negative_words = {
            'terrible', 'awful', 'bad', 'negative', 'down', 'fall', 'bear', 'bearish', 'loss', 'lose',
            'fail', 'failure', 'weak', 'sell', 'short', 'crash', 'dump', 'drop', '📉', '💩',
            'hate', 'dislike', 'horrible', 'disappointing', 'decline', 'plunge', 'collapse', 'tank'
        }

        # 强化词
        self.intensifiers = {
            'very', 'extremely', 'really', 'super', 'absolutely', 'completely', 'totally',
            'incredibly', 'amazingly', 'highly', 'massively', 'hugely'
        }

        # 表情符号映射
        self.emoji_sentiment = {
            '😀': 1, '😃': 1, '😄': 1, '😁': 1, '😆': 1, '😊': 1, '☺️': 1, '🙂': 0.5,
            '😍': 2, '🤩': 2, '😎': 1, '🥳': 2, '🎉': 1, '👍': 1, '💪': 1, '🔥': 1,
            '😢': -1, '😭': -2, '😠': -2, '😡': -2, '🤬': -2, '😤': -1, '👎': -1,
            '💀': -1, '💩': -2, '🤡': -1, '😴': -0.5
        }

    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # 移除@mentions (保留内容用于分析但不影响情绪)
        text = re.sub(r'@\w+', '', text)

        # 标准化空格
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_features(self, text: str) -> Dict[str, Any]:
        """提取文本特征"""
        text_lower = text.lower()

        features = {
            'word_count': len(text.split()),
            'char_count': len(text),
            'emoji_count': len([c for c in text if c in self.emoji_sentiment]),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'capital_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'mentioned_symbols': self.extract_stock_symbols(text),
            'hashtags': self.extract_hashtags(text)
        }

        return features

    def extract_stock_symbols(self, text: str) -> List[str]:
        """提取股票代码"""
        matches = self.stock_pattern.findall(text.upper())
        return list(set(matches))  # 去重

    def extract_hashtags(self, text: str) -> List[str]:
        """提取hashtags"""
        hashtag_pattern = re.compile(r'  # (\w+)')
        matches = hashtag_pattern.findall(text.lower())
        return list(set(matches))  # 去重

    def calculate_lexicon_sentiment(self, text: str) -> Tuple[float, float]:
        """基于词典计算情绪"""
        words = re.findall(r'\b\w+\b', text.lower())

        positive_score = 0
        negative_score = 0
        intensifier_multiplier = 1.0

        for i, word in enumerate(words):
            # 检查强化词
            if word in self.intensifiers:
                intensifier_multiplier = 1.5
                continue

            # 计算情绪得分
            if word in self.positive_words:
                positive_score += 1 * intensifier_multiplier
            elif word in self.negative_words:
                negative_score += 1 * intensifier_multiplier

            # 重置强化倍数
            intensifier_multiplier = 1.0

        # 计算表情符号情绪
        emoji_score = sum(self.emoji_sentiment.get(c, 0) for c in text)

        # 综合得分
        total_positive = positive_score + max(emoji_score, 0)
        total_negative = negative_score + abs(min(emoji_score, 0))

        if total_positive + total_negative == 0:
            return 0.0, 0.0  # 中性

        # 归一化到[-1, 1]
        sentiment = (total_positive - total_negative) / (total_positive + total_negative)
        confidence = min((total_positive + total_negative) / 10.0, 1.0)  # 基于词汇密度的置信度

        return sentiment, confidence

class MockDataSource:
    """模拟数据源"""

    def __init__(self, source_type: SourceType):
        self.source_type = source_type

        # 模拟数据模板
        self.sample_tweets = [
            ("BullishTrader", "🚀 $AAPL is going to the moon! Amazing earnings report!  # bullish  # apple"),
            ("MarketAnalyst", "$TSLA looks weak today. Might be time to take profits. 📉"),
            ("InvestorJoe", "Just bought more $MSFT. Love this dip! 💪  # longterm"),
            ("TechFan2023", "$NVDA crushing it with AI revolution! This is just the beginning 🔥"),
            ("BearishBob", "Market crash incoming? Everything looks overvalued right now 😰"),
            ("CryptoKing", "Bitcoin to 100k! All altcoins following. Bull run confirmed! 🌙"),
            ("StockGuru", "Dividend stocks are the way to go in this market. $KO $PEP steady gains"),
            ("RetailTrader", "Lost money on $GME again... When will I learn? 😭"),
            ("OptionsPlayer", "Calls on $SPY printing money! Easy 200% gains today 📈"),
            ("ValueInvestor", "$BRK.B undervalued as always. Buffett knows what he's doing")
        ]

    async def fetch_data(self, symbols: List[str] = None, limit: int = 50) -> List[SentimentData]:
        """获取模拟数据"""
        data = []

        for i in range(min(limit, len(self.sample_tweets))):
            author, content = self.sample_tweets[i % len(self.sample_tweets)]

            # 添加时间戳变化
            timestamp = datetime.now() - timedelta(minutes=np.random.randint(1, 1440))

            sentiment_data = SentimentData(
                data_id=f"{self.source_type.value}_{i}_{int(timestamp.timestamp())}",
                content=content,
                source=self.source_type,
                author=f"{author}_{np.random.randint(1000, 9999)}",
                published_at=timestamp,
                likes=np.random.randint(0, 1000),
                shares=np.random.randint(0, 100),
                replies=np.random.randint(0, 50),
                views=np.random.randint(100, 10000)
            )

            data.append(sentiment_data)

        return data

class SentimentCalculator:
    """情绪计算器"""

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def analyze_sentiment(self, data: SentimentData) -> SentimentData:
        """分析单条数据的情绪"""
        # 清理文本
        clean_content = self.preprocessor.clean_text(data.content)

        # 提取特征
        features = self.preprocessor.extract_features(data.content)

        # 更新数据特征
        data.word_count = features['word_count']
        data.emoji_count = features['emoji_count']
        data.exclamation_count = features['exclamation_count']
        data.mentioned_symbols = features['mentioned_symbols']
        data.hashtags = features['hashtags']

        # 计算情绪
        sentiment_score, confidence = self.preprocessor.calculate_lexicon_sentiment(clean_content)

        data.sentiment_score = sentiment_score
        data.confidence = confidence

        # 设置情绪极性
        if sentiment_score >= 0.5:
            data.sentiment_polarity = SentimentPolarity.VERY_POSITIVE
        elif sentiment_score >= 0.1:
            data.sentiment_polarity = SentimentPolarity.POSITIVE
        elif sentiment_score <= -0.5:
            data.sentiment_polarity = SentimentPolarity.VERY_NEGATIVE
        elif sentiment_score <= -0.1:
            data.sentiment_polarity = SentimentPolarity.NEGATIVE
        else:
            data.sentiment_polarity = SentimentPolarity.NEUTRAL

        # 计算影响力得分
        data.influence_score = self._calculate_influence_score(data)

        return data

    def _calculate_influence_score(self, data: SentimentData) -> float:
        """计算影响力得分"""
        # 基于互动指标计算影响力
        interaction_score = (
            data.likes * 1.0 +
            data.shares * 2.0 +  # 分享权重更高
            data.replies * 1.5 +
            data.views * 0.01    # 浏览量权重较低
        )

        # 基于内容质量调整
        content_quality = min(data.word_count / 20.0, 2.0)  # 适度的内容长度

        # 基于情绪强度调整
        sentiment_intensity = abs(data.sentiment_score or 0) * 2

        # 综合影响力得分
        influence_score = (interaction_score * 0.7 +
                          content_quality * 0.2 +
                          sentiment_intensity * 0.1)

        # 归一化
        return min(influence_score / 1000.0, 10.0)

    def aggregate_sentiment(self, data_list: List[SentimentData],
                          symbol: str, time_window: datetime,
                          frequency: DataFrequency) -> AggregatedSentiment:
        """聚合情绪数据"""
        if not data_list:
            return AggregatedSentiment(symbol, time_window, frequency)

        # 过滤相关数据
        relevant_data = [
            d for d in data_list
            if symbol.upper() in [s.upper() for s in d.mentioned_symbols]
        ]

        if not relevant_data:
            return AggregatedSentiment(symbol, time_window, frequency)

        # 基础统计
        sentiments = [d.sentiment_score for d in relevant_data if d.sentiment_score is not None]
        influences = [d.influence_score for d in relevant_data]

        aggregated = AggregatedSentiment(symbol, time_window, frequency)
        aggregated.total_mentions = len(relevant_data)

        if sentiments:
            aggregated.average_sentiment = np.mean(sentiments)
            aggregated.sentiment_std = np.std(sentiments)

            # 影响力加权情绪
            if influences and sum(influences) > 0:
                weighted_sentiment = sum(s * i for s, i in zip(sentiments, influences))
                aggregated.weighted_sentiment = weighted_sentiment / sum(influences)
                aggregated.total_influence = sum(influences)
            else:
                aggregated.weighted_sentiment = aggregated.average_sentiment

        # 情绪分布统计
        for data in relevant_data:
            if data.sentiment_polarity:
                if data.sentiment_polarity in [SentimentPolarity.POSITIVE, SentimentPolarity.VERY_POSITIVE]:
                    aggregated.bullish_count += 1
                elif data.sentiment_polarity in [SentimentPolarity.NEGATIVE, SentimentPolarity.VERY_NEGATIVE]:
                    aggregated.bearish_count += 1
                else:
                    aggregated.neutral_count += 1

        # 来源分布
        source_counter = Counter(d.source.value for d in relevant_data)
        aggregated.source_distribution = dict(source_counter)

        # 情绪强度 (标准差作为波动性指标)
        aggregated.sentiment_intensity = aggregated.sentiment_std

        return aggregated

class SentimentAnalyzer:
    """情绪分析器主类"""

    def __init__(self):
        self.data_sources = {}
        self.sentiment_calculator = SentimentCalculator()

        # 添加默认数据源
        self.add_data_source("twitter", MockDataSource(SourceType.TWITTER))
        self.add_data_source("reddit", MockDataSource(SourceType.REDDIT))

        # 数据缓存
        self.raw_data_cache = deque(maxlen=10000)
        self.aggregated_cache = defaultdict(lambda: deque(maxlen=1000))

        # 实时流
        self.subscribers = []

        # 统计信息
        self.stats = {
            'total_data_processed': 0,
            'total_symbols_tracked': 0,
            'average_sentiment': 0.0,
            'data_by_source': defaultdict(int),
            'sentiment_distribution': defaultdict(int),
            'processing_errors': 0
        }

        # 任务管理
        self.is_running = False
        self.processing_task = None

        logger.info("情绪分析器初始化完成")

    def add_data_source(self, name: str, source):
        """添加数据源"""
        self.data_sources[name] = source
        logger.info(f"添加数据源: {name}")

    async def start(self):
        """启动分析器"""
        if self.is_running:
            return

        self.is_running = True

        # 启动定期处理任务
        self.processing_task = asyncio.create_task(self._processing_loop())

        logger.info("情绪分析器启动完成")

    async def stop(self):
        """停止分析器"""
        if not self.is_running:
            return

        logger.info("正在停止情绪分析器...")
        self.is_running = False

        if self.processing_task:
            self.processing_task.cancel()

        logger.info("情绪分析器已停止")

    async def _processing_loop(self):
        """处理循环"""
        while self.is_running:
            try:
                await self.collect_and_analyze_data()
                await asyncio.sleep(300)  # 5分钟处理一次
            except Exception as e:
                logger.error(f"处理循环错误: {e}")
                await asyncio.sleep(60)

    async def collect_and_analyze_data(self, symbols: List[str] = None,
                                     limit: int = 100) -> List[SentimentData]:
        """收集并分析数据"""
        all_data = []

        for source_name, source in self.data_sources.items():
            try:
                # 获取数据
                raw_data = await source.fetch_data(symbols, limit)

                # 分析情绪
                analyzed_data = []
                for data in raw_data:
                    try:
                        analyzed = self.sentiment_calculator.analyze_sentiment(data)
                        analyzed_data.append(analyzed)

                        # 缓存数据
                        self.raw_data_cache.append(analyzed)

                    except Exception as e:
                        logger.error(f"分析数据失败 {data.data_id}: {e}")
                        self.stats['processing_errors'] += 1

                all_data.extend(analyzed_data)

                # 更新统计
                self.stats['total_data_processed'] += len(analyzed_data)
                self.stats['data_by_source'][source_name] += len(analyzed_data)

                logger.info(f"从 {source_name} 处理了 {len(analyzed_data)} 条数据")

            except Exception as e:
                logger.error(f"从数据源 {source_name} 获取数据失败: {e}")

        # 通知订阅者
        if all_data:
            await self._notify_subscribers(all_data)

        # 更新全局统计
        self._update_global_stats(all_data)

        return all_data

    async def _notify_subscribers(self, data_list: List[SentimentData]):
        """通知订阅者"""
        for callback in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data_list)
                else:
                    callback(data_list)
            except Exception as e:
                logger.error(f"通知订阅者失败: {e}")

    def _update_global_stats(self, data_list: List[SentimentData]):
        """更新全局统计"""
        if not data_list:
            return

        sentiments = [d.sentiment_score for d in data_list if d.sentiment_score is not None]

        if sentiments:
            # 更新平均情绪
            current_avg = self.stats['average_sentiment']
            current_count = self.stats['total_data_processed'] - len(sentiments)

            total_sentiment = current_avg * current_count + sum(sentiments)
            self.stats['average_sentiment'] = total_sentiment / self.stats['total_data_processed']

        # 更新情绪分布
        for data in data_list:
            if data.sentiment_polarity:
                self.stats['sentiment_distribution'][data.sentiment_polarity.value] += 1

        # 更新追踪的股票数量
        all_symbols = set()
        for data in data_list:
            all_symbols.update(data.mentioned_symbols)

        self.stats['total_symbols_tracked'] = len(all_symbols)

    def get_sentiment_summary(self, symbol: str, hours_back: int = 24) -> Dict[str, Any]:
        """获取情绪摘要"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        # 过滤相关数据
        relevant_data = []
        for data in self.raw_data_cache:
            if (data.published_at >= cutoff_time and
                symbol.upper() in [s.upper() for s in data.mentioned_symbols]):
                relevant_data.append(data)

        if not relevant_data:
            return {
                'symbol': symbol,
                'total_mentions': 0,
                'average_sentiment': 0.0,
                'bullish_ratio': 0.0,
                'bearish_ratio': 0.0,
                'neutral_ratio': 0.0,
                'sentiment_trend': 'neutral',
                'hours_back': hours_back
            }

        # 计算统计指标
        sentiments = [d.sentiment_score for d in relevant_data if d.sentiment_score is not None]

        bullish_count = sum(1 for d in relevant_data
                           if d.sentiment_polarity in [SentimentPolarity.POSITIVE,
                                                      SentimentPolarity.VERY_POSITIVE])
        bearish_count = sum(1 for d in relevant_data
                           if d.sentiment_polarity in [SentimentPolarity.NEGATIVE,
                                                      SentimentPolarity.VERY_NEGATIVE])
        neutral_count = len(relevant_data) - bullish_count - bearish_count

        total_count = len(relevant_data)

        # 计算趋势
        recent_data = sorted(relevant_data, key=lambda x: x.published_at)[-10:]  # 最近10条
        recent_sentiments = [d.sentiment_score for d in recent_data if d.sentiment_score is not None]

        if len(recent_sentiments) >= 2:
            trend_slope = np.polyfit(range(len(recent_sentiments)), recent_sentiments, 1)[0]
            if trend_slope > 0.1:
                sentiment_trend = 'improving'
            elif trend_slope < -0.1:
                sentiment_trend = 'declining'
            else:
                sentiment_trend = 'stable'
        else:
            sentiment_trend = 'neutral'

        return {
            'symbol': symbol,
            'total_mentions': total_count,
            'average_sentiment': np.mean(sentiments) if sentiments else 0.0,
            'sentiment_std': np.std(sentiments) if sentiments else 0.0,
            'bullish_ratio': bullish_count / total_count,
            'bearish_ratio': bearish_count / total_count,
            'neutral_ratio': neutral_count / total_count,
            'sentiment_trend': sentiment_trend,
            'top_sources': dict(Counter(d.source.value for d in relevant_data).most_common(3)),
            'hours_back': hours_back,
            'data_points': len(relevant_data)
        }

    def get_trending_symbols(self, limit: int = 10,
                           hours_back: int = 24) -> List[Dict[str, Any]]:
        """获取热门股票"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        # 统计股票提及次数
        symbol_mentions = defaultdict(list)

        for data in self.raw_data_cache:
            if data.published_at >= cutoff_time:
                for symbol in data.mentioned_symbols:
                    symbol_mentions[symbol.upper()].append(data)

        # 计算热门股票
        trending_symbols = []

        for symbol, data_list in symbol_mentions.items():
            if len(data_list) >= 5:  # 至少5次提及
                sentiments = [d.sentiment_score for d in data_list if d.sentiment_score is not None]
                influences = [d.influence_score for d in data_list]

                trending_score = (
                    len(data_list) * 1.0 +  # 提及频率
                    sum(influences) * 2.0 +  # 影响力总和
                    abs(np.mean(sentiments)) * 50 if sentiments else 0  # 情绪强度
                )

                trending_symbols.append({
                    'symbol': symbol,
                    'mentions': len(data_list),
                    'average_sentiment': np.mean(sentiments) if sentiments else 0.0,
                    'total_influence': sum(influences),
                    'trending_score': trending_score,
                    'sentiment_polarity': 'bullish' if np.mean(sentiments) > 0.1 else
                                        'bearish' if np.mean(sentiments) < -0.1 else 'neutral'
                })

        # 按热门得分排序
        trending_symbols.sort(key=lambda x: x['trending_score'], reverse=True)

        return trending_symbols[:limit]

    def subscribe_to_sentiment(self, callback: Callable):
        """订阅情绪数据"""
        self.subscribers.append(callback)
        logger.info("新增情绪数据订阅者")

    def unsubscribe_from_sentiment(self, callback: Callable):
        """取消订阅情绪数据"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info("移除情绪数据订阅者")

    async def analyze_custom_text(self, text: str, author: str = "custom") -> SentimentData:
        """分析自定义文本"""
        # 创建临时数据对象
        data = SentimentData(
            data_id=f"custom_{int(datetime.now().timestamp())}",
            content=text,
            source=SourceType.NEWS,  # 默认来源
            author=author,
            published_at=datetime.now()
        )

        return self.sentiment_calculator.analyze_sentiment(data)

    def get_historical_aggregation(self, symbol: str,
                                 frequency: DataFrequency = DataFrequency.HOURLY,
                                 days_back: int = 7) -> List[AggregatedSentiment]:
        """获取历史聚合数据"""
        cutoff_time = datetime.now() - timedelta(days=days_back)

        # 过滤数据
        relevant_data = []
        for data in self.raw_data_cache:
            if (data.published_at >= cutoff_time and
                symbol.upper() in [s.upper() for s in data.mentioned_symbols]):
                relevant_data.append(data)

        if not relevant_data:
            return []

        # 按时间窗口分组
        if frequency == DataFrequency.HOURLY:
            window_size = timedelta(hours=1)
        elif frequency == DataFrequency.DAILY:
            window_size = timedelta(days=1)
        else:
            window_size = timedelta(hours=1)  # 默认

        # 创建时间窗口
        start_time = cutoff_time
        end_time = datetime.now()

        aggregated_data = []
        current_time = start_time

        while current_time < end_time:
            window_end = current_time + window_size

            # 获取窗口内的数据
            window_data = [
                d for d in relevant_data
                if current_time <= d.published_at < window_end
            ]

            if window_data:
                aggregated = self.sentiment_calculator.aggregate_sentiment(
                    window_data, symbol, current_time, frequency
                )
                aggregated_data.append(aggregated)

            current_time = window_end

        return aggregated_data

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'is_running': self.is_running,
            'data_sources_count': len(self.data_sources),
            'cached_data_points': len(self.raw_data_cache),
            'active_subscribers': len(self.subscribers),
            'stats': self.stats,
            'cache_stats': {
                'raw_data_cache_size': len(self.raw_data_cache),
                'aggregated_cache_keys': len(self.aggregated_cache)
            }
        }

# 全局实例
_sentiment_analyzer_instance = None

def get_sentiment_analyzer() -> SentimentAnalyzer:
    """获取情绪分析器实例"""
    global _sentiment_analyzer_instance
    if _sentiment_analyzer_instance is None:
        _sentiment_analyzer_instance = SentimentAnalyzer()
    return _sentiment_analyzer_instance
