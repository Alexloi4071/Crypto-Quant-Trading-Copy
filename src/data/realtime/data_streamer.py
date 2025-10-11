"""
Real-time Data Streamer
实时数据流处理器，负责从多个数据源收集并流式分发数据
支持数据转换、过滤和路由分发
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Any, Callable, Union, Set
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.collectors.base_collector import BaseCollector
from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class DataType(Enum):
    """数据类型枚举"""
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADE = "trade"
    KLINE = "kline"
    VOLUME = "volume"
    NEWS = "news"
    SENTIMENT = "sentiment"

@dataclass

class StreamData:
    """流式数据结构"""
    data_type: DataType
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'data_type': self.data_type.value,
            'symbol': self.symbol,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source
        }

@dataclass

class DataStreamConfig:
    """数据流配置"""
    stream_id: str
    data_types: Set[DataType]
    symbols: Set[str]
    sources: Set[str]
    filters: Dict[str, Any] = field(default_factory=dict)
    transformers: List[Callable] = field(default_factory=list)
    buffer_size: int = 1000
    batch_size: int = 10
    flush_interval: float = 1.0  # seconds

class DataFilter:
    """数据过滤器"""

    @staticmethod

    def price_range_filter(min_price: float = 0, max_price: float = float('inf')):
        """价格范围过滤器"""

        def filter_func(data: StreamData) -> bool:
            if data.data_type == DataType.TICKER:
                price = float(data.data.get('price', 0))
                return min_price <= price <= max_price
            return True
        return filter_func

    @staticmethod

    def volume_filter(min_volume: float = 0):
        """成交量过滤器"""

        def filter_func(data: StreamData) -> bool:
            volume = float(data.data.get('volume', 0))
            return volume >= min_volume
        return filter_func

    @staticmethod

    def symbol_filter(allowed_symbols: Set[str]):
        """交易对过滤器"""

        def filter_func(data: StreamData) -> bool:
            return data.symbol in allowed_symbols
        return filter_func

    @staticmethod

    def time_filter(max_age_seconds: int = 60):
        """时间过滤器"""

        def filter_func(data: StreamData) -> bool:
            age = (datetime.now() - data.timestamp).total_seconds()
            return age <= max_age_seconds
        return filter_func

class DataTransformer:
    """数据转换器"""

    @staticmethod

    def normalize_price(base_currency: str = 'USDT'):
        """价格标准化转换器"""

        def transform_func(data: StreamData) -> StreamData:
            if data.data_type == DataType.TICKER:
                # 简化的价格标准化逻辑
                if not data.symbol.endswith(base_currency):
                    # 需要转换价格，实际应该查汇率
                    pass
            return data
        return transform_func

    @staticmethod

    def add_derived_metrics():
        """添加衍生指标转换器"""

        def transform_func(data: StreamData) -> StreamData:
            if data.data_type == DataType.TICKER:
                price = float(data.data.get('price', 0))
                volume = float(data.data.get('volume', 0))

                # 添加成交额
                data.data['amount'] = price * volume

                # 添加价格变化率
                if 'prev_price' in data.data:
                    prev_price = float(data.data['prev_price'])
                    if prev_price > 0:
                        data.data['price_change_pct'] = (price - prev_price) / prev_price * 100

            return data
        return transform_func

    @staticmethod

    def format_timestamps():
        """时间戳格式化转换器"""

        def transform_func(data: StreamData) -> StreamData:
            data.data['formatted_timestamp'] = data.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            return data
        return transform_func

class DataStream:
    """单个数据流"""

    def __init__(self, config: DataStreamConfig):
        self.config = config
        self.buffer = deque(maxlen=config.buffer_size)
        self.subscribers = []  # 订阅者回调函数
        self.filters = []
        self.transformers = config.transformers.copy()

        # 统计信息
        self.stats = {
            'total_received': 0,
            'total_filtered': 0,
            'total_sent': 0,
            'last_data_time': None,
            'start_time': datetime.now()
        }

        # 批处理
        self.batch_buffer = []
        self.last_flush = time.time()

    def add_filter(self, filter_func: Callable[[StreamData], bool]):
        """添加过滤器"""
        self.filters.append(filter_func)

    def add_transformer(self, transform_func: Callable[[StreamData], StreamData]):
        """添加转换器"""
        self.transformers.append(transform_func)

    def subscribe(self, callback: Callable):
        """订阅数据流"""
        self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable):
        """取消订阅"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    async def push_data(self, data: StreamData):
        """推送数据到流"""
        self.stats['total_received'] += 1
        self.stats['last_data_time'] = datetime.now()

        # 应用过滤器
        for filter_func in self.filters:
            if not filter_func(data):
                self.stats['total_filtered'] += 1
                return

        # 应用转换器
        for transform_func in self.transformers:
            data = transform_func(data)

        # 添加到缓冲区
        self.buffer.append(data)
        self.batch_buffer.append(data)

        # 检查是否需要批量发送
        await self._check_flush()

    async def _check_flush(self):
        """检查是否需要刷新批次"""
        should_flush = (
            len(self.batch_buffer) >= self.config.batch_size or
            time.time() - self.last_flush >= self.config.flush_interval
        )

        if should_flush and self.batch_buffer:
            await self._flush_batch()

    async def _flush_batch(self):
        """刷新批次数据"""
        if not self.batch_buffer:
            return

        batch_data = self.batch_buffer.copy()
        self.batch_buffer.clear()
        self.last_flush = time.time()

        # 发送给所有订阅者
        for callback in self.subscribers:
            try:
                await callback(batch_data)
                self.stats['total_sent'] += len(batch_data)
            except Exception as e:
                logger.error(f"发送数据到订阅者失败: {e}")

    async def force_flush(self):
        """强制刷新"""
        await self._flush_batch()

    def get_recent_data(self, count: int = 10) -> List[StreamData]:
        """获取最近的数据"""
        return list(self.buffer)[-count:]

    def get_stats(self) -> dict:
        """获取统计信息"""
        uptime = datetime.now() - self.stats['start_time']

        return {
            'stream_id': self.config.stream_id,
            'uptime_seconds': uptime.total_seconds(),
            'buffer_size': len(self.buffer),
            'subscribers_count': len(self.subscribers),
            'filters_count': len(self.filters),
            'transformers_count': len(self.transformers),
            **self.stats,
            'last_data_time': self.stats['last_data_time'].isoformat() if self.stats['last_data_time'] else None
        }

class DataStreamer:
    """实时数据流处理器主类"""

    def __init__(self):
        self.streams: Dict[str, DataStream] = {}
        self.data_sources: Dict[str, BaseCollector] = {}

        # 路由配置
        self.routing_rules: Dict[str, List[str]] = defaultdict(list)  # source -> stream_ids

        # 任务管理
        self.collection_tasks = {}
        self.processing_task = None
        self.is_running = False

        # 全局统计
        self.global_stats = {
            'total_streams': 0,
            'total_sources': 0,
            'total_data_points': 0,
            'start_time': datetime.now()
        }

        logger.info("实时数据流处理器初始化完成")

    async def start(self):
        """启动数据流处理器"""
        if self.is_running:
            return

        self.is_running = True

        # 启动数据收集任务
        for source_id, collector in self.data_sources.items():
            if hasattr(collector, 'start'):
                self.collection_tasks[source_id] = asyncio.create_task(
                    self._run_collector(source_id, collector)
                )

        # 启动处理任务
        self.processing_task = asyncio.create_task(self._processing_loop())

        logger.info("实时数据流处理器启动完成")

    async def stop(self):
        """停止数据流处理器"""
        if not self.is_running:
            return

        logger.info("正在停止实时数据流处理器...")
        self.is_running = False

        # 停止数据收集任务
        for task in self.collection_tasks.values():
            task.cancel()

        # 停止处理任务
        if self.processing_task:
            self.processing_task.cancel()

        # 强制刷新所有流
        for stream in self.streams.values():
            await stream.force_flush()

        logger.info("实时数据流处理器已停止")

    def create_stream(self, config: DataStreamConfig) -> DataStream:
        """创建数据流"""
        if config.stream_id in self.streams:
            return self.streams[config.stream_id]

        stream = DataStream(config)
        self.streams[config.stream_id] = stream
        self.global_stats['total_streams'] += 1

        logger.info(f"创建数据流: {config.stream_id}")
        return stream

    def register_data_source(self, source_id: str, collector: BaseCollector):
        """注册数据源"""
        self.data_sources[source_id] = collector
        self.global_stats['total_sources'] += 1

        logger.info(f"注册数据源: {source_id}")

    def add_routing_rule(self, source_id: str, stream_id: str):
        """添加路由规则"""
        self.routing_rules[source_id].append(stream_id)
        logger.debug(f"添加路由规则: {source_id} -> {stream_id}")

    async def _run_collector(self, source_id: str, collector: BaseCollector):
        """运行数据收集器"""
        try:
            async for raw_data in collector.stream_data():
                if not self.is_running:
                    break

                # 转换为标准格式
                stream_data = await self._convert_to_stream_data(source_id, raw_data)

                if stream_data:
                    # 路由到相应的流
                    await self._route_data(source_id, stream_data)
                    self.global_stats['total_data_points'] += 1

        except Exception as e:
            logger.error(f"数据收集器运行错误 {source_id}: {e}")

    async def _convert_to_stream_data(self, source_id: str, raw_data: dict) -> Optional[StreamData]:
        """转换原始数据为流数据格式"""
        try:
            # 根据数据源和数据内容判断数据类型
            data_type = self._detect_data_type(raw_data)
            symbol = raw_data.get('symbol', 'UNKNOWN')
            timestamp = datetime.now()

            # 尝试解析时间戳
            if 'timestamp' in raw_data:
                try:
                    timestamp = datetime.fromtimestamp(raw_data['timestamp'] / 1000)
                except:
                    pass

            return StreamData(
                data_type=data_type,
                symbol=symbol,
                data=raw_data,
                timestamp=timestamp,
                source=source_id
            )

        except Exception as e:
            logger.error(f"转换数据失败: {e}")
            return None

    def _detect_data_type(self, data: dict) -> DataType:
        """检测数据类型"""
        # 简化的数据类型检测逻辑
        if 'price' in data and 'volume' in data:
            return DataType.TICKER
        elif 'bids' in data and 'asks' in data:
            return DataType.ORDERBOOK
        elif 'trade_id' in data:
            return DataType.TRADE
        elif 'open' in data and 'close' in data:
            return DataType.KLINE
        else:
            return DataType.TICKER  # 默认

    async def _route_data(self, source_id: str, data: StreamData):
        """路由数据到流"""
        target_streams = self.routing_rules.get(source_id, [])

        for stream_id in target_streams:
            if stream_id in self.streams:
                stream = self.streams[stream_id]

                # 检查流是否接受这种数据类型和交易对
                if (data.data_type in stream.config.data_types and
                    data.symbol in stream.config.symbols):

                    await stream.push_data(data)

    async def _processing_loop(self):
        """处理循环"""
        while self.is_running:
            try:
                # 定期检查所有流的刷新状态
                for stream in self.streams.values():
                    await stream._check_flush()

                await asyncio.sleep(0.1)  # 100ms检查间隔

            except Exception as e:
                logger.error(f"处理循环错误: {e}")
                await asyncio.sleep(1)

    # 便利方法

    def get_stream(self, stream_id: str) -> Optional[DataStream]:
        """获取数据流"""
        return self.streams.get(stream_id)

    async def create_ticker_stream(self, stream_id: str, symbols: Set[str],
                                 sources: Set[str] = None) -> DataStream:
        """创建行情数据流"""
        config = DataStreamConfig(
            stream_id=stream_id,
            data_types={DataType.TICKER},
            symbols=symbols,
            sources=sources or set(self.data_sources.keys()),
            batch_size=5,
            flush_interval=0.5
        )

        stream = self.create_stream(config)

        # 添加路由规则
        for source_id in config.sources:
            self.add_routing_rule(source_id, stream_id)

        return stream

    async def create_trade_stream(self, stream_id: str, symbols: Set[str],
                                sources: Set[str] = None) -> DataStream:
        """创建交易数据流"""
        config = DataStreamConfig(
            stream_id=stream_id,
            data_types={DataType.TRADE},
            symbols=symbols,
            sources=sources or set(self.data_sources.keys()),
            batch_size=10,
            flush_interval=1.0
        )

        stream = self.create_stream(config)

        # 添加路由规则
        for source_id in config.sources:
            self.add_routing_rule(source_id, stream_id)

        return stream

    def get_stats(self) -> dict:
        """获取统计信息"""
        uptime = datetime.now() - self.global_stats['start_time']

        stream_stats = {
            stream_id: stream.get_stats()
            for stream_id, stream in self.streams.items()
        }

        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime.total_seconds(),
            'global_stats': self.global_stats,
            'stream_stats': stream_stats,
            'routing_rules': dict(self.routing_rules)
        }

# 全局实例
_data_streamer_instance = None

def get_data_streamer() -> DataStreamer:
    """获取数据流处理器实例"""
    global _data_streamer_instance
    if _data_streamer_instance is None:
        _data_streamer_instance = DataStreamer()
    return _data_streamer_instance
