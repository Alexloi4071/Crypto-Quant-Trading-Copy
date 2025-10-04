"""
High-Frequency Data Collector
高频数据收集器，负责实时收集和处理市场数据
与现有数据管理系统深度集成，提供高性能数据流
"""

import asyncio
import aiohttp
import aioredis
import websockets
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import json
import time
from collections import deque, defaultdict
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager
from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class DataSource:
    """数据源基类"""
    
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.is_connected = False
        self.last_update = None
        self.error_count = 0
        self.data_count = 0
        
    async def connect(self):
        """连接到数据源"""
        raise NotImplementedError
        
    async def disconnect(self):
        """断开数据源连接"""
        raise NotImplementedError
        
    async def subscribe(self, symbols: List[str]):
        """订阅数据"""
        raise NotImplementedError
        
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        raise NotImplementedError

class BinanceWebSocketSource(DataSource):
    """币安WebSocket数据源"""
    
    def __init__(self, config: dict):
        super().__init__("binance_ws", config)
        self.websocket = None
        self.subscribed_streams = set()
        self.base_url = "wss://stream.binance.com:9443/ws"
        
    async def connect(self):
        """连接币安WebSocket"""
        try:
            self.websocket = await websockets.connect(self.base_url)
            self.is_connected = True
            logger.info(f"已连接到 WebSocket: {{self.name}")
            
            # 启动消息处理循环
            asyncio.create_task(self._message_loop())
            
        except Exception as e:
            logger.error(f"连接 {self.name} 失败: {e}")
            self.error_count += 1
            raise
    
    async def disconnect(self):
        """断开连接"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            logger.info(f"已断开 {self.name} 连接")
    
    async def subscribe(self, symbols: List[str]):
        """订阅交易对数据"""
        if not self.is_connected:
            await self.connect()
        
        streams = []
        for symbol in symbols:
            symbol_lower = symbol.lower()
            streams.extend([
                f"{symbol_lower}@ticker",      # 24hr ticker
                f"{symbol_lower}@bookTicker",  # 最优买卖价
                f"{symbol_lower}@trade",       # 实时交易
                f"{symbol_lower}@depth5"       # 5档深度
            ])
        
        # 发送订阅消息
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(time.time())
        }
        
        await self.websocket.send(json.dumps(subscribe_msg))
        self.subscribed_streams.update(streams)
        
        logger.info(f"已订阅 {len(symbols)} 个交易对的数据流")
    
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        if not self.is_connected:
            return
        
        streams = []
        for symbol in symbols:
            symbol_lower = symbol.lower()
            symbol_streams = [s for s in self.subscribed_streams 
                            if s.startswith(symbol_lower)]
            streams.extend(symbol_streams)
        
        if streams:
            unsubscribe_msg = {
                "method": "UNSUBSCRIBE", 
                "params": streams,
                "id": int(time.time())
            }
            
            await self.websocket.send(json.dumps(unsubscribe_msg))
            self.subscribed_streams -= set(streams)
    
    async def _message_loop(self):
        """消息处理循环"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    if 'stream' in data:
                        await self._process_market_data(data)
                        self.data_count += 1
                        self.last_update = datetime.now()
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"解析消息失败: {e}")
                    continue
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"{self.name} 连接已关闭")
            self.is_connected = False
        except Exception as e:
            logger.error(f"{self.name} 消息循环错误: {e}")
            self.error_count += 1
    
    async def _process_market_data(self, data: dict):
        """处理市场数据"""
        stream = data.get('stream', '')
        market_data = data.get('data', {})
        
        # 发送到数据收集器
        if hasattr(self, '_data_callback'):
            await self._data_callback(stream, market_data)

class HighFrequencyDataCollector:
    """高频数据收集器主类"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.data_sources = {}
        self.data_callbacks = []
        self.is_running = False
        
        # 数据缓存
        self.data_cache = defaultdict(deque)
        self.cache_size = 1000
        
        # 性能统计
        self.stats = {
            'messages_processed': 0,
            'bytes_received': 0,
            'errors': 0,
            'start_time': None,
            'data_rates': defaultdict(list)
        }
        
        # Redis连接
        self.redis_client = None
        
        # 初始化数据源
        self._init_data_sources()
        
        logger.info("高频数据收集器初始化完成")
    
    def _init_data_sources(self):
        """初始化数据源"""
        # 币安WebSocket
        binance_config = config.get('data_sources', {}).get('binance', {})
        if binance_config.get('enabled', True):
            self.data_sources['binance_ws'] = BinanceWebSocketSource(binance_config)
    
    async def start(self, symbols: List[str] = None):
        """启动数据收集"""
        if self.is_running:
            logger.warning("数据收集器已在运行")
            return
        
        try:
            # 连接Redis
            await self._connect_redis()
            
            # 获取要监控的交易对
            if not symbols:
                symbols = config.get('trading', {}).get('symbols', ['BTCUSDT', 'ETHUSDT'])
            
            # 启动所有数据源
            for source_name, source in self.data_sources.items():
                try:
                    # 设置数据回调
                    source._data_callback = self._on_data_received
                    
                    await source.connect()
                    await source.subscribe(symbols)
                    
                    logger.info(f"数据源 {source_name} 启动成功")
                    
                except Exception as e:
                    logger.error(f"启动数据源 {source_name} 失败: {e}")
            
            self.is_running = True
            self.stats['start_time'] = datetime.now()
            
            # 启动数据处理任务
            asyncio.create_task(self._data_processing_loop())
            asyncio.create_task(self._stats_reporting_loop())
            
            logger.info(f"高频数据收集器启动完成，监控 {len(symbols)} 个交易对")
            
        except Exception as e:
            logger.error(f"启动数据收集器失败: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """停止数据收集"""
        if not self.is_running:
            return
        
        logger.info("正在停止数据收集器...")
        self.is_running = False
        
        # 断开所有数据源
        for source in self.data_sources.values():
            try:
                await source.disconnect()
            except Exception as e:
                logger.error(f"断开数据源失败: {e}")
        
        # 关闭Redis连接
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("数据收集器已停止")
    
    async def _connect_redis(self):
        """连接Redis"""
        try:
            redis_config = config.get('redis', {})
            self.redis_client = await aioredis.from_url(
                f"redis://{redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}",
                db=redis_config.get('db', 0),
                password=redis_config.get('password')
            )
            
            # 测试连接
            await self.redis_client.ping()
            logger.info("Redis连接成功")
            
        except Exception as e:
            logger.warning(f"Redis连接失败: {e}")
            self.redis_client = None
    
    async def _on_data_received(self, stream: str, data: dict):
        """数据接收回调"""
        try:
            # 解析数据类型和交易对
            stream_parts = stream.split('@')
            if len(stream_parts) != 2:
                return
            
            symbol = stream_parts[0].upper()
            data_type = stream_parts[1]
            
            # 标准化数据格式
            standardized_data = await self._standardize_data(symbol, data_type, data)
            
            if standardized_data:
                # 添加到缓存
                cache_key = f"{symbol}_{data_type}"
                self.data_cache[cache_key].append(standardized_data)
                
                # 控制缓存大小
                if len(self.data_cache[cache_key]) > self.cache_size:
                    self.data_cache[cache_key].popleft()
                
                # 发送到Redis
                if self.redis_client:
                    await self._publish_to_redis(cache_key, standardized_data)
                
                # 调用注册的回调函数
                for callback in self.data_callbacks:
                    try:
                        await callback(symbol, data_type, standardized_data)
                    except Exception as e:
                        logger.error(f"数据回调错误: {e}")
                
                # 更新统计
                self.stats['messages_processed'] += 1
                self.stats['data_rates'][data_type].append(time.time())
        
        except Exception as e:
            logger.error(f"处理数据时出错: {e}")
            self.stats['errors'] += 1
    
    async def _standardize_data(self, symbol: str, data_type: str, raw_data: dict) -> Optional[dict]:
        """标准化数据格式"""
        timestamp = datetime.now()
        
        try:
            if data_type == 'ticker':
                return {
                    'type': 'ticker',
                    'symbol': symbol,
                    'timestamp': timestamp.isoformat(),
                    'price': float(raw_data.get('c', 0)),
                    'price_change': float(raw_data.get('P', 0)),
                    'volume': float(raw_data.get('v', 0)),
                    'high': float(raw_data.get('h', 0)),
                    'low': float(raw_data.get('l', 0)),
                    'open': float(raw_data.get('o', 0))
                }
            
            elif data_type == 'bookTicker':
                return {
                    'type': 'book_ticker',
                    'symbol': symbol,
                    'timestamp': timestamp.isoformat(),
                    'bid_price': float(raw_data.get('b', 0)),
                    'bid_qty': float(raw_data.get('B', 0)),
                    'ask_price': float(raw_data.get('a', 0)),
                    'ask_qty': float(raw_data.get('A', 0))
                }
            
            elif data_type == 'trade':
                return {
                    'type': 'trade',
                    'symbol': symbol,
                    'timestamp': timestamp.isoformat(),
                    'price': float(raw_data.get('p', 0)),
                    'quantity': float(raw_data.get('q', 0)),
                    'is_buyer_maker': raw_data.get('m', False),
                    'trade_id': raw_data.get('t', 0)
                }
            
            elif data_type.startswith('depth'):
                return {
                    'type': 'depth',
                    'symbol': symbol,
                    'timestamp': timestamp.isoformat(),
                    'bids': [[float(bid[0]), float(bid[1])] for bid in raw_data.get('bids', [])],
                    'asks': [[float(ask[0]), float(ask[1])] for ask in raw_data.get('asks', [])]
                }
            
        except (KeyError, ValueError) as e:
            logger.warning(f"数据标准化失败 {symbol}_{data_type}: {e}")
            return None
        
        return None
    
    async def _publish_to_redis(self, key: str, data: dict):
        """发布数据到Redis"""
        try:
            await self.redis_client.publish(f"market_data:{key}", json.dumps(data))
        except Exception as e:
            logger.warning(f"发布到Redis失败: {e}")
    
    async def _data_processing_loop(self):
        """数据处理循环"""
        while self.is_running:
            try:
                # 批量处理缓存数据
                for cache_key, cache_queue in self.data_cache.items():
                    if len(cache_queue) >= 10:  # 批量处理阈值
                        batch_data = []
                        for _ in range(min(10, len(cache_queue))):
                            if cache_queue:
                                batch_data.append(cache_queue.popleft())
                        
                        if batch_data:
                            await self._process_batch_data(cache_key, batch_data)
                
                await asyncio.sleep(0.1)  # 100ms间隔
                
            except Exception as e:
                logger.error(f"数据处理循环错误: {e}")
                await asyncio.sleep(1)
    
    async def _process_batch_data(self, cache_key: str, batch_data: List[dict]):
        """批量处理数据"""
        try:
            # 这里可以进行批量数据持久化
            # 或发送给数据处理引擎
            pass
            
        except Exception as e:
            logger.error(f"批量处理数据失败 {cache_key}: {e}")
    
    async def _stats_reporting_loop(self):
        """统计报告循环"""
        while self.is_running:
            try:
                await self._report_stats()
                await asyncio.sleep(60)  # 每分钟报告一次
            except Exception as e:
                logger.error(f"统计报告错误: {e}")
                await asyncio.sleep(60)
    
    async def _report_stats(self):
        """报告统计信息"""
        if not self.stats['start_time']:
            return
        
        uptime = datetime.now() - self.stats['start_time']
        
        # 计算数据速率
        current_time = time.time()
        rates = {}
        for data_type, timestamps in self.stats['data_rates'].items():
            # 只保留最近1分钟的数据
            recent_timestamps = [ts for ts in timestamps if current_time - ts < 60]
            rates[data_type] = len(recent_timestamps)
            self.stats['data_rates'][data_type] = recent_timestamps
        
        # 数据源状态
        source_stats = {}
        for name, source in self.data_sources.items():
            source_stats[name] = {
                'connected': source.is_connected,
                'error_count': source.error_count,
                'data_count': source.data_count,
                'last_update': source.last_update.isoformat() if source.last_update else None
            }
        
        logger.info(f"数据收集统计 - 运行时间: {uptime}, "
                   f"消息处理: {self.stats['messages_processed']}, "
                   f"错误: {self.stats['errors']}, "
                   f"数据速率: {rates}")
    
    def register_callback(self, callback: Callable):
        """注册数据回调函数"""
        self.data_callbacks.append(callback)
    
    def get_cached_data(self, symbol: str, data_type: str, limit: int = 100) -> List[dict]:
        """获取缓存数据"""
        cache_key = f"{symbol}_{data_type}"
        cache_queue = self.data_cache.get(cache_key, deque())
        return list(cache_queue)[-limit:]
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'is_running': self.is_running,
            'stats': self.stats.copy(),
            'sources': {name: {
                'connected': source.is_connected,
                'error_count': source.error_count,
                'data_count': source.data_count,
                'last_update': source.last_update.isoformat() if source.last_update else None
            } for name, source in self.data_sources.items()},
            'cache_sizes': {key: len(queue) for key, queue in self.data_cache.items()}
        }

# 全局实例
_hf_collector_instance = None

def get_hf_data_collector(data_manager: DataManager = None) -> HighFrequencyDataCollector:
    """获取高频数据收集器实例"""
    global _hf_collector_instance
    if _hf_collector_instance is None:
        if data_manager is None:
            data_manager = DataManager()
        _hf_collector_instance = HighFrequencyDataCollector(data_manager)
    return _hf_collector_instance