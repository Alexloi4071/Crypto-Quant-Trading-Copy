"""
Stream Data Processing Engine
数据流处理引擎，负责实时数据的复杂处理和转换
集成机器学习模型和特征工程，提供高性能流式计算
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime, timedelta
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import time
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.features.feature_engineering import FeatureEngineer
from src.models.model_manager import ModelManager
from src.data.preprocessor import DataPreprocessor
from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class DataWindow:
    """滑动数据窗口"""
    
    def __init__(self, size: int, name: str = ""):
        self.size = size
        self.name = name
        self.data = deque(maxlen=size)
        self.last_update = None
        
    def add(self, item: Any):
        """添加数据到窗口"""
        self.data.append(item)
        self.last_update = datetime.now()
        
    def get_data(self) -> List[Any]:
        """获取窗口数据"""
        return list(self.data)
        
    def is_full(self) -> bool:
        """检查窗口是否已满"""
        return len(self.data) >= self.size
        
    def clear(self):
        """清空窗口"""
        self.data.clear()
        
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        if not self.data:
            return pd.DataFrame()
        
        try:
            return pd.DataFrame(list(self.data))
        except Exception as e:
            logger.warning(f"转换窗口数据为DataFrame失败: {e}")
            return pd.DataFrame()

class StreamProcessor:
    """流处理器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_active = False
        self.processed_count = 0
        self.error_count = 0
        self.last_process_time = None
        
    async def process(self, data: Any) -> Optional[Any]:
        """处理单条数据"""
        raise NotImplementedError
        
    async def process_batch(self, batch_data: List[Any]) -> List[Any]:
        """批量处理数据"""
        results = []
        for data in batch_data:
            try:
                result = await self.process(data)
                if result is not None:
                    results.append(result)
                self.processed_count += 1
            except Exception as e:
                logger.error(f"处理数据失败 {self.name}: {e}")
                self.error_count += 1
        
        self.last_process_time = datetime.now()
        return results
    
    def get_stats(self) -> dict:
        """获取处理器统计信息"""
        return {
            'name': self.name,
            'is_active': self.is_active,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'last_process_time': self.last_process_time.isoformat() if self.last_process_time else None
        }

class OHLCVProcessor(StreamProcessor):
    """OHLCV数据流处理器"""
    
    def __init__(self, timeframe: str = '1m'):
        super().__init__(f"ohlcv_{timeframe}")
        self.timeframe = timeframe
        self.timeframe_seconds = self._parse_timeframe(timeframe)
        
        # 存储当前OHLCV数据
        self.current_candles = defaultdict(dict)
        
    def _parse_timeframe(self, timeframe: str) -> int:
        """解析时间框架为秒数"""
        timeframes = {
            '1s': 1, '5s': 5, '15s': 15, '30s': 30,
            '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600, '8h': 28800, '12h': 43200,
            '1d': 86400, '3d': 259200, '1w': 604800
        }
        return timeframes.get(timeframe, 60)
    
    async def process(self, data: dict) -> Optional[dict]:
        """处理交易数据生成OHLCV"""
        try:
            if data.get('type') != 'trade':
                return None
            
            symbol = data['symbol']
            price = data['price']
            quantity = data['quantity']
            timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', ''))
            
            # 计算时间窗口的开始时间
            window_start = self._get_window_start(timestamp)
            
            candle_key = f"{symbol}_{window_start.timestamp()}"
            
            # 初始化或更新蜡烛图数据
            if candle_key not in self.current_candles:
                self.current_candles[candle_key] = {
                    'symbol': symbol,
                    'timestamp': window_start,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': quantity,
                    'trade_count': 1,
                    'last_update': timestamp
                }
            else:
                candle = self.current_candles[candle_key]
                candle['high'] = max(candle['high'], price)
                candle['low'] = min(candle['low'], price)
                candle['close'] = price
                candle['volume'] += quantity
                candle['trade_count'] += 1
                candle['last_update'] = timestamp
            
            # 检查是否需要完成当前蜡烛图
            current_time = datetime.now()
            completed_candles = []
            
            for key, candle in list(self.current_candles.items()):
                candle_end = candle['timestamp'] + timedelta(seconds=self.timeframe_seconds)
                if current_time >= candle_end:
                    completed_candles.append(candle)
                    del self.current_candles[key]
            
            if completed_candles:
                return {
                    'type': 'ohlcv_batch',
                    'timeframe': self.timeframe,
                    'candles': completed_candles
                }
            
            return None
            
        except Exception as e:
            logger.error(f"OHLCV处理错误: {e}")
            return None
    
    def _get_window_start(self, timestamp: datetime) -> datetime:
        """获取时间窗口开始时间"""
        # 将时间戳对齐到时间框架边界
        total_seconds = int(timestamp.timestamp())
        aligned_seconds = (total_seconds // self.timeframe_seconds) * self.timeframe_seconds
        return datetime.fromtimestamp(aligned_seconds)

class FeatureProcessor(StreamProcessor):
    """特征工程处理器"""
    
    def __init__(self, feature_config: dict = None):
        super().__init__("feature_processor")
        self.feature_engineer = FeatureEngineer(feature_config)
        
        # 数据窗口（用于特征计算）
        self.price_windows = defaultdict(lambda: DataWindow(200))  # 价格窗口
        self.volume_windows = defaultdict(lambda: DataWindow(200))  # 成交量窗口
        
        # 计算所需的最小数据量
        self.min_data_points = 50
        
    async def process(self, data: dict) -> Optional[dict]:
        """处理数据生成特征"""
        try:
            if data.get('type') == 'ohlcv_batch':
                results = []
                for candle in data['candles']:
                    feature_data = await self._process_candle(candle)
                    if feature_data:
                        results.append(feature_data)
                
                return {
                    'type': 'features_batch',
                    'features': results
                } if results else None
            
            elif data.get('type') == 'ticker':
                return await self._process_ticker(data)
            
            return None
            
        except Exception as e:
            logger.error(f"特征处理错误: {e}")
            return None
    
    async def _process_candle(self, candle: dict) -> Optional[dict]:
        """处理单个蜡烛图数据"""
        symbol = candle['symbol']
        
        # 添加到数据窗口
        price_data = {
            'timestamp': candle['timestamp'],
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close']
        }
        
        volume_data = {
            'timestamp': candle['timestamp'],
            'volume': candle['volume']
        }
        
        self.price_windows[symbol].add(price_data)
        self.volume_windows[symbol].add(volume_data)
        
        # 检查是否有足够数据进行特征计算
        if not self.price_windows[symbol].is_full() or len(self.price_windows[symbol].data) < self.min_data_points:
            return None
        
        # 转换为DataFrame进行特征计算
        price_df = self.price_windows[symbol].to_dataframe()
        volume_df = self.volume_windows[symbol].to_dataframe()
        
        if price_df.empty or volume_df.empty:
            return None
        
        # 生成特征
        try:
            features, metadata = self.feature_engineer.generate_features(price_df)
            
            if features is not None and not features.empty:
                # 获取最新的特征行
                latest_features = features.iloc[-1].to_dict()
                
                return {
                    'symbol': symbol,
                    'timestamp': candle['timestamp'],
                    'candle_data': candle,
                    'features': latest_features,
                    'feature_metadata': metadata
                }
        
        except Exception as e:
            logger.warning(f"生成特征失败 {symbol}: {e}")
            
        return None
    
    async def _process_ticker(self, ticker_data: dict) -> Optional[dict]:
        """处理ticker数据生成快速特征"""
        try:
            symbol = ticker_data['symbol']
            
            # 简单的实时特征
            features = {
                'price': ticker_data['price'],
                'price_change_pct': ticker_data.get('price_change', 0),
                'volume': ticker_data.get('volume', 0),
                'high': ticker_data.get('high', 0),
                'low': ticker_data.get('low', 0),
                'volatility': abs(ticker_data.get('price_change', 0)),
                'momentum': ticker_data.get('price_change', 0) * ticker_data.get('volume', 0)
            }
            
            return {
                'type': 'realtime_features',
                'symbol': symbol,
                'timestamp': ticker_data['timestamp'],
                'features': features
            }
            
        except Exception as e:
            logger.error(f"处理ticker特征失败: {e}")
            return None

class ModelInferenceProcessor(StreamProcessor):
    """模型推理处理器"""
    
    def __init__(self, model_manager: ModelManager):
        super().__init__("model_inference")
        self.model_manager = model_manager
        self.loaded_models = {}
        
    async def process(self, data: dict) -> Optional[dict]:
        """执行模型推理"""
        try:
            if data.get('type') != 'features_batch':
                return None
            
            results = []
            for feature_data in data['features']:
                prediction = await self._make_prediction(feature_data)
                if prediction:
                    results.append(prediction)
            
            return {
                'type': 'predictions_batch',
                'predictions': results
            } if results else None
            
        except Exception as e:
            logger.error(f"模型推理错误: {e}")
            return None
    
    async def _make_prediction(self, feature_data: dict) -> Optional[dict]:
        """对单个特征数据进行预测"""
        try:
            symbol = feature_data['symbol']
            features = feature_data['features']
            
            # 获取或加载模型
            model = await self._get_model(symbol)
            if not model:
                return None
            
            # 准备特征数据
            feature_array = np.array([list(features.values())])
            
            # 进行预测
            prediction = model.predict(feature_array)
            confidence = model.predict_proba(feature_array) if hasattr(model, 'predict_proba') else None
            
            return {
                'symbol': symbol,
                'timestamp': feature_data['timestamp'],
                'prediction': float(prediction[0]) if isinstance(prediction, np.ndarray) else prediction,
                'confidence': float(confidence[0].max()) if confidence is not None else None,
                'features_used': list(features.keys()),
                'model_version': getattr(model, 'version', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            return None
    
    async def _get_model(self, symbol: str):
        """获取指定交易对的模型"""
        try:
            if symbol not in self.loaded_models:
                # 尝试加载模型
                model = self.model_manager.load_latest_model(symbol)
                if model:
                    self.loaded_models[symbol] = model
            
            return self.loaded_models.get(symbol)
            
        except Exception as e:
            logger.error(f"获取模型失败 {symbol}: {e}")
            return None

class StreamDataProcessingEngine:
    """数据流处理引擎主类"""
    
    def __init__(self):
        self.processors = {}
        self.processor_chains = {}
        self.is_running = False
        
        # 异步任务执行器
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 数据队列
        self.input_queue = asyncio.Queue(maxsize=10000)
        self.output_callbacks = []
        
        # 统计信息
        self.stats = {
            'messages_processed': 0,
            'processing_time_total': 0,
            'errors': 0,
            'start_time': None,
            'throughput_history': deque(maxlen=100)
        }
        
        # 初始化默认处理器
        self._init_default_processors()
        
        logger.info("数据流处理引擎初始化完成")
    
    def _init_default_processors(self):
        """初始化默认处理器"""
        # OHLCV处理器
        self.add_processor('ohlcv_1m', OHLCVProcessor('1m'))
        self.add_processor('ohlcv_5m', OHLCVProcessor('5m'))
        self.add_processor('ohlcv_1h', OHLCVProcessor('1h'))
        
        # 特征处理器
        self.add_processor('features', FeatureProcessor())
        
        # 如果有模型管理器，添加模型推理处理器
        try:
            model_manager = ModelManager()
            self.add_processor('model_inference', ModelInferenceProcessor(model_manager))
        except Exception as e:
            logger.warning(f"无法初始化模型推理处理器: {e}")
        
        # 设置默认处理链
        self.create_processor_chain('main_chain', [
            'ohlcv_1m', 'features', 'model_inference'
        ])
    
    def add_processor(self, name: str, processor: StreamProcessor):
        """添加处理器"""
        self.processors[name] = processor
        logger.info(f"已添加处理器: {name}")
    
    def remove_processor(self, name: str):
        """移除处理器"""
        if name in self.processors:
            del self.processors[name]
            logger.info(f"已移除处理器: {name}")
    
    def create_processor_chain(self, chain_name: str, processor_names: List[str]):
        """创建处理器链"""
        valid_processors = []
        for name in processor_names:
            if name in self.processors:
                valid_processors.append(name)
            else:
                logger.warning(f"处理器 {name} 不存在，跳过")
        
        if valid_processors:
            self.processor_chains[chain_name] = valid_processors
            logger.info(f"已创建处理器链 {chain_name}: {valid_processors}")
    
    async def start(self):
        """启动处理引擎"""
        if self.is_running:
            logger.warning("处理引擎已在运行")
            return
        
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        # 启动数据处理任务
        asyncio.create_task(self._processing_loop())
        asyncio.create_task(self._stats_monitoring_loop())
        
        logger.info("数据流处理引擎启动完成")
    
    async def stop(self):
        """停止处理引擎"""
        if not self.is_running:
            return
        
        logger.info("正在停止数据流处理引擎...")
        self.is_running = False
        
        # 等待队列清空
        while not self.input_queue.empty():
            await asyncio.sleep(0.1)
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        logger.info("数据流处理引擎已停止")
    
    async def process_data(self, data: dict, chain_name: str = 'main_chain') -> Optional[dict]:
        """处理单条数据"""
        if not self.is_running:
            logger.warning("处理引擎未启动")
            return None
        
        try:
            await self.input_queue.put((data, chain_name))
            return True
        except asyncio.QueueFull:
            logger.warning("处理队列已满，丢弃数据")
            return None
    
    async def _processing_loop(self):
        """数据处理循环"""
        while self.is_running:
            try:
                # 获取待处理数据
                data, chain_name = await asyncio.wait_for(
                    self.input_queue.get(), timeout=1.0
                )
                
                start_time = time.time()
                
                # 执行处理链
                result = await self._execute_processor_chain(data, chain_name)
                
                # 更新统计信息
                processing_time = time.time() - start_time
                self.stats['messages_processed'] += 1
                self.stats['processing_time_total'] += processing_time
                self.stats['throughput_history'].append(time.time())
                
                # 调用输出回调
                if result:
                    for callback in self.output_callbacks:
                        try:
                            await callback(result)
                        except Exception as e:
                            logger.error(f"输出回调错误: {e}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"数据处理循环错误: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(0.1)
    
    async def _execute_processor_chain(self, data: dict, chain_name: str) -> Optional[dict]:
        """执行处理器链"""
        if chain_name not in self.processor_chains:
            logger.warning(f"处理器链 {chain_name} 不存在")
            return None
        
        current_data = data
        processor_names = self.processor_chains[chain_name]
        
        for processor_name in processor_names:
            if processor_name not in self.processors:
                continue
                
            processor = self.processors[processor_name]
            
            try:
                # 执行处理器
                current_data = await processor.process(current_data)
                
                if current_data is None:
                    break  # 处理链中断
                    
            except Exception as e:
                logger.error(f"处理器 {processor_name} 执行失败: {e}")
                processor.error_count += 1
                break
        
        return current_data
    
    async def _stats_monitoring_loop(self):
        """统计监控循环"""
        while self.is_running:
            try:
                await self._report_stats()
                await asyncio.sleep(30)  # 每30秒报告一次
            except Exception as e:
                logger.error(f"统计监控错误: {e}")
                await asyncio.sleep(30)
    
    async def _report_stats(self):
        """报告统计信息"""
        if not self.stats['start_time']:
            return
        
        current_time = time.time()
        uptime = datetime.now() - self.stats['start_time']
        
        # 计算吞吐量
        recent_throughput = len([ts for ts in self.stats['throughput_history'] 
                               if current_time - ts < 60])
        
        # 计算平均处理时间
        avg_processing_time = 0
        if self.stats['messages_processed'] > 0:
            avg_processing_time = self.stats['processing_time_total'] / self.stats['messages_processed']
        
        # 处理器统计
        processor_stats = {name: processor.get_stats() 
                         for name, processor in self.processors.items()}
        
        logger.info(f"处理引擎统计 - 运行时间: {uptime}, "
                   f"处理消息: {self.stats['messages_processed']}, "
                   f"错误: {self.stats['errors']}, "
                   f"吞吐量: {recent_throughput}/min, "
                   f"平均处理时间: {avg_processing_time:.4f}s")
    
    def register_output_callback(self, callback: Callable):
        """注册输出回调函数"""
        self.output_callbacks.append(callback)
    
    def get_stats(self) -> dict:
        """获取引擎统计信息"""
        return {
            'is_running': self.is_running,
            'queue_size': self.input_queue.qsize(),
            'processors': {name: processor.get_stats() for name, processor in self.processors.items()},
            'processor_chains': self.processor_chains.copy(),
            'stats': self.stats.copy()
        }

# 全局实例
_processing_engine_instance = None

def get_stream_processing_engine() -> StreamDataProcessingEngine:
    """获取数据流处理引擎实例"""
    global _processing_engine_instance
    if _processing_engine_instance is None:
        _processing_engine_instance = StreamDataProcessingEngine()
    return _processing_engine_instance