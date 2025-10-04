# -*- coding: utf-8 -*-
"""
路径管理器 - 统一管理所有路径
支持变量引用和模板格式化
"""
import yaml
import re
from pathlib import Path
from typing import Dict, Optional, Any


class PathManager:
    """统一路径管理器（单例模式）"""
    
    _instance = None
    
    def __new__(cls, config_file: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_file: Optional[str] = None):
        # 避免重复初始化
        if self._initialized:
            return
        
        if config_file is None:
            config_file = Path(__file__).parent / 'paths.yaml'
        
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self._cache = {}  # 路径缓存
        self._initialized = True
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"路径配置文件不存在: {self.config_file}")
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _resolve_variables(self, value: str, max_depth: int = 10) -> str:
        """递归解析变量引用 ${...}"""
        if not isinstance(value, str) or '${' not in value:
            return value
        
        pattern = r'\$\{([^}]+)\}'
        depth = 0
        
        while '${' in value and depth < max_depth:
            def replacer(match):
                var_path = match.group(1)
                keys = var_path.split('.')
                
                result = self.config
                for key in keys:
                    if isinstance(result, dict) and key in result:
                        result = result[key]
                    else:
                        return match.group(0)  # 无法解析，保持原样
                
                if isinstance(result, str):
                    return result
                return str(result)
            
            new_value = re.sub(pattern, replacer, value)
            if new_value == value:  # 无变化，避免无限循环
                break
            value = new_value
            depth += 1
        
        return value
    
    def get(self, path_key: str, **kwargs) -> Path:
        """
        获取路径
        
        Args:
            path_key: 路径键，如 'data.root' 或 'templates.raw_ohlcv'
            **kwargs: 模板变量，如 symbol='BTCUSDT', timeframe='15m'
        
        Returns:
            Path对象
        
        Examples:
            >>> pm = PathManager()
            >>> pm.get('data.root')
            PosixPath('data')
            >>> pm.get('templates.raw_ohlcv', symbol='BTCUSDT', timeframe='15m')
            PosixPath('data/raw/BTCUSDT/BTCUSDT_15m_ohlcv.parquet')
        """
        # 检查缓存（仅对无参数的键缓存）
        cache_key = path_key if not kwargs else None
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 解析键路径
        keys = path_key.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"路径键不存在: {path_key}")
        
        if not isinstance(value, str):
            raise ValueError(f"路径值不是字符串: {path_key} = {value}")
        
        # 解析变量
        resolved = self._resolve_variables(value)
        
        # 格式化模板变量
        if kwargs:
            try:
                resolved = resolved.format(**kwargs)
            except KeyError as e:
                raise ValueError(f"模板变量缺失: {e}")
        
        result = Path(resolved)
        
        # 缓存结果（仅对无参数的）
        if cache_key:
            self._cache[cache_key] = result
        
        return result
    
    def ensure_exists(self, path_key: str, **kwargs) -> Path:
        """确保路径存在，如不存在则创建"""
        path = self.get(path_key, **kwargs)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_all_paths(self, prefix: str = None) -> Dict[str, Path]:
        """获取所有路径（可指定前缀过滤）"""
        paths = {}
        
        def traverse(d: Dict, current_prefix: str = ''):
            for key, value in d.items():
                full_key = f"{current_prefix}.{key}" if current_prefix else key
                
                if isinstance(value, dict) and key != 'templates':
                    traverse(value, full_key)
                elif isinstance(value, str) and '/' in value:
                    try:
                        paths[full_key] = self.get(full_key)
                    except:
                        pass
        
        traverse(self.config)
        
        if prefix:
            paths = {k: v for k, v in paths.items() if k.startswith(prefix)}
        
        return paths
    
    def reload(self):
        """重新加载配置（清除缓存）"""
        self.config = self._load_config()
        self._cache.clear()


# 全局单例实例
path_manager = PathManager()


# 使用示例
if __name__ == "__main__":
    pm = PathManager()
    
    # 获取简单路径
    data_root = pm.get('data.root')
    print(f"数据根目录: {data_root}")
    
    # 获取模板路径
    btc_15m_ohlcv = pm.get('templates.raw_ohlcv', symbol='BTCUSDT', timeframe='15m')
    print(f"BTC 15m数据: {btc_15m_ohlcv}")
    
    # 确保目录存在
    cleaned_dir = pm.ensure_exists('data.cleaned')
    print(f"清洗数据目录: {cleaned_dir}")
    
    # 获取所有optuna路径
    optuna_paths = pm.get_all_paths('optuna')
    print(f"\nOptuna路径:")
    for key, path in optuna_paths.items():
        print(f"  {key}: {path}")
