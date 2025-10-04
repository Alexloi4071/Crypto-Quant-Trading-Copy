"""
Plugin Manager
插件管理器，支持动态加载和管理交易策略插件、指标插件等
提供插件生命周期管理和热加载功能
"""

import importlib
import importlib.util
import inspect
import sys
from typing import Dict, List, Optional, Any, Type, Callable
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
import time
import json
import hashlib

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class PluginType(Enum):
    """插件类型"""
    STRATEGY = "strategy"
    INDICATOR = "indicator"
    SIGNAL = "signal"
    RISK_MANAGEMENT = "risk_management"
    DATA_SOURCE = "data_source"
    NOTIFICATION = "notification"
    EXTENSION = "extension"

class PluginStatus(Enum):
    """插件状态"""
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UNLOADED = "unloaded"

@dataclass

class PluginInfo:
    """插件信息"""
    plugin_id: str
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType

    # 文件信息
    file_path: Path
    module_name: str
    class_name: str

    # 状态信息
    status: PluginStatus = PluginStatus.LOADED
    load_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)

    # 依赖信息
    dependencies: List[str] = field(default_factory=list)
    required_version: str = "1.0.0"

    # 配置信息
    config_schema: Dict[str, Any] = field(default_factory=dict)
    default_config: Dict[str, Any] = field(default_factory=dict)

    # 运行时信息
    instance: Optional[Any] = None
    error_message: Optional[str] = None
    usage_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'plugin_id': self.plugin_id,
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'plugin_type': self.plugin_type.value,
            'file_path': str(self.file_path),
            'module_name': self.module_name,
            'class_name': self.class_name,
            'status': self.status.value,
            'load_time': self.load_time.isoformat(),
            'last_update': self.last_update.isoformat(),
            'dependencies': self.dependencies,
            'required_version': self.required_version,
            'config_schema': self.config_schema,
            'default_config': self.default_config,
            'error_message': self.error_message,
            'usage_count': self.usage_count,
            'performance_metrics': self.performance_metrics
        }

class PluginBase(ABC):
    """插件基类"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.plugin_id = self.__class__.__name__.lower()
        self.is_initialized = False
        self.performance_stats = {
            'call_count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'error_count': 0
        }

    @abstractmethod

    def get_info(self) -> Dict[str, Any]:
        """获取插件信息"""
        pass

    @abstractmethod

    def initialize(self) -> bool:
        """初始化插件"""
        pass

    @abstractmethod

    def cleanup(self) -> bool:
        """清理插件"""
        pass

    def get_config_schema(self) -> Dict[str, Any]:
        """获取配置模式"""
        return {}

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置"""
        return True

    def update_config(self, config: Dict[str, Any]) -> bool:
        """更新配置"""
        if self.validate_config(config):
            self.config.update(config)
            return True
        return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.performance_stats.copy()

class PluginRegistry:
    """插件注册表"""

    def __init__(self):
        self.plugins: Dict[str, PluginInfo] = {}
        self.plugin_types: Dict[PluginType, List[str]] = defaultdict(list)
        self.dependencies: Dict[str, List[str]] = defaultdict(list)
        self.reverse_dependencies: Dict[str, List[str]] = defaultdict(list)

    def register_plugin(self, plugin_info: PluginInfo):
        """注册插件"""
        plugin_id = plugin_info.plugin_id

        # 检查是否已存在
        if plugin_id in self.plugins:
            logger.warning(f"插件 {plugin_id} 已存在，将被覆盖")

        # 注册插件
        self.plugins[plugin_id] = plugin_info
        self.plugin_types[plugin_info.plugin_type].append(plugin_id)

        # 注册依赖关系
        for dep in plugin_info.dependencies:
            self.dependencies[plugin_id].append(dep)
            self.reverse_dependencies[dep].append(plugin_id)

        logger.info(f"注册插件: {plugin_id} ({plugin_info.plugin_type.value})")

    def unregister_plugin(self, plugin_id: str):
        """注销插件"""
        if plugin_id not in self.plugins:
            return

        plugin_info = self.plugins[plugin_id]

        # 移除类型索引
        if plugin_id in self.plugin_types[plugin_info.plugin_type]:
            self.plugin_types[plugin_info.plugin_type].remove(plugin_id)

        # 移除依赖关系
        for dep in plugin_info.dependencies:
            if plugin_id in self.reverse_dependencies[dep]:
                self.reverse_dependencies[dep].remove(plugin_id)

        # 移除反向依赖
        for dependent in self.reverse_dependencies[plugin_id]:
            if plugin_id in self.dependencies[dependent]:
                self.dependencies[dependent].remove(plugin_id)

        # 删除插件
        del self.plugins[plugin_id]

        logger.info(f"注销插件: {plugin_id}")

    def get_plugin(self, plugin_id: str) -> Optional[PluginInfo]:
        """获取插件信息"""
        return self.plugins.get(plugin_id)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInfo]:
        """按类型获取插件"""
        plugin_ids = self.plugin_types.get(plugin_type, [])
        return [self.plugins[pid] for pid in plugin_ids if pid in self.plugins]

    def get_active_plugins(self) -> List[PluginInfo]:
        """获取活跃插件"""
        return [p for p in self.plugins.values() if p.status == PluginStatus.ACTIVE]

    def check_dependencies(self, plugin_id: str) -> List[str]:
        """检查依赖是否满足"""
        missing_deps = []

        if plugin_id not in self.plugins:
            return [f"插件 {plugin_id} 不存在"]

        plugin = self.plugins[plugin_id]

        for dep in plugin.dependencies:
            if dep not in self.plugins:
                missing_deps.append(f"缺少依赖插件: {dep}")
            elif self.plugins[dep].status != PluginStatus.ACTIVE:
                missing_deps.append(f"依赖插件未激活: {dep}")

        return missing_deps

class PluginLoader:
    """插件加载器"""

    def __init__(self, plugin_paths: List[Path]):
        self.plugin_paths = plugin_paths
        self.loaded_modules = {}
        self.file_timestamps = {}

    def discover_plugins(self) -> List[Dict[str, Any]]:
        """发现插件"""
        plugins = []

        for plugin_path in self.plugin_paths:
            if not plugin_path.exists():
                logger.warning(f"插件路径不存在: {plugin_path}")
                continue

            # 搜索Python文件
            for py_file in plugin_path.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                try:
                    plugin_info = self._extract_plugin_info(py_file)
                    if plugin_info:
                        plugins.append(plugin_info)
                except Exception as e:
                    logger.error(f"解析插件文件失败 {py_file}: {e}")

        logger.info(f"发现 {len(plugins)} 个插件")
        return plugins

    def _extract_plugin_info(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """提取插件信息"""
        # 读取文件内容
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            return None

        # 查找插件标记
        if "PluginBase" not in content:
            return None

        # 动态导入模块
        module_name = f"plugin_{file_path.stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 查找插件类
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, PluginBase) and
                    obj is not PluginBase):
                    plugin_class = obj
                    break

            if not plugin_class:
                return None

            # 创建临时实例获取信息
            temp_instance = plugin_class()
            plugin_info = temp_instance.get_info()

            return {
                'file_path': file_path,
                'module_name': module_name,
                'class_name': plugin_class.__name__,
                'plugin_info': plugin_info,
                'plugin_class': plugin_class
            }

        except Exception as e:
            logger.error(f"导入插件模块失败 {file_path}: {e}")
            return None

    def load_plugin(self, plugin_discovery: Dict[str, Any]) -> Optional[PluginInfo]:
        """加载插件"""
        try:
            file_path = plugin_discovery['file_path']
            module_name = plugin_discovery['module_name']
            class_name = plugin_discovery['class_name']
            info = plugin_discovery['plugin_info']
            plugin_class = plugin_discovery['plugin_class']

            # 创建插件信息对象
            plugin_info = PluginInfo(
                plugin_id=info.get('id', class_name.lower()),
                name=info.get('name', class_name),
                version=info.get('version', '1.0.0'),
                description=info.get('description', ''),
                author=info.get('author', 'Unknown'),
                plugin_type=PluginType(info.get('type', 'extension')),
                file_path=file_path,
                module_name=module_name,
                class_name=class_name,
                dependencies=info.get('dependencies', []),
                required_version=info.get('required_version', '1.0.0'),
                config_schema=info.get('config_schema', {}),
                default_config=info.get('default_config', {})
            )

            # 记录文件时间戳
            self.file_timestamps[str(file_path)] = file_path.stat().st_mtime

            return plugin_info

        except Exception as e:
            logger.error(f"加载插件失败: {e}")
            return None

    def check_file_changes(self) -> List[str]:
        """检查文件变化"""
        changed_files = []

        for file_path_str, old_timestamp in self.file_timestamps.items():
            file_path = Path(file_path_str)

            if file_path.exists():
                current_timestamp = file_path.stat().st_mtime
                if current_timestamp > old_timestamp:
                    changed_files.append(file_path_str)
                    self.file_timestamps[file_path_str] = current_timestamp
            else:
                # 文件被删除
                changed_files.append(file_path_str)
                del self.file_timestamps[file_path_str]

        return changed_files

class HookManager:
    """钩子管理器"""

    def __init__(self):
        self.hooks: Dict[str, List[Callable]] = defaultdict(list)
        self.hook_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'calls': 0, 'errors': 0})

    def register_hook(self, event: str, callback: Callable, plugin_id: str = None):
        """注册钩子"""
        # 添加插件ID信息
        callback._plugin_id = plugin_id
        self.hooks[event].append(callback)

        logger.debug(f"注册钩子: {event} <- {plugin_id or 'unknown'}")

    def unregister_hook(self, event: str, callback: Callable):
        """注销钩子"""
        if callback in self.hooks[event]:
            self.hooks[event].remove(callback)
            logger.debug(f"注销钩子: {event}")

    def unregister_plugin_hooks(self, plugin_id: str):
        """注销插件的所有钩子"""
        removed_count = 0

        for event, callbacks in self.hooks.items():
            to_remove = [cb for cb in callbacks if getattr(cb, '_plugin_id', None) == plugin_id]
            for cb in to_remove:
                callbacks.remove(cb)
                removed_count += 1

        if removed_count > 0:
            logger.info(f"注销插件 {plugin_id} 的 {removed_count} 个钩子")

    def trigger_hook(self, event: str, *args, **kwargs) -> List[Any]:
        """触发钩子"""
        results = []

        if event not in self.hooks:
            return results

        self.hook_stats[event]['calls'] += 1

        for callback in self.hooks[event]:
            try:
                start_time = time.time()
                result = callback(*args, **kwargs)
                execution_time = time.time() - start_time

                results.append({
                    'plugin_id': getattr(callback, '_plugin_id', 'unknown'),
                    'result': result,
                    'execution_time': execution_time
                })

            except Exception as e:
                self.hook_stats[event]['errors'] += 1
                logger.error(f"钩子执行失败 {event}: {e}")

                results.append({
                    'plugin_id': getattr(callback, '_plugin_id', 'unknown'),
                    'error': str(e),
                    'execution_time': 0
                })

        return results

    def get_hook_stats(self) -> Dict[str, Dict[str, int]]:
        """获取钩子统计"""
        return dict(self.hook_stats)

class PluginManager:
    """插件管理器主类"""

    def __init__(self, plugin_paths: List[str] = None):
        # 默认插件路径
        if plugin_paths is None:
            plugin_paths = [
                "plugins/strategies",
                "plugins/indicators",
                "plugins/signals",
                "plugins/extensions"
            ]

        self.plugin_paths = [Path(p) for p in plugin_paths]

        # 核心组件
        self.registry = PluginRegistry()
        self.loader = PluginLoader(self.plugin_paths)
        self.hook_manager = HookManager()

        # 插件实例缓存
        self.instances: Dict[str, PluginBase] = {}

        # 配置管理
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}

        # 热加载
        self.hot_reload_enabled = config.get('plugins', {}).get('hot_reload', False)
        self.reload_check_interval = 5.0  # 秒
        self.reload_thread = None
        self.should_stop_reload = False

        # 统计信息
        self.stats = {
            'total_plugins': 0,
            'active_plugins': 0,
            'failed_plugins': 0,
            'reload_count': 0,
            'hook_calls': 0,
            'hook_errors': 0
        }

        # 确保插件目录存在
        for path in self.plugin_paths:
            path.mkdir(parents=True, exist_ok=True)

        logger.info("插件管理器初始化完成")

    def start(self):
        """启动插件管理器"""
        # 发现和加载插件
        self.discover_and_load_plugins()

        # 启动热加载
        if self.hot_reload_enabled:
            self.start_hot_reload()

        logger.info(f"插件管理器启动完成，共加载 {len(self.registry.plugins)} 个插件")

    def stop(self):
        """停止插件管理器"""
        # 停止热加载
        if self.reload_thread:
            self.should_stop_reload = True
            self.reload_thread.join()

        # 卸载所有插件
        self.unload_all_plugins()

        logger.info("插件管理器已停止")

    def discover_and_load_plugins(self):
        """发现并加载插件"""
        # 发现插件
        discovered = self.loader.discover_plugins()

        # 加载插件信息
        for discovery in discovered:
            plugin_info = self.loader.load_plugin(discovery)
            if plugin_info:
                self.registry.register_plugin(plugin_info)
                self.stats['total_plugins'] += 1

        # 按依赖顺序激活插件
        self.activate_plugins_by_dependency_order()

    def activate_plugins_by_dependency_order(self):
        """按依赖顺序激活插件"""
        activated = set()
        to_activate = list(self.registry.plugins.keys())

        while to_activate:
            # 查找可以激活的插件（依赖已满足）
            can_activate = []

            for plugin_id in to_activate:
                missing_deps = self.registry.check_dependencies(plugin_id)
                if not missing_deps:
                    can_activate.append(plugin_id)

            if not can_activate:
                # 循环依赖或缺少依赖
                for plugin_id in to_activate:
                    missing_deps = self.registry.check_dependencies(plugin_id)
                    logger.error(f"无法激活插件 {plugin_id}: {missing_deps}")

                    plugin_info = self.registry.get_plugin(plugin_id)
                    if plugin_info:
                        plugin_info.status = PluginStatus.ERROR
                        plugin_info.error_message = "; ".join(missing_deps)
                        self.stats['failed_plugins'] += 1

                break

            # 激活可激活的插件
            for plugin_id in can_activate:
                if self.activate_plugin(plugin_id):
                    activated.add(plugin_id)
                    to_activate.remove(plugin_id)
                else:
                    to_activate.remove(plugin_id)
                    self.stats['failed_plugins'] += 1

    def activate_plugin(self, plugin_id: str) -> bool:
        """激活插件"""
        plugin_info = self.registry.get_plugin(plugin_id)
        if not plugin_info:
            logger.error(f"插件不存在: {plugin_id}")
            return False

        try:
            # 检查依赖
            missing_deps = self.registry.check_dependencies(plugin_id)
            if missing_deps:
                logger.error(f"插件 {plugin_id} 依赖未满足: {missing_deps}")
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = "; ".join(missing_deps)
                return False

            # 创建插件实例
            if plugin_id not in self.instances:
                # 动态导入模块
                spec = importlib.util.spec_from_file_location(
                    plugin_info.module_name, plugin_info.file_path
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"无法导入模块: {plugin_info.module_name}")

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # 获取插件类
                plugin_class = getattr(module, plugin_info.class_name)

                # 创建实例
                plugin_config = self.plugin_configs.get(plugin_id, plugin_info.default_config)
                instance = plugin_class(plugin_config)

                # 初始化插件
                if not instance.initialize():
                    raise RuntimeError("插件初始化失败")

                self.instances[plugin_id] = instance
                plugin_info.instance = instance

            # 更新状态
            plugin_info.status = PluginStatus.ACTIVE
            plugin_info.last_update = datetime.now()
            plugin_info.error_message = None

            self.stats['active_plugins'] += 1

            # 触发激活钩子
            self.hook_manager.trigger_hook('plugin_activated', plugin_id, plugin_info)

            logger.info(f"插件 {plugin_id} 激活成功")
            return True

        except Exception as e:
            logger.error(f"激活插件失败 {plugin_id}: {e}")
            plugin_info.status = PluginStatus.ERROR
            plugin_info.error_message = str(e)
            return False

    def deactivate_plugin(self, plugin_id: str) -> bool:
        """停用插件"""
        plugin_info = self.registry.get_plugin(plugin_id)
        if not plugin_info:
            return False

        try:
            # 检查反向依赖
            dependents = self.registry.reverse_dependencies.get(plugin_id, [])
            active_dependents = [
                dep for dep in dependents
                if self.registry.get_plugin(dep) and
                   self.registry.get_plugin(dep).status == PluginStatus.ACTIVE
            ]

            if active_dependents:
                logger.warning(f"插件 {plugin_id} 被以下插件依赖: {active_dependents}")
                # 先停用依赖插件
                for dep in active_dependents:
                    self.deactivate_plugin(dep)

            # 清理插件实例
            if plugin_id in self.instances:
                instance = self.instances[plugin_id]
                instance.cleanup()
                del self.instances[plugin_id]

            # 注销钩子
            self.hook_manager.unregister_plugin_hooks(plugin_id)

            # 更新状态
            plugin_info.status = PluginStatus.INACTIVE
            plugin_info.instance = None
            plugin_info.last_update = datetime.now()

            if plugin_info.status == PluginStatus.ACTIVE:
                self.stats['active_plugins'] -= 1

            # 触发停用钩子
            self.hook_manager.trigger_hook('plugin_deactivated', plugin_id, plugin_info)

            logger.info(f"插件 {plugin_id} 已停用")
            return True

        except Exception as e:
            logger.error(f"停用插件失败 {plugin_id}: {e}")
            return False

    def reload_plugin(self, plugin_id: str) -> bool:
        """重新加载插件"""
        # 先停用
        self.deactivate_plugin(plugin_id)

        # 重新发现和加载
        plugin_info = self.registry.get_plugin(plugin_id)
        if plugin_info:
            discovery = self.loader._extract_plugin_info(plugin_info.file_path)
            if discovery:
                new_plugin_info = self.loader.load_plugin(discovery)
                if new_plugin_info:
                    # 保留一些旧信息
                    new_plugin_info.usage_count = plugin_info.usage_count
                    new_plugin_info.performance_metrics = plugin_info.performance_metrics

                    # 更新注册表
                    self.registry.register_plugin(new_plugin_info)

                    self.stats['reload_count'] += 1

                    # 重新激活
                    return self.activate_plugin(plugin_id)

        return False

    def unload_all_plugins(self):
        """卸载所有插件"""
        for plugin_id in list(self.registry.plugins.keys()):
            self.deactivate_plugin(plugin_id)

        self.registry.plugins.clear()
        self.registry.plugin_types.clear()
        self.instances.clear()

        logger.info("所有插件已卸载")

    def get_plugin_instance(self, plugin_id: str) -> Optional[PluginBase]:
        """获取插件实例"""
        return self.instances.get(plugin_id)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginBase]:
        """按类型获取插件实例"""
        plugin_infos = self.registry.get_plugins_by_type(plugin_type)
        instances = []

        for plugin_info in plugin_infos:
            if (plugin_info.status == PluginStatus.ACTIVE and
                plugin_info.plugin_id in self.instances):
                instances.append(self.instances[plugin_info.plugin_id])

        return instances

    def update_plugin_config(self, plugin_id: str, config: Dict[str, Any]) -> bool:
        """更新插件配置"""
        if plugin_id in self.instances:
            instance = self.instances[plugin_id]
            if instance.update_config(config):
                self.plugin_configs[plugin_id] = config
                return True

        return False

    def start_hot_reload(self):
        """启动热加载"""
        if self.reload_thread is not None:
            return

        self.should_stop_reload = False
        self.reload_thread = threading.Thread(target=self._hot_reload_loop)
        self.reload_thread.daemon = True
        self.reload_thread.start()

        logger.info("插件热加载已启动")

    def _hot_reload_loop(self):
        """热加载循环"""
        while not self.should_stop_reload:
            try:
                # 检查文件变化
                changed_files = self.loader.check_file_changes()

                if changed_files:
                    logger.info(f"检测到插件文件变化: {changed_files}")

                    # 找到对应的插件并重新加载
                    for file_path_str in changed_files:
                        for plugin_id, plugin_info in self.registry.plugins.items():
                            if str(plugin_info.file_path) == file_path_str:
                                logger.info(f"重新加载插件: {plugin_id}")
                                self.reload_plugin(plugin_id)
                                break

                time.sleep(self.reload_check_interval)

            except Exception as e:
                logger.error(f"热加载检查失败: {e}")
                time.sleep(self.reload_check_interval)

    # Hook管理接口

    def register_hook(self, event: str, callback: Callable, plugin_id: str = None):
        """注册钩子"""
        self.hook_manager.register_hook(event, callback, plugin_id)

    def trigger_hook(self, event: str, *args, **kwargs) -> List[Any]:
        """触发钩子"""
        results = self.hook_manager.trigger_hook(event, *args, **kwargs)
        self.stats['hook_calls'] += 1
        self.stats['hook_errors'] += sum(1 for r in results if 'error' in r)
        return results

    def get_plugin_list(self) -> List[Dict[str, Any]]:
        """获取插件列表"""
        return [plugin.to_dict() for plugin in self.registry.plugins.values()]

    def get_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """获取插件详细信息"""
        plugin_info = self.registry.get_plugin(plugin_id)
        if plugin_info:
            info_dict = plugin_info.to_dict()

            # 添加实时统计
            if plugin_id in self.instances:
                instance = self.instances[plugin_id]
                info_dict['performance_stats'] = instance.get_performance_stats()

            return info_dict

        return None

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 更新统计
        self.stats['active_plugins'] = len([
            p for p in self.registry.plugins.values()
            if p.status == PluginStatus.ACTIVE
        ])

        hook_stats = self.hook_manager.get_hook_stats()

        return {
            'stats': self.stats,
            'plugin_paths': [str(p) for p in self.plugin_paths],
            'hot_reload_enabled': self.hot_reload_enabled,
            'hook_stats': hook_stats,
            'plugins_by_type': {
                ptype.value: len(plugins)
                for ptype, plugins in self.registry.plugin_types.items()
            },
            'plugins_by_status': {
                status.value: len([
                    p for p in self.registry.plugins.values()
                    if p.status == status
                ])
                for status in PluginStatus
            }
        }

# 全局实例
_plugin_manager_instance = None

def get_plugin_manager() -> PluginManager:
    """获取插件管理器实例"""
    global _plugin_manager_instance
    if _plugin_manager_instance is None:
        _plugin_manager_instance = PluginManager()
    return _plugin_manager_instance
