# -*- coding: utf-8 -*-
"""
Optuna版本管理器
管理所有Optuna優化結果的版本控制和比較
"""
import json
import logging
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import hashlib

from optuna_system.utils.io_utils import atomic_write_json


class OptunaVersionManager:
    """Optuna優化結果版本管理器"""

    def __init__(self, results_path: str = "results"):
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def create_new_version(self, base_version: Optional[str] = None) -> str:
        """創建新版本號"""
        existing_versions = self.list_versions()

        if not existing_versions:
            new_version = "v1"
        else:
            version_numbers = []
            for v in existing_versions:
                try:
                    num = int(v.replace('v', ''))
                    version_numbers.append(num)
                except ValueError:
                    continue

            if version_numbers:
                new_version = f"v{max(version_numbers) + 1}"
            else:
                new_version = "v1"

        # 創建新版本目錄
        version_path = self.results_path / new_version
        version_path.mkdir(exist_ok=True)

        self.logger.info(f"創建新版本: {new_version}")
        return new_version

    def save_results(self, version: str, module_results: Dict[str, Any]) -> bool:
        """保存優化結果到指定版本"""
        try:
            version_path = self.results_path / version
            version_path.mkdir(exist_ok=True)

            timestamp = datetime.now().isoformat()

            # 保存各模塊結果
            for module_name, result in module_results.items():
                if result and isinstance(result, dict):
                    result['saved_timestamp'] = timestamp
                    result['version'] = version

                    result_file = version_path / f"{module_name}_result.json"
                    atomic_write_json(result_file, result)

            # 創建統一配置文件
            unified_config = self._create_unified_config(module_results, version, timestamp)
            unified_file = version_path / "unified_config.json"
            atomic_write_json(unified_file, unified_config)

            self.logger.info(f"版本 {version} 結果保存成功")
            return True

        except Exception as e:
            self.logger.error(f"保存版本 {version} 失敗: {e}")
            return False

    def create_data_version(self, layer_name: str, params: Dict[str, Any], data_shape: Tuple[int, int]) -> str:
        """為物化資料生成唯一版本代碼"""
        try:
            params_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
        except TypeError:
            params_str = str(params)
        params_hash = hashlib.md5(params_str.encode('utf-8')).hexdigest()[:10]
        rows, cols = data_shape if isinstance(data_shape, (list, tuple)) and len(data_shape) == 2 else (0, 0)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"{layer_name}_{params_hash}_{rows}x{cols}_{timestamp}"

    def save_materialized_data_info(self, version: str, layer_name: str, data_path: str, metadata: Dict[str, Any]) -> bool:
        """保存物化資料版本資訊"""
        record = {
            "version": version,
            "layer_name": layer_name,
            "data_path": data_path,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata,
        }
        try:
            output_file = self.results_path / f"data_version_{version}.json"
            atomic_write_json(output_file, record)
            return True
        except Exception as e:
            self.logger.error(f"保存物化資料資訊失敗: {e}")
            return False

    def get_data_lineage(self, version: str) -> Dict[str, Any]:
        """讀取指定資料版本的血緣資訊"""
        output_file = self.results_path / f"data_version_{version}.json"
        if not output_file.exists():
            return {}
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"讀取資料版本資訊失敗: {e}")
            return {}

    def _create_unified_config(self, module_results: Dict[str, Any],
                             version: str, timestamp: str) -> Dict[str, Any]:
        """創建統一配置"""
        unified_config = {
            'version': version,
            'timestamp': timestamp,
            'kelly_params': {},
            'ensemble_params': {},
            'polynomial_params': {},
            'confidence_params': {}
        }

        for module_name, result in module_results.items():
            if isinstance(result, dict) and 'best_params' in result:
                if module_name == 'kelly':
                    unified_config['kelly_params'] = result['best_params']
                elif module_name == 'ensemble':
                    unified_config['ensemble_params'] = result['best_params']
                elif module_name == 'polynomial':
                    unified_config['polynomial_params'] = result['best_params']
                elif module_name == 'confidence':
                    unified_config['confidence_params'] = result['best_params']

        return unified_config

    def load_version(self, version: str) -> Optional[Dict[str, Any]]:
        """加載指定版本的結果"""
        if version == 'latest':
            version = self.get_latest_version()
            if not version:
                return None

        version_path = self.results_path / version
        if not version_path.exists():
            return None

        try:
            results = {}
            unified_file = version_path / "unified_config.json"
            if unified_file.exists():
                with open(unified_file, 'r', encoding='utf-8') as f:
                    results['unified_config'] = json.load(f)

            return results
        except Exception as e:
            self.logger.error(f"加載版本 {version} 失敗: {e}")
            return None

    def set_latest(self, version: str) -> bool:
        """設置最新版本"""
        try:
            latest_file = self.results_path / "latest.txt"
            with open(latest_file, 'w', encoding='utf-8') as f:
                f.write(version)
            return True
        except Exception as e:
            self.logger.error(f"設置最新版本失敗: {e}")
            return False

    def get_latest_version(self) -> Optional[str]:
        """獲取最新版本號"""
        latest_file = self.results_path / "latest.txt"
        if latest_file.exists():
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception:
                pass

        versions = self.list_versions()
        if versions:
            return versions[-1]
        return None

    def list_versions(self) -> List[str]:
        """列出所有版本"""
        versions = []
        for path in self.results_path.iterdir():
            if path.is_dir() and path.name.startswith('v'):
                versions.append(path.name)

        try:
            versions.sort(key=lambda x: int(x.replace('v', '')))
        except ValueError:
            versions.sort()

        return versions

    def compare_versions(self, v1: str, v2: str) -> Dict[str, Any]:
        """比較兩個版本的差異"""
        results_v1 = self.load_version(v1)
        results_v2 = self.load_version(v2)

        if not results_v1 or not results_v2:
            return {'error': f'無法加載版本 {v1} 或 {v2}'}

        return {
            'versions': {'v1': v1, 'v2': v2},
            'summary': {'total_changes': 0}
        }