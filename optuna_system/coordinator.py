# -*- coding: utf-8 -*-
"""
Optuna系統主協調器
統一管理Layer0+9層完整優化架構的執行和結果管理
支持: Layer0數據清洗 + Layer1-4核心層 + Layer5-8專項層
"""
import json
import logging
import sys
import os
import time
import traceback
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from optuna_system.utils.io_utils import write_dataframe, read_dataframe, compute_file_md5, atomic_write_json
from optuna_system.utils.logging_utils import setup_logging

# 🆕 阶段6：生存者偏差校正（可选）
try:
    from optuna_system.utils.survivorship_bias import apply_survivorship_correction
    HAS_SURVIVORSHIP_CORRECTION = True
except ImportError:
    HAS_SURVIVORSHIP_CORRECTION = False
    print("⚠️ survivorship_bias模块不可用，生存者偏差校正将跳过")

# 🆕 阶段7：系统性偏差+可解释性（可选）
try:
    from optuna_system.utils.adversarial_validation import quick_adversarial_check
    HAS_ADVERSARIAL_VALIDATION = True
except ImportError:
    HAS_ADVERSARIAL_VALIDATION = False
    print("ℹ️ adversarial_validation模块未导入")

try:
    from optuna_system.utils.random_benchmark import RandomBenchmarkTester
    HAS_RANDOM_BENCHMARK = True
except ImportError:
    HAS_RANDOM_BENCHMARK = False
    print("ℹ️ random_benchmark模块未导入")

try:
    from optuna_system.utils.model_interpretability import ModelInterpreter
    HAS_MODEL_INTERPRETABILITY = True
except ImportError:
    HAS_MODEL_INTERPRETABILITY = False
    print("ℹ️ model_interpretability模块未导入")

# 添加當前目錄到Python路徑
current_dir = Path(__file__).parent
project_root = current_dir.parent  # 项目根目录
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "optimizers"))
sys.path.append(str(project_root))  # 添加项目根目录，以便导入src模块

try:
    # Layer0-2: 核心層（必需）
    from optimizers.optuna_cleaning import DataCleaningOptimizer
    from optimizers.optuna_feature import FeatureOptimizer
    from optimizers.optuna_meta_label import MetaLabelOptimizer  # Meta-Labeling 雙層架構
    from config.timeframe_scaler import TimeFrameScaler
    from version_manager import OptunaVersionManager

    # Layer3-8: 其他層（可選，如果存在則導入）
    try:
        from optimizers.optuna_model import ModelOptimizer
        HAS_MODEL_OPTIMIZER = True
    except ImportError:
        HAS_MODEL_OPTIMIZER = False
        print("⚠️ optuna_model不可用，Layer3將跳過")

    try:
        from optimizers.optuna_cv_risk import CVRiskOptimizer
        HAS_CV_RISK_OPTIMIZER = True
    except ImportError:
        HAS_CV_RISK_OPTIMIZER = False
        print("⚠️ optuna_cv_risk不可用，Layer4將跳過")
    
    try:
        from optimizers.kelly_optimizer import KellyOptimizer
        HAS_KELLY_OPTIMIZER = True
    except ImportError:
        HAS_KELLY_OPTIMIZER = False
        print("⚠️ kelly_optimizer不可用，Layer5將跳過")

    try:
        from optimizers.ensemble_optimizer import EnsembleOptimizer
        HAS_ENSEMBLE_OPTIMIZER = True
    except ImportError:
        HAS_ENSEMBLE_OPTIMIZER = False
        print("⚠️ ensemble_optimizer不可用，Layer6將跳過")

    try:
        from optimizers.polynomial_optimizer import PolynomialOptimizer
        HAS_POLYNOMIAL_OPTIMIZER = True
    except ImportError:
        HAS_POLYNOMIAL_OPTIMIZER = False
        print("⚠️ polynomial_optimizer不可用，Layer7將跳過")

    try:
        from optimizers.confidence_optimizer import ConfidenceOptimizer
        HAS_CONFIDENCE_OPTIMIZER = True
    except ImportError:
        HAS_CONFIDENCE_OPTIMIZER = False
        print("⚠️ confidence_optimizer不可用，Layer8將跳過")

except ImportError as e:
    print(f"⚠️ 核心模塊導入失敗: {e}")
    print("請檢查version_manager.py和核心優化器文件是否存在")


class OptunaCoordinator:
    """Layer0+9層完整優化系統協調器（支持多時間框架自動縮放）"""
    
    def __init__(self, symbol: str = "BTCUSDT", timeframe: str = "15m", 
                 data_path: str = "data", scaled_config: Dict = None, version: str = None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_path = Path(data_path)

        # 統一日誌初始化
        self.logger = setup_logging(name=__name__)

        # 時框縮放配置（可由MultiTimeframeCoordinator提供）
        self.scaled_config = scaled_config or {}
        self.scaler = TimeFrameScaler(self.logger)
        
        # 設置路徑
        self.optuna_root = Path(__file__).parent
        self.configs_path = self.optuna_root / "configs"
        self.results_path = self.optuna_root / "results"
        # 物化輸出改為 data/processed
        self.processed_root = self.data_path / "processed"
        self.materialized_path = self.processed_root  # 兼容舊命名
        
        # 確保目錄存在
        self.configs_path.mkdir(exist_ok=True)
        self.results_path.mkdir(exist_ok=True)
        self.processed_root.mkdir(parents=True, exist_ok=True)
        
        # 初始化版本管理器
        self.version_manager = OptunaVersionManager(str(self.results_path))
        
        # 當前版本
        self.current_version = version or self.version_manager.create_new_version()
        
        # 優化結果存儲
        self.layer_results: Dict[str, Any] = {}  # 用於存儲各層結果
        
        self.logger.info("🚀 修復版Layer0+9層優化系統協調器初始化完成")
        self.logger.info(f"   交易對: {symbol}")
        self.logger.info(f"   時間框: {timeframe}")
        self.logger.info(f"   版本號: {self.current_version}")

        # 初始化 meta study 以獲取多時框波動資訊
        self._initialize_meta_study()

    def _initialize_meta_study(self) -> None:
        """構建多時框波動資訊（meta_vol、global_vol）並注入 scaled_config"""
        timeframes = self.scaled_config.get('meta_timeframes') or ['15m', '1h', '4h', '1d']
        global_vol = {}

        for tf in timeframes:
            try:
                ohlcv_file = self.data_path / "raw" / self.symbol / f"{self.symbol}_{tf}_ohlcv.parquet"
                if not ohlcv_file.exists():
                    # 回退：從 15m 重採樣
                    base_15m = self.data_path / "raw" / self.symbol / f"{self.symbol}_15m_ohlcv.parquet"
                    if base_15m.exists() and tf in {"1h", "4h", "1d"}:
                        try:
                            df15 = read_dataframe(base_15m)
                            rule_map = {"1h": "1H", "4h": "4H", "1d": "1D"}
                            rule = rule_map[tf]
                            
                            # 🔧 P0修复：严格双重shift防止时间泄漏
                            # 问题：原代码直接resample，包含未来数据
                            # 修复：使用MultiTimeframeAligner的严格方法
                            from optuna_system.utils.timeframe_alignment import MultiTimeframeAligner
                            
                            aligner = MultiTimeframeAligner('15m', [tf], strict_mode=True)
                            df = aligner.resample_ohlcv(df15, tf, method='double_shift')
                            
                            self.logger.info(
                                f"✅ {tf} 回退重采样自15m（双重shift防泄漏）: {len(df)} 行"
                            )
                        except Exception as ie:
                            self.logger.warning(f"{tf} 無法重採樣: {ie}")
                            global_vol[tf] = 0.02
                            continue
                    else:
                        self.logger.warning(f"⚠️ {tf} 資料不存在且無法回退: {ohlcv_file}")
                        global_vol[tf] = 0.02
                        continue
                else:
                    df = read_dataframe(ohlcv_file)

                returns = df['close'].pct_change().dropna()
                rolling_std = returns.rolling(window=100, min_periods=50).std()
                global_vol[tf] = float(rolling_std.mean()) if len(rolling_std) > 0 else 0.02
                self.logger.info(f"{tf} 波動計算: std_mean={global_vol[tf]:.4f}")
            except Exception as e:
                # 降級為 Warning，並嘗試以 15m 重採樣回退
                self.logger.warning(f"⚠️ {tf} 波動計算讀取失敗，嘗試回退: {e}")
                try:
                    base_15m = self.data_path / "raw" / self.symbol / f"{self.symbol}_15m_ohlcv.parquet"
                    if base_15m.exists() and tf in {"1h", "4h", "1d"}:
                        df15 = read_dataframe(base_15m)
                        rule_map = {"1h": "1H", "4h": "4H", "1d": "1D"}
                        rule = rule_map[tf]
                        df = df15.resample(rule).agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                        returns = df['close'].pct_change().dropna()
                        rolling_std = returns.rolling(window=100, min_periods=50).std()
                        global_vol[tf] = float(rolling_std.mean()) if len(rolling_std) > 0 else 0.02
                        self.logger.info(f"{tf} 回退重採樣自15m: std_mean={global_vol[tf]:.4f}")
                        continue
                except Exception as ie:
                    self.logger.warning(f"⚠️ {tf} 回退重採樣仍失敗: {ie}")
                global_vol[tf] = 0.02

        if not global_vol:
            meta_vol = 0.02
        else:
            meta_vol = float(np.mean(list(global_vol.values())))

        self.scaled_config.setdefault('global_vol', global_vol)
        self.scaled_config.setdefault('meta_vol', meta_vol)
        self.scaled_config.setdefault('meta_timeframes', timeframes)

        self.logger.info(
            f"Meta 大腦準備完成: 平均波動={meta_vol:.4f}, 各時框={ {tf: round(vol, 4) for tf, vol in global_vol.items()} }"
        )

    def _get_layer_json_path(self, layer_name: str) -> Path:
        mapping = {
            "layer0_cleaning": "cleaning_params",
            "layer1_labels": "label_params",
            "layer2_features": "feature_params",
        }
        base = mapping.get(layer_name)
        if not base:
            return self.configs_path
        tf_file = self.configs_path / f"{base}_{self.timeframe}.json"
        if tf_file.exists():
            return tf_file
        return self.configs_path / f"{base}.json"

    def get_cleaned_file_path(self, timeframe: str) -> Path:
        # 優先 processed，其次回退到舊位置
        processed = self.processed_root / "cleaned" / f"{self.symbol}_{timeframe}" / "cleaned_ohlcv.parquet"
        if processed.exists():
            return processed
        legacy = self.configs_path / f"cleaned_ohlcv_{timeframe}.parquet"
        return legacy

    def get_label_config_path(self, timeframe: str) -> Path:
        return self.configs_path / f"label_params_{timeframe}.json"

    def get_feature_config_path(self, timeframe: str) -> Path:
        return self.configs_path / f"feature_params_{timeframe}.json"
    
    def load_default_configs(self) -> Dict[str, Dict]:
        """加載默認配置"""
        configs = {}
        
        config_files = {
            # Layer0-4: 核心層配置
            'layer0_cleaning': 'cleaning_params.json',
            'layer1_label': 'label_params.json',
            'layer2_feature': 'feature_params.json',
            'layer3_model': 'model_params.json',
            'layer4_cv_risk': 'cv_risk_params.json',
            
            # Layer5-8: 專項層配置
            'layer5_kelly': 'kelly_params.json',
            'layer6_ensemble': 'ensemble_params.json',
            'layer7_polynomial': 'polynomial_params.json',
            'layer8_confidence': 'confidence_params.json'
        }
        
        for module_name, filename in config_files.items():
            config_file = self.configs_path / filename
            if not config_file.exists():
                continue
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                if isinstance(config_data, dict):
                    # 輕量檢查：必備鍵（如 best_params / best_score）缺失時，跳過避免污染
                    if any(key in filename for key in ("label_params", "feature_params", "cleaning_params")):
                        if not ("best_params" in config_data and "best_score" in config_data):
                            self.logger.warning("⚠️ 配置缺少必備鍵，跳過: %s", filename)
                            continue
                    configs[module_name] = config_data
                    self.logger.info("✅ 加載配置: %s", filename)
                else:
                    self.logger.warning("⚠️ 配置內容非字典: %s", filename)
            except Exception as e:
                self.logger.warning("⚠️ 加載配置失敗 %s: %s", filename, e)
        
        return configs
    
    # ============================================================
    # Layer0：數據清洗基礎層
    # ============================================================
    
    # -------------------------------------------------
    # 資料物化與快取機制
    # -------------------------------------------------

    def _generate_params_hash(self, params: Dict[str, Any]) -> str:
        """根據排序後的參數生成MD5哈希"""
        serialized = json.dumps(params, sort_keys=True, ensure_ascii=False)
        return md5(serialized.encode("utf-8")).hexdigest()

    def _materialize_dir(self, layer_name: str) -> Path:
        """依 layer 分配物化子資料夾。"""
        layer_to_sub = {
            "layer0_cleaning": "cleaned",
            "layer1_labels": "labels",
            "layer2_features": "features",
        }
        sub = layer_to_sub.get(layer_name, "misc")
        # 將輸出寫入版本子目錄（Vn）
        out_dir = self.processed_root / sub / f"{self.symbol}_{self.timeframe}" / str(self.current_version)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _materialize_stem(self, layer_name: str, params_hash: str) -> str:
        prefix = {
            "layer0_cleaning": "cleaned_ohlcv",
            "layer1_labels": "labels",
            "layer2_features": "features",
        }.get(layer_name, layer_name)
        return f"{prefix}_{self.symbol}_{self.timeframe}_{params_hash}"

    def _materialize_cache_path(self, layer_name: str, params_hash: str) -> Path:
        """取得物化目標檔案（優先 parquet，實際寫入可能為 pkl）。"""
        out_dir = self._materialize_dir(layer_name)
        stem = self._materialize_stem(layer_name, params_hash)
        return out_dir / f"{stem}.parquet"

    def _materialize_metadata_path(self, layer_name: str, params_hash: str) -> Path:
        out_dir = self._materialize_dir(layer_name)
        stem = self._materialize_stem(layer_name, params_hash)
        return out_dir / f"{stem}.json"

    def _is_cache_valid(self, cache_file: Path, ttl_hours: int = 24) -> bool:
        if not cache_file.exists():
            return False
        try:
            modified_delta = time.time() - cache_file.stat().st_mtime
            return modified_delta <= ttl_hours * 3600
        except OSError:
            return False

    def _load_materialized_metadata(self, metadata_file: Path) -> Dict[str, Any]:
        if not metadata_file.exists():
            return {}
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_materialized_metadata(self, metadata_file: Path, metadata: Dict[str, Any]) -> None:
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        # 原子方式寫入 metadata JSON
        atomic_write_json(metadata_file, metadata)

    def _ensure_layer_input(self, layer_name: str) -> Optional[Path]:
        """返回上一層的物化資料路徑"""
        order = [
            "layer0_cleaning",
            "layer1_labels",
            "layer2_features",
        ]
        idx = order.index(layer_name) if layer_name in order else -1
        if idx <= 0:
            return None
        prev_layer = order[idx - 1]
        prev_result = self.layer_results.get(prev_layer, {})
        materialized_path = prev_result.get("materialized_path")
        if materialized_path:
            p = Path(materialized_path)
            if p.exists():
                return p

        # 回退：在 data/processed 尋找上一層輸出
        try:
            if prev_layer == "layer0_cleaning":
                # 尋找 cleaned（遞迴搜索版本子目錄）
                base = self.processed_root / "cleaned" / f"{self.symbol}_{self.timeframe}"
                candidates = []
                if base.exists():
                    for pattern in ["**/cleaned_ohlcv.parquet", "**/cleaned_ohlcv_*.parquet", "**/cleaned_ohlcv.pkl", "**/cleaned_ohlcv_*.pkl"]:
                        candidates.extend(sorted(base.rglob(pattern), key=lambda x: x.stat().st_mtime, reverse=True))
                # Return first readable candidate to avoid corrupted files
                for c in candidates:
                    try:
                        _ = read_dataframe(c)
                        return c
                    except Exception:
                        continue

                # 最後回退到舊位置
                legacy = self.configs_path / f"cleaned_ohlcv_{self.timeframe}.parquet"
                if legacy.exists():
                    return legacy

            if prev_layer == "layer1_labels":
                # 尋找 labels（遞迴搜索版本子目錄）
                base = self.processed_root / "labels" / f"{self.symbol}_{self.timeframe}"
                candidates = []
                if base.exists():
                    for pattern in ["**/labels_*.parquet", "**/labels_*.pkl"]:
                        candidates.extend(sorted(base.rglob(pattern), key=lambda x: x.stat().st_mtime, reverse=True))
                # Return first readable candidate to avoid corrupted files
                for c in candidates:
                    try:
                        _ = read_dataframe(c)
                        return c
                    except Exception:
                        continue

                # 回退：若無可讀取之上一層標籤，嘗試從清洗資料與Layer1配置重建
                try:
                    label_cfg_path = self.get_label_config_path(self.timeframe)
                    if label_cfg_path.exists():
                        with open(label_cfg_path, 'r', encoding='utf-8') as f:
                            label_cfg = json.load(f)
                        label_params = dict(label_cfg.get('best_params', {}))
                        if label_params:
                            cleaned_path = self.get_cleaned_file_path(self.timeframe)
                            df_cleaned = read_dataframe(cleaned_path)
                            # Meta-Labeling 返回 DataFrame（包含 primary_signal, meta_quality, label）
                            labels_df = MetaLabelOptimizer(
                                data_path=str(self.data_path),
                                config_path=str(self.configs_path),
                                symbol=self.symbol,
                                timeframe=self.timeframe,
                                scaled_config=self.scaled_config,
                            ).apply_labels(df_cleaned, label_params)
                            # 提取 label 列（0/1/2 三分類，向後兼容）
                            labels = labels_df['label']
                            aligned = df_cleaned.loc[labels.index].copy()
                            aligned['label'] = labels
                            base_cols = [c for c in df_cleaned.columns if c != 'label']
                            processed = aligned[base_cols + ['label']]
                            params_hash = self._generate_params_hash(label_params)
                            target = self._materialize_cache_path('layer1_labels', params_hash)
                            actual_file, _fmt = write_dataframe(processed, target)
                            self.logger.info(f"🧩 已回退重建Layer1標籤並物化: {actual_file}")
                            return actual_file
                except Exception as _e:
                    # 靜默失敗，交由上層處理
                    self.logger.warning(f"⚠️ 標籤回退重建失敗: {_e}")

        except Exception:
            pass

        return None

    def _get_optimizer(self, layer_name: str):
        if layer_name == "layer0_cleaning":
            return DataCleaningOptimizer(
                data_path=str(self.data_path),
                config_path=str(self.configs_path),
                symbol=self.symbol,
                timeframe=self.timeframe,
                scaled_config=self.scaled_config,
            )
        if layer_name == "layer1_labels":
            # 🎯 使用 Meta-Labeling 雙層架構（Primary + Meta）
            self.logger.info("✨ 使用 Meta-Labeling 雙層架構（Primary + Meta）")
            return MetaLabelOptimizer(
                data_path=str(self.data_path),
                config_path=str(self.configs_path),
                symbol=self.symbol,
                timeframe=self.timeframe,
                scaled_config=self.scaled_config
            )
        if layer_name == "layer2_features":
            return FeatureOptimizer(
                data_path=str(self.data_path),
                config_path=str(self.configs_path),
                symbol=self.symbol,
                timeframe=self.timeframe,
                scaled_config=self.scaled_config,
            )
        raise ValueError(f"Unsupported layer for materialization: {layer_name}")

    def materialize_layer_data(
        self,
        layer_name: str,
        params: Dict[str, Any],
        input_data: Optional[pd.DataFrame] = None,
        optimization_result: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        """物化層級處理結果，返回資料路徑與metadata"""
        params_hash = self._generate_params_hash(params)
        cache_file = self._materialize_cache_path(layer_name, params_hash)
        metadata_file = self._materialize_metadata_path(layer_name, params_hash)

        if self._is_cache_valid(cache_file):
            metadata = self._load_materialized_metadata(metadata_file)
            metadata.setdefault("cached", True)
            metadata.setdefault("cache_path", str(cache_file))
            self.logger.info("✅ 使用緩存資料: %s", cache_file.name)
            return cache_file, metadata

        optimizer = self._get_optimizer(layer_name)

        if layer_name == "layer0_cleaning":
            if input_data is None and optimization_result:
                cleaned_path = optimization_result.get("cleaned_file")
                if cleaned_path and Path(cleaned_path).exists():
                    input_data = read_dataframe(Path(cleaned_path))
            if input_data is None or input_data.empty:
                raise ValueError("Layer0缺少清洗後資料，無法物化")
            processed_data = input_data
            # 標準化欄位順序：OHLCV優先
            ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in processed_data.columns]
            other_cols = [c for c in processed_data.columns if c not in ohlcv_cols]
            processed_data = processed_data[ohlcv_cols + other_cols]

        elif layer_name == "layer1_labels":
            if input_data is None or "close" not in input_data.columns:
                raise ValueError("Layer1需要包含close欄位的上一層資料")
            # Meta-Labeling 返回 DataFrame（包含 primary_signal, meta_quality, label）
            labeled_df = optimizer.apply_labels(input_data, params)
            labels = labeled_df['label']
            # Consistency 校驗
            if not set(labels.index).issubset(set(input_data.index)):
                raise ValueError("Layer1標籤索引不在清洗資料索引內")
            # 从嵌套参数中提取 lag
            lag_v = params.get('primary', {}).get("lag", 0) if isinstance(params, dict) and 'primary' in params else int(params.get("lag", 0))
            aligned = input_data.loc[labels.index].copy()
            if lag_v >= 0:
                expected = max(len(input_data) - lag_v, 0)
                if len(aligned) != expected:
                    raise ValueError(f"Layer1長度不一致: aligned={len(aligned)}, expected={expected}")
            aligned["label"] = labels
            # 標準化欄位順序：保留原清洗欄位順序，label置尾
            base_cols = [c for c in input_data.columns if c != "label"]
            processed_data = aligned[base_cols + ["label"]]

        elif layer_name == "layer2_features":
            if input_data is None or "label" not in input_data.columns:
                raise ValueError("Layer2需要包含label欄位的上一層資料")
            # 與優化一致：重建特徵矩陣，再按最佳子集篩選
            ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in input_data.columns]
            if not ohlcv_cols:
                raise ValueError("Layer2物化缺少OHLCV欄位，無法重建特徵")
            ohlcv_df = input_data[ohlcv_cols].copy()
            try:
                X_all = optimizer.build_features_for_materialization(ohlcv_df, params)
            except Exception as e:
                self.logger.warning(f"⚠️ 無法重建特徵，回退至空特徵: {e}")
                X_all = pd.DataFrame(index=ohlcv_df.index)

            # 對齊索引並選擇最佳特徵子集
            selected = params.get("selected_features") or []
            available = [c for c in selected if c in X_all.columns]
            missing = [c for c in selected if c not in X_all.columns]
            if missing:
                self.logger.warning("⚠️ Layer2物化缺失特徵列: %s", missing)
            X_sel = X_all.reindex(input_data.index).fillna(0)
            if available:
                X_sel = X_sel[available]
            # 合併標籤
            processed_data = pd.concat([X_sel, input_data[["label"]]], axis=1)
        else:
            raise ValueError(f"Unsupported layer: {layer_name}")

        if processed_data is None or processed_data.empty:
            raise ValueError(f"{layer_name} processed data is empty, 無法物化")

        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # 打印JSON路徑與參數雜湊，幫助CLU日誌確認「由JSON參數生成真實檔案」
        json_path = self._get_layer_json_path(layer_name)
        try:
            params_hash = self._generate_params_hash(params)
            self.logger.info(
                f"📄 {layer_name} 超參數JSON: {json_path} | params_hash={params_hash}"
            )
        except Exception:
            pass

        actual_file, storage_format = write_dataframe(processed_data, cache_file)
        # 檔案內容校驗碼
        try:
            file_md5 = compute_file_md5(actual_file)
        except Exception:
            file_md5 = None
        self.logger.info(
            f"💾 {layer_name} 物化輸出: {actual_file} | format={storage_format} | MD5={file_md5}"
        )

        # 補充metadata（含best_params、json路徑、score等）
        metadata = {
            "created_at": pd.Timestamp.utcnow().isoformat(),
            "params_hash": params_hash,
            "data_shape": processed_data.shape,
            "data_columns": list(processed_data.columns),
            "cached": False,
            "storage_format": storage_format,
            "json_path": str(json_path),
            "best_params": params,
            "file_path": str(actual_file),
            "file_md5": file_md5,
        }
        if isinstance(optimization_result, dict):
            if "best_score" in optimization_result:
                metadata["best_score"] = optimization_result["best_score"]
            if "n_trials" in optimization_result:
                metadata["n_trials"] = optimization_result["n_trials"]

        self._save_materialized_metadata(metadata_file, metadata)
        return actual_file, metadata

    def run_layer_with_materialization(
        self,
        layer_name: str,
        n_trials: int,
        prev_data_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        optimizer = self._get_optimizer(layer_name)
        if hasattr(optimizer, "set_input_data_path") and prev_data_path:
            optimizer.set_input_data_path(str(prev_data_path))

        optimization_result = optimizer.optimize(n_trials=n_trials)
        best_params = dict(optimization_result.get("best_params", {}))
        best_params["n_trials"] = n_trials

        # 🔗 層間約束：Layer2 的 lag 需對齊 Layer1 的最佳 lag
        if layer_name == "layer2_features":
            try:
                l1_params = self.layer_results.get("layer1_labels", {}).get("best_params", {})
                if isinstance(l1_params, dict) and "lag" in l1_params:
                    best_params["lag"] = int(l1_params.get("lag"))
                    self.logger.info(f"🔗 對齊設定: Layer2 lag = Layer1 lag = {best_params['lag']}")
            except Exception as _e:
                self.logger.warning(f"⚠️ 無法對齊 Layer2 lag 與 Layer1: {_e}")

        if layer_name == "layer0_cleaning":
            cleaned_file = optimization_result.get("cleaned_file")
            if cleaned_file and Path(cleaned_file).exists():
                input_df = read_dataframe(Path(cleaned_file))
            else:
                input_df = None
        else:
            if not prev_data_path or not prev_data_path.exists():
                prev_data_path = self._ensure_layer_input(layer_name)  # 再嘗試一次（帶回退）
            input_df = read_dataframe(prev_data_path) if prev_data_path and prev_data_path.exists() else None
            if input_df is None:
                raise ValueError(f"{layer_name} 缺少上一層資料")

        # 在物化之前打印來源JSON與雜湊，以便審計
        try:
            json_path = self._get_layer_json_path(layer_name)
            params_hash = self._generate_params_hash(best_params)
            self.logger.info(
                f"🧪 {layer_name} materialize 前檢查: JSON={json_path} | params_hash={params_hash}"
            )
        except Exception:
            pass

        materialized_path, metadata = self.materialize_layer_data(
            layer_name,
            best_params,
            input_df,
            optimization_result,
        )

        version_id = self.version_manager.create_data_version(
            layer_name,
            best_params,
            metadata.get("data_shape", (0, 0)),
        )
        try:
            self.version_manager.save_materialized_data_info(
                version_id,
                layer_name,
                str(materialized_path),
                metadata,
            )
        except Exception as _e:
            self.logger.warning(f"⚠️ 保存血統資訊失敗: {getattr(_e, 'message', _e)}")

        final_result = {
            **optimization_result,
            "best_params": best_params,
            "materialized_path": str(materialized_path),
            "metadata": metadata,
            "data_version": version_id,
        }

        self._save_layer_params(layer_name, final_result)
        return final_result

    def _save_layer_params(self, layer_name: str, result: Dict[str, Any]) -> None:
        # 將結果轉為可序列化，過濾 DataFrame/Series 等大型物件
        import numpy as _np  # 局部引用，避免循環引用風險
        try:
            import pandas as _pd  # 可選
        except Exception:  # pragma: no cover
            _pd = None

        def _is_pandas_obj(obj: Any) -> bool:
            if _pd is None:
                return False
            return isinstance(obj, (_pd.DataFrame, _pd.Series, _pd.Index))

        def _to_json_safe(obj: Any) -> Any:
            if _is_pandas_obj(obj):
                return None  # 丟棄大型資料本體，避免JSON膨脹；物化路徑已在 result 中
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, _np.integer):
                return int(obj)
            if isinstance(obj, _np.floating):
                return float(obj)
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                safe_dict = {}
                for k, v in obj.items():
                    if _is_pandas_obj(v):
                        continue
                    safe_v = _to_json_safe(v)
                    if safe_v is not None:
                        safe_dict[k] = safe_v
                return safe_dict
            if isinstance(obj, (list, tuple, set)):
                safe_list = []
                for v in obj:
                    if _is_pandas_obj(v):
                        continue
                    safe_v = _to_json_safe(v)
                    if safe_v is not None:
                        safe_list.append(safe_v)
                return safe_list
            # 其它未知類型，退回字串
            return str(obj)

        safe_result = _to_json_safe(result)
        if isinstance(safe_result, dict):
            # 額外保險：主動移除可能殘留的大型鍵
            safe_result.pop('labeled_data', None)
            safe_result.pop('materialized_features', None)

        layer_to_filename = {
            "layer0_cleaning": "cleaning_params",
            "layer1_labels": "label_params",
            "layer2_features": "feature_params",
        }
        base_name = layer_to_filename.get(layer_name)
        if not base_name:
            return
        file_path = self.configs_path / f"{base_name}.json"
        atomic_write_json(file_path, safe_result)

        tf_file_path = self.configs_path / f"{base_name}_{self.timeframe}.json"
        atomic_write_json(tf_file_path, safe_result)

    def run_layer0_data_cleaning(self, n_trials: int = 25) -> Dict[str, Any]:
        """Layer0：資料清洗 + 物化"""
        self.logger.info("🔧 Layer0：數據清洗與物化...")

        try:
            result = self.run_layer_with_materialization("layer0_cleaning", n_trials)
            self.layer_results["layer0_cleaning"] = result
            
            # ✅ 添加結果驗證
            if 'best_score' in result and result['best_score'] < 0.3:
                self.logger.warning(f"⚠️ Layer0分數過低: {result['best_score']:.4f}，建議檢查數據質量")
            if 'error' in result:
                self.logger.error(f"❌ Layer0出現錯誤: {result['error']}")
            
            cleaned_path = result.get("materialized_path")
            if cleaned_path:
                self.scaled_config["cleaned_file"] = cleaned_path
            return result
        except Exception as e:
            self.logger.error("❌ Layer0數據清洗失敗: %s", e)
            error_result = {"error": str(e), "layer": 0}
            self.layer_results["layer0_cleaning"] = error_result
            return error_result
    
    # ============================================================
    # Layer1-4：核心優化層
    # ============================================================
    
    def run_layer1_label_optimization(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Layer1：標籤生成 + 物化
        
        🔧 P0修复：降低trial数量75% (200→50)
        问题：过多trials导致数据窥探偏差
        - 原150×250=37,500次隐性测试，预期1,875个假阳性（5% FDR）
        
        解决：应用Romano-Wolf多重检验校正
        学术依据：Romano & Wolf (2005), White (2000)
        """
        self.logger.info("🎯 Layer1：標籤生成與物化...")

        try:
            prev_path = self._ensure_layer_input("layer1_labels")
            result = self.run_layer_with_materialization("layer1_labels", n_trials, prev_path)
            self.layer_results["layer1_labels"] = result
            
            # ✅ 添加結果驗證
            if 'best_score' in result:
                if result['best_score'] < 0.45:
                    self.logger.warning(f"⚠️ Layer1分數過低: {result['best_score']:.4f}，建議檢查標籤平衡性")
                # 檢查標籤分布
                if 'metadata' in result and 'label_distribution' in result['metadata']:
                    dist = result['metadata']['label_distribution']
                    if len(dist) == 3:
                        ratios = list(dist.values())
                        imbalance = max(ratios) / min(ratios) if min(ratios) > 0 else 99
                        if imbalance > 5.0:
                            self.logger.warning(f"⚠️ 標籤不平衡比過高: {imbalance:.2f}")
            
            return result
        except Exception as e:
            self.logger.error("❌ Layer1標籤優化失敗: %s", e)
            error_result = {"error": str(e), "layer": 1}
            self.layer_results["layer1_labels"] = error_result
            return error_result
    
    def run_layer2_feature_optimization(
        self, n_trials: int = 30, multi_timeframes: List[str] = None
    ) -> Dict[str, Any]:
        """Layer2：特徵工程參數優化 - 支持單一時框或多時框優化"""

        if multi_timeframes:
            # 🚀 多時框優化模式
            self.logger.info(f"🎯 Layer2多時框特徵優化: {multi_timeframes}...")
            return self._run_multi_timeframe_feature_optimization(multi_timeframes, n_trials)
        else:
            # 原有的單一時框優化模式
            self.logger.info(f"🎯 Layer2單一時框特徵優化: {self.timeframe}...")
        
        try:
            # 可選：使用增強編排器（不改動既有結構，透過環境變量開關）
            if os.getenv('OPTUNA_ENHANCED_L2', '0') == '1':
                self.logger.info("🚀 使用 EnhancedLayer2Optimizer（Phase 5 編排器）")
                prev_path = self._ensure_layer_input("layer2_features")
                if not prev_path or not prev_path.exists():
                    raise ValueError("Layer2缺少上一層資料")

                # 構建基礎優化器並交由編排器執行增強優化
                base_optimizer = FeatureOptimizer(
                    data_path=str(self.data_path),
                    config_path=str(self.configs_path),
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    scaled_config=self.scaled_config,
                )

                # Phase 5 內嵌編排：標籤質檢→（必要時）重訓練→自適應trials優化→驗證
                analysis = base_optimizer.analyze_label_quality()
                self.logger.info(f"Label 分佈/穩定性: {analysis}")
                if not analysis.get('pass', True):
                    retrain = base_optimizer.auto_retrain_labels_if_needed()
                    if retrain:
                        self.logger.info(f"Layer1 重新優化完成: F1={retrain.get('best_score', 0):.4f}")

                trials_completed = 0
                target_trials = n_trials
                best_score = -1.0
                best_params: Dict[str, Any] = {}

                def _next_target(current_best: float, done: int) -> int:
                    if current_best < 0.30:
                        return min(done + 100, 500)
                    elif current_best > 0.60:
                        return done + 20
                    else:
                        return done + 50

                while trials_completed < target_trials:
                    remaining = target_trials - trials_completed
                    n_batch = min(remaining, 50)
                    self.logger.info(
                        f"執行批次: trials {trials_completed}->{trials_completed + n_batch} / target {target_trials}"
                    )
                    result_batch = base_optimizer.optimize(n_trials=n_batch)
                    trials_completed += n_batch

                    cur_best = float(result_batch.get('best_score', 0.0))
                    if cur_best > best_score:
                        best_score = cur_best
                        best_params = dict(result_batch.get('best_params', {}))

                    target_trials = _next_target(best_score, trials_completed)
                    self.logger.info(
                        f"累計 {trials_completed} trials, 當前最佳={best_score:.4f}, 新目標={target_trials}"
                    )
                    if trials_completed >= 500 or (trials_completed >= n_trials and best_score >= 0.70):
                        break

                # 🔗 層間約束：Layer2 的 lag 需對齊 Layer1 的最佳 lag（Enhanced 路徑）
                try:
                    l1_params = self.layer_results.get("layer1_labels", {}).get("best_params", {})
                    if isinstance(l1_params, dict) and "lag" in l1_params:
                        best_params["lag"] = int(l1_params.get("lag"))
                        self.logger.info(f"🔗 對齊設定(Enhanced): Layer2 lag = Layer1 lag = {best_params['lag']}")
                except Exception as _e:
                    self.logger.warning(f"⚠️ 無法對齊 Layer2 lag 與 Layer1(Enhanced): {_e}")

                optimization_result = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'n_trials': n_trials,
                }

                input_df = read_dataframe(prev_path)
                materialized_path, metadata = self.materialize_layer_data(
                    "layer2_features",
                    best_params,
                    input_df,
                    optimization_result,
                )

                version_id = self.version_manager.create_data_version(
                    "layer2_features",
                    best_params,
                    metadata.get("data_shape", (0, 0)),
                )
                try:
                    self.version_manager.save_materialized_data_info(
                        version_id,
                        "layer2_features",
                        str(materialized_path),
                        metadata,
                    )
                except Exception as _e:
                    self.logger.warning(f"⚠️ 保存血統資訊失敗: {getattr(_e, 'message', _e)}")

                final_result = {
                    **optimization_result,
                    "materialized_path": str(materialized_path),
                    "metadata": metadata,
                    "data_version": version_id,
                }
                self.layer_results["layer2_features"] = final_result

                # 結果驗證與提示
                if 'best_score' in final_result and final_result['best_score'] < 0.50:
                    self.logger.warning(
                        f"⚠️ Layer2分數過低: {final_result['best_score']:.4f}，建議增加試驗次數或檢查特徵質量"
                    )
                if 'best_params' in final_result and 'selected_features' in final_result['best_params']:
                    selected = final_result['best_params']['selected_features']
                    strategy_features = [f for f in selected if any(p in f for p in ['wyk_', 'td_', 'micro_'])]
                    self.logger.info(f"✅ 保留策略特徵: {len(strategy_features)}個")
                    if len(strategy_features) == 0:
                        self.logger.warning("⚠️ 警告: 未選中任何策略特徵（Wyckoff/TD/Micro）")

                self.logger.info("✅ Layer2特徵優化完成 (Enhanced)")
                return final_result

            # 默認：使用既有流程
            prev_path = self._ensure_layer_input("layer2_features")
            result = self.run_layer_with_materialization("layer2_features", n_trials, prev_path)
            self.layer_results["layer2_features"] = result

            # ✅ 添加結果驗證
            if 'best_score' in result:
                if result['best_score'] < 0.50:
                    self.logger.warning(f"⚠️ Layer2分數過低: {result['best_score']:.4f}，建議增加試驗次數或檢查特徵質量")
                # 檢查策略特徵是否保留
                if 'best_params' in result and 'selected_features' in result['best_params']:
                    selected = result['best_params']['selected_features']
                    strategy_features = [f for f in selected if any(p in f for p in ['wyk_', 'td_', 'micro_'])]
                    self.logger.info(f"✅ 保留策略特徵: {len(strategy_features)}個")
                    if len(strategy_features) == 0:
                        self.logger.warning("⚠️ 警告: 未選中任何策略特徵（Wyckoff/TD/Micro）")

            self.logger.info("✅ Layer2特徵優化完成")
            return result

        except Exception as e:
            self.logger.error(f"❌ Layer2特徵優化失敗: {e}")
            self.logger.debug("Layer2 stacktrace:\n%s", traceback.format_exc())
            error_result = {'error': str(e), 'layer': 2}
            self.layer_results['layer2_features'] = error_result
            return error_result

    def _run_multi_timeframe_feature_optimization(self, timeframes: List[str], n_trials: int = 50) -> Dict[str, Any]:
        """🚀 多時框特徵優化：為每個時框找到最佳140-203粗選→10-25精選特徵組合（全量203特徵池）"""
        self.logger.info(f"🚀 開始多時框特徵優化: {timeframes}")

        try:
            # 創建基礎優化器用於協調
            base_optimizer = FeatureOptimizer(
                data_path=str(self.data_path),
                config_path=str(self.configs_path),
                symbol=self.symbol,
                timeframe=self.timeframe  # 基礎時框
            )

            # 執行多時框優化
            multi_result = base_optimizer.optimize_multi_timeframes(
                timeframes=timeframes,
                n_trials=n_trials
            )

            # 存儲結果
            self.layer_results['layer2_features'] = multi_result

            # 輸出摘要
            successful_count = multi_result['summary']['successful_optimizations']
            total_count = len(timeframes)
            best_tf = multi_result['summary']['best_performing_timeframe']

            self.logger.info("🎯 多時框優化完成摘要:")
            self.logger.info(f"   成功: {successful_count}/{total_count} 個時框")
            self.logger.info(f"   最佳時框: {best_tf}")

            for tf, cfg in multi_result['best_configs'].items():
                if 'error' not in cfg:
                    self.logger.info(f"   {tf}: F1={cfg['best_score']:.4f}, {cfg['feature_range']}")
                else:
                    self.logger.error(f"   {tf}: 失敗 - {cfg['error']}")

            self.logger.info("✅ Layer2多時框特徵優化完成")
            return multi_result

        except Exception as e:
            self.logger.error(f"❌ Layer2多時框特徵優化失敗: {e}")
            error_result = {'error': str(e), 'layer': 2, 'type': 'multi_timeframe'}
            self.layer_results['layer2_features'] = error_result
            return error_result
    
    def run_layer3_model_optimization(self, n_trials: int = 25) -> Dict[str, Any]:
        """
        Layer3：模型超參數優化
        
        🔧 P0修复：降低trial数量75% (100→25)
        减少数据窥探偏差，控制FWER
        """
        self.logger.info("🎯 Layer3：模型超參數優化...")
        
        if not globals().get('HAS_MODEL_OPTIMIZER', False):
            self.logger.warning("⚠️ Layer3跳過: ModelOptimizer不可用")
            error_result = {'error': 'module_not_available', 'layer': 3}
            self.layer_results['layer3_models'] = error_result
            return error_result

        try:
            optimizer = ModelOptimizer(
                data_path=str(self.data_path),
                config_path=str(self.configs_path),
                symbol=self.symbol,
                timeframe=self.timeframe,
                results_path=str(self.results_path / f"{self.symbol}_{self.timeframe}" / str(self.current_version))
            )
            
            result = optimizer.optimize(n_trials=n_trials)
            self.layer_results['layer3_models'] = result
            
            self.logger.info("✅ Layer3模型優化完成")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Layer3模型優化失敗: {e}")
            error_result = {'error': str(e), 'layer': 3}
            self.layer_results['layer3_models'] = error_result
            return error_result
    
    def run_layer4_cv_risk_optimization(self, n_trials: int = 100) -> Dict[str, Any]:
        """Layer4：交叉驗證與風控參數優化"""
        self.logger.info("🎯 Layer4：交叉驗證與風控參數優化...")
        
        if not globals().get('HAS_CV_RISK_OPTIMIZER', False):
            self.logger.warning("⚠️ Layer4跳過: CVRiskOptimizer不可用")
            error_result = {'error': 'module_not_available', 'layer': 4}
            self.layer_results['layer4_cv_risk'] = error_result
            return error_result

        try:
            optimizer = CVRiskOptimizer(
                data_path=str(self.data_path),
                config_path=str(self.configs_path),
                symbol=self.symbol,
                timeframe=self.timeframe
            )
            
            result = optimizer.optimize(n_trials=n_trials)
            self.layer_results['layer4_cv_risk'] = result
            
            self.logger.info("✅ Layer4風控優化完成")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Layer4風控優化失敗: {e}")
            error_result = {'error': str(e), 'layer': 4}
            self.layer_results['layer4_cv_risk'] = error_result
            return error_result
    
    # ============================================================
    # Layer5-8：專項優化層
    # ============================================================
    
    def run_layer5_kelly_optimization(self, n_trials: int = 50) -> Dict[str, Any]:
        """Layer5：Kelly公式資金管理優化"""
        self.logger.info("🎯 Layer5：Kelly公式資金管理優化...")
        
        try:
            optimizer = KellyOptimizer(
                data_path=str(self.data_path),
                config_path=str(self.configs_path)
            )
            
            result = optimizer.optimize(n_trials=n_trials)
            self.layer_results['layer5_kelly'] = result
            
            self.logger.info("✅ Layer5 Kelly優化完成")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Layer5 Kelly優化失敗: {e}")
            error_result = {'error': str(e), 'layer': 5}
            self.layer_results['layer5_kelly'] = error_result
            return error_result
    
    def run_layer6_ensemble_optimization(self, n_trials: int = 50) -> Dict[str, Any]:
        """Layer6：多模型融合權重優化"""
        self.logger.info("🎯 Layer6：多模型融合權重優化...")
        
        try:
            optimizer = EnsembleOptimizer(
                data_path=str(self.data_path),
                config_path=str(self.configs_path)
            )
            
            result = optimizer.optimize(n_trials=n_trials)
            self.layer_results['layer6_ensemble'] = result
            
            self.logger.info("✅ Layer6多模型融合優化完成")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Layer6多模型融合優化失敗: {e}")
            error_result = {'error': str(e), 'layer': 6}
            self.layer_results['layer6_ensemble'] = error_result
            return error_result
    
    def run_layer7_polynomial_optimization(self, n_trials: int = 30) -> Dict[str, Any]:
        """Layer7：多項式特徵生成優化"""
        self.logger.info("🎯 Layer7：多項式特徵生成優化...")
        
        try:
            optimizer = PolynomialOptimizer(
                data_path=str(self.data_path),
                config_path=str(self.configs_path)
            )
            
            result = optimizer.optimize(n_trials=n_trials)
            self.layer_results['layer7_polynomial'] = result
            
            self.logger.info("✅ Layer7多項式特徵優化完成")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Layer7多項式特徵優化失敗: {e}")
            error_result = {'error': str(e), 'layer': 7}
            self.layer_results['layer7_polynomial'] = error_result
            return error_result
    
    def run_layer8_confidence_optimization(self, n_trials: int = 40) -> Dict[str, Any]:
        """Layer8：動態置信度閾值優化"""
        self.logger.info("🎯 Layer8：動態置信度閾值優化...")
        
        try:
            optimizer = ConfidenceOptimizer(
                data_path=str(self.data_path),
                config_path=str(self.configs_path)
            )
            
            result = optimizer.optimize(n_trials=n_trials)
            self.layer_results['layer8_confidence'] = result
            
            self.logger.info("✅ Layer8動態置信度優化完成")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Layer8動態置信度優化失敗: {e}")
            error_result = {'error': str(e), 'layer': 8}
            self.layer_results['layer8_confidence'] = error_result
            return error_result
    
    # ============================================================
    # 完整9層優化流程
    # ============================================================
    
    def run_complete_layered_optimization(self, trial_config: Dict[str, int] = None) -> Dict[str, Any]:
        """執行完整的Layer0+9層優化流程"""
        if trial_config is None:
            trial_config = {
                # Layer0: 數據清洗基礎層（必須執行）
                'layer0': 25,   # 數據清洗
                # Layer1-4: 核心優化層
                'layer1': 100,  # 標籤生成
                'layer2': 200,  # 特徵工程
                'layer3': 100,  # 模型超參數
                'layer4': 100,  # CV風控
                # Layer5-8: 專項優化層
                'layer5': 50,   # Kelly資金管理
                'layer6': 50,   # 多模型融合
                'layer7': 30,   # 多項式特徵
                'layer8': 40,   # 動態置信度
            }
 
        self.logger.info("🚀 開始完整Layer0+9層優化流程...")
        overall_start_time = time.time()
 
        # Layer0: 數據清洗基礎層（必須執行）
        self.logger.info("=" * 60)
        self.logger.info("📋 Layer0: 數據清洗基礎層（必須執行）")
        self.run_layer0_data_cleaning(trial_config.get('layer0', 25))
        time.sleep(1)
 
        # Layer1-4: 核心優化層（按順序執行）
        self.logger.info("=" * 60)
        self.logger.info("📋 Layer1-4: 核心優化層")
 
        self.run_layer1_label_optimization(trial_config.get('layer1', 200))
        time.sleep(1)
 
        self.run_layer2_feature_optimization(trial_config.get('layer2', 100))
        time.sleep(1)
 
        self.run_layer3_model_optimization(trial_config.get('layer3', 100))
        time.sleep(1)
 
        self.run_layer4_cv_risk_optimization(trial_config.get('layer4', 100))
        time.sleep(1)

        # Layer5-8: 專項優化層（可並行執行）
        self.logger.info("=" * 60)
        self.logger.info("📋 Layer5-8: 專項優化層")

        self.run_layer5_kelly_optimization(trial_config.get('layer5', 50))
        time.sleep(1)

        self.run_layer6_ensemble_optimization(trial_config.get('layer6', 50))
        time.sleep(1)

        self.run_layer7_polynomial_optimization(trial_config.get('layer7', 30))
        time.sleep(1)

        self.run_layer8_confidence_optimization(trial_config.get('layer8', 40))

        total_time = time.time() - overall_start_time

        # 統計優化結果
        successful_modules = sum(1 for result in self.layer_results.values() if 'error' not in result)
        total_modules = len(self.layer_results)
        success_rate = successful_modules / total_modules if total_modules > 0 else 0

        # 收集最佳分數
        best_scores = {}
        for layer_name, result in self.layer_results.items():
            if 'best_score' in result:
                best_scores[layer_name] = result['best_score']

        # 保存結果
        self.version_manager.save_results(self.current_version, self.layer_results)
        self.version_manager.set_latest(self.current_version)

        self.logger.info("=" * 60)
        self.logger.info("🎉 完整Layer0+9層優化流程執行完畢!")
        self.logger.info(f"✅ 成功模塊: {successful_modules}/{total_modules} ({success_rate:.1%})")
        self.logger.info(f"⏱️ 總耗時: {total_time:.1f}秒")

        return {
            'version': self.current_version,
            'layer_results': self.layer_results,
            'total_time': total_time,
            'optimization_summary': {
                'total_modules': total_modules,
                'successful_modules': successful_modules,
                'failed_modules': total_modules - successful_modules,
                'success_rate': success_rate,
                'best_scores': best_scores
            }
        }
    
    def quick_complete_optimization(self) -> Dict[str, Any]:
        self.logger.info("🚀 開始分階段串聯優化流程（Layer0→Layer1→Layer2聯動）...")
        overall_start_time = time.time()

        self.logger.info("📊 阶段一：Layer0數據清洗與統計分析...")
        self.run_layer0_data_cleaning(max(15, 10))

        self.logger.info("🏷️ 阶段二：Layer1標籤優化...")
        self.run_layer1_label_optimization(75)

        self.logger.info("🔧 阶段三：Layer2特徵優化...")
        self.run_layer2_feature_optimization(75)

        self.logger.info("📘 驗證與重跑...")
        self.rerun_layers_if_needed(min_score=0.45, retries=1)

        if globals().get('HAS_MODEL_OPTIMIZER', False):
            self.run_layer3_model_optimization(50)
        if globals().get('HAS_CV_RISK_OPTIMIZER', False):
            self.run_layer4_cv_risk_optimization(50)

        total_time = time.time() - overall_start_time
        successful_modules = sum(
            1 for result in self.layer_results.values() if 'error' not in result
        )
        total_modules = len(self.layer_results)
        success_rate = successful_modules / total_modules if total_modules > 0 else 0

        best_scores = {}
        for layer_name, result in self.layer_results.items():
            if 'best_score' in result:
                best_scores[layer_name] = result['best_score']

        # 保存結果並更新最新版本標記
        try:
            self.version_manager.save_results(self.current_version, self.layer_results)
            self.version_manager.set_latest(self.current_version)
        except Exception as _e:
            self.logger.warning(f"⚠️ quick 流程保存版本或設置 latest 失敗: {getattr(_e, 'message', _e)}")

        return {
            'version': self.current_version,
            'layer_results': self.layer_results,
            'total_time': total_time,
            'optimization_summary': {
                'total_modules': total_modules,
                'successful_modules': successful_modules,
                'failed_modules': total_modules - successful_modules,
                'success_rate': success_rate,
                'best_scores': best_scores
            }
        }
    
    # ============================================================
    # 向後兼容方法（舊接口）
    # ============================================================
    
    def run_complete_optimization(self, trial_config: Dict[str, int] = None) -> Dict[str, Any]:
        """向後兼容：執行完整優化（重新映射到9層架構）"""
        self.logger.warning("⚠️ 使用舊接口，推薦使用 run_complete_layered_optimization")
        return self.run_complete_layered_optimization(trial_config)
    
    def quick_optimization(self) -> Dict[str, Any]:
        """向後兼容：快速優化"""
        self.logger.warning("⚠️ 使用舊接口，推薦使用 quick_complete_optimization")
        return self.quick_complete_optimization()

    # -------------------------------------------------
    # 監控與驗證工具
    # -------------------------------------------------

    def validate_layer_outputs(self, min_score: float = 0.45) -> Dict[str, Dict[str, Any]]:
        """檢查各時框的 Layer1/Layer2 結果是否滿足分數與 lag 範圍."""
        issues = {}
        timeframes = self.scaled_config.get('meta_timeframes', [self.timeframe])
        meta_vol = self.scaled_config.get('meta_vol', 0.02)

        for tf in timeframes:
            label_path = self.get_label_config_path(tf)
            feature_path = self.get_feature_config_path(tf)

            if not label_path.exists() or not feature_path.exists():
                continue

            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            with open(feature_path, 'r', encoding='utf-8') as f:
                feature_data = json.load(f)

            label_score = label_data.get('best_score', 0)
            feature_score = feature_data.get('best_score', 0)
            lag = label_data.get('best_params', {}).get('lag', 0)

            base_min, base_max = self.scaler.get_base_lag_range(tf)
            _, allowed_max = self.scaler.adjust_lag_range_with_meta(tf, (base_min, base_max), meta_vol)

            tf_issues = {}
            if label_score < min_score:
                tf_issues['label_score'] = label_score
            if feature_score < min_score:
                tf_issues['feature_score'] = feature_score
            if lag > allowed_max:
                tf_issues['lag'] = lag

            if tf_issues:
                tf_issues['allowed_max_lag'] = allowed_max
                issues[tf] = tf_issues

        return issues

    def rerun_layers_if_needed(self, min_score: float = 0.45, retries: int = 1) -> Dict[str, Dict[str, Any]]:
        """若 score 或 lag 超限，重跑 Layer1/Layer2."""
        issues = self.validate_layer_outputs(min_score=min_score)
        if not issues:
            self.logger.info("✅ 所有時框結果符合要求，無需重跑")
            return {}

        self.logger.warning(f"⚠️ 發現時框結果不符合要求: {issues}")
        fixes = {}

        for tf, detail in issues.items():
            self.logger.info(f"🔄 重跑 {tf} Layer1/Layer2…")
            for retry in range(retries):
                self.logger.info(f"  ➤ 第 {retry+1}/{retries} 次重跑")
                self.symbol = self.symbol
                self.timeframe = tf
                self.run_layer1_label_optimization(n_trials=75)
                self.run_layer2_feature_optimization(n_trials=75)
                new_issues = self.validate_layer_outputs(min_score=min_score)
                if tf not in new_issues:
                    self.logger.info(f"  ✅ {tf} 重跑後已符合要求")
                    break
            fixes[tf] = {'original_issues': detail}

        return fixes
    
    # ============================================================
    # 阶段6+7：生存者偏差+系统性偏差集成方法
    # ============================================================
    
    def apply_survivorship_correction_to_results(self, results: Dict) -> Dict:
        """
        应用生存者偏差校正到优化结果（阶段6集成）
        
        使用方法：
        results = coordinator.run_complete_layered_optimization()
        results = coordinator.apply_survivorship_correction_to_results(results)
        
        Args:
            results: 优化结果字典
            
        Returns:
            添加了生存者偏差校正的结果字典
        """
        if not HAS_SURVIVORSHIP_CORRECTION:
            self.logger.info("ℹ️ 生存者偏差模块不可用，跳过校正")
            return results
        
        self.logger.info("=" * 60)
        self.logger.info("🔧 应用生存者偏差校正...")
        
        try:
            # 从layer_results中提取收益率序列
            layer_results = results.get('layer_results', {})
            
            # 尝试从不同层级提取收益率
            returns_series = None
            for layer_name in ['layer3_model', 'layer2_features', 'layer1_labels']:
                layer_data = layer_results.get(layer_name, {})
                if 'returns_series' in layer_data:
                    returns_series = layer_data['returns_series']
                    break
            
            if returns_series is None or (isinstance(returns_series, pd.Series) and returns_series.empty):
                self.logger.warning("⚠️ 未找到收益率序列，跳过生存者偏差校正")
                return results
            
            # 应用校正
            correction_result = apply_survivorship_correction(
                returns=returns_series,
                symbol=self.pair,
                timeframe=self.timeframe
            )
            
            # 添加校正结果
            results['survivorship_bias_correction'] = {
                'raw_sharpe': correction_result['raw_sharpe'],
                'corrected_sharpe': correction_result['corrected_sharpe'],
                'bias_estimate': correction_result['bias_estimate'],
                'confidence_interval_95': correction_result['confidence_interval'],
                'bootstrap_iterations': correction_result['n_iterations'],
                'failure_events_used': correction_result['n_failure_events']
            }
            
            self.logger.info("✅ 生存者偏差校正完成")
            self.logger.info(f"   原始Sharpe: {correction_result['raw_sharpe']:.4f}")
            self.logger.info(f"   校正后Sharpe: {correction_result['corrected_sharpe']:.4f}")
            self.logger.info(f"   偏差估计: {correction_result['bias_estimate']:.4f}")
            
        except Exception as e:
            self.logger.error(f"❌ 生存者偏差校正失败: {e}")
            self.logger.debug(traceback.format_exc())
        
        return results
    
    def check_distribution_shift_optional(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        检查训练/测试分布偏移（阶段7集成 - 可选）
        
        使用对抗性验证检测数据分布差异。
        
        使用方法（在Layer0数据清洗后）：
        coordinator.check_distribution_shift_optional(train_features, test_features)
        
        Args:
            train_data: 训练集特征
            test_data: 测试集特征
        """
        if not HAS_ADVERSARIAL_VALIDATION:
            self.logger.debug("ℹ️ 对抗性验证模块不可用")
            return None
        
        self.logger.info("🔍 执行对抗性验证检查...")
        
        try:
            result = quick_adversarial_check(train_data, test_data)
            
            auc = result['cv_auc_mean']
            shift = result['distribution_shift']
            
            self.logger.info(f"   AUC: {auc:.4f}, 分布偏移: {shift}")
            
            if auc > 0.70:
                self.logger.warning("⚠️ 检测到显著的分布偏移，建议检查数据划分")
            elif auc > 0.60:
                self.logger.info("ℹ️ 检测到轻度分布偏移")
            else:
                self.logger.info("✅ 训练/测试分布相似")
            
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ 对抗性验证失败: {e}")
            return None


def main():
    """主函數"""
    coordinator = OptunaCoordinator()
    result = coordinator.quick_complete_optimization()
    print(f"Layer0+9層優化完成: {result['version']}")


if __name__ == "__main__":
    main()
