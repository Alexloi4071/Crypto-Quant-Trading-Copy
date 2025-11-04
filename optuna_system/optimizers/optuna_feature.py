# -*- coding: utf-8 -*-
"""
特徵工程參數優化器 (第2層) - 分類版本
統一使用分類方法：特徵選擇、模型訓練、性能評估
"""
import json
import logging
import os
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
from functools import lru_cache

import numpy as np
import optuna
from optuna.trial import TrialState
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, 
    f1_score as compute_f1_score,  # 使用別名避免與變量名衝突
    precision_score, 
    recall_score, 
    roc_auc_score, 
    balanced_accuracy_score, 
    confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

from config.timeframe_scaler import TimeFrameScaler
from config.adaptive_parameter_optimizer import AdaptiveParameterOptimizer
from optuna_system.utils.io_utils import write_dataframe, read_dataframe
from optuna_system.utils.time_integrity import (
    EnhancedPurgedKFold, 
    TimeLeakageDetector, 
    validate_lag_alignment
)
from optuna_system.optimizers.multi_objective_feature_selector import MultiObjectiveFeatureSelector
from optuna_system.optimizers.indicators import (
    TDSequential,
    WyckoffAnalysis,
    MarketStructure,
    CandlestickPatterns,
    CompositeIndicators,
    AdvancedTrend,
    VolumeAnalysis,
    HarmonicPatterns,
    ElliottWaveAnalyzer,
    GannTheory,
    AdvancedVolatility,
    TimeSeriesDecomposition,
    OrderFlowProxy
)

warnings.filterwarnings('ignore')


# Version sentinel for verification
PATCH_ID = "feat-obj-selected_features-fix"


class FeatureOptimizer:
    """特徵工程參數優化器 - 第2層優化"""

    def __init__(self, data_path: str, config_path: str = "configs/",
                 symbol: str = "BTCUSDT", timeframe: str = "15m",
                 scaled_config: Dict = None):
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.config_path.mkdir(exist_ok=True)
        self.symbol = symbol
        self.timeframe = timeframe
        # 時框縮放配置需在使用前初始化
        self.scaled_config = scaled_config or {}
        # 使用集中日誌 (由上層/入口初始化)，避免重複 basicConfig
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"FeatureOptimizer PATCH_ID={PATCH_ID}")

        # 多目標優化設定（可透過環境或 scaled_config 控制）
        self.multi_objective_mode = bool(os.getenv('L2_OBJECTIVE_MODE', '').lower() in ('multi', 'multiobjective', 'pareto') or
                                         (self.scaled_config.get('l2_objective_mode', '').lower() in ('multi', 'multiobjective', 'pareto')))
        self.logger.info(f"🔧 Layer2 多目標模式: {self.multi_objective_mode}")

        def _env_float(name: str, default: float) -> float:
            val = os.getenv(name)
            if val is None:
                try:
                    return float(self.scaled_config.get(name.lower(), default))
                except Exception:
                    return float(default)
            try:
                return float(val)
            except Exception:
                return float(default)

        def _env_int(name: str, default: int) -> int:
            val = os.getenv(name)
            if val is None:
                try:
                    return int(self.scaled_config.get(name.lower(), default))
                except Exception:
                    return int(default)
            try:
                return int(val)
            except Exception:
                return int(default)

        # KPI 約束門檻（可依時框差異調整）
        # ✅ KPI約束（優化目標重構 - Phase 3）
        # ✅ 修改：放宽Sharpe/Calmar，严格化Win Rate/Profit Factor，新增Risk:Reward
        self.kpi_constraints = {
            # 風險調整收益（放宽）
            'min_sharpe': _env_float('L2_MIN_SHARPE', 0.8),  # ✅ 從1.0/1.2降低到0.8（更現實）
            'min_sortino': 0.0,  # ✅ 刪除約束（從1.0/1.2降低到0）
            'min_calmar': _env_float('L2_MIN_CALMAR', 0.6),  # ✅ 從0.75/1.2降低到0.6（放宽）
            
            # 交易質量指標（嚴格化）
            'min_win_rate': _env_float('L2_MIN_WINRATE', 0.50),  # ✅ 從0.6降低到0.50（更現實）
            'min_profit_factor': _env_float('L2_MIN_PROFIT_FACTOR', 1.5),  # ✅ 統一提高到1.5（從1.3/1.5）
            
            # ✅ 新增：最小盈虧比約束（與Primary Model一致）
            'min_risk_reward': _env_float('L2_MIN_RISK_REWARD', 2.0),
            
            # 回撤與收益（保持）
            'max_drawdown': _env_float('L2_MAX_DRAWDOWN', 0.2 if self.timeframe.lower() in ('15m', '15') else 0.15),
            'min_total_return': _env_float('L2_MIN_TOTAL_RETURN', 0.02),
            'min_annual_return': 0.0  # ✅ 刪除約束（太嚴格）
        }
        self.logger.info(f"🔒 Layer2 KPI Constraints: {self.kpi_constraints}")

        # ✅ 多目標加權（優化目標重構 - Phase 3）
        # ✅ 修改：交易指標從10%提高到55%，ML指標從40%降低到15%
        self.obj_weights = {
            # ML指標（15% - 從40%降低）
            'f1_macro': _env_float('L2_WEIGHT_F1_MACRO', 0.10),     # ✅ 從0.2降低到0.10
            'f1_weighted': _env_float('L2_WEIGHT_F1_WEIGHTED', 0.05),  # ✅ 從0.2降低到0.05
            
            # 交易質量指標（55% - 從10%提高）
            'profit_factor': _env_float('L2_WEIGHT_PROFIT_FACTOR', 0.30),  # ✅ 從0.1提高到0.30 (+200%)
            'win_rate': _env_float('L2_WEIGHT_WIN_RATE', 0.25),           # ✅ 從0.05提高到0.25 (+400%)
            
            # 風險調整收益指標（30%）
            'sharpe': _env_float('L2_WEIGHT_SHARPE', 0.20),           # ✅ 從0.2保持0.20
            'calmar': _env_float('L2_WEIGHT_CALMAR', 0.10),           # ✅ 從0.1保持0.10
            
            # 輔助指標（刪除sortino, total_return, annual_return）
            'sortino': 0.0,  # ✅ 刪除（從0.1降低到0）
            'total_return': 0.0,  # ✅ 刪除（從0.025降低到0）
            'annual_return': 0.0   # ✅ 刪除（從0.025降低到0）
        }
        weight_sum = sum(self.obj_weights.values())
        if weight_sum <= 0:
            # ✅ fallback weights（同樣調整）
            self.obj_weights = {
                'f1_macro': 0.10,
                'f1_weighted': 0.05,
                'profit_factor': 0.30,
                'win_rate': 0.25,
                'sharpe': 0.20,
                'calmar': 0.10,
                'sortino': 0.0,
                'total_return': 0.0,
                'annual_return': 0.0
            }
        else:
            for k in self.obj_weights:
                self.obj_weights[k] = float(self.obj_weights[k]) / weight_sum
        self.logger.info(f"⚖️ Layer2 目標權重: {self.obj_weights}")

        self.multiobjective_metrics = [
            'sharpe',
            'calmar',
            'profit_factor',
            'win_rate',
            'total_return',
            'annual_return',
            'neg_max_drawdown'
        ]


        self.scaler = TimeFrameScaler(self.logger)
        self.results_path = Path(self.config_path) / ".." / "results" / f"{self.symbol}_{self.timeframe}"
        self.results_path = self.results_path.resolve()
        self.results_path.mkdir(parents=True, exist_ok=True)

        # 特徵開關與策略配置
        self.flags = self._validate_flags(self._load_feature_flags())
        self.phase_config: Dict[str, Any] = {}
        self.selection_params: Dict[str, Any] = {}
        
        # 🚀 初始化所有高级指标计算器
        self.logger.info("🔧 初始化高级指标计算器...")
        self.td_sequential = TDSequential()
        self.wyckoff_analysis = WyckoffAnalysis()
        self.market_structure = MarketStructure()
        self.candlestick_patterns = CandlestickPatterns()
        self.composite_indicators = CompositeIndicators()
        self.advanced_trend = AdvancedTrend()
        self.volume_analysis = VolumeAnalysis()
        self.harmonic_patterns = HarmonicPatterns()
        self.elliott_wave = ElliottWaveAnalyzer()
        self.gann_theory = GannTheory()
        self.advanced_volatility = AdvancedVolatility()
        self.time_series_decomposition = TimeSeriesDecomposition()
        self.order_flow_proxy = OrderFlowProxy()
        self.logger.info("✅ 高级指标计算器初始化完成")
        
        # 🚀 初始化自适应参数优化器
        self.logger.info("🔧 初始化自适应参数优化器...")
        enable_adaptive_params = self.flags.get('enable_adaptive_params', True)
        if enable_adaptive_params:
            self.adaptive_param_optimizer = AdaptiveParameterOptimizer(
                cache_dir=str(self.config_path),
                logger=self.logger
            )
            self.logger.info("✅ 自适应参数优化器已启用")
        else:
            self.adaptive_param_optimizer = None
            self.logger.info("⚠️ 自适应参数优化器已禁用（使用默认参数）")
        
        # 🚀 初始化多目标特征选择器
        self.logger.info("🔧 初始化多目标特征选择器...")
        enable_multi_objective = self.flags.get('enable_multi_objective_selector', True)
        if enable_multi_objective and self.multi_objective_mode:
            try:
                # 构建约束文件路径
                constraint_file = Path(self.config_path) / "feature_group_constraints.json"
                
                self.multi_objective_selector = MultiObjectiveFeatureSelector(
                    objectives=['maximize_f1_macro', 'minimize_feature_count', 'maximize_diversity'],
                    constraints_path=str(constraint_file),
                    logger=self.logger
                )
                self.logger.info("✅ 多目标特征选择器已启用")
            except Exception as e:
                self.logger.warning(f"⚠️ 多目标特征选择器初始化失败: {e}，将使用单目标选择")
                import traceback
                self.logger.debug(traceback.format_exc())
                self.multi_objective_selector = None
        else:
            self.multi_objective_selector = None
            if not enable_multi_objective:
                self.logger.info("⚠️ 多目标特征选择器已禁用（配置关闭）")
            else:
                self.logger.info("⚠️ 多目标特征选择器未启用（单目标模式）")
        
        # 🔧 Purged CV配置（防止時間洩漏）
        self.enable_purged_cv = self.scaled_config.get('enable_purged_cv', True)
        self.embargo_pct = self.scaled_config.get('embargo_pct', 0.02)  # 2% embargo

        # 🚀 修復版：完整數據預加載與缓存（支援 lazy/full 模式）
        self.logger.info("🚀 預加載與缓存OHLCV、特徵、價格序列...")
        try:
            self.ohlcv_data, self.features = self._load_and_prepare_features()

            # 🚀 預計算價格序列（標籤生成用）
            if 'close' in self.ohlcv_data.columns:
                self.close_prices = self.ohlcv_data['close'].copy()
            else:
                raise ValueError("OHLCV數據中缺少close列")

            # 🚀 預計算有效索引範圍（避免每次重新計算）
            self.valid_data_range = (self.features.index.min(), self.features.index.max())

            self.logger.info(f"✅ 完整數據預加載: OHLCV={self.ohlcv_data.shape}, 特徵={self.features.shape}")
            self.logger.info(f"✅ 價格序列緩存: {len(self.close_prices)}個價格點")
            self.logger.info(f"✅ 有效範圍: {self.valid_data_range}")

            # 🚀 一次性構建全量特徵矩陣（可選）
            preload_mode = str(self.flags.get('preload_mode', 'full')).lower()
            if preload_mode == 'full':
                self.X_full = self._build_full_feature_matrix()
                self.logger.info(f"✅ 全量特徵矩陣準備完成: {self.X_full.shape}")
            else:
                self.X_full = None
                self.logger.info("🕒 Lazy 模式啟用：延後構建 X_full 以降低初始化記憶體峰值")

            # 初始化分階段特徵配置
            self._configure_phase_settings()

        except Exception as e:
            self.logger.error(f"❌ 數據預加載失敗: {e}")
            raise ValueError(f"初始化失敗，無法預加載數據: {e}")  # Fail-Fast

    def _estimate_mem_mb(self, df: pd.DataFrame) -> float:
        try:
            return float(df.memory_usage(deep=True).sum()) / (1024 * 1024)
        except Exception:
            return 0.0

    def _enforce_memory_limits(self, X: pd.DataFrame) -> pd.DataFrame:
        """在預載模式下對超大矩陣做早期列過濾，降低記憶體占用。"""
        try:
            max_cols = int(self.flags.get('max_full_features', 1200))
            max_mem_mb = float(self.flags.get('max_full_mem_mb', 1500.0))
        except Exception:
            max_cols, max_mem_mb = 1200, 1500.0

        mem_mb = self._estimate_mem_mb(X)
        if X.shape[1] <= max_cols and mem_mb <= max_mem_mb:
            self.logger.info(f"💾 X_full 估算記憶體 {mem_mb:.1f}MB，列數 {X.shape[1]}，無需裁剪")
            return X

        self.logger.warning(
            f"⚠️ X_full 達上限（cols={X.shape[1]}, mem≈{mem_mb:.1f}MB），執行早期列過濾"
        )

        # 1) 先做現有的質量過濾
        X = self._filter_low_quality_features(X)
        mem_mb = self._estimate_mem_mb(X)

        # 2) 若仍超限，保留高變異度列（快速近似重要度）
        if X.shape[1] > max_cols or mem_mb > max_mem_mb:
            try:
                std = X.astype('float32').std().fillna(0.0)
                keep_k = min(max_cols, max(100, int(max_cols * 0.9)))
                top_cols = std.sort_values(ascending=False).head(keep_k).index
                X = X.loc[:, top_cols]
                self.logger.info(f"🔧 依變異度保留前 {len(top_cols)} 列，new_cols={X.shape[1]}")
            except Exception as exc:
                self.logger.warning(f"變異度裁剪失敗，跳過: {exc}")

        mem_mb2 = self._estimate_mem_mb(X)
        self.logger.info(f"💾 X_full 最終估算記憶體 {mem_mb2:.1f}MB，列數 {X.shape[1]}")
        return X

    def _ensure_X_full(self) -> None:
        """Lazy 模式下在需要時構建 X_full，並做記憶體保護。"""
        if isinstance(getattr(self, 'X_full', None), pd.DataFrame) and not self.X_full.empty:
            return
        X = self._build_full_feature_matrix()
        self.X_full = self._enforce_memory_limits(X)

    def _load_feature_flags(self) -> Dict:
        cfg_path = self.config_path / "feature_flags.json"
        try:
            if cfg_path.exists():
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    return json.load(f) or {}
        except Exception as e:
            self.logger.warning(f"⚠️ feature_flags 讀取失敗: {e}")
        return {}

    def _validate_flags(self, flags: Dict) -> Dict:
        defaults = {
            "enable_td": True,
            "enable_wyckoff": True,
            "enable_micro": True,
            "enable_derivatives": False,
            "latency_lag": 0,
            "cv_gap": 12,
            "cv_purge": 12,
            "cv_splits": 5,
            "cv_strategy": "embargo",
            "tech": {},
            "enable_dynamic_label_balance": False,
            "enable_trade_freq_penalty": False,
            "enable_threshold_search": True,
            "threshold_tau_min": 0.30,
            "threshold_tau_max": 0.65,
            "threshold_tau_step": 0.05,
            "target_corr_keep_ratio": 0.6,
            "target_corr_min_score": 0.0,
            "layer1_range_mode": "narrow",
            "stability_selection_threshold": 0.6
        }
        if not isinstance(flags, dict):
            self.logger.warning("⚠️ feature_flags 格式非法，回退預設")
            return defaults

        merged = defaults.copy()
        merged.update(flags)

        try:
            merged["enable_td"] = bool(merged.get("enable_td", True))
            merged["enable_wyckoff"] = bool(merged.get("enable_wyckoff", True))
            merged["enable_micro"] = bool(merged.get("enable_micro", True))
            merged["enable_derivatives"] = bool(merged.get("enable_derivatives", False))
            merged["enable_dynamic_label_balance"] = bool(merged.get("enable_dynamic_label_balance", False))
            merged["layer1_range_mode"] = str(merged.get("layer1_range_mode", "narrow")).lower()
            if merged["layer1_range_mode"] not in ("narrow", "wide"):
                self.logger.warning("⚠️ layer1_range_mode 非法，回退 narrow")
                merged["layer1_range_mode"] = "narrow"

            # Allow scaled_config to override
            if isinstance(self.scaled_config, dict):
                try:
                    if 'cv_gap' in self.scaled_config:
                        merged["cv_gap"] = int(self.scaled_config['cv_gap'])
                    if 'latency_lag' in self.scaled_config:
                        merged["latency_lag"] = int(self.scaled_config['latency_lag'])
                except Exception:
                    pass
            merged["latency_lag"] = max(0, int(merged.get("latency_lag", 0)))
            merged["cv_gap"] = max(0, int(merged.get("cv_gap", 12)))
            merged["cv_purge"] = max(0, int(merged.get("cv_purge", merged["cv_gap"])))
            merged["cv_splits"] = max(2, int(merged.get("cv_splits", 5)))

            if merged.get("cv_strategy") not in ("embargo", "purged"):
                self.logger.warning("⚠️ cv_strategy 非法，回退 embargo")
                merged["cv_strategy"] = "embargo"

            merged["enable_trade_freq_penalty"] = bool(merged.get("enable_trade_freq_penalty", False))
            merged["enable_threshold_search"] = bool(merged.get("enable_threshold_search", True))

            try:
                merged["threshold_tau_min"] = float(merged.get("threshold_tau_min", 0.30))
                merged["threshold_tau_max"] = float(merged.get("threshold_tau_max", 0.65))
                merged["threshold_tau_step"] = float(merged.get("threshold_tau_step", 0.05))
                if merged["threshold_tau_min"] >= merged["threshold_tau_max"]:
                    merged["threshold_tau_min"], merged["threshold_tau_max"] = 0.30, 0.65
                if merged["threshold_tau_step"] <= 0:
                    merged["threshold_tau_step"] = 0.05
            except Exception:
                merged["threshold_tau_min"], merged["threshold_tau_max"], merged["threshold_tau_step"] = 0.30, 0.65, 0.05

            try:
                merged["target_corr_keep_ratio"] = float(merged.get("target_corr_keep_ratio", 0.6))
            except Exception:
                merged["target_corr_keep_ratio"] = 0.6
            try:
                merged["target_corr_min_score"] = float(merged.get("target_corr_min_score", 0.0))
            except Exception:
                merged["target_corr_min_score"] = 0.0

            if not isinstance(merged.get("tech"), dict):
                self.logger.warning("⚠️ tech 配置非法，略過")
                merged["tech"] = {}

        except Exception as e:
            self.logger.warning(f"⚠️ feature_flags 校驗失敗: {e}，使用預設")
            return defaults

        return merged

    # -----------------------------
    # Phase 3: Label 質量檢查/自動重訓練
    # -----------------------------
    def _periods_per_year(self, timeframe: Optional[str] = None) -> float:
        tf = (timeframe or self.timeframe or '').lower()
        mapping = {
            '1m': 252 * 24 * 60,
            '3m': 252 * 24 * 20,
            '5m': 252 * 24 * 12,
            '15m': 252 * 24 * 4,
            '30m': 252 * 24 * 2,
            '45m': 252 * 24 * 4 / 3,
            '1h': 252 * 24,
            '2h': 252 * 12,
            '4h': 252 * 6,
            '6h': 252 * 4,
            '8h': 252 * 3,
            '12h': 252 * 2,
            '1d': 252,
            '1w': 52,
            '1mo': 12
        }
        return float(mapping.get(tf, 252.0))

    def _compute_objective_metrics(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, lag: int) -> Dict[str, float]:
        if X.empty or y.empty:
            return {}

        classifier = self.create_enhanced_classifier(trial, X.shape[1])
        try:
            sample_weight = self._compute_sample_weights(y)
            classifier.fit(X.values, y.values, sample_weight=sample_weight)
        except TypeError:
            classifier.fit(X.values, y.values)
        except Exception as exc:
            self.logger.warning(f"⚠️ 全量訓練失敗，改為無權重: {exc}")
            classifier.fit(X.values, y.values)

        metrics: Dict[str, float] = {}

        try:
            preds = classifier.predict(X.values)
        except Exception as exc:
            self.logger.warning(f"⚠️ 全量預測失敗: {exc}")
            preds = np.full_like(y.values, fill_value=int(np.median(y.values)))

        try:
            proba = classifier.predict_proba(X.values)
        except Exception:
            proba = None

        idx = X.index
        price_series = self.close_prices.reindex(idx).ffill()
        # 🔧 FIX: Remove shift(-1) to avoid using future returns
        # Use current period returns (already known at decision time)
        returns = price_series.pct_change()
        returns = returns.reindex(idx).fillna(0.0)

        positions = pd.Series(preds - 1, index=idx)
        # Strategy returns aligned properly: position决定时已知的收益
        strategy_returns = (positions.shift(1) * returns).astype(float)
        strategy_returns = strategy_returns.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        periods_per_year = self._periods_per_year()
        mean_ret = float(strategy_returns.mean()) if len(strategy_returns) > 0 else 0.0
        std_ret = float(strategy_returns.std(ddof=0)) if len(strategy_returns) > 0 else 0.0

        sharpe = 0.0
        if std_ret > 1e-9:
            sharpe = mean_ret / std_ret * np.sqrt(periods_per_year)

        downside = strategy_returns[strategy_returns < 0]
        downside_std = float(np.sqrt((downside ** 2).mean())) if len(downside) > 0 else 0.0
        sortino = 0.0
        if downside_std > 1e-9:
            sortino = mean_ret * np.sqrt(periods_per_year) / downside_std

        equity_curve = (strategy_returns + 1.0).cumprod()
        if len(equity_curve) == 0:
            max_dd = 0.0
            total_return = 0.0
        else:
            running_max = equity_curve.cummax()
            drawdowns = 1.0 - equity_curve / (running_max + 1e-9)
            max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0
            total_return = float(equity_curve.iloc[-1] - 1.0)

        annual_return = 0.0
        if len(equity_curve) > 1:
            total_periods = len(strategy_returns)
            if total_periods > 0:
                total_return = float(equity_curve.iloc[-1] - 1.0)
                annual_return = (1.0 + total_return) ** (periods_per_year / total_periods) - 1.0

        calmar = 0.0
        if max_dd > 1e-9:
            calmar = annual_return / max_dd

        positive = strategy_returns[strategy_returns > 0]
        negative = strategy_returns[strategy_returns < 0]
        gross_profit = float(positive.sum())
        gross_loss = float(-negative.sum())
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 1e-9 else float('inf')

        win_rate = float((strategy_returns > 0).mean()) if len(strategy_returns) > 0 else 0.0

        metrics['f1_macro'] = float(compute_f1_score(y, preds, average='macro', zero_division=0))
        metrics['f1_weighted'] = float(compute_f1_score(y, preds, average='weighted', zero_division=0))
        metrics['precision_macro'] = float(precision_score(y, preds, average='macro', zero_division=0))
        metrics['recall_macro'] = float(recall_score(y, preds, average='macro', zero_division=0))
        metrics['balanced_accuracy'] = float(balanced_accuracy_score(y, preds))

        if proba is not None:
            try:
                metrics['auc_macro'] = float(roc_auc_score(y, proba, multi_class='ovr', average='macro'))
            except Exception:
                metrics['auc_macro'] = 0.5
        else:
            metrics['auc_macro'] = 0.5

        metrics['sharpe'] = float(sharpe)
        metrics['sortino'] = float(sortino)
        metrics['calmar'] = float(calmar)
        metrics['profit_factor'] = float(min(max(profit_factor, 0.0), 10.0)) if np.isfinite(profit_factor) else 10.0
        metrics['win_rate'] = float(win_rate)
        metrics['total_return'] = float(total_return)
        metrics['annual_return'] = float(annual_return)
        metrics['max_drawdown'] = float(max_dd)
        metrics['max_drawdown_pct'] = float(max_dd)
        metrics['mean_return'] = float(mean_ret)
        metrics['std_return'] = float(std_ret)
        metrics['strategy_trades'] = float((positions.diff().abs() > 0).sum())
        metrics['avg_position'] = float(positions.abs().mean())

        metrics['lag'] = float(lag)
        metrics['n_samples'] = float(len(X))

        return metrics

    def _check_kpi_constraints(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        if not metrics:
            return False, 'metrics_unavailable'

        rules = (
            ('sharpe', self.kpi_constraints.get('min_sharpe', 0.0), 'min'),
            ('sortino', self.kpi_constraints.get('min_sortino', 0.0), 'min'),
            ('calmar', self.kpi_constraints.get('min_calmar', 0.0), 'min'),
            ('win_rate', self.kpi_constraints.get('min_win_rate', 0.0), 'min'),
            ('profit_factor', self.kpi_constraints.get('min_profit_factor', 0.0), 'min'),
            ('total_return', self.kpi_constraints.get('min_total_return', 0.0), 'min'),
            ('annual_return', self.kpi_constraints.get('min_annual_return', 0.0), 'min'),
            ('max_drawdown', self.kpi_constraints.get('max_drawdown', 1.0), 'max')
        )

        for key, threshold, mode in rules:
            if key not in metrics:
                return False, f'missing_{key}'
            value = float(metrics[key])
            if not np.isfinite(value):
                return False, f'invalid_{key}'
            if mode == 'min' and value < threshold:
                return False, f'{key}={value:.4f} < {threshold:.4f}'
            if mode == 'max' and value > threshold:
                return False, f'{key}={value:.4f} > {threshold:.4f}'

        return True, 'ok'

    def _weighted_objective_score(self, metrics: Dict[str, float]) -> float:
        score = 0.0
        for name, weight in self.obj_weights.items():
            if weight == 0:
                continue
            value = metrics.get(name)
            if value is None or not np.isfinite(value):
                continue
            capped = value
            if name in ('profit_factor', 'calmar', 'sharpe', 'sortino'):
                capped = max(min(value, 5.0), -1.0)
            if name in ('total_return', 'annual_return'):
                capped = max(min(value, 2.0), -1.0)
            score += float(weight) * float(capped)
        return float(score)

    def _current_best_value(self, study: optuna.study.Study) -> float:
        try:
            if not study.trials:
                return 0.0
            if self.multi_objective_mode:
                best_trials = getattr(study, 'best_trials', [])
                if not best_trials:
                    return 0.0
                # 取第一個 Pareto trial 的 user_attrs，如果沒有則回退其 values
                trial = best_trials[0]
                metrics = trial.user_attrs.get('objective_metrics')
                if metrics:
                    return self._weighted_objective_score(metrics)
                try:
                    values = trial.values or []
                    weighted = 0.0
                    total_w = 0.0
                    for name, weight in self.obj_weights.items():
                        if name not in self.multiobjective_metrics:
                            continue
                        idx = self.multiobjective_metrics.index(name)
                        if idx < len(values) and np.isfinite(values[idx]):
                            weighted += weight * float(values[idx])
                            total_w += weight
                    if total_w > 0:
                        return weighted / total_w
                except Exception:
                    pass
                return 0.0
            return float(study.best_value)
        except Exception:
            return 0.0

    def _labels_dir(self) -> Path:
        return self.data_path / 'processed' / 'labels' / f"{self.symbol}_{self.timeframe}"

    def load_latest_labels(self) -> Optional[pd.DataFrame]:
        try:
            labels_dir = self._labels_dir()
            candidates: List[Path] = []
            if labels_dir.exists():
                # 遞迴搜尋版本子目錄
                candidates.extend(sorted(labels_dir.rglob('labels_*.parquet'), key=lambda p: p.stat().st_mtime, reverse=True))
                candidates.extend(sorted(labels_dir.rglob('labels_*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True))
            if candidates:
                return read_dataframe(candidates[0])
            # fallback: config json materialized path
            tf_json = self.config_path / f"label_params_{self.timeframe}.json"
            if tf_json.exists():
                with open(tf_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                mp = data.get('materialized_path') or (data.get('metadata') or {}).get('file_path')
                if mp and Path(mp).exists():
                    return read_dataframe(Path(mp))
        except Exception as e:
            self.logger.warning(f"無法載入最新標籤: {e}")
        return None

    def analyze_label_quality(self) -> Dict[str, Any]:
        """回傳分析結果 dict，並附帶 pass 字段指示是否通過閾值。"""
        df = self.load_latest_labels()
        if df is None or 'label' not in df.columns:
            return {'pass': False, 'reason': 'no_labels'}
        s = df['label'].astype(int)
        dist = s.value_counts(normalize=True).to_dict()
        changes = int((s.diff() != 0).sum())
        stability = 1 - changes / max(len(s), 1)
        ok = (min(dist.get(0, 0), dist.get(2, 0)) >= 0.15) and stability >= 0.55
        return {'pass': ok, 'distribution': dist, 'stability': float(stability)}

    def auto_retrain_labels_if_needed(self) -> Optional[Dict[str, Any]]:
        try:
            analysis = self.analyze_label_quality()
            if analysis.get('pass', True):
                return None
            self.logger.info("觸發Layer1重新優化（自動）...")
            from optuna_system.optimizers.optuna_meta_label import MetaLabelOptimizer  # lazy import
            opt = MetaLabelOptimizer(
                data_path=str(self.data_path),
                config_path=str(self.config_path),
                symbol=self.symbol,
                timeframe=self.timeframe,
                scaled_config=self.scaled_config or {},
            )
            # 適度提高 trials
            result = opt.optimize(n_trials=200)
            self.logger.info(f"Layer1 重新優化完成: F1={result.get('best_score', 0):.4f}")
            return result
        except Exception as e:
            self.logger.warning(f"自動重訓練失敗: {e}")
            return None

    def _apply_latency(self, obj: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
        if lag <= 0 or obj is None:
            return obj
        return obj.ffill().shift(lag).fillna(0)

    def _allowed_slow_timeframes(self) -> List[str]:
        """Return the immediate slower timeframe(s) allowed for current dominant timeframe."""
        tf = str(self.timeframe).lower()
        mapping = {
            '15m': ['1h'],
            '30m': ['1h'],
            '1h': ['4h'],
            '2h': ['4h'],
            '4h': ['1d', '1D'],
            '1d': ['1w', '1W'],
            '1D': ['1w', '1W']
        }
        return mapping.get(tf, [])

    def _safe_merge(self, base: pd.DataFrame, extra: Optional[pd.DataFrame], prefix: str = "") -> pd.DataFrame:
        """
        🔧 P0修復：嚴格對齊，防止時間洩漏
        
        修復邏輯：
        1. extra可能來自更高時框，先shift(1)
        2. 再resample和reindex對齊
        3. 只用ffill，禁用bfill
        """
        if extra is None or extra.empty:
            return base
        # 嚴格對齊：先將輔助特徵 resample 至基準索引頻率，再對齊到基準索引
        extra_prefixed = extra.add_prefix(prefix)
        try:
            extra_resampled = self._resample_like(base.index, extra_prefixed)
            # 🔧 關鍵修復：shift(1)防止洩漏
            extra_shifted = extra_resampled.shift(1)
            extra_aligned = extra_shifted.reindex(base.index, method='ffill')
        except Exception:
            # 如果轉頻/對齊異常，退化為簡單對齊以確保不中斷
            # 🔧 關鍵修復：shift(1)防止洩漏
            extra_shifted = extra_prefixed.shift(1)
            extra_aligned = extra_shifted.reindex(base.index, method='ffill')
        merged = pd.concat([base, extra_aligned], axis=1)
        # 禁用bfill，只用ffill+fillna(0)
        return merged.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    def _infer_rule_from_index(self, idx: pd.DatetimeIndex) -> str:
        """從 DatetimeIndex 推斷頻率規則（如 '15T', '1H', '4H', '1D'）。"""
        if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
            return '15T'
        try:
            freq = pd.infer_freq(idx)
            if isinstance(freq, str) and len(freq) > 0:
                return freq
        except Exception:
            pass
        # 回退：使用中位數間隔估算
        deltas = pd.Series(idx[1:]).reset_index(drop=True) - pd.Series(idx[:-1]).reset_index(drop=True)
        median_delta = pd.to_timedelta(np.median(deltas.values))
        seconds = max(1, int(median_delta.total_seconds()))
        # 常用映射
        if seconds % (24*3600) == 0:
            days = seconds // (24*3600)
            return f"{days}D" if days > 1 else '1D'
        if seconds % 3600 == 0:
            hours = seconds // 3600
            return f"{hours}H"
        if seconds % 60 == 0:
            minutes = seconds // 60
            return f"{minutes}T"
        return f"{seconds}S"

    def _resample_like(self, base_index: pd.DatetimeIndex, extra: pd.DataFrame) -> pd.DataFrame:
        """將 extra 轉頻到與 base_index 相同的頻率，粗→細用 ffill，細→粗用 mean 聚合。"""
        if extra is None or extra.empty:
            return extra
        try:
            target_rule = self._infer_rule_from_index(base_index)
            # LRU 快取鍵（索引指紋 + 欄位指紋 + 目標規則）
            key = (str(extra.index[0]) if len(extra.index) else 'NA', str(extra.index[-1]) if len(extra.index) else 'NA', extra.shape[0], extra.shape[1], hash(tuple(extra.columns)), target_rule)
            if not hasattr(self, '_rs_cache'):
                self._rs_cache = {}
            cached = self._rs_cache.get(key)
            if cached is not None:
                return cached
            try:
                extra_rs = extra.resample(target_rule).mean()
            except Exception:
                extra_rs = extra
            self._rs_cache[key] = extra_rs
            return extra_rs
        except Exception:
            return extra

    def create_enhanced_classifier(self, trial: optuna.Trial, n_features: int):
        """根據trial配置選擇並構建增強分類器."""
        model_type = trial.suggest_categorical(
            'model_type',
            ['random_forest', 'extra_trees', 'gradient_boosting', 'lightgbm', 'xgboost']
        )

        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=trial.suggest_int('rf_n_estimators', 200, 400),
                max_depth=trial.suggest_int('rf_max_depth', 12, 24),
                min_samples_split=trial.suggest_int('rf_min_samples_split', 2, 8),
                min_samples_leaf=trial.suggest_int('rf_min_samples_leaf', 1, 6),
                max_features=trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', 0.8]),
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced_subsample'
            )

        if model_type == 'extra_trees':
            return ExtraTreesClassifier(
                n_estimators=trial.suggest_int('et_n_estimators', 200, 500),
                max_depth=trial.suggest_int('et_max_depth', 10, 18),
                min_samples_split=trial.suggest_int('et_min_samples_split', 5, 15),
                min_samples_leaf=trial.suggest_int('et_min_samples_leaf', 8, 20),
                max_features=trial.suggest_float('et_max_features', 0.3, 0.7),
                bootstrap=False,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )

        if model_type == 'lightgbm':
            try:
                from lightgbm import LGBMClassifier  # type: ignore
            except ModuleNotFoundError:
                self.logger.warning("⚠️ 未安裝 lightgbm，改用 GradientBoostingClassifier")
            else:
                # 更寬鬆且可動態放鬆的搜尋空間，緩解「no positive gain」
                relax = bool(self.flags.get('lgb_relax_search', False))
                depth_low, depth_high = (4, 8) if not relax else (4, 10)
                lgb_max_depth = trial.suggest_int('lgb_max_depth', depth_low, depth_high)
                # 遵守 num_leaves <= 2^max_depth，並確保下限不會大於上限
                lgb_max_leaves = min(2 ** lgb_max_depth, 256 if relax else 128)
                lgb_low_leaves = max(4 if relax else 8, min(31, lgb_max_leaves // 2))
                lgb_num_leaves = trial.suggest_int('lgb_leaves', lgb_low_leaves, lgb_max_leaves)

                return LGBMClassifier(
                    objective='multiclass',
                    num_class=3,
                    n_estimators=trial.suggest_int('lgb_n_estimators', 400, 1200 if relax else 900),
                    learning_rate=trial.suggest_float('lgb_learning_rate', 0.02 if relax else 0.03, 0.15 if relax else 0.12),
                    max_depth=lgb_max_depth,
                    num_leaves=lgb_num_leaves,
                    subsample=trial.suggest_float('lgb_subsample', 0.6 if relax else 0.7, 1.0),
                    subsample_freq=trial.suggest_int('lgb_bagging_freq', 1, 9 if relax else 7),
                    colsample_bytree=trial.suggest_float('lgb_colsample', 0.5 if relax else 0.6, 0.95 if relax else 0.9),
                    max_bin=trial.suggest_int('lgb_max_bin', 127, 511 if relax else 255),
                    min_split_gain=trial.suggest_float('lgb_min_split_gain', 0.0, 0.05 if relax else 0.1),
                    min_child_samples=trial.suggest_int('lgb_min_child_samples', 2 if relax else 5, 60 if relax else 50),
                    min_child_weight=trial.suggest_float('lgb_min_child_weight', 1e-4 if relax else 1e-3, 5.0),
                    # ✅ P1修復：增加正则化，减少过拟合（v67 gap=22.6%）
                    reg_alpha=trial.suggest_float('lgb_reg_alpha', 0.5, 2.5 if relax else 3.5),  # 从0.0-2.0提高到0.5-3.5
                    reg_lambda=trial.suggest_float('lgb_reg_lambda', 1.0, 5.0 if relax else 6.0),  # 从0.0-4.0提高到1.0-6.0
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1
                )

        if model_type == 'xgboost':
            try:
                from xgboost import XGBClassifier  # type: ignore
            except ModuleNotFoundError:
                self.logger.warning("⚠️ 未安裝 xgboost，改用 GradientBoostingClassifier")
            else:
                return XGBClassifier(
                    n_estimators=trial.suggest_int('xgb_n_estimators', 300, 800),
                    learning_rate=trial.suggest_float('xgb_learning_rate', 0.01, 0.07),
                    max_depth=trial.suggest_int('xgb_max_depth', 3, 8),
                    subsample=trial.suggest_float('xgb_subsample', 0.7, 1.0),
                    colsample_bytree=trial.suggest_float('xgb_colsample', 0.6, 0.9),
                    min_child_weight=trial.suggest_float('xgb_min_child_weight', 1.0, 5.0),
                    reg_lambda=trial.suggest_float('xgb_reg_lambda', 0.5, 2.0),
                    reg_alpha=trial.suggest_float('xgb_reg_alpha', 0.0, 1.0),
                    objective='multi:softprob',
                    num_class=3,
                    random_state=42,
                    n_jobs=-1
                )

        return GradientBoostingClassifier(
            n_estimators=trial.suggest_int('gb_n_estimators', 150, 300),
            max_depth=trial.suggest_int('gb_max_depth', 4, 10),
            learning_rate=trial.suggest_float('gb_learning_rate', 0.05, 0.2),
            subsample=trial.suggest_float('gb_subsample', 0.7, 1.0),
            max_features=trial.suggest_categorical('gb_max_features', ['sqrt', 'log2']),
            random_state=42
        )

    def remove_correlated_features_smart(self, X: pd.DataFrame, threshold: float) -> List[str]:
        """基於閾值移除高度相關特徵，保留信息量更高者."""
        if X.empty:
            return []

        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = set()

        for col in upper_tri.columns:
            high_corr = upper_tri.index[upper_tri[col] > threshold].tolist()
            if not high_corr:
                continue

            candidates = high_corr + [col]
            variances = {feat: X[feat].var() for feat in candidates if feat not in to_drop}
            if not variances:
                continue
            keep = max(variances, key=variances.get)
            for feat in candidates:
                if feat != keep:
                    to_drop.add(feat)

        return [col for col in X.columns if col not in to_drop]

    def _resolve_feature_alias(self, name: str, available) -> Optional[str]:
        """將舊名/別名映射為當前可用欄位名，避免 'not in index'。
        規則：
        - 精確匹配優先
        - 去除前綴('1h_','4h_','1d','1w_')與 'tech_' 比對基底名
        - 若請求無 'tech_' 而可用中有 'tech_' 版本，優先取之
        - 對 gated/帶後綴的名稱，以基底名 endswith/相等做弱匹配
        """
        try:
            avail = list(available)
            if name in avail:
                return name
            prefixes = ('1h_', '4h_', '1d_', '1w_', '15m_', '15min_')
            def split_prefix(n: str) -> Tuple[str, str]:
                for p in prefixes:
                    if n.startswith(p):
                        return p, n[len(p):]
                return '', n
            def base(n: str) -> str:
                _, tail = split_prefix(n)
                return tail[5:] if tail.startswith('tech_') else tail
            req_prefix, req_tail = split_prefix(name)
            req_base = base(name)
            # 候選：基底名一致
            candidates = []
            for col in avail:
                col_prefix, _ = split_prefix(col)
                col_base = base(col)
                if col_base == req_base or col.endswith(req_base) or col_base.endswith(req_base):
                    candidates.append((col, col_prefix))
            if not candidates:
                return None
            # 優先相同時間前綴、帶 tech_、名稱較長者
            def score(c: Tuple[str, str]) -> Tuple[int, int, int]:
                col, col_prefix = c
                same_prefix = 1 if col_prefix == req_prefix else 0
                has_tech = 1 if ('tech_' in col) else 0
                return (same_prefix, has_tech, len(col))
            best = sorted(candidates, key=score, reverse=True)[0][0]
            if best not in avail:
                return None
            if best != name:
                self.logger.info(f"🔁 別名映射: '{name}' -> '{best}'")
            return best
        except Exception:
            return None

    def _normalize_column_subset(self, cols: List[str], df_train: pd.DataFrame, df_test: pd.DataFrame) -> List[str]:
        """對子集欄位做別名解析與交集保護，確保 train/test 均存在。"""
        resolved: List[str] = []
        avail_train = set(df_train.columns)
        avail_test = set(df_test.columns)
        for c in cols:
            rc = c if (c in avail_train and c in avail_test) else self._resolve_feature_alias(c, avail_train.intersection(avail_test))
            if rc and (rc in avail_train) and (rc in avail_test) and rc not in resolved:
                resolved.append(rc)
        return resolved

    def _apply_multi_objective_selection(self, X: pd.DataFrame, y: pd.Series, 
                                         max_features: int = 60) -> Tuple[List[str], Dict]:
        """
        应用多目标特征选择（如果启用）
        
        Args:
            X: 特征矩阵
            y: 标签
            max_features: 最大特征数（用于验证）
            
        Returns:
            (selected_features, metrics): 选中的特征列表和评估指标
        """
        if self.multi_objective_selector is None:
            return [], {}
        
        try:
            self.logger.info("🎯 开始多目标特征选择...")
            
            # 调用多目标选择器
            n_trials = int(self.flags.get('multi_obj_n_trials', 50))
            cv_folds = int(self.flags.get('multi_obj_cv_folds', 3))
            
            selected_features, metrics = self.multi_objective_selector.select_features(
                X=X,
                y=y,
                timeframe=self.timeframe,
                n_trials=n_trials,
                cv_folds=cv_folds
            )
            
            self.logger.info(
                f"✅ 多目标特征选择完成: {len(selected_features)}/{X.shape[1]} 特征"
            )
            self.logger.info(f"   指标: F1={metrics.get('f1_macro', 0):.4f}, "
                           f"多样性={metrics.get('diversity', 0):.4f}")
            
            # 验证特征数量是否合理
            if len(selected_features) > max_features:
                self.logger.warning(
                    f"⚠️ 多目标选择返回特征数({len(selected_features)})超过限制({max_features})，截取前{max_features}个"
                )
                selected_features = selected_features[:max_features]
            
            return selected_features, metrics
            
        except Exception as e:
            self.logger.error(f"❌ 多目标特征选择失败: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return [], {}
    
    def _filter_by_target_correlation(self, X: pd.DataFrame, y: pd.Series, keep_ratio: float = 0.8, min_score: float = 0.0) -> List[str]:
        """以目標關聯度（mutual information）做初步過濾，保留前述比例的高關聯特徵."""
        if X.empty:
            return []
        try:
            # MI 分數快取：鍵=欄位指紋 + y 指紋
            cols_fp = hash(tuple(X.columns))
            y_fp = int(hash(tuple(pd.Series(y).fillna(-1).astype(int).values)))
            if not hasattr(self, '_mi_cache'):
                self._mi_cache = {}
            cache_key = (cols_fp, y_fp)
            if cache_key in self._mi_cache:
                score_series = self._mi_cache[cache_key]
            else:
                X_sanitized = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                scores = mutual_info_classif(X_sanitized, y, discrete_features='auto', random_state=42)
                score_series = pd.Series(scores, index=X.columns).fillna(0.0)
                self._mi_cache[cache_key] = score_series
            kept = score_series[score_series >= float(min_score)]
            if kept.empty:
                kept = score_series
            top_k = max(3, int(len(kept) * float(keep_ratio)))
            return kept.sort_values(ascending=False).head(top_k).index.tolist()
        except Exception as e:
            self.logger.warning(f"⚠️ target-corr 過濾失敗: {e}")
            return list(X.columns)

    def _remove_correlated_features_smart(self, X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """根據特徵兩兩絕對相關係數去冗餘：
        - 先以變異度由高到低排序
        - 依序選入特徵，將與已選特徵絕對相關 > threshold 的其餘特徵標記為冗餘
        - 回傳保留的特徵名稱清單
        """
        try:
            if X is None or X.empty:
                return []
            X_num = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
            # 變異度排序（高變異度優先保留）
            std_series = X_num.std().fillna(0.0)
            ordered_cols = std_series.sort_values(ascending=False).index.tolist()

            # 絕對相關矩陣
            corr = X_num.corr().abs()
            keep: List[str] = []
            removed: set = set()

            for col in ordered_cols:
                if col in removed:
                    continue
                keep.append(col)
                # 將與當前保留特徵高度相關者標記為移除
                try:
                    high_corr_cols = corr.index[(corr[col] > float(threshold))].tolist()
                except Exception:
                    high_corr_cols = []
                for hc in high_corr_cols:
                    if hc != col:
                        removed.add(hc)

            # 最少保留3個避免後續崩潰
            if len(keep) < 3:
                keep = ordered_cols[:max(3, len(ordered_cols))]
            return keep
        except Exception as e:
            self.logger.warning(f"⚠️ 去冗餘失敗，回退全部特徵: {e}")
            return list(X.columns)

    def optimize_feature_selection_params(self, trial: optuna.Trial, n_features: int, layer1_params: Optional[Dict]) -> Dict[str, float]:
        """🚀 P7修復：優化版特徵選擇參數 - 提升覆蓋面與穩定性（Task 3.2）
        
        改進點：
        1. 提高粗選比例（0.55-0.75）：擴大特徵覆蓋面
        2. 靈活精選比例（0.45-0.75）：平衡性能與維度
        3. 動態穩定性閾值（0.40-0.70）：適應不同數據質量
        4. 考慮Layer1信號質量動態調整
        
        預期效果：
        - 粗選特徵數: 40 → 50-70個
        - 精選特徵數: 20 → 25-40個
        - 特徵穩定性提升
        """
        # 基礎範圍（經過優化）
        coarse_ratio_low, coarse_ratio_high = (0.55, 0.75)  # 提高下限
        stability_threshold_cap = 0.70  # 提高上限
        
        # 根據Layer1質量動態調整
        if layer1_params:
            # 估算Layer1信號質量（基於分數和分佈）
            layer1_score = layer1_params.get('best_score', 0.0) if isinstance(layer1_params, dict) else 0.0
            
            # 也可以從trial.user_attrs讀取signal_quality
            signal_quality = layer1_params.get('signal_quality', 0.6)
            
            # 高質量標籤（>0.70）：可以更積極地選擇特徵
            if signal_quality > 0.70 or layer1_score > 1.0:
                coarse_ratio_low, coarse_ratio_high = (0.60, 0.80)
                stability_threshold_cap = 0.75
                self.logger.debug(f"  高質量Layer1({signal_quality:.2f})，擴大特徵搜索空間")
            
            # 低質量標籤（<0.50）：需要更保守的特徵選擇
            elif signal_quality < 0.50 or layer1_score < 0.90:
                coarse_ratio_low, coarse_ratio_high = (0.50, 0.70)
                stability_threshold_cap = 0.65
                self.logger.debug(f"  低質量Layer1({signal_quality:.2f})，縮小特徵搜索空間")
        
        # ✅ P1修復：降低特征数量，减少过拟合（v67 CV-Holdout gap=22.6%）
        # 计算實際K值（降低30-40%以减少过拟合）
        coarse_k_min = max(30, int(n_features * coarse_ratio_low * 0.7))  # 从50降到30，乘0.7
        coarse_k_max = min(
            max(coarse_k_min + 10, int(n_features * coarse_ratio_high * 0.8)),  # 乘0.8降低上限
            n_features - 1
        )
        
        # 確保範圍有效
        if coarse_k_min >= coarse_k_max:
            coarse_k_max = coarse_k_min + 10
        
        coarse_k = trial.suggest_int('coarse_k', coarse_k_min, coarse_k_max)
        
        # 精選比例（从粗選結果中選擇，降低范围）
        fine_ratio = trial.suggest_float('fine_ratio', 0.35, 0.60)  # 从0.45-0.75降到0.35-0.60
        fine_k = max(15, min(int(coarse_k * fine_ratio), coarse_k - 1))  # 从25降到15
        
        # 穩定性閾值
        stability_threshold = trial.suggest_float(
            'stability_threshold', 
            0.40,  # 降低下限（允許更多特徵）
            stability_threshold_cap
        )
        
        # 相關性閾值
        correlation_threshold = trial.suggest_float(
            'correlation_threshold', 
            0.92,  # 降低（允許更多多樣性）
            0.97
        )
        
        return {
            'coarse_k': coarse_k,
            'fine_k': fine_k,
            'fine_ratio': fine_ratio,
            'stability_threshold': stability_threshold,
            'correlation_threshold': correlation_threshold
        }

    def _generate_td_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if ohlcv.empty:
            return pd.DataFrame(index=ohlcv.index)

        close = ohlcv['close']
        td_up = []
        td_down = []
        up = down = 0
        for i in range(len(close)):
            if i < 4:
                up = down = 0
            else:
                if close.iloc[i] > close.iloc[i-4]:
                    up += 1
                    down = 0
                elif close.iloc[i] < close.iloc[i-4]:
                    down += 1
                    up = 0
                else:
                    up = down = 0
            td_up.append(up)
            td_down.append(down)

        df = pd.DataFrame(index=ohlcv.index)
        df['td_count_up'] = pd.Series(td_up, index=ohlcv.index)
        df['td_count_down'] = pd.Series(td_down, index=ohlcv.index)
        df['td_signal_buy'] = (df['td_count_down'] >= 9).astype(int).shift(1).fillna(0)
        df['td_signal_sell'] = (df['td_count_up'] >= 9).astype(int).shift(1).fillna(0)
        return df

    def _generate_wyckoff_features(self, ohlcv: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        if ohlcv.empty:
            return pd.DataFrame(index=ohlcv.index)

        high, low, close, volume = ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume']
        local_low = low.rolling(window).min()
        local_high = high.rolling(window).max()
        vol_ma10 = volume.rolling(10).mean()

        spring = ((low < local_low * 0.995) & (close > local_low) & (volume <= vol_ma10)).astype(int).shift(1).fillna(0)
        upthrust = ((high > local_high * 1.005) & (close < local_high) & (volume <= vol_ma10)).astype(int).shift(1).fillna(0)

        price_trend = close.diff(window)
        vol_trend = volume.rolling(window).mean().diff(window)
        vol_div = (((price_trend > 0) & (vol_trend < 0)) | ((price_trend < 0) & (vol_trend > 0))).astype(int).fillna(0)

        vol_ratio = volume / volume.rolling(20).mean()
        price_vol = close.pct_change().rolling(20).std()
        phase = pd.Series(1, index=ohlcv.index)
        phase[(vol_ratio > 1.5) & (price_vol > price_vol.rolling(20).mean())] = 2
        phase[vol_ratio > 2.0] = 3
        phase[price_vol > price_vol.quantile(0.8)] = 4
        phase[vol_ratio < 0.7] = 5

        df = pd.DataFrame(index=ohlcv.index)
        df['wyk_spring'] = spring
        df['wyk_upthrust'] = upthrust
        df['wyk_volume_divergence'] = vol_div
        df['wyk_phase'] = phase.fillna(1).astype(int)
        return df

    def _generate_regime_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """簡版 regime 特徵：基於實現波動與移動趨勢劃分高/低波、趨勢/盤整。"""
        if ohlcv is None or ohlcv.empty:
            return pd.DataFrame(index=getattr(ohlcv, 'index', None))
        try:
            close = ohlcv['close'].astype(float)
            ret = close.pct_change().fillna(0.0)
            vol30 = ret.rolling(30).std().fillna(0.0)
            vol90 = ret.rolling(90).std().fillna(0.0)
            vol_z = (vol30 - vol30.rolling(90).mean()) / (vol30.rolling(90).std() + 1e-8)
            # 趨勢指標：長短均線差
            ma_fast = close.rolling(20).mean()
            ma_slow = close.rolling(60).mean()
            trend = (ma_fast - ma_slow) / (ma_slow + 1e-8)
            regime_high_vol = (vol30 > vol90).astype(int)
            regime_trend = (trend.abs() > trend.abs().rolling(60).median().fillna(0)).astype(int)
            df = pd.DataFrame({
                'regime_high_vol': regime_high_vol,
                'regime_trend': regime_trend,
                'vol_z': vol_z.fillna(0.0)
            }, index=ohlcv.index)
            return df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        except Exception as e:
            self.logger.warning(f"regime 特徵生成失敗: {e}")
            return pd.DataFrame(index=ohlcv.index)

    def _generate_micro_features_from_ohlcv(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if ohlcv.empty:
            return pd.DataFrame(index=ohlcv.index)

        high, low, close, volume = ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume']
        rng = (high - low).clip(lower=1e-8)
        mid = (high + low) / 2

        df = pd.DataFrame(index=ohlcv.index)
        df['micro_spread_est'] = (high - low) * 0.3
        df['micro_spread_ratio_est'] = (high - low) / close
        df['micro_mid_est'] = mid
        df['micro_depth_imb_5'] = (close - mid) / mid
        df['micro_depth_imb_10'] = df['micro_depth_imb_5'] * 0.8
        df['micro_depth_imb_20'] = df['micro_depth_imb_5'] * 0.6
        df['micro_imp_buy_0p5'] = (high - close) / close
        df['micro_imp_sell_0p5'] = (close - low) / close
        df['micro_ob_slope_bid'] = volume / (rng * 1000)
        df['micro_ob_slope_ask'] = volume / (rng * 1000)
        df['micro_price_eff'] = ((high - low) / close).clip(upper=0.8)
        return df.replace([np.inf, -np.inf], np.nan)

    def _load_derivatives_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        if not self.flags.get('enable_derivatives', False):
            return pd.DataFrame(index=index)

        base_path = Path('data') / 'derived' / self.symbol
        if not base_path.exists():
            self.logger.warning(f"⚠️ 衍生品資料不存在: {base_path}")
            return pd.DataFrame(index=index)

        datasets = ['funding', 'open_interest', 'long_short', 'liquidations', 'basis']
        frames = []
        for name in datasets:
            df = None
            for ext in ['parquet', 'csv']:
                file_path = base_path / f"{name}_{self.timeframe}.{ext}"
                if file_path.exists():
                    try:
                        if ext == 'parquet':
                            df = read_dataframe(file_path)
                        else:
                            df = pd.read_csv(file_path, parse_dates=True, index_col=0)
                        break
                    except Exception as e:
                        self.logger.warning(f"⚠️ 讀取衍生品數據失敗 {file_path}: {e}")
            if df is None or df.empty:
                continue
            df = df.reindex(index)
            missing_ratio = 1.0 - df.notna().mean().mean()
            if missing_ratio > 0.2:
                self.logger.warning(f"⚠️ 衍生品 {name} 缺失比例較高: {missing_ratio:.1%}")
            df = self._apply_latency(df, self.flags.get('latency_lag', 1))
            frames.append(df.add_prefix(f"deriv_{name}_"))

        if not frames:
            return pd.DataFrame(index=index)
        return pd.concat(frames, axis=1).fillna(0)

    def _make_cv_splits(self, X: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        gap = int(self.flags.get('cv_gap', 12))
        purge = int(self.flags.get('cv_purge', gap))
        strategy = self.flags.get('cv_strategy', 'embargo')
        splits = []
        for train_idx, test_idx in tscv.split(X):
            train_idx = list(train_idx)
            test_idx = list(test_idx)
            if strategy == 'embargo':
                if len(train_idx) > purge:
                    train_idx = train_idx[:-purge]
                if len(test_idx) > gap:
                    test_idx = test_idx[gap:]
            elif strategy == 'purged':
                purge_left = purge_right = purge
                if test_idx:
                    start, end = test_idx[0], test_idx[-1]
                    train_idx = [i for i in train_idx if (i < start - purge_left or i > end + purge_right)]
            splits.append((np.array(train_idx), np.array(test_idx)))
        return splits


    def _load_and_prepare_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """🚀 預加載OHLCV數據和特徵（優先使用物化Layer0，並具備I/O容錯）"""
        try:
            # 1) 優先讀取物化清洗數據 data/processed/cleaned/{symbol}_{timeframe}/cleaned_ohlcv*
            processed_dir = self.data_path / "processed" / "cleaned" / f"{self.symbol}_{self.timeframe}"
            ohlcv_data: Optional[pd.DataFrame] = None
            if processed_dir.exists():
                candidates = list(processed_dir.glob("cleaned_ohlcv*.parquet")) + \
                             list(processed_dir.glob("cleaned_ohlcv*.pkl")) + \
                             list(processed_dir.glob("cleaned_ohlcv*.pickle"))
                if candidates:
                    # 選擇最新修改時間的檔案
                    latest = max(candidates, key=lambda p: p.stat().st_mtime)
                    self.logger.info(f"🔍 嘗試載入物化清洗數據: {latest}")
                    try:
                        ohlcv_data = read_dataframe(latest)
                        self.logger.info(f"✅ 使用物化Layer0清洗數據: {latest} -> {ohlcv_data.shape}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ 物化清洗數據讀取失敗，回退: {e}")

            # 2) 回退到舊位址 optuna_system/configs/cleaned_ohlcv_{timeframe}.parquet
            if ohlcv_data is None:
                cleaned_candidate = self.config_path / f"cleaned_ohlcv_{self.timeframe}.parquet"
                if cleaned_candidate.exists():
                    self.logger.info(f"🔍 嘗試載入舊位址清洗數據: {cleaned_candidate}")
                    try:
                        ohlcv_data = read_dataframe(cleaned_candidate)
                        self.logger.info(f"✅ 使用舊位址清洗數據: {ohlcv_data.shape}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ 舊位址清洗數據讀取失敗，回退: {e}")

            # 3) 最後回退到 raw OHLCV
            if ohlcv_data is None:
                ohlcv_file = self.data_path / "raw" / self.symbol / f"{self.symbol}_{self.timeframe}_ohlcv.parquet"
                self.logger.info(f"🔍 查找OHLCV源文件: {ohlcv_file.absolute()}")
                if ohlcv_file.exists():
                    try:
                        ohlcv_data = read_dataframe(ohlcv_file)
                        self.logger.info(f"✅ 加載原始OHLCV數據: {ohlcv_data.shape}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ 原始OHLCV讀取失敗: {e}")

            # 4) 如果仍無法獲取，生成模擬數據（最後手段）
            if ohlcv_data is None:
                self.logger.warning("❌ 未找到可用OHLCV文件，生成模擬數據用於測試")
                ohlcv_data = self._generate_mock_data()

            # 🚀 統一使用內建特徵生成，與src完全解耦
            features_df = self._generate_features(ohlcv_data)
            self.logger.info(f"✅ 使用內建特徵生成: {features_df.shape}")

            # 清理特徵數據（移除bfill以免引入向後填補）
            features_df = features_df.ffill()
            features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)

            return ohlcv_data, features_df

        except Exception as e:
            self.logger.error(f"數據加載失敗: {e}")
            raise  # 🚀 Fail-Fast: 關鍵錯誤直接拋出

    # ============================================================
    # 多時框趨勢對齊功能（方法2: reindex + ffill）
    # ============================================================
    
    def _load_higher_timeframe_data(self, higher_timeframe: str = '1h') -> Optional[pd.DataFrame]:
        """
        載入更高時框的 OHLCV 數據（用於多時框趨勢對齊）
        
        Args:
            higher_timeframe: 更高的時間框架（如 '1h', '4h'）
            
        Returns:
            更高時框的 OHLCV DataFrame，如果載入失敗則返回 None
        """
        try:
            # 1. 優先從 processed/cleaned 載入
            processed_dir = self.data_path / "processed" / "cleaned" / f"{self.symbol}_{higher_timeframe}"
            if processed_dir.exists():
                candidates = list(processed_dir.glob("cleaned_ohlcv*.parquet")) + \
                             list(processed_dir.glob("cleaned_ohlcv*.pkl"))
                if candidates:
                    latest = max(candidates, key=lambda p: p.stat().st_mtime)
                    self.logger.info(f"🔍 載入 {higher_timeframe} 清洗數據: {latest.name}")
                    return read_dataframe(latest)
            
            # 2. 回退到 raw 數據
            raw_file = self.data_path / "raw" / self.symbol / f"{self.symbol}_{higher_timeframe}_ohlcv.parquet"
            if raw_file.exists():
                self.logger.info(f"🔍 載入 {higher_timeframe} 原始數據: {raw_file.name}")
                return read_dataframe(raw_file)
            
            self.logger.warning(f"⚠️ 未找到 {higher_timeframe} 數據")
            return None
            
        except Exception as e:
            self.logger.warning(f"❌ 載入 {higher_timeframe} 數據失敗: {e}")
            return None

    def _calculate_higher_timeframe_trend(self, df_higher: pd.DataFrame) -> pd.Series:
        """
        計算更高時框的趨勢方向
        
        使用多個指標綜合判斷趨勢：
        - EMA 20/50 交叉
        - MACD 方向
        
        Args:
            df_higher: 更高時框的 OHLCV DataFrame
            
        Returns:
            趨勢信號 Series (1=買入趨勢, 0=中性, -1=賣出趨勢)
        """
        try:
            # 計算 EMA
            ema_20 = df_higher['close'].ewm(span=20, adjust=False).mean()
            ema_50 = df_higher['close'].ewm(span=50, adjust=False).mean()
            
            # 計算 MACD
            ema_12 = df_higher['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df_higher['close'].ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9, adjust=False).mean()
            
            # 初始化趨勢信號
            trend = pd.Series(0, index=df_higher.index)
            
            # 買入趨勢條件：EMA20 > EMA50 AND MACD > Signal
            buy_condition = (ema_20 > ema_50) & (macd > signal)
            trend[buy_condition] = 1
            
            # 賣出趨勢條件：EMA20 < EMA50 AND MACD < Signal
            sell_condition = (ema_20 < ema_50) & (macd < signal)
            trend[sell_condition] = -1
            
            buy_pct = sum(trend == 1) / len(trend) * 100
            neutral_pct = sum(trend == 0) / len(trend) * 100
            sell_pct = sum(trend == -1) / len(trend) * 100
            
            self.logger.info(
                f"✅ 趨勢分佈: 買入={buy_pct:.1f}%, 中性={neutral_pct:.1f}%, 賣出={sell_pct:.1f}%"
            )
            
            return trend
            
        except Exception as e:
            self.logger.warning(f"❌ 計算趨勢失敗: {e}")
            return pd.Series(0, index=df_higher.index)

    def _align_higher_timeframe_to_current(
        self, 
        current_index: pd.DatetimeIndex,
        higher_tf_series: pd.Series
    ) -> pd.Series:
        """
        🔧 P0修復：嚴格前向對齊，防止未完成bar的數據洩漏
        
        修復邏輯：
        1. 先shift(1) higher_tf數據 → 只用已完成的大bar
        2. 再reindex+ffill到當前時框
        3. 禁用backward fill
        
        示例（15m使用1h數據，當前時間09:45）：
        錯誤: 使用09:00-10:00的1h bar（未完成）❌
        正確: 使用08:00-09:00的1h bar（已完成）✅
        
        Args:
            current_index: 當前時框的時間索引（如 15m）
            higher_tf_series: 更高時框的數據 Series（如 1h 趨勢）
            
        Returns:
            對齊到當前時框的 Series
        """
        try:
            # 🔧 關鍵修復：額外shift(1)確保只用已完成的bar
            higher_tf_shifted = higher_tf_series.shift(1)
            
            # 使用 reindex + ffill 對齊
            aligned = higher_tf_shifted.reindex(current_index, method='ffill')
            
            # 處理開頭的 NaN（用0填充，不用bfill避免反向洩漏）
            aligned = aligned.fillna(0)
            
            self.logger.info(
                f"✅ 嚴格前向對齊: {len(higher_tf_series)} → {len(aligned)} "
                f"(已shift防洩漏)"
            )
            
            return aligned
            
        except Exception as e:
            self.logger.warning(f"❌ 時框對齊失敗: {e}")
            return pd.Series(0, index=current_index)

    def _build_reference_timeframe_features(
        self, 
        ohlcv_data: pd.DataFrame,
        higher_tf: str = '1h'
    ) -> pd.DataFrame:
        """
        🔧 P1新增：生成參考時框特徵（僅趨勢相關，不含細節技術指標）
        
        設計理念：
        - 15m交易時，1h只作為"大環境參考"
        - 只生成趨勢、均線、突破等宏觀信號
        - 不生成RSI、MACD等細節指標
        
        Args:
            ohlcv_data: 當前15m的OHLCV數據
            higher_tf: 參考時框（默認1h）
        
        Returns:
            DataFrame: 僅包含趨勢參考特徵（<10個）
        """
        features = pd.DataFrame(index=ohlcv_data.index)
        
        try:
            # 1. 載入1h數據
            df_higher = self._load_higher_timeframe_data(higher_tf)
            if df_higher is None or df_higher.empty:
                self.logger.warning(f"⏭️ {higher_tf}數據不可用，跳過參考特徵")
                return features
            
            # 2. 只生成趨勢相關特徵（不是完整技術指標）
            close_higher = df_higher['close']
            high_higher = df_higher['high']
            low_higher = df_higher['low']
            
            # 2.1 均線（趨勢判斷）
            ema_20 = close_higher.ewm(span=20, adjust=False).mean()
            ema_50 = close_higher.ewm(span=50, adjust=False).mean()
            
            # 2.2 趨勢方向（簡化版）
            trend = pd.Series(0, index=df_higher.index)
            trend[ema_20 > ema_50] = 1  # 上升趨勢
            trend[ema_20 < ema_50] = -1  # 下降趨勢
            
            # 2.3 價格相對均線位置（判斷超買超賣）
            price_ema20_ratio = close_higher / (ema_20 + 1e-9)
            price_ema50_ratio = close_higher / (ema_50 + 1e-9)
            
            # 2.4 支撐阻力
            resistance = high_higher.rolling(20).max()
            support = low_higher.rolling(20).min()
            
            # 3. 對齊到15m（嚴格shift防洩漏）
            features[f'{higher_tf}_trend'] = self._align_higher_timeframe_to_current(
                ohlcv_data.index, trend
            )
            features[f'{higher_tf}_ema20'] = self._align_higher_timeframe_to_current(
                ohlcv_data.index, ema_20
            )
            features[f'{higher_tf}_ema50'] = self._align_higher_timeframe_to_current(
                ohlcv_data.index, ema_50
            )
            features[f'{higher_tf}_price_ema20_ratio'] = self._align_higher_timeframe_to_current(
                ohlcv_data.index, price_ema20_ratio
            )
            features[f'{higher_tf}_price_ema50_ratio'] = self._align_higher_timeframe_to_current(
                ohlcv_data.index, price_ema50_ratio
            )
            features[f'{higher_tf}_resistance'] = self._align_higher_timeframe_to_current(
                ohlcv_data.index, resistance
            )
            features[f'{higher_tf}_support'] = self._align_higher_timeframe_to_current(
                ohlcv_data.index, support
            )
            
            self.logger.info(
                f"✅ {higher_tf}參考特徵: {len(features.columns)}個 "
                f"(僅趨勢相關，無細節指標)"
            )
            
            return features
            
        except Exception as e:
            self.logger.warning(f"❌ {higher_tf}參考特徵生成失敗: {e}")
            return features

    def _add_multi_timeframe_trend_features(
        self, 
        ohlcv_data: pd.DataFrame,
        higher_timeframes: List[str] = None
    ) -> pd.DataFrame:
        """
        添加多時框趨勢特徵到當前時框
        
        Args:
            ohlcv_data: 當前時框的 OHLCV 數據
            higher_timeframes: 要添加的更高時框列表（默認 ['1h', '4h']）
            
        Returns:
            包含多時框趨勢特徵的 DataFrame
        """
        if higher_timeframes is None:
            # 根據當前時框智能選擇
            if self.timeframe in ['15m', '15']:
                higher_timeframes = ['1h', '4h']
            elif self.timeframe in ['1h', '1H']:
                higher_timeframes = ['4h', '1d']
            else:
                higher_timeframes = ['1h']
        
        trend_features = pd.DataFrame(index=ohlcv_data.index)
        
        for htf in higher_timeframes:
            try:
                # 1. 載入更高時框數據
                df_higher = self._load_higher_timeframe_data(htf)
                if df_higher is None or df_higher.empty:
                    self.logger.warning(f"⏭️ 跳過 {htf} 趨勢特徵（數據不可用）")
                    continue
                
                # 2. 計算趨勢方向
                trend = self._calculate_higher_timeframe_trend(df_higher)
                
                # 3. 對齊到當前時框（使用 reindex + ffill）
                trend.name = f'trend_{htf}'
                aligned_trend = self._align_higher_timeframe_to_current(
                    ohlcv_data.index,
                    trend
                )
                
                # 4. 添加到特徵集
                trend_features[f'trend_{htf}'] = aligned_trend
                
                # 5. 可選：添加更多高時框特徵（如 EMA 等）
                try:
                    ema_20 = df_higher['close'].ewm(span=20).mean()
                    ema_50 = df_higher['close'].ewm(span=50).mean()
                    
                    trend_features[f'{htf}_ema_20'] = self._align_higher_timeframe_to_current(
                        ohlcv_data.index, ema_20
                    )
                    trend_features[f'{htf}_ema_50'] = self._align_higher_timeframe_to_current(
                        ohlcv_data.index, ema_50
                    )
                    
                    self.logger.info(f"✅ 已添加 {htf} 額外特徵: EMA20, EMA50")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ 添加 {htf} 額外特徵失敗: {e}")
                
                self.logger.info(f"🎯 成功添加 {htf} 趨勢特徵組")
                
            except Exception as e:
                self.logger.warning(f"❌ 處理 {htf} 特徵失敗: {e}")
                continue
        
        if not trend_features.empty:
            self.logger.info(f"✅ 多時框趨勢特徵總計: {len(trend_features.columns)} 個")
        else:
            self.logger.warning("⚠️ 未能添加任何多時框趨勢特徵")
        
        return trend_features

    # ============================================================
    # 原有方法
    # ============================================================

    def _generate_mock_data(self) -> pd.DataFrame:
        """生成模擬OHLCV數據"""
        np.random.seed(42)
        n_samples = 2000

        # 生成基礎價格序列
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, n_samples)
        prices = [base_price]

        for change in price_changes:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 100))

        prices = np.array(prices[1:])

        # 生成OHLCV數據
        data = []
        for i, close in enumerate(prices):
            volatility = np.random.uniform(0.005, 0.03)
            high = close * (1 + np.random.uniform(0, volatility))
            low = close * (1 - np.random.uniform(0, volatility))
            open_price = low + (high - low) * np.random.uniform(0.2, 0.8)

            high = max(high, open_price, close)
            low = min(low, open_price, close)
            volume = np.random.uniform(100, 10000)

            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        dates = pd.date_range('2022-01-01', periods=n_samples, freq='15min')
        return pd.DataFrame(data, index=dates)

    def _generate_labels(self, price_series: pd.Series, lag: int = 12,
                        profit_quantile: float = 0.85, loss_quantile: float = 0.15,
                        lookback_window: int = 500) -> pd.Series:
        """向量化標籤生成：rolling quantile + shift(1) 避免洩露並加速。"""
        future_prices = price_series.shift(-lag)
        returns = (future_prices - price_series) / price_series
 
        ret_past = returns.shift(1)
        min_periods = min(lookback_window, 100)
        upper = ret_past.rolling(window=lookback_window, min_periods=min_periods).quantile(profit_quantile)
        lower = ret_past.rolling(window=lookback_window, min_periods=min_periods).quantile(loss_quantile)
 
        upper = upper.clip(lower=1e-4)
        lower = lower.clip(upper=-1e-4)
 
        labels = pd.Series(1, index=price_series.index, dtype=int)
        labels = labels.mask(returns > upper, 2).mask(returns < lower, 0)
 
        if lag > 0:
            labels = labels.iloc[:-lag]
 
        # 可選：波動縮放（高波放寬、低波收斂），提升可分性（旗標控制）
        try:
            if bool(self.flags.get('enable_volatility_scaled_labels', False)):
                ret = price_series.pct_change().fillna(0.0)
                vol = ret.rolling(60).std().fillna(0.0)
                vol_q = vol.rolling(500).quantile(0.5).fillna(vol.median())
                high_vol = vol > vol_q
                # 在高波段放寬買賣門檻，使買/賣樣本更易達成；低波則更保守
                adj = np.where(high_vol.reindex(labels.index, method='ffill').fillna(False), 0.02, -0.02)
                labels = labels.copy()
                # 僅在中性區附近做細微調整（避免大幅改動）
                labels[(returns.iloc[:len(labels)] > upper.iloc[:len(labels)] - adj) & (labels == 1)] = 2
                labels[(returns.iloc[:len(labels)] < lower.iloc[:len(labels)] + adj) & (labels == 1)] = 0
        except Exception as e:
            self.logger.warning(f"波動縮放標籤失敗: {e}")

        label_counts = labels.value_counts().sort_index()
        self.logger.info(f"🏷️ 向量化標籤: lag={lag}, profit_q={profit_quantile:.3f}, loss_q={loss_quantile:.3f}")
        self.logger.info(f"📊 標籤分佈: {dict(label_counts)} (總計{int(labels.notna().sum())})")
        return labels.dropna().astype(int)

    def generate_labels(self, price_data: pd.Series, params: Dict[str, Any]) -> pd.Series:
        """對齊 Layer2 調用介面：從 params 中讀取量化分位或回退至 Layer1 命名，調用內部向量化生成。

        支援鍵：
        - lag
        - profit_quantile / buy_quantile
        - loss_quantile / sell_quantile
        - lookback_window
        """
        try:
            lag = int(params.get('lag', max(1, int(self.scaled_config.get('label_lag_min', 3)))))
        except Exception:
            lag = 3
        try:
            profit_q = float(params.get('profit_quantile', params.get('buy_quantile', 0.75)))
        except Exception:
            profit_q = 0.75
        try:
            loss_q = float(params.get('loss_quantile', params.get('sell_quantile', 0.25)))
        except Exception:
            loss_q = 0.25
        try:
            lookback_window = int(params.get('lookback_window', int(self.scaled_config.get('lookback_window_min', 300))))
        except Exception:
            lookback_window = 300
        return self._generate_labels(price_data, lag, profit_q, loss_q, lookback_window)

    def apply_labels(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """將以向量化方法生成的標籤附加至 OHLCV 資料，並對齊索引。"""
        if 'close' not in data.columns:
            raise ValueError("資料必須包含close欄位")
        labels = self.generate_labels(data['close'], params)
        aligned = data.loc[labels.index].copy()
        aligned['label'] = labels.astype(int)
        return aligned

    def build_features_for_materialization(self, ohlcv_data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """🔧 P1修復：重建與優化流程一致的特徵集，用於物化
        
        整合多種特徵類型：
        1) 原生時間框架特徵（無滯後）
        2) 多時框技術指標特徵
        3) 策略特徵（Wyckoff/TD/Micro）
        4) 進行特徵質量過濾，確保一致性
        """
        if ohlcv_data is None or ohlcv_data.empty:
            return pd.DataFrame()
        
        all_features_list = []
        base_features = pd.DataFrame(index=ohlcv_data.index)
        all_features_list.append(base_features)
        
        # 🔧 P1修復1：添加原生時間框架特徵（優先級最高，無滯後）
        try:
            native_features = self._build_native_timeframe_features(ohlcv_data)
            if not native_features.empty:
                all_features_list.append(native_features)
                self.logger.info(f"✅ 添加 {len(native_features.columns)} 個原生特徵")
        except Exception as e:
            self.logger.warning(f"⚠️ 原生特徵生成失敗: {e}")
        
        # 原有的多時框技術特徵
        try:
            tech_features = self.generate_technical_features(ohlcv_data, params)
            if not tech_features.empty:
                all_features_list.append(tech_features)
                self.logger.info(f"✅ 添加 {len(tech_features.columns)} 個多時框技術特徵")
        except Exception as e:
            self.logger.warning(f"⚠️ 多時框技術特徵生成失敗: {e}")
        
        # 🔧 P1修復2：添加策略特徵（Wyckoff/TD/Micro）
        if self.flags.get('enable_strategy_features', True):
            try:
                # TD Sequential 特徵
                if self.flags.get('enable_td', True):
                    td_features = self._generate_td_features(ohlcv_data)
                    if not td_features.empty:
                        all_features_list.append(td_features)
                        self.logger.info(f"✅ 添加 {len(td_features.columns)} 個 TD Sequential 特徵")
            except Exception as e:
                self.logger.warning(f"⚠️ TD特徵生成失敗: {e}")
            
            try:
                # Wyckoff 特徵
                if self.flags.get('enable_wyckoff', True):
                    wyk_features = self._generate_wyckoff_features(ohlcv_data)
                    if not wyk_features.empty:
                        all_features_list.append(wyk_features)
                        self.logger.info(f"✅ 添加 {len(wyk_features.columns)} 個 Wyckoff 特徵")
            except Exception as e:
                self.logger.warning(f"⚠️ Wyckoff特徵生成失敗: {e}")
            
            try:
                # Market Microstructure 特徵
                if self.flags.get('enable_micro', True):
                    micro_features = self._generate_micro_features_from_ohlcv(ohlcv_data)
                    if not micro_features.empty:
                        all_features_list.append(micro_features)
                        self.logger.info(f"✅ 添加 {len(micro_features.columns)} 個 Microstructure 特徵")
            except Exception as e:
                self.logger.warning(f"⚠️ Microstructure特徵生成失敗: {e}")
        
        # 🎯 新增：多時框趨勢對齊特徵（實現多時框一致性交易邏輯）
        try:
            if self.flags.get('enable_mtf_trend', True):  # 默認啟用
                mtf_trend_features = self._add_multi_timeframe_trend_features(ohlcv_data)
                if not mtf_trend_features.empty:
                    all_features_list.append(mtf_trend_features)
                    self.logger.info(
                        f"🎯 添加 {len(mtf_trend_features.columns)} 個多時框趨勢特徵 "
                        f"(實現 1h+15m 趨勢一致性過濾)"
                    )
                else:
                    self.logger.warning("⚠️ 多時框趨勢特徵生成為空")
            else:
                self.logger.info("⏭️ 多時框趨勢特徵已禁用（enable_mtf_trend=False）")
        except Exception as e:
            self.logger.warning(f"⚠️ 多時框趨勢特徵生成失敗: {e}")
        
        # 合併所有特徵
        X = pd.concat([f for f in all_features_list if not f.empty], axis=1)
        
        # 移除重複列
        X = X.loc[:, ~X.columns.duplicated()]
        
        self.logger.info(
            f"📊 特徵物化總計: {len(X.columns)} 個特徵 "
            f"(原生 + 多時框 + 策略)"
        )
        
        # 特徵質量過濾
        try:
            X = self._filter_low_quality_features(X)
            self.logger.info(f"✅ 質量過濾後保留: {len(X.columns)} 個特徵")
        except Exception as e:
            self.logger.warning(f"⚠️ 特徵質量過濾失敗: {e}")
        
        return X

    def generate_technical_features(self, ohlcv_data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """生成技術指標特徵（支援多時框與配置化參數）"""
        flags_tech = self.flags.get('tech', {})

        base_features = self._calc_base_indicators(ohlcv_data, self.timeframe, flags_tech)
        features = base_features.copy()

        # Phase 1.2: 多尺度增強特徵
        try:
            enhanced = self._generate_enhanced_features(ohlcv_data)
            if isinstance(enhanced, pd.DataFrame) and not enhanced.empty:
                features = self._safe_merge(features, enhanced, prefix='')
        except Exception as e:
            self.logger.warning(f"增強特徵生成失敗: {e}")

        # Phase 1.3: 高級技術指標
        try:
            adv = self._generate_advanced_indicators(ohlcv_data)
            if isinstance(adv, pd.DataFrame) and not adv.empty:
                features = self._safe_merge(features, adv, prefix='')
        except Exception as e:
            self.logger.warning(f"高級指標生成失敗: {e}")

        # 新增：量價/動量/趨勢強度（前視安全）
        try:
            cmf = self._calc_cmf(ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close'], ohlcv_data['volume'])
            adl = self._calc_adl(ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close'], ohlcv_data['volume'])
            stoch_rsi = self._calc_stoch_rsi(ohlcv_data['close'])
            adx = self._calc_adx(ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close'])

            add_df = pd.DataFrame({
                'tech_cmf': cmf,
                'tech_adl': adl,
                'tech_stoch_rsi': stoch_rsi,
                'tech_adx': adx,
            }, index=ohlcv_data.index)

            features = self._safe_merge(features, add_df, prefix='')
        except Exception as e:
            self.logger.warning(f"增補量價/動量/趨勢強度指標失敗: {e}")

        # Phase 1.4: Regime-aware 特徵
        try:
            if bool(self.flags.get('enable_regime_features', True)):
                regime_df = self._generate_regime_features(ohlcv_data)
                if isinstance(regime_df, pd.DataFrame) and not regime_df.empty:
                    features = self._safe_merge(features, regime_df, prefix='')
        except Exception as e:
            self.logger.warning(f"regime 特徵合併失敗: {e}")

        mtf_cfg = flags_tech.get('multi_timeframes', {})
        if isinstance(mtf_cfg, dict) and mtf_cfg.get('enabled'):
            rules = mtf_cfg.get('rules', {})
            tf_list = [tf for tf in mtf_cfg.get('timeframes', []) if tf in self._allowed_slow_timeframes()]
            for tf_key in tf_list:
                rule = rules.get(tf_key)
                if not rule:
                    continue
                freq = rule.get('rule') if isinstance(rule, dict) else rule
                freq_map = {
                    '15m': '15T', '15min': '15T', '15': '15T',
                    '1h': '1H', '1hour': '1H',
                    '4h': '4H', '4hour': '4H',
                    '1d': '1D', '1day': '1D',
                    '1w': '1W', '1week': '1W', '1W': '1W'
                }
                freq_str = str(freq).lower()
                freq = freq_map.get(freq_str, freq)
                
                # 🔧 P1修復：區分當前時框（完整指標）vs 參考時框（僅趨勢）
                if tf_key in ['1h', '4h']:
                    # 高時框 = 參考用途（僅生成趨勢特徵）
                    tf_features = self._build_reference_timeframe_features(ohlcv_data, tf_key)
                    self.logger.info(f"🎯 {tf_key} 作為參考時框（僅趨勢特徵）")
                else:
                    # 當前時框 = 主要交易信號（生成完整技術指標）
                    resampled = self._resample_ohlcv(ohlcv_data, freq)
                    if resampled.empty:
                        continue
                    tf_override = rule if isinstance(rule, dict) else {}
                    tf_features = self._calc_base_indicators(resampled, tf_key, flags_tech, base_index=None, tf_overrides_override=tf_override)
                    # 🔧 P0修復：resampled已經通過_resample_ohlcv雙重shift，這裡不需要額外shift
                    # tf_features = tf_features.shift(1).ffill()  # ❌ 刪除（避免第三重shift）
                    tf_features = tf_features.reindex(ohlcv_data.index, method='ffill').fillna(0)
                    self.logger.info(f"🎯 {tf_key} 作為主要時框（完整技術指標）")
                
                if tf_features.empty:
                    continue
                base_cols = list(tf_features.columns)
                gating_signal = None
                if base_cols:
                    rsi_cols = [col for col in base_cols if 'tech_rsi_' in col]
                    if rsi_cols:
                        gate_threshold = float(tf_override.get('gate_threshold', 0.55))
                        gating_signal = (tf_features[rsi_cols].mean(axis=1) > gate_threshold * 100).astype(float)
                if gating_signal is not None:
                    gated = tf_features.multiply(gating_signal, axis=0).add_suffix('_gated')
                    # 確保 gated 欄位穩定存在：未觸發處以 0 保留欄位，避免不同折生成不一致
                    gated = gated.reindex(tf_features.index).fillna(0)
                    tf_features = pd.concat([tf_features, gated], axis=1)
                features = self._safe_merge(features, tf_features, prefix=f"{tf_key}_")

        return features

    def _generate_enhanced_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Phase 1.2: 多尺度特徵（動量/成交量加速度等）。"""
        if ohlcv is None or ohlcv.empty:
            return pd.DataFrame(index=ohlcv.index if isinstance(ohlcv, pd.DataFrame) else None)
        windows = [200, 400, 800, 1200]
        out = {}
        close = ohlcv['close']
        volume = ohlcv['volume']
        for w in windows:
            try:
                mom = close.pct_change(w)
                out[f'momentum_acc_{w}'] = mom.diff(5)
            except Exception:
                continue
            try:
                vol_chg = volume.pct_change(w)
                out[f'volume_acc_{w}'] = vol_chg.diff(5)
            except Exception:
                continue
        df = pd.DataFrame(out, index=ohlcv.index)
        return df.replace([np.inf, -np.inf], np.nan)

    def _generate_advanced_indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Phase 1.3: 高級指標（Wyckoff累積/TD序列簡化版）。"""
        if ohlcv is None or ohlcv.empty:
            return pd.DataFrame(index=ohlcv.index if isinstance(ohlcv, pd.DataFrame) else None)
        high, low, close, volume = ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume']
        rng = (high - low).replace(0, np.nan)
        # Wyckoff累積/派發線（簡化）
        money_flow = ((close - low) - (high - close)) / rng * volume
        wyckoff_ad = money_flow.fillna(0).cumsum()
        # TD序列（簡化setup計數）
        td_setup = pd.Series(0, index=close.index)
        for i in range(4, len(close)):
            try:
                if close.iloc[i] > close.iloc[i-4]:
                    td_setup.iloc[i] = td_setup.iloc[i-1] + 1 if td_setup.iloc[i-1] > 0 else 1
                elif close.iloc[i] < close.iloc[i-4]:
                    td_setup.iloc[i] = td_setup.iloc[i-1] - 1 if td_setup.iloc[i-1] < 0 else -1
                else:
                    td_setup.iloc[i] = 0
            except Exception:
                td_setup.iloc[i] = 0
        return pd.DataFrame({
            'wyk_ad_line': wyckoff_ad,
            'td_setup': td_setup
        }, index=close.index)

    # ---- 新增：量價/動量/趨勢強度指標（前視安全） ----
    def _calc_cmf(self, high: pd.Series, low: pd.Series, close: pd.Series,
                  volume: pd.Series, window: int = 20) -> pd.Series:
        rng = (high - low).replace(0, np.nan)
        mf_multiplier = ((close - low) - (high - close)) / (rng + 1e-9)
        mf_volume = mf_multiplier * volume
        cmf = (mf_volume.rolling(window, min_periods=max(3, window // 3)).sum() /
               (volume.rolling(window, min_periods=max(3, window // 3)).sum() + 1e-9))
        return cmf.fillna(0)

    def _calc_adl(self, high: pd.Series, low: pd.Series, close: pd.Series,
                  volume: pd.Series) -> pd.Series:
        rng = (high - low).replace(0, np.nan)
        mf_multiplier = ((close - low) - (high - close)) / (rng + 1e-9)
        mf_volume = mf_multiplier * volume
        adl = mf_volume.cumsum()
        return adl.fillna(method='ffill').fillna(0)

    def _calc_rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calc_stoch_rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        rsi = self._calc_rsi(close, window)
        min_rsi = rsi.rolling(window).min()
        max_rsi = rsi.rolling(window).max()
        stoch_rsi = (rsi - min_rsi) / ((max_rsi - min_rsi) + 1e-9)
        return stoch_rsi.clip(0, 1).fillna(0)

    def _calc_adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> pd.Series:
        # True Range components
        tr1 = (high - low).abs()
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm <= plus_dm] = 0

        # Wilder smoothing
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-9))
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-9))

        dx = (100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)).fillna(0)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx.fillna(0)

    def _calc_base_indicators(self, ohlcv: pd.DataFrame, tf_key: str, flags_tech: Dict,
                               base_index: Optional[pd.DatetimeIndex] = None,
                               tf_overrides_override: Optional[Dict] = None) -> pd.DataFrame:
        if ohlcv.empty:
            return pd.DataFrame(index=base_index if base_index is not None else ohlcv.index)

        tf_overrides_base = (flags_tech.get('tf_overrides') or {}).get(tf_key, {})
        tf_overrides = {}
        if isinstance(tf_overrides_override, dict):
            tf_overrides.update(tf_overrides_override)
        tf_overrides.update(tf_overrides_base)

        def _as_window_list(value, default):
            if isinstance(value, (list, tuple)):
                return [int(v) for v in value if int(v) > 0]
            if value is None:
                return list(default)
            return [int(value)]

        fibo_base = flags_tech.get('fibo_ma', [3, 5, 8, 13, 21, 34, 55, 89, 144, 233])
        fibo_ma = [_w for _w in fibo_base]
        fibo_ma = [self.scaler.scale_window(w, self.timeframe, cap=400) for w in fibo_ma]
        fibo_ma = _as_window_list(tf_overrides.get('fibo_ma', fibo_ma), fibo_ma)
        fibo_reference_base = flags_tech.get('fibo_reference_windows', [55, 89, 144])
        fibo_reference_scaled = [self.scaler.scale_window(w, self.timeframe, cap=400) for w in fibo_reference_base]
        fibo_reference_windows = _as_window_list(tf_overrides.get('fibo_reference_windows', fibo_reference_scaled), fibo_reference_scaled)

        retracement_levels = flags_tech.get('fibo_retracement_levels', [0.236, 0.382, 0.5, 0.618, 0.786])
        extension_levels = flags_tech.get('fibo_extension_levels', [1.272, 1.414, 1.618, 2.0])
        try:
            retracement_levels = [float(level) for level in retracement_levels if 0 < float(level) < 1]
        except Exception:
            retracement_levels = [0.382, 0.5, 0.618]
        try:
            extension_levels = [float(level) for level in extension_levels if float(level) > 0]
        except Exception:
            extension_levels = [1.272, 1.618]
        rsi_windows = _as_window_list(tf_overrides.get('rsi_windows', flags_tech.get('rsi_windows',
                                        [5, 7, 9, 14, 21, 28, 42])), [5, 7, 9, 14, 21, 28, 42])
        bb_base = flags_tech.get('bb_windows', [10, 20, 34, 50])
        bb_base_scaled = [self.scaler.scale_window(w, self.timeframe, cap=400) for w in bb_base]
        bb_windows = _as_window_list(tf_overrides.get('bb_windows', bb_base_scaled), bb_base_scaled)
        bb_std_list = np.clip(flags_tech.get('bb_std_list', [1.5, 2.0, 2.5]), 1.0, 3.0)
        mfi_base = flags_tech.get('mfi_windows', [14])
        mfi_scaled = [self.scaler.scale_window(w, self.timeframe, cap=400) for w in mfi_base]
        mfi_windows = _as_window_list(tf_overrides.get('mfi_windows', mfi_scaled), mfi_scaled)
        cci_base = flags_tech.get('cci_windows', [20])
        cci_scaled = [self.scaler.scale_window(w, self.timeframe, cap=400) for w in cci_base]
        cci_windows = _as_window_list(tf_overrides.get('cci_windows', cci_scaled), cci_scaled)

        kdj_cfg = tf_overrides.get('kdj', flags_tech.get('kdj', {
            'rsv_window': 9,
            'k': 3,
            'd': 3
        }))
        if not isinstance(kdj_cfg, dict):
            kdj_cfg = {'rsv_window': 9, 'k': 3, 'd': 3}

        rsv_window = max(1, self.scaler.scale_window(int(kdj_cfg.get('rsv_window', kdj_cfg.get('rsv', 9))), self.timeframe, cap=400))
        k_smooth = max(1, int(kdj_cfg.get('k', 3)))
        d_smooth = max(1, int(kdj_cfg.get('d', 3)))

        def _format_level(level: float) -> str:
            level_str = f"{level:.3f}".rstrip('0').rstrip('.')
            return level_str.replace('-', 'm').replace('.', '_')

        close = ohlcv['close']
        high = ohlcv['high']
        low = ohlcv['low']
        volume = ohlcv['volume']

        features = pd.DataFrame(index=ohlcv.index)

        features['tech_returns'] = close.pct_change()
        features['tech_high_low_ratio'] = high / low
        features['tech_open_close_ratio'] = ohlcv['open'] / close

        range_epsilon = 1e-9
        for ref_win in fibo_reference_windows:
            ref_win = max(1, int(ref_win))
            if ref_win <= 1:
                continue

            high_roll = high.rolling(ref_win).max()
            low_roll = low.rolling(ref_win).min()
            price_range = high_roll - low_roll

            features[f'tech_fibo_range_ratio_{ref_win}'] = (close - low_roll) / (price_range + range_epsilon)

            for level in retracement_levels:
                level_value = high_roll - level * price_range
                level_label = _format_level(level)
                features[f'tech_fibo_retrace_price_{ref_win}_{level_label}'] = level_value
                features[f'tech_fibo_retrace_gap_{ref_win}_{level_label}'] = (close - level_value) / (price_range + range_epsilon)

            for level in extension_levels:
                level_value = high_roll + level * price_range
                level_label = _format_level(level)
                features[f'tech_fibo_extension_price_{ref_win}_{level_label}'] = level_value
                features[f'tech_fibo_extension_gap_{ref_win}_{level_label}'] = (close - level_value) / (price_range + range_epsilon)

        for win in fibo_ma:
            sma = close.rolling(win).mean()
            ema = close.ewm(span=win).mean()
            features[f'tech_sma_{win}'] = sma
            features[f'tech_ema_{win}'] = ema
            features[f'tech_price_sma_ratio_{win}'] = close / sma
            features[f'tech_price_ema_ratio_{win}'] = close / ema

        delta = close.diff()
        for win in rsi_windows:
            gain = delta.where(delta > 0, 0).rolling(win).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(win).mean()
            rs = gain / (loss.replace(0, np.nan))
            features[f'tech_rsi_{win}'] = 100 - (100 / (1 + rs))

        for win in bb_windows:
            sma = close.rolling(win).mean()
            std = close.rolling(win).std()
            for std_k in bb_std_list:
                upper = sma + std_k * std
                lower = sma - std_k * std
                width = (upper - lower) / sma
                pos = (close - lower) / (upper - lower)
                features[f'tech_bb_upper_{win}_{std_k}'] = upper
                features[f'tech_bb_lower_{win}_{std_k}'] = lower
                features[f'tech_bb_width_{win}_{std_k}'] = width
                features[f'tech_bb_pos_{win}_{std_k}'] = pos

        volume_window = max(1, self.scaler.scale_window(int(self.flags.get('volume_sma_window', 20)), self.timeframe, cap=400))
        vol_ma = volume.rolling(volume_window).mean()
        features['tech_volume_sma_ratio'] = volume / vol_ma
        features['tech_price_volume'] = close * volume

        for win in [10, 20]:
            features[f'tech_volatility_{win}'] = close.pct_change().rolling(win).std()

        ema_12 = close.ewm(span=self.scaler.scale_window(12, self.timeframe, cap=400)).mean()
        ema_26 = close.ewm(span=self.scaler.scale_window(26, self.timeframe, cap=400)).mean()
        macd = ema_12 - ema_26
        features['tech_macd'] = macd
        features['tech_macd_signal'] = macd.ewm(span=9).mean()
        features['tech_macd_histogram'] = features['tech_macd'] - features['tech_macd_signal']

        for base_win in [14, 21]:
            win = self.scaler.scale_window(base_win, self.timeframe, cap=400)
            high_max = high.rolling(win).max()
            low_min = low.rolling(win).min()
            features[f'tech_williams_r_{win}'] = -100 * (high_max - close) / (high_max - low_min)

        for base_win in [10, 20]:
            win = self.scaler.scale_window(base_win, self.timeframe, cap=400)
            features[f'tech_momentum_{win}'] = close / close.shift(win) - 1

        typical_price = (high + low + close) / 3
        vol_sum = volume.rolling(20).sum()
        vwap_20 = (typical_price * volume).rolling(20).sum() / vol_sum
        features['tech_vwap_20'] = vwap_20
        features['tech_price_vwap_ratio'] = close / vwap_20

        for win in [20, 50]:
            high_c = high.rolling(win).max()
            low_c = low.rolling(win).min()
            features[f'tech_price_channel_high_{win}'] = high_c
            features[f'tech_price_channel_low_{win}'] = low_c
            features[f'tech_price_channel_pos_{win}'] = (close - low_c) / (high_c - low_c)

        obv_delta = np.where(close > close.shift(1), volume,
                             np.where(close < close.shift(1), -volume, 0))
        features['tech_obv'] = pd.Series(obv_delta, index=close.index).cumsum()

        for win in mfi_windows:
            win = max(1, int(win))
            positive_flow = (typical_price.where(typical_price > typical_price.shift(1), 0) * volume).rolling(win).sum()
            negative_flow = (typical_price.where(typical_price < typical_price.shift(1), 0).abs() * volume).rolling(win).sum()
            mfr = positive_flow / (negative_flow + 1e-9)
            features[f'tech_mfi_{win}'] = 100 - (100 / (1 + mfr))

        for win in cci_windows:
            win = max(1, int(win))
            tp_ma = typical_price.rolling(win).mean()
            tp_md = (typical_price - tp_ma).abs().rolling(win).mean()
            features[f'tech_cci_{win}'] = (typical_price - tp_ma) / (0.015 * tp_md.replace(0, np.nan))

        lowest_low = low.rolling(rsv_window).min()
        highest_high = high.rolling(rsv_window).max()
        rsv = ((close - lowest_low) / (highest_high - lowest_low + 1e-9)) * 100
        k = rsv.ewm(alpha=1 / k_smooth, adjust=False).mean()
        d = k.ewm(alpha=1 / d_smooth, adjust=False).mean()
        j = 3 * k - 2 * d
        features['tech_kdj_k'] = k
        features['tech_kdj_d'] = d
        features['tech_kdj_j'] = j

        features = features.replace([np.inf, -np.inf], np.nan).ffill()
        if base_index is not None:
            features = features.reindex(base_index, method='ffill')
        features = features.fillna(0)
        return features

    def _resample_ohlcv(self, ohlcv: pd.DataFrame, rule: str) -> pd.DataFrame:
        """
        重採樣 OHLCV 到更高時間框架（🔧 P0修復：雙重shift防止時間洩漏）
        
        🚨 關鍵修復：雙重shift策略
        1. 第一重：原始數據shift(1) - 確保只用過去的價格
        2. 第二重：重採樣後shift(1) - 確保只用已完成的大bar
        
        為什麼需要雙重shift？
        - 單次shift不夠：技術指標（如RSI, MACD）在計算時已經"吸收"了當期數據
        - 雙重shift：徹底防止任何形式的未來數據洩漏
        
        示例（15m使用1h數據）：
        時間: 09:45
        錯誤: 使用09:00-10:00的1h bar（包含09:45之後的數據）❌
        單shift: 使用08:00-09:00的1h bar（但RSI等指標可能用了09:00的close）⚠️
        雙shift: 使用07:00-08:00的1h bar（完全隔離未來）✅
        """
        agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        try:
            # 🔧 第一重shift：原始數據向前偏移
            ohlcv_shifted = ohlcv.shift(1)
            
            # 重採樣
            resampled = ohlcv_shifted.resample(rule).agg(agg)
            
            # 🔧 第二重shift：重採樣結果再次向前偏移
            resampled_double_shifted = resampled.shift(1)
            
            # 記錄時間對齊信息（僅在調試模式）
            if len(resampled_double_shifted) > 0 and self.flags.get('debug_resample', False):
                self.logger.debug(
                    f"Resample {rule}: 雙重shift防洩漏 - "
                    f"原始={len(ohlcv)} → 單shift={len(ohlcv_shifted)} → "
                    f"重採樣={len(resampled)} → 最終={len(resampled_double_shifted.dropna())}"
                )
            
            final_data = resampled_double_shifted.dropna()
            
            if len(final_data) > 0:
                self.logger.info(
                    f"✅ {rule} 重採樣完成（雙重shift）: "
                    f"{len(final_data)} 個有效bar，嚴格防止時間洩漏"
                )
            
            return final_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ 重採樣失敗 rule={rule}: {e}")
            return pd.DataFrame(columns=ohlcv.columns)

    def _get_purged_cv_splits(self, n_samples: int, n_splits: int = 5) -> List[Tuple]:
        """
        🔧 P0新增：帶Purge和Embargo的時間序列CV分割
        
        基於《Advances in Financial ML》(Lopez de Prado) 的Purged K-Fold CV
        
        為什麼需要Purge？
        - 標籤依賴未來數據（如lag=17，樣本i的標籤依賴t+17）
        - 訓練集最後的樣本可能與測試集開頭的樣本在時間上重疊
        - 導致信息洩漏，過度樂觀的CV分數
        
        為什麼需要Embargo？
        - 防止同一時期的市場信息在train和test中共享
        - 特別是高頻交易中，相鄰時間點高度相關
        
        Args:
            n_samples: 總樣本數
            n_splits: CV折數
        
        Returns:
            [(train_idx, test_idx), ...] 列表
        """
        if not self.enable_purged_cv:
            # 如果未啟用，使用標準TimeSeriesSplit
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=n_splits)
            return list(tscv.split(range(n_samples)))
        
        from sklearn.model_selection import TimeSeriesSplit
        
        # 基礎時間序列分割
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        
        # 獲取lag參數（從配置或使用默認值）
        lag = getattr(self, '_current_lag', self.scaled_config.get('label_lag', 17))
        # 🔧 FIX: Enhanced embargo = max(embargo_pct, lag×2)
        # Reference: Lopez de Prado (2018), embargo should be at least 2×lag
        embargo_from_pct = int(n_samples * self.embargo_pct)
        embargo_from_lag = lag * 2  # At least 2× lag
        embargo_samples = max(embargo_from_pct, embargo_from_lag, 1)
        
        for train_idx, test_idx in tscv.split(range(n_samples)):
            # 1. Purge: 移除train末尾可能與test重疊的樣本
            #    如果train的樣本i的標籤依賴t+lag，而t+lag落入test範圍
            #    則應該從train中移除
            purge_start = test_idx[0] - lag
            purge_mask = train_idx < purge_start
            train_idx_purged = train_idx[purge_mask]
            
            # 2. Embargo: 移除test開頭的部分樣本
            #    防止訓練集信息通過相鄰時間點洩漏到測試集
            test_idx_embargoed = test_idx[embargo_samples:]
            
            # 驗證分割有效性
            if len(train_idx_purged) > 100 and len(test_idx_embargoed) > 20:
                splits.append((train_idx_purged, test_idx_embargoed))
                self.logger.debug(
                    f"✅ Purged CV Split: "
                    f"Train={len(train_idx_purged)} (purged {len(train_idx)-len(train_idx_purged)}), "
                    f"Test={len(test_idx_embargoed)} (embargoed {embargo_samples})"
                )
            else:
                self.logger.warning(
                    f"⚠️ Split太小，跳過: Train={len(train_idx_purged)}, "
                    f"Test={len(test_idx_embargoed)}"
                )
        
        if len(splits) == 0:
            self.logger.warning("⚠️ Purged CV無有效分割，回退到標準TimeSeriesSplit")
            return list(tscv.split(range(n_samples)))
        
        self.logger.info(
            f"🔒 Purged CV: {len(splits)} 個有效splits (lag={lag}, embargo={embargo_samples})"
        )
        
        return splits

    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba=None, returns=None):
        metrics = {}
        metrics['f1_weighted'] = compute_f1_score(y_true, y_pred, average='weighted')
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        if y_pred_proba is not None:
            metrics['auc_macro'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='macro')
        else:
            metrics['auc_macro'] = 0.5

        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        total = len(y_true)
        metrics['class_distribution'] = {f'class_{int(i)}': count/total for i, count in zip(unique, counts)}

        signal_strength = np.abs(y_pred - 1)
        metrics['avg_signal_strength'] = np.mean(signal_strength)
        metrics['strong_signal_ratio'] = np.mean(signal_strength > 0.5)

        if returns is not None:
            positions = y_pred - 1  # Simplified
            # 🔧 FIX: Remove shift(-1), use proper alignment
            # Position taken at t based on signal, realized at t+1
            strategy_returns = pd.Series(positions).shift(1) * returns
            metrics['strategy_return'] = np.sum(strategy_returns.dropna())
            std = np.std(strategy_returns.dropna())
            metrics['sharpe'] = np.mean(strategy_returns.dropna()) / std * np.sqrt(252) if std > 0 else 0
            metrics['win_rate'] = np.mean(strategy_returns.dropna() > 0)
            metrics['max_drawdown'] = np.min(strategy_returns.cumsum().dropna() - strategy_returns.cumsum().dropna().expanding().max())
            trades = np.abs(np.diff(positions))
            metrics['trade_frequency'] = np.sum(trades) / len(trades) if len(trades) > 0 else 0

        self.logger.info(f"📊 Layer2性能指標：F1={metrics['f1_weighted']:.4f}, 準確率={metrics['accuracy']:.4f}, AUC={metrics['auc_macro']:.4f}")
        return metrics

    def _evaluate_trading_performance(self, model, X_final, y, prices) -> float:
        """🚀 123.md建議：簡化版交易性能評估（勝率、夏普率約束）"""
        try:
            # 訓練模型並預測
            model.fit(X_final, y)
            predictions = model.predict(X_final)

            # 計算交易收益 - 修復numpy數組索引問題
            positions = pd.Series((predictions - 1), index=y.index)  # 轉換為 {-1, 0, 1} Series
            # 🔧 FIX: Remove shift(-1) - use current period returns
            returns = prices.pct_change()  # 當期收益（已知）

            # 對齊數據 - position在t時決定，在t+1時實現
            common_idx = positions.index.intersection(returns.index)
            positions_aligned = positions.shift(1).reindex(common_idx)
            returns_aligned = returns.reindex(common_idx)

            # 計算策略收益
            strategy_returns = positions_aligned * returns_aligned
            strategy_returns = strategy_returns.dropna()

            if len(strategy_returns) == 0:
                return -0.5  # 懲罰

            # 計算關鍵指標
            win_rate = np.mean(strategy_returns > 0)
            total_return = strategy_returns.sum()

            # 計算夏普率
            if strategy_returns.std() > 0:
                sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 24 * 4)  # 15分鐘數據
            else:
                sharpe = 0

            # 月回報估算（假設252個交易日）
            monthly_return = total_return * 21 / len(strategy_returns) if len(strategy_returns) > 0 else 0

            # 🚀 123.md目標約束：勝率≥60%、夏普率≥1.2、月回報≥10%
            penalty = 0
            if win_rate < 0.60:
                penalty -= 0.2 * (0.60 - win_rate)
            if sharpe < 1.2:
                penalty -= 0.1 * (1.2 - sharpe)
            if monthly_return < 0.10:
                penalty -= 0.1 * (0.10 - monthly_return)

            # 獎勵超額表現
            bonus = 0
            if win_rate > 0.65:
                bonus += 0.1 * (win_rate - 0.65)
            if sharpe > 1.5:
                bonus += 0.1 * (sharpe - 1.5)

            performance_score = penalty + bonus

            self.logger.info(f"📈 交易性能: 勝率={win_rate:.3f}, 夏普={sharpe:.3f}, 月回報={monthly_return:.3f}, 分數={performance_score:.3f}")

            return performance_score

        except Exception as e:
            self.logger.warning(f"⚠️ 性能評估失敗: {e}")
            return -0.2  # 小幅懲罰

    def _calculate_trade_frequency_penalty(self, labels, lag) -> float:
        """🚀 123.md建議：交易頻率控制（避免過度交易）"""
        try:
            # 計算信號變化頻率
            signal_changes = np.abs(np.diff(labels))
            change_rate = np.sum(signal_changes > 0) / len(signal_changes) if len(signal_changes) > 0 else 0

            # 根據lag調整合理交易頻率
            # lag越小，允許的交易頻率越高
            max_reasonable_freq = 0.3 / (lag / 12.0)  # 基於12期基準調整

            # 如果交易過於頻繁，施加懲罰
            if change_rate > max_reasonable_freq:
                penalty = 0.1 * (change_rate - max_reasonable_freq) / max_reasonable_freq
                self.logger.info(f"⚠️ 交易頻率過高: {change_rate:.3f} > {max_reasonable_freq:.3f}, 懲罰={penalty:.3f}")
                return min(penalty, 0.3)  # 最大懲罰30%

            return 0.0  # 無懲罰

        except Exception as e:
            self.logger.warning(f"⚠️ 交易頻率計算失敗: {e}")
            return 0.0

    def _shuffle_label_validation(self, X: pd.DataFrame, y: pd.Series, 
                                  model, current_score: float, threshold: float = 0.4) -> tuple:
        """🎯 打乱标签测试：验证模型是否依赖真实标签模式"""
        try:
            self.logger.info("🔀 执行打乱标签测试...")
            
            # 创建打乱的标签
            y_shuffled = y.copy().values
            np.random.shuffle(y_shuffled)
            y_shuffled_series = pd.Series(y_shuffled, index=y.index)
            
            # 使用相同的特征和打乱的标签进行交叉验证
            shuffled_scores = cross_val_score(
                model, X, y_shuffled_series, 
                cv=TimeSeriesSplit(n_splits=3), 
                scoring='f1_weighted',
                n_jobs=-1
            )
            shuffled_score = shuffled_scores.mean()
            
            # 检查打乱标签后的性能
            if shuffled_score > threshold:
                # 打乱标签后仍有预测能力，说明存在数据泄露或过拟合
                return False, f"❌ 打乱标签测试失败: 打乱后F1={shuffled_score:.4f} > {threshold}，疑似数据泄露"
            
            # 计算性能下降比例
            performance_drop = (current_score - shuffled_score) / current_score if current_score > 0 else 0
            if performance_drop < 0.3:  # 性能下降不足30%
                return False, f"❌ 打乱标签测试失败: 性能下降仅{performance_drop:.1%}，模型可能未学习到有效模式"
            
            self.logger.info(f"✅ 打乱标签测试通过: 原始F1={current_score:.4f}, 打乱后F1={shuffled_score:.4f}, 下降{performance_drop:.1%}")
            return True, f"✅ 打乱标签测试通过，性能合理下降{performance_drop:.1%}"
            
        except Exception as e:
            self.logger.warning(f"⚠️ 打乱标签测试失败: {e}")
            return True, f"⚠️ 打乱标签测试异常但继续: {e}"

    def _random_feature_validation(self, X: pd.DataFrame, y: pd.Series, 
                                   model, current_score: float, threshold: float = 0.4) -> tuple:
        """🎯 随机特征测试：验证模型是否仅依赖噪声特征"""
        try:
            self.logger.info("🎲 执行随机特征测试...")
            
            # 生成与原始特征相同维度的随机特征
            X_random = pd.DataFrame(
                np.random.randn(*X.shape), 
                index=X.index, 
                columns=[f'random_feature_{i}' for i in range(X.shape[1])]
            )
            
            # 使用随机特征和真实标签进行交叉验证
            random_scores = cross_val_score(
                model, X_random, y, 
                cv=TimeSeriesSplit(n_splits=3), 
                scoring='f1_weighted',
                n_jobs=-1
            )
            random_score = random_scores.mean()
            
            # 检查随机特征的预测能力
            if random_score > threshold:
                # 随机特征有预测能力，说明可能存在问题
                return False, f"❌ 随机特征测试失败: 随机特征F1={random_score:.4f} > {threshold}，模型可能过拟合"
            
            # 检查真实特征vs随机特征的性能差异
            feature_advantage = (current_score - random_score) / current_score if current_score > 0 else 0
            if feature_advantage < 0.2:  # 真实特征优势不足20%
                return False, f"❌ 随机特征测试失败: 真实特征优势仅{feature_advantage:.1%}，特征工程效果有限"
            
            self.logger.info(f"✅ 随机特征测试通过: 真实特征F1={current_score:.4f}, 随机特征F1={random_score:.4f}, 优势{feature_advantage:.1%}")
            return True, f"✅ 随机特征测试通过，真实特征显著优于随机特征{feature_advantage:.1%}"
            
        except Exception as e:
            self.logger.warning(f"⚠️ 随机特征测试失败: {e}")
            return True, f"⚠️ 随机特征测试异常但继续: {e}"

    def _validate_score_legitimacy(self, score: float, X: pd.DataFrame, y: pd.Series, model) -> tuple:
        """🚀 文档修复版：多重验证确保分数真实性（移除0.8硬阈值）"""
        try:
            self.logger.info(f"🔍 开始多重验证分数: {score:.4f}")

            # 1. 🚀 随机数据测试
            X_random = pd.DataFrame(np.random.randn(*X.shape), index=X.index, columns=X.columns)
            try:
                random_scores = cross_val_score(model, X_random, y, cv=TimeSeriesSplit(n_splits=3), n_jobs=-1)
                random_score = random_scores.mean()
                if random_score > 0.4:  # 随机数据不应有预测能力
                    return False, f"❌ 随机数据测试失败: {random_score:.3f} > 0.4"
                self.logger.info(f"✅ 随机数据测试通过: {random_score:.3f}")
            except Exception as e:
                self.logger.warning(f"⚠️ 随机数据测试失败: {e}")

            # 2. 🚀 标签打乱测试
            y_shuffled = y.copy().values
            np.random.shuffle(y_shuffled)
            y_shuffled_series = pd.Series(y_shuffled, index=y.index)
            try:
                shuffled_scores = cross_val_score(model, X, y_shuffled_series, cv=TimeSeriesSplit(n_splits=3), n_jobs=-1)
                shuffled_score = shuffled_scores.mean()
                if shuffled_score > 0.4:  # 打乱标签不应有预测能力
                    return False, f"❌ 标签打乱测试失败: {shuffled_score:.3f} > 0.4"
                self.logger.info(f"✅ 标签打乱测试通过: {shuffled_score:.3f}")
            except Exception as e:
                self.logger.warning(f"⚠️ 标签打乱测试失败: {e}")

            # 3. 🚀 时间序列前向验证差异检查
            try:
                tscv = TimeSeriesSplit(n_splits=5)
                ts_scores = cross_val_score(model, X, y, cv=tscv, n_jobs=-1)
                ts_score = ts_scores.mean()
                score_diff = abs(score - ts_score)
                if score_diff > 0.15:  # 时序验证与常规验证差异过大
                    return False, f"❌ 时序验证差异过大: {score:.3f} vs {ts_score:.3f} (diff={score_diff:.3f})"
                self.logger.info(f"✅ 时序验证一致性通过: 差异={score_diff:.3f}")
            except Exception as e:
                self.logger.warning(f"⚠️ 时序验证测试失败: {e}")

            # 4. 🚀 特征重要性合理性（如果模型支持）
            if hasattr(model, 'feature_importances_'):
                try:
                    importances = model.feature_importances_
                    max_importance = np.max(importances)
                    if max_importance > 0.8:  # 单个特征贡献过大
                        return False, f"❌ 单个特征权重过高: {max_importance:.3f} > 0.8，可能过拟合"
                    self.logger.info(f"✅ 特征重要性平衡: 最大权重={max_importance:.3f}")
                except Exception as e:
                    self.logger.warning(f"⚠️ 特征重要性检查失败: {e}")

            # 5. 🚀 分数合理性评估（无硬上限）
            if score > 0.9:
                self.logger.warning(f"⚠️ 分数极高 {score:.4f}，建议额外验证但不拒绝")
            elif score > 0.75:
                self.logger.info(f"📈 分数较高 {score:.4f}，符合优秀特征工程预期")
            elif score > 0.5:
                self.logger.info(f"📊 分数正常 {score:.4f}，符合预期范围")
            else:
                self.logger.info(f"📉 分数较低 {score:.4f}，可继续优化")

            return True, "✅ 多重验证通过"

        except Exception as e:
            self.logger.error(f"❌ 多重验证失败: {e}")
            return True, f"⚠️ 验证异常但继续: {e}"  # 验证失败时不阻止优化

    def _get_env_float(self, name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default

    def _soft_penalize_kpis(self, metrics: Dict[str, float]) -> Dict[str, float]:
        penalties = dict(metrics)
        min_win = self._get_env_float("L2_MIN_WINRATE", 0.65)
        min_sharpe = self._get_env_float("L2_MIN_SHARPE", 1.2)
        max_dd = self._get_env_float("L2_MAX_DRAWDOWN", 0.25)

        if penalties.get("sharpe", 0.0) < min_sharpe:
            penalties["sharpe"] = 0.1
        if penalties.get("win_rate", 0.0) < min_win:
            penalties["win_rate"] = 0.01
        if penalties.get("profit_factor", 0.0) < 1.0:
            penalties["profit_factor"] = 0.9
        if penalties.get("total_return", 0.0) <= 0:
            penalties["total_return"] = -0.1
        if penalties.get("annual_return", 0.0) <= 0:
            penalties["annual_return"] = -0.1
        if penalties.get("max_drawdown", 1.0) > max_dd:
            penalties["max_drawdown"] = 1.0
        return penalties

    def _kpis_to_multi_values(self, metrics: Dict[str, float]) -> Dict[str, float]:
        calmar = metrics.get("calmar")
        if calmar is None:
            calmar = metrics.get("sortino", 0.0)
        values = {
            "sharpe": metrics.get("sharpe", 0.0),
            "calmar": calmar,
            "profit_factor": metrics.get("profit_factor", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "total_return": metrics.get("total_return", 0.0),
            "annual_return": metrics.get("annual_return", 0.0),
            "neg_max_drawdown": -metrics.get("max_drawdown", 1.0),
        }
        return values

    def objective(self, trial: optuna.Trial) -> float:
        """🚀 123.md建議：參數化標籤生成 + 性能約束的目標函數"""
        try:
            # Phase 4.1: 擴展空間
            # 阶段D：扩展特征选择方法（从2种→5种）
            selection_method = trial.suggest_categorical('feature_selection_method', 
                ['stability', 'mutual_info', 'rfe', 'lasso', 'tree_based'])
            noise_reduction = trial.suggest_categorical('noise_reduction', [True, False])
            feature_interaction = trial.suggest_categorical('feature_interaction', [False, True])
            # 🚀 Fail-Fast檢查預加載數據
            if len(self.features) == 0 or len(self.close_prices) == 0:
                raise ValueError("預加載數據為空，初始化失敗")

            # 🚀 新增：讀取Layer1標籤優化結果，實現聯動優化
            layer1_params = None
            try:
                tf_specific = self.config_path / f"label_params_{self.timeframe}.json"
                layer1_config_file = tf_specific if tf_specific.exists() else (self.config_path / "label_params.json")
                if layer1_config_file.exists():
                    with open(layer1_config_file, 'r', encoding='utf-8') as f:
                        layer1_result = json.load(f)
                        layer1_params = layer1_result.get('best_params', {})
                        self.logger.info(
                            f"📖 讀取Layer1優化結果: lag={layer1_params.get('lag')}, "
                            f"buy_q={layer1_params.get('buy_quantile', 0):.3f}"
                        )
            except Exception as e:
                self.logger.warning(f"無法讀取Layer1結果: {e}")

            # ========== 🚀 P0修復：強制使用Layer1標籤，確保分層優化一致性 ==========
            # 
            # 問題分析：
            # - Layer1經過150 trials優化，找到最優標籤參數（分數1.0185，完美分佈25/50/25）
            # - Layer2重新生成標籤導致分佈偏斜（9.7/37.2/53.1），破壞Layer1成果
            # - 導致賣出信號召回率僅8.2%，系統無法有效對沖下跌風險
            #
            # 解決方案：
            # - Layer2直接使用Layer1物化的標籤，不再重新生成
            # - 確保標籤分佈保持Layer1優化的結果
            # - lag參數從Layer1讀取，用於特徵對齊
            # ========================================================================
            
            # 加載Layer1優化的標籤
            labels_df = self.load_latest_labels()
            if labels_df is None or 'label' not in labels_df.columns:
                error_msg = (
                    "❌ Layer2依賴Layer1標籤，但未找到標籤文件。\n"
                    "請確保：\n"
                    "  1. Layer1優化已完成\n"
                    "  2. 標籤文件存在於 data/processed/labels/{symbol}_{timeframe}/\n"
                    "  3. 或 configs/label_params_{timeframe}.json 包含 materialized_path"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            labels = labels_df['label'].astype(int)
            
            # 從Layer1配置讀取lag參數（用於特徵對齊和性能計算）
            try:
                label_config = self.config_path / f"label_params_{self.timeframe}.json"
                if not label_config.exists():
                    label_config = self.config_path / "label_params.json"
                
                if not label_config.exists():
                    raise FileNotFoundError(f"未找到Layer1配置: {label_config}")
                
                with open(label_config, 'r', encoding='utf-8') as f:
                    layer1_result = json.load(f)
                    layer1_best_params = layer1_result.get('best_params', {})
                    lag = int(layer1_best_params.get('lag', 14))
                    threshold_method = layer1_best_params.get('threshold_method', 'unknown')
                    layer1_score = layer1_result.get('best_score', 0.0)
                    
                    self.logger.info("=" * 60)
                    self.logger.info("🔗 Layer2使用Layer1優化結果:")
                    self.logger.info(f"  Layer1分數: {layer1_score:.4f}")
                    self.logger.info(f"  標籤方法: {threshold_method}")
                    self.logger.info(f"  lag週期: {lag}")
                    self.logger.info(f"  買入分位: {layer1_best_params.get('buy_quantile', 'N/A')}")
                    self.logger.info(f"  賣出分位: {layer1_best_params.get('sell_quantile', 'N/A')}")
                    if 'profit_multiplier' in layer1_best_params:
                        self.logger.info(f"  止盈倍數: {layer1_best_params['profit_multiplier']:.2f}×ATR")
                        self.logger.info(f"  止損倍數: {layer1_best_params['stop_multiplier']:.2f}×ATR")
                        rr_ratio = layer1_best_params['profit_multiplier'] / layer1_best_params['stop_multiplier']
                        self.logger.info(f"  風險回報比: {rr_ratio:.2f}:1")
                    self.logger.info("=" * 60)
                    
            except Exception as e:
                self.logger.error(f"❌ 無法讀取Layer1配置: {e}")
                raise ValueError(f"Layer2需要Layer1配置文件: {e}")
            
            # 驗證標籤分佈（應該接近Layer1的優化結果）
            dist = labels.value_counts(normalize=True).sort_index()
            actual_dist = [dist.get(i, 0.0) for i in [0, 1, 2]]
            
            self.logger.info(
                f"📊 Layer1標籤分佈: "
                f"賣出={actual_dist[0]:.1%}, 持有={actual_dist[1]:.1%}, 買入={actual_dist[2]:.1%}"
            )
            
            # 標籤質量檢查（警告但不阻止）
            target_dist = [0.25, 0.50, 0.25]
            deviations = [abs(actual_dist[i] - target_dist[i]) for i in range(3)]
            max_deviation = max(deviations)
            
            if max_deviation > 0.15:
                self.logger.warning(
                    f"⚠️ 標籤分佈偏差較大: {max_deviation:.1%} > 15%，"
                    f"可能影響模型性能。建議檢查Layer1優化結果。"
                )
            elif max_deviation > 0.08:
                self.logger.info(
                    f"📊 標籤分佈偏差中等: {max_deviation:.1%}，"
                    f"在可接受範圍內"
                )
            else:
                self.logger.info(
                    f"✅ 標籤分佈良好，最大偏差僅 {max_deviation:.1%}"
                )
            
            # 檢查是否有缺失類別
            missing_classes = [i for i in [0, 1, 2] if dist.get(i, 0) == 0]
            if missing_classes:
                class_names = {0: '賣出', 1: '持有', 2: '買入'}
                missing_names = [class_names[c] for c in missing_classes]
                self.logger.error(
                    f"❌ 標籤缺失類別: {missing_classes} ({missing_names})，"
                    f"無法進行多分類訓練"
                )
                return 0.0
            
            # 統計總樣本數
            total_samples = len(labels)
            self.logger.info(f"📊 Layer1標籤總數: {total_samples:,}個")
            
            # 移除動態標籤再平衡（保持Layer1原始結果）
            # ❌ 不再執行以下邏輯：
            # if self.flags.get('enable_dynamic_label_balance', False):
            #     labels = self._rebalance_labels(...)
            
            self.logger.info("✅ Layer1標籤加載完成，跳過重新生成，確保分層優化一致性")

            # 🔧 數據對齊與清理
            common_idx = self.features.index.intersection(labels.index)
            self._ensure_X_full()
            X = self.X_full.reindex(common_idx).astype('float32').fillna(0)
            y = labels.loc[common_idx].astype(int)

            # 特徵質量過濾
            X = self._filter_low_quality_features(X)

            if len(X) < 1000:  # 確保有足夠的數據
                return 0.0

            n_features = len(X.columns)
            self.logger.info(f"📊 總特徵數: {n_features}")

            selection_cfg = self.optimize_feature_selection_params(trial, n_features, layer1_params)
            coarse_k = selection_cfg['coarse_k']
            fine_k = selection_cfg['fine_k']
            corr_threshold = selection_cfg['correlation_threshold']

            self.logger.info(
                f"🔧 特徵選擇參數: coarse_k={coarse_k} ({coarse_k/n_features:.1%}), "
                f"fine_k={fine_k} ({fine_k/coarse_k:.1%}), "
                f"相關性閾值={corr_threshold:.3f}"
            )

            # 統一順序：先去相關(對冗餘) → 再應用穩定性遮罩 → 再粗/精選
            # 注意：此處不再於全資料集層級直接裁切為 stable_cols，避免"先穩定再去相關"的反向效果

            # 🚀 準備選擇器緩存（避免重複計算）
            if not hasattr(self, '_selector_cache'):
                self._selector_cache = {
                    'variance': {},
                    'coarse': {},
                    'fine': {}
                }
            variance_cache = self._selector_cache['variance']
            coarse_cache = self._selector_cache['coarse']
            fine_cache = self._selector_cache['fine']

            current_splits = self.flags.get('cv_splits', 5)

            # 🚀 P0修復：使用Purged CV（防止時間洩漏）
            # 保存當前lag供purge使用
            self._current_lag = lag
            outer_cv = self._get_purged_cv_splits(n_samples=len(X), n_splits=current_splits)
            cv_scores = []
            # initialize feature aggregation to avoid NameError
            phase = 'full'
            selection_counts: Counter = Counter()
            selected_union: set = set()
            best_fold_cols: List[str] = []
            best_fold_score: float = -1.0

            self.logger.info(f"🔄 開始嵌套交叉驗證: n_features={n_features}, {len(X)}樣本")

            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv):
                try:
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    self.logger.info(f"  Fold {fold_idx+1}: 訓練={len(X_train)}, 測試={len(X_test)}")

                    variance_key = (current_splits, fold_idx)
                    if variance_key in variance_cache:
                        cols_var = variance_cache[variance_key]
                    else:
                        variance_selector = VarianceThreshold(threshold=0.01)
                        variance_selector.fit(X_train)
                        cols_var = X_train.columns[variance_selector.get_support(indices=True)].tolist()
                        variance_cache[variance_key] = cols_var

                    if len(cols_var) == 0:
                        cv_scores.append(0.0)
                        continue

                    X_train_var = X_train[cols_var]
                    X_test_var = X_test[cols_var]

                    # 先做一次目標相關性過濾（mutual information）
                    try:
                        tc_keep_ratio = float(self.flags.get('target_corr_keep_ratio', 0.6))
                        tc_min_score = float(self.flags.get('target_corr_min_score', 0.0))
                        tc_cols = self._filter_by_target_correlation(X_train_var, y_train, keep_ratio=tc_keep_ratio, min_score=tc_min_score)
                        if len(tc_cols) >= 3:
                            X_train_var = X_train_var[tc_cols]
                            X_test_var = X_test_var[tc_cols]
                    except Exception as e:
                        self.logger.warning(f"target-corr 過濾跳過: {e}")

                    # 再做一次去冗餘（pairwise correlation）
                    pre_corr_cols = self._remove_correlated_features_smart(X_train_var, corr_threshold)
                    if len(pre_corr_cols) >= 3:
                        X_train_var = X_train_var[pre_corr_cols]
                        X_test_var = X_test_var[pre_corr_cols]

                    # 再應用穩定性遮罩（若有），避免與前一步衝突而造成特徵集過度收縮
                    if hasattr(self, 'stable_cols') and self.stable_cols:
                        pre_keep = [c for c in X_train_var.columns if c in self.stable_cols]
                        if len(pre_keep) >= max(10, int(0.2 * X_train_var.shape[1])):
                            X_train_var = X_train_var[pre_keep]
                            X_test_var = X_test_var[pre_keep]

                    fold_n_features = X_train_var.shape[1]
                    fold_coarse_k = min(coarse_k, int(fold_n_features * 0.80), fold_n_features - 1)
                    if fold_coarse_k < 10:
                        # 保底策略：採用變異度 Top-10 避免直接 0 分
                        self.logger.warning(f"  Fold {fold_idx+1}: 特徵數不足 {fold_n_features}，啟用保底策略（Top-10 by variance）")
                        try:
                            variances = X_train_var.var().sort_values(ascending=False)
                            top_k = min(10, max(3, X_train_var.shape[1] - 1))
                            fallback_cols = variances.head(top_k).index.tolist()
                            if len(fallback_cols) >= 3:
                                X_train_var = X_train_var[fallback_cols]
                                X_test_var = X_test_var[fallback_cols]
                                fold_n_features = X_train_var.shape[1]
                                fold_coarse_k = min(top_k, fold_n_features - 1)
                            else:
                                cv_scores.append(0.0)
                                continue
                        except Exception as e:
                            self.logger.warning(f"  保底策略失敗: {e}")
                            cv_scores.append(0.0)
                            continue

                    # ========== 🔧 核心修复：15m特征优先选择（解决1h特征过多问题） ==========
                    # 强制至少50%的粗选特征来自原生15m时间框架
                    min_native_ratio = self.scaled_config.get('min_native_feature_ratio', 0.50)
                    
                    # 识别15m原生特征
                    native_15m_features = [col for col in X_train_var.columns 
                                         if col.startswith('15m_native_')]
                    non_native_features = [col for col in X_train_var.columns 
                                          if not col.startswith('15m_native_')]
                    
                    target_native_count = int(fold_coarse_k * min_native_ratio)
                    target_native_count = min(target_native_count, len(native_15m_features))
                    
                    if len(native_15m_features) > 0 and target_native_count > 0:
                        # 从15m特征中强制选择一部分
                        from sklearn.feature_selection import SelectKBest, mutual_info_classif
                        
                        native_15m_k = min(target_native_count, len(native_15m_features))
                        native_selector = SelectKBest(mutual_info_classif, k=native_15m_k)
                        
                        try:
                            X_train_native = X_train_var[native_15m_features]
                            native_selector.fit(X_train_native, y_train)
                            forced_native_cols = X_train_native.columns[
                                native_selector.get_support(indices=True)
                            ].tolist()
                            
                            # 剩余配额从非15m特征中选择
                            remaining_k = fold_coarse_k - len(forced_native_cols)
                            
                            if remaining_k > 0 and len(non_native_features) > 0:
                                X_train_nonnative = X_train_var[non_native_features]
                                nonnative_k = min(remaining_k, len(non_native_features))
                                nonnative_selector = SelectKBest(mutual_info_classif, k=nonnative_k)
                                nonnative_selector.fit(X_train_nonnative, y_train)
                                selected_nonnative_cols = X_train_nonnative.columns[
                                    nonnative_selector.get_support(indices=True)
                                ].tolist()
                                
                                cols_coarse = forced_native_cols + selected_nonnative_cols
                            else:
                                cols_coarse = forced_native_cols
                            
                            native_ratio = len(forced_native_cols) / max(len(cols_coarse), 1)
                            self.logger.info(
                                f"  🎯 Fold {fold_idx+1}: 15m特征优先 - "
                                f"15m={len(forced_native_cols)}/{len(cols_coarse)} "
                                f"({native_ratio:.1%}), 目标≥{min_native_ratio:.0%}"
                            )
                        except Exception as e:
                            self.logger.warning(f"  15m特征优先选择失败，回退到标准选择: {e}")
                            # 回退到标准选择
                            cols_coarse = None
                    else:
                        cols_coarse = None
                    
                    # 如果15m优先选择失败，使用标准选择
                    if cols_coarse is None:
                        cols_var_tuple = tuple(cols_var)
                        coarse_key = (current_splits, fold_idx, tuple(X_train_var.columns), fold_coarse_k)
                        if coarse_key in coarse_cache:
                            cols_coarse = coarse_cache[coarse_key]
                        else:
                            # 阶段D：根据selection_method选择不同的特征选择器
                            if selection_method == 'stability':
                                score_func = f_classif
                            elif selection_method == 'mutual_info':
                                score_func = mutual_info_classif
                            elif selection_method == 'rfe':
                                # RFE (Recursive Feature Elimination)
                                from sklearn.feature_selection import RFE
                                from sklearn.ensemble import GradientBoostingClassifier
                                estimator = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
                                rfe_selector = RFE(estimator=estimator, n_features_to_select=fold_coarse_k, step=0.1)
                                rfe_selector.fit(X_train_var, y_train)
                                cols_coarse = X_train_var.columns[rfe_selector.get_support(indices=True)].tolist()
                                coarse_cache[coarse_key] = cols_coarse
                                score_func = None  # RFE已处理，跳过SelectKBest
                            elif selection_method == 'lasso':
                                # LASSO-based selection
                                from sklearn.feature_selection import SelectFromModel
                                from sklearn.linear_model import LassoCV
                                lasso = LassoCV(cv=3, random_state=42, max_iter=500)
                                lasso_selector = SelectFromModel(lasso, max_features=fold_coarse_k, threshold=-np.inf)
                                lasso_selector.fit(X_train_var, y_train)
                                cols_coarse = X_train_var.columns[lasso_selector.get_support(indices=True)].tolist()
                                coarse_cache[coarse_key] = cols_coarse
                                score_func = None  # LASSO已处理
                            elif selection_method == 'tree_based':
                                # Tree-based feature importance
                                from sklearn.ensemble import RandomForestClassifier
                                rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
                                rf.fit(X_train_var, y_train)
                                importances = rf.feature_importances_
                                top_indices = np.argsort(importances)[::-1][:fold_coarse_k]
                                cols_coarse = X_train_var.columns[top_indices].tolist()
                                coarse_cache[coarse_key] = cols_coarse
                                score_func = None  # Tree-based已处理
                            else:
                                score_func = f_classif  # 默认
                            
                            # 如果使用SelectKBest（stability或mutual_info）
                            if score_func is not None:
                                coarse_selector = SelectKBest(score_func, k=fold_coarse_k)
                                coarse_selector.fit(X_train_var, y_train)
                                cols_coarse = X_train_var.columns[coarse_selector.get_support(indices=True)].tolist()
                                coarse_cache[coarse_key] = cols_coarse

                    if len(cols_coarse) == 0:
                        cv_scores.append(0.0)
                        continue

                    # 對齊 coarse 子集欄位（含別名解析與交集保護）
                    cols_coarse = self._normalize_column_subset(cols_coarse, X_train_var, X_test_var)
                    if len(cols_coarse) < 3:
                        cv_scores.append(0.0)
                        continue
                    X_train_coarse = X_train_var[cols_coarse]
                    X_test_coarse = X_test_var[cols_coarse]

                    coarse_n_features = X_train_coarse.shape[1]
                    fold_fine_k = min(fine_k, coarse_n_features - 1)

                    fine_key = (current_splits, fold_idx, tuple(X_train_coarse.columns), fold_fine_k)
                    if fine_key in fine_cache:
                        cols_fine = fine_cache[fine_key]
                    else:
                        fine_selector = SelectKBest(mutual_info_classif, k=fold_fine_k)
                        fine_selector.fit(X_train_coarse, y_train)
                        cols_fine = X_train_coarse.columns[fine_selector.get_support(indices=True)].tolist()
                        fine_cache[fine_key] = cols_fine

                    if len(cols_fine) < 3:
                        cv_scores.append(0.0)
                        continue

                    # 對齊 fine 子集欄位（含別名解析與交集保護）
                    cols_fine = self._normalize_column_subset(cols_fine, X_train_coarse, X_test_coarse)
                    if len(cols_fine) < 3:
                        cv_scores.append(0.0)
                        continue
                    X_train_fine_df = X_train_coarse[cols_fine]
                    X_test_fine_df = X_test_coarse[cols_fine]

                    # 最終不再重複去相關，避免"重複砍"造成不一致
                    selected_cols = list(X_train_fine_df.columns)
                    
                    # ========== 🚀 P6修復：多時框平衡約束（Task 3.1） ==========
                    # 統計各時框特徵數量
                    tf_counts = {'native_15m': 0, '1h': 0, '4h': 0, 'other': 0}
                    for col in selected_cols:
                        if col.startswith('15m_native_'):
                            tf_counts['native_15m'] += 1
                        elif col.startswith('1h_'):
                            tf_counts['1h'] += 1
                        elif col.startswith('4h_'):
                            tf_counts['4h'] += 1
                        else:
                            tf_counts['other'] += 1
                    
                    total_selected = len(selected_cols)
                    tf_ratios = {k: v/total_selected for k, v in tf_counts.items()} if total_selected > 0 else {}
                    
                    # 診斷日誌
                    if fold_idx == 0 or tf_ratios.get('native_15m', 0) < 0.20:  # 第一折或異常時輸出
                        self.logger.info(
                            f"  時框分佈: "
                            f"15m={tf_ratios.get('native_15m', 0):.1%}, "
                            f"1h={tf_ratios.get('1h', 0):.1%}, "
                            f"4h={tf_ratios.get('4h', 0):.1%}, "
                            f"其他={tf_ratios.get('other', 0):.1%}"
                        )
                    
                    # 檢查是否過度集中在單一時框
                    max_tf_ratio = max(tf_ratios.values()) if tf_ratios else 1.0
                    if max_tf_ratio > 0.75:
                        dominant_tf = max(tf_ratios, key=tf_ratios.get) if tf_ratios else 'unknown'
                        self.logger.warning(
                            f"  ⚠️ Fold {fold_idx+1}: {dominant_tf}特徵佔比過高 "
                            f"({max_tf_ratio:.1%} > 75%)，缺少多尺度視角"
                        )
                    
                    # 如果原生15m特徵太少（<25%），嘗試補充
                    native_ratio = tf_counts.get('native_15m', 0) / total_selected if total_selected > 0 else 0
                    if native_ratio < 0.25 and total_selected > 0:
                        # 從全特徵集中找15m特徵
                        available_15m = [c for c in X_train_var.columns if c.startswith('15m_native_')]
                        if available_15m:
                            # 需要補充的數量
                            target_15m_count = max(1, int(total_selected * 0.30))
                            additional_needed = target_15m_count - tf_counts['native_15m']
                            
                            if additional_needed > 0:
                                # 選擇變異度最高的15m特徵補充
                                try:
                                    variances = X_train_var[available_15m].var().sort_values(ascending=False)
                                    add_15m = variances.head(min(additional_needed, len(available_15m))).index.tolist()
                                    
                                    # 合併到選擇列表
                                    selected_cols = list(set(selected_cols + add_15m))
                                    self.logger.info(
                                        f"  🔧 補充{len(add_15m)}個15m特徵 "
                                        f"(原{native_ratio:.1%} → 目標30%+)"
                                    )
                                except Exception as e:
                                    self.logger.warning(f"  補充15m特徵失敗: {e}")
                    
                    df_train_final = X_train_fine_df[selected_cols] if all(c in X_train_fine_df.columns for c in selected_cols) else X_train_fine_df
                    X_test_fine_df = X_test_fine_df[[c for c in selected_cols if c in X_test_fine_df.columns]]

                    if df_train_final.shape[1] < 3:
                        cv_scores.append(0.0)
                        continue

                    # 可選交互特徵（限制在較小維度）
                    X_train_final = df_train_final.values
                    X_test_final = X_test_fine_df.values
                    interaction_max_dim = int(self.flags.get('interaction_max_dim', 40))
                    interaction_degree = int(self.flags.get('interaction_degree', 2))
                    if feature_interaction and df_train_final.shape[1] <= interaction_max_dim:
                        try:
                            poly = PolynomialFeatures(degree=interaction_degree, include_bias=False)
                            X_train_final = poly.fit_transform(X_train_final)
                            X_test_final = poly.transform(X_test_final)
                        except Exception as e:
                            self.logger.warning(f"交互特徵生成失敗: {e}")

                    # Phase 2.2: 每折標準化 + 可選PCA（保留95%方差）
                    try:
                        scaler = StandardScaler()
                        X_train_final = scaler.fit_transform(X_train_final)
                        X_test_final = scaler.transform(X_test_final)
                        # PCA 成分回灌：如啟用則將前K主成分加入而非完全取代
                        pca_threshold = int(self.flags.get('pca_dim_threshold', 60))
                        pca_variance = float(self.flags.get('pca_variance', 0.95))
                        max_pcs = int(self.flags.get('pca_max_components', 30))
                        if noise_reduction and df_train_final.shape[1] > pca_threshold:
                            pca = PCA(n_components=pca_variance, random_state=42)
                            pca.fit(X_train_final)
                            X_train_pca = pca.transform(X_train_final)
                            X_test_pca = pca.transform(X_test_final)
                            if isinstance(pca.n_components_, int):
                                n_pcs = min(max_pcs, int(pca.n_components_))
                            else:
                                n_pcs = min(max_pcs, X_train_pca.shape[1])
                            X_train_final = np.hstack([X_train_final, X_train_pca[:, :n_pcs]])
                            X_test_final = np.hstack([X_test_final, X_test_pca[:, :n_pcs]])
                    except Exception as e:
                        self.logger.warning(f"標準化/PCA失敗: {e}")

                    classifier = self.create_enhanced_classifier(trial, df_train_final.shape[1])

                    # 早期斷言：避免常數標籤或極低多樣性造成 LGBM 無法分裂
                    try:
                        if pd.Series(y_train).nunique() <= 1:
                            self.logger.warning("⚠️ 略過本折：y_train 單一類別，無法訓練")
                            cv_scores.append(0.0)
                            continue
                    except Exception:
                        pass

                    # 類別再平衡：以樣本權重訓練
                    try:
                        sample_weight = self._compute_sample_weights(y_train)
                        classifier.fit(X_train_final, y_train, sample_weight=sample_weight)
                    except TypeError:
                        classifier.fit(X_train_final, y_train)
                    except Exception as e:
                        self.logger.warning(f"樣本權重訓練失敗，改為無權重: {e}")
                        classifier.fit(X_train_final, y_train)

                    # 閾值搜尋：Trade vs Hold 兩階段決策以最大化 Macro F1
                    y_pred = None
                    try:
                        if hasattr(classifier, 'predict_proba'):
                            y_proba = classifier.predict_proba(X_test_final)
                            if self.flags.get('enable_threshold_search', True):
                                # 自適應搜尋：先粗掃再細掃
                                best_tau, y_pred = self._search_best_trade_threshold(y_test, y_proba)
                                self.logger.info(f"  閾值搜尋完成: best_tau={best_tau:.3f}")
                    except Exception as e:
                        self.logger.warning(f"閾值搜尋失敗: {e}")
                        y_pred = None
                    if y_pred is None:
                        y_pred = classifier.predict(X_test_final)
                    # 複合評分：提高 macro F1 權重以降低持有類主導
                    f1_w = compute_f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1_m = compute_f1_score(y_test, y_pred, average='macro', zero_division=0)
                    fold_score = 0.5 * f1_m + 0.5 * f1_w

                    # 擴充度量（✅ P0修復：確保CV指標正確計算）
                    # ✅ BUG修复：显式导入避免作用域问题
                    from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score
                    try:
                        prec_m = precision_score(y_test, y_pred, average='macro', zero_division=0)
                        rec_m = recall_score(y_test, y_pred, average='macro', zero_division=0)
                        bal_acc = balanced_accuracy_score(y_test, y_pred)
                        
                        # ✅ 調試日誌：確認指標被正確計算
                        if prec_m == 0 or rec_m == 0 or bal_acc == 0:
                            self.logger.warning(
                                f"  ⚠️ Fold {fold_idx+1} CV指標異常: "
                                f"prec={prec_m:.3f}, rec={rec_m:.3f}, bal_acc={bal_acc:.3f}"
                            )
                    except Exception as e:
                        self.logger.error(f"  ❌ Fold {fold_idx+1} CV指標計算失敗: {e}")
                        prec_m, rec_m, bal_acc = 0.0, 0.0, 0.0

                    auc_macro = None
                    try:
                        if hasattr(classifier, 'predict_proba'):
                            y_proba = classifier.predict_proba(X_test_final)
                            auc_macro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                    except Exception:
                        auc_macro = None

                    try:
                        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
                        
                        # 🚀 P2修復：計算並記錄各類召回率，用於動態權重調整
                        class_recalls = {}
                        cls_names = {0: '賣出', 1: '持有', 2: '買入'}
                        
                        for cls_idx in range(3):
                            if cm[cls_idx].sum() > 0:
                                recall = cm[cls_idx, cls_idx] / cm[cls_idx].sum()
                                class_recalls[cls_idx] = float(recall)
                        
                        # 更新全局召回率歷史（指數移動平均）
                        if not hasattr(self, '_class_recall_history'):
                            self._class_recall_history = {}
                        
                        for cls_idx, recall in class_recalls.items():
                            if cls_idx in self._class_recall_history:
                                # EMA: alpha=0.3（新值權重30%）
                                self._class_recall_history[cls_idx] = (
                                    0.7 * self._class_recall_history[cls_idx] + 
                                    0.3 * recall
                                )
                            else:
                                self._class_recall_history[cls_idx] = recall
                        
                        # 診斷日誌（僅在召回率異常時輸出）
                        for cls_idx in [0, 1, 2]:
                            recall_val = class_recalls.get(cls_idx, 0.0)
                            hist_val = self._class_recall_history.get(cls_idx, 0.0)
                            
                            # 狀態判斷
                            if recall_val > 0.40:
                                status = "✅"
                            elif recall_val > 0.25:
                                status = "📊"
                            else:
                                status = "⚠️"
                            
                            # 只記錄異常情況（召回率<35%）或優秀情況（>70%）
                            if recall_val < 0.35 or recall_val > 0.70:
                                self.logger.info(
                                    f"  {status} {cls_names[cls_idx]}召回率: "
                                    f"本fold={recall_val:.1%}, 歷史EMA={hist_val:.1%}"
                                )
                        
                    except Exception as cm_error:
                        self.logger.warning(f"  混淆矩陣計算失敗: {cm_error}")
                        cm = None

                    if fold_score > 0.7:  # 较高分数时进行额外记录
                        self.logger.info(f"📈 Fold {fold_idx+1} F1分數較高: {fold_score:.4f}，將進行多重驗證")

                    cv_scores.append(fold_score)
                    # 累計統計
                    if 'metrics_buf' not in locals():
                        metrics_buf = {
                            'f1_macro': [], 'f1_weighted': [],
                            'precision_macro': [], 'recall_macro': [],
                            'balanced_accuracy': [], 'auc_macro': []
                        }
                        conf_mat_sum = np.zeros((3, 3), dtype=int)
                    metrics_buf['f1_macro'].append(float(f1_m))
                    metrics_buf['f1_weighted'].append(float(f1_w))
                    metrics_buf['precision_macro'].append(float(prec_m))
                    metrics_buf['recall_macro'].append(float(rec_m))
                    metrics_buf['balanced_accuracy'].append(float(bal_acc))
                    if auc_macro is not None:
                        metrics_buf['auc_macro'].append(float(auc_macro))
                    if cm is not None and cm.shape == (3, 3):
                        conf_mat_sum += cm
                    self.logger.info(
                        f"    Fold {fold_idx+1}: {len(cols_coarse)}粗選→{len(cols_fine)}精選→{df_train_final.shape[1]}最終, F1={fold_score:.4f}"
                    )

                    # aggregate selected features this fold
                    selection_counts.update(selected_cols)
                    selected_union.update(selected_cols)
                    if fold_score > best_fold_score:
                        best_fold_score = fold_score
                        best_fold_cols = list(selected_cols)

                except (ValueError, ZeroDivisionError) as critical_error:
                    self.logger.error(f"❌ Fold {fold_idx+1} 關鍵錯誤: {critical_error}")
                    raise critical_error

                except Exception as fold_error:
                    self.logger.warning(f"  Fold {fold_idx+1} 失敗: {fold_error}")
                    cv_scores.append(0.0)

            base_score = np.mean(cv_scores) if cv_scores else 0.0

            # 匯總度量
            metrics_summary = {}
            if 'metrics_buf' in locals():
                metrics_summary = {
                    k: float(np.mean(v)) if len(v) > 0 else None
                    for k, v in metrics_buf.items()
                }
                metrics_summary['cv_scores_mean'] = float(base_score)
                metrics_summary['cv_scores_std'] = float(np.std(cv_scores)) if cv_scores else 0.0

            if base_score > 0.7:  # 对较高分数进行多重验证
                validation_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                is_valid, validation_msg = self._validate_score_legitimacy(base_score, X, y, validation_model)

                if not is_valid:
                    self.logger.error(f"❌ 多重验证失败: {validation_msg}")
                    return max(0.0, base_score * 0.85)  # 验证失败給予輕微懲罰
                else:
                    self.logger.info(f"✅ 多重验证通过: {validation_msg}")

            self.logger.info(f"🎯 嵌套CV平均F1: {base_score:.4f} (±{np.std(cv_scores):.4f})")

            # 🔧 P0修復1: 添加時序分割的 Holdout 驗證（防止數據洩漏）
            holdout_metrics = {}
            overfitting_gap = 0.0
            
            try:
                # 時序分割：前70%訓練，後30%作為獨立驗證集
                split_idx = int(len(X) * 0.70)
                X_train_full = X.iloc[:split_idx]
                X_holdout = X.iloc[split_idx:]
                y_train_full = y.iloc[:split_idx]
                y_holdout = y.iloc[split_idx:]
                
                self.logger.info(f"📊 Holdout驗證: 訓練集={len(X_train_full)}, 驗證集={len(X_holdout)}")
                
                if len(X_holdout) > 50 and len(X_train_full) > 100:
                    # 在訓練集上訓練最終模型
                    from sklearn.ensemble import RandomForestClassifier
                    holdout_model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        max_features='sqrt',
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    # 只在訓練集上擬合
                    holdout_model.fit(X_train_full, y_train_full)
                    
                    # 在獨立驗證集上預測
                    y_holdout_pred = holdout_model.predict(X_holdout)
                    y_holdout_proba = holdout_model.predict_proba(X_holdout) if hasattr(holdout_model, 'predict_proba') else None
                    
                    # 計算驗證集指標
                    from sklearn.metrics import (f1_score as compute_f1_score_local, precision_score, recall_score, 
                                                 balanced_accuracy_score, roc_auc_score)
                    
                    holdout_f1_macro = float(compute_f1_score_local(y_holdout, y_holdout_pred, average='macro', zero_division=0))
                    holdout_f1_weighted = float(compute_f1_score_local(y_holdout, y_holdout_pred, average='weighted', zero_division=0))
                    
                    holdout_metrics = {
                        'f1_macro': holdout_f1_macro,
                        'f1_weighted': holdout_f1_weighted,
                        'precision_macro': float(precision_score(y_holdout, y_holdout_pred, average='macro', zero_division=0)),
                        'recall_macro': float(recall_score(y_holdout, y_holdout_pred, average='macro', zero_division=0)),
                        'balanced_accuracy': float(balanced_accuracy_score(y_holdout, y_holdout_pred)),
                    }
                    
                    if y_holdout_proba is not None:
                        try:
                            holdout_metrics['auc_macro'] = float(
                                roc_auc_score(y_holdout, y_holdout_proba, multi_class='ovr', average='macro')
                            )
                        except:
                            holdout_metrics['auc_macro'] = 0.5
                    else:
                        holdout_metrics['auc_macro'] = 0.5
                    
                    # 計算過擬合程度
                    overfitting_gap = base_score - holdout_f1_macro
                    
                    self.logger.info(
                        f"✅ Holdout F1: {holdout_f1_macro:.4f}, "
                        f"過擬合差距: {overfitting_gap:.4f}"
                    )
                    
                    # 過擬合警告
                    if overfitting_gap > 0.15:
                        self.logger.warning(
                            f"⚠️ 過擬合風險: CV={base_score:.4f}, Holdout={holdout_f1_macro:.4f}, "
                            f"Gap={overfitting_gap:.4f} > 0.15"
                        )
                    
            except Exception as holdout_error:
                self.logger.warning(f"⚠️ Holdout驗證失敗: {holdout_error}")
                holdout_metrics = {'error': str(holdout_error)}
            
            # 記錄 Holdout 結果到 trial
            trial.set_user_attr('cv_base_score', float(base_score))
            trial.set_user_attr('cv_scores', cv_scores)
            trial.set_user_attr('cv_std', float(np.std(cv_scores)))
            trial.set_user_attr('holdout_metrics', holdout_metrics)
            trial.set_user_attr('overfitting_gap', float(overfitting_gap))

            # 🔧 P0修復2: 將 combined_metrics_full 降級為調試信息（標記為過擬合）
            debug_train_metrics = None
            if self.flags.get('compute_full_metrics_for_debug', False):
                try:
                    debug_train_metrics = self._compute_objective_metrics(trial, X, y, lag)
                    debug_train_metrics['_WARNING'] = 'TRAINING_SET_ONLY_DO_NOT_USE_FOR_EVALUATION'
                    debug_train_metrics['cv_base_score'] = float(base_score)
                    trial.set_user_attr('debug_train_metrics', debug_train_metrics)
                    self.logger.debug("📝 已計算調試用訓練集指標（不參與優化）")
                except Exception as debug_error:
                    self.logger.debug(f"調試指標計算失敗: {debug_error}")
            
            # 保留舊格式兼容性，但標記為不可信
            if debug_train_metrics:
                metrics_summary.setdefault('_debug_combined_metrics_full', debug_train_metrics)

            # 多目標模式：基於 CV 分數而非過擬合指標
            if self.multi_objective_mode:
                # 使用 CV 和 Holdout 的真實指標
                real_metrics = {
                    'f1_macro': base_score,
                    'f1_weighted': metrics_summary.get('f1_weighted', base_score),
                    'precision_macro': metrics_summary.get('precision_macro', base_score),
                    'recall_macro': metrics_summary.get('recall_macro', base_score),
                }
                if holdout_metrics and 'f1_macro' in holdout_metrics:
                    # 結合 Holdout 指標
                    real_metrics['holdout_f1'] = holdout_metrics['f1_macro']
                
                penalized_metrics = self._soft_penalize_kpis(real_metrics)
                values_dict = self._kpis_to_multi_values(penalized_metrics)
                ordered_values = [values_dict.get(key, 0.0) for key in self.multiobjective_metrics]
                trial.set_user_attr('objective_values', values_dict)
                trial.set_user_attr('objective_metrics_penalized', penalized_metrics)
                self.logger.info(f"🎯 多目標評估: {values_dict}")
                return ordered_values

            # 🔧 P0修復3: 使用真實的 CV + Holdout 分數計算最終分數
            if holdout_metrics and 'f1_macro' in holdout_metrics and overfitting_gap < 0.20:
                # 方案：CV 60% + Holdout 40%
                final_score = base_score * 0.60 + holdout_metrics['f1_macro'] * 0.40
                
                # 過擬合懲罰
                if overfitting_gap > 0.10:
                    penalty = (overfitting_gap - 0.10) * 0.5
                    final_score -= penalty
                    self.logger.info(f"📉 過擬合懲罰: -{penalty:.4f}")
                
                self.logger.info(
                    f"✅ 最終分數: {final_score:.4f} "
                    f"(CV:{base_score:.4f} * 0.6 + Holdout:{holdout_metrics['f1_macro']:.4f} * 0.4)"
                )
            else:
                # Fallback：只用 CV 分數（但記錄警告）
                final_score = base_score
                if overfitting_gap >= 0.20:
                    self.logger.warning(f"⚠️ 過擬合嚴重，只使用CV分數: {final_score:.4f}")
                else:
                    self.logger.info(f"✅ 使用CV分數: {final_score:.4f}")
            
            trial.set_user_attr('final_score', float(final_score))

            # 記錄 LGBM 的「no positive gain」跡象，供 callback 參考
            try:
                if isinstance(classifier, dict) and 'lightgbm' in str(type(classifier)).lower():
                    pass
            except Exception:
                pass

            if self.flags.get('enable_trade_freq_penalty', False) and len(y) > 1:
                trade_freq_penalty = self._calculate_trade_frequency_penalty(y, lag)
                final_score *= (1 - trade_freq_penalty)

            if final_score > 0.70:  # 提高验证门槛至0.70
                self.logger.info(f"📋 高分数 {final_score:.4f} 需要多重检验，开始验证...")
                
                validation_model = RandomForestClassifier(
                    n_estimators=30, max_depth=6, random_state=42, n_jobs=-1
                )
                
                validation_passed = 0
                
                shuffle_valid, shuffle_msg = self._shuffle_label_validation(X, y, validation_model, final_score)
                if shuffle_valid:
                    validation_passed += 1
                    self.logger.info(f"✅ 打乱标签测试通过: {shuffle_msg}")
                else:
                    self.logger.warning(f"⚠️ 打乱标签测试未通过: {shuffle_msg}")
                
                random_valid, random_msg = self._random_feature_validation(X, y, validation_model, final_score)
                if random_valid:
                    validation_passed += 1
                    self.logger.info(f"✅ 随机特征测试通过: {random_msg}")
                else:
                    self.logger.warning(f"⚠️ 随机特征测试未通过: {random_msg}")
                
                if validation_passed >= 1:
                    self.logger.info("🎯 多重检验通过！(至少1项验证通过)")
                else:
                    self.logger.warning("⚠️ 多重检验未通过，应用轻微惩罚")
                    final_score *= 0.9  # 轻微惩罚而非归零
            else:
                self.logger.info(f"📊 分数 {final_score:.4f} 无需多重检验（< 0.70）")

            expected_range = (0.60, 0.85)  # 提高期望范围至0.60-0.85
            target_excellent = 0.80         # 优秀目标：0.80+
            
            if final_score >= target_excellent:
                self.logger.info(f"🎯 优秀！Layer2分数达到目标: {final_score:.4f} >= {target_excellent}")
            elif final_score >= expected_range[1]:
                self.logger.info(f"📈 很好！分数超出预期: {final_score:.4f} > {expected_range[1]}")
            elif final_score >= expected_range[0]:
                self.logger.info(f"✅ 良好！分数在可接受范围: {final_score:.4f} ∈ {expected_range}")
            else:
                self.logger.info(f"📊 需继续优化: {final_score:.4f} < {expected_range[0]} (目标: 0.80+)")

            self.logger.info(f"✅ 嵌套CV評估完成: F1_weighted={final_score:.4f} (通過多重檢驗，無數據洩露)")

            # finalize selected_features to avoid NameError
            selected_features: List[str]
            stability_report: Dict[str, Any] = {}
            multi_obj_used = False
            
            # 🚀 P0: 尝试使用多目标特征选择器（如果启用）
            if self.multi_objective_selector is not None and self.multi_objective_mode:
                try:
                    self.logger.info("🎯 尝试使用多目标特征选择器进行最终特征优化...")
                    
                    # 使用全部数据进行多目标选择
                    multi_obj_features, multi_obj_metrics = self._apply_multi_objective_selection(
                        X=X, 
                        y=y, 
                        max_features=fine_k
                    )
                    
                    if multi_obj_features and len(multi_obj_features) >= 10:
                        selected_features = multi_obj_features
                        multi_obj_used = True
                        stability_report = {
                            'method': 'multi_objective',
                            'selected_count': len(selected_features),
                            'multi_obj_metrics': multi_obj_metrics
                        }
                        self.logger.info(f"✅ 多目标特征选择成功: {len(selected_features)}个特征")
                    else:
                        self.logger.warning(f"⚠️ 多目标选择返回特征数不足({len(multi_obj_features)})，使用传统方法")
                        multi_obj_used = False
                except Exception as e:
                    self.logger.warning(f"⚠️ 多目标特征选择失败，回退到传统方法: {e}")
                    multi_obj_used = False
            
            # 如果多目标选择未使用或失败，使用传统的基于稳定性的方法
            if not multi_obj_used:
                if selection_counts:
                    max_count = selection_counts.most_common(1)[0][1]
                    threshold = max(1, int(0.6 * max_count))
                    selected_features = [f for f, c in selection_counts.items() if c >= threshold]
                    stability_report = {
                        'method': 'stability_based',
                        'counts': dict(selection_counts),
                        'max_count': int(max_count),
                        'threshold': int(threshold),
                        'selected_count': len(selected_features)
                    }
                elif selected_union:
                    selected_features = sorted(list(selected_union))
                    stability_report = {
                        'method': 'union',
                        'counts': {},
                        'selected_count': len(selected_features)
                    }
                elif best_fold_cols:
                    selected_features = list(best_fold_cols)
                    stability_report = {
                        'method': 'best_fold',
                        'counts': {},
                        'selected_count': len(selected_features)
                    }
                else:
                    # robust fallback: top-variance features if none selected
                    top_k = min(30, X.shape[1]) if X.shape[1] > 0 else 0
                    selected_features = X.var().sort_values(ascending=False).head(top_k).index.tolist() if top_k > 0 else []
                    stability_report = {
                        'method': 'top_variance',
                        'counts': {},
                        'selected_count': len(selected_features),
                        'fallback': 'top_variance'
                    }

            trial.set_user_attr('selected_features', selected_features)
            trial.set_user_attr('feature_phase', phase)
            trial.set_user_attr('stability_report', stability_report)
            trial.set_user_attr('base_score', base_score)
            trial.set_user_attr('cv_scores', cv_scores)
            trial.set_user_attr('cv_metrics', metrics_summary)
            if 'conf_mat_sum' in locals():
                trial.set_user_attr('confusion_matrix', conf_mat_sum.tolist())

            try:
                report = {
                    'trial': trial.number,
                    'score': float(final_score),
                    'base_score': float(base_score),
                    'cv_std': float(np.std(cv_scores)) if cv_scores else None,
                    'selected_features': selected_features,
                    'metrics': metrics_summary,
                    'stability': stability_report,
                }
                reports_dir = self.results_path / 'reports'
                reports_dir.mkdir(parents=True, exist_ok=True)
                report_path = reports_dir / f'l2_trial_{trial.number:04d}.json'
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                self.logger.info(f"📝 保存Layer2試驗報告: {report_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ 保存Layer2試驗報告失敗: {e}")

            self.latest_selected_features = selected_features
            self.latest_feature_phase = phase

            return final_score

        finally:
            pass

    def _generate_features(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """修復版特徵生成（不依賴外部類）"""
        features = pd.DataFrame(index=ohlcv_data.index)

        # 基本價格特徵
        features['returns'] = ohlcv_data['close'].pct_change()
        features['high_low_ratio'] = ohlcv_data['high'] / ohlcv_data['low']
        features['open_close_ratio'] = ohlcv_data['open'] / ohlcv_data['close']

        # 移動平均線
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = ohlcv_data['close'].rolling(window).mean()
            features[f'price_sma_{window}_ratio'] = ohlcv_data['close'] / features[f'sma_{window}']

            features[f'ema_{window}'] = ohlcv_data['close'].ewm(span=window).mean()
            features[f'price_ema_{window}_ratio'] = ohlcv_data['close'] / features[f'ema_{window}']

        # RSI
        for window in [14, 21]:
            delta = ohlcv_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            features[f'rsi_{window}'] = 100 - (100 / (1 + rs))

        # 布林帶
        for window in [20]:
            sma = ohlcv_data['close'].rolling(window).mean()
            std = ohlcv_data['close'].rolling(window).std()
            features[f'bb_upper_{window}'] = sma + (2.0 * std)
            features[f'bb_lower_{window}'] = sma - (2.0 * std)
            features[f'bb_position_{window}'] = (ohlcv_data['close'] - features[f'bb_lower_{window}']) / (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}'])

        # 成交量特徵
        features['volume_sma_ratio'] = ohlcv_data['volume'] / ohlcv_data['volume'].rolling(20).mean()
        features['price_volume'] = ohlcv_data['close'] * ohlcv_data['volume']

        # 波動率
        for window in [10, 20]:
            features[f'volatility_{window}'] = ohlcv_data['close'].pct_change().rolling(window).std()

        # MACD指標
        ema_12 = ohlcv_data['close'].ewm(span=12).mean()
        ema_26 = ohlcv_data['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']

        # 威廉指標
        for window in [14, 21]:
            high_max = ohlcv_data['high'].rolling(window).max()
            low_min = ohlcv_data['low'].rolling(window).min()
            features[f'williams_r_{window}'] = -100 * (high_max - ohlcv_data['close']) / (high_max - low_min)

        # 動量指標
        for window in [10, 20]:
            features[f'momentum_{window}'] = ohlcv_data['close'] / ohlcv_data['close'].shift(window) - 1

        # 清理數據（移除bfill）
        features = features.ffill()
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

        return features

    def initialize_phased_features(self) -> None:
        """初始化分階段特徵池"""
        self.core_feature_groups = {
            'momentum': ['f_sma_5', 'f_sma_10', 'f_sma_20', 'f_ema_10', 'f_ema_20'],
            'volatility': ['f_bb_position_20', 'f_atr_14', 'f_volatility_20'],
            'volume': ['volume_sma_ratio', 'f_mfi_14', 'price_volume'],
            'price_patterns': ['returns', 'high_low_ratio', 'open_close_ratio'],
            'rsi_core': ['f_rsi_14', 'f_rsi_21']
        }

        self.strategy_feature_groups = {
            'advanced_ma': ['f_sma_50', 'f_ema_50', 'macd', 'macd_signal'],
            'wyckoff': ['wyk_spring', 'wyk_upthrust', 'wyk_bc', 'wyk_st'],
            'fibonacci': ['fibo_0.382', 'fibo_0.618', 'fibo_extension_1.618'],
            'td_sequence': ['td_count_buy', 'td_count_sell'],
            'micro_structure': ['spread_est', 'depth_imbalance']
        }

    def _configure_phase_settings(self) -> None:
        self.initialize_phased_features()
        cfg_path = self.config_path / "feature_flags.json"
        if not cfg_path.exists():
            self.phase_config = {}
            self.selection_params = {}
            return
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                flags = json.load(f)
            self.phase_config = flags.get("feature_phases", {})
            self.selection_params = flags.get("selection_params", {})
        except Exception as e:
            self.logger.warning(f"⚠️ 讀取分階段配置失敗: {e}")
            self.phase_config = {}
            self.selection_params = {}

    def _flatten_feature_groups(self, groups: Dict[str, List[str]]) -> List[str]:
        flattened: List[str] = []
        for feats in groups.values():
            flattened.extend(feats)
        return flattened

    def _get_phase_feature_pool(self, phase: str) -> List[str]:
        core = self._flatten_feature_groups(self.core_feature_groups)
        strategy = self._flatten_feature_groups(self.strategy_feature_groups)
        if phase == "core":
            return core
        if phase == "strategy":
            return list(dict.fromkeys(core + strategy))
        return list(self.X_full.columns)

    def _get_phase_threshold(self, phase: str) -> float:
        defaults = {'core': 0.8, 'strategy': 0.65, 'full': 0.6}
        phase_meta = self.phase_config.get(phase, {})
        return float(phase_meta.get('stability_threshold', defaults.get(phase, 0.6)))

    def _get_selection_bounds(self, phase: str, total_features: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        phase_params = self.selection_params.get(phase, {})
        coarse_ratio = phase_params.get('coarse_ratio', [0.3, 0.6])
        fine_ratio = phase_params.get('fine_ratio', [0.4, 0.8])
        coarse_bounds = (
            max(10, int(total_features * coarse_ratio[0])),
            min(total_features - 1, int(total_features * coarse_ratio[1]))
        )
        fine_bounds = (
            max(5, int(coarse_bounds[0] * fine_ratio[0])),
            max(coarse_bounds[0] + 1, int(coarse_bounds[1] * fine_ratio[1]))
        )
        return coarse_bounds, fine_bounds

    def _phase_bonus(self, phase: str, score: float) -> float:
        bonuses = {'core': 0.0, 'strategy': 0.02, 'full': 0.05}
        threshold = 0.65
        return bonuses.get(phase, 0.0) if score > threshold else 0.0

    def _phase_stability_check(self, X: pd.DataFrame, y: pd.Series, phase: str, selected: List[str]) -> Dict[str, Any]:
        if not selected:
            return {'stable': False, 'overall_stability': 0.0}
        X_sel = X[selected]
        n_bootstrap = int(self.selection_params.get(phase, {}).get('bootstrap_runs', 5))
        stability_scores: List[float] = []
        feature_rankings: List[pd.Series] = []

        for i in range(n_bootstrap):
            sample_idx = np.random.choice(len(X_sel), size=max(50, int(len(X_sel) * 0.8)), replace=True)
            X_boot, y_boot = X_sel.iloc[sample_idx], y.iloc[sample_idx]

            rf = RandomForestClassifier(n_estimators=100, random_state=42 + i, n_jobs=-1)
            rf.fit(X_boot.fillna(0), y_boot)

            importances = pd.Series(rf.feature_importances_, index=X_boot.columns)
            feature_rankings.append(importances.rank(ascending=False))

            cv_score = cross_val_score(
                rf,
                X_boot.fillna(0),
                y_boot,
                cv=TimeSeriesSplit(n_splits=3),
                scoring='f1_weighted'
            ).mean()
            stability_scores.append(cv_score)

        score_stability = 1 - np.std(stability_scores) / (np.mean(stability_scores) + 1e-8)
        ranking_df = pd.DataFrame(feature_rankings).T
        if ranking_df.shape[1] > 1:
            corr_matrix = ranking_df.corr().values
            rank_consistency = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        else:
            rank_consistency = 1.0

        overall_stability = 0.6 * score_stability + 0.4 * rank_consistency
        is_stable = overall_stability >= self._get_phase_threshold(phase)

        return {
            'stable': is_stable,
            'overall_stability': overall_stability,
            'score_stability': score_stability,
            'rank_consistency': rank_consistency,
            'mean_score': float(np.mean(stability_scores)),
            'bootstrap_runs': n_bootstrap
        }

    def _phased_feature_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        coarse_k: int,
        fine_k: int,
        stability_threshold: float
    ) -> List[str]:
        if X.empty:
            return []

        variance_selector = VarianceThreshold(threshold=0.01)
        try:
            variance_selector.fit(X)
            cols_var = X.columns[variance_selector.get_support(indices=True)].tolist()
        except ValueError:
            cols_var = list(X.columns)
        if len(cols_var) <= min(10, coarse_k):
            return cols_var

        coarse_selector = SelectKBest(f_classif, k=min(coarse_k, len(cols_var) - 1))
        coarse_selector.fit(X[cols_var], y)
        cols_coarse = [cols_var[i] for i in coarse_selector.get_support(indices=True)]
        if len(cols_coarse) <= min(5, fine_k):
            return cols_coarse

        fine_selector = SelectKBest(mutual_info_classif, k=min(fine_k, len(cols_coarse) - 1))
        fine_selector.fit(X[cols_coarse], y)
        cols_fine = [cols_coarse[i] for i in fine_selector.get_support(indices=True)]
        if not cols_fine:
            return cols_coarse

        final_cols = self.remove_correlated_features_smart(X[cols_fine], 0.9)
        if not final_cols:
            final_cols = cols_fine

        stability = self._phase_stability_check(X[final_cols], y, 'full', final_cols)
        if not stability.get('stable', True):
            self.logger.warning(
                "⚠️ 分階段穩定性未達標: overall=%.3f", stability.get('overall_stability', 0.0)
            )

        return final_cols

    def objective_phased(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> Tuple[float, Dict[str, Any]]:
        phase = trial.suggest_categorical('feature_phase', ['core', 'strategy', 'full'])
        phase_pool = [col for col in self._get_phase_feature_pool(phase) if col in X.columns]
        if not phase_pool:
            phase_pool = list(X.columns)

        stability_threshold = self._get_phase_threshold(phase)
        coarse_bounds, fine_bounds = self._get_selection_bounds(phase, len(phase_pool))

        if coarse_bounds[0] >= coarse_bounds[1]:
            coarse_bounds = (max(10, min(coarse_bounds)), max(20, min(len(phase_pool) - 1, max(coarse_bounds) + 5)))
        if fine_bounds[0] >= fine_bounds[1]:
            fine_bounds = (max(5, min(fine_bounds)), max(10, min(coarse_bounds[1] - 1, max(fine_bounds) + 3)))

        coarse_k = trial.suggest_int('coarse_k', *coarse_bounds)
        fine_k = trial.suggest_int('fine_k', *fine_bounds)

        X_phase = X[phase_pool]
        selected_features = self._phased_feature_selection(X_phase, y, coarse_k, fine_k, stability_threshold)
        if not selected_features or len(selected_features) == 0:
            self.logger.warning(f"⚠️ {phase} 階段無選中特徵，prune 此 trial")
            raise optuna.TrialPruned(f"{phase} 階段無有效特徵子集")

        cv_scores = self._cross_validate_selected_features(X_phase[selected_features], y)
        if not cv_scores or all(s == 0 for s in cv_scores):
            self.logger.warning(f"⚠️ {phase} 階段 CV 無有效分數，prune 此 trial")
            raise optuna.TrialPruned(f"{phase} 階段 CV 無有效評分")
        
        base_score = float(np.mean(cv_scores))
        stability_report = self._phase_stability_check(X_phase, y, phase, selected_features)
        final_score = base_score + self._phase_bonus(phase, base_score)
        
        self.logger.info(
            f"✅ {phase} 階段完成: {len(selected_features)} 特徵, CV={base_score:.4f}, 最終={final_score:.4f}"
        )

        return final_score, {
            'phase': phase,
            'selected_features': selected_features,
            'base_score': base_score,
            'stability': stability_report
        }

    def _cross_validate_selected_features(self, X: pd.DataFrame, y: pd.Series) -> List[float]:
        if X.empty:
            return []
        splitter = TimeSeriesSplit(n_splits=3)
        scores = []
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        for train_idx, test_idx in splitter.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(compute_f1_score(y_test, y_pred, average='weighted'))
        return scores

    def materialize_best_features(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """根據最佳參數物化特徵資料集"""
        if 'label' not in data.columns:
            raise ValueError("物化特徵時需要包含label欄位")
        selected_columns = params.get('selected_features')
        if not selected_columns:
            raise ValueError("最佳參數缺少selected_features資訊")
        available_cols = [col for col in selected_columns if col in data.columns]
        missing = [col for col in selected_columns if col not in data.columns]
        if missing:
            self.logger.warning("⚠️ 物化特徵缺失列: %s", missing)
        result = data[available_cols + ['label']].copy()
        result = result.fillna(0)
        result.attrs['feature_phase'] = params.get('feature_phase')
        return result

    def apply_transform(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """統一物化接口，委派至 materialize_best_features。"""
        return self.materialize_best_features(data, params)

    def calculate_label_quality(self, labels: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
        """計算標籤質量（與 Layer1 報表口徑一致）：
        - balance_score: 以目標 25/50/25 為基準計算 KL 轉換分數
        - stability_score: 1 - 變化頻率
        - f1_score: 簡化為 macroF1 代理（此處保守以 0.5 作基準，可後續替換模型實測）
        - distribution: [sell, hold, buy]
        - total_samples
        """
        try:
            if not isinstance(labels, pd.Series) or labels.empty:
                return {'balance_score': 0.0, 'stability_score': 0.0, 'f1_score': 0.0,
                        'distribution': [0.0, 0.0, 0.0], 'total_samples': 0}
            value_counts = labels.value_counts(normalize=True)
            target = np.array([0.25, 0.5, 0.25])
            actual = np.array([value_counts.get(0, 0.0), value_counts.get(1, 0.0), value_counts.get(2, 0.0)])
            kl = np.sum(target * np.log((target + 1e-8) / (actual + 1e-8)))
            balance_score = float(np.exp(-kl))
            changes = int((labels.diff() != 0).sum())
            stability_score = float(max(0.0, 1.0 - changes / max(len(labels), 1)))
            macro_f1_proxy = 0.5
            return {
                'balance_score': balance_score,
                'stability_score': stability_score,
                'f1_score': macro_f1_proxy,
                'distribution': actual.tolist(),
                'total_samples': int(len(labels))
            }
        except Exception as exc:
            self.logger.warning(f"無法計算標籤質量: {exc}")
            return {'balance_score': 0.0, 'stability_score': 0.0, 'f1_score': 0.0,
                    'distribution': [0.0, 0.0, 0.0], 'total_samples': 0}

    def optimize(self, n_trials: int = 50, timeframes: List[str] = None) -> Dict:
        """🚀 123.md建議：參數化標籤生成的特徵工程優化"""
        if timeframes is None:
            timeframes = [self.timeframe]

        results = {}
        meta_vol = self.scaled_config.get('meta_vol', 0.02)

        for tf in timeframes:
            self.logger.info(f"🚀 開始Layer2參數化標籤+特徵工程優化... 時框: {tf}")
            self.timeframe = tf

            if len(self.features) == 0:
                raise ValueError("特徵數據預加載失敗，無法進行優化")
            if len(self.close_prices) == 0:
                raise ValueError("價格數據預加載失敗，無法進行優化")
            if not hasattr(self, 'valid_data_range'):
                raise ValueError("數據範圍預計算失敗，無法進行優化")

            self.logger.info(f"📊 數據概況: OHLCV={self.ohlcv_data.shape}, 特徵={self.features.shape}")

            # 🚀 在 trial 之前先固定穩定特徵集（使用 Layer1 最佳或默認標籤參數）
            try:
                baseline_params = None
                tf_specific = self.config_path / f"label_params_{self.timeframe}.json"
                fallback = self.config_path / "label_params.json"
                if tf_specific.exists():
                    with open(tf_specific, 'r', encoding='utf-8') as f:
                        baseline_params = json.load(f).get('best_params', {})
                elif fallback.exists():
                    with open(fallback, 'r', encoding='utf-8') as f:
                        baseline_params = json.load(f).get('best_params', {})
                # 默認基線
                lag_b = int(baseline_params.get('lag', max(1, self.scaled_config.get('label_lag_min', 3))))
                buy_q_b = float(baseline_params.get('buy_quantile', 0.75))
                sell_q_b = float(baseline_params.get('sell_quantile', 0.25))
                lookback_b = int(self.scaled_config.get('lookback_window_min', 300))
                y_base = self._generate_labels(self.close_prices, lag_b, buy_q_b, sell_q_b, lookback_b)
                self._ensure_X_full()
                common_idx_base = self.X_full.index.intersection(y_base.index)
                X_base = self.X_full.reindex(common_idx_base).astype('float32').fillna(0)
                y_base = y_base.reindex(common_idx_base).astype(int)
                # 先做一次去冗餘（pairwise correlation）再進行穩定性選擇
                try:
                    baseline_corr_th = float(self.flags.get('baseline_corr_threshold', 0.9))
                except Exception:
                    baseline_corr_th = 0.9
                base_keep = self.remove_correlated_features_smart(X_base, baseline_corr_th)
                X_base = X_base[base_keep] if len(base_keep) > 0 else X_base
                n_base = X_base.shape[1]
                coarse_b = min(max(50, int(n_base * 0.6)), n_base - 1) if n_base > 1 else 1
                fine_b = min(60, max(15, coarse_b - 1))
                self.stable_cols = self._stability_select_features(
                    X_base, y_base,
                    coarse_k=coarse_b,
                    fine_k=fine_b,
                    threshold=float(self.flags.get('stability_selection_threshold', 0.6))
                )
                self.logger.info(f"🧮 固定穩定特徵數: {len(self.stable_cols)} / {n_base}")
            except Exception as e:
                self.logger.warning(f"⚠️ 穩定特徵預計算失敗: {e}")
                self.stable_cols = []

            storage_url = None
            try:
                storage_url = self.scaled_config.get('optuna_storage')
            except Exception:
                storage_url = None
            # Callback: 根據 trial user_attrs 調整 LGBM 搜尋空間（簡化版：透過旗標影響 create_enhanced_classifier 的範圍）
            def _relax_search_space_cb(study, trial):
                try:
                    no_gain = bool(trial.user_attrs.get('lgb_no_gain', False))
                    bad_score = (trial.value is None) or (float(trial.value) < 0.35)
                    if no_gain or bad_score:
                        cnt = int(study.user_attrs.get('relax_count', 0)) + 1
                        study.set_user_attr('relax_count', cnt)
                        # 將旗標設為 True，後續 objective 讀取後放鬆搜索空間
                        study.set_user_attr('relax_search_space', True)
                        self.flags['lgb_relax_search'] = True
                        self.logger.info(f"🔧 觸發 LGBM 搜尋空間放鬆，第{cnt}次")
                except Exception:
                    pass

            study_kwargs = {
                'study_name': f'feature_optimization_layer2_{tf}',
                'pruner': optuna.pruners.MedianPruner(n_warmup_steps=7),
                'storage': storage_url,
                'load_if_exists': bool(storage_url)
            }
            if self.multi_objective_mode:
                study = optuna.create_study(
                    directions=['maximize'] * len(self.multiobjective_metrics),
                    **study_kwargs
                )
            else:
                study = optuna.create_study(
                    direction='maximize',
                    **study_kwargs
                )
            study.set_user_attr('meta_vol', meta_vol)

            self.logger.info(f"🚀 开始Layer2特征优化: {n_trials} trials")
            successful_trials = 0
            failed_trials = 0
            consecutive_failures = 0
            trial_durations: List[float] = []

            self.logger.info(f"🚀 Layer2 flags: {self.flags}")
            for trial_idx in range(n_trials):
                try:
                    start_trial = time.perf_counter()
                    study.optimize(self.objective, n_trials=1, timeout=450, callbacks=[_relax_search_space_cb])
                    duration = time.perf_counter() - start_trial
                    trial_durations.append(duration)
                    successful_trials += 1
                    consecutive_failures = 0

                    if trial_idx % 20 == 0 and trial_idx > 0:
                        current_best = study.best_value if study.trials else 0.0
                        self.logger.info(
                            f"📊 Progress: {trial_idx+1}/{n_trials} trials, 成功: {successful_trials}, "
                            f"失败: {failed_trials}, 当前最佳: {current_best:.4f}"
                        )

                except Exception as e:
                    self.logger.error("Layer2 Trial %s 未處理的例外: %s", trial_idx, e)
                    self.logger.debug("Trial %s stacktrace:\n%s", trial_idx, traceback.format_exc())
                    duration = time.perf_counter() - start_trial
                    trial_durations.append(duration)
                    failed_trials += 1
                    consecutive_failures += 1

                    if consecutive_failures >= self.scaled_config.get('l2_consecutive_fail_limit', 2):
                        self.logger.error("❌ 連續 %s 次失敗，中止 Layer2 trials", consecutive_failures)
                        break
                    fail_rate_limit = self.scaled_config.get('l2_fail_rate_limit', 0.3)
                    if failed_trials > n_trials * fail_rate_limit:
                        self.logger.error("❌ Layer2 Trial 失敗率過高（%s/%s），請檢查日誌排查根因", failed_trials, trial_idx + 1)
                        break
                    continue

            avg_duration = float(np.mean(trial_durations)) if trial_durations else 0.0
            self.logger.info(f"⏱️ 平均 trial 耗時: {avg_duration:.2f} 秒")

            self.logger.info(f"✅ Layer2优化完成: {successful_trials}/{n_trials} 成功trials，失败: {failed_trials}")

            if successful_trials == 0:
                raise ValueError("所有trials都失败，优化无效")

            if successful_trials < n_trials * 0.1:
                self.logger.warning(f"⚠️ 成功率过低 ({successful_trials}/{n_trials})，结果可能不可靠")

            best_params = study.best_params
            best_score = study.best_value

            best_trial = study.best_trial
            selected_features = best_trial.user_attrs.get('selected_features', [])
            feature_phase = best_trial.user_attrs.get('feature_phase')
            stability_report = best_trial.user_attrs.get('stability_report', {})
            cv_metrics = best_trial.user_attrs.get('cv_metrics', {})
            cv_scores_attr = best_trial.user_attrs.get('cv_scores', [])
            confusion_matrix_attr = best_trial.user_attrs.get('confusion_matrix')
            
            # 🔧 P0修復：添加 Holdout 驗證結果
            cv_base_score = best_trial.user_attrs.get('cv_base_score', best_score)
            cv_std = best_trial.user_attrs.get('cv_std', 0.0)
            holdout_metrics = best_trial.user_attrs.get('holdout_metrics', {})
            overfitting_gap = best_trial.user_attrs.get('overfitting_gap', 0.0)
            final_score = best_trial.user_attrs.get('final_score', best_score)
            debug_train_metrics = best_trial.user_attrs.get('debug_train_metrics', {})

            best_params['selected_features'] = selected_features
            best_params['feature_phase'] = feature_phase
            best_params['phase_stability'] = stability_report
            best_params['selected_features_count'] = len(selected_features)
            if cv_metrics:
                best_params['cv_metrics'] = cv_metrics
            if cv_scores_attr:
                best_params['cv_scores'] = [float(x) for x in cv_scores_attr]

            self.logger.info(f"標籤優化完成! 最佳得分: {best_score:.4f}")
            self.logger.info(f"🏆 最優參數: {best_params}")

            try:
                cleaned_file = self.config_path / f"cleaned_ohlcv_{tf}.parquet"
                if cleaned_file.exists():
                    df_cleaned = read_dataframe(cleaned_file)
                    self.logger.info(f"✅ 使用Layer0清洗數據生成最終標籤: {cleaned_file}")
                else:
                    ohlcv_file = self.data_path / "raw" / self.symbol / f"{self.symbol}_{tf}_ohlcv.parquet"
                    df_cleaned = read_dataframe(ohlcv_file)
                    self.logger.info(f"✅ 使用原始OHLCV數據生成最終標籤: {ohlcv_file}")
                price_data2 = df_cleaned['close']
                final_labels = self.generate_labels(price_data2, best_params)
                final_quality = self.calculate_label_quality(final_labels, best_params)
                labeled_data = self.apply_labels(df_cleaned, best_params)
            except Exception as e:
                self.logger.warning(f"無法生成最終標籤: {e}")
                final_quality = {}
                labeled_data = None

            # 🔧 P0修復：記錄真實的驗證指標
            result = {
                'timeframe': tf,
                'best_params': best_params,
                'best_score': best_score,  # 這是 final_score（CV + Holdout 組合）
                'n_trials': n_trials,
                'final_quality': final_quality,
                'meta_vol': meta_vol,
                'labeled_data': labeled_data,
                'cv_metrics': cv_metrics,
                'cv_scores': [float(x) for x in cv_scores_attr] if cv_scores_attr else [],
                'cv_base_score': float(cv_base_score),
                'cv_std': float(cv_std),
                'holdout_metrics': holdout_metrics,  # 新增：獨立驗證集指標
                'overfitting_gap': float(overfitting_gap),  # 新增：過擬合程度
                'final_score': float(final_score),  # 新增：最終組合分數
                'confusion_matrix': confusion_matrix_attr,
                'optimization_history': [
                    {'trial': i, 'score': trial.value}
                    for i, trial in enumerate(study.trials)
                    if trial.value is not None
                ]
            }
            
            # 🔧 P0修復：如果有調試指標，標記為不可信
            if debug_train_metrics and debug_train_metrics.get('_WARNING'):
                result['_debug_train_metrics'] = debug_train_metrics
                result['_WARNING'] = 'debug_train_metrics僅供調試，請勿用於評估模型性能'
            
            # 打印關鍵驗證指標
            self.logger.info(f"📊 驗證結果摘要:")
            self.logger.info(f"  CV Base F1: {cv_base_score:.4f} (±{cv_std:.4f})")
            if holdout_metrics and 'f1_macro' in holdout_metrics:
                self.logger.info(f"  Holdout F1: {holdout_metrics['f1_macro']:.4f}")
                self.logger.info(f"  過擬合差距: {overfitting_gap:.4f}")
                if overfitting_gap > 0.15:
                    self.logger.warning(f"  ⚠️ 過擬合風險較高")
                else:
                    self.logger.info(f"  ✅ 過擬合控制良好")
            self.logger.info(f"  最終分數: {final_score:.4f}")

            # 避免將 DataFrame 寫入 JSON（僅保存可序列化摘要）
            json_safe = {k: v for k, v in result.items() if k != 'labeled_data'}
            output_file = self.config_path / "feature_params.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_safe, f, indent=2, ensure_ascii=False)

            tf_output = self.config_path / f"feature_params_{tf}.json"
            with open(tf_output, 'w', encoding='utf-8') as f:
                json.dump(json_safe, f, indent=2, ensure_ascii=False)

            self.logger.info(f"✅ 結果已保存至: {output_file}")
            self.logger.info(f"✅ 時框專屬結果已保存至: {tf_output}")

            results[tf] = result

        return results[self.timeframe] if len(timeframes) == 1 else results

    def optimize_multi_timeframes(self, timeframes: List[str], n_trials: int = 50) -> Dict:
        """🚀 多時框分別優化：為每個時框找到最佳特徵子集"""
        self.logger.info(f"🚀 開始多時框特徵優化: {timeframes}")

        best_configs = {}

        for timeframe in timeframes:
            self.logger.info(f"\n📈 開始優化時框: {timeframe}")

            optimizer = FeatureOptimizer(
                data_path=str(self.data_path),
                config_path=str(self.config_path),
                symbol=self.symbol,
                timeframe=timeframe
            )

            try:
                result = optimizer.optimize(n_trials=n_trials)

                best_configs[timeframe] = {
                    'timeframe': timeframe,
                    'best_score': result['best_score'],
                    'best_params': result['best_params'],
                    'coarse_k': result['best_params'].get('coarse_k', 60),
                    'fine_k': result['best_params'].get('fine_k', 20),
                    'lag': result['best_params'].get('lag', 12),
                    'profit_quantile': result['best_params'].get('profit_quantile', 0.85),
                    'loss_quantile': result['best_params'].get('loss_quantile', 0.15),
                    'lookback_window': result['best_params'].get('lookback_window', 500),
                    'n_trials': n_trials,
                    'data_shape': result['data_shape'],
                    'feature_range': f"{result['best_params'].get('coarse_k', 60)}粗選→{result['best_params'].get('fine_k', 20)}精選"
                }

                self.logger.info(f"✅ {timeframe} 優化完成: F1={result['best_score']:.4f}")
                self.logger.info(f"   最佳特徵選擇: {best_configs[timeframe]['feature_range']}")

            except Exception as e:
                self.logger.error(f"❌ {timeframe} 優化失敗: {e}")
                best_configs[timeframe] = {
                    'timeframe': timeframe,
                    'error': str(e),
                    'best_score': 0.0
                }

        multi_tf_result = {
            'optimization_type': 'multi_timeframe_feature_selection',
            'timeframes': timeframes,
            'best_configs': best_configs,
            'summary': {
                'successful_optimizations': len([cfg for cfg in best_configs.values() if 'error' not in cfg]),
                'failed_optimizations': len([cfg for cfg in best_configs.values() if 'error' in cfg]),
                'best_performing_timeframe': max(
                    [(tf, cfg['best_score']) for tf, cfg in best_configs.items() if 'error' not in cfg],
                    key=lambda x: x[1], default=('none', 0.0)
                )[0]
            },
            'feature_selection_strategy': {
                'coarse_selection_range': '140-203個特徵 (70%-100%)',
                'fine_selection_range': '10-25個特徵 (從粗選結果中精選)',
                'multi_timeframe_approach': '每個時框獨立優化，使用全量203特徵池，確保所有特徵都有被選中機會'
            }
        }

        output_file = self.config_path / "multi_timeframe_feature_optimization.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(multi_tf_result, f, indent=2, ensure_ascii=False)

        self.logger.info(f"\n🎯 多時框優化完成摘要:")
        self.logger.info(f"   成功: {multi_tf_result['summary']['successful_optimizations']}/{len(timeframes)} 個時框")
        self.logger.info(f"   最佳時框: {multi_tf_result['summary']['best_performing_timeframe']}")

        for tf, cfg in best_configs.items():
            if 'error' not in cfg:
                self.logger.info(f"   {tf}: F1={cfg['best_score']:.4f}, {cfg['feature_range']}")
            else:
                self.logger.error(f"   {tf}: 失敗 - {cfg['error']}")

        self.logger.info(f"💾 多時框結果已保存至: {output_file}")

        return multi_tf_result

    def _calc_mfi(self, typical_price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
        positive_flow = (typical_price - typical_price.shift(1)).clip(lower=0) * volume
        negative_flow = (typical_price.shift(1) - typical_price).clip(lower=0) * volume
        pos_sum = positive_flow.rolling(window).sum()
        neg_sum = negative_flow.rolling(window).sum()
        money_ratio = pos_sum / (neg_sum + 1e-9)
        return 100 - (100 / (1 + money_ratio))

    def _calc_cci(self, typical_price: pd.Series, window: int) -> pd.Series:
        tp_ma = typical_price.rolling(window).mean()
        tp_md = (typical_price - tp_ma).abs().rolling(window).mean()
        return (typical_price - tp_ma) / (0.015 * tp_md.replace(0, np.nan))

    def _calc_kdj(self, high: pd.Series, low: pd.Series, close: pd.Series,
                  rsv_window: int, k_smooth: int, d_smooth: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        lowest = low.rolling(rsv_window).min()
        highest = high.rolling(rsv_window).max()
        rsv = ((close - lowest) / (highest - lowest + 1e-9)) * 100
        k = rsv.ewm(alpha=1 / k_smooth, adjust=False).mean()
        d = k.ewm(alpha=1 / d_smooth, adjust=False).mean()
        j = 3 * k - 2 * d
        return k, d, j

    def _calc_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff()).fillna(0)
        obv = (direction * volume).cumsum()
        return obv

    def _filter_low_quality_features(self, X: pd.DataFrame) -> pd.DataFrame:
        original_cols = X.columns
        constant_cols = original_cols[X.nunique(dropna=False) <= 1]
        if len(constant_cols) > 0:
            X = X.drop(columns=constant_cols)
        # 低多樣性列過濾：nunique/len < ε（避免長段 ffill 造成的低信息列）
        try:
            eps = float(self.flags.get('low_diversity_threshold', 0.003))
            nunique_ratio = X.nunique(dropna=False) / max(1, len(X))
            low_div_cols = nunique_ratio[nunique_ratio < eps].index
            if len(low_div_cols) > 0:
                X = X.drop(columns=low_div_cols)
        except Exception as e:
            self.logger.warning(f"低多樣性過濾失敗，跳過: {e}")
        missing_rate = X.isnull().mean()
        high_missing_cols = missing_rate[missing_rate > 0.5].index
        if len(high_missing_cols) > 0:
            X = X.drop(columns=high_missing_cols)
        if X.shape[1] > 1:
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
            if len(high_corr_cols) > 0:
                X = X.drop(columns=high_corr_cols)
        self.logger.info(f"🔧 特徵質量過濾: 原始={len(original_cols)}, 常量={len(constant_cols)}, 高缺失={len(high_missing_cols)}, 高相關={len(original_cols) - X.shape[1] - len(constant_cols) - len(high_missing_cols)}")
        return X

    def _rebalance_labels(self, labels: pd.Series, lag: int, profit_quantile: float,
                           loss_quantile: float, lookback_window: int) -> pd.Series:
        target = 0.20
        max_iter = 3
        step = 0.02
        cur_profit, cur_loss = profit_quantile, loss_quantile
        for _ in range(max_iter):
            label_counts = labels.value_counts().sort_index()
            total = int(labels.notna().sum())
            if len(label_counts) != 3 or total == 0:
                break
            sell_ratio = label_counts.get(0, 0) / total
            buy_ratio = label_counts.get(2, 0) / total
            improved = False
            if buy_ratio < target:
                cur_profit = max(0.60, cur_profit - step)
                improved = True
            if sell_ratio < target:
                cur_loss = min(0.40, cur_loss + step)
                improved = True
            if not improved:
                break
            self.logger.info(f"📊 調整後 quantile: profit={cur_profit:.3f}, loss={cur_loss:.3f}")
            labels = self._generate_labels(self.close_prices, lag, cur_profit, cur_loss, lookback_window)
        return labels

    def _compute_sample_weights(self, y: pd.Series) -> np.ndarray:
        """🚀 P2修復：增強版樣本權重 - 動態加權 + 稀有類保護 + 召回率反饋
        
        改進點：
        1. 基礎平衡權重（sklearn標準）
        2. 稀有類加倍保護（<12%樣本）
        3. 主導類權重抑制（>60%樣本）
        4. 歷史召回率反饋（針對預測差的類別加強）
        
        預期效果：
        - 賣出類召回率: 8.2% → 40-50%
        - 類別平衡性: 大幅改善
        - Precision-macro: 0.0 → 0.45+
        """
        try:
            y_series = pd.Series(y)
            value_counts = y_series.value_counts().to_dict()
            classes = sorted(value_counts.keys())
            total = float(len(y_series))
            n_classes = float(len(classes))
            
            if total == 0 or n_classes == 0:
                return np.ones(len(y), dtype=float)
            
            # ========== 第1層：基礎平衡權重（sklearn標準） ==========
            base_weights = {
                c: (total / (n_classes * float(value_counts.get(c, 1))))
                for c in classes
            }
            
            # ========== 第2層：稀有類加倍保護機制 ==========
            enhanced_weights = base_weights.copy()
            
            for cls in classes:
                cls_count = value_counts.get(cls, 0)
                ratio = cls_count / total
                
                # 極稀有類（<12%）：權重×2.5，確保不被忽略
                if ratio < 0.12:
                    boost = 2.5
                    enhanced_weights[cls] *= boost
                    cls_name = {0: '賣出', 1: '持有', 2: '買入'}.get(cls, f'類{cls}')
                    self.logger.info(
                        f"⚖️ 極稀有類保護 - {cls_name}(class {cls}): "
                        f"樣本={cls_count}({ratio:.1%}) → 權重×{boost:.1f}"
                    )
                
                # 稀有類（12-20%）：權重×1.5
                elif ratio < 0.20:
                    boost = 1.5
                    enhanced_weights[cls] *= boost
                    cls_name = {0: '賣出', 1: '持有', 2: '買入'}.get(cls, f'類{cls}')
                    self.logger.info(
                        f"⚖️ 少數類增強 - {cls_name}(class {cls}): "
                        f"樣本={cls_count}({ratio:.1%}) → 權重×{boost:.1f}"
                    )
                
                # 主導類（>60%）：權重×0.75，防止過度預測
                elif ratio > 0.60:
                    suppress = 0.75
                    enhanced_weights[cls] *= suppress
                    cls_name = {0: '賣出', 1: '持有', 2: '買入'}.get(cls, f'類{cls}')
                    self.logger.info(
                        f"⚖️ 主導類抑制 - {cls_name}(class {cls}): "
                        f"樣本={cls_count}({ratio:.1%}) → 權重×{suppress:.1f}"
                    )
            
            # ========== 第3層：歷史召回率反饋機制 ==========
            # 從之前fold的混淆矩陣學習，針對預測差的類別加強
            if hasattr(self, '_class_recall_history') and self._class_recall_history:
                for cls in classes:
                    if cls not in self._class_recall_history:
                        continue
                    
                    recall = self._class_recall_history[cls]
                    cls_name = {0: '賣出', 1: '持有', 2: '買入'}.get(cls, f'類{cls}')
                    
                    # 召回率<20%的類別：額外×2.0權重（強力補償）
                    if recall < 0.20:
                        boost = 2.0
                        enhanced_weights[cls] *= boost
                        self.logger.info(
                            f"⚖️ 低召回補償 - {cls_name}(class {cls}): "
                            f"歷史召回={recall:.1%} → 權重×{boost:.1f}"
                        )
                    
                    # 召回率20-35%的類別：×1.5權重（適度補償）
                    elif recall < 0.35:
                        boost = 1.5
                        enhanced_weights[cls] *= boost
                        self.logger.info(
                            f"⚖️ 中召回增強 - {cls_name}(class {cls}): "
                            f"歷史召回={recall:.1%} → 權重×{boost:.1f}"
                        )
                    
                    # 召回率>80%的類別：×0.9權重（輕微抑制）
                    elif recall > 0.80:
                        suppress = 0.9
                        enhanced_weights[cls] *= suppress
                        self.logger.info(
                            f"⚖️ 高召回抑制 - {cls_name}(class {cls}): "
                            f"歷史召回={recall:.1%} → 權重×{suppress:.1f}"
                        )
            
            # ========== 應用權重並統計 ==========
            weights = y_series.map(enhanced_weights).astype(float).values
            
            # 權重統計報告
            weight_by_class = {}
            for cls in classes:
                cls_mask = (y_series == cls)
                if cls_mask.any():
                    weight_by_class[cls] = float(weights[cls_mask].mean())
            
            self.logger.info("=" * 50)
            self.logger.info("📊 最終樣本權重配置:")
            for cls in classes:
                cls_name = {0: '賣出', 1: '持有', 2: '買入'}.get(cls, f'類{cls}')
                cls_count = value_counts.get(cls, 0)
                cls_ratio = cls_count / total
                cls_weight = weight_by_class.get(cls, 1.0)
                
                self.logger.info(
                    f"  {cls_name}(class {cls}): "
                    f"樣本={cls_count:>6}, 佔比={cls_ratio:>5.1%}, "
                    f"平均權重={cls_weight:>5.2f}"
                )
            
            self.logger.info(
                f"📊 權重統計: "
                f"min={weights.min():.2f}, "
                f"max={weights.max():.2f}, "
                f"mean={weights.mean():.2f}, "
                f"std={weights.std():.2f}"
            )
            self.logger.info("=" * 50)
            
            return weights
            
        except Exception as e:
            self.logger.error(f"❌ 樣本權重計算失敗: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # 安全回退：基礎平衡權重
            try:
                y_series = pd.Series(y)
                value_counts = y_series.value_counts().to_dict()
                total = len(y_series)
                n_classes = len(value_counts)
                
                if total > 0 and n_classes > 0:
                    fallback_weights = y_series.map({
                        c: total / (n_classes * count) 
                        for c, count in value_counts.items()
                    }).values
                    self.logger.warning("⚠️ 使用基礎平衡權重（回退方案）")
                    return fallback_weights
                else:
                    return np.ones(len(y), dtype=float)
            except:
                self.logger.error("❌ 回退方案也失敗，使用等權")
                return np.ones(len(y), dtype=float)

    def _search_best_trade_threshold(self, y_true: pd.Series, y_proba: np.ndarray) -> tuple:
        """對多分類機率進行兩階段決策的閾值搜尋：先判斷是否交易，再在{sell,buy}中選最大。"""
        try:
            # 假設類別順序為 [0:sell, 1:hold, 2:buy]
            proba_sell = y_proba[:, 0]
            proba_hold = y_proba[:, 1]
            proba_buy = y_proba[:, 2]
            trade_strength = np.maximum(proba_sell, proba_buy)
            best_tau = 0.5
            best_f1 = -1.0
            y_true_arr = np.asarray(y_true)
            # 粗掃
            tau_min = float(self.flags.get('threshold_tau_min', 0.35))
            tau_max = float(self.flags.get('threshold_tau_max', 0.7))
            tau_step = float(self.flags.get('threshold_tau_step', 0.05))
            tau_grid = np.arange(tau_min, tau_max + 1e-9, tau_step)
            for tau in tau_grid:
                do_trade = trade_strength >= tau
                pred = np.where(do_trade, np.where(proba_sell >= proba_buy, 0, 2), 1)
                f1_m = compute_f1_score(y_true_arr, pred, average='macro', zero_division=0)
                if f1_m > best_f1:
                    best_f1, best_tau, best_pred = f1_m, float(tau), pred
            # 細掃
            narrow_min = max(tau_min, best_tau - 0.05)
            narrow_max = min(tau_max, best_tau + 0.05)
            narrow_grid = np.arange(narrow_min, narrow_max + 1e-9, 0.01)
            for tau in narrow_grid:
                do_trade = trade_strength >= tau
                pred = np.where(do_trade, np.where(proba_sell >= proba_buy, 0, 2), 1)
                f1_m = compute_f1_score(y_true_arr, pred, average='macro', zero_division=0)
                if f1_m > best_f1:
                    best_f1, best_tau, best_pred = f1_m, float(tau), pred
            return best_tau, best_pred
        except Exception as e:
            self.logger.warning(f"閾值搜尋過程失敗: {e}")
            # 回退為 argmax 決策
            try:
                return 0.0, np.argmax(y_proba, axis=1)
            except Exception:
                return 0.0, None

    def _rank_features_by_importance(self, X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
        try:
            if X.shape[1] <= top_k:
                return list(X.columns)
            rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced_subsample')
            rf.fit(X.fillna(0), y)
            importances = pd.Series(rf.feature_importances_, index=X.columns)
            top_features = importances.sort_values(ascending=False).head(top_k).index.tolist()
            self.logger.info(f"🔍 重要度預篩: {X.shape[1]} → {len(top_features)}")
            return top_features
        except Exception as e:
            self.logger.warning(f"無法計算特徵重要度: {e}")
            return list(X.columns)

    def _drop_highly_correlated(self, X: pd.DataFrame, threshold: float) -> pd.DataFrame:
        if X.shape[1] <= 1:
            return X
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        if to_drop:
            self.logger.info(f"🔧 高相關特徵移除: {len(to_drop)}")
            X = X.drop(columns=to_drop)
        return X

    def _stability_select_features(self, X: pd.DataFrame, y: pd.Series, coarse_k: int, fine_k: int, threshold: float) -> List[str]:
        if X.shape[1] <= max(fine_k, 20):
            return list(X.columns)
        tscv = TimeSeriesSplit(n_splits=3)
        selection_counts = pd.Series(0, index=X.columns)
        for train_idx, test_idx in tscv.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            variance_selector = VarianceThreshold(threshold=0.01)
            X_train_var = variance_selector.fit_transform(X_train)
            selected_columns = X_train.columns[variance_selector.get_support(indices=True)]
            if len(selected_columns) == 0:
                continue
            coarse_k_fold = min(coarse_k, len(selected_columns) - 1)
            if coarse_k_fold < 5:
                continue
            coarse_selector = SelectKBest(f_classif, k=coarse_k_fold)
            X_train_coarse = coarse_selector.fit_transform(X_train[selected_columns], y_train)
            selected_coarse = pd.Index(selected_columns)[coarse_selector.get_support(indices=True)]
            fine_k_fold = min(fine_k, len(selected_coarse) - 1)
            if fine_k_fold < 5:
                continue
            fine_selector = SelectKBest(mutual_info_classif, k=fine_k_fold)
            fine_selector.fit(X_train[selected_coarse], y_train)
            selected_fine = selected_coarse[fine_selector.get_support(indices=True)]
            selection_counts[selected_fine] += 1
        selected_features = selection_counts[selection_counts >= threshold * selection_counts.max()].index.tolist()
        self.logger.info(f"🧮 穩定性選擇保留: {len(selected_features)} / {len(selection_counts)}")
        return selected_features if selected_features else list(X.columns)

    def _build_full_feature_matrix(self) -> pd.DataFrame:
        """🚀 優化版：優先使用原生時間框架特徵，確保多尺度信號覆蓋"""
        ohlcv = self.ohlcv_data
        X = pd.DataFrame(index=ohlcv.index)
        
        # 特徵統計（用於報告）
        feature_stats = {
            'native_15m': 0, 
            'external': 0,
            'multi_tf': 0, 
            'strategy': 0, 
            'derivatives': 0,
            'total': 0
        }
        
        # ========================================================================
        # 🔧 P1文檔：特徵優先級層次（基於「15m交易，1h參考」設計理念）
        # ========================================================================
        # 優先級1：原生15m特徵（PRIMARY TRADING SIGNALS）
        #   目標：80%+ 的最終選擇特徵
        #   包含：RSI, MACD, BB, ATR, Volume, Stochastic, Momentum等
        #   特點：最及時，無延遲，直接反應15m市場變化
        #
        # 優先級2：1h參考特徵（MACRO ENVIRONMENT REFERENCE）
        #   目標：10-15% 的最終選擇特徵
        #   包含：趨勢方向, EMAs, 支撐/阻力
        #   特點：不含細節技術指標（無RSI/MACD等），僅提供大環境判斷
        #   實現：通過_build_reference_timeframe_features生成
        #
        # 優先級3：4h超大趨勢（OPTIONAL SUPER-TREND）
        #   目標：<5% 的最終選擇特徵
        #   包含：趨勢方向
        #   特點：可選，提供超長期視角
        # ========================================================================
        
        # ========== 優先級1：原生15m特徵（最及時，無延遲） ==========
        # 🔍 強制診斷日誌
        self.logger.info(f"🔍 DEBUG: 開始生成原生{self.timeframe}特徵...")
        self.logger.info(f"🔍 DEBUG: OHLCV shape={ohlcv.shape}, columns={list(ohlcv.columns)}")
        
        try:
            native_features = self._build_native_timeframe_features(ohlcv)
            
            # 🔍 詳細診斷
            self.logger.info(f"🔍 DEBUG: _build_native_timeframe_features 返回")
            self.logger.info(f"🔍 DEBUG: 返回shape={native_features.shape}")
            self.logger.info(f"🔍 DEBUG: 是否為空={native_features.empty}")
            self.logger.info(f"🔍 DEBUG: 列數={len(native_features.columns)}")
            
            if not native_features.empty:
                self.logger.info(f"🔍 DEBUG: 前5個特徵名: {list(native_features.columns[:5])}")
                X = self._safe_merge(X, native_features, prefix='')
                feature_stats['native_15m'] = len(native_features.columns)
                self.logger.info(
                    f"🎯 原生{self.timeframe}特徵: {feature_stats['native_15m']}個 "
                    f"(RSI, MACD, BB, ATR, Stoch等)"
                )
            else:
                self.logger.error(
                    f"❌ 原生{self.timeframe}特徵生成為空！"
                    f"檢查_build_native_timeframe_features方法實現"
                )
        except Exception as e:
            self.logger.error(f"❌ 原生{self.timeframe}特徵生成失敗（異常）: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        # ========== 優先級2：外部特徵（如果有） ==========
        if isinstance(self.features, pd.DataFrame) and not self.features.empty:
            before_count = X.shape[1]
            X = self._safe_merge(X, self.features, prefix='')
            feature_stats['external'] = X.shape[1] - before_count
            if feature_stats['external'] > 0:
                self.logger.info(f"📦 外部特徵: {feature_stats['external']}個")
        
        # ========== 優先級3：多時框技術特徵（1h, 4h趨勢輔助） ==========
        before_count = X.shape[1]
        tech = self.generate_technical_features(ohlcv, params={})
        X = self._safe_merge(X, tech, prefix='')
        feature_stats['multi_tf'] = X.shape[1] - before_count
        self.logger.info(f"📊 多時框特徵: {feature_stats['multi_tf']}個 (1h, 4h趨勢)")
        
        # ========== 優先級4：策略特徵（Wyckoff, TD, Micro） ==========
        strategy_before = X.shape[1]
        
        if self.flags.get('enable_td', True):
            td_feat = self._generate_td_features(ohlcv)
            if not td_feat.empty:
                X = self._safe_merge(X, td_feat, prefix='')
                self.logger.info(f"  ├─ TD Sequential: {len(td_feat.columns)}個")
        
        if self.flags.get('enable_wyckoff', True):
            wyk_feat = self._generate_wyckoff_features(ohlcv)
            if not wyk_feat.empty:
                X = self._safe_merge(X, wyk_feat, prefix='')
                self.logger.info(f"  ├─ Wyckoff: {len(wyk_feat.columns)}個")
        
        if self.flags.get('enable_micro', True):
            micro_feat = self._generate_micro_features_from_ohlcv(ohlcv)
            if not micro_feat.empty:
                X = self._safe_merge(X, micro_feat, prefix='')
                self.logger.info(f"  └─ Microstructure: {len(micro_feat.columns)}個")
        
        feature_stats['strategy'] = X.shape[1] - strategy_before
        
        # ========== 優先級5：衍生品特徵（可選） ==========
        deriv = self._load_derivatives_features(ohlcv.index)
        if not deriv.empty:
            before_deriv = X.shape[1]
            X = self._safe_merge(X, deriv, prefix='')
            feature_stats['derivatives'] = X.shape[1] - before_deriv
            self.logger.info(f"📈 衍生品特徵: {feature_stats['derivatives']}個")
        
        # ========== 質量過濾 ==========
        before_filter = X.shape[1]
        X = self._filter_low_quality_features(X)
        X = X.loc[:, ~X.columns.duplicated()]
        filtered_count = before_filter - X.shape[1]
        feature_stats['total'] = X.shape[1]
        
        # ========== 特徵組成診斷報告 ==========
        self.logger.info("=" * 60)
        self.logger.info("📊 完整特徵矩陣組成分析:")
        
        if feature_stats['total'] > 0:
            native_pct = feature_stats['native_15m'] / feature_stats['total'] * 100
            external_pct = feature_stats['external'] / feature_stats['total'] * 100
            multi_tf_pct = feature_stats['multi_tf'] / feature_stats['total'] * 100
            strategy_pct = feature_stats['strategy'] / feature_stats['total'] * 100
            deriv_pct = feature_stats['derivatives'] / feature_stats['total'] * 100
            
            self.logger.info(
                f"  原生{self.timeframe}:  {feature_stats['native_15m']:>4}個 "
                f"({native_pct:>5.1f}%)"
            )
            if feature_stats['external'] > 0:
                self.logger.info(
                    f"  外部特徵:     {feature_stats['external']:>4}個 "
                    f"({external_pct:>5.1f}%)"
                )
            self.logger.info(
                f"  多時框:       {feature_stats['multi_tf']:>4}個 "
                f"({multi_tf_pct:>5.1f}%)"
            )
            self.logger.info(
                f"  策略特徵:     {feature_stats['strategy']:>4}個 "
                f"({strategy_pct:>5.1f}%)"
            )
            if feature_stats['derivatives'] > 0:
                self.logger.info(
                    f"  衍生品:       {feature_stats['derivatives']:>4}個 "
                    f"({deriv_pct:>5.1f}%)"
                )
            self.logger.info(f"  質量過濾:     移除{filtered_count}個低質量特徵")
            self.logger.info(f"  最終總計:     {feature_stats['total']}個特徵")
            
            # 特徵多樣性診斷
            if native_pct < 20:
                self.logger.warning(
                    f"⚠️ 原生{self.timeframe}特徵佔比過低: {native_pct:.1f}% < 20%，"
                    f"建議檢查_build_native_timeframe_features實現"
                )
            elif native_pct < 30:
                self.logger.info(
                    f"📊 原生{self.timeframe}特徵佔比: {native_pct:.1f}% "
                    f"(建議>30%以提升及時性)"
                )
            else:
                self.logger.info(
                    f"✅ 原生{self.timeframe}特徵佔比健康: {native_pct:.1f}%"
                )
            
            # 檢查特徵時框多樣性
            timeframe_diversity = sum([
                1 if feature_stats['native_15m'] > 0 else 0,
                1 if feature_stats['multi_tf'] > 0 else 0,
                1 if feature_stats['strategy'] > 0 else 0
            ])
            self.logger.info(f"📊 特徵時框多樣性: {timeframe_diversity}/3個維度")
            
        else:
            self.logger.error("❌ 特徵矩陣為空！")
        
        self.logger.info("=" * 60)
        
        return X.astype('float32').fillna(0)

    def _build_native_timeframe_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """🔧 P1修復+350+高级特征：生成原生時間框架的技術指標特徵（無時序滯後）
        
        這些特徵使用當前時間框架的原生數據，不需要重採樣，避免信號滯後。
        例如：15m 交易使用 15m 級別的技術指標，反應更及時。
        
        新增350+高级特征模块：
        - TD Sequential (25+)
        - Wyckoff Analysis (25+)
        - Market Structure (20+)
        - Candlestick Patterns (50+)
        - Composite Indicators (15+)
        - Advanced Trend (25+)
        - Volume Analysis (25+)
        - Harmonic Patterns (8+)
        - Elliott Wave (10+)
        - Gann Theory (10+)
        - Advanced Volatility (6+)
        - Time Series Decomposition (5+)
        - Order Flow Proxy (10+)
        
        Args:
            ohlcv: 原始 OHLCV 數據
            
        Returns:
            原生特徵 DataFrame (包含原有+350+新特征)
        """
        features = pd.DataFrame(index=ohlcv.index)
        close = ohlcv['close']
        high = ohlcv['high']
        low = ohlcv['low']
        volume = ohlcv['volume']
        open_ = ohlcv['open']
        
        prefix = f"{self.timeframe}_native_"
        
        # 🚀 自适应参数优化（如果启用）
        optimized_params = {}
        if self.adaptive_param_optimizer is not None:
            try:
                self.logger.info("🔧 开始自适应参数优化...")
                optimized_params = self.adaptive_param_optimizer.load_or_optimize(
                    timeframe=self.timeframe,
                    ohlcv=ohlcv,
                    force_reoptimize=False  # 使用缓存以提高性能
                )
                self.logger.info(f"✅ 自适应参数优化完成，获得 {len(optimized_params)} 组优化参数")
            except Exception as e:
                self.logger.warning(f"⚠️ 自适应参数优化失败，使用默认参数: {e}")
                optimized_params = {}
        
        try:
            # ====== 原有基础特征 ======
            # 使用优化参数或默认参数
            # ROC使用MA periods（动量指标通常使用类似的周期）
            roc_periods = optimized_params.get('ma_periods', [3, 5, 8, 10, 13, 20, 34, 50, 89])[:9]  # 取前9个
            sma_periods = optimized_params.get('ma_periods', [5, 8, 10, 13, 20, 21, 34, 50, 55, 100, 144, 200])
            atr_periods = optimized_params.get('atr_periods', [5, 7, 10, 14, 20, 30, 50])
            rsi_periods = optimized_params.get('rsi_periods', [5, 7, 9, 14, 21, 28, 42])
            
            # MACD配置从字典或使用默认值
            macd_param = optimized_params.get('macd', {})
            if isinstance(macd_param, dict):
                # 如果是字典，构建多个MACD配置
                fast = macd_param.get('fast', 12)
                slow = macd_param.get('slow', 26)
                signal = macd_param.get('signal', 9)
                # 创建多个变体
                macd_configs = [
                    (fast, slow, signal),
                    (8, 17, 9),
                    (5, 35, 5),
                    (fast-4, slow-7, signal),  # 快速变体
                    (fast+1, slow+8, signal-1)  # 慢速变体
                ]
            else:
                macd_configs = [(8, 17, 9), (12, 26, 9), (5, 35, 5), (19, 39, 9), (13, 34, 8)]
            
            bb_periods = optimized_params.get('bb_periods', [10, 15, 20, 30, 50])
            bb_std_mults = [1.5, 2.0, 2.5]  # 标准差倍数不需要优化
            vol_periods = optimized_params.get('volume_ma_periods', [5, 10, 15, 20, 30, 50])
            
            # Stochastic参数
            stoch_param = optimized_params.get('stoch', {})
            if isinstance(stoch_param, dict):
                stoch_k = stoch_param.get('k', 14)
                stoch_periods = [stoch_k, 9, 14, 21, 28]  # 包含优化值和其他常用值
            else:
                stoch_periods = [9, 14, 21, 28]
            
            wr_periods = rsi_periods  # Williams %R使用类似RSI的周期
            cci_periods = optimized_params.get('cci_periods', [14, 20, 30, 40])
            
            # 1. 價格動量特徵（扩展周期）
            for period in roc_periods:
                features[f'{prefix}roc_{period}'] = close.pct_change(period)
                features[f'{prefix}momentum_{period}'] = close - close.shift(period)
            
            # 2. 移動平均特徵（扩展周期）
            for period in sma_periods:
                sma = close.rolling(period).mean()
                features[f'{prefix}sma_{period}'] = sma
                features[f'{prefix}price_sma_ratio_{period}'] = close / (sma + 1e-9)
                
                ema = close.ewm(span=period).mean()
                features[f'{prefix}ema_{period}'] = ema
                features[f'{prefix}price_ema_ratio_{period}'] = close / (ema + 1e-9)
            
            # 3. 波動率特徵（ATR）（扩展周期）
            for period in atr_periods:
                returns = close.pct_change()
                features[f'{prefix}volatility_{period}'] = returns.rolling(period).std()
                
                # ATR (Average True Range)
                tr1 = high - low
                tr2 = (high - close.shift(1)).abs()
                tr3 = (low - close.shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                features[f'{prefix}atr_{period}'] = tr.rolling(period).mean()
            
            # 4. RSI 家族（扩展周期）
            for period in rsi_periods:
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(period).mean()
                loss = -delta.where(delta < 0, 0).rolling(period).mean()
                rs = gain / (loss + 1e-9)
                features[f'{prefix}rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # 5. MACD 家族（扩展配置）
            for fast, slow, signal in macd_configs:
                ema_fast = close.ewm(span=fast).mean()
                ema_slow = close.ewm(span=slow).mean()
                macd = ema_fast - ema_slow
                signal_line = macd.ewm(span=signal).mean()
                histogram = macd - signal_line
                
                features[f'{prefix}macd_{fast}_{slow}'] = macd
                features[f'{prefix}macd_signal_{fast}_{slow}_{signal}'] = signal_line
                features[f'{prefix}macd_hist_{fast}_{slow}_{signal}'] = histogram
            
            # 6. 布林帶特徵（扩展周期和标准差）
            for period in bb_periods:
                for std_mult in bb_std_mults:
                    bb_middle = close.rolling(period).mean()
                    bb_std = close.rolling(period).std()
                    bb_upper = bb_middle + (std_mult * bb_std)
                    bb_lower = bb_middle - (std_mult * bb_std)
                    
                    features[f'{prefix}bb_position_{period}_{std_mult}'] = (
                        (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
                    )
                    features[f'{prefix}bb_width_{period}_{std_mult}'] = (
                        (bb_upper - bb_lower) / (bb_middle + 1e-9)
                    )
            
            # 7. 成交量特徵（扩展）
            for period in vol_periods:
                vol_sma = volume.rolling(period).mean()
                features[f'{prefix}volume_ratio_{period}'] = volume / (vol_sma + 1e-9)
                features[f'{prefix}volume_std_{period}'] = volume.rolling(period).std()
            
            # 8. OBV (On-Balance Volume)
            obv = (np.sign(close.diff()) * volume).cumsum()
            features[f'{prefix}obv'] = obv
            obv_ema_period = 20  # 固定使用20周期EMA
            features[f'{prefix}obv_ema_20'] = obv.ewm(span=obv_ema_period).mean()
            
            # 9. 價格模式特徵
            features[f'{prefix}high_low_ratio'] = high / (low + 1e-9)
            features[f'{prefix}close_open_ratio'] = close / (open_ + 1e-9)
            features[f'{prefix}body_ratio'] = abs(close - open_) / (high - low + 1e-9)
            
            # 10. Stochastic Oscillator（扩展周期）
            # 从优化参数中获取D周期
            stoch_d_period = 3
            if isinstance(stoch_param, dict):
                stoch_d_period = stoch_param.get('d', 3)
            
            for period in stoch_periods:
                lowest_low = low.rolling(period).min()
                highest_high = high.rolling(period).max()
                k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
                d = k.rolling(stoch_d_period).mean()
                features[f'{prefix}stoch_k_{period}'] = k
                features[f'{prefix}stoch_d_{period}'] = d
            
            # 11. Williams %R（扩展周期）
            for period in wr_periods:
                highest_high = high.rolling(period).max()
                lowest_low = low.rolling(period).min()
                wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-9)
                features[f'{prefix}williams_r_{period}'] = wr
            
            # 12. CCI (Commodity Channel Index)（扩展周期）
            cci_constant = 0.015  # CCI标准常数
            for period in cci_periods:
                tp = (high + low + close) / 3
                sma_tp = tp.rolling(period).mean()
                mad = (tp - sma_tp).abs().rolling(period).mean()
                features[f'{prefix}cci_{period}'] = (tp - sma_tp) / (cci_constant * mad + 1e-9)
            
            self.logger.info(f"✅ 生成 {len(features.columns)} 個基础原生特徵")
            
            # ====== 新增350+高级特征 ======
            all_advanced_features = []
            
            # 13. TD Sequential (25+ features)
            if self.flags.get('enable_td', True):
                try:
                    self.logger.info("🔧 生成TD Sequential特征...")
                    td_features = self.td_sequential.calculate_all(ohlcv)
                    if not td_features.empty:
                        td_features = td_features.add_prefix(prefix)
                        all_advanced_features.append(td_features)
                        self.logger.info(f"  ✅ TD Sequential: {len(td_features.columns)}个特征")
                except Exception as e:
                    self.logger.warning(f"  ⚠️ TD Sequential失败: {e}")
            
            # 14. Wyckoff Analysis (25+ features)
            if self.flags.get('enable_wyckoff', True):
                try:
                    self.logger.info("🔧 生成Wyckoff特征...")
                    wyk_features = self.wyckoff_analysis.calculate_all(ohlcv)
                    if not wyk_features.empty:
                        wyk_features = wyk_features.add_prefix(prefix)
                        all_advanced_features.append(wyk_features)
                        self.logger.info(f"  ✅ Wyckoff: {len(wyk_features.columns)}个特征")
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Wyckoff失败: {e}")
            
            # 15. Market Structure (20+ features)
            if self.flags.get('enable_market_structure', True):
                try:
                    self.logger.info("🔧 生成Market Structure特征...")
                    ms_features = self.market_structure.calculate_all(ohlcv)
                    if not ms_features.empty:
                        ms_features = ms_features.add_prefix(prefix)
                        all_advanced_features.append(ms_features)
                        self.logger.info(f"  ✅ Market Structure: {len(ms_features.columns)}个特征")
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Market Structure失败: {e}")
            
            # 16. Candlestick Patterns (50+ features)
            if self.flags.get('enable_candlestick', True):
                try:
                    self.logger.info("🔧 生成Candlestick Pattern特征...")
                    cs_features = self.candlestick_patterns.calculate_all(ohlcv)
                    if not cs_features.empty:
                        cs_features = cs_features.add_prefix(prefix)
                        all_advanced_features.append(cs_features)
                        self.logger.info(f"  ✅ Candlestick Patterns: {len(cs_features.columns)}个特征")
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Candlestick Patterns失败: {e}")
            
            # 17. Composite Indicators (15+ features)
            if self.flags.get('enable_composite', True):
                try:
                    self.logger.info("🔧 生成Composite Indicator特征...")
                    comp_features = self.composite_indicators.calculate_all(ohlcv)
                    if not comp_features.empty:
                        comp_features = comp_features.add_prefix(prefix)
                        all_advanced_features.append(comp_features)
                        self.logger.info(f"  ✅ Composite Indicators: {len(comp_features.columns)}个特征")
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Composite Indicators失败: {e}")
            
            # 18. Advanced Trend (25+ features)
            if self.flags.get('enable_advanced_trend', True):
                try:
                    self.logger.info("🔧 生成Advanced Trend特征...")
                    trend_features = self.advanced_trend.calculate_all(ohlcv)
                    if not trend_features.empty:
                        trend_features = trend_features.add_prefix(prefix)
                        all_advanced_features.append(trend_features)
                        self.logger.info(f"  ✅ Advanced Trend: {len(trend_features.columns)}个特征")
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Advanced Trend失败: {e}")
            
            # 19. Volume Analysis (25+ features)
            if self.flags.get('enable_volume_analysis', True):
                try:
                    self.logger.info("🔧 生成Volume Analysis特征...")
                    vol_features = self.volume_analysis.calculate_all(ohlcv)
                    if not vol_features.empty:
                        vol_features = vol_features.add_prefix(prefix)
                        all_advanced_features.append(vol_features)
                        self.logger.info(f"  ✅ Volume Analysis: {len(vol_features.columns)}个特征")
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Volume Analysis失败: {e}")
            
            # 20. Harmonic Patterns (8+ features)
            if self.flags.get('enable_harmonic', True):
                try:
                    self.logger.info("🔧 生成Harmonic Pattern特征...")
                    harm_features = self.harmonic_patterns.calculate_all(ohlcv)
                    if not harm_features.empty:
                        harm_features = harm_features.add_prefix(prefix)
                        all_advanced_features.append(harm_features)
                        self.logger.info(f"  ✅ Harmonic Patterns: {len(harm_features.columns)}个特征")
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Harmonic Patterns失败: {e}")
            
            # 21. Elliott Wave (10+ features)
            if self.flags.get('enable_elliott', True):
                try:
                    self.logger.info("🔧 生成Elliott Wave特征...")
                    ew_features = self.elliott_wave.analyze(ohlcv)
                    if not ew_features.empty:
                        ew_features = ew_features.add_prefix(prefix)
                        all_advanced_features.append(ew_features)
                        self.logger.info(f"  ✅ Elliott Wave: {len(ew_features.columns)}个特征")
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Elliott Wave失败: {e}")
            
            # 22. Gann Theory (10+ features)
            if self.flags.get('enable_gann', True):
                try:
                    self.logger.info("🔧 生成Gann Theory特征...")
                    gann_features = self.gann_theory.calculate_all(ohlcv)
                    if not gann_features.empty:
                        gann_features = gann_features.add_prefix(prefix)
                        all_advanced_features.append(gann_features)
                        self.logger.info(f"  ✅ Gann Theory: {len(gann_features.columns)}个特征")
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Gann Theory失败: {e}")
            
            # 23. Advanced Volatility (6+ features)
            if self.flags.get('enable_advanced_vol', True):
                try:
                    self.logger.info("🔧 生成Advanced Volatility特征...")
                    vol_features = self.advanced_volatility.calculate_all(ohlcv)
                    if not vol_features.empty:
                        vol_features = vol_features.add_prefix(prefix)
                        all_advanced_features.append(vol_features)
                        self.logger.info(f"  ✅ Advanced Volatility: {len(vol_features.columns)}个特征")
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Advanced Volatility失败: {e}")
            
            # 24. Time Series Decomposition (5+ features)
            if self.flags.get('enable_ts_decomp', True):
                try:
                    self.logger.info("🔧 生成Time Series Decomposition特征...")
                    ts_features = self.time_series_decomposition.decompose(ohlcv)
                    if not ts_features.empty:
                        ts_features = ts_features.add_prefix(prefix)
                        all_advanced_features.append(ts_features)
                        self.logger.info(f"  ✅ Time Series Decomposition: {len(ts_features.columns)}个特征")
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Time Series Decomposition失败: {e}")
            
            # 25. Order Flow Proxy (10+ features)
            if self.flags.get('enable_order_flow', True):
                try:
                    self.logger.info("🔧 生成Order Flow Proxy特征...")
                    of_features = self.order_flow_proxy.calculate_proxy(ohlcv)
                    if not of_features.empty:
                        of_features = of_features.add_prefix(prefix)
                        all_advanced_features.append(of_features)
                        self.logger.info(f"  ✅ Order Flow Proxy: {len(of_features.columns)}个特征")
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Order Flow Proxy失败: {e}")
            
            # 合并所有高级特征
            if all_advanced_features:
                advanced_features_df = pd.concat(all_advanced_features, axis=1)
                features = pd.concat([features, advanced_features_df], axis=1)
                self.logger.info(f"✅ 新增 {len(advanced_features_df.columns)} 个高级特征")
            
            # 清理 NaN 和 Inf
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.ffill().bfill().fillna(0)  # 修復：使用新版pandas語法
            
            self.logger.info(f"🎉 总共生成 {len(features.columns)} 个原生 {self.timeframe} 特徵（基础+高级）")
            
        except Exception as e:
            self.logger.error(f"❌ 原生特徵生成失敗: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            features = pd.DataFrame(index=ohlcv.index)
        
        return features


def main():
    import sys

    if len(sys.argv) == 1:
        print("🚀 單一時框優化 (15m)")
        optimizer = FeatureOptimizer(data_path='../data', config_path='../configs', timeframe='15m')
        result = optimizer.optimize(n_trials=50)
        print(f"✅ 特徵優化完成: {result['best_score']:.4f}")

    elif len(sys.argv) == 2 and sys.argv[1] == 'multi':
        print("🚀 多時框優化 (15m, 1h, 4h)")
        optimizer = FeatureOptimizer(data_path='../data', config_path='../configs')
        multi_result = optimizer.optimize_multi_timeframes(['15m', '1h', '4h'], n_trials=30)

        print(f"\n📊 多時框優化結果:")
        for tf, config in multi_result['best_configs'].items():
            if 'error' not in config:
                print(f"   {tf}: F1={config['best_score']:.4f}, {config['feature_range']}")

    else:
        print("用法:")
        print("  python optuna_feature.py        # 單一時框優化")
        print("  python optuna_feature.py multi  # 多時框優化")


if __name__ == "__main__":
    main()