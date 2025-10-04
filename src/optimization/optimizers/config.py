#!/usr/bin/env python3
"""
配置文件 - Optuna優化系統統一配置管理（改進版）

修復過擬合問題：增加特徵數量，降低模型複雜度，增強正則化
"""

import os
from typing import Dict, Tuple, Any


class OptimizationConfig:
    """Optuna優化系統配置類（改進版）"""

    # 基礎配置
    BASE_CONFIG = {
        "random_state": 42,
        "n_jobs": 2,  # 保守設置，適合VPS環境
        "verbosity": 1,
        "timeout": None,  # 無時間限制
        "cv_folds": 5,
        "test_size": 0.2,
        "validation_method": "TimeSeriesSplit"
    }

    # 時間框架特定配置 - 🔧 基於學術文獻的4階段架構設計
    TIMEFRAME_CONFIG = {
        # 🚀 新增：超高頻時框 (基於HFT研究) - 🔧 標籤平衡優化
        "5m": {
            "lag_range": (1, 6),             # 極短滯後期，捕捉微觀結構
            "threshold_range": (0.0008, 0.0020), # 🔧 降低閾值範圍促進平衡（方案B）
            
            # 🎯 兩階段特徵選擇 (基於文獻：高頻需要更多技術指標)
            "coarse_k_range": (35, 50),     # 粗選：保留35-50個候選特徵
            "fine_k_range": (8, 12),        # 精選：最終8-12個（控制過擬合）
            
            "label_trials": 200,             # 較少試驗（高頻數據計算密集）
            "feature_coarse_trials": 80,     # 粗選試驗數
            "feature_fine_trials": 120,      # 精選試驗數  
            "model_trials": 300,
            
            # 🆕 5m專用標籤平衡配置（超高頻特化）
            "label_balance_override": {
                "quantile_range": (0.20, 0.80),     # 更寬鬆分位數（捕捉微小波動）
                "preferred_method": "quantile",      # 優先分位數方法
                "balance_penalty_weight": 0.4,      # 高不平衡懲罰權重
                "micro_movement_threshold": 0.0005  # 微觀價格移動閾值
            }
        },
        
        # 🔥 高頻時框 (基於日內交易研究) - 🔧 修復標籤平衡問題
        "15m": {
            # 🚀 階段2: 基於盈利的標籤配置
            "lag_range": (4, 20),  # 預測1-5小時後的實際盈利 (4*15分-20*15分)
            "pos_threshold_range": (0.002, 0.015),  # 最小盈利閾值: 0.2%-1.5%（扣除交易成本後）
            "neg_threshold_range": (-0.015, -0.002),  # 最大虧損閾值: -1.5%到-0.2%
            "threshold_range": (0.002, 0.012),  # 綜合閾值範圍
            
            "coarse_k_range": (60, 80),     # 🔧 增加粗選特徵數（利用203特徵）
            "fine_k_range": (20, 30),       # 🔧 最終選擇20-30個特徵
            
            "label_trials": 300,  # 🔧 增加試驗數尋找平衡解
            "feature_coarse_trials": 120,
            "feature_fine_trials": 180,
            "model_trials": 400,
            
            # 🆕 15m專用標籤平衡配置（強化版）
            "label_balance_override": {
                "quantile_range": (0.20, 0.80),     # 🔧 更寬鬆分位數
                "preferred_method": "quantile",      # 🔧 強制quantile方法
                "force_quantile": True,              # 🆕 強制使用quantile
                "balance_penalty_weight": 0.5,      # 🔧 提高懲罰權重
                "target_distribution": [0.25, 0.50, 0.25],  # 🆕 目標分佈
                "min_class_ratio": 0.15,            # 🆕 最小類別比例15%
                "enable_smote": True                 # 🆕 自動啟用SMOTE
            }
        },
        
        # 📈 中頻時框 (基於量化交易最佳實踐) - 🔧 標籤平衡優化
        "1h": {
            "lag_range": (4, 16),
            "threshold_range": (0.0015, 0.0035), # 🔧 降低閾值範圍促進平衡（方案B）
            
            "coarse_k_range": (45, 70),     # 更大候選池（1h數據質量更好）
            "fine_k_range": (15, 25),       # 經典範圍：15-25個特徵
            
            "label_trials": 300,
            "feature_coarse_trials": 120,
            "feature_fine_trials": 200,     # 更多精選試驗
            "model_trials": 400,
            
            # 🆕 1h專用標籤平衡配置（中頻時框最佳實踐）
            "label_balance_override": {
                "quantile_range": (0.25, 0.75),     # 標準平衡分位數
                "preferred_method": "quantile",      # 優先分位數方法
                "balance_penalty_weight": 0.25,     # 中等不平衡懲罰權重
                "trend_sensitivity": 1.2            # 趨勢敏感度因子
            }
        },
        
        # 📊 中長期時框 (基於趨勢跟蹤研究) - 🔧 標籤平衡優化
        "4h": {
            "lag_range": (6, 24),
            "threshold_range": (0.0025, 0.0080), # 🔧 降低閾值範圍促進平衡（方案B）
            
            "coarse_k_range": (35, 50),     # 中等候選池
            "fine_k_range": (12, 18),       # 適中特徵數
            
            "label_trials": 280,
            "feature_coarse_trials": 100,
            "feature_fine_trials": 150,
            "model_trials": 350,
            
            # 🆕 4h專用標籤平衡配置（中長期趨勢特化）
            "label_balance_override": {
                "quantile_range": (0.30, 0.70),     # 趨勢導向分位數
                "preferred_method": "quantile",      # 優先分位數方法
                "balance_penalty_weight": 0.2,      # 較低不平衡懲罰權重
                "trend_confirmation_window": 6      # 趨勢確認窗口（6*4h=24h）
            }
        },
        
        # 📈 長期時框 (基於趨勢投資研究) - 🔧 標籤平衡優化
        "1D": {
            "lag_range": (8, 30),
            "threshold_range": (0.0050, 0.0150), # 🔧 降低閾值範圍促進平衡（方案B）
            
            "coarse_k_range": (25, 40),     # 較少候選（日線噪聲較少）
            "fine_k_range": (8, 15),        # 精簡特徵（避免過擬合）
            
            "label_trials": 200,             # 較少試驗（日線數據有限）
            "feature_coarse_trials": 80,
            "feature_fine_trials": 120,
            "model_trials": 250,
            
            # 🆕 1D專用標籤平衡配置（長期投資特化）
            "label_balance_override": {
                "quantile_range": (0.35, 0.65),     # 保守的分位數（長期穩定）
                "preferred_method": "quantile",      # 優先分位數方法
                "balance_penalty_weight": 0.15,     # 最低不平衡懲罰權重
                "long_term_trend_factor": 1.5,      # 長期趨勢因子
                "volatility_adjustment": True       # 波動率調整
            }
        }
    }

    # 標籤優化配置 - 🆕 增強平衡性控制（方案A+B綜合）
    LABEL_OPTIMIZER_CONFIG = {
        "objectives": ["f1_score", "label_stability", "label_balance"],  # 🆕 新增平衡性目標
        "pareto_selection": {
            "f1_weight": 0.5,          # 🔧 調整權重以平衡三個目標
            "stability_weight": 0.25,
            "balance_weight": 0.25     # 🆕 平衡性權重
        },
        "label_types": ["ternary"],  # 🚀 階段2: 使用新的三分類邏輯 (買入/持有/賣出)
        "threshold_methods": ["quantile"],  # 🔧 強制使用分位數，禁用fixed
        "quantile_range": (0.20, 0.80),  # 🔧 更寬範圍，增加平衡性
        "force_quantile_15m": True,  # 🆕 15m時框強制quantile
        
        # 🆕 標籤平衡約束（強化版）
        "label_balance_constraint": {
            "enabled": True,
            "min_class_ratio": 0.15,        # 🔧 降低至15%，更嚴格
            "max_class_ratio": 0.55,        # 🔧 降低至55%，避免單一類別主導
            "target_distribution": {         # 🔧 彈性分佈範圍
                "multiclass": {
                    "short_range": (0.20, 0.30),    # 20%-30%
                    "neutral_range": (0.50, 0.60),  # 50%-60%  
                    "long_range": (0.20, 0.30)      # 20%-30%
                },
                "binary": {
                    "negative_range": (0.35, 0.45), # 35%-45%
                    "positive_range": (0.55, 0.65)  # 55%-65%
                }
            },
            "balance_score_threshold": 0.30,  # 🔧 提高閾值到30%
            "enable_auto_smote": True,       # 🆕 自動啟用SMOTE
            "smote_trigger_threshold": 0.25  # 🆕 平衡分數<25%時觸發SMOTE
        },
        
        # 🆕 動態閾值調整（方案B的補充）
        "adaptive_threshold": {
            "enabled": True,
            "auto_adjust_range": True,      # 自動調整threshold_range
            "balance_check_interval": 10,   # 每10個trial檢查一次平衡性
            "adjustment_factor": 0.8        # 閾值調整因子
        }
    }

    # 特徵選擇配置 - 🔧 兩階段選擇架構 (基於學術最佳實踐)
    FEATURE_SELECTOR_CONFIG = {
        "selector_methods": ["lightgbm", "mutual_info", "f_classif", "rfe"],
        "feature_importance_threshold": 0.0005,
        "correlation_threshold": 0.90,
        "variance_threshold": 0.005,
        
        # 🆕 兩階段特徵選擇配置 (基於金融ML文獻)
        "two_stage_selection": {
            "enabled": True,
            
            # 🔍 第2A階段：粗選候選特徵池 (快速篩選)
            "coarse_selection": {
                "primary_method": "lightgbm",        # 主要方法：LightGBM重要性
                "secondary_method": "mutual_info",   # 輔助方法：互信息
                "correlation_threshold": 0.95,      # 寬鬆去相關
                "importance_threshold": 0.0001,     # 寬鬆重要性閾值
                "stability_check": True,             # 穩定性檢查
                "cv_folds": 3                        # 快速交叉驗證
            },
            
            # 🎯 第2B階段：精選最終特徵 (精細優化)
            "fine_selection": {
                "method": "combined",                # 組合多種方法
                "methods_weight": {                  # 方法權重 (基於文獻)
                    "lightgbm": 0.35,               # 樹模型重要性
                    "mutual_info": 0.25,            # 互信息
                    "f_classif": 0.20,              # F檢驗
                    "rfe": 0.20                     # 遞歸特徵消除
                },
                "correlation_threshold": 0.85,      # 嚴格去相關
                "stability_threshold": 0.7,         # 穩定性要求
                "cv_folds": 5,                      # 完整交叉驗證
                "ensemble_validation": True         # 集成驗證
            },
            
            # 📊 時間框架特定調整 (基於HFT vs 長期投資文獻)
            "timeframe_adjustments": {
                "5m": {
                    "noise_filtering": "aggressive",  # 激進去噪
                    "technical_indicators_weight": 1.2,
                    "volume_features_weight": 1.3
                },
                "15m": {
                    "noise_filtering": "moderate",
                    "technical_indicators_weight": 1.1,
                    "volume_features_weight": 1.2
                },
                "1h": {
                    "noise_filtering": "moderate",
                    "technical_indicators_weight": 1.0,
                    "trend_features_weight": 1.1
                },
                "4h": {
                    "noise_filtering": "conservative",
                    "trend_features_weight": 1.2,
                    "momentum_features_weight": 1.1
                },
                "1D": {
                    "noise_filtering": "minimal",
                    "trend_features_weight": 1.3,
                    "fundamental_features_weight": 1.1
                }
            }
        },
        
        # 🆕 特徵重要性穩定性分析配置 - 🔧 針對不同時框優化
        "feature_importance_stability": {
            "enabled": True,                        # 啟用穩定性分析
            "min_stability_score": 0.3,            # 最小穩定性評分
            "stability_methods": ["jaccard", "spearman", "variance"],  # 穩定性計算方法
            "cross_window_threshold": 0.6,         # 跨窗口一致性閾值
            "output_detailed_report": True,        # 輸出詳細報告
            "validate_prediction_stability": True,  # 驗證預測穩定性
            
            # 🎯 時框特定配置（基於市場週期分析）
            "timeframe_configs": {
                "5m": {
                    "rolling_windows": [144, 288, 432],     # 12h, 24h, 36h
                    "top_k_features": [8, 10, 12]          # 適合超高頻
                },
                "15m": {
                    "rolling_windows": [96, 192, 288],      # 24h, 48h, 72h  
                    "top_k_features": [10, 15, 20]         # 高頻標準
                },
                "1h": {
                    "rolling_windows": [48, 96, 168],       # 48h, 96h, 168h (7天)
                    "top_k_features": [12, 18, 24]         # 中頻範圍
                },
                "4h": {
                    "rolling_windows": [18, 36, 54],        # 72h, 144h, 216h
                    "top_k_features": [10, 15, 18]         # 中長期
                },
                "1D": {
                    "rolling_windows": [14, 30, 60],        # 14天, 30天, 60天
                    "top_k_features": [8, 12, 15]          # 長期趨勢
                }
            },
            
            # 🔧 向後兼容：預設配置（15m標準）
            "rolling_windows": [96, 192, 288],      
            "top_k_features": [10, 15, 20]
        }
    }

    # 模型優化配置 - 🔧 修改：降低複雜度，增強正則化
    MODEL_OPTIMIZER_CONFIG = {
        "model_type": "lightgbm",
        "param_ranges": {
            # 🔧 針對15m高頻過擬合問題的特別調整
            "num_leaves": (12, 35),      # 🔧 進一步減少，防止過擬合
            "max_depth": (3, 7),         # 🔧 限制深度，提高泛化能力
            "learning_rate": (0.06, 0.18),  # 🔧 更保守的學習率
            "n_estimators": (80, 200),   # 🔧 減少樹數量，防止過度學習
            "feature_fraction": (0.5, 0.8),  # 🔧 更激進的特徵採樣
            "bagging_fraction": (0.5, 0.8),  # 🔧 更激進的樣本採樣
            "bagging_freq": (1, 5),
            "min_child_samples": (50, 150),  # 🔧 大幅提高最小樣本數
            "reg_alpha": (0.5, 3.0),     # 🔧 強化L1正則化
            "reg_lambda": (0.5, 3.0),    # 🔧 強化L2正則化
            "min_split_gain": (0.5, 2.5),  # 🔧 提高分裂閾值
            "subsample": (0.5, 0.8)      # 🔧 降低子採樣比例
        },
        "early_stopping_rounds": 30,  # 🔧 更嚴格的早停機制
        "verbose": False,
        # 🔧 新增：增強正則化設置
        "enhanced_regularization": {
            "min_reg_alpha": 0.1,    # L1正則化最小值
            "min_reg_lambda": 0.1,   # L2正則化最小值
            "max_complexity_score": 2000,  # 最大複雜度評分 (leaves * depth)
            "max_estimators_per_feature": 20,  # 每個特徵最多樹數
            "use_walk_forward": True,  # 🔧 啟用Walk-Forward Analysis
            
            # 🆕 CV vs WFA 一致性對比配置
            "compare_cv_vs_oos": True,        # 啟用CV與樣本外對比
            "max_oos_delta": 0.10,            # 最大容許差異(10%)
            "oos_confidence_level": 0.95,     # 樣本外置信水平
            "require_oos_validation": True    # 強制樣本外驗證
        }
    }

    # 方案C擴展配置
    ADVANCED_CONFIG = {
        # 高階特徵配置
        "advanced_features": {
            "microstructure": {
                "bid_ask_spread": True,
                "order_flow_imbalance": True,
                "trade_size_distribution": True,
                "price_impact": True
            },
            "macro_economic": {
                "vix_index": True,
                "dxy_index": True,
                "us10y_yield": True,
                "gold_price": True,
                "crypto_dominance": True
            },
            "sentiment": {
                "fear_greed_index": True,
                "social_sentiment": False,  # 需要額外API
                "news_sentiment": False     # 需要額外API
            }
        },

        # 多模型集成配置
        "ensemble": {
            "models": {
                "lightgbm": {
                    "enabled": True,
                    "weight_range": (0.0, 1.0),
                    "trials": 200
                },
                "xgboost": {
                    "enabled": True,
                    "weight_range": (0.0, 1.0),
                    "trials": 200
                },
                "catboost": {
                    "enabled": False,
                    "weight_range": (0.0, 1.0),
                    "trials": 100
                },
                "neural_net": {
                    "enabled": False,  # 方案C後期啟用
                    "weight_range": (0.0, 1.0),
                    "trials": 100
                }
            },
            "ensemble_trials": 300,
            "optimization_targets": ["f1_score", "sharpe_ratio", "max_drawdown"]
        }
    }

    # 評估指標配置 - 🔧 修改：更保守的性能期望
    EVALUATION_CONFIG = {
        "primary_metric": "f1_score",
        "secondary_metrics": [
            "precision",
            "recall",
            "accuracy",
            "roc_auc"
        ],
        "trading_metrics": [
            "sharpe_ratio",
            "profit_factor",
            "max_drawdown",
            "win_rate",
            "avg_trade_duration"
        ],
        "stability_metrics": [
            "label_consistency",
            "prediction_stability",
            "feature_stability"
        ]
    }

    # 輸出配置
    OUTPUT_CONFIG = {
        "results_dir": "results/modular_optimization",
        "visualization_dir": "results/visualization",
        "models_dir": "results/models",
        "logs_dir": "logs/optimization",
        "save_study": True,
        "save_trials": True,
        "save_models": True,
        "generate_reports": True,
        "create_visualizations": True
    }

    # 監控配置 - 🔧 修改：更現實的性能期望
    MONITORING_CONFIG = {
        "performance_thresholds": {
            "min_f1_score": 0.55,        # 🔧 提高：原來0.15，更現實
            "max_f1_score": 0.80,        # 🔧 新增：防止過擬合
            "min_sharpe_ratio": 1.0,
            "max_drawdown": 0.15,
            "min_win_rate": 0.5
        },
        "convergence": {
            "patience": 50,
            "min_trials": 30,
            "threshold": 0.001
        },
        "resource_limits": {
            "max_memory_gb": 8,
            "max_time_hours": 6
        },
        # 🔧 新增：過擬合檢測
        "overfitting_detection": {
            "max_cv_std": 0.08,          # CV標準差上限
            "min_features": 10,          # 最少特徵數
            "max_model_complexity": 300  # 最大樹數
        }
    }

    @classmethod
    def get_timeframe_config(cls, timeframe: str) -> Dict[str, Any]:
        """獲取特定時間框架的配置"""
        return cls.TIMEFRAME_CONFIG.get(timeframe, cls.TIMEFRAME_CONFIG["1h"])

    @classmethod
    def get_label_config(cls, timeframe: str) -> Dict[str, Any]:
        """獲取標籤優化配置"""
        base_config = cls.LABEL_OPTIMIZER_CONFIG.copy()
        timeframe_config = cls.get_timeframe_config(timeframe)

        base_config.update({
            "lag_range": timeframe_config["lag_range"],
            "threshold_range": timeframe_config["threshold_range"],
            "n_trials": timeframe_config["label_trials"]
        })
        return base_config

    @classmethod
    def get_feature_config(cls, timeframe: str) -> Dict[str, Any]:
        """獲取特徵選擇配置 - 🆕 支持兩階段選擇"""
        base_config = cls.FEATURE_SELECTOR_CONFIG.copy()
        timeframe_config = cls.get_timeframe_config(timeframe)

        # 🔧 傳統k_range配置 (向後兼容)
        if "k_range" in timeframe_config:
            base_config.update({
                "k_range": timeframe_config["k_range"],
                "n_trials": timeframe_config["feature_trials"]
            })
        else:
            # 🆕 新的兩階段配置
            base_config.update({
                # 粗選階段配置
                "coarse_k_range": timeframe_config["coarse_k_range"],
                "coarse_n_trials": timeframe_config["feature_coarse_trials"],
                
                # 精選階段配置  
                "fine_k_range": timeframe_config["fine_k_range"],
                "fine_n_trials": timeframe_config["feature_fine_trials"],
                
                # 🔧 向後兼容：為舊代碼提供k_range
                "k_range": timeframe_config["fine_k_range"],  # 使用精選範圍作為兼容
                "n_trials": timeframe_config["feature_fine_trials"],
                
                # 時間框架特定調整
                "timeframe_specific": timeframe_config.get("timeframe_adjustments", {}).get(timeframe, {})
            })
        
        return base_config

    @classmethod
    def get_model_config(cls, timeframe: str) -> Dict[str, Any]:
        """獲取模型優化配置"""
        base_config = cls.MODEL_OPTIMIZER_CONFIG.copy()
        timeframe_config = cls.get_timeframe_config(timeframe)

        base_config.update({
            "n_trials": timeframe_config["model_trials"]
        })
        return base_config

    @classmethod
    def create_directories(cls):
        """創建必要的目錄結構"""
        for dir_path in cls.OUTPUT_CONFIG.values():
            if isinstance(dir_path, str) and dir_path.endswith(('results', 'models', 'logs')):
                os.makedirs(dir_path, exist_ok=True)

    @classmethod
    def validate_config(cls, symbol: str, timeframe: str) -> bool:
        """驗證配置是否有效"""
        if timeframe not in cls.TIMEFRAME_CONFIG:
            print(f"警告: 時間框架 {timeframe} 不在支持列表中，使用默認配置")
            return False

        config = cls.get_timeframe_config(timeframe)
        # 🔧 更新驗證邏輯 - 支持新的兩階段配置
        required_keys = ["lag_range", "threshold_range",
                         "label_trials", "model_trials"]
        
        # 檢查特徵選擇配置 (兼容新舊格式)
        has_legacy_config = "k_range" in config and "feature_trials" in config
        has_new_config = ("coarse_k_range" in config and "fine_k_range" in config and 
                         "feature_coarse_trials" in config and "feature_fine_trials" in config)
        
        if not (has_legacy_config or has_new_config):
            print("❌ 特徵選擇配置不完整")
            return False

        for key in required_keys:
            if key not in config:
                print(f"錯誤: 配置缺少必需鍵 {key}")
                return False

        return True


# 全局配置實例
config = OptimizationConfig()


# 便捷函數
def get_config(component: str, timeframe: str = "1h") -> Dict[str, Any]:
    """獲取指定組件的配置

    Args:
        component: 組件名稱 ("label", "feature", "model", "base")
        timeframe: 時間框架

    Returns:
        對應的配置字典
    """
    if component == "label":
        return config.get_label_config(timeframe)
    elif component == "feature":
        return config.get_feature_config(timeframe)
    elif component == "model":
        return config.get_model_config(timeframe)
    elif component == "base":
        return config.BASE_CONFIG
    else:
        raise ValueError(f"未知組件: {component}")


# 🆕 分層時框配置工廠（為多幣種多時框設計）
class TimeframeConfigFactory:
    """分層時框配置工廠，支持多幣種和時框的統一配置管理"""
    
    @staticmethod
    def get_crypto_volatility_multiplier(symbol: str) -> float:
        """根據幣種獲取波動率乘數"""
        volatility_map = {
            "BTCUSDT": 1.0,     # 基準
            "ETHUSDT": 1.2,     # ETH波動性更高
            "ADAUSDT": 1.5,     # 小幣種波動性更高
            "DOTUSDT": 1.4,
            "LINKUSDT": 1.3
        }
        return volatility_map.get(symbol, 1.2)  # 默認1.2倍
    
    @staticmethod
    def get_adaptive_thresholds(symbol: str, timeframe: str) -> Dict[str, Tuple[float, float]]:
        """獲取自適應閾值範圍（考慮幣種和時框）"""
        base_ranges = {
            "5m": (0.005, 0.015),
            "15m": (0.008, 0.020),   # 🔧 修復後的15m範圍
            "1h": (0.012, 0.030),
            "4h": (0.020, 0.050),
            "1D": (0.030, 0.080)
        }
        
        # 根據幣種調整
        multiplier = TimeframeConfigFactory.get_crypto_volatility_multiplier(symbol)
        base_min, base_max = base_ranges.get(timeframe, (0.01, 0.03))
        
        return {
            "pos_threshold_range": (base_min * multiplier, base_max * multiplier),
            "neg_threshold_range": (-base_max * multiplier, -base_min * multiplier)
        }
    
    @staticmethod
    def get_optimized_config_for_symbol_timeframe(symbol: str, timeframe: str) -> Dict[str, Any]:
        """為特定幣種和時框生成優化配置"""
        
        # 獲取基礎配置
        base_config = OptimizationConfig()
        tf_config = base_config.get_timeframe_config(timeframe)
        
        # 獲取自適應閾值
        adaptive_thresholds = TimeframeConfigFactory.get_adaptive_thresholds(symbol, timeframe)
        
        # 特殊處理：15m時框的過擬合修復
        if timeframe == "15m":
            optimized_config = tf_config.copy()
            optimized_config.update({
                **adaptive_thresholds,
                "force_quantile": True,
                "enable_auto_smote": True,
                "smote_trigger_threshold": 0.25,
                "target_balance_score": 0.40,
                "max_complexity_control": True
            })
            print(f"🔧 應用15m時框過擬合修復配置 for {symbol}")
            return optimized_config
        
        # 其他時框使用標準配置
        else:
            optimized_config = tf_config.copy()
            optimized_config.update(adaptive_thresholds)
            return optimized_config
