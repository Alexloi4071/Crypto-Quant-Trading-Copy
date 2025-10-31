# -*- coding: utf-8 -*-
"""
Meta-Labeling Coordinator
整合 Primary Model + Meta Model

Meta-Labeling 雙層架構協調器
- Layer 1A: PrimaryLabelOptimizer (方向預測)
- Layer 1B: MetaQualityOptimizer (質量評估)
- 輸出：三分類標籤（向後兼容 Layer2）

參考文獻：
- Marcos López de Prado (2018), "Advances in Financial Machine Learning", Ch.3
"""
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from .optuna_primary_label import PrimaryLabelOptimizer
from .optuna_meta_quality import MetaQualityOptimizer
from optuna_system.utils.timeframe_alignment import TimeframeAlignmentError


class MetaLabelOptimizer:
    """
    Meta-Labeling 雙層架構協調器
    
    整合：
    - Layer 1A: PrimaryLabelOptimizer (方向預測)
    - Layer 1B: MetaQualityOptimizer (質量評估)
    
    輸出：
    - label: 0/1/2 (賣/持/買) - 向後兼容
    - primary_signal: 1/-1 (買/賣) - 新增
    - meta_quality: 1/0 (執行/跳過) - 新增
    """
    
    def __init__(self, **kwargs):
        """
        初始化 Meta-Labeling 協調器
        
        Args:
            **kwargs: 傳遞給 Primary 和 Meta 優化器的參數
        """
        self.logger = logging.getLogger(__name__)
        
        # 初始化雙優化器
        self.primary_optimizer = PrimaryLabelOptimizer(**kwargs)
        self.meta_optimizer = MetaQualityOptimizer(**kwargs)
        
        # 保存參數（用於後續物化）
        self.data_path = kwargs.get('data_path')
        self.symbol = kwargs.get('symbol', 'BTCUSDT')
        self.timeframe = kwargs.get('timeframe', '15m')
        
        self.logger.info("✅ Meta-Labeling 雙層協調器初始化完成")
    
    def optimize(self, n_trials: int = 200) -> Dict:
        """
        兩階段優化
        
        階段 1：優化 Primary Model（n_trials // 2）
        階段 2：優化 Meta Model（n_trials // 2）
        
        Args:
            n_trials: 總試驗次數
        
        Returns:
            Dict: 包含兩階段結果的字典
        """
        self.logger.info("🚀 Meta-Labeling 兩階段優化開始（多目標優化）...")
        self.logger.info(f"   總試驗次數: {n_trials}")
        primary_trials = n_trials // 3  # 分配 1/3 給 Primary
        meta_trials = (n_trials * 2) // 3  # 分配 2/3 給 Meta（多目標需要更多）
        self.logger.info(f"   Primary Model: {primary_trials} trials")
        self.logger.info(f"   Meta Model: {meta_trials} trials (多目標優化)")
        
        # ===== 階段 1：Primary Model =====
        self.logger.info("\n" + "="*60)
        self.logger.info("📍 階段 1/2：優化方向預測器 (Primary Model)")
        self.logger.info("="*60)
        
        primary_result = self.primary_optimizer.optimize(n_trials=primary_trials)
        primary_params = primary_result['best_params']
        
        self.logger.info(f"\n✅ Primary Model 優化完成:")
        self.logger.info(f"   最佳得分: {primary_result['best_score']:.4f}")
        self.logger.info(f"   方向準確率: {primary_result.get('accuracy', 0):.3f}")
        self.logger.info(f"   Sharpe: {primary_result.get('sharpe', 0):.2f}")
        self.logger.info(f"   買入比例: {primary_result.get('buy_ratio', 0):.2%}")
        
        # 生成 Primary 信號（供 Meta Model 使用）
        self.logger.info("\n🔄 生成 Primary 信號供 Meta Model 使用...")
        primary_signals = self.primary_optimizer.generate_primary_signals(
            self.primary_optimizer.price_data['close'],
            primary_params
        )
        self.logger.info(f"✅ Primary 信號已生成: {len(primary_signals)} 筆")
        
        # ===== 階段 2：Meta Model（多目標優化）=====
        self.logger.info("\n" + "="*60)
        self.logger.info("📍 階段 2/2：優化質量評估器 (Meta Model - 多目標)")
        self.logger.info("="*60)
        
        # 設定 Meta Model 的輸入
        self.meta_optimizer.set_primary_signals(primary_signals)
        self.meta_optimizer.price_data = self.primary_optimizer.price_data
        
        # 優化 Meta Model（多目標優化）
        meta_result = self.meta_optimizer.optimize(n_trials=meta_trials)
        meta_params = meta_result['best_params']
        
        self.logger.info(f"\n✅ Meta Model 優化完成:")
        self.logger.info(f"   性能得分: {meta_result['best_score']:.4f}")
        self.logger.info(f"   標籤偏差: {meta_result.get('label_deviation', 0):.2%}")
        self.logger.info(f"   F1 分數: {meta_result.get('f1_score', 0):.3f}")
        self.logger.info(f"   執行率: {meta_result.get('execution_ratio', 0):.2%}")
        self.logger.info(f"   最終分布: 卖{meta_result.get('sell_pct', 0):.1%} / "
                        f"持{meta_result.get('hold_pct', 0):.1%} / "
                        f"买{meta_result.get('buy_pct', 0):.1%}")
        
        # ===== 組合結果 =====
        combined_score = (
            primary_result['best_score'] * 0.4 +
            meta_result['best_score'] * 0.6
        )
        
        self.logger.info("\n" + "="*60)
        self.logger.info("🎉 Meta-Labeling 兩階段優化完成!")
        self.logger.info("="*60)
        self.logger.info(f"   綜合得分: {combined_score:.4f}")
        self.logger.info(f"   Primary 得分: {primary_result['best_score']:.4f}")
        self.logger.info(f"   Meta 得分: {meta_result['best_score']:.4f}")
        self.logger.info(f"   Pareto 前沿: {meta_result.get('pareto_front_size', 0)} 個解")
        
        return {
            'best_params': {
                'primary': primary_params,
                'meta': meta_params,
                'n_trials': n_trials
            },
            'best_score': combined_score,
            'primary_result': primary_result,
            'meta_result': meta_result,
            'n_trials': n_trials
        }
    
    def apply_labels(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        應用 Meta-Labeling（物化輸出）
        
        輸出欄位：
        - primary_signal: 1/-1 (方向)
        - meta_quality: 1/0 (質量)
        - label: 2/1/0 (最終三分類，向後兼容)
        
        Args:
            data: 輸入數據（OHLCV）
            params: 包含 primary 和 meta 參數的字典
        
        Returns:
            pd.DataFrame: 包含標籤的數據
        """
        self.logger.info("🔄 應用 Meta-Labeling 標籤...")
        
        # 提取參數
        primary_params = params.get('primary', {})
        meta_params = params.get('meta', {})
        
        # 🔧 P0修复：验证嵌套lag一致性
        # 问题：Primary和Meta模型可能使用不同的lag，导致时间对齐错误
        # 修复：确保两个模型的lag一致，或明确记录差异
        primary_lag = primary_params.get('lag', 0)
        meta_lag = meta_params.get('lag', primary_lag)  # 默认使用primary的lag
        
        if primary_lag != meta_lag:
            error_msg = (
                f"⚠️ 嵌套lag不一致: Primary lag={primary_lag}, Meta lag={meta_lag}. "
                f"这可能导致时间对齐错误！建议使用相同的lag。"
            )
            self.logger.warning(error_msg)
            
            # 强制统一lag（使用primary的lag）
            meta_params['lag'] = primary_lag
            self.logger.info(f"🔧 已强制统一lag={primary_lag}")
        else:
            self.logger.info(f"✅ Lag对齐检查通过: Primary lag={primary_lag}, Meta lag={meta_lag}")
        
        # ===== 步驟 1：生成 Primary 信號 =====
        self.logger.info("   步驟 1/3: 生成 Primary 信號...")
        primary_signals = self.primary_optimizer.generate_primary_signals(
            data['close'],
            primary_params
        )
        
        buy_count = (primary_signals == 1).sum()
        sell_count = (primary_signals == -1).sum()
        self.logger.info(f"   ✅ Primary 信號: {len(primary_signals)} 筆 "
                        f"(買={buy_count}, 賣={sell_count})")
        
        # ===== 步驟 2：生成 Meta 特徵和評估質量 =====
        self.logger.info("   步驟 2/3: 評估信號質量...")
        
        # 合併 Primary 和 Meta 參數（Meta 需要一些 Primary 參數）
        combined_params = {**primary_params, **meta_params}
        
        meta_features = self.meta_optimizer.generate_meta_features(
            primary_signals,
            data,
            combined_params
        )
        
        meta_quality = self.meta_optimizer.evaluate_quality(
            meta_features,
            meta_params
        )
        
        execute_count = (meta_quality == 1).sum()
        skip_count = (meta_quality == 0).sum()
        execution_ratio = execute_count / len(meta_quality) if len(meta_quality) > 0 else 0
        
        self.logger.info(f"   ✅ Meta 評估: 執行={execute_count}, 跳過={skip_count} "
                        f"(執行率={execution_ratio:.1%})")
        
        # ===== 步驟 3：組合最終標籤 =====
        self.logger.info("   步驟 3/3: 組合最終標籤...")
        
        result = data.loc[primary_signals.index].copy()
        result['primary_signal'] = primary_signals  # 1/-1
        result['meta_quality'] = meta_quality       # 1/0
        
        # ✅ 關鍵：生成三分類 label（向後兼容 Layer2）
        final_label = pd.Series(1, index=primary_signals.index, dtype=int)  # 默認持有
        
        # 買入信號 + 高質量 → 執行買入
        final_label[(primary_signals == 1) & (meta_quality == 1)] = 2
        
        # 賣出信號 + 高質量 → 執行賣出
        final_label[(primary_signals == -1) & (meta_quality == 1)] = 0
        
        # meta_quality == 0 的信號保持為 1 (持有)
        
        result['label'] = final_label  # 0/1/2（與 Legacy 兼容）
        
        # 統計最終標籤分佈
        label_counts = final_label.value_counts().sort_index()
        total = len(final_label)
        
        self.logger.info(f"\n   ✅ 最終標籤分佈:")
        for label_val in [0, 1, 2]:
            count = label_counts.get(label_val, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            label_name = {0: '賣出', 1: '持有', 2: '買入'}[label_val]
            self.logger.info(f"      {label_name}({label_val}): {count:,} ({percentage:.1f}%)")
        
        return result
    
    def apply_transform(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        統一物化接口（Coordinator 調用）
        
        這是 Coordinator 期望的接口名稱
        """
        return self.apply_labels(data, params)


if __name__ == "__main__":
    # 獨立測試腳本
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    print("🚀 Meta-Labeling Coordinator 測試")
    
    optimizer = MetaLabelOptimizer(
        data_path='../../data',
        config_path='../../configs',
        symbol='BTCUSDT',
        timeframe='15m'
    )
    
    print(f"✅ 協調器初始化成功")
    print(f"   Primary Optimizer: {optimizer.primary_optimizer.__class__.__name__}")
    print(f"   Meta Optimizer: {optimizer.meta_optimizer.__class__.__name__}")
    
    print("\n🔬 開始兩階段優化測試（10 trials）...")
    result = optimizer.optimize(n_trials=10)
    
    print(f"\n✅ 優化完成!")
    print(f"   綜合得分: {result['best_score']:.4f}")
    print(f"   Primary 得分: {result['primary_result']['best_score']:.4f}")
    print(f"   Meta 得分: {result['meta_result']['best_score']:.4f}")
    
    print("\n📋 最優參數:")
    print(f"   Primary: {result['best_params']['primary']}")
    print(f"   Meta: {result['best_params']['meta']}")
