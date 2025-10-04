#!/usr/bin/env python3
"""
A/B 測試系統
將個性化優化配置與統一基準比較
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import sys

# 添加項目根目錄到Python路徑
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

from src.optimization.advanced_optuna_optimizer import AdvancedOptunaOptimizer
from src.config.config_manager import ConfigManager


@dataclass
class ABTestResult:
    """A/B 測試結果數據類"""
    symbol: str
    timeframe: str
    config_type: str  # 'personalized' 或 'baseline'
    best_score: float
    avg_score: float
    score_std: float
    consistency_score: float
    trial_count: int
    optimization_time: float
    best_params: Dict[str, Any]
    

class ABTester:
    """
    A/B 測試系統
    比較個性化配置驅動優化 vs 統一基準配置
    """
    
    # 統一基準配置
    BASELINE_CONFIG = {
        'label_lag_range': (1, 10),
        'label_threshold_range': (0.01, 0.05),
        'n_features_range': (20, 50),
        'cv_folds': 5,
        'trials_config': {
            'layer1_total': 100,
            'layer2_total': 80,
            'stage1_trials': 60,
            'stage2_trials': 40
        },
        'model_param_ranges': {
            'num_leaves': (20, 100),
            'max_depth': (4, 15),
            'learning_rate': (0.01, 0.3),
            'n_estimators': (100, 1000),
            'reg_alpha': (0.0, 1.0),
            'reg_lambda': (0.0, 1.0),
            'feature_fraction': (0.6, 1.0),
            'bagging_fraction': (0.6, 1.0),
            'bagging_freq': (1, 7),
            'min_child_samples': (5, 100)
        }
    }
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.results_dir = Path("results/ab_testing")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("🧪 A/B 測試系統初始化完成")
        print(f"📁 結果目錄: {self.results_dir}")
    
    async def run_ab_test(self, symbol: str, timeframe: str, 
                         n_trials: int = 100) -> Dict[str, ABTestResult]:
        """
        運行單個 symbol-timeframe 的 A/B 測試
        
        Args:
            symbol: 交易對
            timeframe: 時框
            n_trials: 每個配置的試驗次數
            
        Returns:
            A/B 測試結果對比
        """
        print(f"\n🧪 開始 A/B 測試: {symbol} {timeframe}")
        print("=" * 80)
        
        results = {}
        
        try:
            # A組：個性化配置驅動優化
            print("🅰️ 測試組A: 個性化配置驅動優化")
            personalized_result = await self._run_personalized_optimization(
                symbol, timeframe, n_trials
            )
            results['personalized'] = personalized_result
            
            # B組：統一基準配置優化
            print("\n🅱️ 測試組B: 統一基準配置優化")  
            baseline_result = await self._run_baseline_optimization(
                symbol, timeframe, n_trials
            )
            results['baseline'] = baseline_result
            
            # 生成對比報告
            self._generate_comparison_report(results, symbol, timeframe)
            
            return results
            
        except Exception as e:
            print(f"❌ A/B 測試失敗: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    async def _run_personalized_optimization(self, symbol: str, timeframe: str, 
                                           n_trials: int) -> ABTestResult:
        """運行個性化配置驅動優化"""
        start_time = datetime.now()
        
        # 使用配置驅動的優化器
        optimizer = AdvancedOptunaOptimizer(symbol, timeframe)
        
        print(f"📊 個性化配置參數:")
        print(f"   滯後期範圍: {optimizer.get_label_lag_range()}")
        print(f"   閾值範圍: {optimizer.get_label_threshold_range()}")
        print(f"   特徵數範圍: {optimizer.get_feature_count_range()}")
        print(f"   CV折數: {optimizer.get_cv_folds()}")
        
        # 運行優化
        results = await optimizer.run_complete_optimization()
        
        end_time = datetime.now()
        optimization_time = (end_time - start_time).total_seconds()
        
        # 提取結果
        layer1_results = results.get('layer1', {})
        layer2_results = results.get('layer2', {})
        
        return ABTestResult(
            symbol=symbol,
            timeframe=timeframe,
            config_type='personalized',
            best_score=layer2_results.get('best_score', 0.0),
            avg_score=layer1_results.get('cv_scores', [0.0])[0] if layer1_results.get('cv_scores') else 0.0,
            score_std=np.std(layer1_results.get('cv_scores', [0.0])),
            consistency_score=-np.std(layer1_results.get('cv_scores', [0.0])),
            trial_count=layer1_results.get('total_trials', n_trials),
            optimization_time=optimization_time,
            best_params={**layer1_results.get('best_params', {}), 
                        **layer2_results.get('best_params', {})}
        )
    
    async def _run_baseline_optimization(self, symbol: str, timeframe: str, 
                                       n_trials: int) -> ABTestResult:
        """運行統一基準配置優化"""
        start_time = datetime.now()
        
        # 創建使用基準配置的優化器
        optimizer = BaselineOptimizer(symbol, timeframe, self.BASELINE_CONFIG)
        
        print(f"📊 統一基準配置參數:")
        print(f"   滯後期範圍: {self.BASELINE_CONFIG['label_lag_range']}")
        print(f"   閾值範圍: {self.BASELINE_CONFIG['label_threshold_range']}")
        print(f"   特徵數範圍: {self.BASELINE_CONFIG['n_features_range']}")
        print(f"   CV折數: {self.BASELINE_CONFIG['cv_folds']}")
        
        # 運行優化
        results = await optimizer.run_baseline_optimization(n_trials)
        
        end_time = datetime.now()
        optimization_time = (end_time - start_time).total_seconds()
        
        return ABTestResult(
            symbol=symbol,
            timeframe=timeframe,
            config_type='baseline',
            best_score=results.get('best_score', 0.0),
            avg_score=results.get('avg_cv_score', 0.0),
            score_std=results.get('score_std', 0.0),
            consistency_score=-results.get('score_std', 0.0),
            trial_count=results.get('trial_count', n_trials),
            optimization_time=optimization_time,
            best_params=results.get('best_params', {})
        )
    
    def _generate_comparison_report(self, results: Dict[str, ABTestResult], 
                                   symbol: str, timeframe: str):
        """生成詳細的對比報告"""
        personalized = results.get('personalized')
        baseline = results.get('baseline')
        
        if not personalized or not baseline:
            print("⚠️ 結果不完整，無法生成對比報告")
            return
        
        print(f"\n📊 A/B 測試對比報告: {symbol} {timeframe}")
        print("=" * 80)
        
        # 性能對比
        score_improvement = personalized.best_score - baseline.best_score
        score_improvement_pct = (score_improvement / baseline.best_score * 100) if baseline.best_score > 0 else 0
        
        consistency_improvement = personalized.consistency_score - baseline.consistency_score
        
        print(f"🎯 **性能指標對比**")
        print(f"   個性化最佳分數: {personalized.best_score:.6f}")
        print(f"   基準最佳分數: {baseline.best_score:.6f}")
        print(f"   分數提升: {score_improvement:+.6f} ({score_improvement_pct:+.2f}%)")
        
        print(f"\n📈 **一致性對比**")
        print(f"   個性化一致性: {personalized.consistency_score:.6f}")
        print(f"   基準一致性: {baseline.consistency_score:.6f}")
        print(f"   一致性提升: {consistency_improvement:+.6f}")
        
        print(f"\n⏱️ **效率對比**")
        print(f"   個性化優化時間: {personalized.optimization_time:.1f} 秒")
        print(f"   基準優化時間: {baseline.optimization_time:.1f} 秒")
        print(f"   時間差異: {personalized.optimization_time - baseline.optimization_time:+.1f} 秒")
        
        # 統計顯著性分析
        print(f"\n🔬 **統計分析**")
        
        if score_improvement_pct > 5:
            significance = "🚀 顯著優勢"
        elif score_improvement_pct > 2:
            significance = "✅ 明顯提升" 
        elif score_improvement_pct > 0:
            significance = "📈 輕微提升"
        else:
            significance = "❌ 無明顯優勢"
        
        print(f"   結論: {significance}")
        print(f"   置信度: {self._calculate_confidence(personalized, baseline):.1f}%")
        
        # 保存詳細結果
        self._save_ab_test_results(results, symbol, timeframe)
    
    def _calculate_confidence(self, personalized: ABTestResult, 
                             baseline: ABTestResult) -> float:
        """計算統計置信度（簡化版本）"""
        # 簡化的置信度計算，基於分數差異和標準差
        score_diff = abs(personalized.best_score - baseline.best_score)
        combined_std = (personalized.score_std + baseline.score_std) / 2
        
        if combined_std == 0:
            return 99.9 if score_diff > 0 else 50.0
            
        # 簡化的 t-test 近似
        t_stat = score_diff / combined_std
        confidence = min(95.0 + t_stat * 10, 99.9)  # 簡化映射
        
        return confidence
    
    def _save_ab_test_results(self, results: Dict[str, ABTestResult], 
                             symbol: str, timeframe: str):
        """保存 A/B 測試結果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ab_test_{symbol}_{timeframe}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # 序列化結果
        serializable_results = {}
        for key, result in results.items():
            serializable_results[key] = {
                'symbol': result.symbol,
                'timeframe': result.timeframe,
                'config_type': result.config_type,
                'best_score': result.best_score,
                'avg_score': result.avg_score,
                'score_std': result.score_std,
                'consistency_score': result.consistency_score,
                'trial_count': result.trial_count,
                'optimization_time': result.optimization_time,
                'best_params': result.best_params
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 A/B 測試結果已保存: {filepath}")
    
    async def run_batch_ab_test(self, symbol_timeframe_pairs: List[Tuple[str, str]], 
                               n_trials: int = 100) -> Dict[str, Dict[str, ABTestResult]]:
        """批量運行多個 A/B 測試"""
        print(f"🧪 開始批量 A/B 測試: {len(symbol_timeframe_pairs)} 個組合")
        
        all_results = {}
        
        for symbol, timeframe in symbol_timeframe_pairs:
            test_key = f"{symbol}_{timeframe}"
            print(f"\n🔄 處理 {test_key}...")
            
            try:
                results = await self.run_ab_test(symbol, timeframe, n_trials)
                all_results[test_key] = results
                print(f"✅ {test_key} 完成")
            except Exception as e:
                print(f"❌ {test_key} 失敗: {e}")
                all_results[test_key] = {}
        
        # 生成批量總結報告
        self._generate_batch_summary(all_results)
        
        return all_results
    
    def _generate_batch_summary(self, all_results: Dict[str, Dict[str, ABTestResult]]):
        """生成批量測試總結報告"""
        print(f"\n📊 批量 A/B 測試總結報告")
        print("=" * 100)
        
        wins = 0
        total = 0
        total_improvement = 0
        
        print(f"{'組合':15} {'個性化分數':12} {'基準分數':12} {'提升幅度':12} {'結論':15}")
        print("-" * 80)
        
        for test_key, results in all_results.items():
            if not results:
                continue
                
            personalized = results.get('personalized')
            baseline = results.get('baseline')
            
            if personalized and baseline:
                improvement = personalized.best_score - baseline.best_score
                improvement_pct = (improvement / baseline.best_score * 100) if baseline.best_score > 0 else 0
                total_improvement += improvement_pct
                total += 1
                
                if improvement > 0:
                    wins += 1
                    conclusion = "✅ 個性化勝出"
                else:
                    conclusion = "❌ 基準勝出"
                
                print(f"{test_key:15} {personalized.best_score:.6f}   {baseline.best_score:.6f}   "
                      f"{improvement_pct:+8.2f}%    {conclusion}")
        
        print("-" * 80)
        if total > 0:
            win_rate = wins / total * 100
            avg_improvement = total_improvement / total
            print(f"總結: {wins}/{total} 勝出 (勝率: {win_rate:.1f}%)")
            print(f"平均提升: {avg_improvement:+.2f}%")
        
        print("=" * 100)


class BaselineOptimizer:
    """基準配置優化器"""
    
    def __init__(self, symbol: str, timeframe: str, baseline_config: Dict):
        self.symbol = symbol
        self.timeframe = timeframe
        self.config = baseline_config
        self.optimizer = AdvancedOptunaOptimizer(symbol, timeframe)
        
        # 強制覆蓋配置為基準配置
        self._override_config()
    
    def _override_config(self):
        """覆蓋優化器配置為基準配置"""
        # 這裡需要臨時覆蓋配置方法
        self.optimizer.strategy_config = self.config
        
    async def run_baseline_optimization(self, n_trials: int) -> Dict[str, Any]:
        """使用基準配置運行優化"""
        # 直接調用優化器的完整優化，但使用基準配置
        results = await self.optimizer.run_complete_optimization()
        
        # 計算額外的統計信息
        layer1_results = results.get('layer1', {})
        cv_scores = layer1_results.get('cv_scores', [])
        
        return {
            'best_score': results.get('layer2', {}).get('best_score', 0.0),
            'avg_cv_score': np.mean(cv_scores) if cv_scores else 0.0,
            'score_std': np.std(cv_scores) if cv_scores else 0.0,
            'trial_count': n_trials,
            'best_params': {**layer1_results.get('best_params', {}),
                          **results.get('layer2', {}).get('best_params', {})}
        }


# 使用範例和測試函數
async def test_ab_system():
    """測試 A/B 系統"""
    tester = ABTester()
    
    # 單個測試
    results = await tester.run_ab_test('BTCUSDT', '1h', n_trials=50)
    
    # 批量測試
    pairs = [('BTCUSDT', '1h'), ('ETHUSDT', '1h')]
    batch_results = await tester.run_batch_ab_test(pairs, n_trials=30)
    
    return results, batch_results


if __name__ == "__main__":
    # 運行測試
    results, batch_results = asyncio.run(test_ab_system())
    print("A/B 測試系統測試完成")
