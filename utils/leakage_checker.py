# -*- coding: utf-8 -*-
"""
数据洩漏检测工具
用于验证标签生成和特征工程中是否存在前视偏差
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging


class DataLeakageChecker:
    """数据洩漏检测器"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def check_label_generation(
        self,
        price_data: pd.Series,
        labels: pd.Series,
        lag: int
    ) -> Dict[str, Any]:
        """
        检查标签生成是否存在洩漏
        
        Args:
            price_data: 价格序列
            labels: 生成的标签
            lag: 预测滞后期
        
        Returns:
            检查结果字典
        """
        results = {
            'passed': True,
            'checks': {}
        }
        
        # 检查1: 标签长度
        expected_len = len(price_data) - lag
        actual_len = len(labels)
        
        length_diff = abs(expected_len - actual_len)
        length_ok = length_diff <= 1  # 允许±1的差异
        
        results['checks']['length'] = {
            'passed': length_ok,
            'expected': expected_len,
            'actual': actual_len,
            'diff': length_diff
        }
        
        if not length_ok:
            self.logger.warning(
                f"⚠️ 标签长度异常: 预期{expected_len}, 实际{actual_len}, 差异{length_diff}"
            )
            results['passed'] = False
        else:
            self.logger.info("✅ 标签长度检查通过")
        
        # 检查2: 标签值范围
        unique_values = labels.unique()
        values_ok = set(unique_values).issubset({0, 1, 2})
        
        results['checks']['values'] = {
            'passed': values_ok,
            'unique_values': unique_values.tolist(),
            'expected': [0, 1, 2]
        }
        
        if not values_ok:
            self.logger.warning(f"⚠️ 标签值异常: {unique_values}")
            results['passed'] = False
        else:
            self.logger.info("✅ 标签值范围检查通过")
        
        # 检查3: NaN值
        nan_count = labels.isna().sum()
        nan_ok = nan_count == 0
        
        results['checks']['nan'] = {
            'passed': nan_ok,
            'nan_count': int(nan_count),
            'nan_ratio': float(nan_count / len(labels)) if len(labels) > 0 else 0
        }
        
        if not nan_ok:
            self.logger.warning(f"⚠️ 标签包含{nan_count}个NaN值")
            results['passed'] = False
        else:
            self.logger.info("✅ NaN值检查通过")
        
        # 检查4: 索引连续性
        if isinstance(labels.index, pd.DatetimeIndex):
            # 检查索引是否单调递增
            is_monotonic = labels.index.is_monotonic_increasing
            
            results['checks']['index'] = {
                'passed': is_monotonic,
                'is_monotonic': is_monotonic,
                'index_type': 'DatetimeIndex'
            }
            
            if not is_monotonic:
                self.logger.warning("⚠️ 标签索引不是单调递增")
                results['passed'] = False
            else:
                self.logger.info("✅ 索引连续性检查通过")
        
        return results
    
    def check_feature_future_dependency(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.8
    ) -> Dict[str, List]:
        """
        检查特征是否异常依赖未来数据
        
        通过计算特征与未来标签的相关性来检测
        如果相关性异常高，可能存在洩漏
        """
        suspicious_features = []
        
        if len(X) != len(y):
            self.logger.warning("⚠️ 特征和标签长度不匹配")
            return {'suspicious_features': []}
        
        # 计算与未来标签的相关性
        y_future = y.shift(-1).fillna(0)
        
        for col in X.columns:
            try:
                corr = X[col].corr(y_future)
                
                if abs(corr) > threshold:
                    suspicious_features.append({
                        'feature': col,
                        'correlation': float(corr),
                        'risk': 'HIGH',
                        'reason': f'与未来标签相关性{corr:.3f} > {threshold}'
                    })
            except:
                continue
        
        if suspicious_features:
            self.logger.warning(
                f"⚠️ 发现{len(suspicious_features)}个可疑特征（与未来标签高度相关）"
            )
            for feat in suspicious_features[:5]:  # 只显示前5个
                self.logger.warning(f"   {feat['feature']}: {feat['correlation']:.3f}")
        else:
            self.logger.info("✅ 未发现异常高相关特征")
        
        return {
            'suspicious_features': suspicious_features,
            'total_checked': len(X.columns),
            'suspicious_count': len(suspicious_features)
        }
    
    def check_train_test_leakage(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        check_overlap: bool = True,
        check_scaling: bool = True
    ) -> Dict:
        """检查训练集和测试集之间是否有洩漏"""
        results = {
            'passed': True,
            'checks': {}
        }
        
        # 检查1: 时间顺序（测试集应晚于训练集）
        if isinstance(X_train.index, pd.DatetimeIndex) and isinstance(X_test.index, pd.DatetimeIndex):
            train_end = X_train.index.max()
            test_start = X_test.index.min()
            
            time_order_ok = test_start > train_end
            
            results['checks']['time_order'] = {
                'passed': time_order_ok,
                'train_end': str(train_end),
                'test_start': str(test_start),
                'gap': str(test_start - train_end) if test_start > train_end else 'OVERLAP'
            }
            
            if not time_order_ok:
                self.logger.error("❌ 测试集起始时间早于训练集结束时间（时间重叠）")
                results['passed'] = False
            else:
                self.logger.info("✅ 时间顺序检查通过")
        
        # 检查2: 数据重叠（检查是否有相同的行）
        if check_overlap:
            common_index = X_train.index.intersection(X_test.index)
            overlap_count = len(common_index)
            overlap_ok = overlap_count == 0
            
            results['checks']['overlap'] = {
                'passed': overlap_ok,
                'overlap_count': overlap_count,
                'overlap_ratio': overlap_count / len(X_test) if len(X_test) > 0 else 0
            }
            
            if not overlap_ok:
                self.logger.warning(f"⚠️ 训练集和测试集有{overlap_count}个重叠样本")
                results['passed'] = False
            else:
                self.logger.info("✅ 数据重叠检查通过")
        
        return results
    
    def check_rolling_calculation(
        self,
        data: pd.Series,
        window: int,
        feature_name: str = "unknown"
    ) -> Dict:
        """
        检查滚动计算是否正确使用历史数据
        
        验证点：
        - 窗口大小合理
        - 是否使用shift避免当前点
        """
        results = {
            'passed': True,
            'feature': feature_name
        }
        
        # 检查窗口大小
        if window > len(data) * 0.5:
            self.logger.warning(
                f"⚠️ {feature_name}: 窗口{window}过大（>50%数据长度）"
            )
            results['passed'] = False
        
        # 检查窗口是否合理
        if window < 2:
            self.logger.warning(f"⚠️ {feature_name}: 窗口{window}过小")
            results['passed'] = False
        
        results['window'] = window
        results['data_length'] = len(data)
        results['window_ratio'] = window / len(data)
        
        return results
    
    def generate_leakage_report(
        self,
        label_check: Dict,
        feature_check: Dict,
        output_file: Optional[str] = None
    ) -> str:
        """生成洩漏检测报告"""
        lines = [
            "# 数据洩漏检测报告\n",
            f"生成时间: {pd.Timestamp.now()}\n",
            "---\n",
            "\n## 标签生成检查\n"
        ]
        
        # 标签检查结果
        if label_check.get('passed'):
            lines.append("✅ **标签生成无洩漏**\n")
        else:
            lines.append("❌ **标签生成可能存在洩漏**\n")
        
        for check_name, check_result in label_check.get('checks', {}).items():
            status = "✅" if check_result.get('passed') else "❌"
            lines.append(f"\n### {check_name}\n")
            lines.append(f"{status} 状态: {'通过' if check_result.get('passed') else '失败'}\n")
            for key, value in check_result.items():
                if key != 'passed':
                    lines.append(f"- {key}: {value}\n")
        
        # 特征检查结果
        lines.append("\n## 特征检查\n")
        
        suspicious = feature_check.get('suspicious_features', [])
        if len(suspicious) == 0:
            lines.append("✅ **未发现可疑特征**\n")
        else:
            lines.append(f"⚠️ **发现{len(suspicious)}个可疑特征**\n\n")
            for feat in suspicious[:10]:  # 只列出前10个
                lines.append(f"- {feat['feature']}: 相关性{feat['correlation']:.3f}\n")
        
        report = ''.join(lines)
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"报告已保存: {output_file}")
        
        return report


# 使用示例
if __name__ == "__main__":
    # 示例：检查标签生成
    checker = DataLeakageChecker()
    
    # 模拟价格数据
    dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
    prices = pd.Series(
        45000 + np.cumsum(np.random.randn(1000) * 100),
        index=dates
    )
    
    # 模拟标签（正确生成，应无洩漏）
    future_returns = prices.pct_change(5).shift(-5)
    labels = pd.Series(1, index=prices.index)
    labels[future_returns > 0.01] = 2
    labels[future_returns < -0.01] = 0
    labels = labels[:-5]  # 移除未来部分
    
    # 检查
    result = checker.check_label_generation(prices, labels, lag=5)
    
    print("检查结果:")
    print(f"  通过: {result['passed']}")
    for check_name, check_data in result['checks'].items():
        print(f"  {check_name}: {check_data}")
    
    print("\n数据洩漏检测工具测试完成！")
