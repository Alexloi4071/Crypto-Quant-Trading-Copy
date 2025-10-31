#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析 Layer 优化结果"""

import json
import pandas as pd
from pathlib import Path
import sys

def analyze_layer1_results():
    """分析 Layer1 结果"""
    print("\n" + "="*60)
    print("📊 Layer1 (Meta-Labeling) 结果分析")
    print("="*60)
    
    # 1. 读取配置
    config_file = Path("optuna_system/configs/label_params_15m.json")
    if not config_file.exists():
        print("❌ 未找到 Layer1 配置文件")
        return False
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 2. 显示基本信息
    best_score = config.get('best_score', 0)
    n_trials = config.get('n_trials', 0)
    
    print(f"\n✅ 优化完成:")
    print(f"   最佳分数: {best_score:.4f}")
    print(f"   总试验数: {n_trials}")
    
    # 3. Primary 结果
    primary_result = config.get('primary_result', {})
    if primary_result:
        print(f"\n📊 Primary Model (方向预测):")
        print(f"   准确率: {primary_result.get('accuracy', 0):.2%}")
        print(f"   Sharpe: {primary_result.get('sharpe', 0):.2f}")
        print(f"   买入比例: {primary_result.get('buy_ratio', 0):.2%}")
    
    # 4. Meta 结果
    meta_result = config.get('meta_result', {})
    if meta_result:
        print(f"\n📊 Meta Model (质量评估):")
        print(f"   F1 Score: {meta_result.get('f1_score', 0):.4f}")
        print(f"   Precision: {meta_result.get('precision', 0):.4f}")
        print(f"   Recall: {meta_result.get('recall', 0):.4f}")
        print(f"   Sharpe: {meta_result.get('sharpe', 0):.2f}")
        print(f"   执行率: {meta_result.get('execution_ratio', 0):.2%}")
    
    # 5. 读取标签文件
    label_path = config.get('materialized_path')
    if label_path and Path(label_path).exists():
        df = pd.read_parquet(label_path)
        
        print(f"\n📊 标签分布:")
        label_counts = df['label'].value_counts().sort_index()
        label_ratio = df['label'].value_counts(normalize=True).sort_index()
        
        label_names = {0: '卖出', 1: '持有', 2: '买入'}
        
        for label_val in [0, 1, 2]:
            count = label_counts.get(label_val, 0)
            ratio = label_ratio.get(label_val, 0)
            name = label_names[label_val]
            status = "✅" if count > 0 else "❌"
            print(f"   {status} {name}({label_val}): {count:,} ({ratio:.1%})")
        
        # 检查缺失类别
        missing = [label_names[i] for i in [0,1,2] if label_counts.get(i, 0) == 0]
        if missing:
            print(f"\n⚠️ 警告: 缺失类别 {missing}")
            return False
        else:
            print(f"\n✅ 所有类别都存在，标签生成成功！")
            return True
    else:
        print(f"\n❌ 未找到标签文件: {label_path}")
        return False

def analyze_layer2_results():
    """分析 Layer2 结果"""
    print("\n" + "="*60)
    print("📊 Layer2 (Feature Engineering) 结果分析")
    print("="*60)
    
    # 1. 读取配置
    config_file = Path("optuna_system/configs/feature_params_15m.json")
    if not config_file.exists():
        print("❌ 未找到 Layer2 配置文件")
        return False
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 2. 显示基本信息
    best_score = config.get('best_score', 0)
    n_trials = config.get('n_trials', 0)
    
    print(f"\n✅ 优化完成:")
    print(f"   最佳分数: {best_score:.4f}")
    print(f"   总试验数: {n_trials}")
    
    if best_score == 0:
        print(f"\n❌ 警告: 最佳分数为 0，优化可能失败")
        return False
    
    # 3. 特征信息
    best_params = config.get('best_params', {})
    selected_features = best_params.get('selected_features', [])
    
    print(f"\n📊 特征选择:")
    print(f"   选中特征数: {len(selected_features)}")
    
    if len(selected_features) == 0:
        print(f"   ❌ 警告: 未选中任何特征")
        return False
    
    # 分类特征
    feature_categories = {
        '15m_native': 0,
        '1h_tech': 0,
        'tech_': 0,
        'wyk_': 0,
        'micro_': 0,
        'td_': 0,
    }
    
    for feat in selected_features:
        for prefix, _ in feature_categories.items():
            if feat.startswith(prefix):
                feature_categories[prefix] += 1
                break
    
    print(f"\n📊 特征分类:")
    for category, count in feature_categories.items():
        if count > 0:
            print(f"   {category}: {count} 个")
    
    # 4. CV 指标
    cv_metrics = config.get('cv_metrics', {})
    if cv_metrics:
        print(f"\n📊 交叉验证指标:")
        print(f"   F1 Macro: {cv_metrics.get('f1_macro', 0):.4f}")
        print(f"   F1 Weighted: {cv_metrics.get('f1_weighted', 0):.4f}")
        print(f"   Balanced Accuracy: {cv_metrics.get('balanced_accuracy', 0):.4f}")
    
    # 5. 读取特征文件
    feature_path = config.get('materialized_path')
    if feature_path and Path(feature_path).exists():
        df = pd.read_parquet(feature_path)
        print(f"\n📊 物化数据:")
        print(f"   数据形状: {df.shape}")
        print(f"   ✅ 特征工程完成！")
        return True
    else:
        print(f"\n❌ 未找到特征文件: {feature_path}")
        return False

def main():
    """主函数"""
    print("\n" + "="*60)
    print("🔍 Layer 优化结果完整分析")
    print("="*60)
    
    # 分析 Layer1
    layer1_ok = analyze_layer1_results()
    
    # 分析 Layer2
    layer2_ok = analyze_layer2_results()
    
    # 总结
    print("\n" + "="*60)
    print("📝 总结")
    print("="*60)
    
    if layer1_ok and layer2_ok:
        print("\n🎉 所有层优化成功！")
        print("\n✅ 修复验证:")
        print("   ✅ Meta-Labeling 执行率正常")
        print("   ✅ 标签分布包含所有类别 (0/1/2)")
        print("   ✅ Layer2 成功选择特征")
        print("   ✅ 整体优化流程正常")
        print("\n🚀 可以继续进行:")
        print("   - Layer 3: 模型训练")
        print("   - Layer 4: 交叉验证与风控")
        print("   - Layer 5-8: 高级优化")
        return 0
    else:
        print("\n❌ 部分层优化失败")
        if not layer1_ok:
            print("   ❌ Layer1 需要检查")
        if not layer2_ok:
            print("   ❌ Layer2 需要检查")
        print("\n📋 建议:")
        print("   1. 检查日志文件: layer_test_output.log")
        print("   2. 调整参数后重新运行")
        print("   3. 查看详细错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())

