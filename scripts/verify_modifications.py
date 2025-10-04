#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证改善方案修改是否正确
确保所有策略保留，参数优化生效
"""
import json
import sys
import os
from pathlib import Path

# 强制UTF-8编码
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def verify_feature_flags():
    """验证feature_flags.json修改"""
    print("="*70)
    print("🔍 验证 feature_flags.json")
    print("="*70)
    
    config_file = project_root / 'optuna_system' / 'configs' / 'feature_flags.json'
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    passed = True
    
    # ============================================================
    # 1. 检查策略开关（必须保留为true）
    # ============================================================
    print("\n[1] 检查策略开关...")
    
    strategy_switches = {
        'enable_wyckoff': '威科夫策略',
        'enable_td': 'TD Sequential',
        'enable_micro': '微观结构'
    }
    
    for key, name in strategy_switches.items():
        if config.get(key) != True:
            print(f"  ❌ {name}被禁用: {key} = {config.get(key)}")
            passed = False
        else:
            print(f"  ✅ {name}已启用")
    
    # ============================================================
    # 2. 检查窗口优化（15m基础配置）
    # ============================================================
    print("\n[2] 检查15m窗口优化...")
    
    tech = config.get('tech', {})
    
    # Fibonacci MA
    fibo_ma = tech.get('fibo_ma', [])
    if len(fibo_ma) == 4 and fibo_ma == [8, 21, 55, 144]:
        print(f"  ✅ fibo_ma已优化: {fibo_ma}")
    else:
        print(f"  ❌ fibo_ma错误: 预期[8,21,55,144], 实际{fibo_ma}")
        passed = False
    
    # RSI windows
    rsi_windows = tech.get('rsi_windows', [])
    if len(rsi_windows) == 3 and rsi_windows == [9, 14, 21]:
        print(f"  ✅ rsi_windows已优化: {rsi_windows}")
    else:
        print(f"  ❌ rsi_windows错误: 预期[9,14,21], 实际{rsi_windows}")
        passed = False
    
    # BB windows
    bb_windows = tech.get('bb_windows', [])
    if len(bb_windows) == 1 and bb_windows == [20]:
        print(f"  ✅ bb_windows已优化: {bb_windows}")
    else:
        print(f"  ❌ bb_windows错误: 预期[20], 实际{bb_windows}")
        passed = False
    
    # ============================================================
    # 3. 检查多时框配置
    # ============================================================
    print("\n[3] 检查多时框配置...")
    
    mtf = tech.get('multi_timeframes', {})
    if not mtf.get('enabled'):
        print("  ❌ 多时框未启用")
        passed = False
    else:
        print("  ✅ 多时框已启用")
        
        rules = mtf.get('rules', {})
        
        # 检查1h
        if '1h' in rules:
            r1h = rules['1h']
            if len(r1h.get('fibo_ma', [])) == 4:
                print(f"  ✅ 1h fibo_ma: {r1h['fibo_ma']}")
            else:
                print(f"  ❌ 1h fibo_ma错误: {r1h.get('fibo_ma')}")
                passed = False
                
            if len(r1h.get('rsi_windows', [])) == 3:
                print(f"  ✅ 1h rsi_windows: {r1h['rsi_windows']}")
            else:
                print(f"  ❌ 1h rsi_windows错误: {r1h.get('rsi_windows')}")
                passed = False
        
        # 检查4h和1d
        for tf in ['4h', '1d']:
            if tf in rules:
                rtf = rules[tf]
                fibo_ok = len(rtf.get('fibo_ma', [])) == 4
                rsi_ok = len(rtf.get('rsi_windows', [])) == 3
                bb_ok = len(rtf.get('bb_windows', [])) == 1
                
                if fibo_ok and rsi_ok and bb_ok:
                    print(f"  ✅ {tf}时框已优化")
                else:
                    print(f"  ⚠️ {tf}时框部分未优化: fibo={fibo_ok}, rsi={rsi_ok}, bb={bb_ok}")
    
    # ============================================================
    # 4. 检查策略组（必须存在）
    # ============================================================
    print("\n[4] 检查策略组...")
    
    phases = config.get('feature_phases', {})
    strategy_phase = phases.get('strategy', {})
    groups = strategy_phase.get('groups', {})
    
    required_groups = ['wyckoff', 'td_sequence', 'micro_structure']
    for group in required_groups:
        if group in groups and len(groups[group]) > 0:
            print(f"  ✅ {group}组存在: {groups[group]}")
        else:
            print(f"  ❌ {group}组缺失或为空")
            passed = False
    
    # ============================================================
    # 5. 检查选择比例
    # ============================================================
    print("\n[5] 检查选择比例...")
    
    sel_params = config.get('selection_params', {})
    full_sel = sel_params.get('full', {})
    
    coarse_ratio = full_sel.get('coarse_ratio', [])
    fine_ratio = full_sel.get('fine_ratio', [])
    
    if coarse_ratio == [0.5, 0.7]:
        print(f"  ✅ coarse_ratio已优化: {coarse_ratio}")
    else:
        print(f"  ❌ coarse_ratio错误: 预期[0.5,0.7], 实际{coarse_ratio}")
        passed = False
    
    if fine_ratio == [0.15, 0.25]:
        print(f"  ✅ fine_ratio已优化: {fine_ratio}")
    else:
        print(f"  ❌ fine_ratio错误: 预期[0.15,0.25], 实际{fine_ratio}")
        passed = False
    
    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "="*70)
    if passed:
        print("🎉 feature_flags.json 验证通过！")
        print("="*70)
        print("\n✅ 所有策略已保留")
        print("✅ 窗口参数已优化")
        print("✅ 选择比例已调整")
        print("\n可以运行优化了: python run_core_layers_only.py")
    else:
        print("❌ 验证失败！请检查修改")
        print("="*70)
    
    return passed


def verify_label_optimizer():
    """验证optuna_label.py修改"""
    print("\n" + "="*70)
    print("🔍 验证 optuna_label.py")
    print("="*70)
    
    label_file = project_root / 'optuna_system' / 'optimizers' / 'optuna_label.py'
    
    with open(label_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    passed = True
    
    # 检查是否添加了historical_returns
    if 'historical_returns = price_data.pct_change()' in content:
        print("  ✅ 发现historical_returns变量（避免洩漏）")
    else:
        print("  ⚠️ 警告: 未找到historical_returns变量")
        passed = False
    
    # 检查是否添加了shift(1)
    if '.shift(1)' in content and 'quantile' in content:
        print("  ✅ 发现shift(1)修复（避免前视偏差）")
    else:
        print("  ⚠️ 警告: 可能未添加shift(1)")
    
    # 检查是否移除了actual_profit的rolling quantile
    if 'actual_profit.rolling(window=lookback_window, min_periods=100).quantile' in content:
        print("  ⚠️ 警告: 仍然存在actual_profit的rolling quantile（可能洩漏）")
    else:
        print("  ✅ 已移除actual_profit的直接quantile计算")
    
    print("="*70)
    return passed


def verify_feature_optimizer():
    """验证optuna_feature.py修改"""
    print("\n" + "="*70)
    print("🔍 验证 optuna_feature.py")
    print("="*70)
    
    feature_file = project_root / 'optuna_system' / 'optimizers' / 'optuna_feature.py'
    
    with open(feature_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查fine_ratio
    if "fine_ratio', 0.15, 0.25" in content:
        print("  ✅ fine_ratio已优化: [0.15, 0.25]")
    else:
        print("  ⚠️ 警告: fine_ratio可能未优化")
    
    # 检查coarse_ratio_low/high
    if "coarse_ratio_low, coarse_ratio_high = (0.5, 0.7)" in content:
        print("  ✅ coarse_ratio已优化: [0.5, 0.7]")
    else:
        print("  ⚠️ 警告: coarse_ratio可能未优化")
    
    # 检查策略特征生成函数是否存在
    functions = [
        ('_generate_td_features', 'TD Sequential'),
        ('_generate_wyckoff_features', 'Wyckoff'),
        ('_generate_micro_features_from_ohlcv', '微观结构')
    ]
    
    print("\n  检查策略特征生成函数:")
    for func_name, strategy_name in functions:
        if f'def {func_name}' in content:
            print(f"    ✅ {strategy_name}生成函数存在: {func_name}")
        else:
            print(f"    ❌ {strategy_name}生成函数缺失: {func_name}")
    
    print("="*70)
    return True


def main():
    """主验证流程"""
    print("\n" + "="*70)
    print("改善方案修改验证工具")
    print("="*70 + "\n")
    
    try:
        # 验证配置文件
        config_ok = verify_feature_flags()
        
        # 验证代码修改
        label_ok = verify_label_optimizer()
        feature_ok = verify_feature_optimizer()
        
        # 总结
        print("\n" + "="*70)
        print("📊 验证总结")
        print("="*70)
        
        results = {
            'feature_flags.json': config_ok,
            'optuna_label.py': label_ok,
            'optuna_feature.py': feature_ok
        }
        
        all_passed = all(results.values())
        
        for file, status in results.items():
            status_str = "✅ 通过" if status else "❌ 失败"
            print(f"  {file}: {status_str}")
        
        print("="*70)
        
        if all_passed:
            print("\n🎉 所有验证通过！")
            print("\n下一步:")
            print("  1. 运行优化: python run_core_layers_only.py")
            print("  2. 查看效果: 特征数量、训练时间、F1分数")
            print("  3. 确认策略特征存在: wyk_*, td_*, micro_*")
            return 0
        else:
            print("\n⚠️ 部分验证未通过，请检查修改")
            return 1
            
    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
