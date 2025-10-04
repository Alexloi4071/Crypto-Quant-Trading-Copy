#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证Layer0-2修复是否正确
检查关键代码段是否包含修复标记
"""
import re
from pathlib import Path


def check_file_contains(file_path: Path, patterns: list, description: str) -> bool:
    """检查文件是否包含指定模式"""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        print(f"\n{'='*60}")
        print(f"检查: {description}")
        print(f"文件: {file_path}")
        print(f"{'='*60}")
        
        all_found = True
        for pattern_desc, pattern in patterns:
            if re.search(pattern, content, re.DOTALL):
                print(f"✅ {pattern_desc}")
            else:
                print(f"❌ {pattern_desc} - 未找到！")
                all_found = False
        
        return all_found
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def main():
    """主验证函数"""
    base_path = Path("optuna_system/optimizers")
    
    all_checks_passed = True
    
    # ========== Layer0检查 ==========
    layer0_file = base_path / "optuna_cleaning.py"
    layer0_patterns = [
        ("数据泄漏修复: 使用历史收益率", r"current_returns = cleaned_data\['close'\]\.pct_change\(\)"),
        ("数据泄漏修复: 无未来数据", r"cleaned_volatility = float\(current_returns_clean\.std\(\)\)"),
        ("参数优化: IQR倍数2.5-4.0", r"'iqr_multiplier'.*2\.5.*4\.0"),
        ("参数优化: jump_threshold 0.05-0.12", r"'jump_threshold'.*0\.05.*0\.12"),
    ]
    
    if not check_file_contains(layer0_file, layer0_patterns, "Layer0修复验证"):
        all_checks_passed = False
    
    # ========== Layer1检查 ==========
    layer1_file = base_path / "optuna_label.py"
    layer1_patterns = [
        ("核心修复: 计算未来收益率", r"future_returns = \(future_prices - price_data\) / price_data"),
        ("核心修复: 使用历史的未来收益率", r"historical_future_returns = future_returns\.shift\(1\)"),
        ("核心修复: 基于历史未来收益率计算阈值", r"rolling_upper = historical_future_returns\.rolling"),
        ("核心修复: 类型一致的标签分配", r"future_returns\.iloc\[valid_range\] > rolling_upper"),
        ("分位数修复: buy 0.70-0.85", r"'buy_quantile'.*0\.70.*0\.85"),
        ("分位数修复: sell 0.15-0.30", r"'sell_quantile'.*0\.15.*0\.30"),
        ("ATR实现", r"def calculate_atr\(self.*high.*low.*close"),
        ("Triple-Barrier: profit_multiplier", r"'profit_multiplier'.*1\.5.*3\.0"),
        ("Triple-Barrier: stop_multiplier", r"'stop_multiplier'.*0\.8.*1\.5"),
        ("Triple-Barrier: max_holding 16-24", r"'max_holding'.*16.*24"),
        ("Triple-Barrier: 风险收益比检查", r"actual_profit_distance / actual_stop_distance < 2\.0"),
    ]
    
    if not check_file_contains(layer1_file, layer1_patterns, "Layer1修复验证"):
        all_checks_passed = False
    
    # ========== 输出总结 ==========
    print(f"\n{'='*60}")
    print("验证总结")
    print(f"{'='*60}")
    
    if all_checks_passed:
        print("✅ 所有修复验证通过！")
        print("\n下一步:")
        print("1. 重新运行Layer0优化: python run_core_layers_only.py --layer 0 --n_trials 25")
        print("2. 重新运行Layer1优化: python run_core_layers_only.py --layer 1 --n_trials 150")
        print("3. 运行Layer3训练: python run_layer3_fixed.bat")
        return 0
    else:
        print("❌ 部分修复未通过验证！")
        print("\n请检查上述标记为❌的项目")
        return 1


if __name__ == "__main__":
    exit(main())

