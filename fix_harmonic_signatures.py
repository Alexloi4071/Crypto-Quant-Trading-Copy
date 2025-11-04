# -*- coding: utf-8 -*-
"""批量修复harmonic_patterns.py的方法签名"""

import re

file_path = 'optuna_system/optimizers/indicators/harmonic_patterns.py'

# 读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 需要修复的方法列表（除了gartley已经修复）
methods_to_fix = [
    'detect_butterfly',
    'detect_bat',
    'detect_crab',
    'detect_shark',
    'detect_cypher',
    'detect_abcd',
    'detect_three_drives'
]

for method in methods_to_fix:
    # 查找方法签名并添加data_index参数
    # 旧模式：pattern_type: str) -> Tuple
    # 新模式：pattern_type: str, data_index: pd.Index) -> Tuple
    
    old_pattern = rf'(def {method}\(.*?pattern_type: str)\) -> Tuple'
    new_pattern = r'\1, data_index: pd.Index) -> Tuple'
    
    content = re.sub(old_pattern, new_pattern, content, flags=re.DOTALL)
    
    # 修复返回空Series的部分
    # 旧：return pd.Series(signals), pd.Series(strengths)
    # 新：return pd.Series(0, index=data_index), pd.Series(0.0, index=data_index)
    
    # 同时修复初始化signals和strengths的部分
    # 旧：signals = np.zeros(len(primary_swings[0].index) if primary_swings else 0)
    # 新：signals = np.zeros(len(data_index))

# 使用更精确的替换
# 替换所有 "if len(primary_swings) < X or len(secondary_swings) < Y:" 后的 return 语句
content = re.sub(
    r'if len\(primary_swings\) < \d+ or len\(secondary_swings\) < \d+:\s+return pd\.Series\([^)]+\), pd\.Series\([^)]+\)',
    lambda m: m.group(0).replace(
        'return pd.Series(signals), pd.Series(strengths)',
        'return pd.Series(0, index=data_index), pd.Series(0.0, index=data_index)'
    ).replace(
        'return pd.Series(np.zeros(len(primary_swings) if primary_swings else 0)), pd.Series(np.zeros(len(primary_swings) if primary_swings else 0))',
        'return pd.Series(0, index=data_index), pd.Series(0.0, index=data_index)'
    ),
    content,
    flags=re.MULTILINE | re.DOTALL
)

# 替换所有初始化signals和strengths的语句
content = re.sub(
    r'signals = np\.zeros\(len\(primary_swings\[0\]\.index\) if primary_swings else 0\)\s+strengths = np\.zeros\(len\(signals\)\)',
    'signals = np.zeros(len(data_index))\n        strengths = np.zeros(len(data_index))',
    content
)

# 也替换其他可能的初始化模式
content = re.sub(
    r'n_points = max\(\[sp\.index for sp in primary_swings \+ secondary_swings\]\) \+ 1 if \(primary_swings or secondary_swings\) else 0\s+signals = np\.zeros\(n_points\)\s+strengths = np\.zeros\(n_points\)',
    'signals = np.zeros(len(data_index))\n        strengths = np.zeros(len(data_index))',
    content
)

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed harmonic pattern method signatures")

