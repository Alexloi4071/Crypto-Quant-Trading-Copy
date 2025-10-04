#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查最新的Layer1和Layer2文件"""

import pandas as pd
from pathlib import Path
import os
from datetime import datetime

def check_files():
    configs = Path('configs')
    
    print("=" * 60)
    print("=== Layer1 标签文件检查 ===")
    print("=" * 60)
    label_files = sorted(configs.glob('label*.parquet'), key=lambda x: os.path.getmtime(x), reverse=True)
    if label_files:
        for f in label_files[:5]:
            size_kb = os.path.getsize(f) / 1024
            mtime = datetime.fromtimestamp(os.path.getmtime(f))
            df = pd.read_parquet(f)
            print(f"✅ {f.name}")
            print(f"   大小: {size_kb:.1f}KB")
            print(f"   修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   形状: {df.shape}")
            if 'label' in df.columns:
                print(f"   标签分布: {df['label'].value_counts().to_dict()}")
            print()
    else:
        print("❌ 未找到标签文件")
    
    print("=" * 60)
    print("=== Layer2 特征文件检查 ===")
    print("=" * 60)
    feat_files = sorted(configs.glob('*feature*.parquet'), key=lambda x: os.path.getmtime(x), reverse=True)
    if feat_files:
        for f in feat_files[:5]:
            size_kb = os.path.getsize(f) / 1024
            mtime = datetime.fromtimestamp(os.path.getmtime(f))
            df = pd.read_parquet(f)
            print(f"✅ {f.name}")
            print(f"   大小: {size_kb:.1f}KB")
            print(f"   修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   形状: {df.shape} (样本数 × 特征数)")
            print()
    else:
        print("❌ 未找到特征文件")
    
    print("=" * 60)
    print("=== Layer0 清洗文件检查 ===")
    print("=" * 60)
    clean_files = list(configs.glob('cleaned*.parquet'))
    if clean_files:
        for f in clean_files[:3]:
            size_kb = os.path.getsize(f) / 1024
            mtime = datetime.fromtimestamp(os.path.getmtime(f))
            print(f"✅ {f.name}")
            print(f"   大小: {size_kb:.1f}KB")
            print(f"   修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
    else:
        print("⚠️ 未找到清洗文件（如需计算收益率）")
    
    # 检查是否可以开始Layer3
    print("=" * 60)
    print("=== Layer3 准备状态 ===")
    print("=" * 60)
    if label_files and feat_files:
        print("✅ 可以开始Layer3优化！")
        print(f"   将使用: {label_files[0].name}")
        print(f"   将使用: {feat_files[0].name}")
        return True
    else:
        print("❌ 缺少必要文件，无法开始Layer3")
        return False

if __name__ == "__main__":
    check_files()

