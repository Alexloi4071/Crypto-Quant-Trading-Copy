#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实时监控 Layer 优化进度"""

import time
from pathlib import Path
import re

def monitor_log_file(log_path="layer_test_output.log", refresh_seconds=2):
    """实时监控日志文件并显示关键信息"""
    
    print("=" * 60)
    print("🔍 Layer 优化进度监控")
    print("=" * 60)
    print(f"\n监控文件: {log_path}")
    print(f"刷新间隔: {refresh_seconds} 秒")
    print("\n按 Ctrl+C 停止监控\n")
    
    last_size = 0
    layer1_trials = 0
    layer2_trials = 0
    current_layer = "准备中"
    
    try:
        while True:
            log_file = Path(log_path)
            
            if not log_file.exists():
                print(f"⏳ 等待日志文件创建...")
                time.sleep(refresh_seconds)
                continue
            
            current_size = log_file.stat().st_size
            
            if current_size > last_size:
                # 读取新内容
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(last_size)
                    new_content = f.read()
                    last_size = current_size
                
                # 解析关键信息
                lines = new_content.split('\n')
                for line in lines:
                    # Layer 进度
                    if 'Layer0' in line and '跳過' in line:
                        current_layer = "Layer0 (跳过)"
                        print(f"✅ Layer0: 使用已有清洗数据")
                    
                    elif 'Layer1：標籤生成' in line or 'Layer1' in line:
                        current_layer = "Layer1 优化中"
                        print(f"\n{'='*60}")
                        print(f"🎯 开始 Layer1 优化...")
                        print(f"{'='*60}")
                    
                    elif 'primary_label' in line and 'Trial' in line and 'finished' in line:
                        layer1_trials += 1
                        # 提取分数
                        match = re.search(r'value: ([\d.]+)', line)
                        if match:
                            score = float(match.group(1))
                            print(f"  Primary Trial {layer1_trials}: score={score:.4f}")
                    
                    elif 'meta_quality' in line and 'Trial' in line and 'finished' in line:
                        # Meta trial
                        match = re.search(r'value: ([-\d.]+)', line)
                        if match:
                            score = float(match.group(1))
                            print(f"  Meta Trial: score={score:.4f}")
                    
                    elif 'Layer1' in line and '物化輸出' in line:
                        print(f"✅ Layer1 物化完成")
                        print(f"   Total trials: {layer1_trials}")
                    
                    elif 'Layer2：' in line or 'Layer2單一時框' in line:
                        current_layer = "Layer2 优化中"
                        print(f"\n{'='*60}")
                        print(f"🎯 开始 Layer2 优化...")
                        print(f"{'='*60}")
                    
                    elif 'feature_optimization_layer2' in line and 'Trial' in line and 'finished' in line:
                        layer2_trials += 1
                        match = re.search(r'value: ([\d.]+)', line)
                        if match:
                            score = float(match.group(1))
                            print(f"  Layer2 Trial {layer2_trials}: score={score:.4f}")
                    
                    elif 'Layer2' in line and '物化輸出' in line:
                        print(f"✅ Layer2 物化完成")
                        print(f"   Total trials: {layer2_trials}")
                    
                    elif 'Layer2特徵優化完成' in line or '✅ Layer2' in line:
                        print(f"\n{'='*60}")
                        print(f"🎉 优化完成!")
                        print(f"{'='*60}")
                        print(f"\n📊 最终统计:")
                        print(f"  Layer1 trials: {layer1_trials}")
                        print(f"  Layer2 trials: {layer2_trials}")
                        return
                    
                    # 错误检测
                    elif 'ERROR' in line or '❌' in line:
                        print(f"⚠️ 错误: {line.strip()}")
            
            else:
                # 显示当前状态
                print(f"\r🔄 {current_layer} | L1: {layer1_trials} trials | L2: {layer2_trials} trials", end='', flush=True)
            
            time.sleep(refresh_seconds)
            
    except KeyboardInterrupt:
        print(f"\n\n⏸️ 监控已停止")
        print(f"📊 当前进度: {current_layer}")
        print(f"   Layer1: {layer1_trials} trials")
        print(f"   Layer2: {layer2_trials} trials")

if __name__ == "__main__":
    print("\n⚠️ 请先在另一个终端运行:")
    print("   run_layer_test.bat")
    print("\n然后运行本脚本监控进度\n")
    
    input("按 Enter 开始监控...")
    monitor_log_file()

