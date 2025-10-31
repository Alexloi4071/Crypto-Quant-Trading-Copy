"""实时监控优化进度"""
import time
import re
from pathlib import Path

log_file = Path("optuna_system/logs/coordinator.log")
if not log_file.exists():
    print(f"❌ 日志文件不存在: {log_file}")
    print("等待程序启动...")

last_size = 0
trial_count = 0
layer_status = {"Layer1": "等待中", "Layer2": "等待中"}

print("=" * 60)
print("🔍 实时监控 - Layer1 & Layer2 优化")
print("=" * 60)
print("\n按 Ctrl+C 停止监控（不会中断优化）\n")

try:
    while True:
        if log_file.exists():
            current_size = log_file.stat().st_size
            
            if current_size > last_size:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(last_size)
                    new_lines = f.readlines()
                    
                    for line in new_lines:
                        # 检测 Trial 完成
                        if 'Trial' in line and 'finished' in line:
                            trial_match = re.search(r'Trial (\d+)', line)
                            if trial_match:
                                trial_num = int(trial_match.group(1))
                                trial_count = trial_num + 1
                                
                                # 提取分数
                                score_match = re.search(r'value[:\s]+([0-9.]+)', line)
                                score = score_match.group(1) if score_match else 'N/A'
                                
                                print(f"✅ Trial {trial_num} 完成 | 分数: {score}")
                        
                        # 检测层状态
                        if 'Layer1' in line and '標籤生成' in line:
                            layer_status['Layer1'] = "运行中"
                            print("\n" + "="*60)
                            print("🎯 Layer1 (Meta-Labeling) 开始优化")
                            print("="*60)
                        
                        if 'Layer1' in line and '物化' in line:
                            layer_status['Layer1'] = "完成"
                            print("\n✅ Layer1 完成并物化")
                        
                        if 'Layer2' in line and '特徵優化' in line:
                            layer_status['Layer2'] = "运行中"
                            print("\n" + "="*60)
                            print("🎯 Layer2 (Feature Engineering) 开始优化")
                            print("="*60)
                        
                        # 检测错误
                        if 'ERROR' in line or '失败' in line or '錯誤' in line:
                            print(f"❌ 错误: {line.strip()}")
                
                last_size = current_size
            
            # 显示进度摘要
            print(f"\r⏳ 状态: Layer1[{layer_status['Layer1']}] | Layer2[{layer_status['Layer2']}] | Trials: {trial_count}  ", end='', flush=True)
        
        time.sleep(2)  # 每2秒检查一次

except KeyboardInterrupt:
    print("\n\n" + "="*60)
    print("🛑 监控已停止（优化继续在后台运行）")
    print("="*60)
    print(f"\n最终状态:")
    print(f"  Layer1: {layer_status['Layer1']}")
    print(f"  Layer2: {layer_status['Layer2']}")
    print(f"  完成 Trials: {trial_count}")
    print(f"\n查看完整日志: {log_file}")

