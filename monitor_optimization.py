#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
實時監控優化運行狀態
記錄BUG和性能問題
"""
import subprocess
import time
import json
from pathlib import Path
import signal
import sys


class OptimizationMonitor:
    """優化過程實時監控器"""
    
    def __init__(self):
        self.log_file = "optimization_monitor.log"
        self.bug_log = "bugs_detected.log"
        self.performance_log = "performance_monitor.json"
        
    def monitor_run(self, script_name="run_9layers_optimization.py"):
        """監控運行過程"""
        print("🔍 開始實時監控優化運行...")
        print(f"監控腳本: {script_name}")
        print(f"日志文件: {self.log_file}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # 啟動子進程
            process = subprocess.Popen(
                [sys.executable, script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            layer_status = {}
            current_layer = None
            
            # 實時讀取輸出
            for line in iter(process.stdout.readline, ''):
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] {line.strip()}")
                
                # 記錄到文件
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{timestamp}] {line}")
                
                # 檢測層級狀態
                if "Layer" in line and "優化" in line:
                    if "開始" in line:
                        current_layer = self._extract_layer_name(line)
                        layer_status[current_layer] = {"start_time": time.time(), "status": "running"}
                        print(f"📋 檢測到{current_layer}開始")
                    elif "完成" in line:
                        if current_layer:
                            layer_status[current_layer]["end_time"] = time.time()
                            layer_status[current_layer]["status"] = "completed"
                            duration = layer_status[current_layer]["end_time"] - layer_status[current_layer]["start_time"]
                            print(f"✅ {current_layer}完成，耗時{duration:.1f}秒")
                
                # 檢測錯誤
                if "ERROR" in line or "Exception" in line or "Traceback" in line:
                    self._log_bug(line, timestamp)
                    print(f"🚨 檢測到錯誤: {line.strip()}")
                
                # 檢測性能問題
                if "Trial" in line and "finished" in line:
                    self._log_performance(line, current_layer)
            
            # 等待進程結束
            return_code = process.wait()
            total_time = time.time() - start_time
            
            print(f"\n📊 運行完成")
            print(f"總耗時: {total_time:.1f}秒")
            print(f"返回碼: {return_code}")
            
            # 生成報告
            self._generate_report(layer_status, total_time, return_code)
            
        except KeyboardInterrupt:
            print("\n⚠️ 用戶中斷監控")
            process.terminate()
            
        except Exception as e:
            print(f"❌ 監控錯誤: {e}")
    
    def _extract_layer_name(self, line):
        """提取層級名稱"""
        if "Layer0" in line:
            return "Layer0"
        elif "Layer1" in line:
            return "Layer1"
        elif "Layer2" in line:
            return "Layer2"
        elif "Layer3" in line:
            return "Layer3"
        elif "Layer4" in line:
            return "Layer4"
        elif "Layer5" in line:
            return "Layer5"
        elif "Layer6" in line:
            return "Layer6"
        elif "Layer7" in line:
            return "Layer7"
        elif "Layer8" in line:
            return "Layer8"
        return "Unknown"
    
    def _log_bug(self, line, timestamp):
        """記錄BUG"""
        with open(self.bug_log, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] BUG: {line}\n")
    
    def _log_performance(self, line, layer):
        """記錄性能數據"""
        try:
            if "value:" in line:
                # 提取分數
                score_part = line.split("value:")[1].split(" ")[1]
                score = float(score_part)
                
                perf_data = {
                    "timestamp": time.time(),
                    "layer": layer,
                    "score": score,
                    "line": line.strip()
                }
                
                # 追加到性能日志
                if Path(self.performance_log).exists():
                    with open(self.performance_log, 'r') as f:
                        data = json.load(f)
                else:
                    data = []
                
                data.append(perf_data)
                
                with open(self.performance_log, 'w') as f:
                    json.dump(data, f, indent=2)
                    
        except Exception:
            pass  # 忽略解析錯誤
    
    def _generate_report(self, layer_status, total_time, return_code):
        """生成監控報告"""
        print("\n" + "="*60)
        print("📊 監控報告")
        print("="*60)
        
        for layer, status in layer_status.items():
            if status["status"] == "completed":
                duration = status["end_time"] - status["start_time"]
                print(f"{layer}: ✅ 完成 ({duration:.1f}秒)")
            else:
                print(f"{layer}: ❌ 未完成")
        
        print(f"\n總耗時: {total_time:.1f}秒")
        print(f"狀態碼: {'✅ 成功' if return_code == 0 else '❌ 失敗'}")
        
        # 檢查是否有BUG日志
        if Path(self.bug_log).exists():
            with open(self.bug_log, 'r', encoding='utf-8') as f:
                bugs = f.readlines()
            print(f"\n🚨 發現{len(bugs)}個BUG記錄")
            print("最近5個BUG:")
            for bug in bugs[-5:]:
                print(f"  {bug.strip()}")


if __name__ == "__main__":
    monitor = OptimizationMonitor()
    monitor.monitor_run()
