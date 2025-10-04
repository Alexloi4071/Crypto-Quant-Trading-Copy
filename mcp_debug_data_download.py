#!/usr/bin/env python3
"""
使用MCP Debug系統分析數據下載問題
"""

import asyncio
import sys
from pathlib import Path

# 設置真實 API Keys
from real_api_config import setup_real_api_keys
setup_real_api_keys()

# 導入 MCP Debug 系統
sys.path.append(str(Path.cwd()))
from mcp_python_debug_server import QuantTradingDebugTools

async def analyze_data_download_issue():
    """使用MCP分析數據下載問題"""
    print("🤖 使用 MCP Debug 分析數據下載問題")
    print("=" * 50)
    
    debug_tools = QuantTradingDebugTools()
    
    # 分析問題：為什麼只下載了15,217條記錄而不是150萬條？
    data_issue = {
        'error_type': 'DataIncompleteError',
        'error_message': 'Downloaded only 15,217 records instead of ~1.5 million for 3 years of 1m data',
        'file_path': 'scripts/data_downloader.py',
        'line_number': 82,
        'code_context': '''
# 問題出現在這裡：
ohlcv = await self._fetch_ohlcv_batch(symbol, timeframe, current_ts, limit)

# Binance API 限制：每次最多1000條記錄
# 對於3年數據需要約1500次請求
# 可能的問題：
# 1. 請求次數限制
# 2. 速率限制導致中斷  
# 3. 時間戳計算錯誤
# 4. 循環提前結束
'''
    }
    
    print("🔍 MCP AI 分析中...")
    analysis = await debug_tools.analyze_python_error(data_issue)
    
    print("\n📊 MCP AI 分析結果：")
    print("-" * 30)
    print(f"AI來源: {analysis['primary_solution']['source_ai']}")
    print(f"信心度: {analysis['primary_solution']['confidence']:.2f}")
    print("\n🔧 主要解決方案：")
    print(analysis['primary_solution']['solution'])
    
    return analysis

async def debug_existing_downloader():
    """Debug現有的data_downloader.py代碼"""
    print("\n🔍 Debug 現有數據下載器代碼")
    print("=" * 30)
    
    debug_tools = QuantTradingDebugTools()
    
    # 掃描實際的數據下載器文件
    batch_results = debug_tools.batch_scan_issues(['scripts/data_downloader.py'])
    
    print(f"📊 掃描結果：")
    print(f"  文件數: {batch_results['total_files']}")
    print(f"  問題數: {batch_results['total_issues']}")
    
    if batch_results['total_issues'] > 0:
        print("🚨 發現的問題：")
        for issue_type, count in batch_results['issues_by_type'].items():
            if count > 0:
                print(f"  - {issue_type}: {count}")
    
    return batch_results

if __name__ == "__main__":
    asyncio.run(analyze_data_download_issue())
    asyncio.run(debug_existing_downloader())
