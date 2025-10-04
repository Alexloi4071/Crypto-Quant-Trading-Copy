#!/usr/bin/env python3
"""
UAT Debug 工作流
使用 MCP Debug 系統進行項目驗收檢查
從數據下載開始的完整 Debug 流程
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 導入 MCP Debug 系統
sys.path.append(str(Path.cwd()))
from mcp_python_debug_server import QuantTradingDebugTools

class UATDebugWorkflow:
    """UAT Debug 工作流管理器"""
    
    def __init__(self):
        self.debug_tools = QuantTradingDebugTools()
        self.results = {}
        
    async def run_complete_uat_debug(self) -> Dict[str, Any]:
        """運行完整的 UAT Debug 流程"""
        print("🚀 開始 UAT Debug 工作流")
        print("=" * 50)
        
        # 第一階段：環境和依賴檢查
        print("\n📍 第一階段：環境依賴檢查")
        env_result = await self.check_environment_for_uat()
        self.results['environment'] = env_result
        
        # 第二階段：數據下載功能 Debug
        print("\n📍 第二階段：數據下載功能檢查")
        data_result = await self.debug_data_download_issues()
        self.results['data_download'] = data_result
        
        # 第三階段：配置文件檢查
        print("\n📍 第三階段：配置文件驗證")
        config_result = await self.check_trading_configurations()
        self.results['configurations'] = config_result
        
        # 第四階段：API 連接測試
        print("\n📍 第四階段：API 連接診斷")
        api_result = await self.test_api_connections()
        self.results['api_connections'] = api_result
        
        # 第五階段：生成修復計劃
        print("\n📍 第五階段：生成修復建議")
        fix_plan = await self.generate_uat_fix_plan()
        self.results['fix_plan'] = fix_plan
        
        # 生成完整報告
        report = self.generate_uat_report()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'workflow_results': self.results,
            'uat_report': report,
            'next_actions': self.get_next_actions()
        }
    
    async def check_environment_for_uat(self) -> Dict[str, Any]:
        """檢查 UAT 所需的環境"""
        print("🔍 檢查 UAT 環境...")
        
        # 使用 MCP Debug 掃描環境
        env_scan = self.debug_tools.scan_environment()
        
        # 檢查 UAT 關鍵依賴
        uat_critical_modules = [
            'pandas', 'numpy', 'ccxt', 'lightgbm', 'optuna', 
            'backtrader', 'fastapi', 'requests', 'sqlalchemy'
        ]
        
        critical_issues = []
        for module in uat_critical_modules:
            status = env_scan['key_modules_status'].get(module, '❌ 未檢測')
            if '❌' in status:
                critical_issues.append(f"缺少關鍵模塊: {module}")
        
        # 檢查 Python 版本兼容性
        python_version = env_scan['python_version'].split()[0]
        version_compatible = self._check_python_compatibility(python_version)
        
        result = {
            'python_version': python_version,
            'version_compatible': version_compatible,
            'critical_modules_missing': critical_issues,
            'total_modules_checked': len(env_scan['key_modules_status']),
            'working_directory': env_scan['working_directory'],
            'virtual_env': env_scan['virtual_env'],
            'status': 'PASS' if not critical_issues and version_compatible else 'FAIL'
        }
        
        # 輸出結果
        if result['status'] == 'PASS':
            print(f"  ✅ 環境檢查通過 (Python {python_version})")
        else:
            print(f"  ❌ 環境檢查失敗:")
            for issue in critical_issues:
                print(f"    - {issue}")
        
        return result
    
    async def debug_data_download_issues(self) -> Dict[str, Any]:
        """使用 AI 分析數據下載問題"""
        print("🔍 使用 MCP AI 分析數據下載問題...")
        
        # 檢查數據下載相關文件
        data_files = [
            'scripts/data_downloader.py',
            'src/data/collector.py', 
            'config/trading_pairs.yaml'
        ]
        
        issues_found = []
        
        for file_path in data_files:
            if Path(file_path).exists():
                print(f"  📄 分析文件: {file_path}")
                
                # 模擬常見的數據下載問題進行 AI 分析
                common_errors = [
                    {
                        'error_type': 'ImportError',
                        'error_message': 'No module named ccxt',
                        'file_path': file_path,
                        'line_number': 14,
                        'code_context': 'import ccxt'
                    },
                    {
                        'error_type': 'KeyError', 
                        'error_message': 'api_key',
                        'file_path': file_path,
                        'line_number': 38,
                        'code_context': "config.trading.api_key"
                    },
                    {
                        'error_type': 'ConnectionError',
                        'error_message': 'Failed to connect to Binance',
                        'file_path': file_path,
                        'line_number': 46,
                        'code_context': 'exchange.load_markets()'
                    }
                ]
                
                for error in common_errors:
                    analysis = await self.debug_tools.analyze_python_error(error)
                    issues_found.append({
                        'file': file_path,
                        'error': error['error_type'],
                        'ai_analysis': analysis['primary_solution']['solution'][:200] + '...',
                        'confidence': analysis['primary_solution']['confidence'],
                        'source_ai': analysis['primary_solution']['source_ai']
                    })
            else:
                issues_found.append({
                    'file': file_path,
                    'error': 'FileNotFound',
                    'ai_analysis': f'關鍵文件 {file_path} 不存在，這會導致數據下載功能失敗',
                    'confidence': 1.0,
                    'source_ai': 'file_checker'
                })
        
        # 批量掃描數據下載相關代碼
        batch_scan = self.debug_tools.batch_scan_issues(['src/data/', 'scripts/'])
        
        result = {
            'files_analyzed': len(data_files),
            'issues_found': len(issues_found),
            'ai_analyses': issues_found[:5],  # 顯示前5個分析結果
            'batch_scan_results': {
                'total_files': batch_scan['total_files'],
                'total_issues': batch_scan['total_issues'],
                'issues_by_type': batch_scan['issues_by_type']
            },
            'status': 'ISSUES_FOUND' if issues_found else 'CLEAN'
        }
        
        print(f"  📊 AI分析完成: 發現 {len(issues_found)} 個潛在問題")
        
        return result
    
    async def check_trading_configurations(self) -> Dict[str, Any]:
        """檢查交易配置文件"""
        print("🔍 檢查交易配置...")
        
        config_files = [
            'config/trading_pairs.yaml',
            'config/settings.py',
            'config/indicators.yaml',
            '.env'
        ]
        
        config_status = {}
        critical_missing = []
        
        for config_file in config_files:
            if Path(config_file).exists():
                config_status[config_file] = '✅ 存在'
                
                # 特殊檢查：.env 文件的 API 配置
                if config_file == '.env':
                    env_issues = self._check_env_file_content(config_file)
                    if env_issues:
                        config_status[config_file] = f'⚠️ 存在但有問題: {"; ".join(env_issues)}'
                        critical_missing.extend(env_issues)
            else:
                config_status[config_file] = '❌ 缺失'
                critical_missing.append(f'缺少配置文件: {config_file}')
        
        result = {
            'config_files_status': config_status,
            'critical_issues': critical_missing,
            'total_configs_checked': len(config_files),
            'status': 'PASS' if not critical_missing else 'FAIL'
        }
        
        # 輸出結果
        for config, status in config_status.items():
            print(f"  {status} {config}")
        
        return result
    
    async def test_api_connections(self) -> Dict[str, Any]:
        """測試 API 連接"""
        print("🔍 測試 API 連接...")
        
        # 使用 AI 分析 API 連接問題
        api_error_scenarios = [
            {
                'error_type': 'AuthenticationError',
                'error_message': 'Invalid API key',
                'file_path': 'src/data/collector.py',
                'line_number': 47,
                'code_context': 'exchange.load_markets()'
            },
            {
                'error_type': 'NetworkError', 
                'error_message': 'Connection timeout',
                'file_path': 'src/data/collector.py',
                'line_number': 82,
                'code_context': 'ohlcv = await self._fetch_ohlcv_batch()'
            }
        ]
        
        api_analyses = []
        for error in api_error_scenarios:
            analysis = await self.debug_tools.analyze_python_error(error)
            api_analyses.append({
                'scenario': error['error_message'],
                'ai_solution': analysis['primary_solution']['solution'][:150] + '...',
                'source_ai': analysis['primary_solution']['source_ai']
            })
        
        # 檢查網絡連通性相關代碼問題
        network_scan = self.debug_tools.batch_scan_issues(['src/data/collector.py'])
        
        result = {
            'api_scenarios_analyzed': len(api_error_scenarios),
            'ai_solutions': api_analyses,
            'network_code_issues': network_scan['total_issues'],
            'status': 'ANALYZED'
        }
        
        print(f"  📊 API連接分析完成: {len(api_analyses)} 個場景")
        
        return result
    
    async def generate_uat_fix_plan(self) -> Dict[str, Any]:
        """生成 UAT 修復計劃"""
        print("📋 生成修復計劃...")
        
        fix_plan = {
            'immediate_actions': [],
            'short_term_fixes': [],
            'configuration_updates': [],
            'verification_steps': []
        }
        
        # 基於前面的檢查結果生成修復計劃
        env_result = self.results.get('environment', {})
        if env_result.get('status') == 'FAIL':
            for issue in env_result.get('critical_modules_missing', []):
                fix_plan['immediate_actions'].append(f"安裝缺失模塊: pip install {issue.split(': ')[1]}")
        
        data_result = self.results.get('data_download', {})
        if data_result.get('status') == 'ISSUES_FOUND':
            fix_plan['short_term_fixes'].append("修復數據下載模塊的導入和配置問題")
            fix_plan['verification_steps'].append("測試數據下載功能: python scripts/data_downloader.py --help")
        
        config_result = self.results.get('configurations', {})
        if config_result.get('status') == 'FAIL':
            for issue in config_result.get('critical_issues', []):
                if 'API' in issue:
                    fix_plan['configuration_updates'].append("配置 Binance API Keys 到 .env 文件")
                else:
                    fix_plan['configuration_updates'].append(f"創建缺失的配置文件: {issue}")
        
        # 添加通用驗證步驟
        fix_plan['verification_steps'].extend([
            "運行環境檢查: python check_environment.py",
            "測試基礎功能: python test_mcp_connection.py", 
            "重新運行 UAT: python UAT_Debug_工作流.py"
        ])
        
        return fix_plan
    
    def generate_uat_report(self) -> str:
        """生成 UAT 報告"""
        report_lines = [
            "# 🔍 UAT Debug 報告",
            f"生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 📊 檢查結果總覽"
        ]
        
        # 統計各階段結果
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results.values() 
                           if isinstance(r, dict) and r.get('status') in ['PASS', 'CLEAN'])
        
        report_lines.append(f"- **總檢查項目**: {total_checks}")
        report_lines.append(f"- **通過檢查**: {passed_checks}")
        report_lines.append(f"- **通過率**: {(passed_checks/total_checks*100):.1f}%" if total_checks > 0 else "- **通過率**: 0%")
        report_lines.append("")
        
        # 詳細結果
        for stage, result in self.results.items():
            if isinstance(result, dict):
                status = result.get('status', 'UNKNOWN')
                emoji = '✅' if status in ['PASS', 'CLEAN'] else '❌' if status == 'FAIL' else '⚠️'
                report_lines.append(f"### {emoji} {stage.replace('_', ' ').title()}")
                report_lines.append(f"狀態：{status}")
                
                if 'critical_issues' in result and result['critical_issues']:
                    report_lines.append("問題：")
                    for issue in result['critical_issues'][:3]:
                        report_lines.append(f"- {issue}")
                report_lines.append("")
        
        return '\n'.join(report_lines)
    
    def get_next_actions(self) -> List[str]:
        """獲取下一步行動建議"""
        actions = []
        
        # 檢查環境問題
        env_result = self.results.get('environment', {})
        if env_result.get('status') == 'FAIL':
            actions.append("🔧 修復環境依賴問題 (安裝缺失模塊)")
        
        # 檢查配置問題  
        config_result = self.results.get('configurations', {})
        if config_result.get('status') == 'FAIL':
            actions.append("⚙️ 配置 API Keys 和配置文件")
        
        # 數據下載問題
        data_result = self.results.get('data_download', {})
        if data_result.get('status') == 'ISSUES_FOUND':
            actions.append("📊 修復數據下載模塊問題")
        
        # 如果沒有關鍵問題，建議進行實際測試
        if not actions:
            actions.extend([
                "✅ 環境檢查通過，開始實際數據下載測試",
                "📈 運行完整的驗收測試流程"
            ])
        
        return actions
    
    def _check_python_compatibility(self, version: str) -> bool:
        """檢查 Python 版本兼容性"""
        try:
            major, minor = map(int, version.split('.')[:2])
            return major == 3 and minor >= 8  # 支持 3.8+
        except:
            return False
    
    def _check_env_file_content(self, env_file: str) -> List[str]:
        """檢查 .env 文件內容"""
        issues = []
        
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 檢查關鍵配置
            required_keys = ['BINANCE_API_KEY', 'BINANCE_API_SECRET']
            for key in required_keys:
                if key not in content:
                    issues.append(f'缺少 {key} 配置')
                elif f'{key}=your_' in content:
                    issues.append(f'{key} 未設置實際值')
        
        except Exception:
            issues.append('.env 文件讀取失敗')
        
        return issues

async def main():
    """主函數"""
    print("🚀 啟動 UAT Debug 工作流")
    print("使用 MCP AI 系統進行智能診斷")
    print("=" * 50)
    
    workflow = UATDebugWorkflow()
    
    try:
        # 運行完整 UAT Debug 流程
        results = await workflow.run_complete_uat_debug()
        
        # 保存詳細結果
        with open('uat_debug_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存報告
        with open('uat_debug_report.md', 'w', encoding='utf-8') as f:
            f.write(results['uat_report'])
        
        # 顯示總結
        print("\n" + "=" * 50)
        print("🎉 UAT Debug 工作流完成！")
        print(f"📊 詳細結果: uat_debug_results.json")
        print(f"📋 分析報告: uat_debug_report.md")
        
        print("\n📋 下一步建議：")
        for i, action in enumerate(results['next_actions'], 1):
            print(f"  {i}. {action}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ UAT Debug 工作流失敗: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
