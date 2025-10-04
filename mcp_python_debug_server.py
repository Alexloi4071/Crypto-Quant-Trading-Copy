#!/usr/bin/env python3
"""
MCP Python Debug 服務器
專為加密貨幣量化交易系統設計

此文件是 Cursor IDE 的 MCP 服務器，提供智能 Debug 功能
"""

import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import ast
import os
import importlib.util
from datetime import datetime
import re

# 檢查並安裝必要依賴


def check_and_install_dependencies():
    """檢查並安裝必要的依賴"""
    required_packages = [
        ('anthropic', 'anthropic>=0.7.0'),
        ('openai', 'openai>=1.0.0'),
        ('google.generativeai', 'google-generativeai>=0.3.0'),
        ('aiohttp', 'aiohttp>=3.8.0'),
        ('yaml', 'pyyaml>=6.0')
    ]

    missing_packages = []

    for module_name, pip_name in required_packages:
        try:
            if '.' in module_name:
                # 處理子模塊導入
                parent_module = module_name.split('.')[0]
                importlib.import_module(parent_module)
            else:
                importlib.import_module(module_name)
        except ImportError:
            missing_packages.append(pip_name)

    if missing_packages:
        print(f"🔧 正在安裝缺失的依賴包：{', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package],
                             check=True, capture_output=True)
                print(f"✅ 已安裝：{package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ 安裝失敗：{package} - {e}")

# 安裝依賴
check_and_install_dependencies()

# 現在導入所需的模塊
try:
    import anthropic
    import openai
    import google.generativeai as genai
    import aiohttp
    import yaml
    HAS_AI_LIBS = True
except ImportError as e:
    print(f"⚠️ AI 庫導入失敗：{e}")
    print("將使用模擬模式運行")
    HAS_AI_LIBS = False


class MultiAIDebugger:
    """多 AI 助手協同 Debug 系統"""

    def __init__(self):
        self.config = self._load_config()
        self.clients = self._init_ai_clients() if HAS_AI_LIBS else {}

    def _load_config(self) -> Dict:
        """載入配置文件"""
        config_file = Path("mcp_servers_config.yaml")
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"配置文件讀取失敗：{e}")

        # 默認配置
        return {
            'ai_backends': {
                'claude': {'weight': 0.4, 'speciality': 'deep_logic_analysis'},
                'gpt4': {'weight': 0.35, 'speciality': 'syntax_repair'},
                'gemini': {'weight': 0.25, 'speciality': 'quick_response'}
            }
        }


    def _init_ai_clients(self) -> Dict:
        """初始化 AI 客戶端"""
        if not HAS_AI_LIBS:
            return {}

        clients = {}

        # Claude 客戶端
        claude_key = os.getenv('CLAUDE_API_KEY')
        if claude_key and claude_key != 'your_claude_api_key_here':
            try:
                clients['claude'] = anthropic.Anthropic(api_key=claude_key)
            except Exception as e:
                print(f"Claude 初始化失敗：{e}")

        # OpenAI 客戶端
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key and openai_key != 'your_openai_api_key_here':
            try:
                clients['gpt4'] = openai.OpenAI(api_key=openai_key)
                print(f"✅ OpenAI GPT-4 已連接")
            except Exception as e:
                print(f"OpenAI 初始化失敗：{e}")

        # Gemini 客戶端
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key and gemini_key != 'your_gemini_api_key_here':
            try:
                genai.configure(api_key=gemini_key)
                clients['gemini'] = genai.GenerativeModel('gemini-1.5-flash')
                print(f"✅ Gemini 已連接")
            except Exception as e:
                print(f"Gemini 初始化失敗：{e}")

        return clients


    async def analyze_error(self, error_info: Dict) -> Dict:
        """多 AI 協同分析錯誤"""
        if not self.clients:
            return self._mock_analysis(error_info)

        prompt = self._build_debug_prompt(error_info)

        # 並行請求可用的 AI
        tasks = []
        if 'claude' in self.clients:
            tasks.append(self._query_claude(prompt, error_info))
        if 'gpt4' in self.clients:
            tasks.append(self._query_gpt4(prompt, error_info))
        if 'gemini' in self.clients:
            tasks.append(self._query_gemini(prompt, error_info))

        if not tasks:
            return self._mock_analysis(error_info)

        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return self._aggregate_responses(responses, error_info.get('error_type', 'Unknown'))
        except Exception as e:
            print(f"AI 分析失敗：{e}")
            return self._mock_analysis(error_info)


    def _build_debug_prompt(self, error_info: Dict) -> str:
        """構建結構化的 Debug Prompt"""
        return f"""
你是量化交易系統的資深 Python 工程師。請分析以下錯誤並提供修復方案：

## 錯誤詳情
- 錯誤類型：{error_info.get('error_type', 'Unknown')}
- 錯誤訊息：{error_info.get('error_message', '')}
- 發生位置：{error_info.get('file_path', '')}:{error_info.get('line_number', 0)}
- Python版本：{error_info.get('python_version', sys.version.split()[0])}

## 程式碼上下文
```python
{error_info.get('code_context', '')}
```

## 系統環境
- 專案類型：加密貨幣量化交易系統
- 主要依賴：pandas, numpy, lightgbm, ccxt, TA-Lib
- 模組範圍：{error_info.get('module_type', 'unknown')}

## 期望輸出
請提供：
1. **根本原因分析** (1-2句話簡潔說明)
2. **修復方案** (具體可執行的代碼改動)
3. **風險評估** (修改可能影響的其他部分)
4. **測試建議** (如何驗證修復效果)

請確保方案適用於量化交易系統的特殊需求。
"""


    async def _query_claude(self, prompt: str, error_info: Dict) -> Dict:
        """查詢 Claude"""
        try:
            message = await self.clients['claude'].messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return {
                'ai': 'claude',
                'response': message.content[0].text,
                'confidence': 0.9
            }
        except Exception as e:
            return {'ai': 'claude', 'error': str(e), 'confidence': 0.0}


    async def _query_gpt4(self, prompt: str, error_info: Dict) -> Dict:
        """查詢 GPT-4"""
        try:
            response = await self.clients['gpt4'].chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            return {
                'ai': 'gpt4',
                'response': response.choices[0].message.content,
                'confidence': 0.85
            }
        except Exception as e:
            return {'ai': 'gpt4', 'error': str(e), 'confidence': 0.0}


    async def _query_gemini(self, prompt: str, error_info: Dict) -> Dict:
        """查詢 Gemini"""
        try:
            response = self.clients['gemini'].generate_content(prompt)
            return {
                'ai': 'gemini',
                'response': response.text,
                'confidence': 0.8
            }
        except Exception as e:
            return {'ai': 'gemini', 'error': str(e), 'confidence': 0.0}


    def _aggregate_responses(self, responses: List[Dict], error_type: str) -> Dict:
        """聚合多個 AI 的回應"""
        valid_responses = [r for r in responses if isinstance(r, dict) and 'error' not in r]

        if not valid_responses:
            return {
                'error': 'All AI services failed',
                'suggestions': [],
                'mock_mode': True
            }

        return {
            'primary_solution': self._extract_primary_solution(valid_responses),
            'alternative_solutions': self._extract_alternatives(valid_responses),
            'consensus_confidence': self._calculate_consensus(valid_responses),
            'individual_responses': valid_responses,
            'timestamp': datetime.now().isoformat()
        }


    def _extract_primary_solution(self, responses: List[Dict]) -> Dict:
        """提取主要解決方案"""
        if not responses:
            return {'solution': 'No solution available', 'confidence': 0.0}

        # 選擇信心度最高的回應
        best_response = max(responses, key=lambda r: r.get('confidence', 0))
        return {
            'solution': best_response.get('response', ''),
            'confidence': best_response.get('confidence', 0.0),
            'source_ai': best_response.get('ai', 'unknown')
        }


    def _extract_alternatives(self, responses: List[Dict]) -> List[Dict]:
        """提取替代方案"""
        return [
            {
                'solution': r.get('response', ''),
                'source_ai': r.get('ai', 'unknown'),
                'confidence': r.get('confidence', 0.0)
            }
            for r in responses[1:3]  # 最多3個替代方案
        ]


    def _calculate_consensus(self, responses: List[Dict]) -> float:
        """計算共識度"""
        if not responses:
            return 0.0

        total_confidence = sum(r.get('confidence', 0.0) for r in responses)
        return total_confidence / len(responses)


    def _mock_analysis(self, error_info: Dict) -> Dict:
        """模擬分析（當 AI 服務不可用時）"""
        error_type = error_info.get('error_type', 'Unknown')

        mock_solutions = {
            'ImportError': {
                'solution': '導入錯誤通常是由於模塊未安裝。請檢查：\n1. pip install 相關模塊\n2. 確認虛擬環境已激活\n3. 檢查模塊名稱拼寫',
                'confidence': 0.8
            },
            'KeyError': {
                'solution': 'KeyError 表示字典中不存在該鍵。請檢查：\n1. 鍵名拼寫是否正確\n2. 數據結構是否符合預期\n3. 添加 .get() 方法防護',
                'confidence': 0.8
            },
            'AttributeError': {
                'solution': '屬性錯誤通常是對象沒有該屬性。請檢查：\n1. 對象類型是否正確\n2. 屬性名稱是否正確\n3. 對象是否已正確初始化',
                'confidence': 0.75
            }
        }

        solution = mock_solutions.get(error_type, {
            'solution': f'{error_type} 錯誤需要具體分析。請提供更多上下文信息。',
            'confidence': 0.6
        })

        return {
            'primary_solution': {
                'solution': solution['solution'],
                'confidence': solution['confidence'],
                'source_ai': 'mock_analyzer'
            },
            'alternative_solutions': [],
            'consensus_confidence': solution['confidence'],
            'mock_mode': True,
            'timestamp': datetime.now().isoformat()
        }


class QuantTradingDebugTools:
    """量化交易系統專用 Debug 工具"""

    def __init__(self):
        self.project_root = Path.cwd()
        self.debugger = MultiAIDebugger()

    def scan_environment(self) -> Dict:
        """掃描環境狀況"""
        return {
            'python_version': sys.version,
            'working_directory': str(self.project_root),
            'virtual_env': os.getenv('VIRTUAL_ENV'),
            'key_modules_status': self._check_key_modules(),
            'config_files_status': self._check_config_files(),
            'timestamp': datetime.now().isoformat()
        }


    def _check_key_modules(self) -> Dict:
        """檢查關鍵模組狀態"""
        key_modules = [
            'pandas', 'numpy', 'lightgbm', 'ccxt',
            'talib', 'backtrader', 'fastapi', 'optuna',
            'sklearn', 'tensorflow', 'torch'
        ]

        status = {}
        for module in key_modules:
            try:
                mod = importlib.import_module(module)
                version = getattr(mod, '__version__', 'Unknown')
                status[module] = f'✅ v{version}'
            except ImportError:
                status[module] = '❌ 未安裝'
            except Exception as e:
                status[module] = f'⚠️ 問題: {e}'

        return status


    def _check_config_files(self) -> Dict:
        """檢查配置文件狀態"""
        config_files = [
            '.env',
            'config/settings.py',
            'config/trading_pairs.yaml',
            'config/indicators.yaml',
            'requirements.txt',
            'mcp_servers_config.yaml'
        ]

        status = {}
        for file_path in config_files:
            path = self.project_root / file_path
            if path.exists():
                status[file_path] = '✅ 存在'
            else:
                status[file_path] = '❌ 缺失'

        return status


    async def analyze_python_error(self, error_data: Dict) -> Dict:
        """分析 Python 錯誤"""
        error_info = self._parse_error_data(error_data)

        # 添加系統上下文
        error_info.update({
            'python_version': sys.version.split()[0],
            'module_type': self._detect_module_type(error_info['file_path'])
        })

        # 使用多 AI 分析
        result = await self.debugger.analyze_error(error_info)

        return result


    def _parse_error_data(self, error_data: Dict) -> Dict:
        """解析錯誤數據"""
        return {
            'error_type': error_data.get('error_type', 'UnknownError'),
            'error_message': error_data.get('error_message', ''),
            'file_path': error_data.get('file_path', ''),
            'line_number': error_data.get('line_number', 0),
            'code_context': error_data.get('code_context', ''),
            'stack_trace': error_data.get('stack_trace', '')
        }


    def _detect_module_type(self, file_path: str) -> str:
        """檢測模塊類型"""
        if 'data/' in file_path or 'collector' in file_path:
            return 'data_processing'
        elif 'models/' in file_path:
            return 'machine_learning'
        elif 'trading/' in file_path:
            return 'trading_logic'
        elif 'api/' in file_path:
            return 'web_api'
        elif 'features/' in file_path:
            return 'feature_engineering'
        else:
            return 'general'


    def batch_scan_issues(self, scan_paths: List[str] = None) -> Dict:
        """批量掃描問題"""
        if scan_paths is None:
            scan_paths = [
                'src/', 'api/', 'scripts/', 'config/',
                'realtime/', 'advanced/', 'ai_enhanced/'
            ]

        issues = []
        total_files = 0

        for path_str in scan_paths:
            path = self.project_root / path_str
            if path.exists():
                issues_found, files_count = self._scan_directory(path)
                issues.extend(issues_found)
                total_files += files_count

        return {
            'total_files': total_files,
            'total_issues': len(issues),
            'issues_by_type': self._categorize_issues(issues),
            'issues': issues[:50],  # 限制返回數量
            'timestamp': datetime.now().isoformat()
        }


    def _scan_directory(self, directory: Path) -> tuple:
        """掃描目錄中的問題"""
        issues = []
        file_count = 0

        for py_file in directory.rglob('*.py'):
            try:
                file_count += 1
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # AST 解析檢查語法錯誤
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    issues.append({
                        'type': 'SyntaxError',
                        'file': str(py_file),
                        'line': e.lineno,
                        'message': str(e),
                        'severity': 'CRITICAL'
                    })

                # 檢查常見問題模式
                issues.extend(self._check_common_patterns(py_file, content))

            except Exception as e:
                issues.append({
                    'type': 'FileReadError',
                    'file': str(py_file),
                    'message': str(e),
                    'severity': 'MEDIUM'
                })

        return issues, file_count


    def _check_common_patterns(self, file_path: Path, content: str) -> List[Dict]:
        """檢查常見問題模式"""
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # 檢查硬編碼配置
            if any(keyword in line.lower() for keyword in ['api_key =', 'secret =', 'password =']):
                if 'your_' not in line and 'example' not in line:
                    issues.append({
                        'type': 'SecurityIssue',
                        'file': str(file_path),
                        'line': i,
                        'message': '可能的硬編碼敏感信息',
                        'severity': 'HIGH'
                    })

            # 檢查性能問題
            if '.iterrows()' in line:
                issues.append({
                    'type': 'PerformanceIssue',
                    'file': str(file_path),
                    'line': i,
                    'message': '使用 .iterrows() 可能影響性能',
                    'severity': 'MEDIUM'
                })

            # 檢查 TODO/FIXME 標記
            if any(marker in line.upper() for marker in ['TODO', 'FIXME', 'HACK']):
                issues.append({
                    'type': 'TodoFixme',
                    'file': str(file_path),
                    'line': i,
                    'message': line.strip(),
                    'severity': 'LOW'
                })

        return issues


    def _categorize_issues(self, issues: List[Dict]) -> Dict:
        """分類問題"""
        categories = {}
        for issue in issues:
            issue_type = issue['type']
            if issue_type not in categories:
                categories[issue_type] = 0
            categories[issue_type] += 1

        return categories


    def run_flake8(self, paths=None) -> Dict[str, Any]:
        """呼叫 flake8 並回傳解析後的錯誤列表"""
        if paths is None:
            paths = ["src/", "scripts/"]
        cmd = ["flake8", "--format=default"] + paths
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            issues = []
            for line in result.stdout.splitlines():
                # 格式: file.py:line:col: code message
                m = re.match(r"(.+?):(\d+):(\d+):\s+(\w+)\s+(.*)", line)
                if m:
                    issues.append({
                        "file": m.group(1),
                        "line": int(m.group(2)),
                        "col": int(m.group(3)),
                        "code": m.group(4),
                        "message": m.group(5)
                    })
            return {"tool": "flake8", "issues": issues, "total_issues": len(issues)}
        except FileNotFoundError:
            return {"tool": "flake8", "error": "flake8 未安裝或未找到", "issues": []}
        except Exception as e:
            return {"tool": "flake8", "error": str(e), "issues": []}


    def run_mypy(self, paths=None) -> Dict[str, Any]:
        """呼叫 mypy 並回傳解析後的錯誤列表"""
        if paths is None:
            paths = ["src/", "scripts/"]
        cmd = ["mypy", "--show-error-codes"] + paths
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            issues = []
            for line in result.stdout.splitlines():
                # 格式: file.py:line: error: message  [error-code]
                m = re.match(r"(.+?):(\d+):\s+error:\s+(.*)\s+\[(.*)\]", line)
                if m:
                    issues.append({
                        "file": m.group(1),
                        "line": int(m.group(2)),
                        "message": m.group(3),
                        "code": m.group(4)
                    })
            return {"tool": "mypy", "issues": issues, "total_issues": len(issues)}
        except FileNotFoundError:
            return {"tool": "mypy", "error": "mypy 未安裝或未找到", "issues": []}
        except Exception as e:
            return {"tool": "mypy", "error": str(e), "issues": []}


# 簡化的 MCP 服務器實現


class SimpleMCPServer:
    """簡化的 MCP 服務器"""

    def __init__(self):
        self.debug_tools = QuantTradingDebugTools()
        print("🤖 MCP Python Debug 服務器已啟動")
        print(f"📁 工作目錄: {Path.cwd()}")
        print(f"🐍 Python 版本: {sys.version.split()[0]}")

    async def handle_request(self, request: Dict) -> Dict:
        """處理請求"""
        try:
            method = request.get('method', '')
            params = request.get('params', {})

            if method == 'scan_environment':
                return self.debug_tools.scan_environment()

            elif method == 'analyze_error':
                return await self.debug_tools.analyze_python_error(params)

            elif method == 'batch_scan':
                return self.debug_tools.batch_scan_issues(
                    params.get('scan_paths')
                )

            elif method == 'static_check':
                # 執行靜態檢查
                flake8_result = self.debug_tools.run_flake8(params.get('paths'))
                mypy_result = self.debug_tools.run_mypy(params.get('paths'))
                return {
                    "flake8": flake8_result,
                    "mypy": mypy_result,
                    "total_issues": flake8_result.get("total_issues", 0) + mypy_result.get("total_issues", 0)
                }

            elif method == 'chat':
                # 使用多 AI 聯合回答自然語言提問
                question = params.get("question", "")
                if not question:
                    return {"error": "請提供問題內容", "chat_response": ""}

                # 重用 MultiAIDebugger 來處理通用 Chat
                chat_error_info = {
                    "error_type": "ChatRequest",
                    "error_message": question,
                    "file_path": "",
                    "line_number": 0,
                    "code_context": ""
                }
                responses = await self.debug_tools.debugger.analyze_error(chat_error_info)
                return {"chat_response": responses.get("primary_solution", {}).get("solution", "抱歉，無法生成回應")}

            else:
                return {
                    'error': f'Unknown method: {method}',
                    'available_methods': [
                        'scan_environment',
                        'analyze_error',
                        'batch_scan',
                        'static_check',
                        'chat'
                    ]
                }

        except Exception as e:
            return {
                'error': f'Request handling failed: {str(e)}',
                'traceback': traceback.format_exc()
            }


    def run_interactive_mode(self):
        """運行交互模式（用於測試）"""
        print("\n🎮 MCP Debug 交互模式")
        print("可用命令：")
        print("  scan    - 掃描環境")
        print("  batch   - 批量掃描問題")
        print("  test    - 測試錯誤分析")
        print("  static  - 靜態代碼檢查 (flake8 + mypy)")
        print("  chat    - 自然語言對話模式")
        print("  quit    - 退出")

        while True:
            try:
                command = input("\n> ").strip().lower()

                if command == 'quit':
                    break
                elif command == 'scan':
                    result = self.debug_tools.scan_environment()
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                elif command == 'batch':
                    result = self.debug_tools.batch_scan_issues()
                    print(f"掃描結果：{result['total_issues']} 個問題")
                    for issue in result['issues'][:5]:
                        print(f"  - {issue['type']}: {issue['file']}:{issue.get('line', '?')}")
                elif command == 'test':
                    print("執行錯誤分析測試...")
                    test_error = {
                        'error_type': 'ImportError',
                        'error_message': 'No module named talib',
                        'file_path': 'src/features/indicators.py',
                        'line_number': 15,
                        'code_context': 'import talib'
                    }
                    # 使用同步的mock分析而不是異步
                    result = self.debug_tools.debugger._mock_analysis(test_error)
                    print("🤖 AI 分析結果：")
                    print(f"解決方案: {result['primary_solution']['solution']}")
                    print(f"信心度: {result['primary_solution']['confidence']}")
                    print(f"來源: {result['primary_solution']['source_ai']}")
                elif command == 'static':
                    print("執行靜態檢查...")
                    # 直接調用靜態檢查方法而不是通過async handler
                    flake8_result = self.debug_tools.run_flake8()
                    mypy_result = self.debug_tools.run_mypy()

                    flake8_issues = flake8_result.get('total_issues', 0)
                    mypy_issues = mypy_result.get('total_issues', 0)
                    total_issues = flake8_issues + mypy_issues

                    print(f"\n📊 靜態檢查結果：共 {total_issues} 個問題")
                    print(f"  - flake8: {flake8_issues} 個問題")
                    print(f"  - mypy: {mypy_issues} 個問題")

                    # 顯示前5個flake8問題
                    if flake8_issues > 0:
                        print("\n🔍 flake8 問題:")
                        for issue in flake8_result.get('issues', [])[:5]:
                            print(f"  {issue['file']}:{issue['line']}:{issue['col']}: {issue['code']} {issue['message']}")

                    # 顯示前5個mypy問題
                    if mypy_issues > 0:
                        print("\n🔍 mypy 問題:")
                        for issue in mypy_result.get('issues', [])[:5]:
                            print(f"  {issue['file']}:{issue['line']}: {issue['message']} [{issue['code']}]")

                elif command == 'chat':
                    question = input("\n請輸入您的問題：\n> ")
                    if question.strip():
                        print("🤖 AI 正在分析您的問題...")
                        # 使用同步方式處理聊天
                        if self.debug_tools.debugger.clients:
                            print("✅ AI 後端可用，正在生成回應...")
                            print("💡 提示：完整的AI功能請使用 Cursor IDE 或 API 調用")
                        else:
                            print("⚠️ 未檢測到AI後端，使用基礎回應")

                        # 提供基礎的聊天回應
                        basic_responses = {
                            "量化交易": "量化交易是使用數學模型和統計方法來分析金融市場並執行交易策略的方法。",
                            "特徵工程": "特徵工程是從原始數據中提取和構造有意義特徵的過程，對機器學習模型性能至關重要。",
                            "回測": "回測是使用歷史數據來測試交易策略表現的過程。",
                            "風險管理": "風險管理包括識別、評估和控制投資風險的方法和策略。"
                        }

                        # 簡單關鍵詞匹配
                        response = "這是一個關於量化交易系統的問題。建議查看項目文檔或使用完整的AI功能。"
                        for keyword, answer in basic_responses.items():
                            if keyword in question:
                                response = answer
                                break

                        print(f"\n🤖 基礎回覆：\n{response}")
                    else:
                        print("請輸入有效的問題")
                else:
                    print("未知命令，請輸入 scan, batch, test, static, chat 或 quit")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"錯誤：{e}")

        print("👋 再見！")


async def main():
    """主函數"""
    server = SimpleMCPServer()

    # 檢查命令行參數
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            # 測試模式
            print("🧪 運行測試模式")
            result = server.debug_tools.scan_environment()
            print("✅ 環境掃描測試通過")
            print(f"Python: {result['python_version'].split()[0]}")
            print(f"工作目錄: {result['working_directory']}")
            return
        elif sys.argv[1] == '--interactive':
            # 交互模式
            server.run_interactive_mode()
            return

    # 默認：顯示服務器信息
    print("✅ MCP Python Debug 服務器準備就緒")
    print("\n📋 可用功能：")
    print("  - scan_environment: 掃描環境狀態")
    print("  - analyze_error: 分析 Python 錯誤")
    print("  - batch_scan: 批量掃描問題")
    print("  - static_check: flake8 + mypy 靜態檢查")
    print("  - chat: 自然語言對話模式")
    print("\n🚀 使用方式：")
    print("  - 在 Cursor IDE 中：@mcp <your_request>")
    print("  - 測試模式：python mcp_python_debug_server.py --test")
    print("  - 交互模式：python mcp_python_debug_server.py --interactive")
    print("\n💡 交互模式命令：scan, batch, test, static, chat, quit")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 服務器已停止")
    except Exception as e:
        print(f"❌ 服務器錯誤：{e}")
        traceback.print_exc()
