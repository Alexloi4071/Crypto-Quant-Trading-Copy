"""
配置管理器
統一管理所有配置和參數計算
"""

from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
from .parameter_calculator import ParameterCalculator
from config.symbol_profiles import SYMBOL_PROFILES
from config.timeframe_profiles import TIMEFRAME_PROFILES


class ConfigManager:

    def __init__(self):
        self._calculators = {}  # 緩存計算器

    def get_strategy_config(self, symbol: str, 
                            timeframe: str) -> Dict[str, Any]:
        """獲取策略配置"""
        key = f"{symbol}_{timeframe}"

        if key not in self._calculators:
            self._calculators[key] = ParameterCalculator(
                symbol, timeframe)

        return self._calculators[key].calculate_all_parameters()

    def get_available_symbols(self) -> List[str]:
        """獲取可用幣種列表"""
        return [symbol for symbol in SYMBOL_PROFILES.keys()
                if symbol != 'TEMPLATE']

    def get_available_timeframes(self) -> List[str]:
        """獲取可用時框列表"""
        return list(TIMEFRAME_PROFILES.keys())

    def validate_symbol_timeframe(self, symbol: str, timeframe: str
                                  ) -> bool:
        """驗證幣種和時框組合是否有效 (支援多目標優化驗證)"""
        basic_valid = (symbol in SYMBOL_PROFILES
                      and timeframe in TIMEFRAME_PROFILES)
        
        if not basic_valid:
            return False
            
        # 🔧 檢查多目標優化支援
        try:
            config = self.get_strategy_config(symbol, timeframe)
            directions = config.get('optimization_directions', [])
            # 若有directions且長度大於1，才允許進入多目標模式
            if directions and len(directions) > 1:
                print(f"✅ {symbol} {timeframe} 支援多目標優化: {directions}")
            return True
        except Exception as e:
            print(f"⚠️ {symbol} {timeframe} 配置驗證失敗: {e}")
            return False

    def print_all_configurations(self):
        """打印所有配置組合"""
        symbols = self.get_available_symbols()
        timeframes = self.get_available_timeframes()

        print("\n" + "=" * 80)
        print(f"所有可用配置組合 "
              f"({len(symbols)} 幣種 × {len(timeframes)} 時框 = "
              f"{len(symbols)*len(timeframes)} 組合)")
        print("=" * 80)

        for symbol in symbols:
            print(f"\n🪙 {symbol}:")
            for timeframe in timeframes:
                calculator = ParameterCalculator(symbol, timeframe)
                params = calculator.calculate_all_parameters()

                lag_range = params['label_lag_range']
                threshold_range = params['label_threshold_range']
                feature_range = params['n_features_range']
                trials = params['trials_config']

                print(f"  {timeframe:>3}: lag={lag_range}, "
                      f"threshold={threshold_range}, "
                      f"features={feature_range}, "
                      f"trials=({trials['layer1_total']},"
                      f"{trials['layer2_total']})")
    
    def batch_validate_configurations(self) -> Dict[str, Any]:
        """批量驗證所有配置組合"""
        print("🔍 開始批量驗證配置...")
        
        symbols = self.get_available_symbols()
        timeframes = self.get_available_timeframes()
        
        validation_results = {
            'total_combinations': len(symbols) * len(timeframes),
            'valid_combinations': 0,
            'invalid_combinations': [],
            'warnings': [],
            'summary': {},
            'detailed_results': {}
        }
        
        for symbol in symbols:
            validation_results['detailed_results'][symbol] = {}
            
            for timeframe in timeframes:
                try:
                    calculator = ParameterCalculator(symbol, timeframe)
                    params = calculator.calculate_all_parameters()
                    
                    # 驗證參數合理性
                    issues = self._validate_single_config(symbol, timeframe, params)
                    
                    if not issues:
                        validation_results['valid_combinations'] += 1
                        validation_results['detailed_results'][symbol][timeframe] = {
                            'status': 'valid',
                            'params': params
                        }
                    else:
                        validation_results['invalid_combinations'].append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'issues': issues
                        })
                        validation_results['detailed_results'][symbol][timeframe] = {
                            'status': 'invalid',
                            'issues': issues,
                            'params': params
                        }
                        
                except Exception as e:
                    error_info = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'error': str(e)
                    }
                    validation_results['invalid_combinations'].append(error_info)
                    validation_results['detailed_results'][symbol][timeframe] = {
                        'status': 'error',
                        'error': str(e)
                    }
        
        # 生成摘要
        validation_results['summary'] = {
            'success_rate': validation_results['valid_combinations'] / 
                           validation_results['total_combinations'] * 100,
            'error_count': len(validation_results['invalid_combinations']),
            'most_common_issues': self._analyze_common_issues(validation_results['invalid_combinations'])
        }
        
        self._print_validation_report(validation_results)
        return validation_results
    
    def _validate_single_config(self, symbol: str, timeframe: str, 
                                params: Dict[str, Any]) -> List[str]:
        """驗證單個配置的合理性"""
        issues = []
        
        # 檢查滯後期範圍
        lag_range = params.get('label_lag_range', (1, 10))
        if lag_range[0] < 1 or lag_range[1] > 150:  # 放寬至150週，支援極長週期分析（約3年）
            issues.append(f"滯後期範圍異常: {lag_range}")
        
        if lag_range[0] >= lag_range[1]:
            issues.append(f"滯後期範圍無效: min >= max")
        
        # 檢查閾值範圍  
        threshold_range = params.get('label_threshold_range', (0.01, 0.05))
        if threshold_range[0] <= 0 or threshold_range[1] > 1.0:  # 放寬至100%，支援極端波動場景
            issues.append(f"閾值範圍異常: {threshold_range}")
        
        # 檢查特徵數範圍
        feature_range = params.get('n_features_range', (20, 50))
        if feature_range[0] < 5 or feature_range[1] > 200:
            issues.append(f"特徵數範圍異常: {feature_range}")
        
        # 檢查CV折數
        cv_folds = params.get('cv_folds', 5)
        if cv_folds < 2 or cv_folds > 10:
            issues.append(f"CV折數異常: {cv_folds}")
        
        # 🔧 檢查兩階段多目標試驗配置
        trials_config = params.get('trials_config', {})
        required_keys = ['layer1_total', 'layer2_total', 'stage1_multi_trials', 'stage1_single_trials']
        missing_keys = [k for k in required_keys if k not in trials_config]
        if missing_keys:
            issues.append(f"trial配置不一致，缺少字段: {missing_keys}")
        else:
            if trials_config.get('layer1_total', 0) < 10:
                issues.append("第一層試驗數過少")
            if trials_config.get('layer2_total', 0) < 10:
                issues.append("第二層試驗數過少")
            
            # 檢查多目標與單目標試驗數加總是否合理
            stage1_total = trials_config.get('stage1_multi_trials', 0) + trials_config.get('stage1_single_trials', 0)
            layer1_total = trials_config.get('layer1_total', 0)
            if abs(stage1_total - layer1_total) > 5:  # 允許小幅誤差
                issues.append(f"stage1試驗數不一致: {stage1_total} != {layer1_total}")
        
        # 🔧 檢查Warm-Start種子參數
        seed_params = params.get('seed_params')
        if not seed_params:
            issues.append("缺少seed_params")
        else:
            required_seed_keys = ['label_lag', 'label_threshold', 'label_type', 'n_features']
            missing_seed_keys = [k for k in required_seed_keys if k not in seed_params]
            if missing_seed_keys:
                issues.append(f"seed_params結構不完整，缺少: {missing_seed_keys}")
        
        # 🔧 檢查多目標優化方向
        directions = params.get('optimization_directions', [])
        if not directions or len(directions) != 2:
            issues.append(f"多目標優化方向配置錯誤: 應為2個方向，實際: {directions}")
        
        return issues
    
    def _analyze_common_issues(self, invalid_combinations: List[Dict]) -> Dict[str, int]:
        """分析常見問題"""
        issue_counts = {}
        
        for combination in invalid_combinations:
            issues = combination.get('issues', [])
            for issue in issues:
                # 提取問題類型
                if '滯後期' in issue:
                    key = '滯後期問題'
                elif '閾值' in issue:
                    key = '閾值問題'
                elif '特徵數' in issue:
                    key = '特徵數問題'
                elif 'CV' in issue:
                    key = 'CV配置問題'
                elif '試驗' in issue:
                    key = '試驗數問題'
                else:
                    key = '其他問題'
                    
                issue_counts[key] = issue_counts.get(key, 0) + 1
        
        return issue_counts
    
    def _print_validation_report(self, results: Dict[str, Any]):
        """打印驗證報告"""
        print(f"\n📊 配置驗證報告")
        print("=" * 80)
        
        summary = results['summary']
        print(f"總配置數: {results['total_combinations']}")
        print(f"有效配置: {results['valid_combinations']}")
        print(f"問題配置: {len(results['invalid_combinations'])}")
        print(f"成功率: {summary['success_rate']:.1f}%")
        
        if results['invalid_combinations']:
            print(f"\n❌ 問題配置詳情:")
            for item in results['invalid_combinations'][:5]:  # 只顯示前5個
                symbol = item.get('symbol', 'Unknown')
                timeframe = item.get('timeframe', 'Unknown')
                if 'issues' in item:
                    issues_str = ', '.join(item['issues'])
                    print(f"   {symbol} {timeframe}: {issues_str}")
                else:
                    print(f"   {symbol} {timeframe}: {item.get('error', 'Unknown error')}")
        
        if summary['most_common_issues']:
            print(f"\n🔍 常見問題統計:")
            for issue_type, count in summary['most_common_issues'].items():
                print(f"   {issue_type}: {count} 次")
        
        print("=" * 80)
    
    def export_all_configurations_json(self, output_path: str = None) -> str:
        """導出所有配置到 JSON 文件"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"config/exported_configs_{timestamp}.json"
        
        all_configs = {}
        symbols = self.get_available_symbols()
        timeframes = self.get_available_timeframes()
        
        for symbol in symbols:
            all_configs[symbol] = {}
            for timeframe in timeframes:
                calculator = ParameterCalculator(symbol, timeframe)
                params = calculator.calculate_all_parameters()
                all_configs[symbol][timeframe] = params
        
        # 保存到文件
        import json
        from datetime import datetime
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_symbols': len(symbols),
            'total_timeframes': len(timeframes),
            'total_combinations': len(symbols) * len(timeframes),
            'configurations': all_configs
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, 
                     default=str)  # 處理特殊類型序列化
        
        print(f"📁 配置已導出到: {output_path}")
        return output_path
    
    def export_all_configurations_yaml(self, output_path: str = None) -> str:
        """導出所有配置到 YAML 文件"""
        try:
            import yaml
        except ImportError:
            print("⚠️ 需要安裝 PyYAML: pip install PyYAML")
            return ""
        
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"config/exported_configs_{timestamp}.yaml"
        
        all_configs = {}
        symbols = self.get_available_symbols()
        timeframes = self.get_available_timeframes()
        
        for symbol in symbols:
            all_configs[symbol] = {}
            for timeframe in timeframes:
                calculator = ParameterCalculator(symbol, timeframe)
                params = calculator.calculate_all_parameters()
                
                # 轉換為 YAML 友好格式
                yaml_params = self._convert_to_yaml_friendly(params)
                all_configs[symbol][timeframe] = yaml_params
        
        # 保存到文件
        from datetime import datetime
        
        export_data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'total_symbols': len(symbols),
                'total_timeframes': len(timeframes),
                'total_combinations': len(symbols) * len(timeframes)
            },
            'configurations': all_configs
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(export_data, f, default_flow_style=False, 
                     allow_unicode=True, sort_keys=True)
        
        print(f"📁 配置已導出到: {output_path}")
        return output_path
    
    def _convert_to_yaml_friendly(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """轉換參數為 YAML 友好格式"""
        yaml_params = {}
        
        for key, value in params.items():
            if isinstance(value, tuple):
                yaml_params[key] = list(value)  # tuple -> list
            elif isinstance(value, dict):
                yaml_params[key] = self._convert_to_yaml_friendly(value)
            else:
                yaml_params[key] = value
                
        return yaml_params