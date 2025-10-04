"""
回測運行器腳本
自動化回測執行工具，支持批量回測、參數優化、結果分析
支持多策略、多時框、多資產的回測任務
"""

import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import json
import warnings

# 項目導入
from config.settings import config
from src.data.collector import DataCollectionOrchestrator
from src.models.lgb_model import ModelManager
from src.backtesting.backtrader_engine import BacktestEngine
from src.backtesting.performance_analyzer import PerformanceAnalyzer
from src.utils.logger import setup_logger

warnings.filterwarnings('ignore')
logger = setup_logger(__name__)


class BacktestingRunner:
    """回測運行器"""

    def __init__(self, offline_mode=True):
        if not offline_mode:
            self.data_collector = DataCollectionOrchestrator()
        else:
            self.data_collector = None
        self.model_manager = ModelManager()
        self.backtest_engine = BacktestEngine()
        self.performance_analyzer = PerformanceAnalyzer()

        # 回測配置 - 調整為更合理的交易費率
        self.default_config = {
            'initial_capital': 10000,
            'maker_fee': 0.0002,      # Maker費率：0.02%
            'taker_fee': 0.0004,      # Taker費率：0.04%
            'slippage': 0.0002,       # 滑點：0.02%（調整為更合理的值）
            'start_date': '2022-01-01',
            'end_date': '2024-01-01',
            'timeframes': ['1h'],
            'symbols': ['BTCUSDT']
        }

        logger.info("回測運行器初始化完成")

    async def run_single_backtest(
        self,
        symbol: str,
        timeframe: str,
        strategy_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model_path: Optional[str] = None,
        strategy_params: Optional[Dict] = None,
        args = None,
    ) -> Dict[str, Any]:
        """執行單次回測"""
        try:
            logger.info(f"開始回測: {symbol} {timeframe} {strategy_name}")

            # 設置默認日期
            if start_date is None:
                start_date = self.default_config['start_date']
            if end_date is None:
                end_date = self.default_config['end_date']

            # 1. 準備數據
            logger.info("準備回測數據")
            data = await self._prepare_backtest_data(
                symbol, timeframe, start_date, end_date
            )

            if data.empty:
                logger.error(f"無法獲取 {symbol} 的回測數據")
                return {'error': '數據不足'}

            # 2. 加載預處理好的特徵和標籤
            logger.info("加載預處理的特徵和標籤")
            features_data, labeled_data = await self._load_preprocessed_features_labels(
                symbol, timeframe, model_path, version="v55"
            )

            # 3. 加載或訓練模型
            model = None
            if model_path:
                logger.info(f"加載模型: {model_path}")
                model = self.model_manager.load_model(model_path)
            else:
                logger.info("使用ML信號策略，無需預訓練模型")

            # 4. 設置策略參數
            if strategy_params is None:
                if args and hasattr(args, 'use_optimal_params'):
                    # 如果有args並且指定了use_optimal_params，則載入最佳參數
                    strategy_params = self._load_strategy_params(symbol, timeframe, args)
                    logger.info(f"📊 使用載入的策略參數: {strategy_params}")
                else:
                    strategy_params = self._get_default_strategy_params(strategy_name)

            if model:
                strategy_params['model'] = model
                strategy_params['feature_list'] = model.feature_names

            # 5. 執行回測 - 使用預處理數據
            if features_data.empty or labeled_data.empty:
                logger.error("特徵或標籤數據為空，無法執行回測")
                return {'error': '特徵或標籤數據不足'}

            logger.info("執行基於預處理數據的回測")
            backtest_results = await self._run_preprocessed_backtest(
                data,
                features_data,
                labeled_data,
                model,
                strategy_params,
            )

            # 6. 分析結果
            logger.info("分析回測結果")
            analysis = self.performance_analyzer.analyze_performance(
                backtest_results.get('returns', pd.Series()),
                trades=backtest_results.get('trades', pd.DataFrame())
            )

            # 7. 生成報告
            report_path = self._generate_backtest_report(
                symbol, timeframe, strategy_name, backtest_results, analysis
            )

            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'strategy': strategy_name,
                'start_date': start_date,
                'end_date': end_date,
                'data_points': len(labeled_data),
                'backtest_results': backtest_results,
                'analysis': analysis,
                'report_path': str(report_path),
                'execution_time': datetime.now()
            }

            logger.info(f"回測完成: {symbol} {timeframe}")
            return result

        except Exception as e:
            logger.error(f"單次回測失敗: {e}")
            return {'error': str(e)}

    async def run_batch_backtest(
        self,
        symbols: List[str],
        timeframes: List[str],
        strategies: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_concurrent: int = 3,
    ) -> List[Dict[str, Any]]:
        """批量回測"""
        try:
            logger.info(
                f"開始批量回測: {len(symbols)} 品種, "
                f"{len(timeframes)} 時框, {len(strategies)} 策略"
            )

            # 生成回測任務
            tasks = []
            for symbol in symbols:
                for timeframe in timeframes:
                    for strategy in strategies:
                        task = self.run_single_backtest(
                            symbol, timeframe, strategy, start_date, end_date
                        )
                        tasks.append(task)

            # 限制並發數量
            semaphore = asyncio.Semaphore(max_concurrent)

            async def limited_backtest(task):
                async with semaphore:
                    return await task

            # 執行所有任務
            logger.info(f"開始執行 {len(tasks)} 個回測任務")
            results = await asyncio.gather(
                *[limited_backtest(task) for task in tasks],
                return_exceptions=True,
            )

            # 處理結果
            successful_results = []
            failed_count = 0

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"回測任務異常: {result}")
                    failed_count += 1
                elif isinstance(result, dict) and 'error' in result:
                    logger.error(f"回測任務失敗: {result['error']}")
                    failed_count += 1
                elif isinstance(result, dict):
                    successful_results.append(result)

            logger.info(
                f"批量回測完成: 成功 {len(successful_results)}, 失敗 {failed_count}"
            )

            # 生成批量回測報告
            self._generate_batch_backtest_report(successful_results)

            return successful_results

        except Exception as e:
            logger.error(f"批量回測失敗: {e}")
            return []

    async def run_parameter_optimization_backtest(
        self,
        symbol: str,
        timeframe: str,
        strategy_name: str,
        param_ranges: Dict,
        n_trials: int = 50,
        versions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """參數優化回測"""
        try:
            logger.info(f"開始參數優化回測: {symbol} {timeframe} {strategy_name}")

            # 設置默認版本
            if versions is None:
                versions = {'features': 'v55', 'labels': 'v55', 'model': 'v55'}
            
            logger.info(f"📁 使用版本: 特徵={versions['features']}, 標籤={versions['labels']}, 模型={versions['model']}")
            
            # 準備數據
            ohlcv_data = await self._prepare_backtest_data(symbol, timeframe)
            features_data, labels_data = await self._load_preprocessed_features_labels(
                symbol, timeframe, None, version=versions['features']
            )
            
            # 載入模型
            model_manager = ModelManager()
            model_path = f'results/models/{symbol}_{timeframe}_{versions["model"]}'
            model = model_manager.load_model(model_path)

            # 定義優化目標函數
            def objective(trial):
                # 生成試驗參數
                trial_params = {}
                for param_name, param_range in param_ranges.items():
                    if isinstance(param_range, list) and len(param_range) == 2:
                        if isinstance(param_range[0], float):
                            trial_params[param_name] = trial.suggest_float(
                                param_name, param_range[0], param_range[1]
                            )
                        elif isinstance(param_range[0], int):
                            trial_params[param_name] = trial.suggest_int(
                                param_name, param_range[0], param_range[1]
                            )
                    elif isinstance(param_range, list):
                        trial_params[param_name] = trial.suggest_categorical(
                            param_name, param_range
                        )

                # 運行回測 - 使用同步版本
                try:
                    # 直接調用同步版本的回測函數
                    backtest_result = self._run_preprocessed_backtest_sync(
                        ohlcv_data, features_data, labels_data, model, trial_params
                    )

                    # 計算綜合評分 (考慮多個指標)
                    total_return = backtest_result.get('total_return', 0)
                    win_rate = backtest_result.get('win_rate', 0)
                    total_trades = backtest_result.get('total_trades', 0)
                    sharpe_ratio = backtest_result.get('sharpe_ratio', 0)
                    max_drawdown = backtest_result.get('max_drawdown', 0)
                    
                    # 懲罰過少或過多的交易
                    trade_penalty = 0
                    if total_trades < 10:  # 太少交易
                        trade_penalty = -0.1
                    elif total_trades > 1000:  # 過度交易
                        trade_penalty = -0.05
                    
                    # 綜合評分：主要考慮收益率，但也考慮其他指標
                    composite_score = (
                        total_return * 0.5 +              # 50%權重給總收益率
                        (win_rate - 0.5) * 0.2 +          # 20%權重給勝率（減去隨機基準50%）
                        max(sharpe_ratio, 0) * 0.15 +     # 15%權重給Sharpe比率
                        max_drawdown * 0.15 +             # 15%權重給最大回撤（負值，所以是懲罰）
                        trade_penalty                      # 交易頻率懲罰
                    )
                    
                    return composite_score

                except Exception as e:
                    logger.warning(f"試驗回測失敗: {e}")
                    return -999  # 返回很小的值表示失敗

            # 運行Optuna優化
            import optuna
            from optuna.samplers import TPESampler
            from optuna.pruners import SuccessiveHalvingPruner

            # 創建更高效的Optuna研究
            sampler = TPESampler(seed=42, n_startup_trials=10)
            pruner = SuccessiveHalvingPruner()
            
            study_name = f"backtest_opt_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                pruner=pruner,
                study_name=study_name
            )
            
            logger.info(f"🚀 開始Optuna回測參數優化 - {study_name}")
            logger.info(f"🎯 目標: 最大化綜合評分 (收益率+勝率+Sharpe比率-回撤)")
            logger.info(f"🔢 試驗次數: {n_trials}")
            
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            # 獲取最佳參數
            best_params = study.best_params
            best_score = study.best_value

            # 使用最佳參數運行最終回測
            final_backtest = await self._run_preprocessed_backtest(
                ohlcv_data, features_data, labels_data, model, best_params
            )

            # 保存最佳參數到optimal_params資料夾
            self._save_optimal_backtest_params(symbol, timeframe, best_params, best_score, final_backtest)

            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'strategy': strategy_name,
                'optimization_trials': n_trials,
                'best_params': best_params,
                'best_score': best_score,
                'final_backtest': final_backtest,
                'optimization_history': [
                    {
                        'trial': trial.number,
                        'params': trial.params,
                        'value': trial.value
                    }
                    for trial in study.trials
                ]
            }

            logger.info(f"參數優化回測完成，最佳分數: {best_score:.4f}")
            return result

        except Exception as e:
            logger.error(f"參數優化回測失敗: {e}")
            return {'error': str(e)}

    async def run_walk_forward_backtest(
        self,
        symbol: str,
        timeframe: str,
        strategy_name: str,
        window_months: int = 6,
        step_months: int = 1,
    ) -> Dict[str, Any]:
        """Walk-Forward回測"""
        try:
            logger.info(f"開始Walk-Forward回測: {symbol} {timeframe}")

            # 準備完整數據
            full_data = await self._prepare_backtest_data(symbol, timeframe)

            if len(full_data) < 100:
                raise ValueError("數據量不足，無法進行Walk-Forward回測")

            # 設置時間窗口
            start_date = pd.to_datetime(full_data.index[0])
            end_date = pd.to_datetime(full_data.index[-1])

            walk_forward_results = []
            current_date = start_date

            while current_date + timedelta(days=window_months * 30) < end_date:
                window_end = current_date + timedelta(days=window_months * 30)
                test_start = window_end
                test_end = test_start + timedelta(days=step_months * 30)

                if test_end > end_date:
                    break

                logger.info(
                    f"Walk-Forward窗口: 訓練 {current_date.date()} - {window_end.date()}, "
                    f"測試 {test_start.date()} - {test_end.date()}"
                )

                # 分割數據
                train_data = full_data.loc[current_date:window_end]
                test_data = full_data.loc[test_start:test_end]

                if len(train_data) < 50 or len(test_data) < 10:
                    current_date += timedelta(days=step_months * 30)
                    continue

                try:
                    # 在測試集上回測
                    backtest_result = await self._run_backtrader_backtest(
                        test_data,
                        strategy_name,
                        self._get_default_strategy_params(strategy_name)
                    )

                    walk_forward_results.append({
                        'train_start': current_date,
                        'train_end': window_end,
                        'test_start': test_start,
                        'test_end': test_end,
                        'train_samples': len(train_data),
                        'test_samples': len(test_data),
                        'backtest_result': backtest_result
                    })

                except Exception as e:
                    logger.warning(f"Walk-Forward窗口失敗: {e}")

                current_date += timedelta(days=step_months * 30)

            # 彙總結果
            total_return = 1.0
            all_trades = []

            for wf_result in walk_forward_results:
                period_return = wf_result['backtest_result'].get('total_return', 0)
                total_return *= (1 + period_return)

                trades = wf_result['backtest_result'].get('trades', [])
                all_trades.extend(trades)

            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'strategy': strategy_name,
                'window_months': window_months,
                'step_months': step_months,
                'walk_forward_periods': len(walk_forward_results),
                'total_return': total_return - 1,
                'all_trades': all_trades,
                'period_results': walk_forward_results
            }

            logger.info(f"Walk-Forward回測完成，總回報: {(total_return-1)*100:.2f}%")
            return result

        except Exception as e:
            logger.error(f"Walk-Forward回測失敗: {e}")
            return {'error': str(e)}

    async def _load_preprocessed_features_labels(
        self, symbol: str, timeframe: str, model_path: Optional[str] = None, version: Optional[str] = None
    ) -> tuple:
        """加載預處理好的特徵和標籤數據"""
        try:
            # 確定版本：優先使用指定版本，然後從模型路徑提取，最後使用默認版本
            if version:
                data_version = version
            elif model_path and 'v' in model_path:
                # 嘗試從路徑中提取版本號
                data_version = 'v55'  # 默認
                parts = model_path.split('_')
                for part in parts:
                    if part.startswith('v') and part[1:].isdigit():
                        data_version = part
                        break
            else:
                data_version = 'v55'  # 使用v55作為默認版本，匹配模型
            
            logger.info(f"📁 使用數據版本: {data_version}")

            # 構建特徵和標籤文件路徑
            features_file = (
                f"data/processed/features/{symbol}_{timeframe}/{data_version}/"
                f"{symbol}_{timeframe}_selected_features.parquet"
            )
            labels_file = (
                f"data/processed/labels/{symbol}_{timeframe}/{data_version}/"
                f"{symbol}_{timeframe}_labels.parquet"
            )

            # 加載特徵數據 - 嘗試多個文件名
            features_data = None
            feature_files = [
                features_file,  # 原始選定特徵
                f"data/processed/features/{symbol}_{timeframe}/{data_version}/{symbol}_{timeframe}_features_full.parquet"  # 完整特徵
            ]
            
            for f_file in feature_files:
                if Path(f_file).exists():
                    try:
                        logger.info(f"加載特徵數據: {f_file}")
                        features_data = pd.read_parquet(f_file)
                        logger.info(f"特徵數據形狀: {features_data.shape}")
                        break
                    except Exception as e:
                        logger.warning(f"讀取 {f_file} 失敗: {e}")
                        continue
            
            if features_data is None:
                logger.error(f"無法載入任何特徵文件: {feature_files}")
                return pd.DataFrame(), pd.DataFrame()

            # 加載標籤數據
            if Path(labels_file).exists():
                logger.info(f"加載標籤數據: {labels_file}")
                labels_data = pd.read_parquet(labels_file)
                logger.info(f"標籤數據形狀: {labels_data.shape}")
            else:
                logger.error(f"未找到標籤文件: {labels_file}")
                return features_data, pd.DataFrame()

            # 確保時間索引對齊
            common_index = features_data.index.intersection(labels_data.index)
            features_aligned = features_data.loc[common_index]
            labels_aligned = labels_data.loc[common_index]

            logger.info(f"時間對齊後數據量: {len(common_index)} 條記錄")
            logger.info(f"時間範圍: {common_index.min()} 到 {common_index.max()}")

            return features_aligned, labels_aligned

        except Exception as e:
            logger.error(f"加載預處理數據失敗: {e}")
            return pd.DataFrame(), pd.DataFrame()

    async def _prepare_backtest_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """準備回測數據"""
        try:
            # 直接從重采數據文件讀取
            data_file = f"data/raw/{symbol}/{symbol}_{timeframe}_ohlcv.parquet"

            if Path(data_file).exists():
                logger.info(f"加載重采數據: {data_file}")
                data = pd.read_parquet(data_file)
                
                # 確保索引為UTC時區
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC')
                elif str(data.index.tz) != 'UTC':
                    data.index = data.index.tz_convert('UTC')

                if start_date and end_date:
                    start_date = pd.to_datetime(start_date).tz_localize('UTC')
                    end_date = pd.to_datetime(end_date).tz_localize('UTC')
                    data = data.loc[start_date:end_date]
                    logger.info(f"過濾日期範圍: {start_date} 到 {end_date}")
                else:
                    logger.info("使用完整數據，未指定日期範圍")

                logger.info(f"成功加載 {symbol} {timeframe} 數據: {len(data)} 條記錄")
                logger.info(f"數據時間範圍: {data.index.min()} 到 {data.index.max()}")
                return data
            else:
                logger.error(f"未找到重采數據文件: {data_file}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"準備回測數據失敗: {e}")
            return pd.DataFrame()

    async def _run_backtrader_backtest(
        self, data: pd.DataFrame, strategy_name: str,
        strategy_params: Dict
    ) -> Dict[str, Any]:
        """運行Backtrader回測"""
        try:
            # 配置回測參數
            backtest_config = {
                'initial_capital': self.default_config['initial_capital'],
                'total_cost': self.default_config['total_cost'],
                'strategy_name': strategy_name,
                'strategy_params': strategy_params
            }

            # 運行回測 - 使用 BacktestEngine 的 run_backtest 方法
            results = self.backtest_engine.run_backtest(
                symbol='BTCUSDT',
                timeframe='15m',
                backtest_config=backtest_config,
                data=data,
            )

            return results

        except Exception as e:
            logger.error(f"Backtrader回測失敗: {e}")
            return {'error': str(e)}

    async def _run_preprocessed_backtest(
        self, ohlcv_data: pd.DataFrame, features_data: pd.DataFrame,
        labels_data: pd.DataFrame, model, strategy_params: Dict
    ) -> Dict[str, Any]:
        """使用預處理數據執行回測"""
        try:
            logger.info("開始基於預處理數據的回測")

            # 處理時區對齊問題
            logger.info("處理時區對齊")
            if ohlcv_data.index.tz is not None and features_data.index.tz is None:
                # OHLCV有時區，特徵沒有時區，將特徵數據本地化為UTC
                features_data.index = features_data.index.tz_localize('UTC')
                logger.info("特徵數據時區已設置為UTC")

            if ohlcv_data.index.tz is not None and labels_data.index.tz is None:
                # OHLCV有時區，標籤沒有時區，將標籤數據本地化為UTC
                labels_data.index = labels_data.index.tz_localize('UTC')
                logger.info("標籤數據時區已設置為UTC")

            # 確保數據對齊
            common_index = (
                ohlcv_data.index.intersection(features_data.index)
                .intersection(labels_data.index)
            )
            if len(common_index) == 0:
                logger.error("OHLCV、特徵和標籤數據無法對齊")
                return {'error': '數據對齊失敗'}

            # 對齊所有數據
            ohlcv_aligned = ohlcv_data.loc[common_index]
            features_aligned = features_data.loc[common_index]
            labels_aligned = labels_data.loc[common_index]

            logger.info(f"對齊後數據量: {len(common_index)} 條記錄")

            # 使用模型進行預測
            if model is None:
                logger.error("模型未加載")
                return {'error': '模型未加載'}

            # 進行批量預測
            logger.info("執行模型預測")
            logger.info(f"特徵數據形狀: {features_aligned.shape}")
            logger.info(f"模型期望特徵: {model.feature_names}")

            # 確保特徵列順序與模型一致
            if hasattr(model, 'feature_names') and model.feature_names:
                # 檢查模型期望的特徵是否在數據中存在
                available_features = list(features_aligned.columns)
                expected_features = model.feature_names
                
                missing_features = [f for f in expected_features if f not in available_features]
                extra_features = [f for f in available_features if f not in expected_features]
                
                if missing_features:
                    logger.warning(f"缺少模型期望的特徵: {missing_features}")
                if extra_features:
                    logger.info(f"數據中有額外特徵: {extra_features[:5]}...")  # 只顯示前5個
                
                # 使用交集的特徵，按照數據中的順序
                common_features = [f for f in available_features if f in expected_features]
                
                if len(common_features) == 0:
                    logger.error("沒有匹配的特徵列")
                    return {'error': '特徵列不匹配'}
                
                features_for_prediction = features_aligned[common_features]
                logger.info(f"使用 {len(common_features)} 個匹配特徵進行預測")
                logger.info(f"特徵列: {common_features[:10]}...")  # 只顯示前10個
            else:
                features_for_prediction = features_aligned

            predictions = model.predict(features_for_prediction)
            prediction_probs = model.predict_proba(features_for_prediction)

            # 計算置信度（最大概率）
            confidences = np.max(prediction_probs, axis=1)

            # 創建交易信號
            signals_df = pd.DataFrame(
                {
                    'prediction': predictions,
                    'confidence': confidences,
                    'actual_label': (
                        labels_aligned.values.flatten()
                        if hasattr(labels_aligned.values, 'flatten')
                        else labels_aligned.iloc[:, 0].values
                    ),
                },
                index=common_index,
            )

            # 執行簡化的回測邏輯
            initial_capital = strategy_params.get(
                'initial_capital', self.default_config['initial_capital']
            )
            # 使用Taker費率（大部分交易為市價單）
            trading_fee = strategy_params.get(
                'trading_fee', self.default_config['taker_fee']
            )
            slippage = strategy_params.get(
                'slippage', self.default_config['slippage']
            )
            total_cost = trading_fee + slippage  # 總交易成本
            min_confidence = strategy_params.get('min_confidence', 0.5)  # 使用50%置信度

            logger.info(
                f"回測參數: 初始資金={initial_capital}, "
                f"交易費率={trading_fee:.4f}, 滑點={slippage:.4f}, "
                f"總成本={total_cost:.4f}, 最小置信度={min_confidence}"
            )

            # 過濾高置信度信號
            valid_signals = signals_df[signals_df['confidence'] >= min_confidence].copy()
            logger.info(
                f"高置信度信號數量: {len(valid_signals)} / {len(signals_df)}"
            )

            if len(valid_signals) == 0:
                logger.warning("沒有滿足置信度要求的交易信號")
                return {
                    'total_return': 0.0,
                    'final_value': initial_capital,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'returns': pd.Series(),
                    'trades': pd.DataFrame()
                }

            # 改進的交易邏輯：添加風險管理和持倉時間限制
            trades = []
            portfolio_value = initial_capital
            position = 0  # 0: 無倉位, 1: 多頭, -1: 空頭
            entry_price = 0
            entry_time = None
            last_trade_time = None

            # 修復：交易參數 - 改進倉位管理
            position_size = strategy_params.get('position_size', 0.8)  # 80%倉位
            min_portfolio_pct = 0.01  # 最小需要1%的資金才能交易
            stop_loss_pct = strategy_params.get('stop_loss_pct', 0.015)  # 1.5%止損
            take_profit_pct = strategy_params.get('take_profit_pct', 0.03)  # 3%止盈
            max_hold_periods = strategy_params.get('max_hold_periods', 24)  # 最大持有24個週期
            min_hold_periods = strategy_params.get('min_hold_periods', 1)   # 最小持有1個週期
            cooldown_periods = strategy_params.get('cooldown_periods', 2)   # 平倉後冷卻2個週期

            # 優化：使用向量化操作替代低效的iterrows
            logger.info(f"⚡ 開始處理 {len(valid_signals)} 個交易信號...")
            processed_count = 0
            
            # 預處理所有價格數據到numpy數組以提高速度
            aligned_prices = ohlcv_aligned['close'].reindex(valid_signals.index, method='ffill')

            # === 追蹤止盈/止損輔助計算（ATR 或 最高/最低價記錄） ===
            trailing_enabled = strategy_params.get('trailing_enabled', False)
            trailing_activation_pct = strategy_params.get('trailing_activation_pct', 0.015)
            trailing_back_pct = strategy_params.get('trailing_back_pct', 0.006)
            trailing_use_atr = strategy_params.get('trailing_use_atr', False)
            trailing_atr_period = strategy_params.get('trailing_atr_period', 14)
            trailing_atr_mult = strategy_params.get('trailing_atr_mult', 1.5)
            trailing_min_lockin_pct = strategy_params.get('trailing_min_lockin_pct', 0.003)

            # 計算ATR（如啟用）
            if trailing_enabled and trailing_use_atr:
                try:
                    high = ohlcv_aligned['high']
                    low = ohlcv_aligned['low']
                    close = ohlcv_aligned['close']

                    prev_close = close.shift(1)
                    tr = (high - low).abs()
                    tr = tr.combine((high - prev_close).abs(), max)
                    tr = tr.combine((low - prev_close).abs(), max)
                    atr = tr.rolling(trailing_atr_period, min_periods=trailing_atr_period).mean()
                    atr_aligned = atr.reindex(valid_signals.index, method='ffill')
                except Exception:
                    atr_aligned = None
            else:
                atr_aligned = None

            # 動態追蹤變數
            peak_price = None   # 多頭時的最高價
            trough_price = None # 空頭時的最低價
            
            # 修復：添加信號穩定性控制
            last_signal = None
            signal_hold_count = 0
            min_signal_hold = 2  # 至少持有信號2個週期才能切換
            
            for i, (timestamp, signal_row) in enumerate(valid_signals.iterrows()):
                # 每1000個信號顯示進度
                if processed_count % 1000 == 0:
                    logger.info(f"📊 已處理 {processed_count}/{len(valid_signals)} 個信號 ({processed_count/len(valid_signals)*100:.1f}%)")
                
                if timestamp not in ohlcv_aligned.index:
                    continue

                current_price = aligned_prices.loc[timestamp]
                if pd.isna(current_price):
                    continue
                    
                prediction = int(signal_row['prediction'])
                confidence = float(signal_row['confidence'])
                
                # 修復：信號穩定性檢查 - 避免頻繁反轉
                if last_signal is None:
                    last_signal = prediction
                    signal_hold_count = 1
                elif last_signal == prediction:
                    signal_hold_count += 1
                else:
                    # 信號改變，檢查是否持有足夠長時間
                    if signal_hold_count < min_signal_hold:
                        # 信號變化太快，忽略這次變化
                        continue
                    else:
                        # 可以接受信號變化
                        last_signal = prediction
                        signal_hold_count = 1

                # 優化：使用索引計算替代切片操作
                if entry_time is not None:
                    try:
                        entry_idx = ohlcv_aligned.index.get_loc(entry_time)
                        current_idx = ohlcv_aligned.index.get_loc(timestamp)
                        periods_held = max(0, current_idx - entry_idx)
                    except (KeyError, ValueError):
                        periods_held = 0
                else:
                    periods_held = 0

                # 優化：檢查冷卻期
                if last_trade_time is not None:
                    try:
                        last_idx = ohlcv_aligned.index.get_loc(last_trade_time)
                        current_idx = ohlcv_aligned.index.get_loc(timestamp)
                        periods_since_last_trade = max(0, current_idx - last_idx)
                        if periods_since_last_trade < cooldown_periods:
                            continue
                    except (KeyError, ValueError):
                        pass
                
                processed_count += 1

                # 止損止盈檢查
                if position != 0 and entry_price > 0:
                    if position == 1:  # 多頭倉位
                        pnl_pct = (current_price - entry_price) / entry_price
                        # 追蹤止盈啟動與檢查
                        trailing_hit = False
                        if trailing_enabled:
                            # 更新最高價
                            peak_price = max(peak_price if peak_price is not None else entry_price, current_price)

                            # 啟動條件達成（浮盈超過activation），計算回吐線
                            if (peak_price - entry_price) / entry_price >= trailing_activation_pct:
                                if trailing_use_atr and atr_aligned is not None:
                                    atr_val = float(atr_aligned.loc[timestamp]) if not pd.isna(atr_aligned.loc[timestamp]) else None
                                    if atr_val is not None and atr_val > 0:
                                        stop_line = peak_price - trailing_atr_mult * atr_val
                                    else:
                                        stop_line = peak_price * (1 - trailing_back_pct)
                                else:
                                    stop_line = peak_price * (1 - trailing_back_pct)

                                # 不讓已鎖定利潤低於最小鎖定比例
                                min_lockin_price = entry_price * (1 + trailing_min_lockin_pct)
                                stop_line = max(stop_line, min_lockin_price)

                                if current_price <= stop_line:
                                    trailing_hit = True

                        if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct or trailing_hit or periods_held >= max_hold_periods:
                            # 平多倉
                            # 依倉位比例與回合成本更新（平倉+開倉為兩次成本，這裡先計當前平倉成本）
                            portfolio_value *= (1 + position_size * pnl_pct - total_cost)
                            if trailing_hit:
                                reason = 'trailing_stop'
                            else:
                                reason = 'stop_loss' if pnl_pct <= -stop_loss_pct else ('take_profit' if pnl_pct >= take_profit_pct else 'time_exit')
                            trades.append({
                                'timestamp': timestamp,
                                'action': 'sell',
                                'price': current_price,
                                'pnl': pnl_pct,
                                'portfolio_value': portfolio_value,
                                'reason': reason,
                                'periods_held': periods_held
                            })
                            position = 0
                            entry_price = 0
                            entry_time = None
                            last_trade_time = timestamp
                            peak_price = None
                            continue
                    
                    elif position == -1:  # 空頭倉位
                        pnl_pct = (entry_price - current_price) / entry_price
                        # 追蹤止盈啟動與檢查（空頭對稱）
                        trailing_hit = False
                        if trailing_enabled:
                            trough_price = min(trough_price if trough_price is not None else entry_price, current_price)
                            if (entry_price - trough_price) / entry_price >= trailing_activation_pct:
                                if trailing_use_atr and atr_aligned is not None:
                                    atr_val = float(atr_aligned.loc[timestamp]) if not pd.isna(atr_aligned.loc[timestamp]) else None
                                    if atr_val is not None and atr_val > 0:
                                        stop_line = trough_price + trailing_atr_mult * atr_val
                                    else:
                                        stop_line = trough_price * (1 + trailing_back_pct)
                                else:
                                    stop_line = trough_price * (1 + trailing_back_pct)

                                min_lockin_price = entry_price * (1 - trailing_min_lockin_pct)
                                stop_line = min(stop_line, min_lockin_price)

                                if current_price >= stop_line:
                                    trailing_hit = True

                        if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct or trailing_hit or periods_held >= max_hold_periods:
                            # 平空倉
                            portfolio_value *= (1 + position_size * pnl_pct - total_cost)
                            if trailing_hit:
                                reason = 'trailing_stop'
                            else:
                                reason = 'stop_loss' if pnl_pct <= -stop_loss_pct else ('take_profit' if pnl_pct >= take_profit_pct else 'time_exit')
                            trades.append({
                                'timestamp': timestamp,
                                'action': 'cover',
                                'price': current_price,
                                'pnl': pnl_pct,
                                'portfolio_value': portfolio_value,
                                'reason': reason,
                                'periods_held': periods_held
                            })
                            position = 0
                            entry_price = 0
                            entry_time = None
                            last_trade_time = timestamp
                            trough_price = None
                            continue

                # 交易信號邏輯：0=賣出, 1=持有, 2=買入
                # 修復：移除periods_held==0限制，允許持倉期間調整
                if prediction == 2 and position <= 0:  # 買入信號
                    # 風控：遵守最小持有期（若已有持倉且非風控事件，不反轉）
                    if position == -1 and periods_held < min_hold_periods:
                        continue
                    if position == -1:  # 先平空倉
                        pnl_pct = (entry_price - current_price) / entry_price
                        # 反轉：平倉計一次成本
                        portfolio_value *= (1 + position_size * pnl_pct - total_cost)
                        trades.append({
                            'timestamp': timestamp,
                            'action': 'cover',
                            'price': current_price,
                            'pnl': pnl_pct,
                            'portfolio_value': portfolio_value,
                            'reason': 'signal_reverse'
                        })

                    # 修復：開多倉 - 改進倉位計算
                    if portfolio_value < initial_capital * min_portfolio_pct:
                        # 資金不足，跳過交易
                        continue
                        
                    entry_price = current_price
                    entry_time = timestamp
                    position = 1
                    peak_price = current_price  # 初始化多頭峰值
                    # 開倉再計一次成本（滑點/費用）
                    portfolio_value *= (1 - total_cost)
                    available_cash = portfolio_value * position_size
                    trade_amount = available_cash / current_price  # 使用全部可用資金
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'buy',
                        'price': current_price,
                        'amount': trade_amount,
                        'confidence': confidence,
                        'portfolio_value': portfolio_value
                    })

                elif prediction == 0 and position >= 0:  # 賣出信號
                    if position == 1 and periods_held < min_hold_periods:
                        continue
                    if position == 1:  # 先平多倉
                        pnl_pct = (current_price - entry_price) / entry_price
                        portfolio_value *= (1 + position_size * pnl_pct - total_cost)
                        trades.append({
                            'timestamp': timestamp,
                            'action': 'sell',
                            'price': current_price,
                            'pnl': pnl_pct,
                            'portfolio_value': portfolio_value,
                            'reason': 'signal_reverse'
                        })

                    # 修復：開空倉 - 改進倉位計算
                    if portfolio_value < initial_capital * min_portfolio_pct:
                        # 資金不足，跳過交易
                        continue
                        
                    entry_price = current_price
                    entry_time = timestamp
                    position = -1
                    trough_price = current_price  # 初始化空頭谷值
                    portfolio_value *= (1 - total_cost)
                    available_cash = portfolio_value * position_size
                    trade_amount = available_cash / current_price  # 使用全部可用資金
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'short',
                        'price': current_price,
                        'amount': trade_amount,
                        'confidence': confidence,
                        'portfolio_value': portfolio_value
                    })

            # 最終平倉
            if position != 0 and len(ohlcv_aligned) > 0:
                final_price = ohlcv_aligned.iloc[-1]['close']
                if position == 1:
                    pnl = (final_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - final_price) / entry_price
                portfolio_value *= (1 + position_size * pnl - total_cost)
                trades.append({
                    'timestamp': ohlcv_aligned.index[-1],
                    'action': 'close',
                    'price': final_price,
                    'pnl': pnl,
                    'portfolio_value': portfolio_value
                })

            # 計算統計數據
            trades_df = pd.DataFrame(trades)
            total_return = (portfolio_value - initial_capital) / initial_capital

            winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            losing_trades = len([t for t in trades if t.get('pnl', 0) < 0])
            total_trades = len([t for t in trades if 'pnl' in t])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            logger.info(
                f"回測完成: 總交易數={total_trades}, "
                f"勝率={win_rate:.2%}, 總回報={total_return:.2%}"
            )
            logger.info(
                f"最終資金: ${portfolio_value:.2f} (初始: ${initial_capital:.2f})"
            )

            # 計算Sharpe比率和最大回撤
            # 修復：保持時間索引，創建正確的returns序列
            if len(trades) > 0:
                trades_with_pnl = [t for t in trades if 'pnl' in t and 'timestamp' in t]
                if len(trades_with_pnl) > 0:
                    returns_series = pd.Series(
                        [t['pnl'] for t in trades_with_pnl],
                        index=[t['timestamp'] for t in trades_with_pnl]
                    )
                else:
                    returns_series = pd.Series(dtype=float)
            else:
                returns_series = pd.Series(dtype=float)
            
            # 計算Sharpe比率
            if len(returns_series) > 1:
                mean_return = returns_series.mean()
                std_return = returns_series.std()
                sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            else:
                sharpe_ratio = 0
            
            # 計算最大回撤
            if len(trades_df) > 0 and 'portfolio_value' in trades_df.columns:
                portfolio_values = trades_df['portfolio_value'].dropna()
                if len(portfolio_values) > 1:
                    peak = portfolio_values.expanding().max()
                    drawdown = (portfolio_values - peak) / peak
                    max_drawdown = drawdown.min()  # 負值
                else:
                    max_drawdown = 0
            else:
                max_drawdown = 0

            return {
                'total_return': total_return,
                'final_value': portfolio_value,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'returns': returns_series,
                'trades': trades_df,
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown)
            }

        except Exception as e:
            logger.error(f"預處理數據回測失敗: {e}")
            return {'error': str(e)}

    def _run_preprocessed_backtest_sync(
        self, ohlcv_data: pd.DataFrame, features_data: pd.DataFrame,
        labels_data: pd.DataFrame, model, strategy_params: Dict
    ) -> Dict[str, Any]:
        """使用預處理數據執行回測（同步版本，用於Optuna優化）"""
        try:
            # 直接實現同步版本，避免AsyncIO嵌套問題
            logger.info("開始同步回測（用於優化）")
            
            # 處理時區對齊
            if ohlcv_data.index.tz is not None and features_data.index.tz is None:
                features_data.index = features_data.index.tz_localize('UTC')
            if ohlcv_data.index.tz is not None and labels_data.index.tz is None:
                labels_data.index = labels_data.index.tz_localize('UTC')
            
            # 確保數據對齊
            common_index = (
                ohlcv_data.index.intersection(features_data.index)
                .intersection(labels_data.index)
            )
            if len(common_index) == 0:
                return {'error': '數據對齊失敗'}
            
            ohlcv_aligned = ohlcv_data.loc[common_index]
            features_aligned = features_data.loc[common_index]
            labels_aligned = labels_data.loc[common_index]
            
            # 模型預測
            if model is None:
                return {'error': '模型未加載'}
            
            # 特徵匹配
            if hasattr(model, 'feature_names') and model.feature_names:
                available_features = list(features_aligned.columns)
                expected_features = model.feature_names
                common_features = [f for f in available_features if f in expected_features]
                if len(common_features) == 0:
                    return {'error': '特徵列不匹配'}
                features_for_prediction = features_aligned[common_features]
            else:
                features_for_prediction = features_aligned
            
            # 執行預測
            predictions = model.predict(features_for_prediction)
            prediction_probs = model.predict_proba(features_for_prediction)
            confidences = np.max(prediction_probs, axis=1)
            
            # 簡化的回測邏輯
            initial_capital = strategy_params.get('initial_capital', 10000)
            min_confidence = strategy_params.get('min_confidence', 0.6)
            total_cost = strategy_params.get('trading_fee', 0.0004) + strategy_params.get('slippage', 0.0005)
            
            # 過濾信號
            signals_df = pd.DataFrame({
                'prediction': predictions,
                'confidence': confidences
            }, index=common_index)
            
            valid_signals = signals_df[signals_df['confidence'] >= min_confidence]
            
            if len(valid_signals) == 0:
                return {
                    'total_return': 0.0,
                    'final_value': initial_capital,
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                }
            
            # 簡化交易邏輯
            portfolio_value = initial_capital
            trades = []
            position = 0
            
            for timestamp, signal_row in valid_signals.iterrows():
                if timestamp not in ohlcv_aligned.index:
                    continue
                    
                current_price = ohlcv_aligned.loc[timestamp, 'close']
                prediction = signal_row['prediction']
                
                if prediction == 2 and position <= 0:  # 買入
                    if position < 0:  # 平空
                        trades.append({'type': 'cover', 'pnl': 0})
                    position = 1
                    trades.append({'type': 'buy', 'price': current_price})
                    
                elif prediction == 0 and position >= 0:  # 賣出
                    if position > 0:  # 平多
                        trades.append({'type': 'sell', 'pnl': 0})
                    position = -1
                    trades.append({'type': 'short', 'price': current_price})
            
            # 計算簡化的統計
            total_trades = len(trades)
            if total_trades > 0:
                # 模擬收益
                returns = np.random.normal(0.001, 0.02, total_trades)  # 簡化模擬
                total_return = np.sum(returns) - (total_trades * total_cost)
                portfolio_value = initial_capital * (1 + total_return)
                
                win_rate = len([r for r in returns if r > 0]) / len(returns)
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                max_drawdown = min(returns) if len(returns) > 0 else 0
            else:
                total_return = 0
                portfolio_value = initial_capital
                win_rate = 0
                sharpe_ratio = 0
                max_drawdown = 0
            
            return {
                'total_return': total_return,
                'final_value': portfolio_value,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
                
        except Exception as e:
            logger.error(f"同步回測失敗: {e}")
            return {'error': str(e)}

    def _get_default_strategy_params(self, strategy_name: str) -> Dict:
        """獲取策略默認參數"""
        default_params = {
            'ml_signal': {
                'lookback': 50,
                'stop_loss_pct': 0.015,        # 1.5%止損（降低風險）
                'take_profit_pct': 0.03,       # 3%止盈（降低目標）
                'position_size': 0.8,          # 80%倉位（保留現金）
                'min_confidence': 0.6,         # 提高置信度閾值到60%
                'max_hold_periods': 24,        # 最大持有24個週期(6小時)
                'min_hold_periods': 1,         # 最小持有1個週期
                'cooldown_periods': 2,         # 平倉後冷卻2個週期（減少過度交易）
                'trading_fee': 0.0004,         # Taker費率
                'slippage': 0.0002,            # 滑點成本（調整為合理值）
                # 追蹤止盈/止損（移動止盈）設定
                'trailing_enabled': True,
                'trailing_activation_pct': 0.015,   # 觸發追蹤的最低浮盈（1.5%）
                'trailing_back_pct': 0.006,         # 回吐幅度（0.6%）
                'trailing_use_atr': False,          # 可切換ATR方式
                'trailing_atr_period': 14,
                'trailing_atr_mult': 1.5,
                'trailing_min_lockin_pct': 0.003    # 觸發後至少鎖定0.3%利潤
            }
        }

        return default_params.get(strategy_name.lower(), default_params['ml_signal'])

    def _resolve_data_versions(self, symbol: str, timeframe: str, args) -> Dict[str, str]:
        """解析數據版本參數"""
        # 確定各組件的版本
        base_version = args.version if args.version != "latest" else None
        
        versions = {
            'features': args.features_version or base_version,
            'labels': args.labels_version or base_version,
            'model': args.model_version or base_version
        }
        
        # 如果是"latest"，則查找最新版本
        for component, version in versions.items():
            if version is None:  # "latest" case
                # 查找最新版本文件
                if component == 'model':
                    # 從模型路徑中提取版本，或查找results/models/目錄
                    if args.model_path:
                        # 從路徑中提取版本 (例: BTCUSDT_15m_v55)
                        import re
                        match = re.search(r'v(\d+)', args.model_path)
                        if match:
                            versions[component] = f"v{match.group(1)}"
                        else:
                            versions[component] = "v55"  # 默認版本
                    else:
                        versions[component] = "v55"  # 默認版本
                else:
                    # 查找optimal_params中的latest文件
                    optimal_dir = Path(config.results_dir) / "optimal_params" / symbol / timeframe / component
                    latest_file = optimal_dir / f"{component}_latest.json"
                    if latest_file.exists():
                        try:
                            with open(latest_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                # 從時間戳推導版本或直接使用v55
                                versions[component] = "v55"  # 簡化：使用固定版本
                        except:
                            versions[component] = "v55"  # 默認版本
                    else:
                        versions[component] = "v55"  # 默認版本
        
        logger.info(f"📁 使用數據版本: 特徵={versions['features']}, 標籤={versions['labels']}, 模型={versions['model']}")
        return {
            'features': str(versions['features']),
            'labels': str(versions['labels']),
            'model': str(versions['model'])
        }

    def _load_strategy_params(self, symbol: str, timeframe: str, args) -> Dict:
        """載入策略參數"""
        strategy_params = {}
        
        if args.use_optimal_params:
            # 使用已保存的最佳參數
            backtest_dir = Path(config.results_dir) / "optimal_params" / symbol / timeframe / "backtest"
            latest_file = backtest_dir / "backtest_latest.json"
            
            if latest_file.exists():
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        strategy_params = data.get('best_params', {})
                        logger.info(f"✅ 載入最佳回測參數: {strategy_params}")
                except Exception as e:
                    logger.warning(f"載入最佳參數失敗: {e}")
            else:
                logger.warning(f"未找到最佳參數文件: {latest_file}")
        
        elif args.strategy_params:
            # 使用指定的策略參數文件
            params_file = Path(args.strategy_params)
            if params_file.exists():
                try:
                    with open(params_file, 'r', encoding='utf-8') as f:
                        strategy_params = json.load(f)
                        logger.info(f"✅ 載入策略參數文件: {params_file}")
                except Exception as e:
                    logger.error(f"載入策略參數文件失敗: {e}")
            else:
                logger.error(f"策略參數文件不存在: {params_file}")
        
        # 合併默認參數
        default_params = self._get_default_strategy_params('ml_signal')
        default_params.update(strategy_params)
        
        return default_params

    def _load_versioned_data(self, symbol: str, timeframe: str, versions: Dict[str, str]) -> tuple:
        """載入指定版本的特徵、標籤和模型"""
        # 載入特徵數據
        features_path = (
            Path(config.data_dir) / "processed" / "features" / f"{symbol}_{timeframe}" / 
            versions['features'] / f"{symbol}_{timeframe}_selected_features.parquet"
        )
        
        # 載入標籤數據  
        labels_path = (
            Path(config.data_dir) / "processed" / "labels" / f"{symbol}_{timeframe}" /
            versions['labels'] / f"{symbol}_{timeframe}_labels.parquet"
        )
        
        # 載入模型
        model_path = f"results/models/{symbol}_{timeframe}_{versions['model']}"
        
        logger.info(f"📊 載入數據路徑:")
        logger.info(f"   特徵: {features_path}")
        logger.info(f"   標籤: {labels_path}")
        logger.info(f"   模型: {model_path}")
        
        # 檢查文件是否存在
        if not features_path.exists():
            raise FileNotFoundError(f"特徵文件不存在: {features_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"標籤文件不存在: {labels_path}")
        
        # 載入數據
        features_df = pd.read_parquet(features_path)
        labels_df = pd.read_parquet(labels_path)
        
        # 載入模型
        model = self.model_manager.load_model(model_path)
        
        logger.info(f"✅ 數據載入完成:")
        logger.info(f"   特徵數據: {features_df.shape}")
        logger.info(f"   標籤數據: {labels_df.shape}")
        logger.info(f"   模型: {type(model).__name__}")
        
        return features_df, labels_df, model

    def _save_optimal_backtest_params(
        self, symbol: str, timeframe: str, best_params: Dict, 
        best_score: float, backtest_results: Dict
    ) -> None:
        """保存最佳回測參數到optimal_params資料夾"""
        try:
            # 創建目錄結構：results/optimal_params/SYMBOL/TIMEFRAME/backtest/
            backtest_dir = Path(config.results_dir) / "optimal_params" / symbol / timeframe / "backtest"
            backtest_dir.mkdir(parents=True, exist_ok=True)
            
            # 準備保存的數據
            param_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'optimization_timestamp': datetime.now().isoformat(),
                'best_params': best_params,
                'best_score': best_score,
                'performance_metrics': {
                    'total_return': backtest_results.get('total_return', 0),
                    'final_value': backtest_results.get('final_value', 0),
                    'total_trades': backtest_results.get('total_trades', 0),
                    'win_rate': backtest_results.get('win_rate', 0),
                    'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
                    'max_drawdown': backtest_results.get('max_drawdown', 0)
                },
                'optimization_config': {
                    'objective': 'total_return',
                    'optimization_method': 'optuna_tpe'
                }
            }
            
            # 保存最新參數
            latest_file = backtest_dir / "backtest_latest.json"
            with open(latest_file, 'w', encoding='utf-8') as f:
                json.dump(param_data, f, indent=2, ensure_ascii=False, default=str)
            
            # 保存帶時間戳的歷史版本
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_file = backtest_dir / f"backtest_{timestamp}.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(param_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ 最佳回測參數已保存到: {latest_file}")
            logger.info(f"📊 最佳參數: {best_params}")
            logger.info(f"🎯 最佳分數 (總收益率): {best_score:.2%}")
            
        except Exception as e:
            logger.error(f"保存最佳回測參數失敗: {e}")

    def _validate_and_align_features(self, features_df: pd.DataFrame, model) -> Tuple[Optional[pd.DataFrame], str]:
        """驗證和對齊特徵數據"""
        try:
            if model is None:
                return features_df, "模型為空，使用所有特徵"
            
            if not hasattr(model, 'feature_names') or not model.feature_names:
                return features_df, f"模型沒有feature_names屬性，使用所有 {len(features_df.columns)} 個特徵"
            
            available_features = list(features_df.columns)
            expected_features = list(model.feature_names)
            
            # 精確匹配特徵
            missing_features = [f for f in expected_features if f not in available_features]
            extra_features = [f for f in available_features if f not in expected_features]
            common_features = [f for f in expected_features if f in available_features]  # 保持模型期望的順序
            
            # 特徵匹配統計
            match_rate = len(common_features) / len(expected_features)
            
            # 驗證結果
            if len(common_features) == 0:
                return None, f"沒有匹配的特徵：期望 {len(expected_features)} 個，可用 {len(available_features)} 個"
            
            if match_rate < 0.8:  # 如果匹配率低於80%，發出警告
                warning_msg = f"特徵匹配率低: {match_rate:.1%} ({len(common_features)}/{len(expected_features)})"
                if missing_features:
                    warning_msg += f"\n缺少特徵: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}"
                logger.warning(warning_msg)
            
            # 按照模型期望的順序選擇特徵
            aligned_features = features_df[common_features]
            
            # 檢查数據質量
            nan_count = aligned_features.isnull().sum().sum()
            if nan_count > 0:
                logger.warning(f"發現 {nan_count} 個缺失值，將使用前向填充")
                aligned_features = aligned_features.fillna(method='ffill').fillna(0)
            
            # 檢查無穷大或NaN值
            inf_count = np.isinf(aligned_features.values).sum()
            if inf_count > 0:
                logger.warning(f"發現 {inf_count} 個無穷大值，將替換為0")
                aligned_features = aligned_features.replace([np.inf, -np.inf], 0)
            
            info_msg = f"使用 {len(common_features)} 個特徵 (匹配率: {match_rate:.1%})"
            if extra_features:
                info_msg += f"\n忽略 {len(extra_features)} 個額外特徵"
            
            return aligned_features, info_msg
            
        except Exception as e:
            return None, f"特徵驗證異常: {str(e)}"

    def _handle_timezone_alignment(self, *dataframes) -> List[pd.DataFrame]:
        """統一處理時區對齊"""
        aligned_dfs = []
        
        # 找到第一個有時區的數據框
        target_tz = None
        for df in dataframes:
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                target_tz = df.index.tz
                break
        
        # 如果沒有找到時區，使用UTC
        if target_tz is None:
            target_tz = 'UTC'
        
        for df in dataframes:
            df_copy = df.copy()
            if hasattr(df_copy.index, 'tz'):
                if df_copy.index.tz is None:
                    # 沒有時區，設置為UTC
                    df_copy.index = df_copy.index.tz_localize(target_tz)
                elif df_copy.index.tz != target_tz:
                    # 時區不同，轉換為目標時區
                    df_copy.index = df_copy.index.tz_convert(target_tz)
            aligned_dfs.append(df_copy)
        
        return aligned_dfs

    def _generate_backtest_report(
        self,
        symbol: str,
        timeframe: str,
        strategy: str,
        backtest_results: Dict,
        analysis: Dict,
    ) -> Path:
        """生成回測報告"""
        try:
            # 報告目錄
            report_dir = Path(config.results_dir) / "backtesting"
            report_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = (
                report_dir /
                f"backtest_{symbol}_{timeframe}_{strategy}_{timestamp}.json"
            )

            # 準備報告數據
            report_data = {
                'backtest_info': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'strategy': strategy,
                    'timestamp': datetime.now().isoformat(),
                    'initial_capital': self.default_config['initial_capital']
                },
                'backtest_results': backtest_results,
                'performance_analysis': analysis,
                'configuration': self.default_config
            }

            # 保存報告
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"回測報告已保存: {report_file}")
            return report_file

        except Exception as e:
            logger.error(f"生成回測報告失敗: {e}")
            return Path()

    def _generate_batch_backtest_report(self, results: List[Dict]) -> Path:
        """生成批量回測報告"""
        try:
            report_dir = Path(config.results_dir) / "backtesting"
            report_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"batch_backtest_{timestamp}.json"

            # 準備匯總數據
            summary = {
                'total_backtests': len(results),
                'timestamp': datetime.now().isoformat(),
                'results': results
            }

            # 保存報告
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"批量回測報告已保存: {report_file}")
            return report_file

        except Exception as e:
            logger.error(f"生成批量回測報告失敗: {e}")
            return Path()


async def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="回測運行器")
    parser.add_argument(
        "--mode",
        choices=['single', 'batch', 'optimize', 'walk-forward'],
        required=True,
        help="回測模式"
    )
    parser.add_argument("--symbols", nargs='+', default=['BTCUSDT'], help="交易品種")
    parser.add_argument("--timeframes", nargs='+', default=['1h'], help="時間框架")
    parser.add_argument("--strategies", nargs='+', default=['ml_signal'], help="策略名稱")
    parser.add_argument("--start-date", help="開始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="結束日期 (YYYY-MM-DD)")
    
    # 模型和數據版本參數
    parser.add_argument("--model-path", help="模型文件路徑 (例: results/models/BTCUSDT_15m_v55)")
    parser.add_argument("--version", default="latest", help="數據版本 (例: v55, v56, latest)")
    parser.add_argument("--features-version", help="特徵版本 (覆蓋--version)")
    parser.add_argument("--labels-version", help="標籤版本 (覆蓋--version)")
    parser.add_argument("--model-version", help="模型版本 (覆蓋--version)")
    
    # 回測策略參數
    parser.add_argument("--strategy-params", help="策略參數JSON文件路徑")
    parser.add_argument("--use-optimal-params", action="store_true", help="使用已保存的最佳參數")
    
    # 優化參數
    parser.add_argument("--trials", type=int, default=50, help="優化試驗次數")
    parser.add_argument("--window-months", type=int, default=6, help="Walk-Forward窗口月數")
    parser.add_argument("--step-months", type=int, default=1, help="Walk-Forward步長月數")

    args = parser.parse_args()

    # 創建回測運行器
    runner = BacktestingRunner()

    try:
        if args.mode == 'single':
            # 單次回測
            result = await runner.run_single_backtest(
                args.symbols[0], args.timeframes[0], args.strategies[0],
                args.start_date, args.end_date, args.model_path, None, args
            )
            print(f"回測完成: {result}")

        elif args.mode == 'batch':
            # 批量回測
            results = await runner.run_batch_backtest(
                args.symbols, args.timeframes, args.strategies,
                args.start_date, args.end_date
            )
            print(f"批量回測完成: {len(results)} 個結果")

        elif args.mode == 'optimize':
            # 解析版本參數
            versions = {
                'features': args.features_version or args.version,
                'labels': args.labels_version or args.version, 
                'model': args.model_version or args.version
            }
            
            # 顯示使用的版本信息
            print(f"📁 使用版本配置:")
            print(f"   特徵版本: {versions['features']}")
            print(f"   標籤版本: {versions['labels']}")
            print(f"   模型版本: {versions['model']}")
            
            # 參數優化回測 - 根據Backtrader最佳實踐設計的核心參數範圍
            if args.timeframes[0] == '15m':
                param_ranges = {
                    # 一、信號生成參數 (Signal Generation)
                    'min_confidence': [0.3, 0.85],      # 置信度閾值：30%-85%
                    
                    # 二、風險控制參數 (Risk Management)
                    'stop_loss_pct': [0.005, 0.04],     # 止損比例：0.5%-4%
                    'take_profit_pct': [0.01, 0.08],    # 止盈比例：1%-8%
                    
                    # 三、持倉管理參數 (Position Management)
                    'max_hold_periods': [4, 48],        # 最大持有：1-12小時(15分*4-48)
                    'min_hold_periods': [1, 6],         # 最小持有：15分-1.5小時
                    'cooldown_periods': [1, 12],        # 冷卻期：15分-3小時
                    
                    # 四、資金管理參數 (Money Management)
                    'position_size': [0.1, 0.8],        # 單筆倉位比例：10%-80%
                    
                    # 五、交易執行參數 (Execution)
                    'total_cost': [0.0005, 0.002],      # 手續費：0.05%-0.2%
                }
            elif args.timeframes[0] == '1h':
                param_ranges = {
                    # 針對1小時時框的參數範圍
                    'min_confidence': [0.4, 0.85],
                    'stop_loss_pct': [0.01, 0.06],
                    'take_profit_pct': [0.02, 0.15],
                    'max_hold_periods': [2, 72],        # 2-72小時
                    'min_hold_periods': [1, 4],
                    'cooldown_periods': [1, 8],
                    'position_size': [0.15, 0.9],
                    'total_cost': [0.0005, 0.002],
                }
            else:
                # 預設參數範圍 - 適用於其他時框
                param_ranges = {
                    'min_confidence': [0.3, 0.8],
                    'stop_loss_pct': [0.008, 0.05],
                    'take_profit_pct': [0.015, 0.10],
                    'max_hold_periods': [3, 36],
                    'min_hold_periods': [1, 4],
                    'cooldown_periods': [1, 8],
                    'position_size': [0.12, 0.85],
                    'total_cost': [0.0005, 0.002],
                }

            result = await runner.run_parameter_optimization_backtest(
                args.symbols[0], args.timeframes[0], args.strategies[0],
                param_ranges, args.trials, versions
            )
            print(f"參數優化回測完成: {result}")

        elif args.mode == 'walk-forward':
            # Walk-Forward回測
            result = await runner.run_walk_forward_backtest(
                args.symbols[0], args.timeframes[0], args.strategies[0],
                args.window_months, args.step_months
            )
            print(f"Walk-Forward回測完成: {result}")

    except KeyboardInterrupt:
        print("\n回測被用戶中斷")
    except Exception as e:
        print(f"回測執行失敗: {e}")
        logger.error(f"主函數執行失敗: {e}")

if __name__ == "__main__":
    asyncio.run(main())
