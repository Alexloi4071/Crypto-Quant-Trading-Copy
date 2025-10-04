#!/usr/bin/env python3
"""
Model Training Script
Automated machine learning model training pipeline
Supports batch training, hyperparameter optimization, and model versioning
"""
__file__ = r'D:\crypto-quant-trading-copy\scripts\model_trainer.py'

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import config
from src.utils.logger import setup_logger
from src.data.collector import BinanceDataCollector
from src.data.storage import DataStorage
# DEPRECATED: 舊特徵/標籤管線，改走新B方案的版本化數據
from src.optimization.main_optimizer import ModularOptunaOptimizer  # 新B方案入口（如需觸發優化）
from src.models.lgb_model import ModelManager
from src.backtesting.performance_analyzer import PerformanceAnalyzer

logger = setup_logger(__name__)


class TrainingPipeline:
    """Automated model training pipeline"""


    def __init__(self):
        self.data_collector = None
        self.storage = None
        self.feature_engine = None
        self.feature_selector = None
        self.label_generator = None
        self.model_manager = None
        self.performance_analyzer = None

        # Training statistics
        self.stats: Dict[str, Any] = {
            'models_trained': 0,
            'successful_trainings': 0,
            'failed_trainings': 0,
            'best_models': {},
            'training_times': [],
            'start_time': None,
            'end_time': None
        }

        logger.info("Training pipeline initialized")


    async def initialize(self) -> bool:
        """Initialize all pipeline components"""
        try:
            logger.info("Initializing training pipeline components")

            # Initialize components (no API needed for optimization)
            self.data_collector = None  # Skip API component
            self.storage = None  # Use direct parquet access
            self.feature_engine = None
            self.feature_selector = None
            self.label_generator = None
            self.model_manager = ModelManager()
            self.performance_analyzer = PerformanceAnalyzer()

            # No initialization needed for parquet file access
            storage_success = True  # Direct file access
            data_success = True    # Direct file access

            if not data_success or not storage_success:
                logger.error("Failed to initialize core components")
                return False

            logger.info("Training pipeline initialization completed")
            return True

        except Exception as e:
            logger.error(f"Training pipeline initialization failed: {e}")
            return False


    async def train_models_batch(self, symbols: List[str],
                               timeframes: List[str],
                               version: Optional[str] = None,
                               model_types: Optional[List[str]] = None,
                               optimize_hyperparameters: bool = True,
                               feature_selection: bool = True,
                               test_size: float = 0.2,
                               validation_size: float = 0.2) -> Dict[str, Any]:
        """Train models for multiple symbols in batch"""
        try:
            logger.info(f"Starting batch training for {len(symbols)} symbols and {len(timeframes)} timeframes")

            self.stats['start_time'] = datetime.now()

            if version is None:
                version = datetime.now().strftime('%Y%m%d_%H%M%S')

            if model_types is None:
                model_types = ['lightgbm', 'xgboost', 'random_forest']

            results: Dict[str, Any] = {
                'success': True,
                'version': version,
                'trained_models': {},
                'failed_trainings': [],
                'best_models': {},
                'training_summary': {}
            }

            # Train models for each symbol/timeframe combination
            for symbol in symbols:
                symbol_results = {}

                for timeframe in timeframes:
                    combination_key = f"{symbol}_{timeframe}"
                    logger.info(f"Training models for {combination_key}")

                    try:
                        # Train models for this combination
                        training_result = await self._train_symbol_timeframe(
                            symbol=symbol,
                            timeframe=timeframe,
                            version=version,
                            model_types=model_types,
                            optimize_hyperparameters=optimize_hyperparameters,
                            feature_selection=feature_selection,
                            test_size=test_size,
                            validation_size=validation_size
                        )

                        if training_result['success']:
                            symbol_results[timeframe] = training_result
                            self.stats['successful_trainings'] += 1

                            # Track best model
                            best_model = training_result.get('best_model')
                            if best_model:
                                self.stats['best_models'][combination_key] = best_model

                            logger.info(f"Successfully trained models for {combination_key}")
                        else:
                            results['failed_trainings'].append({
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'error': training_result.get('error', 'Unknown error')
                            })
                            self.stats['failed_trainings'] += 1

                        self.stats['models_trained'] += len(model_types)

                    except Exception as e:
                        logger.error(f"Training failed for {combination_key}: {e}")
                        results['failed_trainings'].append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'error': str(e)
                        })
                        self.stats['failed_trainings'] += 1
                        continue

                if symbol_results:
                    results['trained_models'][symbol] = symbol_results

            # Generate training summary
            self.stats['end_time'] = datetime.now()
            results['training_summary'] = self._generate_training_summary()
            results['best_models'] = self._select_best_models()

            logger.info(f"Batch training completed: {self.stats['successful_trainings']} successful, {self.stats['failed_trainings']} failed")

            return results

        except Exception as e:
            logger.error(f"Batch training failed: {e}")
            return {'success': False, 'error': str(e)}


    async def _train_symbol_timeframe(self, symbol: str, timeframe: str,
                                    version: str, model_types: List[str],
                                    optimize_hyperparameters: bool,
                                    feature_selection: bool,
                                    test_size: float,
                                    validation_size: float) -> Dict[str, Any]:
        """Train models for a specific symbol/timeframe combination"""
        try:
            training_start = datetime.now()

            # Step 1: Load or generate features
            features_df = await self._get_or_generate_features(symbol, timeframe, version)

            if features_df is None or features_df.empty:
                return {'success': False, 'error': 'No features available'}

            logger.debug(f"Loaded {len(features_df)} feature records for {symbol}_{timeframe}")

            # Step 2: Generate labels
            labels_df = await self._generate_labels(features_df, symbol, timeframe, version)

            if labels_df is None or labels_df.empty:
                return {'success': False, 'error': 'No labels generated'}

            logger.debug(f"Generated {len(labels_df)} labels for {symbol}_{timeframe}")

            # Step 3: Feature selection (if enabled)
            if feature_selection:
                features_df = await self._perform_feature_selection(features_df, labels_df, symbol, timeframe)
                logger.debug(f"Selected {len(features_df.columns)} features for {symbol}_{timeframe}")

            # Step 4: Train models
            model_results = {}
            best_model = None
            best_score = -float('inf')

            for model_type in model_types:
                try:
                    logger.debug(f"Training {model_type} for {symbol}_{timeframe}")

                    # Train model
                    # Create and train model
                    # Create model instance
                    # 當 --no-optimization 時，自動使用優化參數
                    use_optimal = not optimize_hyperparameters
                    model = self.model_manager.create_model(
                        model_type=model_type,
                        symbol=symbol,
                        timeframe=timeframe,
                        version=version,
                        use_optimal_params=use_optimal
                    )
                    
                    # Prepare training data
                    X_train = features_df
                    y_train = labels_df['label'] if 'label' in labels_df.columns else labels_df.iloc[:, 0]
                    
                    # Train model
                    trained_model = self.model_manager.train_model(
                        model=model,
                        X_train=X_train,
                        y_train=y_train
                    )
                    
                    # Save trained model
                    try:
                        model_path = trained_model.save_model()
                        logger.info(f"Model saved to: {model_path}")
                    except Exception as e:
                        logger.error(f"Failed to save model: {e}")
                    
                    # Create training result
                    training_result = {
                        'success': True,
                        'model': trained_model,
                        'model_type': model_type,
                        'performance': getattr(trained_model, 'performance_metrics', {}),
                        'model_path': getattr(trained_model, 'model_path', None)
                    }

                    if training_result.get('success', False):
                        model_results[model_type] = training_result

                        # Track best model
                        model_score = training_result.get('best_score', -float('inf'))
                        if model_score > best_score:
                            best_score = model_score
                            best_model = {
                                'type': model_type,
                                'score': model_score,
                                'metrics': training_result.get('metrics', {}),
                                'model_path': training_result.get('model_path')
                            }

                        logger.debug(f"{model_type} training completed - Score: {model_score:.4f}")
                    else:
                        logger.warning(f"{model_type} training failed for {symbol}_{timeframe}")
                        model_results[model_type] = {'success': False, 'error': training_result.get('error', 'Unknown error')}

                except Exception as e:
                    logger.error(f"{model_type} training failed for {symbol}_{timeframe}: {e}")
                    model_results[model_type] = {'success': False, 'error': str(e)}

            # Calculate training time
            training_time = (datetime.now() - training_start).total_seconds()
            self.stats['training_times'].append(training_time)

            # Generate model comparison
            model_comparison = self._compare_models(model_results)

            return {
                'success': True,
                'model_results': model_results,
                'best_model': best_model,
                'model_comparison': model_comparison,
                'feature_count': len(features_df.columns),
                'training_samples': len(features_df),
                'training_time_seconds': training_time
            }

        except Exception as e:
            logger.error(f"Symbol/timeframe training failed for {symbol}_{timeframe}: {e}")
            return {'success': False, 'error': str(e)}


    async def _get_or_generate_features(self, symbol: str, timeframe: str, version: str):
        """Get existing features or generate new ones using direct parquet access"""
        try:
            import pandas as pd
            
            # Try to load existing features from parquet files
            feature_path = f"data/processed/features/{symbol}_{timeframe}/{version}/{symbol}_{timeframe}_selected_features.parquet"
            
            try:
                features_df = pd.read_parquet(feature_path)
                if not features_df.empty:
                    logger.debug(f"Loaded existing features for {symbol}_{timeframe} from {feature_path}")
                    return features_df
            except FileNotFoundError:
                logger.debug(f"No existing features found at {feature_path}")

            # If no existing features, return None (skip feature generation for now)
            logger.warning(f"No pre-generated features available for {symbol}_{timeframe} version {version}")
            logger.warning(f"Please run feature generation first for {symbol}_{timeframe}")
            return None

        except Exception as e:
            logger.error(f"Feature loading failed for {symbol}_{timeframe}: {e}")
            return None


    async def _generate_labels(self, features_df, symbol: str, timeframe: str, version: str):
        """Generate labels for the features using direct parquet access"""
        try:
            import pandas as pd
            
            # Try to load existing labels from parquet files
            label_path = f"data/processed/labels/{symbol}_{timeframe}/{version}/{symbol}_{timeframe}_labels.parquet"
            
            try:
                labels_df = pd.read_parquet(label_path)
                if not labels_df.empty:
                    logger.debug(f"Loaded existing labels for {symbol}_{timeframe} from {label_path}")
                    
                    # Log label distribution
                    if 'label' in labels_df.columns:
                        label_dist = labels_df['label'].value_counts()
                        logger.debug(f"Label distribution for {symbol}_{timeframe}: {dict(label_dist)}")
                    
                    return labels_df
            except FileNotFoundError:
                logger.debug(f"No existing labels found at {label_path}")

            # If no existing labels, return None (skip label generation for now)
            logger.warning(f"No pre-generated labels available for {symbol}_{timeframe} version {version}")
            logger.warning(f"Please run label generation first for {symbol}_{timeframe}")
            return None

        except Exception as e:
            logger.error(f"Label loading failed for {symbol}_{timeframe}: {e}")
            return None


    async def _perform_feature_selection(self, features_df, labels_df, symbol: str, timeframe: str):
        """Perform feature selection - skip since we already have selected features"""
        try:
            # Features are already selected in the parquet files, so skip additional selection
            logger.debug(f"Using pre-selected features for {symbol}_{timeframe}: {len(features_df.columns)} features")
            return features_df

        except Exception as e:
            logger.error(f"Feature selection failed for {symbol}_{timeframe}: {e}")
            return features_df  # Return original features if selection fails


    def _compare_models(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare model performance"""
        try:
            comparison: Dict[str, Any] = {
                'model_count': len(model_results),
                'successful_models': 0,
                'failed_models': 0,
                'scores': {},
                'best_model': None,
                'model_rankings': []
            }

            successful_models = []

            for model_type, result in model_results.items():
                if result.get('success', False):
                    comparison['successful_models'] += 1
                    score = result.get('best_score', 0)
                    comparison['scores'][model_type] = score
                    successful_models.append((model_type, score))
                else:
                    comparison['failed_models'] += 1
                    comparison['scores'][model_type] = None

            # Rank models by score
            successful_models.sort(key=lambda x: x[1], reverse=True)
            comparison['model_rankings'] = successful_models

            if successful_models:
                comparison['best_model'] = successful_models[0][0]

            return comparison

        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {}


    def _generate_training_summary(self) -> Dict[str, Any]:
        """Generate training summary statistics"""
        try:
            duration = None
            if self.stats['start_time'] and self.stats['end_time']:
                duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

            success_rate = 0
            total_combinations = self.stats['successful_trainings'] + self.stats['failed_trainings']
            if total_combinations > 0:
                success_rate = (self.stats['successful_trainings'] / total_combinations) * 100

            avg_training_time = 0
            if self.stats['training_times']:
                avg_training_time = sum(self.stats['training_times']) / len(self.stats['training_times'])

            return {
                'total_models_trained': self.stats['models_trained'],
                'successful_combinations': self.stats['successful_trainings'],
                'failed_combinations': self.stats['failed_trainings'],
                'success_rate_percent': round(success_rate, 1),
                'total_duration_seconds': round(duration, 1) if duration else None,
                'average_training_time_seconds': round(avg_training_time, 1),
                'best_models_count': len(self.stats['best_models'])
            }

        except Exception as e:
            logger.error(f"Training summary generation failed: {e}")
            return {}


    def _select_best_models(self) -> Dict[str, Any]:
        """Select best performing models"""
        try:
            best_models = {}

            for combination, model_info in self.stats['best_models'].items():
                best_models[combination] = {
                    'model_type': model_info['type'],
                    'score': model_info['score'],
                    'metrics': model_info.get('metrics', {}),
                    'model_path': model_info.get('model_path')
                }

            # Sort by score
            sorted_models = sorted(
                best_models.items(),
                key=lambda x: x[1]['score'],
                reverse=True
            )

            return {
                'all_best_models': dict(sorted_models),
                'top_model': dict(sorted_models[:1]) if sorted_models else {},
                'top_5_models': dict(sorted_models[:5]) if sorted_models else {}
            }

        except Exception as e:
            logger.error(f"Best model selection failed: {e}")
            return {}


    async def train_single_model(self, symbol: str, timeframe: str,
                               model_type: str = 'lightgbm',
                               version: Optional[str] = None,
                               optimize_hyperparameters: bool = True) -> Dict[str, Any]:
        """Train a single model for a specific symbol/timeframe"""
        try:
            logger.info(f"Training single {model_type} model for {symbol}_{timeframe}")

            if version is None:
                version = datetime.now().strftime('%Y%m%d_%H%M%S')

            result = await self._train_symbol_timeframe(
                symbol=symbol,
                timeframe=timeframe,
                version=version,
                model_types=[model_type],
                optimize_hyperparameters=optimize_hyperparameters,
                feature_selection=True,
                test_size=0.2,
                validation_size=0.2
            )

            return result

        except Exception as e:
            logger.error(f"Single model training failed: {e}")
            return {'success': False, 'error': str(e)}


    async def retrain_best_models(self, symbols: List[str], timeframes: List[str],
                                version: Optional[str] = None) -> Dict[str, Any]:
        """Retrain the best performing models with latest data"""
        try:
            logger.info(f"Retraining best models for {len(symbols)} symbols")

            if version is None:
                version = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            results = {
                'success': True,
                'retrained_models': {},
                'failed_retrainings': []
            }

            for symbol in symbols:
                for timeframe in timeframes:
                    try:
                        # Get best model type for this combination
                        best_model_type = await self._get_best_model_type(symbol, timeframe)

                        if not best_model_type:
                            logger.warning(f"No previous model found for {symbol}_{timeframe}")
                            continue

                        # Retrain the best model
                        retrain_result = await self.train_single_model(
                            symbol=symbol,
                            timeframe=timeframe,
                            model_type=best_model_type,
                            version=version,
                            optimize_hyperparameters=False  # Use previous best params
                        )

                        if retrain_result['success']:
                            results['retrained_models'][f"{symbol}_{timeframe}"] = retrain_result
                            logger.info(f"Successfully retrained {best_model_type} for {symbol}_{timeframe}")
                        else:
                            results['failed_retrainings'].append({
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'model_type': best_model_type,
                                'error': retrain_result.get('error', 'Unknown error')
                            })

                    except Exception as e:
                        logger.error(f"Retraining failed for {symbol}_{timeframe}: {e}")
                        results['failed_retrainings'].append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'error': str(e)
                        })
                        continue

            return results

        except Exception as e:
            logger.error(f"Best models retraining failed: {e}")
            return {'success': False, 'error': str(e)}


    async def _get_best_model_type(self, symbol: str, timeframe: str) -> Optional[str]:
        """Get the best model type for a symbol/timeframe combination"""
        try:
            # This would typically query the model manager for the best performing model
            # For now, return a default
            return 'lightgbm'

        except Exception as e:
            logger.error(f"Best model type retrieval failed: {e}")
            return None


    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training pipeline statistics"""
        return self.stats.copy()


    async def optimize_features_with_optuna(self, symbols: List[str], timeframes: List[str],
                                          version: Optional[str] = None, stages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform multi-stage feature optimization using Optuna

        Args:
            symbols: List of trading symbols
            timeframes: List of timeframes
            version: Version tag
            stages: Optimization stages ['top30', 'top15_stage1', 'top15_stage2']

        Returns:
            Optimization results
        """
        try:
            logger.info(f"Starting feature optimization for {len(symbols)} symbols, {len(timeframes)} timeframes")
            self.stats['start_time'] = datetime.now()

            # Import optimization components from new modular system
            from src.optimization.feature_selector import FeatureSelector
            from src.optimization.feature_optimizer import FeatureOptimizer

            results = {
                'success': True,
                'optimization_results': {},
                'stages_completed': [],
                'failed_optimizations': [],
                'summary': {}
            }

            stages = stages or ['top30', 'top15_stage1', 'top15_stage2']
            total_operations = len(symbols) * len(timeframes) * len(stages)
            current_operation = 0
            successful_operations = 0

            print(f"🔬 開始多階段特徵優化")
            print(f"   💰 交易對: {', '.join(symbols)}")
            print(f"   ⏰ 時框: {', '.join(timeframes)}")
            print(f"   🎯 優化階段: {', '.join(stages)}")
            print(f"   📊 總操作數: {total_operations}")
            print("-" * 60)

            for symbol in symbols:
                for timeframe in timeframes:
                    symbol_tf_key = f"{symbol}_{timeframe}"
                    results['optimization_results'][symbol_tf_key] = {}

                    try:
                        print(f"\n🎯 處理: {symbol} {timeframe}")

                        # Load OHLCV data directly from parquet files
                        import pandas as pd
                        parquet_path = f"data/raw/{symbol}_{timeframe}_ohlcv.parquet"

                        try:
                            df = pd.read_parquet(parquet_path)
                            print(f"   ✅ 從Parquet文件載入: {parquet_path}")
                        except FileNotFoundError:
                            # Try alternative path
                            alt_path = f"data/raw/{symbol}/{symbol}_{timeframe}_ohlcv.parquet"
                            try:
                                df = pd.read_parquet(alt_path)
                                print(f"   ✅ 從替代路徑載入: {alt_path}")
                            except FileNotFoundError:
                                print(f"   ❌ 無法找到數據文件: {symbol}_{timeframe}")
                                df = pd.DataFrame()

                        if df.empty:
                            logger.warning(f"No data for {symbol}_{timeframe}")
                            for stage in stages:
                                current_operation += 1
                                results['failed_optimizations'].append(f"{symbol_tf_key}_{stage}")
                            continue

                        print(f"   ✅ 載入數據: {len(df):,} 條記錄")
                        print(f"   📅 時間範圍: {df.index.min()} 到 {df.index.max()}")

                        # Stage 1: Generate features and labels, find top 30
                        if 'top30' in stages:
                            current_operation += 1
                            stage_progress = (current_operation / total_operations) * 100

                            print(f"\n   🔬 階段1: 特徵標籤生成 + Top30優化 ({stage_progress:.1f}%)")

                            # Generate comprehensive features using new system
                            print(f"      🔧 生成全量特徵...")
                            
                            # Use the new FeatureOptimizer to generate features
                            feature_optimizer = FeatureOptimizer()
                            
                            # Generate all features
                            features_df = feature_optimizer.generate_all_features(df)
                            initial_feature_count = len([col for col in features_df.columns
                                                        if col not in ['open', 'high', 'low', 'close', 'volume']])
                            
                            # Generate labels using the new system
                            print(f"      🏷️ 生成標籤...")
                            from src.optimization.label_optimizer import LabelOptimizer
                            label_optimizer = LabelOptimizer()
                            
                            # Generate labels with default parameters using new API
                            # 需要價格數據來生成標籤
                            price_data = df['close']  # 假設df包含OHLCV數據
                            
                            labels_series = label_optimizer.generate_labels(
                                price_data=price_data,
                                lag=6,  # 默認滯後期
                                pos_threshold=0.005,  # 0.5%正閾值
                                neg_threshold=-0.005,  # -0.5%負閾值
                                label_type='multiclass',  # 強制三分類
                                threshold_method='fixed'
                            )
                            
                            if labels_series is not None and len(labels_series) > 0:
                                labeled_df = features_df.copy()
                                labeled_df['label'] = labels_series
                                print(f"      ✅ 特徵和標籤生成完成")
                                
                                # 顯示標籤分布
                                label_dist = labels_series.value_counts(normalize=True).sort_index()
                                print(f"      📊 標籤分布: {dict(label_dist)}")
                            else:
                                print(f"      ❌ 標籤生成失敗，跳過此組合")
                                continue

                            if labeled_df.empty or 'label' not in labeled_df.columns:
                                logger.error(f"Label generation failed for {symbol}_{timeframe}")
                                results['failed_optimizations'].append(f"{symbol_tf_key}_top30")
                                continue

                            # 保存完整特徵和標籤數據到processed目錄
                            print(f"      💾 保存特徵和標籤數據...")
                            try:
                                # 分離特徵和標籤
                                exclude_cols = ['label', 'future_return']
                                X_all = labeled_df.drop(columns=[col for col in exclude_cols if col in labeled_df.columns])
                                y_all = labeled_df['label'] if 'label' in labeled_df.columns else None

                                # 使用現有的DataStorage保存方法
                                if self.storage is None:
                                    from src.data.storage import DataStorage
                                    self.storage = DataStorage()
                                    await self.storage.initialize()

                                # 保存完整特徵數據
                                feature_metadata = {
                                    'symbol': symbol,
                                    'timeframe': timeframe,
                                    'total_features': X_all.shape[1],
                                    'total_samples': X_all.shape[0],
                                    'generation_timestamp': datetime.now().isoformat()
                                }
                                await self.storage.store_feature_data(
                                    symbol, timeframe, f"{version}_full", X_all, feature_metadata
                                )

                                # 另外也保存到parquet文件
                                from pathlib import Path
                                features_dir = Path('data/processed/features')
                                labels_dir = Path('data/processed/labels')
                                features_dir.mkdir(exist_ok=True)
                                labels_dir.mkdir(exist_ok=True)

                                features_file = features_dir / f'{symbol}_{timeframe}_full_features.parquet'
                                X_all.to_parquet(features_file)
                                print(f"         ✅ 完整特徵已保存: {features_file}")

                                if y_all is not None:
                                    labels_file = labels_dir / f'{symbol}_{timeframe}_labels.parquet'
                                    pd.DataFrame({'label': y_all}).to_parquet(labels_file)
                                    print(f"         ✅ 標籤已保存: {labels_file}")

                            except Exception as save_error:
                                logger.error(f"Failed to save feature/label data: {save_error}")
                                print(f"         ⚠️ 數據保存失敗: {save_error}")

                            print(f"      ✅ 特徵數量: {initial_feature_count}")
                            print(f"      ✅ 樣本數量: {len(labeled_df):,}")

                            # Label distribution
                            label_dist = labeled_df['label'].value_counts(normalize=True)
                            print(f"      📊 標籤分佈: Long:{label_dist.get(1, 0)*100:.1f}% | "
                                  f"Neutral:{label_dist.get(0, 0)*100:.1f}% | Short:{label_dist.get(-1, 0)*100:.1f}%")

                            # Feature optimization for top 30
                            print(f"      🎯 特徵優化 - 選擇Top30特徵...")

                            # Use new FeatureSelector
                            feature_selector = FeatureSelector()
                            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'label', 'future_return']
                            X = labeled_df.drop(columns=[col for col in exclude_cols if col in labeled_df.columns])
                            y = labeled_df['label']

                            # Select top 30 features using combined method
                            selection_result = feature_selector.select_features(
                                X, y,
                                method='combined',
                                n_features=30,
                                params={
                                    'importance_threshold': 0.01,
                                    'correlation_threshold': 0.95,
                                    'variance_threshold': 0.01
                                }
                            )
                            
                            if selection_result and 'selected_features' in selection_result:
                                top30_features = selection_result['selected_features']
                                top30_scores = selection_result.get('feature_scores', {})
                            else:
                                # Fallback to basic selection
                                from sklearn.ensemble import RandomForestClassifier
                                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                                rf.fit(X, y)
                                feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
                                top30_features = feature_importance.nlargest(30).index.tolist()
                                top30_scores = dict(feature_importance.nlargest(30))

                            if top30_features:
                                results['optimization_results'][symbol_tf_key]['top30'] = {
                                    'features': top30_features,
                                    'scores': top30_scores,
                                    'feature_count': len(top30_features),
                                    'optimization_method': 'combined'
                                }
                                successful_operations += 1
                                print(f"      ✅ Top30特徵選出: {len(top30_features)}個")
                                print(f"         前5特徵: {', '.join(top30_features[:5])}")
                            else:
                                results['failed_optimizations'].append(f"{symbol_tf_key}_top30")
                                continue

                        # Stage 2: First refinement to top 15
                        if 'top15_stage1' in stages and 'top30' in results['optimization_results'][symbol_tf_key]:
                            current_operation += 1
                            stage_progress = (current_operation / total_operations) * 100

                            print(f"\n   🔬 階段2: Top15第一輪優化 ({stage_progress:.1f}%)")

                            # Use top 30 features from stage 1
                            top30_features = results['optimization_results'][symbol_tf_key]['top30']['features']
                            X_top30 = labeled_df[top30_features]

                            print(f"      📊 輸入特徵: {len(top30_features)}個")

                            # Second round optimization using FeatureSelector
                            selection_result_stage1 = feature_selector.select_features(
                                X_top30, y,
                                method='mutual_info',
                                n_features=15,
                                params={'importance_threshold': 0.02}
                            )
                            
                            if selection_result_stage1 and 'selected_features' in selection_result_stage1:
                                top15_stage1_features = selection_result_stage1['selected_features']
                                top15_stage1_scores = selection_result_stage1.get('feature_scores', {})
                            else:
                                # Fallback
                                from sklearn.feature_selection import SelectKBest, mutual_info_classif
                                selector = SelectKBest(mutual_info_classif, k=min(15, len(X_top30.columns)))
                                selector.fit(X_top30, y)
                                feature_scores = pd.Series(selector.scores_, index=X_top30.columns)
                                top15_stage1_features = feature_scores.nlargest(15).index.tolist()
                                top15_stage1_scores = dict(feature_scores.nlargest(15))

                            if top15_stage1_features:
                                results['optimization_results'][symbol_tf_key]['top15_stage1'] = {
                                    'features': top15_stage1_features,
                                    'scores': top15_stage1_scores,
                                    'feature_count': len(top15_stage1_features),
                                    'optimization_method': 'lightgbm'
                                }
                                successful_operations += 1
                                print(f"      ✅ Top15第一輪: {len(top15_stage1_features)}個")
                                print(f"         前5特徵: {', '.join(top15_stage1_features[:5])}")
                            else:
                                results['failed_optimizations'].append(f"{symbol_tf_key}_top15_stage1")

                        # Stage 3: Final refinement to top 15
                        if 'top15_stage2' in stages and 'top15_stage1' in results['optimization_results'][symbol_tf_key]:
                            current_operation += 1
                            stage_progress = (current_operation / total_operations) * 100

                            print(f"\n   🔬 階段3: Top15第二輪優化 ({stage_progress:.1f}%)")

                            # Use top 15 features from stage 2
                            top15_stage1_features = results['optimization_results'][symbol_tf_key]['top15_stage1']['features']
                            X_top15_stage1 = labeled_df[top15_stage1_features]

                            print(f"      📊 輸入特徵: {len(top15_stage1_features)}個")

                            # Final optimization using tree-based method
                            selection_result_final = feature_selector.select_features(
                                X_top15_stage1, y,
                                method='tree_based',
                                n_features=15,
                                params={
                                    'importance_threshold': 0.01,
                                    'model_type': 'lightgbm'
                                }
                            )
                            
                            if selection_result_final and 'selected_features' in selection_result_final:
                                top15_final_features = selection_result_final['selected_features']
                                top15_final_scores = selection_result_final.get('feature_scores', {})
                            else:
                                # Fallback to chi2
                                from sklearn.feature_selection import chi2
                                from sklearn.preprocessing import MinMaxScaler
                                
                                scaler = MinMaxScaler()
                                X_scaled = pd.DataFrame(
                                    scaler.fit_transform(X_top15_stage1),
                                    columns=X_top15_stage1.columns,
                                    index=X_top15_stage1.index
                                )
                                
                                chi2_scores, _ = chi2(X_scaled, y)
                                feature_scores_final = pd.Series(chi2_scores, index=X_top15_stage1.columns)
                                top15_final_features = feature_scores_final.nlargest(min(15, len(feature_scores_final))).index.tolist()
                                top15_final_scores = dict(feature_scores_final.nlargest(min(15, len(feature_scores_final))))

                            if top15_final_features:
                                results['optimization_results'][symbol_tf_key]['top15_stage2'] = {
                                    'features': top15_final_features,
                                    'scores': top15_final_scores,
                                    'feature_count': len(top15_final_features),
                                    'optimization_method': 'mutual_info'
                                }
                                successful_operations += 1
                                print(f"      ✅ Top15最終: {len(top15_final_features)}個")
                                print(f"         最終特徵: {', '.join(top15_final_features)}")

                                # Save final feature configuration
                                if version:
                                    try:
                                        # Manual save as JSON
                                        config_data = {
                                            'symbol': symbol,
                                            'timeframe': timeframe,
                                            'version': version,
                                            'features': top15_final_features,
                                            'scores': top15_final_scores,
                                            'timestamp': datetime.now().isoformat()
                                        }
                                        config_path = f"data/processed/features/{symbol}_{timeframe}_{version}_features.json"
                                        import json
                                        from pathlib import Path
                                        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
                                        with open(config_path, 'w') as f:
                                            json.dump(config_data, f, indent=2)
                                        print(f"      💾 手動保存特徵配置: {config_path}")
                                    except Exception as e:
                                        print(f"      ⚠️ 特徵配置保存失敗: {e}")

                                # 保存最終選出的特徵數據
                                print(f"      💾 保存Top15特徵數據...")
                                try:
                                    # 獲取最終特徵的數據
                                    X_top15 = labeled_df[top15_final_features]
                                    y_final = labeled_df['label'] if 'label' in labeled_df.columns else None

                                    # 保存Top15特徵數據
                                    from pathlib import Path
                                    features_dir = Path('data/processed/features')
                                    labels_dir = Path('data/processed/labels')
                                    features_dir.mkdir(exist_ok=True)
                                    labels_dir.mkdir(exist_ok=True)

                                    top15_features_file = features_dir / f'{symbol}_{timeframe}_top15_features.parquet'
                                    X_top15.to_parquet(top15_features_file)
                                    print(f"         ✅ Top15特徵數據: {top15_features_file}")

                                    # 保存訓練就緒的數據集
                                    training_data = X_top15.copy()
                                    training_data['label'] = y_final
                                    training_ready_file = features_dir / f'{symbol}_{timeframe}_training_ready.parquet'
                                    training_data.to_parquet(training_ready_file)
                                    print(f"         ✅ 訓練就緒數據: {training_ready_file}")

                                    # 使用現有方法保存到數據庫
                                    feature_metadata_top15 = {
                                        'symbol': symbol,
                                        'timeframe': timeframe,
                                        'selected_features': top15_final_features,
                                        'feature_scores': top15_final_scores,
                                        'optimization_method': 'mutual_info',
                                        'samples_count': len(X_top15),
                                        'generation_timestamp': datetime.now().isoformat()
                                    }

                                    if self.storage:
                                        await self.storage.store_feature_data(
                                            symbol, timeframe, version, X_top15, feature_metadata_top15
                                        )
                                        print(f"         ✅ 數據庫保存完成")

                                except Exception as save_error:
                                    logger.error(f"Failed to save Top15 feature data: {save_error}")
                                    print(f"         ⚠️ Top15數據保存失敗: {save_error}")
                            else:
                                results['failed_optimizations'].append(f"{symbol_tf_key}_top15_stage2")

                        results['stages_completed'].extend([f"{symbol_tf_key}_{stage}" for stage in stages
                                                          if stage in results['optimization_results'][symbol_tf_key]])

                    except Exception as e:
                        logger.error(f"Feature optimization failed for {symbol}_{timeframe}: {e}")
                        for stage in stages:
                            if f"{symbol_tf_key}_{stage}" not in results['failed_optimizations']:
                                results['failed_optimizations'].append(f"{symbol_tf_key}_{stage}")

            # Generate summary
            self.stats['end_time'] = datetime.now()
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

            results['summary'] = {
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'failed_operations': len(results['failed_optimizations']),
                'success_rate_percent': (successful_operations / total_operations * 100) if total_operations > 0 else 0,
                'duration_seconds': duration,
                'symbols_processed': len(symbols),
                'timeframes_processed': len(timeframes),
                'stages_processed': len(stages),
                'stages_completed': len(results['stages_completed'])
            }

            print(f"\n" + "=" * 60)
            print(f"🎉 特徵優化完成！")
            print(f"   成功: {successful_operations}/{total_operations} ({successful_operations/total_operations*100:.1f}%)")
            print(f"   完成階段: {len(results['stages_completed'])}")
            print(f"   耗時: {duration:.1f} 秒")

            if results['failed_optimizations']:
                print(f"   ⚠️ 失敗的操作: {len(results['failed_optimizations'])}")
                for failed in results['failed_optimizations'][:5]:
                    print(f"     - {failed}")

            # Show final results for each symbol/timeframe
            for symbol_tf_key, optimization_result in results['optimization_results'].items():
                if 'top15_stage2' in optimization_result:
                    final_features = optimization_result['top15_stage2']['features']
                    print(f"\n📊 {symbol_tf_key} 最終Top15特徵:")
                    for i, feature in enumerate(final_features, 1):
                        score = optimization_result['top15_stage2']['scores'].get(feature, 0)
                        print(f"   {i:2d}. {feature} (score: {score:.4f})")

            logger.info(f"Feature optimization completed: {successful_operations} successful, {len(results['failed_optimizations'])} failed")

            return results

        except Exception as e:
            logger.error(f"Feature optimization with Optuna failed: {e}")
            return {'success': False, 'error': str(e)}


async def main():
    """Main function for model training script"""
    parser = argparse.ArgumentParser(description='Model Training Script')

    # Command selection
    parser.add_argument('command', choices=[
        'batch', 'single', 'retrain', 'compare', 'feature-optimize'
    ], help='Training command')

    # Symbol and timeframe selection
    parser.add_argument('--symbols', nargs='+',
                       default=['BTCUSDT', 'ETHUSDT'],
                       help='Trading symbols')

    parser.add_argument('--timeframes', nargs='+',
                       default=['1h', '4h'],
                       help='Timeframes')

    parser.add_argument('--version',
                       help='Version tag for models/features')

    # Model options
    parser.add_argument('--models', nargs='+',
                       default=['lightgbm', 'xgboost', 'random_forest'],
                       choices=['lightgbm', 'xgboost', 'random_forest', 'svm', 'neural_network', 'lstm'],
                       help='Model types to train')

    parser.add_argument('--no-optimization', action='store_true',
                       help='Skip hyperparameter optimization')

    parser.add_argument('--no-feature-selection', action='store_true',
                       help='Skip feature selection')

    # Data split options
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (0-1)')

    parser.add_argument('--validation-size', type=float, default=0.2,
                       help='Validation set size (0-1)')

    # Output options
    parser.add_argument('--output',
                       help='Output file for results (JSON format)')

    # Logging
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Setup logging
    import logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Initialize pipeline
    pipeline = TrainingPipeline()

    if not await pipeline.initialize():
        logger.error("Failed to initialize training pipeline")
        return False

    try:
        results = None

        if args.command == 'batch':
            # Batch training
            results = await pipeline.train_models_batch(
                symbols=args.symbols,
                timeframes=args.timeframes,
                version=args.version,
                model_types=args.models,
                optimize_hyperparameters=not args.no_optimization,
                feature_selection=not args.no_feature_selection,
                test_size=args.test_size,
                validation_size=args.validation_size
            )

        elif args.command == 'single':
            # Single model training
            if len(args.symbols) != 1 or len(args.timeframes) != 1 or len(args.models) != 1:
                logger.error("Single command requires exactly one symbol, timeframe, and model")
                return False

            results = await pipeline.train_single_model(
                symbol=args.symbols[0],
                timeframe=args.timeframes[0],
                model_type=args.models[0],
                version=args.version,
                optimize_hyperparameters=not args.no_optimization
            )

        elif args.command == 'retrain':
            # Retrain best models
            results = await pipeline.retrain_best_models(
                symbols=args.symbols,
                timeframes=args.timeframes,
                version=args.version
            )

        elif args.command == 'compare':
            # Model comparison (placeholder)
            logger.info("Model comparison not yet implemented")
            return False

        elif args.command == 'feature-optimize':
            # Feature optimization with Optuna
            results = await pipeline.optimize_features_with_optuna(
                symbols=args.symbols,
                timeframes=args.timeframes,
                version=args.version,
                stages=['top30', 'top15_stage1', 'top15_stage2']
            )

        # Output results
        if results:
            if args.output:
                # Save to file
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Results saved to {args.output}")
            else:
                # Print summary
                if results.get('training_summary'):
                    summary = results['training_summary']
                    print(f"Training Summary:")
                    print(f"  Success Rate: {summary.get('success_rate_percent', 0):.1f}%")
                    print(f"  Models Trained: {summary.get('total_models_trained', 0)}")
                    print(f"  Avg Training Time: {summary.get('average_training_time_seconds', 0):.1f}s")

                if results.get('best_models', {}).get('top_model'):
                    top_model = list(results['best_models']['top_model'].values())[0]
                    print(f"  Best Model: {top_model['model_type']} (Score: {top_model['score']:.4f})")

            return results.get('success', True)
        else:
            logger.error("No results generated")
            return False

    except KeyboardInterrupt:
        logger.info("Training cancelled by user")
        return False
    except Exception as e:
        logger.error(f"Training script failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
