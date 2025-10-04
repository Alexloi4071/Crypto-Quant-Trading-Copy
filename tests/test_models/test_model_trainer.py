"""
Test Model Trainer Module
Comprehensive tests for machine learning model training functionality
Tests model training, hyperparameter optimization, and model management
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import pickle
from unittest.mock import Mock, patch, AsyncMock
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.model_trainer import ModelTrainer
from src.models.model_manager import ModelManager
from src.features.feature_engineering import FeatureEngineer
# NOTE: 測試文件：舊label生成器依賴移除，對應測試需改為讀取版本化標籤


class TestModelTrainer:
    """Test suite for ModelTrainer class"""

    @pytest.fixture
    def model_trainer(self):
        """Create ModelTrainer instance for testing"""
        return ModelTrainer()

    @pytest.fixture
    def sample_features_labels(self):
        """Generate sample features and labels for testing"""
        np.random.seed(42)  # For reproducible tests

        # Create feature data
        n_samples = 1000
        n_features = 20

        dates = pd.date_range('2024-01-01', periods=n_samples, freq='1H')

        # Generate features with some correlation structure
        features = np.random.randn(n_samples, n_features)
        features[:, 1] = features[:, 0] * 0.5 + np.random.randn(n_samples) * 0.5  # Correlated feature

        feature_names = [f'feature_{i}' for i in range(n_features)]
        features_df = pd.DataFrame(features, columns=feature_names, index=dates)

        # Generate labels (classification: 0=sell, 1=hold, 2=buy)
        # Make labels somewhat predictable based on features for testing
        labels = np.zeros(n_samples)
        labels[features[:, 0] > 0.5] = 2  # Buy signal
        labels[features[:, 0] < -0.5] = 0  # Sell signal
        labels[(features[:, 0] >= -0.5) & (features[:, 0] <= 0.5)] = 1  # Hold signal

        labels_df = pd.DataFrame({'label': labels}, index=dates)

        return features_df, labels_df

    @pytest.fixture
    def small_dataset(self):
        """Generate small dataset for quick testing"""
        np.random.seed(42)

        n_samples = 100
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='1H')

        features_df = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'feature_4': np.random.randn(n_samples),
            'feature_5': np.random.randn(n_samples)
        }, index=dates)

        # Simple binary classification
        labels = (features_df['feature_1'] + features_df['feature_2'] > 0).astype(int)
        labels_df = pd.DataFrame({'label': labels}, index=dates)

        return features_df, labels_df

    @pytest.mark.asyncio
    async def test_train_model_success(self, model_trainer, small_dataset):
        """Test successful model training"""
        features_df, labels_df = small_dataset

        result = await model_trainer.train_model(
            symbol='BTCUSDT',
            timeframe='1h',
            version='test_v1',
            features_df=features_df,
            labels_df=labels_df,
            model_type='lightgbm',
            optimize_hyperparameters=False  # Skip optimization for speed
        )

        # Should return success
        assert result['success'] is True
        assert 'best_score' in result
        assert 'metrics' in result
        assert 'model_path' in result

        # Score should be reasonable
        assert 0 <= result['best_score'] <= 1

    @pytest.mark.asyncio
    async def test_train_model_with_optimization(self, model_trainer, small_dataset):
        """Test model training with hyperparameter optimization"""
        features_df, labels_df = small_dataset

        with patch.object(model_trainer.hyperparameter_optimizer, 'optimize_hyperparameters') as mock_optimize:
            mock_optimize.return_value = {
                'best_params': {'n_estimators': 100, 'learning_rate': 0.1},
                'best_score': 0.85,
                'optimization_time': 10.5
            }

            result = await model_trainer.train_model(
                symbol='BTCUSDT',
                timeframe='1h',
                version='test_v1',
                features_df=features_df,
                labels_df=labels_df,
                model_type='lightgbm',
                optimize_hyperparameters=True
            )

            assert result['success'] is True
            assert result['best_score'] == 0.85
            mock_optimize.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_model_invalid_data(self, model_trainer):
        """Test model training with invalid data"""
        # Empty DataFrame
        empty_df = pd.DataFrame()

        result = await model_trainer.train_model(
            symbol='BTCUSDT',
            timeframe='1h',
            version='test_v1',
            features_df=empty_df,
            labels_df=empty_df,
            model_type='lightgbm'
        )

        assert result['success'] is False
        assert 'error' in result

    @pytest.mark.asyncio
    async def test_train_model_mismatched_indices(self, model_trainer):
        """Test model training with mismatched feature and label indices"""
        features_df = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1H'))

        labels_df = pd.DataFrame({
            'label': [0, 1, 0]
        }, index=pd.date_range('2024-01-02', periods=3, freq='1H'))  # Different dates

        result = await model_trainer.train_model(
            symbol='BTCUSDT',
            timeframe='1h',
            version='test_v1',
            features_df=features_df,
            labels_df=labels_df,
            model_type='lightgbm'
        )

        # Should handle mismatched indices gracefully
        # Either succeed with aligned data or fail with appropriate error
        if result['success']:
            assert len(result['metrics']) > 0
        else:
            assert 'error' in result


    def test_supported_models(self, model_trainer):
        """Test that trainer supports expected model types"""
        expected_models = ['lightgbm', 'xgboost', 'random_forest', 'svm', 'neural_network', 'lstm']

        for model_type in expected_models:
            assert model_type in model_trainer.supported_models


    def test_prepare_data(self, model_trainer, sample_features_labels):
        """Test data preparation method"""
        features_df, labels_df = sample_features_labels

        X_train, X_test, y_train, y_test = model_trainer._prepare_data(
            features_df, labels_df, test_size=0.2
        )

        # Check dimensions
        assert len(X_train) + len(X_test) == len(features_df)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

        # Check no data leakage (test set should be after train set in time)
        assert X_train.index.max() <= X_test.index.min()


    def test_evaluate_model(self, model_trainer, small_dataset):
        """Test model evaluation"""
        features_df, labels_df = small_dataset

        # Train a simple model for testing
        X_train, X_test, y_train, y_test = model_trainer._prepare_data(
            features_df, labels_df, test_size=0.3
        )

        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=10, random_state=42, verbosity=-1)
        model.fit(X_train, y_train)

        metrics = model_trainer._evaluate_model(model, X_test, y_test)

        # Should return comprehensive metrics
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1


    def test_save_and_load_model(self, model_trainer, small_dataset):
        """Test model saving and loading"""
        features_df, labels_df = small_dataset

        # Train a simple model
        X_train, X_test, y_train, y_test = model_trainer._prepare_data(
            features_df, labels_df, test_size=0.3
        )

        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=10, random_state=42, verbosity=-1)
        model.fit(X_train, y_train)

        # Test saving
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / 'test_model.pkl'

            success = model_trainer._save_model(model, str(model_path), {
                'symbol': 'BTCUSDT',
                'timeframe': '1h',
                'version': 'test',
                'model_type': 'lightgbm'
            })

            assert success is True
            assert model_path.exists()

            # Test loading
            loaded_model, metadata = model_trainer._load_model(str(model_path))

            assert loaded_model is not None
            assert metadata['symbol'] == 'BTCUSDT'
            assert metadata['model_type'] == 'lightgbm'

            # Test that loaded model makes same predictions
            original_pred = model.predict(X_test[:5])
            loaded_pred = loaded_model.predict(X_test[:5])

            np.testing.assert_array_equal(original_pred, loaded_pred)


class TestModelManager:
    """Test suite for ModelManager class"""

    @pytest.fixture
    def model_manager(self):
        """Create ModelManager instance for testing"""
        return ModelManager()

    @pytest.fixture
    def sample_model_metadata(self):
        """Generate sample model metadata"""
        return {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'version': 'v1.0',
            'model_type': 'lightgbm',
            'training_date': datetime.now(),
            'metrics': {
                'accuracy': 0.75,
                'precision': 0.72,
                'recall': 0.78,
                'f1_score': 0.75
            },
            'feature_count': 20,
            'training_samples': 1000
        }


    def test_register_model(self, model_manager, sample_model_metadata):
        """Test model registration"""
        model_id = 'test_model_001'

        success = model_manager.register_model(model_id, sample_model_metadata)

        assert success is True
        assert model_id in model_manager.models

        # Test retrieving registered model
        retrieved_metadata = model_manager.get_model_metadata(model_id)
        assert retrieved_metadata == sample_model_metadata


    def test_get_best_model(self, model_manager):
        """Test getting best model for symbol/timeframe"""
        # Register multiple models
        models = [
            ('model_1', {'symbol': 'BTCUSDT', 'timeframe': '1h', 'metrics': {'accuracy': 0.70}}),
            ('model_2', {'symbol': 'BTCUSDT', 'timeframe': '1h', 'metrics': {'accuracy': 0.85}}),
            ('model_3', {'symbol': 'BTCUSDT', 'timeframe': '1h', 'metrics': {'accuracy': 0.60}}),
            ('model_4', {'symbol': 'ETHUSDT', 'timeframe': '1h', 'metrics': {'accuracy': 0.75}})
        ]

        for model_id, metadata in models:
            model_manager.register_model(model_id, metadata)

        # Get best model for BTCUSDT 1h
        best_model_id = model_manager.get_best_model('BTCUSDT', '1h')

        assert best_model_id == 'model_2'  # Highest accuracy

        # Test with non-existent symbol/timeframe
        no_model = model_manager.get_best_model('UNKNOWN', '1d')
        assert no_model is None


    def test_model_comparison(self, model_manager):
        """Test model comparison functionality"""
        # Register models with different performance
        models = [
            ('model_A', {'metrics': {'accuracy': 0.80, 'precision': 0.75, 'recall': 0.85}}),
            ('model_B', {'metrics': {'accuracy': 0.85, 'precision': 0.80, 'recall': 0.80}}),
            ('model_C', {'metrics': {'accuracy': 0.75, 'precision': 0.85, 'recall': 0.70}})
        ]

        for model_id, metadata in models:
            model_manager.register_model(model_id, metadata)

        comparison = model_manager.compare_models(['model_A', 'model_B', 'model_C'])

        assert 'model_rankings' in comparison
        assert 'best_model' in comparison
        assert len(comparison['model_rankings']) == 3

        # Best model should be model_B (highest accuracy)
        assert comparison['best_model'] == 'model_B'


    def test_model_versioning(self, model_manager):
        """Test model versioning functionality"""
        base_metadata = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'model_type': 'lightgbm'
        }

        # Register different versions
        versions = ['v1.0', 'v1.1', 'v2.0']
        for version in versions:
            metadata = base_metadata.copy()
            metadata['version'] = version
            metadata['metrics'] = {'accuracy': np.random.random()}

            model_manager.register_model(f'model_{version}', metadata)

        # Get all versions
        all_versions = model_manager.get_model_versions('BTCUSDT', '1h', 'lightgbm')

        assert len(all_versions) == 3
        assert all(version in [m['version'] for m in all_versions] for version in versions)


    def test_model_lifecycle(self, model_manager, sample_model_metadata):
        """Test complete model lifecycle"""
        model_id = 'lifecycle_test_model'

        # 1. Register model
        success = model_manager.register_model(model_id, sample_model_metadata)
        assert success is True

        # 2. Update model status
        model_manager.update_model_status(model_id, 'active')
        metadata = model_manager.get_model_metadata(model_id)
        assert metadata.get('status') == 'active'

        # 3. Archive model
        model_manager.archive_model(model_id)
        metadata = model_manager.get_model_metadata(model_id)
        assert metadata.get('status') == 'archived'

        # 4. Delete model
        model_manager.delete_model(model_id)
        assert model_id not in model_manager.models


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class"""

    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance for testing"""
        return FeatureEngineer()

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Generate sample OHLCV data for feature engineering"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='1H')

        base_price = 45000
        prices = base_price + np.cumsum(np.random.randn(200) * 50)

        return pd.DataFrame({
            'open': prices,
            'high': prices * 1.002 + np.random.randn(200) * 10,
            'low': prices * 0.998 - np.random.randn(200) * 10,
            'close': prices + np.random.randn(200) * 20,
            'volume': np.random.randint(1000, 5000, 200)
        }, index=dates)


    def test_generate_features(self, feature_engineer, sample_ohlcv_data):
        """Test feature generation"""
        features_df, feature_info = feature_engineer.generate_features(sample_ohlcv_data)

        # Should return non-empty DataFrame
        assert not features_df.empty
        assert len(features_df) <= len(sample_ohlcv_data)  # May be shorter due to lookback periods

        # Should have feature metadata
        assert isinstance(feature_info, dict)
        assert 'feature_count' in feature_info
        assert 'feature_names' in feature_info

        # Should have generated multiple features
        assert len(features_df.columns) > 10

        # Should not have any infinite values
        assert not np.isinf(features_df.select_dtypes(include=[np.number])).any().any()


    def test_technical_indicators(self, feature_engineer, sample_ohlcv_data):
        """Test specific technical indicators"""
        # Test RSI calculation
        rsi = feature_engineer._calculate_rsi(sample_ohlcv_data['close'])

        # RSI should be between 0 and 100
        assert (rsi >= 0).all() and (rsi <= 100).all()

        # Test MACD
        macd_line, signal_line, histogram = feature_engineer._calculate_macd(sample_ohlcv_data['close'])

        # Should have same length as input (excluding NaN values)
        assert len(macd_line.dropna()) > 0
        assert len(signal_line.dropna()) > 0
        assert len(histogram.dropna()) > 0

        # Test Bollinger Bands
        upper, middle, lower = feature_engineer._calculate_bollinger_bands(sample_ohlcv_data['close'])

        # Upper should be >= middle >= lower
        valid_indices = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_indices] >= middle[valid_indices]).all()
        assert (middle[valid_indices] >= lower[valid_indices]).all()


    def test_feature_engineering_edge_cases(self, feature_engineer):
        """Test feature engineering with edge cases"""
        # Test with minimal data
        minimal_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1H'))

        features_df, feature_info = feature_engineer.generate_features(minimal_data)

        # Should handle minimal data gracefully
        assert isinstance(features_df, pd.DataFrame)
        assert isinstance(feature_info, dict)

        # Test with constant prices
        constant_data = pd.DataFrame({
            'open': [100] * 50,
            'high': [100] * 50,
            'low': [100] * 50,
            'close': [100] * 50,
            'volume': [1000] * 50
        }, index=pd.date_range('2024-01-01', periods=50, freq='1H'))

        features_df, feature_info = feature_engineer.generate_features(constant_data)

        # Should handle constant data without errors
        assert isinstance(features_df, pd.DataFrame)


class TestLabelGenerator:
    """Test suite for LabelGenerator class"""

    @pytest.fixture
    def label_generator(self):
        """Create LabelGenerator instance for testing"""
        return LabelGenerator()

    @pytest.fixture
    def sample_price_data(self):
        """Generate sample price data for label generation"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=500, freq='1H')

        base_price = 45000
        prices = base_price + np.cumsum(np.random.randn(500) * 50)

        return pd.DataFrame({
            'close': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'volume': np.random.randint(1000, 5000, 500)
        }, index=dates)


    def test_generate_labels(self, label_generator, sample_price_data):
        """Test label generation"""
        labels_df = label_generator.generate_labels(sample_price_data)

        # Should return non-empty DataFrame
        assert not labels_df.empty
        assert 'label' in labels_df.columns

        # Labels should be valid (0, 1, 2 for sell, hold, buy)
        unique_labels = labels_df['label'].unique()
        assert all(label in [0, 1, 2] for label in unique_labels)

        # Should have fewer rows than input due to future return calculation
        assert len(labels_df) <= len(sample_price_data)


    def test_label_distribution(self, label_generator, sample_price_data):
        """Test label distribution"""
        labels_df = label_generator.generate_labels(sample_price_data)

        # Get label distribution
        label_counts = labels_df['label'].value_counts()

        # Should have reasonable distribution (not all one class)
        assert len(label_counts) > 1

        # Each class should have some representation
        min_class_count = len(labels_df) * 0.1  # At least 10% for each class
        for count in label_counts.values:
            assert count >= min_class_count * 0.5  # Allow some imbalance


    def test_label_consistency(self, label_generator):
        """Test label generation consistency"""
        # Create deterministic price data
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        prices = [45000 + i * 10 for i in range(100)]  # Steadily increasing

        price_data = pd.DataFrame({
            'close': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'volume': [1000] * 100
        }, index=dates)

        # Generate labels twice
        labels_1 = label_generator.generate_labels(price_data)
        labels_2 = label_generator.generate_labels(price_data)

        # Should be identical
        pd.testing.assert_frame_equal(labels_1, labels_2)


    def test_future_return_calculation(self, label_generator):
        """Test future return calculation"""
        # Create simple price series
        prices = pd.Series([100, 102, 104, 103, 101, 105],
                          index=pd.date_range('2024-01-01', periods=6, freq='1H'))

        future_returns = label_generator._calculate_future_returns(prices, periods=2)

        # Check specific calculations
        # Price at t=0 is 100, at t=2 is 104, so return should be 4%
        expected_return_0 = (104 - 100) / 100
        np.testing.assert_almost_equal(future_returns.iloc[0], expected_return_0, decimal=4)

        # Price at t=1 is 102, at t=3 is 103, so return should be ~0.98%
        expected_return_1 = (103 - 102) / 102
        np.testing.assert_almost_equal(future_returns.iloc[1], expected_return_1, decimal=4)


    def test_return_to_label_conversion(self, label_generator):
        """Test conversion of returns to labels"""
        # Create returns with known distribution
        returns = pd.Series([-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05])

        labels = label_generator._returns_to_labels(returns)

        # Check that extreme negative returns become sell signals (0)
        assert labels.iloc[0] == 0  # -5% return

        # Check that extreme positive returns become buy signals (2)
        assert labels.iloc[-1] == 2  # +5% return

        # Check that small returns become hold signals (1)
        assert labels.iloc[3] == 1  # 0% return

# Integration Tests


class TestModelingIntegration:
    """Integration tests for modeling components working together"""

    @pytest.fixture
    def integrated_pipeline(self):
        """Create integrated modeling pipeline"""
        return {
            'feature_engineer': FeatureEngineer(),
            'label_generator': LabelGenerator(),
            'model_trainer': ModelTrainer(),
            'model_manager': ModelManager()
        }


    def test_full_modeling_pipeline(self, integrated_pipeline):
        """Test complete modeling pipeline"""
        # Create sample OHLCV data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=500, freq='1H')

        base_price = 45000
        prices = base_price + np.cumsum(np.random.randn(500) * 50)

        ohlcv_data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices + np.random.randn(500) * 20,
            'volume': np.random.randint(1000, 5000, 500)
        }, index=dates)

        # Step 1: Feature Engineering
        features_df, feature_info = integrated_pipeline['feature_engineer'].generate_features(ohlcv_data)
        assert not features_df.empty

        # Step 2: Label Generation
        labels_df = integrated_pipeline['label_generator'].generate_labels(features_df)
        assert not labels_df.empty

        # Step 3: Model Training (async test would be more complex, so we'll mock it)
        with patch.object(integrated_pipeline['model_trainer'], 'train_model') as mock_train:
            mock_train.return_value = {
                'success': True,
                'best_score': 0.75,
                'metrics': {'accuracy': 0.75, 'precision': 0.73},
                'model_path': '/path/to/model.pkl'
            }

            # This would be async in real usage
            training_result = mock_train.return_value

            assert training_result['success'] is True

            # Step 4: Model Management
            model_metadata = {
                'symbol': 'BTCUSDT',
                'timeframe': '1h',
                'version': 'test_v1',
                'model_type': 'lightgbm',
                'metrics': training_result['metrics'],
                'feature_count': len(features_df.columns)
            }

            success = integrated_pipeline['model_manager'].register_model('test_model', model_metadata)
            assert success is True

# Performance Tests


class TestModelingPerformance:
    """Performance tests for modeling operations"""

    def test_feature_generation_performance(self):
        """Test feature generation performance with large dataset"""
        # Create large dataset
        size = 5000
        dates = pd.date_range('2020-01-01', periods=size, freq='1H')

        np.random.seed(42)
        base_price = 45000
        prices = base_price + np.cumsum(np.random.randn(size) * 50)

        large_ohlcv = pd.DataFrame({
            'open': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices + np.random.randn(size) * 20,
            'volume': np.random.randint(1000, 5000, size)
        }, index=dates)

        feature_engineer = FeatureEngineer()

        import time
        start_time = time.time()

        features_df, feature_info = feature_engineer.generate_features(large_ohlcv)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete within reasonable time
        assert processing_time < 30.0  # 30 seconds max
        assert not features_df.empty
        assert len(features_df.columns) > 10

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
