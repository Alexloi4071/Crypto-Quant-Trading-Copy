"""
Test Data Manager Module
Comprehensive tests for data management functionality
Tests data download, storage, preprocessing, and retrieval
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager
from src.data.storage import DataStorage
from src.data.preprocessor import DataPreprocessor
from src.data.external_apis import ExternalDataManager


class TestDataManager:
    """Test suite for DataManager class"""

    @pytest.fixture
    async def data_manager(self):
        """Create DataManager instance for testing"""
        manager = DataManager()
        await manager.initialize()
        yield manager
        # Cleanup
        try:
            await manager.close()
        except:
            pass

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Generate sample OHLCV data for testing"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)  # For reproducible tests

        base_price = 45000
        prices = base_price + np.cumsum(np.random.randn(100) * 50)

        return pd.DataFrame({
            'open': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices + np.random.randn(100) * 10,
            'volume': np.random.randint(1000, 5000, 100)
        }, index=dates)

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test DataManager initialization"""
        manager = DataManager()
        result = await manager.initialize()

        assert result is True
        assert manager.storage is not None
        assert manager.preprocessor is not None

        # Cleanup
        await manager.close()

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_data_success(self, data_manager, sample_ohlcv_data):
        """Test successful OHLCV data fetching"""
        # Mock the exchange manager
        with patch.object(data_manager, 'exchange_manager') as mock_exchange:
            mock_exchange.fetch_ohlcv = AsyncMock(return_value=sample_ohlcv_data)

            result = await data_manager.fetch_ohlcv_data(
                symbol='BTCUSDT',
                timeframe='1h',
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5)
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])

            # Verify the mock was called
            mock_exchange.fetch_ohlcv.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_data_failure(self, data_manager):
        """Test OHLCV data fetching with failure"""
        with patch.object(data_manager, 'exchange_manager') as mock_exchange:
            mock_exchange.fetch_ohlcv = AsyncMock(side_effect=Exception("API Error"))

            result = await data_manager.fetch_ohlcv_data(
                symbol='BTCUSDT',
                timeframe='1h'
            )

            assert isinstance(result, pd.DataFrame)
            assert result.empty

    @pytest.mark.asyncio
    async def test_store_ohlcv_data(self, data_manager, sample_ohlcv_data):
        """Test storing OHLCV data"""
        with patch.object(data_manager, 'storage') as mock_storage:
            mock_storage.store_ohlcv_data = AsyncMock(return_value=True)

            result = await data_manager.store_ohlcv_data(
                symbol='BTCUSDT',
                timeframe='1h',
                data=sample_ohlcv_data
            )

            assert result is True
            mock_storage.store_ohlcv_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_ohlcv_data(self, data_manager, sample_ohlcv_data):
        """Test loading OHLCV data"""
        with patch.object(data_manager, 'storage') as mock_storage:
            mock_storage.load_ohlcv_data = AsyncMock(return_value=sample_ohlcv_data)

            result = await data_manager.load_ohlcv_data(
                symbol='BTCUSDT',
                timeframe='1h'
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_ohlcv_data)
            mock_storage.load_ohlcv_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_feature_data(self, data_manager):
        """Test storing feature data"""
        # Create sample feature data
        feature_data = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100)
        }, index=pd.date_range('2024-01-01', periods=100, freq='1H'))

        feature_metadata = {
            'feature_count': 3,
            'generation_time': datetime.now().isoformat(),
            'feature_names': ['feature_1', 'feature_2', 'feature_3']
        }

        with patch.object(data_manager, 'storage') as mock_storage:
            mock_storage.store_feature_data = AsyncMock(return_value=True)

            result = await data_manager.store_feature_data(
                symbol='BTCUSDT',
                timeframe='1h',
                version='test_v1',
                features_df=feature_data,
                feature_metadata=feature_metadata
            )

            assert result is True
            mock_storage.store_feature_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_available_symbols(self, data_manager):
        """Test getting available symbols"""
        expected_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

        with patch.object(data_manager, 'storage') as mock_storage:
            mock_storage.get_available_symbols = Mock(return_value=expected_symbols)

            result = await data_manager.get_available_symbols()

            assert result == expected_symbols
            mock_storage.get_available_symbols.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_data_summary(self, data_manager):
        """Test getting data summary"""
        expected_summary = {
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'timeframes': ['1h', '4h'],
            'total_records': 10000,
            'date_range': {
                'start': '2024-01-01',
                'end': '2024-12-31'
            }
        }

        with patch.object(data_manager, 'storage') as mock_storage:
            mock_storage.get_data_summary = AsyncMock(return_value=expected_summary)

            result = await data_manager.get_data_summary()

            assert result == expected_summary
            mock_storage.get_data_summary.assert_called_once()


class TestDataStorage:
    """Test suite for DataStorage class"""

    @pytest.fixture
    async def storage(self):
        """Create DataStorage instance for testing"""
        storage = DataStorage()
        await storage.initialize()
        yield storage
        # Cleanup
        storage.clear_cache()

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        dates = pd.date_range('2024-01-01', periods=50, freq='1H')
        return pd.DataFrame({
            'open': np.random.randn(50) + 45000,
            'high': np.random.randn(50) + 45100,
            'low': np.random.randn(50) + 44900,
            'close': np.random.randn(50) + 45000,
            'volume': np.random.randint(1000, 5000, 50)
        }, index=dates)


    def test_cache_functionality(self, storage):
        """Test cache operations"""
        # Test cache setting and getting
        test_key = "test_cache_key"
        test_data = {"test": "data"}

        storage.cache[test_key] = test_data
        storage.cache_ttl[test_key] = datetime.now() + timedelta(minutes=5)

        # Test cache validity check
        assert storage._is_cache_valid(test_key)

        # Test expired cache
        storage.cache_ttl[test_key] = datetime.now() - timedelta(minutes=5)
        assert not storage._is_cache_valid(test_key)


    def test_cache_clearing(self, storage):
        """Test cache clearing functionality"""
        # Add some test data to cache
        storage.cache['ohlcv_BTCUSDT_1h'] = pd.DataFrame()
        storage.cache['features_ETHUSDT_4h'] = pd.DataFrame()
        storage.cache['signals_test'] = []

        # Clear specific cache type
        storage.clear_cache('ohlcv')

        # Check that only ohlcv cache was cleared
        assert 'ohlcv_BTCUSDT_1h' not in storage.cache
        assert 'features_ETHUSDT_4h' in storage.cache
        assert 'signals_test' in storage.cache

        # Clear all cache
        storage.clear_cache()
        assert len(storage.cache) == 0


    def test_get_cache_stats(self, storage):
        """Test cache statistics"""
        # Add some test data
        storage.cache['test1'] = [1, 2, 3]
        storage.cache['test2'] = {'key': 'value'}
        storage.cache['ohlcv_test'] = pd.DataFrame()

        stats = storage.get_cache_stats()

        assert 'total_entries' in stats
        assert 'cache_types' in stats
        assert stats['total_entries'] == 3


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class"""

    @pytest.fixture
    def preprocessor(self):
        """Create DataPreprocessor instance"""
        return DataPreprocessor()

    @pytest.fixture
    def messy_ohlcv_data(self):
        """Generate messy OHLCV data for testing preprocessing"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)

        prices = 45000 + np.cumsum(np.random.randn(100) * 50)

        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices + np.random.randn(100) * 10,
            'volume': np.random.randint(1000, 5000, 100)
        }, index=dates)

        # Add some data quality issues
        # Missing values
        df.loc[df.index[10:15], 'volume'] = np.nan

        # Zero volume
        df.loc[df.index[20:25], 'volume'] = 0

        # OHLC inconsistencies (high < low)
        df.loc[df.index[30], 'high'] = df.loc[df.index[30], 'low'] - 100

        # Outliers
        df.loc[df.index[40], 'high'] = df.loc[df.index[40], 'close'] * 2

        # Duplicates
        df.loc[df.index[50]] = df.loc[df.index[49]]

        return df


    def test_preprocess_ohlcv_data(self, preprocessor, messy_ohlcv_data):
        """Test OHLCV data preprocessing"""
        original_length = len(messy_ohlcv_data)

        cleaned_df = preprocessor.preprocess_ohlcv_data(messy_ohlcv_data)

        # Should return a DataFrame
        assert isinstance(cleaned_df, pd.DataFrame)

        # Should have fewer or equal records (due to cleaning)
        assert len(cleaned_df) <= original_length

        # Should not have missing values in volume
        assert not cleaned_df['volume'].isnull().any()

        # Should not have zero volume (assuming remove_zero_volume=True)
        assert not (cleaned_df['volume'] == 0).any()

        # Should have valid OHLC relationships
        assert (cleaned_df['high'] >= cleaned_df['low']).all()
        assert (cleaned_df['high'] >= cleaned_df['open']).all()
        assert (cleaned_df['high'] >= cleaned_df['close']).all()
        assert (cleaned_df['low'] <= cleaned_df['open']).all()
        assert (cleaned_df['low'] <= cleaned_df['close']).all()


    def test_validate_ohlc_relationships(self, preprocessor):
        """Test OHLC relationship validation"""
        # Create data with OHLC issues
        df = pd.DataFrame({
            'open': [100, 200, 300],
            'high': [90, 250, 350],   # high < open for first row
            'low': [110, 180, 280],   # low > open for first row
            'close': [95, 220, 320],
            'volume': [1000, 2000, 3000]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1H'))

        fixed_df, fixes = preprocessor._validate_ohlc_relationships(df)

        # Should have made fixes
        assert fixes > 0

        # Should have valid relationships
        assert (fixed_df['high'] >= fixed_df['low']).all()
        assert (fixed_df['high'] >= fixed_df['open']).all()
        assert (fixed_df['high'] >= fixed_df['close']).all()
        assert (fixed_df['low'] <= fixed_df['open']).all()
        assert (fixed_df['low'] <= fixed_df['close']).all()


    def test_handle_missing_values(self, preprocessor):
        """Test missing value handling"""
        # Create data with missing values
        df = pd.DataFrame({
            'value1': [1, 2, np.nan, 4, 5],
            'value2': [10, np.nan, 30, np.nan, 50],
            'value3': [100, 200, 300, 400, 500]
        })

        cleaned_df, handled_count = preprocessor._handle_missing_values(df, 'forward_fill')

        # Should have handled missing values
        assert handled_count > 0

        # Should not have any missing values
        assert not cleaned_df.isnull().any().any()


    def test_outlier_detection_iqr(self, preprocessor):
        """Test IQR outlier detection"""
        # Create data with outliers
        data = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10])  # 100 is an outlier

        outliers = preprocessor._detect_outliers_iqr(data)

        # Should detect the outlier
        assert len(outliers) > 0
        assert data.loc[outliers].max() >= 100


    def test_outlier_detection_zscore(self, preprocessor):
        """Test Z-score outlier detection"""
        # Create data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        outlier_data = np.concatenate([normal_data, [10, -10]])  # Add extreme outliers
        data = pd.Series(outlier_data)

        outliers = preprocessor._detect_outliers_zscore(data, threshold=3.0)

        # Should detect outliers
        assert len(outliers) > 0


    def test_preprocess_features(self, preprocessor):
        """Test feature preprocessing"""
        # Create feature data with various issues
        np.random.seed(42)
        features_df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'constant_feature': [1.0] * 100,  # Constant feature
            'feature3': np.random.randn(100),
            'highly_correlated': np.random.randn(100)  # Will make it highly correlated
        })

        # Make feature3 and highly_correlated very similar
        features_df['highly_correlated'] = features_df['feature3'] * 0.99 + np.random.randn(100) * 0.01

        # Add some missing values and infinities
        features_df.loc[0:5, 'feature1'] = np.nan
        features_df.loc[10:15, 'feature2'] = np.inf

        config = {
            'handle_missing': True,
            'remove_constant_features': True,
            'handle_inf_values': True,
            'correlation_threshold': 0.95
        }

        processed_df = preprocessor.preprocess_features(features_df, config)

        # Should not have missing values
        assert not processed_df.isnull().any().any()

        # Should not have infinite values
        assert not np.isinf(processed_df.select_dtypes(include=[np.number])).any().any()

        # Should have removed constant feature
        assert 'constant_feature' not in processed_df.columns

        # Should have fewer features due to correlation removal
        assert len(processed_df.columns) <= len(features_df.columns)


    def test_validate_data_quality(self, preprocessor, messy_ohlcv_data):
        """Test data quality validation"""
        quality_report = preprocessor.validate_data_quality(messy_ohlcv_data)

        # Should return a quality report
        assert isinstance(quality_report, dict)
        assert 'quality_score' in quality_report
        assert 'missing_values' in quality_report
        assert 'data_types' in quality_report
        assert 'issues' in quality_report

        # Quality score should be between 0 and 100
        assert 0 <= quality_report['quality_score'] <= 100

        # Should detect issues in messy data
        assert len(quality_report['issues']) > 0


class TestExternalDataManager:
    """Test suite for ExternalDataManager class"""

    @pytest.fixture
    async def external_data_manager(self):
        """Create ExternalDataManager instance"""
        manager = ExternalDataManager()
        await manager.initialize()
        yield manager
        # Cleanup
        manager.clear_cache()

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test ExternalDataManager initialization"""
        manager = ExternalDataManager()
        result = await manager.initialize()

        # Should always return True or False
        assert isinstance(result, bool)

        # Should have initialized components
        assert hasattr(manager, 'config')
        assert hasattr(manager, 'cache')

    @pytest.mark.asyncio
    async def test_get_market_data_caching(self, external_data_manager):
        """Test market data caching"""
        with patch.object(external_data_manager, '_get_coingecko_market_data') as mock_coingecko:
            mock_data = {
                'symbol': 'BTCUSDT',
                'price_usd': 45000,
                'market_cap_usd': 900000000,
                'volume_24h_usd': 50000000,
                'change_24h': 2.5,
                'source': 'coingecko',
                'timestamp': datetime.now()
            }
            mock_coingecko.return_value = mock_data

            # First call should hit the API
            result1 = await external_data_manager.get_market_data('BTCUSDT')

            # Second call should use cache
            result2 = await external_data_manager.get_market_data('BTCUSDT')

            # Should have called the API only once
            mock_coingecko.assert_called_once()

            # Both results should be the same
            assert result1 == result2


    def test_symbol_to_coingecko_id(self, external_data_manager):
        """Test symbol to CoinGecko ID conversion"""
        # Test known mappings
        assert external_data_manager._symbol_to_coingecko_id('BTCUSDT') == 'bitcoin'
        assert external_data_manager._symbol_to_coingecko_id('ETHUSDT') == 'ethereum'

        # Test unknown symbol
        result = external_data_manager._symbol_to_coingecko_id('UNKNOWNUSDT')
        assert result == 'unknown'


    def test_crypto_name_mapping(self, external_data_manager):
        """Test cryptocurrency name mapping"""
        assert external_data_manager._get_crypto_name('BTCUSDT') == 'Bitcoin'
        assert external_data_manager._get_crypto_name('ETHUSDT') == 'Ethereum'
        assert external_data_manager._get_crypto_name('UNKNOWN') == 'UNKNOWN'


    def test_cache_validity(self, external_data_manager):
        """Test cache validity checking"""
        cache_key = 'test_cache'

        # Test non-existent cache
        assert not external_data_manager._is_cache_valid(cache_key)

        # Add cache entry
        external_data_manager.cache[cache_key] = {'test': 'data'}
        external_data_manager.cache_ttl[cache_key] = datetime.now() + timedelta(minutes=5)

        # Should be valid
        assert external_data_manager._is_cache_valid(cache_key, ttl_minutes=10)

        # Make it expired
        external_data_manager.cache_ttl[cache_key] = datetime.now() - timedelta(minutes=5)

        # Should be invalid
        assert not external_data_manager._is_cache_valid(cache_key)

# Integration Tests


class TestDataIntegration:
    """Integration tests for data components working together"""

    @pytest.fixture
    async def integrated_system(self):
        """Create integrated data system for testing"""
        manager = DataManager()
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.mark.asyncio
    async def test_full_data_pipeline(self, integrated_system):
        """Test complete data pipeline from fetch to storage"""
        # Create sample data
        sample_data = pd.DataFrame({
            'open': [45000, 45100, 45200],
            'high': [45050, 45150, 45250],
            'low': [44950, 45050, 45150],
            'close': [45100, 45200, 45300],
            'volume': [1000, 1500, 2000]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1H'))

        with patch.object(integrated_system, 'exchange_manager') as mock_exchange:
            mock_exchange.fetch_ohlcv = AsyncMock(return_value=sample_data)

            with patch.object(integrated_system, 'storage') as mock_storage:
                mock_storage.store_ohlcv_data = AsyncMock(return_value=True)

                # Test the full pipeline
                fetched_data = await integrated_system.fetch_ohlcv_data(
                    symbol='BTCUSDT',
                    timeframe='1h'
                )

                stored = await integrated_system.store_ohlcv_data(
                    symbol='BTCUSDT',
                    timeframe='1h',
                    data=fetched_data
                )

                assert not fetched_data.empty
                assert stored is True
                mock_exchange.fetch_ohlcv.assert_called_once()
                mock_storage.store_ohlcv_data.assert_called_once()

# Performance Tests


class TestDataPerformance:
    """Performance tests for data operations"""

    def test_large_data_preprocessing_performance(self):
        """Test preprocessing performance with large dataset"""
        # Create large dataset
        size = 10000
        dates = pd.date_range('2020-01-01', periods=size, freq='1H')

        large_df = pd.DataFrame({
            'open': np.random.randn(size) + 45000,
            'high': np.random.randn(size) + 45100,
            'low': np.random.randn(size) + 44900,
            'close': np.random.randn(size) + 45000,
            'volume': np.random.randint(1000, 5000, size)
        }, index=dates)

        preprocessor = DataPreprocessor()

        import time
        start_time = time.time()

        cleaned_df = preprocessor.preprocess_ohlcv_data(large_df)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 10.0  # 10 seconds max
        assert not cleaned_df.empty
        assert len(cleaned_df) <= size

# Fixtures for pytest
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
