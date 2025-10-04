"""
Features Package
Feature engineering and selection modules for quantitative trading
"""

from .feature_generator import FeatureEngineering, TechnicalIndicators, TimeFeatures

try:
    from .feature_selector import FeatureOptimizer
except ImportError:
    # Handle case where feature_selector might not be available
    FeatureOptimizer = None

__all__ = [
    'FeatureEngineering',
    'TechnicalIndicators', 
    'TimeFeatures',
    'FeatureOptimizer'
]

__version__ = '1.0.0'
