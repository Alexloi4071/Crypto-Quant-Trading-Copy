"""
Feature Engineering Module
Comprehensive technical indicator calculation and feature generation
Supports multi-timeframe analysis and customizable parameters
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import warnings

from config.settings import config
from src.utils.logger import setup_logger

warnings.filterwarnings('ignore')

# 嘗試導入 optuna，如果不可用則設置標誌
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = setup_logger(__name__)


class AdaptiveIndicatorParams:
    """為不同時框定義智能參數搜索範圍"""
    
    @staticmethod
    def get_indicator_ranges(timeframe: str) -> Dict[str, Dict[str, Tuple[int, int]]]:
        """
        根據時框返回各指標的合理參數範圍
        
        Returns:
            Dict[指標名稱, Dict[參數名, (最小值, 最大值)]]
        """
        
        # 基於時框特性的參數範圍映射
        ranges = {
            "1m": {
                "rsi_period": (5, 21),
                "ma_period": (3, 50),
                "macd_fast": (5, 15),
                "macd_slow": (15, 35),
                "macd_signal": (3, 12),
                "bb_period": (10, 30),
                "atr_period": (7, 21),
                "adx_period": (7, 21),
                "cci_period": (7, 30),
                "stoch_k": (5, 21),
                "stoch_d": (2, 10),
                "williams_r": (7, 21),
                "roc_period": (5, 20),
                "mfi_period": (7, 21),
                "volume_ma": (5, 30)
            },
            "5m": {
                "rsi_period": (7, 30),
                "ma_period": (5, 100),
                "macd_fast": (8, 18),
                "macd_slow": (20, 40),
                "macd_signal": (5, 15),
                "bb_period": (15, 40),
                "atr_period": (10, 30),
                "adx_period": (10, 30),
                "cci_period": (10, 40),
                "stoch_k": (8, 30),
                "stoch_d": (3, 12),
                "williams_r": (10, 30),
                "roc_period": (8, 30),
                "mfi_period": (10, 30),
                "volume_ma": (10, 50)
            },
            "15m": {
                "rsi_period": (10, 35),
                "ma_period": (10, 150),
                "macd_fast": (10, 20),
                "macd_slow": (22, 45),
                "macd_signal": (7, 18),
                "bb_period": (15, 50),
                "atr_period": (12, 35),
                "adx_period": (12, 35),
                "cci_period": (12, 50),
                "stoch_k": (10, 35),
                "stoch_d": (3, 15),
                "williams_r": (12, 35),
                "roc_period": (10, 35),
                "mfi_period": (12, 35),
                "volume_ma": (15, 75)
            },
            "1h": {
                "rsi_period": (14, 50),
                "ma_period": (20, 200),
                "macd_fast": (12, 25),
                "macd_slow": (26, 50),
                "macd_signal": (9, 20),
                "bb_period": (20, 60),
                "atr_period": (14, 50),
                "adx_period": (14, 50),
                "cci_period": (14, 60),
                "stoch_k": (14, 50),
                "stoch_d": (3, 20),
                "williams_r": (14, 50),
                "roc_period": (12, 50),
                "mfi_period": (14, 50),
                "volume_ma": (20, 100)
            },
            "4h": {
                "rsi_period": (21, 70),
                "ma_period": (30, 300),
                "macd_fast": (15, 30),
                "macd_slow": (30, 70),
                "macd_signal": (12, 25),
                "bb_period": (25, 80),
                "atr_period": (21, 70),
                "adx_period": (21, 70),
                "cci_period": (20, 80),
                "stoch_k": (21, 70),
                "stoch_d": (5, 25),
                "williams_r": (21, 70),
                "roc_period": (15, 70),
                "mfi_period": (21, 70),
                "volume_ma": (25, 150)
            },
            "1d": {
                "rsi_period": (30, 100),
                "ma_period": (50, 500),
                "macd_fast": (20, 40),
                "macd_slow": (40, 100),
                "macd_signal": (15, 30),
                "bb_period": (30, 120),
                "atr_period": (30, 100),
                "adx_period": (30, 100),
                "cci_period": (30, 120),
                "stoch_k": (30, 100),
                "stoch_d": (7, 30),
                "williams_r": (30, 100),
                "roc_period": (20, 100),
                "mfi_period": (30, 100),
                "volume_ma": (30, 200)
            },
            "1w": {
                "rsi_period": (50, 150),
                "ma_period": (100, 1000),
                "macd_fast": (30, 60),
                "macd_slow": (60, 150),
                "macd_signal": (20, 40),
                "bb_period": (50, 200),
                "atr_period": (50, 150),
                "adx_period": (50, 150),
                "cci_period": (50, 200),
                "stoch_k": (50, 150),
                "stoch_d": (10, 50),
                "williams_r": (50, 150),
                "roc_period": (30, 150),
                "mfi_period": (50, 150),
                "volume_ma": (50, 300)
            }
        }
        
        return ranges.get(timeframe, ranges["1h"])  # 默認使用1h參數


class OptunaTechnicalIndicators:
    """集成Optuna優化的技術指標計算引擎"""
    
    def __init__(self, timeframe: str = "1h"):
        self.timeframe = timeframe
        self.param_ranges = AdaptiveIndicatorParams.get_indicator_ranges(timeframe)
        self.optimized_params = {}  # 存儲優化后的參數
        
    def suggest_indicator_params(self, trial) -> Dict[str, int]:
        """為當前trial建議指標參數"""
        if not OPTUNA_AVAILABLE:
            # 如果沒有optuna，使用參數範圍的中值
            suggested_params = {}
            for param_name, (min_val, max_val) in self.param_ranges.items():
                suggested_params[param_name] = (min_val + max_val) // 2
            return suggested_params
        
        suggested_params = {}
        for param_name, (min_val, max_val) in self.param_ranges.items():
            suggested_params[param_name] = trial.suggest_int(param_name, min_val, max_val)
        
        return suggested_params
    
    def calculate_adaptive_indicators(self, df: pd.DataFrame, params: Dict[str, int]) -> pd.DataFrame:
        """使用優化參數計算所有技術指標"""
        result_df = df.copy()
        
        try:
            # RSI - 使用優化參數
            rsi_period = params.get('rsi_period', 14)
            result_df['RSI_optimized'] = talib.RSI(df['close'], timeperiod=rsi_period)
            
            # Moving Averages - 使用優化參數
            ma_period = params.get('ma_period', 20)
            result_df['MA_optimized'] = talib.SMA(df['close'], timeperiod=ma_period)
            result_df['EMA_optimized'] = talib.EMA(df['close'], timeperiod=ma_period)
            
            # MACD - 使用優化參數
            macd_fast = params.get('macd_fast', 12)
            macd_slow = params.get('macd_slow', 26)
            macd_signal = params.get('macd_signal', 9)
            macd, macdsignal, macdhist = talib.MACD(
                df['close'], 
                fastperiod=macd_fast,
                slowperiod=macd_slow,
                signalperiod=macd_signal
            )
            result_df['MACD_optimized'] = macd
            result_df['MACD_Signal_optimized'] = macdsignal
            result_df['MACD_Hist_optimized'] = macdhist
            
            # Bollinger Bands - 使用優化參數
            bb_period = params.get('bb_period', 20)
            upper, middle, lower = talib.BBANDS(df['close'], timeperiod=bb_period)
            result_df['BB_Upper_optimized'] = upper
            result_df['BB_Middle_optimized'] = middle
            result_df['BB_Lower_optimized'] = lower
            result_df['BB_Width_optimized'] = (upper - lower) / middle
            result_df['BB_Position_optimized'] = (df['close'] - lower) / (upper - lower)
            
            # ATR - 使用優化參數
            atr_period = params.get('atr_period', 14)
            result_df['ATR_optimized'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
            result_df['ATR_Norm_optimized'] = result_df['ATR_optimized'] / df['close']
            
            # ADX - 使用優化參數
            adx_period = params.get('adx_period', 14)
            result_df['ADX_optimized'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=adx_period)
            
            # CCI - 使用優化參數
            cci_period = params.get('cci_period', 14)
            result_df['CCI_optimized'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=cci_period)
            
            # Stochastic - 使用優化參數
            stoch_k = params.get('stoch_k', 14)
            stoch_d = params.get('stoch_d', 3)
            slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'],
                                     fastk_period=stoch_k, slowk_period=stoch_d, slowd_period=stoch_d)
            result_df['STOCH_K_optimized'] = slowk
            result_df['STOCH_D_optimized'] = slowd
            
            # Williams %R - 使用優化參數
            wr_period = params.get('williams_r', 14)
            result_df['WR_optimized'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=wr_period)
            
            # ROC - 使用優化參數
            roc_period = params.get('roc_period', 12)
            result_df['ROC_optimized'] = talib.ROC(df['close'], timeperiod=roc_period)
            
            # MFI - 使用優化參數
            mfi_period = params.get('mfi_period', 14)
            result_df['MFI_optimized'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=mfi_period)
            
            # Volume MA - 使用優化參數 - 🔧 修復數據洩漏
            vol_ma_period = params.get('volume_ma', 20)
            result_df['Vol_MA_optimized'] = df['volume'].rolling(window=vol_ma_period).mean().shift(1)
            result_df['Vol_Ratio_optimized'] = df['volume'] / result_df['Vol_MA_optimized']
            
            logger.debug(f"Calculated adaptive indicators with params: {params}")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating adaptive indicators: {e}")
            return result_df
    
    def set_optimized_params(self, params: Dict[str, int]):
        """設置經Optuna優化後的最佳參數"""
        self.optimized_params = params
        logger.info(f"Set optimized parameters for {self.timeframe}: {params}")


class TechnicalIndicators:
    """Technical indicator calculation engine with timeframe adaptation"""

    def __init__(self, timeframe_multiplier: int = 60):
        self.indicators_config = config.indicators
        self.timeframe_multiplier = timeframe_multiplier
        self.scale_factor = max(1, timeframe_multiplier // 60)  # 基於1h的縮放因子

    def calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-based indicators"""
        result_df = df.copy()

        try:
            # Moving Averages - fix config key
            trend_config = self.indicators_config.get('trend_indicators', {})
            ma_periods = trend_config.get('moving_averages', {}).get('sma', {}).get('periods', [5, 10, 20, 50, 200])
            for period in ma_periods:
                result_df[f'MA_{period}'] = talib.SMA(df['close'], timeperiod=period)

            # Exponential Moving Averages
            ema_periods = trend_config.get('moving_averages', {}).get('ema', {}).get('periods', [9, 12, 21, 26, 50])
            for period in ema_periods:
                result_df[f'EMA_{period}'] = talib.EMA(df['close'], timeperiod=period)

            # MACD
            macd_config = trend_config.get('trend_following', {}).get('macd', {})
            macd, macdsignal, macdhist = talib.MACD(
                df['close'],
                fastperiod=macd_config.get('fast_period', 12),
                slowperiod=macd_config.get('slow_period', 26),
                signalperiod=macd_config.get('signal_period', 9)
            )
            result_df['MACD'] = macd
            result_df['MACD_Signal'] = macdsignal
            result_df['MACD_Hist'] = macdhist

            # ADX (Average Directional Movement Index)
            adx_periods = trend_config.get('trend_following', {}).get('adx', {}).get('periods', [14, 21])
            for period in adx_periods:
                result_df[f'ADX_{period}'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=period)

            # Parabolic SAR
            sar_config = trend_config.get('trend_following', {}).get('psar', {})
            result_df['SAR'] = talib.SAR(df['high'], df['low'],
                                       acceleration=sar_config.get('acceleration', 0.02),
                                       maximum=sar_config.get('maximum', 0.2))

            # Donchian Channel (simplified)
            for period in [20, 55]:
                result_df[f'DC_High_{period}'] = df['high'].rolling(window=period).max()
                result_df[f'DC_Low_{period}'] = df['low'].rolling(window=period).min()
                result_df[f'DC_Mid_{period}'] = (result_df[f'DC_High_{period}'] + result_df[f'DC_Low_{period}']) / 2

            logger.debug("Calculated trend indicators")
            return result_df

        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
            return result_df


    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based indicators"""
        result_df = df.copy()

        try:
            # Get momentum config
            momentum_config = self.indicators_config.get('momentum_indicators', {})

            # RSI (Relative Strength Index) - 時間框架自適應
            rsi_base_periods = momentum_config.get('oscillators', {}).get('rsi', {}).get('periods', [14, 21])
            for base_period in rsi_base_periods:
                adapted_period = max(2, base_period // self.scale_factor)  # 確保最小為2
                result_df[f'RSI_{base_period}'] = talib.RSI(df['close'], timeperiod=adapted_period)

            # Stochastic RSI - 時間框架自適應
            stochrsi_base_periods = momentum_config.get('oscillators', {}).get('stoch_rsi', {}).get('periods', [14])
            for base_period in stochrsi_base_periods:
                adapted_period = max(2, base_period // self.scale_factor)
                adapted_fastk = max(2, 5 // self.scale_factor)
                adapted_fastd = max(2, 3 // self.scale_factor)
                fastk, fastd = talib.STOCHRSI(df['close'], timeperiod=adapted_period,
                                            fastk_period=adapted_fastk, fastd_period=adapted_fastd)
                result_df[f'StochRSI_K_{base_period}'] = fastk
                result_df[f'StochRSI_D_{base_period}'] = fastd

            # KDJ Indicator - 時間框架自適應
            adapted_fastk_period = max(2, 9 // self.scale_factor)
            adapted_slowk_period = max(2, 3 // self.scale_factor)
            adapted_slowd_period = max(2, 3 // self.scale_factor)
            slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'],
                                     fastk_period=adapted_fastk_period, 
                                     slowk_period=adapted_slowk_period, 
                                     slowd_period=adapted_slowd_period)
            result_df['KDJ_K'] = slowk
            result_df['KDJ_D'] = slowd
            result_df['KDJ_J'] = 3 * slowk - 2 * slowd

            # CCI (Commodity Channel Index) - 時間框架自適應
            cci_base_periods = momentum_config.get('oscillators', {}).get('cci', {}).get('periods', [14, 20])
            for base_period in cci_base_periods:
                adapted_period = max(2, base_period // self.scale_factor)
                result_df[f'CCI_{base_period}'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=adapted_period)

            # Williams %R
            wr_periods = momentum_config.get('oscillators', {}).get('williams_r', {}).get('periods', [14])
            for period in wr_periods:
                result_df[f'WR_{period}'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=period)

            # Rate of Change
            roc_periods = momentum_config.get('momentum_oscillators', {}).get('roc', {}).get('periods', [12, 25])
            for period in roc_periods:
                result_df[f'ROC_{period}'] = talib.ROC(df['close'], timeperiod=period)

            # Momentum
            mom_periods = momentum_config.get('momentum_oscillators', {}).get('momentum', {}).get('periods', [10, 14])
            for period in mom_periods:
                result_df[f'MOM_{period}'] = talib.MOM(df['close'], timeperiod=period)

            logger.debug("Calculated momentum indicators")
            return result_df

        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return result_df


    def calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based indicators"""
        result_df = df.copy()

        try:
            # Get volatility config
            volatility_config = self.indicators_config.get('volatility_indicators', {})

            # Bollinger Bands
            boll_config = volatility_config.get('bollinger_bands', {})
            bb_periods = boll_config.get('periods', [20])
            bb_std_devs = boll_config.get('std_dev', [2])
            for period in bb_periods:
                for std_dev in bb_std_devs:
                    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
                    result_df[f'BB_Upper_{period}_{std_dev}'] = upper
                    result_df[f'BB_Middle_{period}_{std_dev}'] = middle
                    result_df[f'BB_Lower_{period}_{std_dev}'] = lower
                    result_df[f'BB_Width_{period}_{std_dev}'] = (upper - lower) / middle
                    result_df[f'BB_Position_{period}_{std_dev}'] = (df['close'] - lower) / (upper - lower)

            # Average True Range
            atr_periods = volatility_config.get('atr', {}).get('periods', [14])
            for period in atr_periods:
                result_df[f'ATR_{period}'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
                # Normalized ATR
                result_df[f'ATR_Norm_{period}'] = result_df[f'ATR_{period}'] / df['close']

            # Keltner Channel
            keltner_config = volatility_config.get('keltner_channel', {})
            kc_periods = keltner_config.get('periods', [20])
            kc_multipliers = keltner_config.get('multiplier', [2])
            for period in kc_periods:
                for multiplier in kc_multipliers:
                    ema = talib.EMA(df['close'], timeperiod=period)
                    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
                    result_df[f'KC_Upper_{period}_{multiplier}'] = ema + (multiplier * atr)
                    result_df[f'KC_Lower_{period}_{multiplier}'] = ema - (multiplier * atr)
                    result_df[f'KC_Position_{period}_{multiplier}'] = (df['close'] - result_df[f'KC_Lower_{period}_{multiplier}']) / \
                                                                    (result_df[f'KC_Upper_{period}_{multiplier}'] - result_df[f'KC_Lower_{period}_{multiplier}'])

            logger.debug("Calculated volatility indicators")
            return result_df

        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
            return result_df


    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        result_df = df.copy()

        try:
            # Get volume config
            volume_config = self.indicators_config.get('volume_indicators', {})

            # On-Balance Volume
            result_df['OBV'] = talib.OBV(df['close'], df['volume'])

            # Price-Volume Trend
            pv_change = df['close'].pct_change()
            result_df['PVT'] = (pv_change * df['volume']).cumsum()

            # Money Flow Index
            mfi_periods = volume_config.get('flow_indicators', {}).get('mfi', {}).get('periods', [14])
            for period in mfi_periods:
                result_df[f'MFI_{period}'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=period)

            # Volume Weighted Average Price
            vwap_periods = volume_config.get('price_volume', {}).get('vwap', {}).get('periods', [20])
            for period in vwap_periods:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                # 🔧 修復數據洩漏：VWAP計算使用歷史數據
                vwap_num = (typical_price * df['volume']).rolling(window=period).sum().shift(1)
                vwap_den = df['volume'].rolling(window=period).sum().shift(1)
                result_df[f'VWAP_{period}'] = vwap_num / vwap_den

            # Volume Moving Averages - 🔧 修復數據洩漏
            vol_ma_periods = volume_config.get('volume_patterns', {}).get('volume_ma', {}).get('periods', [20, 50])
            for period in vol_ma_periods:
                result_df[f'Vol_MA_{period}'] = df['volume'].rolling(window=period).mean().shift(1)
                result_df[f'Vol_Ratio_{period}'] = df['volume'] / result_df[f'Vol_MA_{period}']

            logger.debug("Calculated volume indicators")
            return result_df

        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return result_df


class TimeFeatures:
    """
    Comprehensive time-based feature engineering for quantitative trading
    Enhanced with advanced calendar effects, market sessions, and volatility features
    """

    def __init__(self, timezone: str = 'UTC'):
        self.timezone = timezone
        logger.info("Enhanced TimeFeatures module initialized")

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive time-based features"""
        try:
            logger.debug("Creating comprehensive time features")

            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning("Index is not DatetimeIndex, attempting conversion")
                df.index = pd.to_datetime(df.index)

            # Start with basic time features
            result_df = self._create_basic_time_features(df)
            
            # Add cyclical features
            result_df = self._create_cyclical_features(result_df)
            
            # Add market session features
            result_df = self._create_market_session_features(result_df)
            
            # Add calendar features
            result_df = self._create_calendar_features(result_df)
            
            # Add time statistics features
            result_df = self._create_time_statistics_features(result_df)

            logger.debug("Comprehensive time features created successfully")
            return result_df

        except Exception as e:
            logger.error(f"Error creating time features: {e}")
            return df

    def _create_basic_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic time component features"""
        result_df = df.copy()

        # Basic time components
        result_df['year'] = df.index.year
        result_df['month'] = df.index.month
        result_df['day'] = df.index.day
        result_df['hour'] = df.index.hour
        result_df['minute'] = df.index.minute
        result_df['weekday'] = df.index.weekday  # 0=Monday, 6=Sunday
        result_df['day_of_year'] = df.index.dayofyear
        result_df['week_of_year'] = df.index.isocalendar().week
        result_df['quarter'] = df.index.quarter

        # Day of week indicators
        for i, day in enumerate(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
            result_df[f'is_{day}'] = (result_df['weekday'] == i).astype(int)

        # Weekend/weekday indicators
        result_df['is_weekday'] = (result_df['weekday'] < 5).astype(int)
        result_df['is_weekend'] = (result_df['weekday'] >= 5).astype(int)

        # Period start/end indicators
        result_df['is_month_start'] = df.index.is_month_start.astype(int)
        result_df['is_month_end'] = df.index.is_month_end.astype(int)
        result_df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        result_df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        result_df['is_year_start'] = df.index.is_year_start.astype(int)
        result_df['is_year_end'] = df.index.is_year_end.astype(int)

        return result_df

    def _create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical time features using sine/cosine encoding"""
        # Hour cycle (24 hours)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Minute cycle (60 minutes)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

        # Weekday cycle (7 days)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

        # Month cycle (12 months)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Day cycle (31 days max)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

        # Day of year cycle (365 days)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        # Quarter cycle (4 quarters)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

        return df

    def _create_market_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market trading session features"""
        # Global market sessions (UTC time)
        # US market (EST/EDT, UTC-5/-4)
        us_open, us_close = 13, 20  # Adjusted for daylight saving
        df['us_market_hours'] = ((df['hour'] >= us_open) &
                                 (df['hour'] < us_close) &
                                 (df['weekday'] < 5)).astype(int)

        # European market (CET/CEST, UTC+1/+2)
        eu_open, eu_close = 8, 16
        df['eu_market_hours'] = ((df['hour'] >= eu_open) &
                                 (df['hour'] < eu_close) &
                                 (df['weekday'] < 5)).astype(int)

        # Asian market (JST, UTC+9)
        asia_open, asia_close = 0, 7
        df['asia_market_hours'] = ((df['hour'] >= asia_open) &
                                   (df['hour'] < asia_close) &
                                   (df['weekday'] < 5)).astype(int)

        # Market overlap periods
        df['us_eu_overlap'] = (df['us_market_hours'] & df['eu_market_hours']).astype(int)
        df['asia_eu_overlap'] = (df['asia_market_hours'] & df['eu_market_hours']).astype(int)

        # Major trading hours
        df['major_trading_hours'] = (df['us_market_hours'] |
                                     df['eu_market_hours'] |
                                     df['asia_market_hours']).astype(int)

        # Daily time periods
        df['early_morning'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
        df['morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['evening'] = ((df['hour'] >= 18) & (df['hour'] < 24)).astype(int)

        # Market microstructure features
        df['is_hour_open'] = (df['minute'] <= 5).astype(int)
        df['is_hour_close'] = (df['minute'] >= 55).astype(int)
        df['is_news_time'] = (df['minute'] == 0).astype(int)

        return df

    def _create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create calendar and holiday effect features"""
        # Month effects
        df['is_first_week_of_month'] = (df.index.day <= 7).astype(int)
        df['is_last_week_of_month'] = (df.index.day >= 22).astype(int)

        # Quarter effects
        df['is_end_of_quarter'] = (
            ((df['month'] == 3) | (df['month'] == 6) |
             (df['month'] == 9) | (df['month'] == 12)) &
            (df.index.day >= 25)
        ).astype(int)

        # Seasonal effects
        df['is_january'] = (df['month'] == 1).astype(int)
        df['is_december'] = (df['month'] == 12).astype(int)

        # Market anomalies
        df['is_jan_effect'] = (df['month'] == 1).astype(int)  # January effect
        df['is_sell_in_may'] = (df['month'] == 5).astype(int)  # Sell in May
        df['is_halloween_effect'] = ((df['month'] >= 11) | (df['month'] <= 4)).astype(int)

        # Holiday proximity (simplified)
        def is_near_holiday(date):
            month, day = date.month, date.day
            # New Year
            if (month == 1 and day <= 3) or (month == 12 and day >= 29):
                return True
            # Thanksgiving/Christmas period
            if month == 11 and 20 <= day <= 30:
                return True
            if month == 12 and 20 <= day <= 31:
                return True
            return False

        df['near_holiday'] = df.index.map(is_near_holiday).astype(int)

        return df

    def _create_time_statistics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based statistical features"""
        # Time intervals
        if len(df) > 1:
            time_diff = df.index.to_series().diff()
            df['time_since_last'] = time_diff.dt.total_seconds() / 60  # minutes
            df['time_since_last'].fillna(0, inplace=True)
        else:
            df['time_since_last'] = 0

        # Progress through periods
        df['hour_progress'] = df['minute'] / 60.0
        df['day_progress'] = (df['hour'] * 60 + df['minute']) / (24 * 60)
        df['week_progress'] = (df['weekday'] * 24 * 60 + df['hour'] * 60 + df['minute']) / (7 * 24 * 60)
        
        # Month progress
        days_in_month = df.index.days_in_month
        df['month_progress'] = (df['day'] - 1) / (days_in_month - 1)
        df['year_progress'] = df['day_of_year'] / 365.0

        # Distance features
        df['days_since_year_start'] = df['day_of_year']
        df['days_until_year_end'] = 365 - df['day_of_year']
        df['days_since_month_start'] = df['day'] - 1
        df['days_until_month_end'] = days_in_month - df['day']

        return df

    def create_advanced_time_features(self, df: pd.DataFrame, 
                                     include_seasonal: bool = True,
                                     include_volatility: bool = True) -> pd.DataFrame:
        """Create advanced time features"""
        result_df = self.create_time_features(df)

        if include_seasonal:
            result_df = self._create_seasonal_features(result_df)

        if include_volatility and 'close' in df.columns:
            result_df = self._create_volatility_time_features(result_df)

        return result_df

    def _create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal features"""
        # Season definitions
        df['is_spring'] = ((df['month'] >= 3) & (df['month'] <= 5)).astype(int)
        df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        df['is_autumn'] = ((df['month'] >= 9) & (df['month'] <= 11)).astype(int)
        df['is_winter'] = ((df['month'] == 12) | (df['month'] <= 2)).astype(int)

        # Half-year periods
        df['is_first_half'] = (df['month'] <= 6).astype(int)
        df['is_second_half'] = (df['month'] > 6).astype(int)

        # Quarter position
        quarter_month = ((df['month'] - 1) % 3) + 1
        df['quarter_month'] = quarter_month
        df['is_quarter_start_month'] = (quarter_month == 1).astype(int)
        df['is_quarter_middle_month'] = (quarter_month == 2).astype(int)
        df['is_quarter_end_month'] = (quarter_month == 3).astype(int)

        return df

    def _create_volatility_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based volatility features"""
        if 'close' not in df.columns:
            return df

        returns = df['close'].pct_change()

        # Hourly volatility expectations
        hourly_vol = returns.groupby(df['hour']).std()
        df['hourly_expected_vol'] = df['hour'].map(hourly_vol)

        # Weekday volatility expectations
        weekday_vol = returns.groupby(df['weekday']).std()
        df['weekday_expected_vol'] = df['weekday'].map(weekday_vol)

        # High volatility period indicators
        df['is_high_vol_hour'] = ((df['hour'] >= 13) & (df['hour'] <= 16)).astype(int)
        df['is_high_vol_day'] = ((df['weekday'] >= 0) & (df['weekday'] <= 4)).astype(int)

        return df

    @staticmethod
    def get_time_feature_groups():
        """Get time feature groups for selection"""
        return {
            'basic_time': ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'quarter'],
            'cyclical': ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos'],
            'calendar': ['is_weekday', 'is_weekend', 'is_month_start', 'is_month_end'],
            'market_sessions': ['us_market_hours', 'eu_market_hours', 'asia_market_hours', 'major_trading_hours'],
            'seasonal': ['is_spring', 'is_summer', 'is_autumn', 'is_winter'],
            'statistical': ['hour_progress', 'day_progress', 'month_progress', 'year_progress']
        }


class AdvancedFeatures:
    """Advanced feature engineering techniques"""

    @staticmethod
    def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based derived features"""
        result_df = df.copy()

        try:
            # Price ratios and relationships
            result_df['hl_ratio'] = df['high'] / df['low']
            result_df['oc_ratio'] = df['open'] / df['close']
            result_df['ho_ratio'] = df['high'] / df['open']
            result_df['lo_ratio'] = df['low'] / df['open']

            # Intraday price ranges
            result_df['daily_range'] = (df['high'] - df['low']) / df['open']
            result_df['body_size'] = abs(df['close'] - df['open']) / df['open']
            result_df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
            result_df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']

            # Price position within the day
            result_df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

            # Gap analysis
            result_df['gap_up'] = np.maximum(0, df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            result_df['gap_down'] = np.maximum(0, df['close'].shift(1) - df['open']) / df['close'].shift(1)

            # Returns at different periods
            for period in [1, 3, 5, 10, 20, 50]:
                result_df[f'return_{period}'] = df['close'].pct_change(period)
                result_df[f'return_{period}_abs'] = abs(result_df[f'return_{period}'])

                # Rolling statistics - 🔧 修復數據洩漏：加入shift(1)確保只使用歷史數據
                result_df[f'return_mean_{period}'] = result_df[f'return_{period}'].rolling(window=period).mean().shift(1)
                result_df[f'return_std_{period}'] = result_df[f'return_{period}'].rolling(window=period).std().shift(1)
                result_df[f'return_skew_{period}'] = result_df[f'return_{period}'].rolling(window=period).skew().shift(1)
                result_df[f'return_kurt_{period}'] = result_df[f'return_{period}'].rolling(window=period).kurt().shift(1)

            logger.debug("Created price-based features")
            return result_df

        except Exception as e:
            logger.error(f"Error creating price features: {e}")
            return result_df

    @staticmethod
    def create_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based derived features"""
        result_df = df.copy()

        try:
            # Volume ratios and statistics - 🔧 修復數據洩漏：加入shift(1)
            for period in [5, 10, 20, 50]:
                vol_ma = df['volume'].rolling(window=period).mean().shift(1)  # 使用歷史數據
                result_df[f'volume_ratio_{period}'] = df['volume'] / vol_ma
                result_df[f'volume_std_{period}'] = df['volume'].rolling(window=period).std().shift(1)
                result_df[f'volume_zscore_{period}'] = (df['volume'] - vol_ma) / result_df[f'volume_std_{period}']

            # Volume-price relationships - 🔧 修復數據洩漏
            result_df['vp_ratio'] = df['volume'] / df['close']
            result_df['volume_return_corr'] = df['volume'].rolling(window=20).corr(df['close'].pct_change()).shift(1)

            # Volume momentum - 🔧 修復數據洩漏：使用歷史數據
            vol_ma_5 = df['volume'].rolling(window=5).mean().shift(1)
            vol_ma_20 = df['volume'].rolling(window=20).mean().shift(1)
            vol_ma_50 = df['volume'].rolling(window=50).mean().shift(1)
            result_df['volume_momentum_5'] = vol_ma_5 / vol_ma_20
            result_df['volume_momentum_20'] = vol_ma_20 / vol_ma_50

            # Accumulation/Distribution features
            result_df['ad_line'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
            result_df['ad_oscillator'] = result_df['ad_line'].pct_change()

            logger.debug("Created volume-based features")
            return result_df

        except Exception as e:
            logger.error(f"Error creating volume features: {e}")
            return result_df


class FeatureSelector:
    """Feature selection and dimensionality reduction"""

    def __init__(self, correlation_threshold: float = 0.95, max_features: int = 30):
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features
        self.selected_features = None
        self.feature_importance = None

    def remove_correlated_features(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Remove highly correlated features"""
        try:
            # Calculate correlation matrix
            corr_matrix = X.corr().abs()

            # Find pairs of highly correlated features
            upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))

            # Find features to drop
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.correlation_threshold)]

            logger.info(f"Removing {len(to_drop)} highly correlated features")
            return X.drop(columns=to_drop)

        except Exception as e:
            logger.error(f"Error removing correlated features: {e}")
            return X


    def select_k_best_features(self, X: pd.DataFrame, y: pd.Series, k: int = None) -> pd.DataFrame:
        """Select k best features using statistical tests"""
        try:
            if k is None:
                k = min(self.max_features, X.shape[1])

            # Use mutual information for feature selection
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_selected = selector.fit_transform(X, y)

            # Get selected feature names
            selected_mask = selector.get_support()
            self.selected_features = X.columns[selected_mask].tolist()

            # Get feature scores
            feature_scores = selector.scores_
            self.feature_importance = dict(zip(X.columns, feature_scores))

            logger.info(f"Selected {len(self.selected_features)} best features")
            return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)

        except Exception as e:
            logger.error(f"Error selecting k best features: {e}")
            return X


    def apply_pca(self, X: pd.DataFrame, variance_threshold: float = 0.95) -> Tuple[pd.DataFrame, PCA]:
        """Apply PCA for dimensionality reduction"""
        try:
            # Standardize features before PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Apply PCA
            pca = PCA(n_components=variance_threshold)
            X_pca = pca.fit_transform(X_scaled)

            # Create DataFrame with PCA components
            pca_columns = [f'PCA_{i+1}' for i in range(X_pca.shape[1])]
            X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)

            logger.info(f"PCA reduced features from {X.shape[1]} to {X_pca.shape[1]} components")
            logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

            return X_pca_df, pca

        except Exception as e:
            logger.error(f"Error applying PCA: {e}")
            return X, None


class AdaptiveFeatureEngineering:
    """支持Optuna自適應參數優化的特徵工程"""
    
    def __init__(self, timeframe: str = "1h"):
        self.timeframe = timeframe
        self.optuna_indicators = OptunaTechnicalIndicators(timeframe)
        self.time_features = TimeFeatures()
        self.advanced_features = AdvancedFeatures()
        self.feature_selector = FeatureSelector(
            correlation_threshold=config.optimization.correlation_threshold,
            max_features=config.optimization.max_features
        )
        
        logger.info(f"AdaptiveFeatureEngineering initialized for {timeframe}")
        
    def generate_adaptive_features(self, df: pd.DataFrame, trial=None, 
                                 optimized_params: Dict[str, int] = None) -> pd.DataFrame:
        """
        生成自適應特徵集
        
        Args:
            df: OHLCV數據
            trial: Optuna trial對象（用於參數建議）
            optimized_params: 已優化的參數（如果有）
        """
        try:
            logger.info(f"Generating adaptive features for {self.timeframe}")
            
            # 獲取指標參數
            if optimized_params:
                params = optimized_params
            elif trial:
                params = self.optuna_indicators.suggest_indicator_params(trial)
            else:
                # 使用默認參數範圍的中值
                ranges = self.optuna_indicators.param_ranges
                params = {name: (min_val + max_val) // 2 for name, (min_val, max_val) in ranges.items()}
            
            # 開始特徵生成
            features_df = df.copy()
            
            # 計算自適應技術指標
            features_df = self.optuna_indicators.calculate_adaptive_indicators(features_df, params)
            
            # 添加原有的時間特徵和價格特徵（這些不需要優化）
            features_df = self.time_features.create_time_features(features_df)
            features_df = self.advanced_features.create_price_features(features_df)
            features_df = self.advanced_features.create_volume_features(features_df)
            
            # 數據清理
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(method='ffill')  # 🔧 只使用前向填充，避免數據洩漏
            features_df = features_df.fillna(0)
            
            logger.info(f"Generated {features_df.shape[1]} adaptive features using params: {params}")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error generating adaptive features: {e}")
            return df


class FeatureEngineering:
    """Main feature engineering orchestrator with timeframe adaptation"""

    def __init__(self, timeframe: str = "1h"):
        self.timeframe = timeframe
        self.timeframe_multiplier = self._get_timeframe_multiplier(timeframe)
        
        # 同時支持傳統方式和自適應方式
        self.technical_indicators = TechnicalIndicators(self.timeframe_multiplier)
        self.adaptive_fe = AdaptiveFeatureEngineering(timeframe)
        self.time_features = TimeFeatures()
        self.advanced_features = AdvancedFeatures()
        self.feature_selector = FeatureSelector(
            correlation_threshold=config.optimization.correlation_threshold,
            max_features=config.optimization.max_features
        )
        
        logger.info(f"FeatureEngineering initialized for {timeframe} (multiplier: {self.timeframe_multiplier})")
    
    def _get_timeframe_multiplier(self, timeframe: str) -> int:
        """獲取時間框架乘數用於調整技術指標參數"""
        multipliers = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "8h": 480,
            "12h": 720,
            "1d": 1440,
            "1w": 10080
        }
        return multipliers.get(timeframe, 60)  # 默認為1h

    def generate_all_features(self, df: pd.DataFrame, external_data: Dict = None) -> pd.DataFrame:
        """Generate comprehensive feature set"""
        try:
            logger.info("Starting comprehensive feature generation")

            # Start with original OHLCV data
            features_df = df.copy()

            # Technical indicators
            features_df = self.technical_indicators.calculate_trend_indicators(features_df)
            features_df = self.technical_indicators.calculate_momentum_indicators(features_df)
            features_df = self.technical_indicators.calculate_volatility_indicators(features_df)
            features_df = self.technical_indicators.calculate_volume_indicators(features_df)

            # Time-based features
            features_df = self.time_features.create_time_features(features_df)

            # Advanced price and volume features
            features_df = self.advanced_features.create_price_features(features_df)
            features_df = self.advanced_features.create_volume_features(features_df)

            # Add external data if provided
            if external_data:
                features_df = self._add_external_features(features_df, external_data)

            # Remove infinite values and replace with NaN
            features_df = features_df.replace([np.inf, -np.inf], np.nan)

            # 🔧 修復數據洩漏：只使用前向填充，禁止後向填充
            features_df = features_df.fillna(method='ffill')  # 只使用歷史數據填充

            # Only drop rows where ALL values are NaN (more conservative)
            initial_rows = len(features_df)
            features_df = features_df.dropna(how='all')  # Only drop if ALL values are NaN

            # For remaining NaN values, fill with 0 (for technical indicators)
            features_df = features_df.fillna(0)
            final_rows = len(features_df)

            if initial_rows != final_rows:
                logger.warning(f"Dropped {initial_rows - final_rows} rows due to all-NaN values")

            logger.info(f"Generated {features_df.shape[1]} features for {features_df.shape[0]} samples")
            return features_df

        except Exception as e:
            logger.error(f"Error in feature generation: {e}")
            return df

    def generate_adaptive_features(self, df: pd.DataFrame, trial=None, 
                                 optimized_params: Dict[str, int] = None) -> pd.DataFrame:
        """
        使用自適應特徵工程生成特徵
        
        Args:
            df: OHLCV數據
            trial: Optuna trial對象（用於參數建議）
            optimized_params: 已優化的參數（如果有）
        """
        return self.adaptive_fe.generate_adaptive_features(df, trial, optimized_params)


    def _add_external_features(self, df: pd.DataFrame, external_data: Dict) -> pd.DataFrame:
        """Add external data features"""
        try:
            # Add external features (on-chain, sentiment, etc.)
            for key, value in external_data.items():
                if isinstance(value, (int, float)):
                    df[f'external_{key}'] = value
                elif isinstance(value, pd.Series):
                    df[f'external_{key}'] = value

            return df

        except Exception as e:
            logger.error(f"Error adding external features: {e}")
            return df


    def prepare_features_for_training(self, df: pd.DataFrame, target: pd.Series = None,
                                    feature_selection: bool = True) -> Tuple[pd.DataFrame, List[str], Dict]:
        """Prepare features for model training"""
        try:
            # Exclude non-feature columns
            exclude_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            X = df[feature_cols]

            # Feature selection pipeline
            if feature_selection and target is not None:
                # Remove correlated features
                X = self.feature_selector.remove_correlated_features(X, target)

                # Select best features
                X = self.feature_selector.select_k_best_features(X, target)

            # Get final feature list
            final_features = X.columns.tolist()

            # Get feature importance scores
            feature_importance = self.feature_selector.feature_importance or {}

            logger.info(f"Prepared {len(final_features)} features for training")
            return X, final_features, feature_importance

        except Exception as e:
            logger.error(f"Error preparing features for training: {e}")
            return df, [], {}


    def save_feature_config(self, symbol: str, timeframe: str, version: str,
                          feature_list: List[str], feature_importance: Dict) -> str:
        """Save feature configuration to file"""
        try:
            config_path = config.get_processed_path(symbol, timeframe, version)
            config_path.mkdir(parents=True, exist_ok=True)

            # Feature configuration
            feature_config = {
                'symbol': symbol,
                'timeframe': timeframe,
                'version': version,
                'feature_count': len(feature_list),
                'selected_features': feature_list,
                'feature_importance': feature_importance,
                'generation_date': pd.Timestamp.now().isoformat(),
                'correlation_threshold': self.feature_selector.correlation_threshold,
                'max_features': self.feature_selector.max_features
            }

            # Save to YAML file
            import yaml
            config_file = config_path / 'feature_config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(feature_config, f, default_flow_style=False)

            # Save feature importance as JSON
            import json
            importance_file = config_path / 'feature_importance.json'
            with open(importance_file, 'w') as f:
                json.dump(feature_importance, f, indent=2)

            logger.info(f"Saved feature configuration to {config_path}")
            return str(config_path)

        except Exception as e:
            logger.error(f"Error saving feature config: {e}")
            return ""

# Usage example functions


def generate_features_for_symbol(symbol: str, timeframe: str, df: pd.DataFrame,
                                external_data: Dict = None) -> pd.DataFrame:
    """Generate features for a specific symbol/timeframe"""
    fe = FeatureEngineering()
    return fe.generate_all_features(df, external_data)


def prepare_training_data(symbol: str, timeframe: str, df: pd.DataFrame,
                        target: pd.Series) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Prepare data for model training with feature selection"""
    fe = FeatureEngineering()
    return fe.prepare_features_for_training(df, target, feature_selection=True)
