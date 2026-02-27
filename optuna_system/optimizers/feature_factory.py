# -*- coding: utf-8 -*-
"""
P0 Fix: Clean Crypto Feature Factory
=====================================
Replaces the bloated indicator soup in optuna_feature.py.

Design principles:
1. Only factors with proven crypto alpha (or solid academic backing)
2. No fake microstructure from OHLCV (no spread estimation, no OB slope)
3. No Elliott Wave, Gann, Harmonic Patterns
4. Enable real crypto-specific factors: funding rate, OI, long/short ratio
5. Single-shift rule: higher timeframe features shift(1) once, that's it
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# SECTION 1: Core Price/Momentum Features (HIGH alpha, LOW noise)
# ============================================================

def calc_returns(close: pd.Series, windows: List[int] = [1, 3, 5, 10, 20, 60]) -> pd.DataFrame:
    """Log returns and momentum at multiple horizons."""
    out = {}
    log_close = np.log(close + 1e-10)
    for w in windows:
        out[f'log_ret_{w}'] = log_close.diff(w)
        out[f'mom_{w}'] = close.pct_change(w)
    return pd.DataFrame(out, index=close.index)


def calc_volatility(close: pd.Series, windows: List[int] = [10, 20, 60, 120]) -> pd.DataFrame:
    """Realized volatility (rolling std of log returns)."""
    log_ret = np.log(close / close.shift(1))
    out = {}
    for w in windows:
        rv = log_ret.rolling(w).std() * np.sqrt(252 * 24 * 4)  # annualized for 15m
        out[f'rv_{w}'] = rv
    # Volatility regime: short/long ratio
    out['vol_regime'] = (log_ret.rolling(10).std() / (log_ret.rolling(60).std() + 1e-9))
    return pd.DataFrame(out, index=close.index)


def calc_rsi(close: pd.Series, windows: List[int] = [7, 14, 21]) -> pd.DataFrame:
    """RSI at multiple periods."""
    delta = close.diff()
    out = {}
    for w in windows:
        gain = delta.clip(lower=0).ewm(span=w, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(span=w, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        out[f'rsi_{w}'] = 100 - (100 / (1 + rs))
    return pd.DataFrame(out, index=close.index)


def calc_macd(close: pd.Series) -> pd.DataFrame:
    """MACD with histogram and normalized version."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    # Normalize by price to make cross-asset comparable
    return pd.DataFrame({
        'macd': macd / (close + 1e-9),
        'macd_signal': signal / (close + 1e-9),
        'macd_hist': hist / (close + 1e-9),
        'macd_cross': (macd > signal).astype(int),
    }, index=close.index)


def calc_bollinger(close: pd.Series, windows: List[int] = [20, 50], stds: List[float] = [1.5, 2.0]) -> pd.DataFrame:
    """Bollinger Bands position and width (not raw price levels)."""
    out = {}
    for w in windows:
        ma = close.rolling(w).mean()
        std = close.rolling(w).std()
        for s in stds:
            upper = ma + s * std
            lower = ma - s * std
            out[f'bb_pos_{w}_{s}'] = (close - lower) / (upper - lower + 1e-9)
            out[f'bb_width_{w}_{s}'] = (upper - lower) / (ma + 1e-9)
    return pd.DataFrame(out, index=close.index)


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, windows: List[int] = [14, 21]) -> pd.DataFrame:
    """ATR normalized by price."""
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    out = {}
    for w in windows:
        atr = tr.ewm(span=w, adjust=False).mean()
        out[f'atr_{w}'] = atr / (close + 1e-9)  # normalized
    return pd.DataFrame(out, index=close.index)


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """ADX trend strength."""
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0.0
    minus_dm[minus_dm <= plus_dm] = 0.0

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-9)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return pd.DataFrame({'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}, index=close.index)


def calc_volume_features(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """Volume-based features that actually work."""
    out = {}
    vol_ma20 = volume.rolling(20).mean()
    vol_ma5 = volume.rolling(5).mean()

    out['vol_ratio_20'] = volume / (vol_ma20 + 1e-9)  # volume spike
    out['vol_ratio_5'] = volume / (vol_ma5 + 1e-9)
    out['vol_momentum'] = vol_ma5 / (vol_ma20 + 1e-9)  # volume acceleration

    # OBV normalized (cumulative directional volume)
    direction = np.sign(close.diff())
    obv = (direction * volume).cumsum()
    obv_ma = obv.rolling(20).mean()
    out['obv_slope'] = (obv - obv_ma) / (obv.rolling(20).std() + 1e-9)  # OBV deviation from mean

    # CMF (Chaikin Money Flow)
    high = close  # approximation if only close/volume available
    low = close
    mf_vol = ((close - close.rolling(1).min()) - (close.rolling(1).max() - close)) * volume
    out['cmf_20'] = mf_vol.rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)

    # Price-volume divergence: price up + volume down = bearish
    price_direction = np.sign(close.pct_change(5))
    vol_direction = np.sign(volume.pct_change(5))
    out['pv_divergence'] = (price_direction != vol_direction).astype(int)

    return pd.DataFrame(out, index=close.index)


def calc_volume_features_full(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """Volume features using full OHLCV."""
    out = {}
    vol_ma20 = volume.rolling(20).mean()
    vol_ma5 = volume.rolling(5).mean()
    out['vol_ratio_20'] = volume / (vol_ma20 + 1e-9)
    out['vol_ratio_5'] = volume / (vol_ma5 + 1e-9)
    out['vol_momentum'] = vol_ma5 / (vol_ma20 + 1e-9)

    # True OBV
    direction = np.sign(close.diff())
    obv = (direction * volume).cumsum()
    obv_ma = obv.rolling(20).mean()
    out['obv_slope'] = (obv - obv_ma) / (obv.rolling(20).std() + 1e-9)

    # True CMF
    rng = (high - low).replace(0, np.nan)
    mf_mult = ((close - low) - (high - close)) / (rng + 1e-9)
    mf_vol = mf_mult * volume
    out['cmf_20'] = mf_vol.rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)

    # VWAP deviation
    tp = (high + low + close) / 3
    vwap = (tp * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    out['vwap_dev'] = (close - vwap) / (vwap + 1e-9)

    # Price-volume divergence
    price_direction = np.sign(close.pct_change(5))
    vol_direction = np.sign(volume.pct_change(5))
    out['pv_divergence'] = (price_direction != vol_direction).astype(int)

    return pd.DataFrame(out, index=close.index)


# ============================================================
# SECTION 2: Market Structure Features (Wyckoff-inspired but simple)
# ============================================================

def calc_market_structure(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.DataFrame:
    """Simple Wyckoff-inspired market structure.

    Only uses proven concepts:
    - Price relative to range
    - Support/resistance proximity
    - Volume-price agreement (spring/upthrust)
    """
    out = {}
    local_high = high.rolling(window).max()
    local_low = low.rolling(window).min()
    price_range = local_high - local_low + 1e-9

    # Price position in range [0, 1]
    out['price_pos_in_range'] = (close - local_low) / price_range

    # Distance to support/resistance (normalized)
    out['dist_to_resistance'] = (local_high - close) / price_range
    out['dist_to_support'] = (close - local_low) / price_range

    # Trend: close vs midpoint of range
    midpoint = (local_high + local_low) / 2
    out['above_midpoint'] = (close > midpoint).astype(int)

    return pd.DataFrame(out, index=close.index)


def calc_td_sequential(close: pd.Series) -> pd.DataFrame:
    """TD Sequential setup count (simplified, proven version)."""
    td_up = pd.Series(0, index=close.index)
    td_down = pd.Series(0, index=close.index)
    vals = close.values
    up_count = 0
    down_count = 0
    for i in range(len(vals)):
        if i < 4:
            up_count = down_count = 0
        else:
            if vals[i] > vals[i - 4]:
                up_count = up_count + 1 if up_count > 0 else 1
                down_count = 0
            elif vals[i] < vals[i - 4]:
                down_count = down_count + 1 if down_count > 0 else 1
                up_count = 0
            else:
                up_count = down_count = 0
        td_up.iloc[i] = min(up_count, 13)
        td_down.iloc[i] = min(down_count, 13)

    return pd.DataFrame({
        'td_up': td_up.shift(1).fillna(0),    # shift to avoid lookahead
        'td_down': td_down.shift(1).fillna(0),
        'td_buy_signal': (td_down >= 9).astype(int).shift(1).fillna(0),
        'td_sell_signal': (td_up >= 9).astype(int).shift(1).fillna(0),
    }, index=close.index)


def calc_regime(close: pd.Series) -> pd.DataFrame:
    """Market regime detection (trend vs chop, high/low vol)."""
    log_ret = np.log(close / close.shift(1)).fillna(0)
    rv10 = log_ret.rolling(10).std().fillna(0)
    rv60 = log_ret.rolling(60).std().fillna(0)

    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    return pd.DataFrame({
        'regime_vol_ratio': rv10 / (rv60 + 1e-9),         # >1 = vol expanding
        'regime_trend': ((ma20 - ma60) / (ma60 + 1e-9)),  # trend direction
        'regime_high_vol': (rv10 > rv60).astype(int),
        'regime_trending': (ma20 > ma60).astype(int),
    }, index=close.index)


# ============================================================
# SECTION 3: Crypto-Specific Alpha Factors
# These are the factors that ACTUALLY matter in crypto
# ============================================================

def calc_funding_rate_features(funding_df: Optional[pd.DataFrame], target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Funding rate features - #1 crypto alpha factor.

    Funding rate predicts mean reversion:
    - High positive funding -> longs pay shorts -> price pressure DOWN
    - High negative funding -> shorts pay longs -> price pressure UP

    Args:
        funding_df: DataFrame with 'funding_rate' column, 8h frequency
        target_index: DatetimeIndex of the target timeframe
    """
    empty = pd.DataFrame(index=target_index)
    if funding_df is None or funding_df.empty or 'funding_rate' not in funding_df.columns:
        logger.warning("⚠️ Funding rate data not available - critical alpha missing!")
        return empty

    fr = funding_df['funding_rate']

    out = {}
    # Raw funding rate
    out['funding_rate'] = fr
    # Cumulative 3-period (24h) funding
    out['funding_rate_cum24h'] = fr.rolling(3).sum()
    # Funding rate z-score (regime normalized)
    fr_mean = fr.rolling(30).mean()
    fr_std = fr.rolling(30).std()
    out['funding_rate_zscore'] = (fr - fr_mean) / (fr_std + 1e-9)
    # Extreme funding (>0.1% = extremely bullish leverage)
    out['funding_extreme_long'] = (fr > 0.001).astype(int)
    out['funding_extreme_short'] = (fr < -0.0005).astype(int)

    df = pd.DataFrame(out, index=funding_df.index)
    # Align to target: shift(1) = use last completed funding period only
    aligned = df.shift(1).reindex(target_index, method='ffill').fillna(0)
    return aligned


def calc_open_interest_features(oi_df: Optional[pd.DataFrame], target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Open Interest features.

    OI captures total leveraged positions:
    - OI up + price up = trend continuation (healthy)
    - OI up + price down = shorts building (bearish signal)
    - OI down + price move = deleveraging (trend exhaustion)
    """
    empty = pd.DataFrame(index=target_index)
    if oi_df is None or oi_df.empty:
        logger.warning("⚠️ Open Interest data not available")
        return empty

    oi_col = next((c for c in oi_df.columns if 'open_interest' in c.lower() or c == 'oi'), None)
    if oi_col is None:
        return empty

    oi = oi_df[oi_col].astype(float)
    out = {}
    out['oi'] = oi
    out['oi_change_1'] = oi.pct_change(1)
    out['oi_change_8'] = oi.pct_change(8)   # 8-period = ~2h for 15m data
    out['oi_ma_ratio'] = oi / (oi.rolling(20).mean() + 1e-9)

    df = pd.DataFrame(out, index=oi_df.index)
    aligned = df.shift(1).reindex(target_index, method='ffill').fillna(0)
    return aligned


def calc_oi_price_divergence(oi_aligned: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """
    OI vs Price divergence - strong directional signal.
    Must be called AFTER oi features are aligned to target timeframe.
    """
    if oi_aligned.empty or 'oi_change_1' not in oi_aligned.columns:
        return pd.DataFrame(index=close.index)

    price_change = close.pct_change(1)
    oi_change = oi_aligned['oi_change_1'].reindex(close.index).fillna(0)

    # OI and price moving together = trend, diverging = reversal
    oi_price_agree = (np.sign(price_change) == np.sign(oi_change)).astype(int)

    return pd.DataFrame({
        'oi_price_agree': oi_price_agree,
        'oi_price_diverge': 1 - oi_price_agree,
    }, index=close.index)


def calc_long_short_ratio(ls_df: Optional[pd.DataFrame], target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Long/Short ratio - market sentiment.
    Extreme readings are contrarian signals.
    """
    empty = pd.DataFrame(index=target_index)
    if ls_df is None or ls_df.empty:
        return empty

    ls_col = next((c for c in ls_df.columns if 'long_short' in c.lower() or 'ls_ratio' in c.lower()), None)
    if ls_col is None:
        return empty

    ls = ls_df[ls_col].astype(float)
    out = {}
    out['ls_ratio'] = ls
    ls_mean = ls.rolling(24).mean()
    ls_std = ls.rolling(24).std()
    out['ls_zscore'] = (ls - ls_mean) / (ls_std + 1e-9)
    # Contrarian extremes
    out['ls_extreme_long'] = (ls > ls.rolling(100).quantile(0.90)).astype(int)
    out['ls_extreme_short'] = (ls < ls.rolling(100).quantile(0.10)).astype(int)

    df = pd.DataFrame(out, index=ls_df.index)
    aligned = df.shift(1).reindex(target_index, method='ffill').fillna(0)
    return aligned


def calc_liquidations(liq_df: Optional[pd.DataFrame], target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Liquidation data - identifies forced selling/buying.
    Large liquidation cascades predict short-term reversal.
    """
    empty = pd.DataFrame(index=target_index)
    if liq_df is None or liq_df.empty:
        return empty

    long_liq_col = next((c for c in liq_df.columns if 'long_liq' in c.lower()), None)
    short_liq_col = next((c for c in liq_df.columns if 'short_liq' in c.lower()), None)

    if long_liq_col is None and short_liq_col is None:
        return empty

    out = {}
    if long_liq_col:
        ll = liq_df[long_liq_col].astype(float)
        ll_ma = ll.rolling(20).mean()
        out['long_liq_ratio'] = ll / (ll_ma + 1e-9)
        out['long_liq_spike'] = (ll > ll.rolling(100).quantile(0.95)).astype(int)

    if short_liq_col:
        sl = liq_df[short_liq_col].astype(float)
        sl_ma = sl.rolling(20).mean()
        out['short_liq_ratio'] = sl / (sl_ma + 1e-9)
        out['short_liq_spike'] = (sl > sl.rolling(100).quantile(0.95)).astype(int)

    df = pd.DataFrame(out, index=liq_df.index)
    aligned = df.shift(1).reindex(target_index, method='ffill').fillna(0)
    return aligned


# ============================================================
# SECTION 4: Higher Timeframe Reference (Single Shift Only)
# ============================================================

def calc_higher_tf_trend(
    df_higher: pd.DataFrame,
    current_index: pd.DatetimeIndex,
    tf_label: str
) -> pd.DataFrame:
    """
    Higher timeframe trend features with SINGLE shift.

    Bug fix: The original code used double-shift (_resample_ohlcv shift(1) +
    _align_higher_timeframe_to_current shift(1)), causing 2h lag on 1h data.
    This function uses a single shift(1) on the higher TF bar.

    Rule: use the LAST COMPLETED higher TF bar. One shift is enough.
    """
    close = df_higher['close']
    high = df_higher['high']
    low = df_higher['low']
    volume = df_higher.get('volume', pd.Series(1, index=df_higher.index))

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    trend = pd.Series(0, index=df_higher.index)
    trend[(ema20 > ema50) & (macd > macd_signal)] = 1
    trend[(ema20 < ema50) & (macd < macd_signal)] = -1

    resistance = high.rolling(20).max()
    support = low.rolling(20).min()
    price_range = resistance - support + 1e-9

    features = pd.DataFrame({
        f'{tf_label}_trend': trend,
        f'{tf_label}_ema20_ratio': close / (ema20 + 1e-9),
        f'{tf_label}_ema50_ratio': close / (ema50 + 1e-9),
        f'{tf_label}_price_pos': (close - support) / price_range,
        f'{tf_label}_macd_hist': (macd - macd_signal) / (close + 1e-9),
    }, index=df_higher.index)

    # SINGLE shift(1): only use last completed bar
    shifted = features.shift(1)
    aligned = shifted.reindex(current_index, method='ffill').fillna(0)
    return aligned


# ============================================================
# SECTION 5: Main Factory Function
# ============================================================

def build_feature_matrix(
    ohlcv: pd.DataFrame,
    derivatives_path: Optional[Path] = None,
    symbol: str = 'BTCUSDT',
    higher_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix from clean, non-leaking factors.

    Feature groups and expected counts:
    - Returns/Momentum: ~12 features
    - Volatility: ~5 features
    - RSI: ~3 features
    - MACD: ~4 features
    - Bollinger: ~8 features
    - ATR: ~2 features
    - ADX: ~3 features
    - Volume: ~6 features
    - Market Structure (Wyckoff-lite): ~4 features
    - TD Sequential: ~4 features
    - Regime: ~4 features
    - Higher TF (1h/4h trend): ~10 features
    - Crypto derivatives (if available): ~15 features
    TOTAL: ~80 clean features (vs 200+ bloated originals)

    Args:
        ohlcv: OHLCV DataFrame with columns [open, high, low, close, volume]
        derivatives_path: Path to derivatives data folder
        symbol: Trading symbol for derivative data lookup
        higher_tf_data: Dict of {'1h': df_1h, '4h': df_4h}
    """
    if ohlcv is None or ohlcv.empty:
        raise ValueError("OHLCV data is required")

    close = ohlcv['close']
    high = ohlcv['high']
    low = ohlcv['low']
    volume = ohlcv['volume']
    idx = ohlcv.index

    frames: List[pd.DataFrame] = []

    # --- Group 1: Returns & Momentum ---
    try:
        frames.append(calc_returns(close))
    except Exception as e:
        logger.warning(f"returns failed: {e}")

    # --- Group 2: Volatility ---
    try:
        frames.append(calc_volatility(close))
    except Exception as e:
        logger.warning(f"volatility failed: {e}")

    # --- Group 3: RSI ---
    try:
        frames.append(calc_rsi(close))
    except Exception as e:
        logger.warning(f"RSI failed: {e}")

    # --- Group 4: MACD ---
    try:
        frames.append(calc_macd(close))
    except Exception as e:
        logger.warning(f"MACD failed: {e}")

    # --- Group 5: Bollinger Bands ---
    try:
        frames.append(calc_bollinger(close))
    except Exception as e:
        logger.warning(f"Bollinger failed: {e}")

    # --- Group 6: ATR ---
    try:
        frames.append(calc_atr(high, low, close))
    except Exception as e:
        logger.warning(f"ATR failed: {e}")

    # --- Group 7: ADX ---
    try:
        frames.append(calc_adx(high, low, close))
    except Exception as e:
        logger.warning(f"ADX failed: {e}")

    # --- Group 8: Volume ---
    try:
        frames.append(calc_volume_features_full(high, low, close, volume))
    except Exception as e:
        logger.warning(f"Volume failed: {e}")

    # --- Group 9: Market Structure ---
    try:
        frames.append(calc_market_structure(high, low, close))
    except Exception as e:
        logger.warning(f"Market structure failed: {e}")

    # --- Group 10: TD Sequential ---
    try:
        frames.append(calc_td_sequential(close))
    except Exception as e:
        logger.warning(f"TD Sequential failed: {e}")

    # --- Group 11: Regime ---
    try:
        frames.append(calc_regime(close))
    except Exception as e:
        logger.warning(f"Regime failed: {e}")

    # --- Group 12: Higher Timeframe Trend (single shift) ---
    if higher_tf_data:
        for tf_label, df_htf in higher_tf_data.items():
            try:
                htf_feat = calc_higher_tf_trend(df_htf, idx, tf_label)
                frames.append(htf_feat)
                logger.info(f"✅ Added {tf_label} trend features ({len(htf_feat.columns)} cols)")
            except Exception as e:
                logger.warning(f"Higher TF {tf_label} failed: {e}")

    # --- Group 13: Crypto Derivatives (optional but critical alpha) ---
    if derivatives_path is not None:
        funding_df = _load_derivative(derivatives_path, symbol, 'funding')
        oi_df = _load_derivative(derivatives_path, symbol, 'open_interest')
        ls_df = _load_derivative(derivatives_path, symbol, 'long_short')
        liq_df = _load_derivative(derivatives_path, symbol, 'liquidations')

        try:
            fr_feat = calc_funding_rate_features(funding_df, idx)
            if not fr_feat.empty:
                frames.append(fr_feat)
                logger.info(f"✅ Funding rate features: {len(fr_feat.columns)} cols")
        except Exception as e:
            logger.warning(f"Funding rate failed: {e}")

        try:
            oi_feat = calc_open_interest_features(oi_df, idx)
            if not oi_feat.empty:
                frames.append(oi_feat)
                # Add OI-price divergence
                div_feat = calc_oi_price_divergence(oi_feat, close)
                if not div_feat.empty:
                    frames.append(div_feat)
                logger.info(f"✅ Open Interest features: {len(oi_feat.columns)} cols")
        except Exception as e:
            logger.warning(f"OI failed: {e}")

        try:
            ls_feat = calc_long_short_ratio(ls_df, idx)
            if not ls_feat.empty:
                frames.append(ls_feat)
                logger.info(f"✅ Long/Short ratio features: {len(ls_feat.columns)} cols")
        except Exception as e:
            logger.warning(f"Long/short failed: {e}")

        try:
            liq_feat = calc_liquidations(liq_df, idx)
            if not liq_feat.empty:
                frames.append(liq_feat)
                logger.info(f"✅ Liquidation features: {len(liq_feat.columns)} cols")
        except Exception as e:
            logger.warning(f"Liquidations failed: {e}")
    else:
        logger.warning(
            "⚠️ derivatives_path=None. Funding rate, OI, L/S ratio not included. "
            "These are critical crypto alpha factors. "
            "Pass derivatives_path to enable them."
        )

    # --- Concatenate ---
    X = pd.concat([f for f in frames if not f.empty], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    logger.info(f"✅ Feature matrix built: {X.shape[0]} rows x {X.shape[1]} features")
    return X


def _load_derivative(base_path: Path, symbol: str, name: str) -> Optional[pd.DataFrame]:
    """Load derivative data from standard paths."""
    search_dirs = [
        base_path / symbol,
        base_path / 'derived' / symbol,
        base_path,
    ]
    for d in search_dirs:
        for ext in ['parquet', 'csv']:
            p = d / f"{name}.{ext}"
            if p.exists():
                try:
                    if ext == 'parquet':
                        return pd.read_parquet(p)
                    else:
                        df = pd.read_csv(p, index_col=0, parse_dates=True)
                        return df
                except Exception as e:
                    logger.warning(f"Failed to load {p}: {e}")
    return None


# ============================================================
# SECTION 6: Feature Quality Filter (keep this, it's good)
# ============================================================

def filter_low_quality(X: pd.DataFrame, min_variance: float = 1e-6, max_nan_ratio: float = 0.3) -> pd.DataFrame:
    """Remove near-zero variance and high-NaN features."""
    # NaN filter
    nan_ratio = X.isnull().mean()
    X = X.loc[:, nan_ratio <= max_nan_ratio]

    # Variance filter
    var = X.var()
    X = X.loc[:, var > min_variance]

    return X


def remove_correlated(X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """
    Remove highly correlated features.
    Priority: keep higher-variance features.
    Returns list of columns to keep.
    """
    if X.empty:
        return []

    X_num = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
    std_series = X_num.std().fillna(0)
    ordered_cols = std_series.sort_values(ascending=False).index.tolist()

    corr = X_num.corr().abs()
    keep: List[str] = []
    removed: set = set()

    for col in ordered_cols:
        if col in removed:
            continue
        keep.append(col)
        try:
            high_corr = corr.index[(corr[col] > threshold) & (corr.index != col)].tolist()
            removed.update(high_corr)
        except Exception:
            pass

    return keep
