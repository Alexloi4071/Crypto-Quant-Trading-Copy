# -*- coding: utf-8 -*-
"""
P0 Fix: Label Engine
====================
Replaces the label generation in optuna_feature.py._generate_labels()

THE BUG FIXED HERE:
-------------------
Original code:
    future_prices = price_series.shift(-lag)
    returns = (future_prices - price_series) / price_series
    ret_past = returns.shift(1)                              # OK so far
    upper = ret_past.rolling(lookback).quantile(profit_q)  # BUG!

The BUG: `ret_past.rolling(lookback).quantile(profit_q)` uses a ROLLING
window that shifts over the data. At each timestamp t, the threshold uses
returns from t-lookback to t-1. But the DISTRIBUTION of returns in a bull
market is systematically higher, so:
- In a bull market: upper threshold is high -> almost nothing triggers BUY
- In a bear market: lower threshold is low -> almost nothing triggers SELL
This creates severe label imbalance that changes with market regime.

THE FIX:
--------
Option A (simple): Use FIXED quantiles computed on the training portion only.
Option B (better): ATR-based triple barrier labels.

We implement both and let the user choose via method parameter.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_labels_fixed_quantile(
    close: pd.Series,
    lag: int = 12,
    buy_quantile: float = 0.70,
    sell_quantile: float = 0.30,
    train_end_idx: Optional[int] = None,
) -> pd.Series:
    """
    Fixed quantile labels - NO rolling lookahead.

    Labels:
        0 = SELL (return < sell_quantile threshold)
        1 = HOLD
        2 = BUY  (return > buy_quantile threshold)

    The thresholds are computed ONCE on the training portion of the data.
    They do NOT change over time, so there is NO rolling lookahead bias.

    Args:
        close: Close price series
        lag: Future periods to compute return (e.g., 12 = 3h for 15m data)
        buy_quantile: Return percentile above which = BUY (e.g., 0.70)
        sell_quantile: Return percentile below which = SELL (e.g., 0.30)
        train_end_idx: Index position of training end (to compute thresholds only
                       on training data). If None, uses first 60% of data.
    """
    if len(close) < lag + 100:
        raise ValueError(f"Need at least {lag + 100} rows, got {len(close)}")

    # Future return: return from t to t+lag
    future_return = close.shift(-lag) / close - 1

    # Compute thresholds on training data ONLY
    if train_end_idx is None:
        train_end_idx = int(len(close) * 0.6)

    train_returns = future_return.iloc[:train_end_idx].dropna()

    if len(train_returns) < 100:
        # Fallback: use all available
        train_returns = future_return.dropna()

    upper_threshold = float(train_returns.quantile(buy_quantile))
    lower_threshold = float(train_returns.quantile(sell_quantile))

    # Ensure thresholds make sense
    if upper_threshold <= 0:
        upper_threshold = max(float(train_returns.std() * 0.5), 0.002)
        logger.warning(f"buy_quantile={buy_quantile} gave threshold <= 0, using {upper_threshold:.4f}")
    if lower_threshold >= 0:
        lower_threshold = min(-float(train_returns.std() * 0.5), -0.002)
        logger.warning(f"sell_quantile={sell_quantile} gave threshold >= 0, using {lower_threshold:.4f}")

    logger.info(
        f"🏷️ Fixed quantile labels: lag={lag}, "
        f"BUY if return > {upper_threshold:.4f} ({buy_quantile:.0%}), "
        f"SELL if return < {lower_threshold:.4f} ({sell_quantile:.0%})"
    )

    labels = pd.Series(1, index=close.index, dtype=int)  # default HOLD
    labels[future_return > upper_threshold] = 2   # BUY
    labels[future_return < lower_threshold] = 0   # SELL

    # Drop the last `lag` rows (their future return is NaN)
    labels = labels.iloc[:-lag]

    dist = labels.value_counts(normalize=True).sort_index()
    logger.info(
        f"📊 Label distribution: "
        f"SELL={dist.get(0, 0):.1%}, HOLD={dist.get(1, 0):.1%}, BUY={dist.get(2, 0):.1%}"
    )

    return labels.astype(int)


def generate_labels_atr_barrier(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lag: int = 12,
    atr_period: int = 14,
    profit_multiplier: float = 2.0,
    stop_multiplier: float = 1.0,
) -> pd.Series:
    """
    ATR-based triple barrier labels.

    THIS IS THE BETTER METHOD FOR CRYPTO.

    Labels by which barrier is hit first within `lag` periods:
        2 = BUY:  profit target (close + profit_multiplier * ATR) hit first
        0 = SELL: stop loss (close - stop_multiplier * ATR) hit first
        1 = HOLD: neither barrier hit within lag periods (time expiry)

    Advantages over return quantile:
    - Adapts to volatility: wider barriers in volatile markets
    - Risk:Reward ratio is explicit (profit_mult / stop_mult)
    - No future distribution lookahead
    - Produces balanced classes by tuning multipliers

    Args:
        high/low/close: OHLCV series
        lag: Max periods to hold (time barrier)
        atr_period: ATR period
        profit_multiplier: Take profit = close + N * ATR
        stop_multiplier: Stop loss = close - M * ATR
    """
    # Compute ATR
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False).mean()

    n = len(close)
    labels = pd.Series(1, index=close.index, dtype=int)  # default HOLD

    vals_close = close.values
    vals_high = high.values
    vals_low = low.values
    vals_atr = atr.values

    for i in range(n - lag):
        entry = vals_close[i]
        atr_val = vals_atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        target = entry + profit_multiplier * atr_val
        stop = entry - stop_multiplier * atr_val

        label = 1  # HOLD by default (time barrier)
        for j in range(i + 1, min(i + lag + 1, n)):
            if vals_high[j] >= target:
                label = 2  # BUY (profit hit)
                break
            if vals_low[j] <= stop:
                label = 0  # SELL (stop hit)
                break

        labels.iloc[i] = label

    # Mark last `lag` rows as NaN (no future data)
    labels.iloc[-lag:] = np.nan
    labels = labels.dropna().astype(int)

    dist = labels.value_counts(normalize=True).sort_index()
    logger.info(
        f"🏷️ ATR barrier labels: lag={lag}, profit={profit_multiplier}xATR, stop={stop_multiplier}xATR, "
        f"RR={profit_multiplier/stop_multiplier:.1f}:1"
    )
    logger.info(
        f"📊 Label distribution: "
        f"SELL={dist.get(0, 0):.1%}, HOLD={dist.get(1, 0):.1%}, BUY={dist.get(2, 0):.1%}"
    )

    return labels


def validate_label_quality(labels: pd.Series) -> Dict:
    """
    Validate label quality before training.
    Returns dict with quality metrics and pass/fail.
    """
    if labels is None or len(labels) == 0:
        return {'pass': False, 'reason': 'empty labels'}

    dist = labels.value_counts(normalize=True).sort_index()
    sell_pct = dist.get(0, 0)
    hold_pct = dist.get(1, 0)
    buy_pct = dist.get(2, 0)

    issues = []

    # Check class balance
    if sell_pct < 0.10:
        issues.append(f"SELL class too rare: {sell_pct:.1%} < 10%")
    if buy_pct < 0.10:
        issues.append(f"BUY class too rare: {buy_pct:.1%} < 10%")
    if hold_pct > 0.80:
        issues.append(f"HOLD class too dominant: {hold_pct:.1%} > 80%")

    # Check minimum samples per class
    counts = labels.value_counts()
    for cls in [0, 1, 2]:
        if counts.get(cls, 0) < 100:
            issues.append(f"Class {cls} has fewer than 100 samples: {counts.get(cls, 0)}")

    # Check label stability (not all noise)
    changes = int((labels.diff() != 0).sum())
    change_rate = changes / len(labels)
    if change_rate > 0.8:
        issues.append(f"Label change rate too high: {change_rate:.1%} (possible noise)")
    if change_rate < 0.05:
        issues.append(f"Label change rate too low: {change_rate:.1%} (no signal)")

    result = {
        'pass': len(issues) == 0,
        'issues': issues,
        'distribution': {
            'sell': float(sell_pct),
            'hold': float(hold_pct),
            'buy': float(buy_pct),
        },
        'total_samples': int(len(labels)),
        'change_rate': float(change_rate),
    }

    if issues:
        logger.warning(f"⚠️ Label quality issues: {issues}")
    else:
        logger.info(f"✅ Labels passed quality check: {len(labels)} samples")

    return result


def optimize_label_params(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    method: str = 'fixed_quantile',
) -> Dict:
    """
    Grid search over label parameters to find balanced distribution.
    Run this ONCE before training to find good label params.

    Returns best params that give close to 25/50/25 distribution.
    """
    if method == 'atr_barrier':
        candidates = [
            {'lag': 8, 'profit_multiplier': 1.5, 'stop_multiplier': 0.75},
            {'lag': 12, 'profit_multiplier': 2.0, 'stop_multiplier': 1.0},
            {'lag': 16, 'profit_multiplier': 2.0, 'stop_multiplier': 1.0},
            {'lag': 12, 'profit_multiplier': 1.5, 'stop_multiplier': 1.0},
            {'lag': 20, 'profit_multiplier': 2.5, 'stop_multiplier': 1.0},
        ]
        best = None
        best_score = float('inf')
        for p in candidates:
            try:
                lbls = generate_labels_atr_barrier(
                    high, low, close,
                    lag=p['lag'],
                    profit_multiplier=p['profit_multiplier'],
                    stop_multiplier=p['stop_multiplier']
                )
                quality = validate_label_quality(lbls)
                dist = quality['distribution']
                # Score = deviation from target [25%, 50%, 25%]
                score = abs(dist['sell'] - 0.25) + abs(dist['buy'] - 0.25) + abs(dist['hold'] - 0.50)
                if score < best_score and quality['pass']:
                    best_score = score
                    best = {**p, 'score': score, 'distribution': dist}
            except Exception as e:
                logger.warning(f"Failed params {p}: {e}")
        if best is None:
            best = {'lag': 12, 'profit_multiplier': 2.0, 'stop_multiplier': 1.0}
        return best

    else:  # fixed_quantile
        candidates = [
            {'lag': 8, 'buy_quantile': 0.75, 'sell_quantile': 0.25},
            {'lag': 12, 'buy_quantile': 0.75, 'sell_quantile': 0.25},
            {'lag': 16, 'buy_quantile': 0.75, 'sell_quantile': 0.25},
            {'lag': 12, 'buy_quantile': 0.70, 'sell_quantile': 0.30},
            {'lag': 20, 'buy_quantile': 0.70, 'sell_quantile': 0.30},
        ]
        best = None
        best_score = float('inf')
        for p in candidates:
            try:
                lbls = generate_labels_fixed_quantile(close, **p)
                quality = validate_label_quality(lbls)
                dist = quality['distribution']
                score = abs(dist['sell'] - 0.25) + abs(dist['buy'] - 0.25) + abs(dist['hold'] - 0.50)
                if score < best_score and quality['pass']:
                    best_score = score
                    best = {**p, 'score': score, 'distribution': dist}
            except Exception as e:
                logger.warning(f"Failed params {p}: {e}")
        if best is None:
            best = {'lag': 12, 'buy_quantile': 0.75, 'sell_quantile': 0.25}
        return best
