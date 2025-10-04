"""
Cross-sectional factor research toolkit.

This module integrates FinLab-inspired methodologies (rank normalization, excess return labels,
IC/IC decay, factor long-short analytics) into the project without any external dependency
on the FinLab platform. It provides utilities for analyzing factor quality, generating reports,
and exporting metrics that can support Layer2 scoring and optimization decisions.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr, ttest_1samp

import plotly.graph_objects as go

warnings.filterwarnings("ignore")

try:
    import networkx as nx  # Optional for factor centrality analytics

    NX_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    NX_AVAILABLE = False


# ======================================================================================
# Helpers for data alignment and preprocessing
# ======================================================================================

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame index is a monotonic increasing DateTimeIndex."""

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()]

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    return df


def _to_wide(panel: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Convert a panel that may be in long or multi-index form into a wide date x asset
    DataFrame. Accepts DataFrame or Series with either:
      - Wide format already (columns=assets)
      - MultiIndex (date, asset)
      - Columns containing ['date', 'asset', 'value']
    """

    if isinstance(panel, pd.Series):
        panel = panel.to_frame("value")

    if isinstance(panel.index, pd.MultiIndex):
        if len(panel.index.levels) == 2:
            reshaped = panel.unstack()
            if isinstance(reshaped, pd.Series):
                reshaped = reshaped.to_frame()
            if isinstance(reshaped.columns, pd.MultiIndex):
                reshaped.columns = [c[1] if isinstance(c, tuple) else c for c in reshaped.columns]
            return reshaped

    return panel


def align_panels(
    prices: pd.DataFrame,
    factor_panel: pd.DataFrame,
    shift_factor_by: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align prices and factor panels (wide format). Optionally shift the factor panel forward
    to enforce the use of T-1 information.
    """

    prices = _ensure_datetime_index(_to_wide(prices).copy())
    factor_panel = _ensure_datetime_index(_to_wide(factor_panel).copy())

    if shift_factor_by > 0:
        factor_panel = factor_panel.shift(shift_factor_by)

    common_index = prices.index.intersection(factor_panel.index)
    prices = prices.loc[common_index]
    factor_panel = factor_panel.loc[common_index]

    common_assets = prices.columns.intersection(factor_panel.columns)
    return prices[common_assets], factor_panel[common_assets]


# ======================================================================================
# Cross-sectional normalization transforms
# ======================================================================================

def winsorize_by_date(factor_panel: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Apply cross-sectional winsorization per date."""

    def _clip_row(row: pd.Series) -> pd.Series:
        lo = row.quantile(lower)
        hi = row.quantile(upper)
        return row.clip(lo, hi)

    return factor_panel.apply(_clip_row, axis=1)


def zscore_by_date(factor_panel: pd.DataFrame) -> pd.DataFrame:
    """Apply cross-sectional z-score standardization per date."""

    mean = factor_panel.mean(axis=1)
    std = factor_panel.std(axis=1).replace(0, np.nan)
    return factor_panel.sub(mean, axis=0).div(std, axis=0)


def rank_pct_by_date(factor_panel: pd.DataFrame) -> pd.DataFrame:
    """Apply cross-sectional percentile ranking per date (FinLab style)."""

    return factor_panel.rank(axis=1, pct=True)


def normalize_pipeline(
    factor_panel: pd.DataFrame,
    do_winsor: bool = True,
    do_rank: bool = True,
    do_zscore: bool = False,
) -> pd.DataFrame:
    """
    Apply a recommended normalization pipeline (winsor -> rank -> optional z-score).
    """

    transformed = factor_panel.copy()
    if do_winsor:
        transformed = winsorize_by_date(transformed)
    if do_rank:
        transformed = rank_pct_by_date(transformed)
    if do_zscore:
        transformed = zscore_by_date(transformed)
    return transformed


# ======================================================================================
# Label generation and forward returns
# ======================================================================================

def future_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Compute forward returns over the given horizon."""

    return prices.shift(-horizon) / prices - 1


def excess_over_mean_label(
    prices: pd.DataFrame,
    horizon: int = 20,
    pos_q: float = 0.7,
    neg_q: float = 0.3,
) -> pd.DataFrame:
    """Generate multi-class labels based on excess returns versus cross-sectional mean."""

    fwd = future_returns(prices, horizon)
    excess = fwd.sub(fwd.mean(axis=1), axis=0)
    ranks = excess.rank(axis=1, pct=True)
    buy = (ranks >= pos_q).astype(np.int8) * 2
    hold = ((ranks < pos_q) & (ranks > neg_q)).astype(np.int8)
    sell = (ranks <= neg_q).astype(np.int8) * 0
    return (buy + hold + sell).astype("int8")


# ======================================================================================
# Information Coefficient analytics
# ======================================================================================

def daily_ic(
    factor_panel: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """Compute daily cross-sectional IC series."""

    index = factor_panel.index.intersection(fwd_ret.index)
    ic_values: List[float] = []
    for dt in index:
        x = factor_panel.loc[dt].dropna()
        y = fwd_ret.loc[dt].reindex(x.index).dropna()
        common = x.index.intersection(y.index)
        if len(common) >= 5:
            if method == "spearman":
                correlation = spearmanr(x.loc[common], y.loc[common]).correlation
            else:
                correlation = pearsonr(x.loc[common], y.loc[common])[0]
            ic_values.append(correlation)
        else:
            ic_values.append(np.nan)
    return pd.Series(ic_values, index=index).dropna()


def ic_stats(ic_series: pd.Series) -> Dict[str, float]:
    """Summarize IC statistics (mean, std, t-value, p-value, IR)."""

    if ic_series is None or len(ic_series) == 0:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_t": np.nan,
            "ic_p": np.nan,
            "ic_ir": np.nan,
        }

    mean = float(ic_series.mean())
    std = float(ic_series.std(ddof=1))
    try:
        t_stat, p_val = ttest_1samp(ic_series.values, 0.0, nan_policy="omit")
    except Exception:  # pragma: no cover - fallback
        t_stat, p_val = np.nan, np.nan
    information_ratio = mean / std if std and std > 0 else np.nan

    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_t": float(t_stat),
        "ic_p": float(p_val),
        "ic_ir": float(information_ratio),
    }


def ic_decay(
    factor_panel: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: List[int],
    method: str = "spearman",
) -> Dict[int, float]:
    """Compute average IC for multiple horizons (IC decay)."""

    decay: Dict[int, float] = {}
    for horizon in horizons:
        ic_series = daily_ic(factor_panel, future_returns(prices, horizon), method=method)
        decay[horizon] = float(ic_series.mean()) if len(ic_series) else np.nan
    return decay


def rolling_ic(
    factor_panel: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    window: int = 60,
    method: str = "spearman",
) -> pd.Series:
    """Compute rolling mean IC (stability check)."""

    ic_series = daily_ic(factor_panel, fwd_ret, method=method)
    return ic_series.rolling(window).mean().dropna()


# ======================================================================================
# Quantile analysis, long-short, turnover, coverage
# ======================================================================================

def quantile_groups(factor_panel: pd.DataFrame, n_quantiles: int = 5) -> Dict[int, pd.DataFrame]:
    """Return boolean masks for each quantile bucket per date."""

    ranks = factor_panel.rank(axis=1, pct=True)
    masks: Dict[int, pd.DataFrame] = {}
    for q in range(1, n_quantiles + 1):
        lower = (q - 1) / n_quantiles
        upper = q / n_quantiles
        masks[q] = (ranks > lower) & (ranks <= upper)
    return masks


def quantile_returns(
    factor_panel: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_quantiles: int = 5,
) -> Dict[int, pd.Series]:
    """Compute equal-weighted returns for each quantile bucket."""

    masks = quantile_groups(factor_panel, n_quantiles=n_quantiles)
    index = factor_panel.index.intersection(fwd_ret.index)
    results: Dict[int, List[float]] = {q: [] for q in masks}

    for dt in index:
        forward_returns = fwd_ret.loc[dt]
        for q, mask in masks.items():
            assets = mask.loc[dt]
            if assets.any():
                results[q].append(forward_returns.loc[assets].mean())
            else:
                results[q].append(np.nan)

    return {q: pd.Series(values, index=index).dropna() for q, values in results.items()}


def long_short_series(
    factor_panel: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.2,
) -> pd.Series:
    """Compute long-short series using top q and bottom q percentile buckets."""

    ranks = factor_panel.rank(axis=1, pct=True)
    index = factor_panel.index.intersection(fwd_ret.index)
    long_short_values: List[float] = []

    for dt in index:
        cross_section = ranks.loc[dt].dropna()
        if len(cross_section) < 10:
            long_short_values.append(np.nan)
            continue

        low, high = cross_section.quantile(q), cross_section.quantile(1 - q)
        long_idx = cross_section[cross_section >= high].index
        short_idx = cross_section[cross_section <= low].index
        returns = fwd_ret.loc[dt]
        long_short_values.append(returns.reindex(long_idx).mean() - returns.reindex(short_idx).mean())

    return pd.Series(long_short_values, index=index).dropna()


def turnover_rate(factor_panel: pd.DataFrame, q: float = 0.2) -> Dict[str, pd.Series]:
    """Compute daily turnover rates for long and short baskets."""

    ranks = factor_panel.rank(axis=1, pct=True)
    index = ranks.index
    long_sets: List[set] = []
    short_sets: List[set] = []

    for dt in index:
        cross_section = ranks.loc[dt].dropna()
        lo, hi = cross_section.quantile(q), cross_section.quantile(1 - q)
        long_sets.append(set(cross_section[cross_section >= hi].index))
        short_sets.append(set(cross_section[cross_section <= lo].index))

    def _turnover(sets: List[set]) -> pd.Series:
        values = [np.nan]
        for i in range(1, len(sets)):
            prev = sets[i - 1]
            curr = sets[i]
            union = prev | curr
            if not union:
                values.append(np.nan)
            else:
                values.append(1 - len(prev & curr) / len(union))
        return pd.Series(values, index=index)

    return {
        "long_turnover": _turnover(long_sets),
        "short_turnover": _turnover(short_sets),
    }


def coverage(factor_panel: pd.DataFrame) -> pd.Series:
    """Compute factor coverage ratio per date."""

    return factor_panel.notna().sum(axis=1) / factor_panel.shape[1]


# ======================================================================================
# Factor correlation and centrality analytics
# ======================================================================================

def factor_correlation(factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute Spearman correlation matrix among multiple factor panels."""

    names = list(factors.keys())
    common_index = None
    common_cols = None

    for panel in factors.values():
        wide = _to_wide(panel)
        common_index = wide.index if common_index is None else common_index.intersection(wide.index)
        common_cols = wide.columns if common_cols is None else common_cols.intersection(wide.columns)

    aligned = {name: _to_wide(panel).loc[common_index, common_cols] for name, panel in factors.items()}
    matrix = np.column_stack([aligned[name].values.reshape(-1) for name in names])
    flattened = pd.DataFrame(matrix, columns=names)
    return flattened.corr(method="spearman")


def factor_centrality_from_corr(corr_df: pd.DataFrame, threshold: float = 0.5) -> Optional[pd.Series]:
    """Estimate factor centrality using a correlation network (requires networkx)."""

    if not NX_AVAILABLE:
        return None

    graph = nx.Graph()
    nodes = corr_df.columns.tolist()
    graph.add_nodes_from(nodes)

    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if j <= i:
                continue
            weight = abs(corr_df.loc[a, b])
            if np.isfinite(weight) and weight >= threshold:
                graph.add_edge(a, b, weight=weight)

    if graph.number_of_edges() == 0:
        return pd.Series(0.0, index=nodes)

    centrality = nx.degree_centrality(graph)
    return pd.Series(centrality).sort_values(ascending=False)


# ======================================================================================
# Plotly-based visualization helpers
# ======================================================================================

def plot_ic_series(ic_series: pd.Series, out_html: str) -> str:
    fig = go.Figure(go.Scatter(x=ic_series.index, y=ic_series.values, name="IC", mode="lines"))
    fig.add_hline(y=float(ic_series.mean()), line_dash="dash", annotation_text=f"Mean={ic_series.mean():.3f}")
    fig.update_layout(title="Daily IC", template="plotly_white", xaxis_title="Date", yaxis_title="IC")
    fig.write_html(out_html)
    return out_html


def plot_ic_hist(ic_series: pd.Series, out_html: str) -> str:
    fig = go.Figure(go.Histogram(x=ic_series.values, nbinsx=40))
    fig.add_vline(x=float(ic_series.mean()), line_dash="dash", annotation_text=f"Mean={ic_series.mean():.3f}")
    fig.update_layout(title="IC Distribution", template="plotly_white", xaxis_title="IC", yaxis_title="Frequency")
    fig.write_html(out_html)
    return out_html


def plot_ic_decay_bar(decay: Dict[int, float], out_html: str) -> str:
    horizons = list(decay.keys())
    values = [decay[h] for h in horizons]
    fig = go.Figure(go.Bar(x=[str(h) for h in horizons], y=values))
    fig.update_layout(title="IC Decay by Horizon", template="plotly_white", xaxis_title="Horizon", yaxis_title="Mean IC")
    fig.write_html(out_html)
    return out_html


def plot_quantile_returns(quantile_series: Dict[int, pd.Series], out_html: str) -> str:
    fig = go.Figure()
    for quantile, series in sorted(quantile_series.items()):
        cumret = (1 + series.fillna(0)).cumprod()
        fig.add_trace(go.Scatter(x=cumret.index, y=cumret.values, mode="lines", name=f"Q{quantile}"))
    fig.update_layout(title="Quantile Cumulative Returns", template="plotly_white", xaxis_title="Date", yaxis_title="Cumulative Return")
    fig.write_html(out_html)
    return out_html


def plot_long_short(ls_series: pd.Series, out_html: str) -> str:
    cumret = (1 + ls_series.fillna(0)).cumprod()
    drawdown = cumret / cumret.cummax() - 1

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=cumret.index, y=cumret.values, name="L-S CumRet", mode="lines"), secondary_y=False)
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, name="Drawdown", mode="lines", line=dict(color="red", width=1)), secondary_y=True)
    fig.update_layout(title="Long-Short Cumulative Return & Drawdown", template="plotly_white")
    fig.update_yaxes(title_text="Cumulative Return", secondary_y=False)
    fig.update_yaxes(title_text="Drawdown", secondary_y=True)
    fig.write_html(out_html)
    return out_html


def plot_turnover(turnover: Dict[str, pd.Series], out_html: str) -> str:
    fig = go.Figure()
    for name, series in turnover.items():
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=name, mode="lines"))
    fig.update_layout(title="Turnover (Daily)", template="plotly_white", xaxis_title="Date", yaxis_title="Rate")
    fig.write_html(out_html)
    return out_html


def plot_coverage(cov_series: pd.Series, out_html: str) -> str:
    fig = go.Figure(go.Scatter(x=cov_series.index, y=cov_series.values, name="Coverage", mode="lines"))
    fig.update_layout(title="Coverage over Time", template="plotly_white", xaxis_title="Date", yaxis_title="Coverage")
    fig.write_html(out_html)
    return out_html


def plot_factor_corr_heatmap(corr_df: pd.DataFrame, out_html: str) -> str:
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns.tolist(),
            y=corr_df.index.tolist(),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
        )
    )
    fig.update_layout(title="Factor Correlation (Spearman)", template="plotly_white")
    fig.write_html(out_html)
    return out_html


# ======================================================================================
# Layer2 summary helper
# ======================================================================================

def _annualized_ratio(series: pd.Series, periods_per_year: int = 252) -> float:
    if len(series) < 5:
        return np.nan
    mean = series.mean() * periods_per_year
    std = series.std(ddof=1) * np.sqrt(periods_per_year)
    return float(mean / std) if std and std > 0 else np.nan


def summarize_for_layer2(
    factor_name: str,
    ic_series: pd.Series,
    ls_series: pd.Series,
    ic_decay_values: Dict[int, float],
    turnover: Dict[str, pd.Series],
    coverage_series: pd.Series,
) -> Dict[str, float]:
    """Export compact metrics consumable by Layer2 scoring logic."""

    summary = {
        "factor": factor_name,
    }
    summary.update(ic_stats(ic_series))

    for horizon, value in ic_decay_values.items():
        summary[f"ic_decay_{horizon}"] = float(value) if value is not None else np.nan

    summary["ls_cumret"] = float((1 + ls_series.fillna(0)).prod() - 1) if len(ls_series) else np.nan
    summary["ls_ann_sharpe"] = _annualized_ratio(ls_series.fillna(0))

    if "long_turnover" in turnover:
        summary["turnover_long_mean"] = float(turnover["long_turnover"].mean())
    if "short_turnover" in turnover:
        summary["turnover_short_mean"] = float(turnover["short_turnover"].mean())

    summary["coverage_mean"] = float(coverage_series.mean()) if len(coverage_series) else np.nan
    return summary


# ======================================================================================
# Main reporting function
# ======================================================================================

def generate_factor_report(
    factor_name: str,
    prices: pd.DataFrame,
    raw_factor_panel: pd.DataFrame,
    out_dir: str,
    horizons: List[int] = None,
    shift_factor_by: int = 1,
    n_quantiles: int = 5,
    normalize_cfg: Optional[Dict[str, bool]] = None,
    compute_corr_with: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Union[str, Dict[str, Union[str, float]]]]:
    """
    Generate a comprehensive factor quality report with Plotly HTML outputs and JSON summary.

    Args:
        factor_name:      Name of the factor (for filenames/report titles).
        prices:           Price panel (date x asset) used for future returns.
        raw_factor_panel: Raw factor panel (date x asset).
        out_dir:          Directory to store generated files.
        horizons:         List of horizons for IC decay analysis (default [1,5,10,20]).
        shift_factor_by:  Positive integer to shift factor forward (enforce T-1 info).
        n_quantiles:      Number of quantile buckets for analysis.
        normalize_cfg:    Dict controlling winsor/rank/zscore (default winsor+rank).
        compute_corr_with:Optional dict of other factor panels for correlation analysis.
    """

    os.makedirs(out_dir, exist_ok=True)
    horizons = horizons or [1, 5, 10, 20]
    normalize_cfg = normalize_cfg or {"do_winsor": True, "do_rank": True, "do_zscore": False}

    prices_aligned, factor_panel = align_panels(prices, raw_factor_panel, shift_factor_by=shift_factor_by)
    factor_panel = normalize_pipeline(factor_panel, **normalize_cfg)

    main_horizon = horizons[-1]
    forward_returns = future_returns(prices_aligned, main_horizon)

    ic_series = daily_ic(factor_panel, forward_returns, method="spearman")
    ic_summary = ic_stats(ic_series)
    ic_decay_values = ic_decay(factor_panel, prices_aligned, horizons=horizons, method="spearman")

    quantile_ret = quantile_returns(factor_panel, forward_returns, n_quantiles=n_quantiles)
    ls_series = long_short_series(factor_panel, forward_returns, q=1.0 / n_quantiles)
    turnover_series = turnover_rate(factor_panel, q=1.0 / n_quantiles)
    coverage_series = coverage(factor_panel)

    paths: Dict[str, str] = {}
    paths["ic_series"] = plot_ic_series(ic_series, os.path.join(out_dir, f"{factor_name}_ic.html"))
    paths["ic_hist"] = plot_ic_hist(ic_series, os.path.join(out_dir, f"{factor_name}_ic_hist.html"))
    paths["ic_decay"] = plot_ic_decay_bar(ic_decay_values, os.path.join(out_dir, f"{factor_name}_ic_decay.html"))
    paths["quantile"] = plot_quantile_returns(quantile_ret, os.path.join(out_dir, f"{factor_name}_quantiles.html"))
    paths["long_short"] = plot_long_short(ls_series, os.path.join(out_dir, f"{factor_name}_ls.html"))
    paths["turnover"] = plot_turnover(turnover_series, os.path.join(out_dir, f"{factor_name}_turnover.html"))
    paths["coverage"] = plot_coverage(coverage_series, os.path.join(out_dir, f"{factor_name}_coverage.html"))

    corr_df = None
    centrality = None
    if compute_corr_with:
        combined = {"__SELF__": factor_panel.copy()}
        combined.update({name: normalize_pipeline(panel) for name, panel in compute_corr_with.items()})
        corr_df = factor_correlation(combined)
        paths["corr_heatmap"] = plot_factor_corr_heatmap(corr_df, os.path.join(out_dir, f"{factor_name}_corr_heatmap.html"))
        centrality = factor_centrality_from_corr(corr_df) if corr_df is not None else None

    summary = summarize_for_layer2(factor_name, ic_series, ls_series, ic_decay_values, turnover_series, coverage_series)

    if corr_df is not None:
        values = corr_df.loc["__SELF__", corr_df.columns != "__SELF__"].values
        summary["avg_abs_corr_to_others"] = float(np.nanmean(np.abs(values))) if len(values) else np.nan
    if centrality is not None:
        summary["corr_network_centrality"] = float(centrality.get("__SELF__", np.nan))

    summary_path = os.path.join(out_dir, f"{factor_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    # Optional index file with links and summary snippet
    index_path = os.path.join(out_dir, f"{factor_name}_index.html")
    with open(index_path, "w", encoding="utf-8") as file:
        file.write("<html><head><meta charset='utf-8'><title>Factor Report</title></head><body>")
        file.write(f"<h2>Factor Report: {factor_name}</h2><ul>")
        for key, path in paths.items():
            file.write(f"<li><a href='{os.path.basename(path)}' target='_blank'>{key}</a></li>")
        file.write("</ul><pre>")
        file.write(json.dumps(summary, indent=2))
        file.write("</pre></body></html>")

    return {
        "summary": summary,
        "summary_json": summary_path,
        "plots": paths,
        "index_html": index_path,
    }


__all__ = [
    "align_panels",
    "normalize_pipeline",
    "winsorize_by_date",
    "rank_pct_by_date",
    "zscore_by_date",
    "future_returns",
    "excess_over_mean_label",
    "daily_ic",
    "ic_stats",
    "ic_decay",
    "rolling_ic",
    "quantile_groups",
    "quantile_returns",
    "long_short_series",
    "turnover_rate",
    "coverage",
    "factor_correlation",
    "factor_centrality_from_corr",
    "plot_ic_series",
    "plot_ic_hist",
    "plot_ic_decay_bar",
    "plot_quantile_returns",
    "plot_long_short",
    "plot_turnover",
    "plot_coverage",
    "plot_factor_corr_heatmap",
    "summarize_for_layer2",
    "generate_factor_report",
]

