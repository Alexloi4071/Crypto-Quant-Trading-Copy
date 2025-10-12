#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
專供 CI 使用的 Layer0→Layer2 一鍵流程。
- 自動執行資料清洗、標籤優化、特徵優化
- trials 數量可由環境變數指定
- 最後輸出 Layer2 物化檔案與 JSON 結果
"""

import json
import os
from pathlib import Path

from optuna_system.coordinator import OptunaCoordinator
from config.timeframe_scaler import MultiTimeframeCoordinator

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data"


def get_trial_from_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except Exception:
        return default


def main() -> None:
    symbol = os.getenv("PIPELINE_SYMBOL", "BTCUSDT")
    timeframe = os.getenv("PIPELINE_TIMEFRAME", "15m")
    l0_trials = get_trial_from_env("L0_TRIALS", 50)
    l1_trials = get_trial_from_env("L1_TRIALS", 150)
    l2_trials = get_trial_from_env("L2_TRIALS", 250)

    print(f"🚀 全流程啟動: {symbol}_{timeframe}")
    print(f"  ⚙️ L0 trials={l0_trials}, L1 trials={l1_trials}, L2 trials={l2_trials}")

    multi_scaler = MultiTimeframeCoordinator(symbol=symbol, data_path=str(DATA_PATH))
    scaled_config = multi_scaler.get_scaled_config_for_timeframe(timeframe)

    coordinator = OptunaCoordinator(
        symbol=symbol,
        timeframe=timeframe,
        data_path=str(DATA_PATH),
        scaled_config=scaled_config,
    )

    print("\n--- Layer0: Data Cleaning ---")
    l0_result = coordinator.run_layer0_data_cleaning(n_trials=l0_trials)
    print(l0_result)

    print("\n--- Layer1: Label Optimization ---")
    l1_result = coordinator.run_layer1_label_optimization(n_trials=l1_trials)
    print(l1_result)

    print("\n--- Layer2: Feature Optimization ---")
    l2_result = coordinator.run_layer2_feature_optimization(n_trials=l2_trials)
    print(l2_result)

    if isinstance(l2_result, dict) and 'best_score' in l2_result:
        output_dir = PROJECT_ROOT / 'optuna_system' / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'feature_pipeline_{symbol}_{timeframe}.json'
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(l2_result, f, indent=2, ensure_ascii=False)
        print(f"💾 Layer2 結果已保存: {output_file}")

    print("\n✅ Layer0→Layer2 全流程完成")


if __name__ == "__main__":
    main()
