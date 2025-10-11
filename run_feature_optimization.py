import os
import json
import pandas as pd
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# 引入你的 feature pipeline 和 model trainer
from feature_pipeline import load_features, evaluate_model

def objective(trial: Trial):
    # 定義你要優化的超參數範圍
    params = {
        'feature_selection_method': trial.suggest_categorical('feature_selection_method', ['mutual_info', 'f_classif']),
        'noise_reduction': trial.suggest_categorical('noise_reduction', [True, False]),
        'feature_interaction': trial.suggest_categorical('feature_interaction', [True, False]),
        'lag': trial.suggest_int('lag', 1, 20),
        'profit_quantile': trial.suggest_float('profit_quantile', 0.5, 1.0),
        'loss_quantile': trial.suggest_float('loss_quantile', 0.0, 0.5),
        'lookback_window': trial.suggest_int('lookback_window', 100, 1000),
        'coarse_k': trial.suggest_int('coarse_k', 10, 200),
        'fine_ratio': trial.suggest_float('fine_ratio', 0.1, 1.0),
        'stability_threshold': trial.suggest_float('stability_threshold', 0.1, 1.0),
        'correlation_threshold': trial.suggest_float('correlation_threshold', 0.1, 1.0),
        'model_type': trial.suggest_categorical('model_type', ['xgboost', 'extra_trees']),
    }

    # 依 model_type 設定模型相關參數
    if params['model_type'] == 'xgboost':
        params.update({
            'xgb_n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
            'xgb_learning_rate': trial.suggest_float('xgb_learning_rate', 0.001, 0.1),
            'xgb_max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
            'xgb_subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
            'xgb_colsample': trial.suggest_float('xgb_colsample', 0.5, 1.0),
            'xgb_min_child_weight': trial.suggest_float('xgb_min_child_weight', 1, 10),
            'xgb_reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.1, 10.0),
            'xgb_reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.0, 1.0),
        })
    else:
        params.update({
            'et_n_estimators': trial.suggest_int('et_n_estimators', 50, 300),
            'et_max_depth': trial.suggest_int('et_max_depth', 3, 20),
            'et_min_samples_split': trial.suggest_int('et_min_samples_split', 2, 20),
            'et_min_samples_leaf': trial.suggest_int('et_min_samples_leaf', 1, 20),
            'et_max_features': trial.suggest_float('et_max_features', 0.1, 1.0),
        })

    # 讀取前置特徵
    df = load_features(
        symbol='BTCUSDT',
        timeframe='15m',
        version='v30'
    )

    # 評估模型，返回 CV 分數 dict
    metrics = evaluate_model(df, params)

    # Optuna objective 只需要一個 scalar
    return metrics['cv_scores_mean']


def main():
    # 設定 Optuna 存儲
    study = optuna.create_study(
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_warmup_steps=5),
        direction='maximize'
    )
    # 確保跑足 250 次試驗
    study.optimize(objective, n_trials=250)

    # 取得最佳結果
    best = study.best_trial
    result = {
        'best_value': best.value,
        'best_params': best.params,
        'n_trials': best.number,
        'state': str(best.state),
        'cv_scores': best.user_attrs.get('cv_scores', []),
        'cv_scores_mean': best.user_attrs.get('cv_scores_mean', None),
        'cv_scores_std': best.user_attrs.get('cv_scores_std', None)
    }

    # 避免序列化 DataFrame：將所有 DataFrame 轉為 dict
    for k, v in list(result.items()):
        if isinstance(v, pd.DataFrame):
            result[k] = v.to_dict(orient='records')

    # 確保目錄存在
    os.makedirs('optuna_results', exist_ok=True)
    output_json = os.path.join('optuna_results', 'layer2_optimization_result.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Layer2 優化完成，結果保存在 {output_json}")


if __name__ == '__main__':
    main()
    for vdir in version_dirs:
        selected = vdir / 'BTCUSDT_15m_selected_features.parquet'
        if selected.exists():
            return vdir.name, str(selected)

        # fallback: 任一 features_*.parquet
        candidates = sorted(vdir.glob('features_BTCUSDT_15m_*.parquet'))
        if candidates:
            return vdir.name, str(candidates[0])

    return None, None


def main():
    print("🚀 開始 BTCUSDT_15m 特徵選擇超參數優化...")

    # 自動偵測最新版本特徵檔案
    version, feature_file = find_latest_feature_artifact()
    if not version or not feature_file:
        print("❌ 找不到任何可用的特徵檔 (data/processed/features/BTCUSDT_15m/v*/)")
        sys.exit(1)

    print(f"✅ 找到特徵文件: {feature_file} (version={version})")

    # 從零生成流程：先 L0→L1，最後執行 L2
    try:
        n_trials = int(os.getenv("L2_TRIALS", "250"))
    except Exception:
        n_trials = 250
    

    coordinator = OptunaCoordinator(
        symbol="BTCUSDT",
        timeframe="15m",
        data_path="data",
    )

    print("🔧 先執行 Layer0 數據清洗與物化…")
    coordinator.run_layer0_data_cleaning(n_trials=max(10, 15))

    print("🏷️ 接著執行 Layer1 標籤優化與物化…")
    coordinator.run_layer1_label_optimization(n_trials=max(50, 75))

    # 最後執行 L2 特徵優化（使用前兩層物化結果作為輸入）
    print("📊 執行第2層：特徵工程參數優化…")
    result = coordinator.run_layer2_feature_optimization(n_trials=n_trials)

    if 'error' in result:
        print(f"❌ 優化失敗: {result['error']}")
    else:
        print(f"✅ 優化完成! 最佳得分: {result.get('best_score', 'N/A')}")
        print(f"📊 最優參數: {result.get('best_params', 'N/A')}")

        # 保存結果
        result_file = f"optuna_system/results/feature_optimization_BTCUSDT_15m.json"
        os.makedirs("optuna_system/results", exist_ok=True)

        import json
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"💾 結果已保存至: {result_file}")


if __name__ == "__main__":
    main()
