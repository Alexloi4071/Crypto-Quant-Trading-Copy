# 修复版标签生成 - 避免未来数据泄露

def _generate_labels_fixed(self, price_series: pd.Series, lag: int = 12,
                          profit_quantile: float = 0.85, loss_quantile: float = 0.15,
                          lookback_window: int = 500) -> pd.Series:
    """
    修复版标籤生成 - 滾動窗口分位數，嚴格避免未來數據洩露
    """
    # 計算未來收益
    future_prices = price_series.shift(-lag)
    returns = (future_prices - price_series) / price_series
    
    # 初始化標籤
    labels = pd.Series(1, index=price_series.index, dtype=int)  # 默認持有
    
    # 滾動窗口計算分位數閾值
    for i in range(lookback_window, len(returns) - lag):
        # 只使用過去數據計算分位數
        start_idx = max(0, i - lookback_window)
        historical_returns = returns.iloc[start_idx:i]
        
        if len(historical_returns.dropna()) < 100:  # 確保有足夠歷史數據
            continue
            
        # 基於歷史數據計算動態閾值
        upper_threshold = historical_returns.quantile(profit_quantile)
        lower_threshold = historical_returns.quantile(loss_quantile)
        
        # 應用閾值到當前時點
        current_return = returns.iloc[i]
        if pd.notna(current_return):
            if current_return > upper_threshold:
                labels.iloc[i] = 2  # 買入
            elif current_return < lower_threshold:
                labels.iloc[i] = 0  # 賣出
    
    # 移除未來數據
    labels = labels[:-lag] if lag > 0 else labels
    
    # 統計信息
    label_counts = labels.value_counts()
    total = len(labels)
    distribution = {k: v/total for k, v in label_counts.items()}
    
    print(f"修復版標籤分佈: {distribution}")
    print(f"滾動窗口: {lookback_window}, lag: {lag}")
    
    return labels.dropna().astype(int)


# 修復版特徵選擇 - 嵌套交叉驗證

def objective_fixed(self, trial):
    """
    修復版目標函數 - 嵌套交叉驗證避免數據洩露
    """
    # 參數
    lag = trial.suggest_int('lag', 8, 24)
    profit_quantile = trial.suggest_float('profit_quantile', 0.75, 0.90)
    loss_quantile = trial.suggest_float('loss_quantile', 0.10, 0.25)
    lookback_window = trial.suggest_int('lookback_window', 300, 800)
    
    # 修復版標籤生成
    labels = self._generate_labels_fixed(
        self.ohlcv_data['close'],
        lag=lag,
        profit_quantile=profit_quantile,
        loss_quantile=loss_quantile,
        lookback_window=lookback_window
    )
    
    # 數據對齊
    common_idx = self.features.index.intersection(labels.index)
    X = self.features.loc[common_idx].fillna(0)
    y = labels.loc[common_idx]
    
    if len(X) < 1000:
        return 0.0
    
    # 🚀 嵌套交叉驗證 - 在每個fold內部進行特徵選擇
    outer_cv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    
    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx] 
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 內部特徵選擇（只在訓練集上）
        n_features = len(X_train.columns)
        coarse_k = trial.suggest_int('coarse_k', int(n_features * 0.2), int(n_features * 0.6))
        fine_k = trial.suggest_int('fine_k', 10, min(30, coarse_k // 2))
        
        # 方差篩選
        var_selector = VarianceThreshold(threshold=0.01)
        X_train_var = var_selector.fit_transform(X_train)
        X_test_var = var_selector.transform(X_test)
        
        if X_train_var.shape[1] == 0:
            cv_scores.append(0.0)
            continue
        
        # 統計顯著性選擇（只在訓練集上fit）
        coarse_k = min(coarse_k, X_train_var.shape[1])
        coarse_selector = SelectKBest(f_classif, k=coarse_k)
        X_train_coarse = coarse_selector.fit_transform(X_train_var, y_train)
        X_test_coarse = coarse_selector.transform(X_test_var)
        
        # 互信息選擇
        fine_k = min(fine_k, X_train_coarse.shape[1])
        fine_selector = SelectKBest(mutual_info_classif, k=fine_k)
        X_train_final = fine_selector.fit_transform(X_train_coarse, y_train)
        X_test_final = fine_selector.transform(X_test_coarse)
        
        # 模型訓練與評估
        model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        
        # F1分數
        fold_score = f1_score(y_test, y_pred, average='weighted')
        cv_scores.append(fold_score)
    
    final_score = np.mean(cv_scores)
    
    # 檢查分數合理性
    if final_score > 0.8:  # 可疑的高分數
        print(f"⚠️ 可疑高分數 {final_score:.4f}，請檢查數據洩露")
        
    return final_score