# 修復 optuna_feature.py 中的關鍵問題

## 1. 修復重採樣方法
```python
def resample_ohlcv(self, ohlcv: pd.DataFrame, rule) -> pd.DataFrame:
    """修復版本 - 正確提取頻率字符串並處理多時框配置"""
    agg = {
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    try:
        # 修復：正確提取頻率字符串
        if isinstance(rule, dict):
            frequency = rule.get('rule', '1H')  # 從字典中提取頻率
        else:
            frequency = str(rule)  # 如果已經是字符串則直接使用
            
        # 驗證頻率格式
        valid_frequencies = ['15T', '1H', '4H', '1D', '15m', '1h', '4h', '1d']
        if frequency not in valid_frequencies:
            self.logger.warning(f"⚠️ 無效頻率 {frequency}，使用默認 1H")
            frequency = '1H'
            
        resampled = ohlcv.resample(frequency).agg(agg)
        self.logger.info(f"✅ 成功重採樣到 {frequency}: {resampled.shape}")
        return resampled.dropna()
        
    except Exception as e:
        self.logger.warning(f"⚠️ 重採樣失敗 rule={rule}: {e}")
        return pd.DataFrame(columns=ohlcv.columns)
```

## 2. 修復多時框特徵生成邏輯
```python
def generate_technical_features(self, ohlcv_data: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """修復版本 - 正確處理多時框特徵生成"""
    flags_tech = self.flags.get('tech', {})
    base_features = self._calc_base_indicators(ohlcv_data, self.timeframe, flags_tech)
    features = base_features.copy()
    
    # 多時框配置
    mtf_cfg = flags_tech.get('multi_timeframes', {})
    if isinstance(mtf_cfg, dict) and mtf_cfg.get('enabled'):
        rules = mtf_cfg.get('rules', {})
        tf_list = mtf_cfg.get('timeframes', [])
        
        for tf_key in tf_list:
            rule_config = rules.get(tf_key)
            if not rule_config:
                continue
                
            # 修復：正確提取頻率字符串 
            frequency = rule_config.get('rule') if isinstance(rule_config, dict) else str(rule_config)
            
            resampled = self.resample_ohlcv(ohlcv_data, frequency)  # 傳遞字符串而不是字典
            if resampled.empty:
                continue
                
            # 使用特定時框的參數生成指標
            tf_features = self._calc_base_indicators(
                resampled, tf_key, flags_tech, 
                base_index=ohlcv_data.index,
                tf_overrides=rule_config  # 傳遞時框特定配置
            )
            
            features = self.safe_merge(features, tf_features, prefix=f'{tf_key}_')
            
    return features
```

## 3. 調整 Layer1 聯動參數範圍
```python
def objective(self, trial) -> float:
    """修復版本 - 擴大 Layer1 聯動搜索範圍"""
    # ... 前面代碼不變 ...
    
    # 修復：擴大 Layer1 聯動搜索範圍
    if layer1_params:
        layer1_lag = layer1_params.get('lag', 15)
        layer1_buy_q = layer1_params.get('buy_quantile', 0.7)
        layer1_sell_q = layer1_params.get('sell_quantile', 0.3)
        
        # 擴大搜索範圍，增加靈活性
        lag_range = max(3, layer1_lag - 5), min(30, layer1_lag + 5)  # 從 ±3 擴大到 ±5
        profit_q_range = max(0.55, layer1_buy_q - 0.1), min(0.95, layer1_buy_q + 0.1)  # 從 ±0.05 擴大到 ±0.1
        loss_q_range = max(0.05, layer1_sell_q - 0.1), min(0.45, layer1_sell_q + 0.1)
        
        lag = trial.suggest_int('lag', *lag_range)
        profit_quantile = trial.suggest_float('profit_quantile', *profit_q_range)
        loss_quantile = trial.suggest_float('loss_quantile', *loss_q_range)
        
        self.logger.info(f"🔗 Layer1聯動擴大範圍: lag={layer1_lag} → 搜索範圍{lag_range}")
        self.logger.info(f"🔗 Layer1聯動擴大範圍: buy_q={layer1_buy_q:.3f} → profit_q{profit_q_range}")
        self.logger.info(f"🔗 Layer1聯動擴大範圍: sell_q={layer1_sell_q:.3f} → loss_q{loss_q_range}")
    else:
        # 默認範圍保持不變
        lag = trial.suggest_int('lag', 3, 20)
        profit_quantile = trial.suggest_float('profit_quantile', 0.6, 0.9)
        loss_quantile = trial.suggest_float('loss_quantile', 0.1, 0.4)
    
    # ... 後續代碼不變 ...
```

## 4. 增加特徵質量過濾
```python
def nested_cv_evaluation(self, X: pd.DataFrame, y: pd.Series, coarse_k: int, fine_k: int) -> Dict:
    """修復版本 - 增加特徵質量過濾"""
    # ... 前面代碼不變 ...
    
    # 在特徵選擇前增加質量過濾
    X_filtered = self._filter_low_quality_features(X)
    self.logger.info(f"🔧 特徵質量過濾: {X.shape[1]} → {X_filtered.shape[1]}")
    
    # 後續使用 X_filtered 進行特徵選擇
    # ... 剩餘代碼使用 X_filtered 替代 X
    
def _filter_low_quality_features(self, X: pd.DataFrame) -> pd.DataFrame:
    """過濾低質量特徵"""
    # 1. 移除常量特徵
    constant_features = X.columns[X.nunique() <= 1]
    X = X.drop(columns=constant_features)
    
    # 2. 移除高缺失率特徵 (>50%)
    missing_rate = X.isnull().mean()
    high_missing_features = missing_rate[missing_rate > 0.5].index
    X = X.drop(columns=high_missing_features)
    
    # 3. 移除高相關性特徵 (>0.95)
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_features = [
        column for column in upper_triangle.columns 
        if any(upper_triangle[column] > 0.95)
    ]
    X = X.drop(columns=high_corr_features)
    
    self.logger.info(f"📊 過濾統計: 常量={len(constant_features)}, 高缺失={len(high_missing_features)}, 高相關={len(high_corr_features)}")
    
    return X
```

## 5. 優化標籤平衡
```python
def generate_rolling_labels(self, returns: pd.Series, lag: int, profit_quantile: float, 
                          loss_quantile: float, lookback_window: int) -> pd.Series:
    """修復版本 - 優化標籤平衡"""
    # ... 前面代碼不變 ...
    
    # 檢查標籤分布並調整
    label_counts = labels.value_counts().sort_index()
    total = len(labels.dropna())
    
    # 計算類別比例
    if len(label_counts) == 3:
        ratios = [count/total for count in label_counts.values]
        min_ratio = min(ratios)
        
        # 如果最小類別比例 < 25%，調整 quantile
        if min_ratio < 0.25:
            self.logger.warning(f"⚠️ 標籤不平衡嚴重: {dict(label_counts)}, 最小比例={min_ratio:.3f}")
            
            # 動態調整 quantile 以改善平衡
            if ratios[2] < 0.25:  # 類別2偏少，降低 profit_quantile
                profit_quantile = max(0.65, profit_quantile - 0.05)
            if ratios[0] < 0.25:  # 類別0偏少，提高 loss_quantile  
                loss_quantile = min(0.35, loss_quantile + 0.05)
            
            self.logger.info(f"📊 調整後 quantile: profit={profit_quantile:.3f}, loss={loss_quantile:.3f}")
            
            # 重新生成標籤
            return self.generate_rolling_labels(returns, lag, profit_quantile, loss_quantile, lookback_window)
    
    self.logger.info(f"📊 標籤分布: {dict(label_counts)} (總計{total})")
    return labels.dropna().astype(int)
```

## 修復總結
1. **重採樣問題**：修復頻率字符串提取邏輯
2. **Layer1 聯動**：擴大搜索範圍，增加靈活性
3. **特徵質量**：增加過濾機制，提升特徵有效性
4. **標籤平衡**：動態調整 quantile，改善類別分布
5. **錯誤處理**：增強異常處理和日誌記錄

這些修復應該能顯著改善 Layer2 的優化效果，預期 F1 分數從 0.24 提升到 0.4+ 。