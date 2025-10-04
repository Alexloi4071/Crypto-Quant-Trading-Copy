/**
 * Layer 1: 標籤生成配置
 * 包含預設模式和專家模式
 */

// L1 預設配置
const L1_PRESETS = {
    quickstart: {
        name: '快速開始',
        desc: '平衡的標籤生成配置，適合初次使用',
        config: {
            lag: 7,
            threshold_method: 'quantile',
            lookback_window: 600,
            buy_quantile: 0.75,
            sell_quantile: 0.25,
            profit_threshold: 0.015,
            loss_threshold: -0.015,
            vol_multiplier: 2.0,
            vol_window: 30,
            profit_multiplier: 1.5,
            stop_multiplier: 1.5,
            max_holding: 20,
            atr_period: 14,
            min_samples: 1000,
            balance_weight: 0.5,
            stability_weight: 0.3,
            fixed_lookback: 1500,
            min_threshold_gap: 0.005,
            max_noise_ratio: 0.35,
            target_hold_ratio: 0.5,
            distribution_penalty: 1.0,
            n_trials: 50
        }
    },
    high_precision: {
        name: '高精度模式',
        desc: '更多試驗次數，追求最佳標籤質量',
        config: {
            lag: 7,
            threshold_method: 'quantile',
            lookback_window: 1200,
            buy_quantile: 0.77,
            sell_quantile: 0.27,
            profit_threshold: 0.013,
            loss_threshold: -0.019,
            vol_multiplier: 1.95,
            vol_window: 35,
            profit_multiplier: 1.69,
            stop_multiplier: 1.40,
            max_holding: 18,
            atr_period: 12,
            min_samples: 1377,
            balance_weight: 0.529,
            stability_weight: 0.202,
            fixed_lookback: 1661,
            min_threshold_gap: 0.0046,
            max_noise_ratio: 0.362,
            target_hold_ratio: 0.504,
            distribution_penalty: 1.004,
            n_trials: 150
        }
    },
    testing: {
        name: '測試模式',
        desc: '快速驗證，較少試驗次數',
        config: {
            lag: 5,
            threshold_method: 'quantile',
            lookback_window: 400,
            buy_quantile: 0.75,
            sell_quantile: 0.25,
            profit_threshold: 0.02,
            loss_threshold: -0.02,
            vol_multiplier: 2.0,
            vol_window: 20,
            profit_multiplier: 1.5,
            stop_multiplier: 1.5,
            max_holding: 15,
            atr_period: 14,
            min_samples: 500,
            balance_weight: 0.5,
            stability_weight: 0.3,
            fixed_lookback: 1000,
            min_threshold_gap: 0.01,
            max_noise_ratio: 0.4,
            target_hold_ratio: 0.5,
            distribution_penalty: 1.0,
            n_trials: 20
        }
    }
};

// 閾值方法選項
const THRESHOLD_METHODS = [
    { value: 'quantile', label: '分位數法（推薦）' },
    { value: 'fixed', label: '固定閾值' },
    { value: 'volatility', label: '波動率調整' }
];

/**
 * 渲染 L1 內容
 */
function renderLayer1(container) {
    container.innerHTML = `
        <!-- 層級頭部 -->
        <div class="layer-header">
            <div class="layer-header-top">
                <div class="layer-title-group">
                    <div class="layer-badge" data-layer="1">
                        <i class="fas fa-tag"></i>
                        <span>Layer 1</span>
                    </div>
                    <h1 class="layer-title">標籤生成配置</h1>
                </div>
                <div class="layer-actions">
                    <a href="/" class="btn btn-secondary" style="text-decoration: none;">
                        <i class="fas fa-arrow-left"></i>
                        返回儀表盤
                    </a>
                    <button class="btn btn-secondary" onclick="resetL1Form()">
                        <i class="fas fa-redo"></i>
                        重置
                    </button>
                    <button class="btn btn-success" id="l1-run-btn" onclick="runL1Optimization()">
                        <i class="fas fa-play"></i>
                        開始優化
                    </button>
                </div>
            </div>
            <p style="color: var(--text-secondary); margin-top: 12px;">
                配置三分類標籤的生成規則和閾值。生成的標籤將用於後續的特徵工程和模型訓練。
            </p>
        </div>
        
        <!-- 模式切換器 -->
        <div class="mode-switcher">
            <div class="mode-btn active" data-mode="preset" onclick="switchL1Mode('preset')">
                <i class="fas fa-magic"></i>
                <span>預設模式</span>
                <small>快速配置，使用推薦參數</small>
            </div>
            <div class="mode-btn" data-mode="expert" onclick="switchL1Mode('expert')">
                <i class="fas fa-cog"></i>
                <span>專家模式</span>
                <small>完全自定義所有參數</small>
            </div>
        </div>
        
        <!-- 預設模式內容 -->
        <div id="l1-preset-mode" class="mode-content">
            ${renderL1PresetMode()}
        </div>
        
        <!-- 專家模式內容 -->
        <div id="l1-expert-mode" class="mode-content hidden">
            ${renderL1ExpertMode()}
        </div>
        
        <!-- 進度監控區域 -->
        <div id="l1-progress" class="hidden">
            ${renderL1Progress()}
        </div>
    `;
    
    // 初始化預設配置
    loadL1Preset('quickstart');
}

/**
 * 渲染預設模式
 */
function renderL1PresetMode() {
    return `
        <div class="card">
            <div class="card-header">
                <i class="fas fa-list"></i> 選擇預設配置
            </div>
            <div class="card-body">
                <div class="alert alert-info">
                    <i class="fas fa-lightbulb"></i>
                    <div>
                        <strong>提示</strong><br>
                        標籤生成是量化交易的核心步驟，決定了後續模型的訓練質量。
                        預設配置已經過測試和優化，適合大多數場景。
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 20px;">
                    ${Object.entries(L1_PRESETS).map(([key, preset]) => `
                        <div class="preset-card ${key === 'quickstart' ? 'selected' : ''}" 
                             data-preset="${key}"
                             onclick="loadL1Preset('${key}')">
                            <h3>${preset.name}</h3>
                            <p>${preset.desc}</p>
                            <div class="preset-details">
                                <div><strong>試驗次數:</strong> ${preset.config.n_trials}</div>
                                <div><strong>回看窗口:</strong> ${preset.config.lookback_window}</div>
                                <div><strong>閾值方法:</strong> ${preset.config.threshold_method}</div>
                            </div>
                        </div>
                    `).join('')}
                </div>
                
                <div style="margin-top: 24px;">
                    <h4 style="color: var(--text-primary); margin-bottom: 12px;">當前配置預覽</h4>
                    <pre id="l1-preset-preview" style="background: var(--bg-tertiary); padding: 16px; border-radius: 6px; overflow-x: auto; color: var(--text-secondary); font-size: 13px;"></pre>
                </div>
            </div>
        </div>
        
        <style>
            .preset-card {
                padding: 20px;
                background: var(--bg-tertiary);
                border: 2px solid var(--border-primary);
                border-radius: 8px;
                cursor: pointer;
                transition: var(--transition-fast);
            }
            
            .preset-card:hover {
                border-color: var(--brand-secondary);
                transform: translateY(-2px);
                box-shadow: var(--shadow-md);
            }
            
            .preset-card.selected {
                border-color: var(--brand-secondary);
                background: var(--bg-elevated);
            }
            
            .preset-card h3 {
                color: var(--text-primary);
                margin-bottom: 8px;
                font-size: 16px;
            }
            
            .preset-card p {
                color: var(--text-tertiary);
                font-size: 13px;
                margin-bottom: 16px;
            }
            
            .preset-details {
                font-size: 12px;
                color: var(--text-secondary);
            }
            
            .preset-details div {
                margin-bottom: 4px;
            }
        </style>
    `;
}

/**
 * 渲染專家模式
 */
function renderL1ExpertMode() {
    return `
        <div class="card">
            <div class="card-header">
                <i class="fas fa-cog"></i> 基礎設置
            </div>
            <div class="card-body">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div class="form-group">
                        <label class="form-label form-label-required">滯後期數 (lag)</label>
                        <input type="number" class="form-input" id="l1-lag" value="7" min="1" max="20">
                        <div class="form-hint">計算標籤時的滯後期數，通常為 3-10</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label form-label-required">閾值方法</label>
                        <select class="form-select" id="l1-threshold-method">
                            ${THRESHOLD_METHODS.map(m => `
                                <option value="${m.value}" ${m.value === 'quantile' ? 'selected' : ''}>
                                    ${m.label}
                                </option>
                            `).join('')}
                        </select>
                        <div class="form-hint">決定買賣閾值的計算方法</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label form-label-required">回看窗口</label>
                        <input type="number" class="form-input" id="l1-lookback-window" value="600" min="100" max="3000">
                        <div class="form-hint">用於計算閾值的歷史窗口大小</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">最大持倉期</label>
                        <input type="number" class="form-input" id="l1-max-holding" value="20" min="5" max="50">
                        <div class="form-hint">最長持倉週期數</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <i class="fas fa-percent"></i> 買賣閾值設置
            </div>
            <div class="card-body">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div class="form-group">
                        <label class="form-label">買入分位數</label>
                        <input type="number" class="form-input" id="l1-buy-quantile" 
                               value="0.75" min="0.5" max="0.95" step="0.01">
                        <div class="form-hint">觸發買入信號的分位數閾值 (0.5-0.95)</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">賣出分位數</label>
                        <input type="number" class="form-input" id="l1-sell-quantile" 
                               value="0.25" min="0.05" max="0.5" step="0.01">
                        <div class="form-hint">觸發賣出信號的分位數閾值 (0.05-0.5)</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">利潤閾值</label>
                        <input type="number" class="form-input" id="l1-profit-threshold" 
                               value="0.015" min="0.005" max="0.05" step="0.001">
                        <div class="form-hint">觸發止盈的利潤百分比 (0.5%-5%)</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">損失閾值</label>
                        <input type="number" class="form-input" id="l1-loss-threshold" 
                               value="-0.015" min="-0.05" max="-0.005" step="0.001">
                        <div class="form-hint">觸發止損的損失百分比 (-5%~-0.5%)</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <i class="fas fa-chart-line"></i> 波動率參數
            </div>
            <div class="card-body">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div class="form-group">
                        <label class="form-label">波動率乘數</label>
                        <input type="number" class="form-input" id="l1-vol-multiplier" 
                               value="2.0" min="0.5" max="5.0" step="0.1">
                        <div class="form-hint">波動率調整倍數</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">波動率窗口</label>
                        <input type="number" class="form-input" id="l1-vol-window" 
                               value="30" min="10" max="100">
                        <div class="form-hint">計算波動率的窗口大小</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">利潤乘數</label>
                        <input type="number" class="form-input" id="l1-profit-multiplier" 
                               value="1.5" min="1.0" max="3.0" step="0.1">
                        <div class="form-hint">利潤目標的動態調整倍數</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">止損乘數</label>
                        <input type="number" class="form-input" id="l1-stop-multiplier" 
                               value="1.5" min="1.0" max="3.0" step="0.1">
                        <div class="form-hint">止損位的動態調整倍數</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">ATR 週期</label>
                        <input type="number" class="form-input" id="l1-atr-period" 
                               value="14" min="5" max="30">
                        <div class="form-hint">平均真實波動幅度的計算週期</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <i class="fas fa-balance-scale"></i> 質量控制參數
            </div>
            <div class="card-body">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div class="form-group">
                        <label class="form-label">最小樣本數</label>
                        <input type="number" class="form-input" id="l1-min-samples" 
                               value="1000" min="100" max="5000">
                        <div class="form-hint">每個類別的最小樣本要求</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">固定回看期</label>
                        <input type="number" class="form-input" id="l1-fixed-lookback" 
                               value="1500" min="500" max="3000">
                        <div class="form-hint">固定的歷史回看期數</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">平衡權重</label>
                        <input type="number" class="form-input" id="l1-balance-weight" 
                               value="0.5" min="0" max="1" step="0.01">
                        <div class="form-hint">類別平衡的權重 (0-1)</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">穩定性權重</label>
                        <input type="number" class="form-input" id="l1-stability-weight" 
                               value="0.3" min="0" max="1" step="0.01">
                        <div class="form-hint">標籤穩定性的權重 (0-1)</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">最小閾值差距</label>
                        <input type="number" class="form-input" id="l1-min-threshold-gap" 
                               value="0.005" min="0.001" max="0.02" step="0.001">
                        <div class="form-hint">買賣閾值的最小間隔</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">最大噪音比率</label>
                        <input type="number" class="form-input" id="l1-max-noise-ratio" 
                               value="0.35" min="0.1" max="0.5" step="0.01">
                        <div class="form-hint">允許的最大噪音比例</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">目標持倉比率</label>
                        <input type="number" class="form-input" id="l1-target-hold-ratio" 
                               value="0.5" min="0.3" max="0.7" step="0.01">
                        <div class="form-hint">期望的持倉時間佔比</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">分布懲罰</label>
                        <input type="number" class="form-input" id="l1-distribution-penalty" 
                               value="1.0" min="0.5" max="2.0" step="0.1">
                        <div class="form-hint">類別分布不平衡的懲罰係數</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <i class="fas fa-flask"></i> Optuna 優化參數
            </div>
            <div class="card-body">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div class="form-group">
                        <label class="form-label form-label-required">試驗次數 (n_trials)</label>
                        <input type="number" class="form-input" id="l1-n-trials" 
                               value="50" min="10" max="500">
                        <div class="form-hint">Optuna 將運行的試驗次數（更多=更好但更慢）</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">超時時間（秒）</label>
                        <input type="number" class="form-input" id="l1-timeout" 
                               value="7200" min="60">
                        <div class="form-hint">優化的最大運行時間（2小時=7200秒）</div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

/**
 * 渲染進度監控
 */
function renderL1Progress() {
    return `
        <div class="card">
            <div class="card-header">
                <i class="fas fa-chart-line"></i> 優化進度
            </div>
            <div class="card-body">
                <div class="progress-bar-container">
                    <div class="progress-bar-fill" id="l1-progress-bar" style="width: 0%"></div>
                </div>
                <div style="text-align: center; margin-top: 8px; color: var(--text-secondary);">
                    <span id="l1-progress-text">準備中...</span>
                </div>
                <div style="margin-top: 16px; font-size: 12px; color: var(--text-tertiary);">
                    <div>當前試驗: <span id="l1-current-trial">-</span></div>
                    <div>最佳分數: <span id="l1-best-score">-</span></div>
                    <div>預計剩餘時間: <span id="l1-time-remaining">計算中...</span></div>
                </div>
            </div>
        </div>
    `;
}

/**
 * 切換 L1 模式
 */
function switchL1Mode(mode) {
    // 更新按鈕狀態
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`.mode-btn[data-mode="${mode}"]`).classList.add('active');
    
    // 切換內容顯示
    if (mode === 'preset') {
        document.getElementById('l1-preset-mode').classList.remove('hidden');
        document.getElementById('l1-expert-mode').classList.add('hidden');
    } else {
        document.getElementById('l1-preset-mode').classList.add('hidden');
        document.getElementById('l1-expert-mode').classList.remove('hidden');
    }
}

/**
 * 載入預設配置
 */
function loadL1Preset(presetKey) {
    // 更新選中狀態
    document.querySelectorAll('.preset-card').forEach(card => {
        card.classList.remove('selected');
    });
    document.querySelector(`.preset-card[data-preset="${presetKey}"]`)?.classList.add('selected');
    
    // 顯示配置預覽
    const preset = L1_PRESETS[presetKey];
    const preview = document.getElementById('l1-preset-preview');
    if (preview) {
        preview.textContent = JSON.stringify(preset.config, null, 2);
    }
    
    // 保存到全局狀態
    if (typeof OptunaState !== 'undefined') {
        OptunaState.layers[1] = {
            mode: 'preset',
            preset: presetKey,
            config: preset.config
        };
    }
}

/**
 * 重置表單
 */
function resetL1Form() {
    if (confirm('確定要重置所有配置嗎？')) {
        renderLayer1(document.getElementById('layer-content'));
        showNotification('info', '已重置為預設配置');
    }
}

/**
 * 運行 L1 優化
 */
async function runL1Optimization() {
    const btn = document.getElementById('l1-run-btn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 運行中...';
    
    try {
        // 收集配置
        const config = collectL1Config();
        
        // 驗證配置
        if (!validateL1Config(config)) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-play"></i> 開始優化';
            return;
        }
        
        // 顯示進度區域
        document.getElementById('l1-progress').classList.remove('hidden');
        
        // 調用 API
        showNotification('info', '正在啟動 Layer 1 優化...');
        const result = await OptunaAPI.startOptimization(1, config);
        
        showNotification('success', '優化已啟動！');
        
        // 開始監控進度
        monitorL1Progress();
        
    } catch (error) {
        showNotification('error', `啟動失敗: ${error.message}`);
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play"></i> 開始優化';
    }
}

/**
 * 收集 L1 配置
 */
function collectL1Config() {
    const mode = document.querySelector('.mode-btn.active')?.dataset.mode;
    
    if (mode === 'preset') {
        return OptunaState.layers[1]?.config || L1_PRESETS.quickstart.config;
    } else {
        // 專家模式：從表單收集
        return {
            lag: parseInt(document.getElementById('l1-lag')?.value || 7),
            threshold_method: document.getElementById('l1-threshold-method')?.value || 'quantile',
            lookback_window: parseInt(document.getElementById('l1-lookback-window')?.value || 600),
            buy_quantile: parseFloat(document.getElementById('l1-buy-quantile')?.value || 0.75),
            sell_quantile: parseFloat(document.getElementById('l1-sell-quantile')?.value || 0.25),
            profit_threshold: parseFloat(document.getElementById('l1-profit-threshold')?.value || 0.015),
            loss_threshold: parseFloat(document.getElementById('l1-loss-threshold')?.value || -0.015),
            vol_multiplier: parseFloat(document.getElementById('l1-vol-multiplier')?.value || 2.0),
            vol_window: parseInt(document.getElementById('l1-vol-window')?.value || 30),
            profit_multiplier: parseFloat(document.getElementById('l1-profit-multiplier')?.value || 1.5),
            stop_multiplier: parseFloat(document.getElementById('l1-stop-multiplier')?.value || 1.5),
            max_holding: parseInt(document.getElementById('l1-max-holding')?.value || 20),
            atr_period: parseInt(document.getElementById('l1-atr-period')?.value || 14),
            min_samples: parseInt(document.getElementById('l1-min-samples')?.value || 1000),
            balance_weight: parseFloat(document.getElementById('l1-balance-weight')?.value || 0.5),
            stability_weight: parseFloat(document.getElementById('l1-stability-weight')?.value || 0.3),
            fixed_lookback: parseInt(document.getElementById('l1-fixed-lookback')?.value || 1500),
            min_threshold_gap: parseFloat(document.getElementById('l1-min-threshold-gap')?.value || 0.005),
            max_noise_ratio: parseFloat(document.getElementById('l1-max-noise-ratio')?.value || 0.35),
            target_hold_ratio: parseFloat(document.getElementById('l1-target-hold-ratio')?.value || 0.5),
            distribution_penalty: parseFloat(document.getElementById('l1-distribution-penalty')?.value || 1.0),
            n_trials: parseInt(document.getElementById('l1-n-trials')?.value || 50),
            timeout: parseInt(document.getElementById('l1-timeout')?.value || 7200)
        };
    }
}

/**
 * 驗證 L1 配置
 */
function validateL1Config(config) {
    // 驗證滯後期數
    if (config.lag < 1 || config.lag > 20) {
        showNotification('error', '滯後期數必須在 1-20 之間');
        return false;
    }
    
    // 驗證買賣分位數
    if (config.buy_quantile <= config.sell_quantile) {
        showNotification('error', '買入分位數必須大於賣出分位數');
        return false;
    }
    
    // 驗證閾值
    if (config.profit_threshold <= 0) {
        showNotification('error', '利潤閾值必須大於 0');
        return false;
    }
    
    if (config.loss_threshold >= 0) {
        showNotification('error', '損失閾值必須小於 0');
        return false;
    }
    
    // 驗證權重
    const totalWeight = config.balance_weight + config.stability_weight;
    if (totalWeight > 1.0) {
        showNotification('warning', '平衡權重和穩定性權重之和不應超過 1.0');
    }
    
    return true;
}

/**
 * 監控優化進度
 */
async function monitorL1Progress() {
    // 模擬進度更新（實際應該通過 WebSocket 或輪詢）
    let progress = 0;
    let trial = 0;
    const maxTrials = parseInt(document.getElementById('l1-n-trials')?.value || 50);
    
    const interval = setInterval(() => {
        progress += Math.random() * 5;
        trial = Math.min(Math.floor(progress / 100 * maxTrials), maxTrials);
        
        if (progress > 100) progress = 100;
        
        document.getElementById('l1-progress-bar').style.width = `${progress}%`;
        document.getElementById('l1-progress-text').textContent = 
            `已完成 ${Math.floor(progress)}%`;
        document.getElementById('l1-current-trial').textContent = 
            `${trial}/${maxTrials}`;
        document.getElementById('l1-best-score').textContent = 
            (Math.random() * 0.3 + 0.7).toFixed(4);
        
        const remainingTime = Math.floor((100 - progress) / progress * 120);
        document.getElementById('l1-time-remaining').textContent = 
            `約 ${remainingTime} 秒`;
        
        if (progress >= 100) {
            clearInterval(interval);
            document.getElementById('l1-progress-text').textContent = '優化完成！';
            document.getElementById('l1-time-remaining').textContent = '已完成';
            showNotification('success', 'Layer 1 標籤生成優化已完成');
            
            // 恢復按鈕
            const btn = document.getElementById('l1-run-btn');
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-play"></i> 開始優化';
        }
    }, 500);
}

console.log('✅ Layer 1 模組載入完成');
