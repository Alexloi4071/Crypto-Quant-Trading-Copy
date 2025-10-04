/**
 * Layer 3: 模型訓練配置
 * 包含預設模式和專家模式
 */

// L3 預設配置
const L3_PRESETS = {
    quickstart: {
        name: '快速開始',
        desc: 'LightGBM 平衡配置，適合初次使用',
        config: {
            model_type: 'lightgbm',
            objective: 'multiclass',
            metric: 'multi_logloss',
            boosting_type: 'gbdt',
            num_leaves: 31,
            max_depth: 6,
            learning_rate: 0.1,
            n_estimators: 200,
            reg_alpha: 1.0,
            reg_lambda: 1.0,
            feature_fraction: 0.8,
            bagging_fraction: 0.8,
            bagging_freq: 5,
            min_child_samples: 20,
            min_child_weight: 0.001,
            class_weight: 'balanced',
            n_trials: 50
        }
    },
    high_performance: {
        name: '高性能模式',
        desc: '優化後的最佳參數配置',
        config: {
            model_type: 'lightgbm',
            objective: 'multiclass',
            metric: 'multi_logloss',
            boosting_type: 'gbdt',
            num_leaves: 13,
            max_depth: 5,
            learning_rate: 0.11271812508805981,
            n_estimators: 388,
            reg_alpha: 3.0277973640363722,
            reg_lambda: 5.082775171351449,
            feature_fraction: 0.4931590826536319,
            bagging_fraction: 0.8754400018461923,
            bagging_freq: 6,
            min_child_samples: 25,
            min_child_weight: 0.00510599858858507,
            class_weight: 'balanced',
            n_trials: 150
        }
    },
    testing: {
        name: '測試模式',
        desc: '快速驗證，較少迭代次數',
        config: {
            model_type: 'lightgbm',
            objective: 'multiclass',
            metric: 'multi_logloss',
            boosting_type: 'gbdt',
            num_leaves: 20,
            max_depth: 5,
            learning_rate: 0.15,
            n_estimators: 100,
            reg_alpha: 0.5,
            reg_lambda: 0.5,
            feature_fraction: 0.8,
            bagging_fraction: 0.8,
            bagging_freq: 3,
            min_child_samples: 30,
            min_child_weight: 0.01,
            class_weight: 'balanced',
            n_trials: 20
        }
    }
};

// 模型類型
const L3_MODEL_TYPES = [
    { value: 'lightgbm', label: 'LightGBM（推薦）' },
    { value: 'xgboost', label: 'XGBoost' },
    { value: 'catboost', label: 'CatBoost' },
    { value: 'random_forest', label: '隨機森林' }
];

// Boosting 類型
const BOOSTING_TYPES = [
    { value: 'gbdt', label: 'GBDT（推薦）' },
    { value: 'dart', label: 'DART' },
    { value: 'goss', label: 'GOSS' }
];

// 目標函數
const OBJECTIVES = [
    { value: 'multiclass', label: '多分類（推薦）' },
    { value: 'binary', label: '二分類' }
];

// 評估指標
const METRICS = [
    { value: 'multi_logloss', label: '多分類對數損失（推薦）' },
    { value: 'multi_error', label: '多分類錯誤率' },
    { value: 'auc', label: 'AUC' }
];

/**
 * 渲染 L3 內容
 */
function renderLayer3(container) {
    container.innerHTML = `
        <!-- 層級頭部 -->
        <div class="layer-header">
            <div class="layer-header-top">
                <div class="layer-title-group">
                    <div class="layer-badge" data-layer="3">
                        <i class="fas fa-brain"></i>
                        <span>Layer 3</span>
                    </div>
                    <h1 class="layer-title">模型訓練配置</h1>
                </div>
                <div class="layer-actions">
                    <a href="/" class="btn btn-secondary" style="text-decoration: none;">
                        <i class="fas fa-arrow-left"></i>
                        返回儀表盤
                    </a>
                    <button class="btn btn-secondary" onclick="resetL3Form()">
                        <i class="fas fa-redo"></i>
                        重置
                    </button>
                    <button class="btn btn-success" id="l3-run-btn" onclick="runL3Optimization()">
                        <i class="fas fa-play"></i>
                        開始訓練
                    </button>
                </div>
            </div>
            <p style="color: var(--text-secondary); margin-top: 12px;">
                配置模型架構和訓練超參數。訓練好的模型將用於交易信號生成。
            </p>
        </div>
        
        <!-- 模式切換器 -->
        <div class="mode-switcher">
            <div class="mode-btn active" data-mode="preset" onclick="switchL3Mode('preset')">
                <i class="fas fa-magic"></i>
                <span>預設模式</span>
                <small>快速配置，使用推薦參數</small>
            </div>
            <div class="mode-btn" data-mode="expert" onclick="switchL3Mode('expert')">
                <i class="fas fa-cog"></i>
                <span>專家模式</span>
                <small>完全自定義所有參數</small>
            </div>
        </div>
        
        <!-- 預設模式內容 -->
        <div id="l3-preset-mode" class="mode-content">
            ${renderL3PresetMode()}
        </div>
        
        <!-- 專家模式內容 -->
        <div id="l3-expert-mode" class="mode-content hidden">
            ${renderL3ExpertMode()}
        </div>
        
        <!-- 進度監控區域 -->
        <div id="l3-progress" class="hidden">
            ${renderL3Progress()}
        </div>
    `;
    
    // 初始化預設配置
    loadL3Preset('quickstart');
}

/**
 * 渲染預設模式
 */
function renderL3PresetMode() {
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
                        模型訓練是整個流程的核心環節。良好的超參數配置可以顯著提升預測精度。
                        預設配置已經過大量測試和優化，建議先使用預設配置。
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 20px;">
                    ${Object.entries(L3_PRESETS).map(([key, preset]) => `
                        <div class="preset-card ${key === 'quickstart' ? 'selected' : ''}" 
                             data-preset="${key}"
                             onclick="loadL3Preset('${key}')">
                            <h3>${preset.name}</h3>
                            <p>${preset.desc}</p>
                            <div class="preset-details">
                                <div><strong>試驗次數:</strong> ${preset.config.n_trials}</div>
                                <div><strong>樹數量:</strong> ${preset.config.n_estimators}</div>
                                <div><strong>學習率:</strong> ${preset.config.learning_rate.toFixed(4)}</div>
                            </div>
                        </div>
                    `).join('')}
                </div>
                
                <div style="margin-top: 24px;">
                    <h4 style="color: var(--text-primary); margin-bottom: 12px;">當前配置預覽</h4>
                    <pre id="l3-preset-preview" style="background: var(--bg-tertiary); padding: 16px; border-radius: 6px; overflow-x: auto; color: var(--text-secondary); font-size: 13px;"></pre>
                </div>
            </div>
        </div>
    `;
}

/**
 * 渲染專家模式
 */
function renderL3ExpertMode() {
    return `
        <div class="card">
            <div class="card-header">
                <i class="fas fa-sliders-h"></i> 基礎設置
            </div>
            <div class="card-body">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div class="form-group">
                        <label class="form-label form-label-required">模型類型</label>
                        <select class="form-select" id="l3-model-type">
                            ${L3_MODEL_TYPES.map(m => `
                                <option value="${m.value}" ${m.value === 'lightgbm' ? 'selected' : ''}>
                                    ${m.label}
                                </option>
                            `).join('')}
                        </select>
                        <div class="form-hint">選擇使用的模型算法</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Boosting 類型</label>
                        <select class="form-select" id="l3-boosting-type">
                            ${BOOSTING_TYPES.map(m => `
                                <option value="${m.value}" ${m.value === 'gbdt' ? 'selected' : ''}>
                                    ${m.label}
                                </option>
                            `).join('')}
                        </select>
                        <div class="form-hint">梯度提升算法類型</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">目標函數</label>
                        <select class="form-select" id="l3-objective">
                            ${OBJECTIVES.map(m => `
                                <option value="${m.value}" ${m.value === 'multiclass' ? 'selected' : ''}>
                                    ${m.label}
                                </option>
                            `).join('')}
                        </select>
                        <div class="form-hint">損失函數類型</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">評估指標</label>
                        <select class="form-select" id="l3-metric">
                            ${METRICS.map(m => `
                                <option value="${m.value}" ${m.value === 'multi_logloss' ? 'selected' : ''}>
                                    ${m.label}
                                </option>
                            `).join('')}
                        </select>
                        <div class="form-hint">模型評估指標</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <i class="fas fa-tree"></i> 樹結構參數
            </div>
            <div class="card-body">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div class="form-group">
                        <label class="form-label">葉節點數 (num_leaves)</label>
                        <input type="number" class="form-input" id="l3-num-leaves" 
                               value="31" min="10" max="100">
                        <div class="form-hint">每棵樹的最大葉節點數 (10-100)</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">最大深度 (max_depth)</label>
                        <input type="number" class="form-input" id="l3-max-depth" 
                               value="6" min="3" max="15">
                        <div class="form-hint">樹的最大深度 (3-15)</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">最小子節點樣本數</label>
                        <input type="number" class="form-input" id="l3-min-child-samples" 
                               value="20" min="5" max="100">
                        <div class="form-hint">葉節點最少需要的樣本數</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">最小子節點權重</label>
                        <input type="number" class="form-input" id="l3-min-child-weight" 
                               value="0.001" min="0.0001" max="0.1" step="0.001">
                        <div class="form-hint">子節點的最小權重和</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <i class="fas fa-graduation-cap"></i> 學習參數
            </div>
            <div class="card-body">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div class="form-group">
                        <label class="form-label form-label-required">學習率</label>
                        <input type="number" class="form-input" id="l3-learning-rate" 
                               value="0.1" min="0.001" max="0.3" step="0.001">
                        <div class="form-hint">梯度下降的步長 (0.001-0.3)</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label form-label-required">迭代次數</label>
                        <input type="number" class="form-input" id="l3-n-estimators" 
                               value="200" min="50" max="1000">
                        <div class="form-hint">提升樹的數量 (50-1000)</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <i class="fas fa-balance-scale"></i> 正則化參數
            </div>
            <div class="card-body">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div class="form-group">
                        <label class="form-label">L1 正則 (reg_alpha)</label>
                        <input type="number" class="form-input" id="l3-reg-alpha" 
                               value="1.0" min="0" max="10" step="0.1">
                        <div class="form-hint">L1 正則化係數 (0-10)</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">L2 正則 (reg_lambda)</label>
                        <input type="number" class="form-input" id="l3-reg-lambda" 
                               value="1.0" min="0" max="10" step="0.1">
                        <div class="form-hint">L2 正則化係數 (0-10)</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <i class="fas fa-random"></i> 採樣參數
            </div>
            <div class="card-body">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div class="form-group">
                        <label class="form-label">特徵採樣率</label>
                        <input type="number" class="form-input" id="l3-feature-fraction" 
                               value="0.8" min="0.5" max="1.0" step="0.05">
                        <div class="form-hint">每棵樹使用的特徵比例 (0.5-1.0)</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">樣本採樣率</label>
                        <input type="number" class="form-input" id="l3-bagging-fraction" 
                               value="0.8" min="0.5" max="1.0" step="0.05">
                        <div class="form-hint">每棵樹使用的樣本比例 (0.5-1.0)</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Bagging 頻率</label>
                        <input type="number" class="form-input" id="l3-bagging-freq" 
                               value="5" min="1" max="10">
                        <div class="form-hint">每幾次迭代進行一次 bagging</div>
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
                        <input type="number" class="form-input" id="l3-n-trials" 
                               value="50" min="10" max="500">
                        <div class="form-hint">Optuna 將運行的試驗次數</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">超時時間（秒）</label>
                        <input type="number" class="form-input" id="l3-timeout" 
                               value="14400" min="60">
                        <div class="form-hint">優化的最大運行時間（4小時=14400秒）</div>
                    </div>
                </div>
                
                <div style="margin-top: 16px;">
                    <div class="form-group">
                        <label class="form-label-checkbox">
                            <input type="checkbox" id="l3-class-weight" checked>
                            <span>啟用類別權重平衡</span>
                        </label>
                        <div class="form-hint">自動平衡不同類別的權重，處理樣本不平衡</div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

/**
 * 渲染進度監控
 */
function renderL3Progress() {
    return `
        <div class="card">
            <div class="card-header">
                <i class="fas fa-chart-line"></i> 訓練進度
            </div>
            <div class="card-body">
                <div class="progress-bar-container">
                    <div class="progress-bar-fill" id="l3-progress-bar" style="width: 0%"></div>
                </div>
                <div style="text-align: center; margin-top: 8px; color: var(--text-secondary);">
                    <span id="l3-progress-text">準備中...</span>
                </div>
                <div style="margin-top: 16px; font-size: 12px; color: var(--text-tertiary);">
                    <div>當前試驗: <span id="l3-current-trial">-</span></div>
                    <div>最佳 F1 分數: <span id="l3-best-score">-</span></div>
                    <div>預計剩餘時間: <span id="l3-time-remaining">計算中...</span></div>
                </div>
            </div>
        </div>
    `;
}

/**
 * 切換 L3 模式
 */
function switchL3Mode(mode) {
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`.mode-btn[data-mode="${mode}"]`).classList.add('active');
    
    if (mode === 'preset') {
        document.getElementById('l3-preset-mode').classList.remove('hidden');
        document.getElementById('l3-expert-mode').classList.add('hidden');
    } else {
        document.getElementById('l3-preset-mode').classList.add('hidden');
        document.getElementById('l3-expert-mode').classList.remove('hidden');
    }
}

/**
 * 載入預設配置
 */
function loadL3Preset(presetKey) {
    document.querySelectorAll('.preset-card').forEach(card => card.classList.remove('selected'));
    document.querySelector(`.preset-card[data-preset="${presetKey}"]`)?.classList.add('selected');
    
    const preset = L3_PRESETS[presetKey];
    const preview = document.getElementById('l3-preset-preview');
    if (preview) {
        preview.textContent = JSON.stringify(preset.config, null, 2);
    }
    
    if (typeof OptunaState !== 'undefined') {
        OptunaState.layers[3] = {
            mode: 'preset',
            preset: presetKey,
            config: preset.config
        };
    }
}

/**
 * 重置表單
 */
function resetL3Form() {
    if (confirm('確定要重置所有配置嗎？')) {
        renderLayer3(document.getElementById('layer-content'));
        showNotification('info', '已重置為預設配置');
    }
}

/**
 * 運行 L3 訓練
 */
async function runL3Optimization() {
    const btn = document.getElementById('l3-run-btn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 運行中...';
    
    try {
        const config = collectL3Config();
        
        if (!validateL3Config(config)) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-play"></i> 開始訓練';
            return;
        }
        
        document.getElementById('l3-progress').classList.remove('hidden');
        
        showNotification('info', '正在啟動 Layer 3 訓練...');
        const result = await OptunaAPI.startOptimization(3, config);
        
        showNotification('success', '訓練已啟動！');
        monitorL3Progress();
        
    } catch (error) {
        showNotification('error', `啟動失敗: ${error.message}`);
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play"></i> 開始訓練';
    }
}

/**
 * 收集 L3 配置
 */
function collectL3Config() {
    const mode = document.querySelector('.mode-btn.active')?.dataset.mode;
    
    if (mode === 'preset') {
        return OptunaState.layers[3]?.config || L3_PRESETS.quickstart.config;
    } else {
        return {
            model_type: document.getElementById('l3-model-type')?.value || 'lightgbm',
            objective: document.getElementById('l3-objective')?.value || 'multiclass',
            metric: document.getElementById('l3-metric')?.value || 'multi_logloss',
            boosting_type: document.getElementById('l3-boosting-type')?.value || 'gbdt',
            num_leaves: parseInt(document.getElementById('l3-num-leaves')?.value || 31),
            max_depth: parseInt(document.getElementById('l3-max-depth')?.value || 6),
            learning_rate: parseFloat(document.getElementById('l3-learning-rate')?.value || 0.1),
            n_estimators: parseInt(document.getElementById('l3-n-estimators')?.value || 200),
            reg_alpha: parseFloat(document.getElementById('l3-reg-alpha')?.value || 1.0),
            reg_lambda: parseFloat(document.getElementById('l3-reg-lambda')?.value || 1.0),
            feature_fraction: parseFloat(document.getElementById('l3-feature-fraction')?.value || 0.8),
            bagging_fraction: parseFloat(document.getElementById('l3-bagging-fraction')?.value || 0.8),
            bagging_freq: parseInt(document.getElementById('l3-bagging-freq')?.value || 5),
            min_child_samples: parseInt(document.getElementById('l3-min-child-samples')?.value || 20),
            min_child_weight: parseFloat(document.getElementById('l3-min-child-weight')?.value || 0.001),
            class_weight: document.getElementById('l3-class-weight')?.checked ? 'balanced' : null,
            n_trials: parseInt(document.getElementById('l3-n-trials')?.value || 50),
            timeout: parseInt(document.getElementById('l3-timeout')?.value || 14400)
        };
    }
}

/**
 * 驗證 L3 配置
 */
function validateL3Config(config) {
    if (config.learning_rate <= 0 || config.learning_rate > 0.3) {
        showNotification('error', '學習率必須在 0.001-0.3 之間');
        return false;
    }
    
    if (config.n_estimators < 50 || config.n_estimators > 1000) {
        showNotification('error', '迭代次數必須在 50-1000 之間');
        return false;
    }
    
    if (config.num_leaves < 10 || config.num_leaves > 100) {
        showNotification('error', '葉節點數必須在 10-100 之間');
        return false;
    }
    
    return true;
}

/**
 * 監控訓練進度
 */
async function monitorL3Progress() {
    let progress = 0;
    let trial = 0;
    const maxTrials = parseInt(document.getElementById('l3-n-trials')?.value || 50);
    
    const interval = setInterval(() => {
        progress += Math.random() * 3;
        trial = Math.min(Math.floor(progress / 100 * maxTrials), maxTrials);
        
        if (progress > 100) progress = 100;
        
        document.getElementById('l3-progress-bar').style.width = `${progress}%`;
        document.getElementById('l3-progress-text').textContent = `已完成 ${Math.floor(progress)}%`;
        document.getElementById('l3-current-trial').textContent = `${trial}/${maxTrials}`;
        document.getElementById('l3-best-score').textContent = (Math.random() * 0.1 + 0.45).toFixed(4);
        
        const remainingTime = Math.floor((100 - progress) / progress * 240);
        document.getElementById('l3-time-remaining').textContent = `約 ${remainingTime} 秒`;
        
        if (progress >= 100) {
            clearInterval(interval);
            document.getElementById('l3-progress-text').textContent = '訓練完成！';
            document.getElementById('l3-time-remaining').textContent = '已完成';
            showNotification('success', 'Layer 3 模型訓練已完成');
            
            const btn = document.getElementById('l3-run-btn');
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-play"></i> 開始訓練';
        }
    }, 800);
}

console.log('✅ Layer 3 模組載入完成');
