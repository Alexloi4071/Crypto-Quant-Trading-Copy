/**
 * Layer 2: 特徵工程配置
 * 包含預設模式和專家模式
 */

// L2 預設配置
const L2_PRESETS = {
    quickstart: {
        name: '快速開始',
        desc: '平衡的特徵工程配置，適合初次使用',
        config: {
            feature_selection_method: 'mutual_info',
            noise_reduction: false,
            feature_interaction: true,
            coarse_k: 70,
            fine_ratio: 0.4,
            stability_threshold: 0.5,
            correlation_threshold: 0.8,
            model_type: 'random_forest',
            rf_n_estimators: 200,
            rf_max_depth: 15,
            rf_min_samples_split: 5,
            rf_min_samples_leaf: 2,
            rf_max_features: 'sqrt',
            n_trials: 50
        }
    },
    high_quality: {
        name: '高質量模式',
        desc: '追求特徵質量，更多試驗次數',
        config: {
            feature_selection_method: 'mutual_info',
            noise_reduction: true,
            feature_interaction: true,
            coarse_k: 69,
            fine_ratio: 0.41723054332688336,
            stability_threshold: 0.48991402826168695,
            correlation_threshold: 0.8192238734950134,
            model_type: 'random_forest',
            rf_n_estimators: 249,
            rf_max_depth: 19,
            rf_min_samples_split: 7,
            rf_min_samples_leaf: 4,
            rf_max_features: 'log2',
            n_trials: 100
        }
    },
    testing: {
        name: '測試模式',
        desc: '快速驗證，較少試驗次數',
        config: {
            feature_selection_method: 'mutual_info',
            noise_reduction: false,
            feature_interaction: false,
            coarse_k: 50,
            fine_ratio: 0.3,
            stability_threshold: 0.6,
            correlation_threshold: 0.85,
            model_type: 'random_forest',
            rf_n_estimators: 100,
            rf_max_depth: 10,
            rf_min_samples_split: 10,
            rf_min_samples_leaf: 5,
            rf_max_features: 'sqrt',
            n_trials: 20
        }
    }
};

// 特徵選擇方法
const FEATURE_SELECTION_METHODS = [
    { value: 'mutual_info', label: '互信息（推薦）' },
    { value: 'f_classif', label: 'F統計量' },
    { value: 'chi2', label: '卡方檢驗' },
    { value: 'recursive', label: '遞歸特徵消除' }
];

// 模型類型
const MODEL_TYPES = [
    { value: 'random_forest', label: '隨機森林（推薦）' },
    { value: 'gradient_boosting', label: '梯度提升' },
    { value: 'extra_trees', label: '極端隨機樹' }
];

// Max Features 選項
const MAX_FEATURES_OPTIONS = [
    { value: 'sqrt', label: 'sqrt（平方根）' },
    { value: 'log2', label: 'log2（對數）' },
    { value: 'auto', label: 'auto（自動）' },
    { value: 'none', label: 'None（全部）' }
];

// 特徵池配置（基於 indicators.yaml）
const FEATURE_POOL = {
    total: 200,
    categories: {
        trend: { name: '趨勢指標', count: 45, color: '#8b5cf6' },
        momentum: { name: '動量指標', count: 38, color: '#ec4899' },
        volatility: { name: '波動率指標', count: 32, color: '#06b6d4' },
        volume: { name: '成交量指標', count: 28, color: '#10b981' },
        market_structure: { name: '市場結構', count: 22, color: '#f59e0b' },
        wyckoff: { name: '威科夫指標', count: 18, color: '#ef4444' },
        advanced: { name: '高級指標', count: 17, color: '#6366f1' }
    }
};

/**
 * 渲染 L2 內容
 */
function renderLayer2(container) {
    container.innerHTML = `
        <!-- 層級頭部 -->
        <div class="layer-header">
            <div class="layer-header-top">
                <div class="layer-title-group">
                    <div class="layer-badge" data-layer="2">
                        <i class="fas fa-cogs"></i>
                        <span>Layer 2</span>
                    </div>
                    <h1 class="layer-title">特徵工程配置</h1>
                </div>
                <div class="layer-actions">
                    <a href="/" class="btn btn-secondary" style="text-decoration: none;">
                        <i class="fas fa-arrow-left"></i>
                        返回儀表盤
                    </a>
                    <button class="btn btn-secondary" onclick="resetL2Form()">
                        <i class="fas fa-redo"></i>
                        重置
                    </button>
                    <button class="btn btn-success" id="l2-run-btn" onclick="runL2Optimization()">
                        <i class="fas fa-play"></i>
                        開始優化
                    </button>
                </div>
            </div>
            <p style="color: var(--text-secondary); margin-top: 12px;">
                配置特徵選擇、降噪、交互等工程策略。生成的特徵集將用於模型訓練。
            </p>
        </div>
        
        <!-- 模式切換器 -->
        <div class="mode-switcher">
            <div class="mode-btn active" data-mode="preset" onclick="switchL2Mode('preset')">
                <i class="fas fa-magic"></i>
                <span>預設模式</span>
                <small>快速配置，使用推薦參數</small>
            </div>
            <div class="mode-btn" data-mode="expert" onclick="switchL2Mode('expert')">
                <i class="fas fa-cog"></i>
                <span>專家模式</span>
                <small>完全自定義所有參數</small>
            </div>
        </div>
        
        <!-- 預設模式內容 -->
        <div id="l2-preset-mode" class="mode-content">
            ${renderL2PresetMode()}
        </div>
        
        <!-- 專家模式內容 -->
        <div id="l2-expert-mode" class="mode-content hidden">
            ${renderL2ExpertMode()}
        </div>
        
        <!-- 進度監控區域 -->
        <div id="l2-progress" class="hidden">
            ${renderL2Progress()}
        </div>
    `;
    
    // 初始化預設配置
    loadL2Preset('quickstart');
}

/**
 * 渲染預設模式
 */
function renderL2PresetMode() {
    return `
        <!-- 特徵池總覽 -->
        <div class="card" style="margin-bottom: 20px;">
            <div class="card-header">
                <i class="fas fa-database"></i> 特徵池總覽
            </div>
            <div class="card-body">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                    <div>
                        <div style="font-size: 14px; color: var(--text-secondary); margin-bottom: 4px;">可用特徵總數</div>
                        <div style="font-size: 32px; font-weight: 600; color: var(--brand-secondary);">${FEATURE_POOL.total}+</div>
                        <div style="font-size: 12px; color: var(--text-tertiary); margin-top: 4px;">來自 indicators.yaml</div>
                    </div>
                    <div style="flex: 1; margin: 0 30px;">
                        ${renderFeatureDistributionChart()}
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px;">
                    ${Object.entries(FEATURE_POOL.categories).map(([key, cat]) => `
                        <div style="padding: 12px; background: var(--bg-tertiary); border-radius: 6px; border-left: 3px solid ${cat.color};">
                            <div style="font-size: 11px; color: var(--text-tertiary); margin-bottom: 4px;">${cat.name}</div>
                            <div style="font-size: 18px; font-weight: 600; color: var(--text-primary);">${cat.count}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <i class="fas fa-list"></i> 選擇預設配置
            </div>
            <div class="card-body">
                <div class="alert alert-info">
                    <i class="fas fa-lightbulb"></i>
                    <div>
                        <strong>提示</strong><br>
                        特徵工程從 ${FEATURE_POOL.total}+ 個技術指標中智能選擇最優特徵。
                        良好的特徵選擇可以提升模型性能並減少過擬合。
                        預設配置已經過優化測試，適合大多數場景。
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 20px;">
                    ${Object.entries(L2_PRESETS).map(([key, preset]) => `
                        <div class="preset-card ${key === 'quickstart' ? 'selected' : ''}" 
                             data-preset="${key}"
                             onclick="loadL2Preset('${key}')">
                            <h3>${preset.name}</h3>
                            <p>${preset.desc}</p>
                            <div class="preset-details">
                                <div><strong>試驗次數:</strong> ${preset.config.n_trials}</div>
                                <div><strong>粗選數:</strong> ${preset.config.coarse_k}</div>
                                <div><strong>模型:</strong> ${preset.config.model_type}</div>
                            </div>
                        </div>
                    `).join('')}
                </div>
                
                <div style="margin-top: 24px;">
                    <h4 style="color: var(--text-primary); margin-bottom: 12px;">當前配置預覽</h4>
                    <pre id="l2-preset-preview" style="background: var(--bg-tertiary); padding: 16px; border-radius: 6px; overflow-x: auto; color: var(--text-secondary); font-size: 13px;"></pre>
                </div>
            </div>
        </div>
    `;
}

/**
 * 渲染專家模式
 */
function renderL2ExpertMode() {
    return `
        <!-- 特徵選擇流程可視化 -->
        <div class="card" style="margin-bottom: 20px;">
            <div class="card-header">
                <i class="fas fa-project-diagram"></i> 特徵選擇流程
            </div>
            <div class="card-body">
                <div id="l2-feature-flow" style="padding: 20px;">
                    ${renderFeatureSelectionFlow()}
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <i class="fas fa-filter"></i> 特徵選擇設置
            </div>
            <div class="card-body">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div class="form-group">
                        <label class="form-label form-label-required">特徵選擇方法</label>
                        <select class="form-select" id="l2-feature-selection-method">
                            ${FEATURE_SELECTION_METHODS.map(m => `
                                <option value="${m.value}" ${m.value === 'mutual_info' ? 'selected' : ''}>
                                    ${m.label}
                                </option>
                            `).join('')}
                        </select>
                        <div class="form-hint">選擇特徵重要性的評估方法</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">粗選特徵數 (coarse_k)</label>
                        <input type="number" class="form-input" id="l2-coarse-k" 
                               value="70" min="20" max="200" onchange="updateL2FeatureFlow()">
                        <div class="form-hint">第一階段粗選保留的特徵數量</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">精選比率 (fine_ratio)</label>
                        <input type="number" class="form-input" id="l2-fine-ratio" 
                               value="0.4" min="0.2" max="0.8" step="0.05" onchange="updateL2FeatureFlow()">
                        <div class="form-hint">第二階段精選的特徵比例 (0.2-0.8)</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">穩定性閾值</label>
                        <input type="number" class="form-input" id="l2-stability-threshold" 
                               value="0.5" min="0.3" max="0.9" step="0.05">
                        <div class="form-hint">特徵穩定性要求 (0.3-0.9)</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">相關性閾值</label>
                        <input type="number" class="form-input" id="l2-correlation-threshold" 
                               value="0.8" min="0.5" max="0.95" step="0.05">
                        <div class="form-hint">去除高相關特徵的閾值 (0.5-0.95)</div>
                    </div>
                </div>
                
                <div style="margin-top: 16px;">
                    <div class="form-group">
                        <label class="form-label-checkbox">
                            <input type="checkbox" id="l2-noise-reduction">
                            <span>啟用降噪處理</span>
                        </label>
                        <div class="form-hint">對特徵進行降噪預處理，可能提升穩定性</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label-checkbox">
                            <input type="checkbox" id="l2-feature-interaction" checked>
                            <span>啟用特徵交互</span>
                        </label>
                        <div class="form-hint">生成特徵之間的交互項，增強表達能力</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <i class="fas fa-brain"></i> 特徵評估模型設置
            </div>
            <div class="card-body">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div class="form-group">
                        <label class="form-label form-label-required">模型類型</label>
                        <select class="form-select" id="l2-model-type">
                            ${MODEL_TYPES.map(m => `
                                <option value="${m.value}" ${m.value === 'random_forest' ? 'selected' : ''}>
                                    ${m.label}
                                </option>
                            `).join('')}
                        </select>
                        <div class="form-hint">用於評估特徵重要性的模型</div>
                    </div>
                </div>
                
                <div id="l2-rf-params" style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px;">
                    <div class="form-group">
                        <label class="form-label">樹的數量 (n_estimators)</label>
                        <input type="number" class="form-input" id="l2-rf-n-estimators" 
                               value="200" min="50" max="500">
                        <div class="form-hint">隨機森林中樹的數量</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">最大深度 (max_depth)</label>
                        <input type="number" class="form-input" id="l2-rf-max-depth" 
                               value="15" min="5" max="30">
                        <div class="form-hint">每棵樹的最大深度</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">最小分割樣本數</label>
                        <input type="number" class="form-input" id="l2-rf-min-samples-split" 
                               value="5" min="2" max="20">
                        <div class="form-hint">分割節點所需的最小樣本數</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">最小葉節點樣本數</label>
                        <input type="number" class="form-input" id="l2-rf-min-samples-leaf" 
                               value="2" min="1" max="10">
                        <div class="form-hint">葉節點所需的最小樣本數</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">最大特徵數</label>
                        <select class="form-select" id="l2-rf-max-features">
                            ${MAX_FEATURES_OPTIONS.map(m => `
                                <option value="${m.value}" ${m.value === 'sqrt' ? 'selected' : ''}>
                                    ${m.label}
                                </option>
                            `).join('')}
                        </select>
                        <div class="form-hint">每次分割考慮的最大特徵數</div>
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
                        <input type="number" class="form-input" id="l2-n-trials" 
                               value="50" min="10" max="500">
                        <div class="form-hint">Optuna 將運行的試驗次數</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">超時時間（秒）</label>
                        <input type="number" class="form-input" id="l2-timeout" 
                               value="10800" min="60">
                        <div class="form-hint">優化的最大運行時間（3小時=10800秒）</div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

/**
 * 渲染進度監控
 */
function renderL2Progress() {
    return `
        <div class="card">
            <div class="card-header">
                <i class="fas fa-chart-line"></i> 優化進度
            </div>
            <div class="card-body">
                <div class="progress-bar-container">
                    <div class="progress-bar-fill" id="l2-progress-bar" style="width: 0%"></div>
                </div>
                <div style="text-align: center; margin-top: 8px; color: var(--text-secondary);">
                    <span id="l2-progress-text">準備中...</span>
                </div>
                <div style="margin-top: 16px; font-size: 12px; color: var(--text-tertiary);">
                    <div>當前試驗: <span id="l2-current-trial">-</span></div>
                    <div>最佳分數: <span id="l2-best-score">-</span></div>
                    <div>預計剩餘時間: <span id="l2-time-remaining">計算中...</span></div>
                </div>
            </div>
        </div>
    `;
}

/**
 * 切換 L2 模式
 */
function switchL2Mode(mode) {
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`.mode-btn[data-mode="${mode}"]`).classList.add('active');
    
    if (mode === 'preset') {
        document.getElementById('l2-preset-mode').classList.remove('hidden');
        document.getElementById('l2-expert-mode').classList.add('hidden');
    } else {
        document.getElementById('l2-preset-mode').classList.add('hidden');
        document.getElementById('l2-expert-mode').classList.remove('hidden');
    }
}

/**
 * 載入預設配置
 */
function loadL2Preset(presetKey) {
    document.querySelectorAll('.preset-card').forEach(card => card.classList.remove('selected'));
    document.querySelector(`.preset-card[data-preset="${presetKey}"]`)?.classList.add('selected');
    
    const preset = L2_PRESETS[presetKey];
    const preview = document.getElementById('l2-preset-preview');
    if (preview) {
        preview.textContent = JSON.stringify(preset.config, null, 2);
    }
    
    if (typeof OptunaState !== 'undefined') {
        OptunaState.layers[2] = {
            mode: 'preset',
            preset: presetKey,
            config: preset.config
        };
    }
}

/**
 * 重置表單
 */
function resetL2Form() {
    if (confirm('確定要重置所有配置嗎？')) {
        renderLayer2(document.getElementById('layer-content'));
        showNotification('info', '已重置為預設配置');
    }
}

/**
 * 運行 L2 優化
 */
async function runL2Optimization() {
    const btn = document.getElementById('l2-run-btn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 運行中...';
    
    try {
        const config = collectL2Config();
        
        if (!validateL2Config(config)) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-play"></i> 開始優化';
            return;
        }
        
        document.getElementById('l2-progress').classList.remove('hidden');
        
        showNotification('info', '正在啟動 Layer 2 優化...');
        const result = await OptunaAPI.startOptimization(2, config);
        
        showNotification('success', '優化已啟動！');
        monitorL2Progress();
        
    } catch (error) {
        showNotification('error', `啟動失敗: ${error.message}`);
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play"></i> 開始優化';
    }
}

/**
 * 收集 L2 配置
 */
function collectL2Config() {
    const mode = document.querySelector('.mode-btn.active')?.dataset.mode;
    
    if (mode === 'preset') {
        return OptunaState.layers[2]?.config || L2_PRESETS.quickstart.config;
    } else {
        return {
            feature_selection_method: document.getElementById('l2-feature-selection-method')?.value || 'mutual_info',
            noise_reduction: document.getElementById('l2-noise-reduction')?.checked || false,
            feature_interaction: document.getElementById('l2-feature-interaction')?.checked || true,
            coarse_k: parseInt(document.getElementById('l2-coarse-k')?.value || 70),
            fine_ratio: parseFloat(document.getElementById('l2-fine-ratio')?.value || 0.4),
            stability_threshold: parseFloat(document.getElementById('l2-stability-threshold')?.value || 0.5),
            correlation_threshold: parseFloat(document.getElementById('l2-correlation-threshold')?.value || 0.8),
            model_type: document.getElementById('l2-model-type')?.value || 'random_forest',
            rf_n_estimators: parseInt(document.getElementById('l2-rf-n-estimators')?.value || 200),
            rf_max_depth: parseInt(document.getElementById('l2-rf-max-depth')?.value || 15),
            rf_min_samples_split: parseInt(document.getElementById('l2-rf-min-samples-split')?.value || 5),
            rf_min_samples_leaf: parseInt(document.getElementById('l2-rf-min-samples-leaf')?.value || 2),
            rf_max_features: document.getElementById('l2-rf-max-features')?.value || 'sqrt',
            n_trials: parseInt(document.getElementById('l2-n-trials')?.value || 50),
            timeout: parseInt(document.getElementById('l2-timeout')?.value || 10800)
        };
    }
}

/**
 * 驗證 L2 配置
 */
function validateL2Config(config) {
    if (config.coarse_k < 20 || config.coarse_k > 200) {
        showNotification('error', '粗選特徵數必須在 20-200 之間');
        return false;
    }
    
    if (config.fine_ratio < 0.2 || config.fine_ratio > 0.8) {
        showNotification('error', '精選比率必須在 0.2-0.8 之間');
        return false;
    }
    
    return true;
}

/**
 * 監控優化進度
 */
async function monitorL2Progress() {
    let progress = 0;
    let trial = 0;
    const maxTrials = parseInt(document.getElementById('l2-n-trials')?.value || 50);
    
    const interval = setInterval(() => {
        progress += Math.random() * 4;
        trial = Math.min(Math.floor(progress / 100 * maxTrials), maxTrials);
        
        if (progress > 100) progress = 100;
        
        document.getElementById('l2-progress-bar').style.width = `${progress}%`;
        document.getElementById('l2-progress-text').textContent = `已完成 ${Math.floor(progress)}%`;
        document.getElementById('l2-current-trial').textContent = `${trial}/${maxTrials}`;
        document.getElementById('l2-best-score').textContent = (Math.random() * 0.15 + 0.4).toFixed(4);
        
        const remainingTime = Math.floor((100 - progress) / progress * 180);
        document.getElementById('l2-time-remaining').textContent = `約 ${remainingTime} 秒`;
        
        if (progress >= 100) {
            clearInterval(interval);
            document.getElementById('l2-progress-text').textContent = '優化完成！';
            document.getElementById('l2-time-remaining').textContent = '已完成';
            showNotification('success', 'Layer 2 特徵工程優化已完成');
            
            const btn = document.getElementById('l2-run-btn');
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-play"></i> 開始優化';
        }
    }, 600);
}

/**
 * 渲染特徵分布圖表
 */
function renderFeatureDistributionChart() {
    const categories = Object.values(FEATURE_POOL.categories);
    const maxCount = Math.max(...categories.map(c => c.count));
    
    return `
        <div style="display: flex; flex-direction: column; gap: 8px;">
            ${categories.map(cat => {
                const percentage = (cat.count / FEATURE_POOL.total * 100).toFixed(1);
                const width = (cat.count / maxCount * 100).toFixed(1);
                return `
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div style="width: 80px; font-size: 11px; color: var(--text-tertiary); text-align: right;">
                            ${cat.name}
                        </div>
                        <div style="flex: 1; background: var(--bg-tertiary); border-radius: 4px; height: 20px; position: relative; overflow: hidden;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: ${width}%; background: ${cat.color}; border-radius: 4px; transition: width 0.3s;"></div>
                        </div>
                        <div style="width: 50px; font-size: 11px; color: var(--text-secondary); text-align: right;">
                            ${cat.count} (${percentage}%)
                        </div>
                    </div>
                `;
            }).join('')}
        </div>
    `;
}

/**
 * 渲染特徵選擇流程
 */
function renderFeatureSelectionFlow() {
    const coarseK = 70;
    const fineRatio = 0.4;
    const finalCount = Math.round(coarseK * fineRatio);
    
    return `
        <div style="display: flex; align-items: center; justify-content: space-around; position: relative;">
            <!-- 特徵池 -->
            <div class="flow-node" style="text-align: center;">
                <div class="flow-icon" style="width: 80px; height: 80px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 12px;">
                    <i class="fas fa-database" style="font-size: 32px; color: white;"></i>
                </div>
                <div style="font-weight: 600; font-size: 16px; color: var(--text-primary); margin-bottom: 4px;">特徵池</div>
                <div style="font-size: 24px; font-weight: 700; color: var(--brand-secondary);" id="flow-total">${FEATURE_POOL.total}+</div>
                <div style="font-size: 11px; color: var(--text-tertiary); margin-top: 4px;">可用特徵</div>
            </div>
            
            <!-- 箭頭 1 -->
            <div style="display: flex; flex-direction: column; align-items: center; margin: 0 20px;">
                <i class="fas fa-arrow-right" style="font-size: 24px; color: var(--brand-secondary);"></i>
                <div style="font-size: 11px; color: var(--text-tertiary); margin-top: 4px; white-space: nowrap;">粗選階段</div>
            </div>
            
            <!-- 粗選結果 -->
            <div class="flow-node" style="text-align: center;">
                <div class="flow-icon" style="width: 80px; height: 80px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 12px;">
                    <i class="fas fa-filter" style="font-size: 32px; color: white;"></i>
                </div>
                <div style="font-weight: 600; font-size: 16px; color: var(--text-primary); margin-bottom: 4px;">粗選</div>
                <div style="font-size: 24px; font-weight: 700; color: var(--brand-secondary);" id="flow-coarse">${coarseK}</div>
                <div style="font-size: 11px; color: var(--text-tertiary); margin-top: 4px;">保留特徵</div>
            </div>
            
            <!-- 箭頭 2 -->
            <div style="display: flex; flex-direction: column; align-items: center; margin: 0 20px;">
                <i class="fas fa-arrow-right" style="font-size: 24px; color: var(--brand-secondary);"></i>
                <div style="font-size: 11px; color: var(--text-tertiary); margin-top: 4px; white-space: nowrap;">精選階段</div>
            </div>
            
            <!-- 精選結果 -->
            <div class="flow-node" style="text-align: center;">
                <div class="flow-icon" style="width: 80px; height: 80px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 12px;">
                    <i class="fas fa-star" style="font-size: 32px; color: white;"></i>
                </div>
                <div style="font-weight: 600; font-size: 16px; color: var(--text-primary); margin-bottom: 4px;">最終特徵</div>
                <div style="font-size: 24px; font-weight: 700; color: var(--brand-secondary);" id="flow-final">${finalCount}</div>
                <div style="font-size: 11px; color: var(--text-tertiary); margin-top: 4px;">用於訓練</div>
            </div>
        </div>
        
        <!-- 統計信息 -->
        <div style="margin-top: 30px; padding: 16px; background: var(--bg-tertiary); border-radius: 8px;">
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; text-align: center;">
                <div>
                    <div style="font-size: 12px; color: var(--text-tertiary); margin-bottom: 4px;">粗選保留率</div>
                    <div style="font-size: 20px; font-weight: 600; color: var(--text-primary);" id="flow-coarse-rate">${(coarseK / FEATURE_POOL.total * 100).toFixed(1)}%</div>
                </div>
                <div>
                    <div style="font-size: 12px; color: var(--text-tertiary); margin-bottom: 4px;">精選比率</div>
                    <div style="font-size: 20px; font-weight: 600; color: var(--text-primary);" id="flow-fine-ratio">${(fineRatio * 100).toFixed(0)}%</div>
                </div>
                <div>
                    <div style="font-size: 12px; color: var(--text-tertiary); margin-bottom: 4px;">總體保留率</div>
                    <div style="font-size: 20px; font-weight: 600; color: var(--text-primary);" id="flow-total-rate">${(finalCount / FEATURE_POOL.total * 100).toFixed(1)}%</div>
                </div>
            </div>
        </div>
    `;
}

/**
 * 更新特徵選擇流程圖
 */
function updateL2FeatureFlow() {
    const coarseK = parseInt(document.getElementById('l2-coarse-k')?.value || 70);
    const fineRatio = parseFloat(document.getElementById('l2-fine-ratio')?.value || 0.4);
    const finalCount = Math.round(coarseK * fineRatio);
    
    // 更新數字
    const flowCoarse = document.getElementById('flow-coarse');
    const flowFinal = document.getElementById('flow-final');
    const flowCoarseRate = document.getElementById('flow-coarse-rate');
    const flowFineRatio = document.getElementById('flow-fine-ratio');
    const flowTotalRate = document.getElementById('flow-total-rate');
    
    if (flowCoarse) flowCoarse.textContent = coarseK;
    if (flowFinal) flowFinal.textContent = finalCount;
    if (flowCoarseRate) flowCoarseRate.textContent = `${(coarseK / FEATURE_POOL.total * 100).toFixed(1)}%`;
    if (flowFineRatio) flowFineRatio.textContent = `${(fineRatio * 100).toFixed(0)}%`;
    if (flowTotalRate) flowTotalRate.textContent = `${(finalCount / FEATURE_POOL.total * 100).toFixed(1)}%`;
}

console.log('✅ Layer 2 模組載入完成');
