/**
 * Layer 0: 數據基座配置
 * 包含預設模式和專家模式
 */

// L0 預設配置
const L0_PRESETS = {
    quickstart: {
        name: '快速開始',
        desc: '推薦新手使用，使用常用交易對和參數',
        config: {
            symbols: ['BTCUSDT', 'ETHUSDT'],
            timeframe: '15m',
            date_range_type: 'recent',
            recent_days: 30,
            cleaning_method: 'standard'
        }
    },
    production: {
        name: '生產環境',
        desc: '完整配置，適合正式交易',
        config: {
            symbols: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT'],
            timeframe: '15m',
            date_range_type: 'recent',
            recent_days: 90,
            cleaning_method: 'advanced'
        }
    },
    testing: {
        name: '測試環境',
        desc: '小數據集，快速驗證',
        config: {
            symbols: ['BTCUSDT'],
            timeframe: '15m',
            date_range_type: 'recent',
            recent_days: 7,
            cleaning_method: 'minimal'
        }
    }
};

// 可選交易對列表
const AVAILABLE_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
    'SOLUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'LTCUSDT',
    'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'XLMUSDT'
];

// 時間框架選項
const TIMEFRAMES = [
    { value: '1m', label: '1分鐘' },
    { value: '5m', label: '5分鐘' },
    { value: '15m', label: '15分鐘（推薦）' },
    { value: '1h', label: '1小時' },
    { value: '4h', label: '4小時' },
    { value: '1d', label: '1天' },
    { value: '1w', label: '1週' }
];

/**
 * 渲染 L0 內容
 */
function renderLayer0(container) {
    container.innerHTML = `
        <!-- 層級頭部 -->
        <div class="layer-header">
            <div class="layer-header-top">
                <div class="layer-title-group">
                    <div class="layer-badge" data-layer="0">
                        <i class="fas fa-database"></i>
                        <span>Layer 0</span>
                    </div>
                    <h1 class="layer-title">數據基座配置</h1>
                </div>
                <div class="layer-actions">
                    <a href="/" class="btn btn-secondary" style="text-decoration: none;">
                        <i class="fas fa-arrow-left"></i>
                        返回儀表盤
                    </a>
                    <button class="btn btn-secondary" onclick="resetL0Form()">
                        <i class="fas fa-redo"></i>
                        重置
                    </button>
                    <button class="btn btn-success" id="l0-run-btn" onclick="runL0Optimization()">
                        <i class="fas fa-play"></i>
                        開始優化
                    </button>
                </div>
            </div>
            <p style="color: var(--text-secondary); margin-top: 12px;">
                配置原始數據的下載、清洗和格式化參數。這是所有後續層級的基礎。
            </p>
        </div>
        
        <!-- 模式切換器 -->
        <div class="mode-switcher">
            <div class="mode-btn active" data-mode="preset" onclick="switchL0Mode('preset')">
                <i class="fas fa-magic"></i>
                <span>預設模式</span>
                <small>快速配置，使用推薦參數</small>
            </div>
            <div class="mode-btn" data-mode="expert" onclick="switchL0Mode('expert')">
                <i class="fas fa-cog"></i>
                <span>專家模式</span>
                <small>完全自定義所有參數</small>
            </div>
        </div>
        
        <!-- 預設模式內容 -->
        <div id="l0-preset-mode" class="mode-content">
            ${renderL0PresetMode()}
        </div>
        
        <!-- 專家模式內容 -->
        <div id="l0-expert-mode" class="mode-content hidden">
            ${renderL0ExpertMode()}
        </div>
        
        <!-- 進度監控區域 -->
        <div id="l0-progress" class="hidden">
            ${renderL0Progress()}
        </div>
    `;
    
    // 初始化預設配置
    loadL0Preset('quickstart');
}

/**
 * 渲染預設模式
 */
function renderL0PresetMode() {
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
                        預設配置已經過測試和優化，適合大多數場景。
                        如需精細控制，請切換到專家模式。
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 20px;">
                    ${Object.entries(L0_PRESETS).map(([key, preset]) => `
                        <div class="preset-card ${key === 'quickstart' ? 'selected' : ''}" 
                             data-preset="${key}"
                             onclick="loadL0Preset('${key}')">
                            <h3>${preset.name}</h3>
                            <p>${preset.desc}</p>
                            <div class="preset-details">
                                <div><strong>交易對:</strong> ${preset.config.symbols.length} 個</div>
                                <div><strong>時間框架:</strong> ${preset.config.timeframe}</div>
                                <div><strong>數據範圍:</strong> 最近 ${preset.config.recent_days} 天</div>
                            </div>
                        </div>
                    `).join('')}
                </div>
                
                <div style="margin-top: 24px;">
                    <h4 style="color: var(--text-primary); margin-bottom: 12px;">當前配置預覽</h4>
                    <pre id="l0-preset-preview" style="background: var(--bg-tertiary); padding: 16px; border-radius: 6px; overflow-x: auto; color: var(--text-secondary); font-size: 13px;"></pre>
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
function renderL0ExpertMode() {
    return `
        <div class="card">
            <div class="card-header">
                <i class="fas fa-sliders-h"></i> 交易對選擇
            </div>
            <div class="card-body">
                <div class="form-group">
                    <label class="form-label form-label-required">選擇交易對</label>
                    <div class="multi-selector" id="l0-symbols-selector">
                        ${AVAILABLE_SYMBOLS.map(symbol => `
                            <div class="selector-item" data-symbol="${symbol}" onclick="toggleSymbol('${symbol}')">
                                ${symbol}
                            </div>
                        `).join('')}
                    </div>
                    <div class="form-hint">點擊選擇/取消選擇交易對。建議選擇 2-10 個交易對。</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <i class="fas fa-clock"></i> 時間配置
            </div>
            <div class="card-body">
                <div class="form-group">
                    <label class="form-label form-label-required">時間框架</label>
                    <select class="form-select" id="l0-timeframe">
                        ${TIMEFRAMES.map(tf => `
                            <option value="${tf.value}" ${tf.value === '15m' ? 'selected' : ''}>
                                ${tf.label}
                            </option>
                        `).join('')}
                    </select>
                    <div class="form-hint">數據的時間粒度。15分鐘是最常用的時間框架。</div>
                </div>
                
                <div class="form-group">
                    <label class="form-label form-label-required">日期範圍</label>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                        <div>
                            <label style="font-size: 12px; color: var(--text-secondary); display: block; margin-bottom: 4px;">開始日期</label>
                            <input type="date" class="form-input" id="l0-start-date" value="${getDefaultStartDate()}">
                        </div>
                        <div>
                            <label style="font-size: 12px; color: var(--text-secondary); display: block; margin-bottom: 4px;">結束日期</label>
                            <input type="date" class="form-input" id="l0-end-date" value="${getDefaultEndDate()}">
                        </div>
                    </div>
                    <div class="form-hint">選擇歷史數據的時間範圍。</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <i class="fas fa-broom"></i> 數據清洗配置
            </div>
            <div class="card-body">
                <div class="form-group">
                    <label class="form-label">缺失值處理</label>
                    <select class="form-select" id="l0-missing-method">
                        <option value="ffill">前向填充（推薦）</option>
                        <option value="bfill">後向填充</option>
                        <option value="interpolate">線性插值</option>
                        <option value="drop">刪除缺失行</option>
                    </select>
                    <div class="form-hint">處理數據中的缺失值。</div>
                </div>
                
                <div class="form-group">
                    <label class="form-label">異常值檢測</label>
                    <select class="form-select" id="l0-outlier-method">
                        <option value="iqr">IQR 方法（推薦）</option>
                        <option value="zscore">Z-Score 方法</option>
                        <option value="none">不檢測</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label class="form-label">異常值閾值</label>
                    <input type="range" class="form-range" id="l0-outlier-threshold" 
                           min="1" max="5" step="0.5" value="3"
                           oninput="updateThresholdDisplay(this.value)">
                    <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                        <span style="font-size: 12px; color: var(--text-tertiary);">嚴格</span>
                        <span id="threshold-display" style="font-size: 14px; font-weight: 600; color: var(--text-primary);">3.0</span>
                        <span style="font-size: 12px; color: var(--text-tertiary);">寬鬆</span>
                    </div>
                    <div class="form-hint">較小的值會更嚴格地識別異常值。</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <i class="fas fa-database"></i> Optuna 優化參數
            </div>
            <div class="card-body">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div class="form-group">
                        <label class="form-label">試驗次數 (n_trials)</label>
                        <input type="number" class="form-input" id="l0-n-trials" 
                               value="50" min="10" max="500">
                        <div class="form-hint">Optuna 將運行的試驗次數。</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">超時時間（秒）</label>
                        <input type="number" class="form-input" id="l0-timeout" 
                               value="3600" min="60">
                        <div class="form-hint">優化的最大運行時間。</div>
                    </div>
                </div>
            </div>
        </div>
        
        <style>
            .form-range {
                width: 100%;
                height: 6px;
                background: var(--bg-tertiary);
                border-radius: 3px;
                outline: none;
                -webkit-appearance: none;
            }
            
            .form-range::-webkit-slider-thumb {
                -webkit-appearance: none;
                width: 18px;
                height: 18px;
                background: var(--brand-secondary);
                border-radius: 50%;
                cursor: pointer;
            }
            
            .form-range::-moz-range-thumb {
                width: 18px;
                height: 18px;
                background: var(--brand-secondary);
                border-radius: 50%;
                cursor: pointer;
                border: none;
            }
        </style>
    `;
}

/**
 * 渲染進度監控
 */
function renderL0Progress() {
    return `
        <div class="card">
            <div class="card-header">
                <i class="fas fa-chart-line"></i> 優化進度
            </div>
            <div class="card-body">
                <div class="progress-bar-container">
                    <div class="progress-bar-fill" id="l0-progress-bar" style="width: 0%"></div>
                </div>
                <div style="text-align: center; margin-top: 8px; color: var(--text-secondary);">
                    <span id="l0-progress-text">準備中...</span>
                </div>
            </div>
        </div>
    `;
}

/**
 * 切換 L0 模式
 */
function switchL0Mode(mode) {
    // 更新按鈕狀態
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`.mode-btn[data-mode="${mode}"]`).classList.add('active');
    
    // 切換內容顯示
    if (mode === 'preset') {
        document.getElementById('l0-preset-mode').classList.remove('hidden');
        document.getElementById('l0-expert-mode').classList.add('hidden');
    } else {
        document.getElementById('l0-preset-mode').classList.add('hidden');
        document.getElementById('l0-expert-mode').classList.remove('hidden');
    }
}

/**
 * 載入預設配置
 */
function loadL0Preset(presetKey) {
    // 更新選中狀態
    document.querySelectorAll('.preset-card').forEach(card => {
        card.classList.remove('selected');
    });
    document.querySelector(`.preset-card[data-preset="${presetKey}"]`).classList.add('selected');
    
    // 顯示配置預覽
    const preset = L0_PRESETS[presetKey];
    const preview = document.getElementById('l0-preset-preview');
    preview.textContent = JSON.stringify(preset.config, null, 2);
    
    // 保存到全局狀態
    OptunaState.layers[0] = {
        mode: 'preset',
        preset: presetKey,
        config: preset.config
    };
}

/**
 * 切換交易對選擇
 */
function toggleSymbol(symbol) {
    const item = document.querySelector(`.selector-item[data-symbol="${symbol}"]`);
    item.classList.toggle('selected');
}

/**
 * 更新閾值顯示
 */
function updateThresholdDisplay(value) {
    document.getElementById('threshold-display').textContent = parseFloat(value).toFixed(1);
}

/**
 * 獲取預設開始日期（30天前）
 */
function getDefaultStartDate() {
    const date = new Date();
    date.setDate(date.getDate() - 30);
    return date.toISOString().split('T')[0];
}

/**
 * 獲取預設結束日期（今天）
 */
function getDefaultEndDate() {
    return new Date().toISOString().split('T')[0];
}

/**
 * 重置表單
 */
function resetL0Form() {
    if (confirm('確定要重置所有配置嗎？')) {
        renderLayer0(document.getElementById('layer-content'));
        showNotification('info', '已重置為預設配置');
    }
}

/**
 * 運行 L0 優化
 */
async function runL0Optimization() {
    const btn = document.getElementById('l0-run-btn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 運行中...';
    
    try {
        // 收集配置
        const config = collectL0Config();
        
        // 驗證配置
        if (!validateL0Config(config)) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-play"></i> 開始優化';
            return;
        }
        
        // 顯示進度區域
        document.getElementById('l0-progress').classList.remove('hidden');
        
        // 調用 API
        showNotification('info', '正在啟動 Layer 0 優化...');
        const result = await OptunaAPI.startOptimization(0, config);
        
        showNotification('success', '優化已啟動！');
        
        // 開始監控進度
        monitorL0Progress();
        
    } catch (error) {
        showNotification('error', `啟動失敗: ${error.message}`);
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play"></i> 開始優化';
    }
}

/**
 * 收集 L0 配置
 */
function collectL0Config() {
    const mode = document.querySelector('.mode-btn.active').dataset.mode;
    
    if (mode === 'preset') {
        return OptunaState.layers[0].config;
    } else {
        // 專家模式：從表單收集
        const selectedSymbols = Array.from(document.querySelectorAll('.selector-item.selected'))
            .map(item => item.dataset.symbol);
        
        return {
            symbols: selectedSymbols,
            timeframe: document.getElementById('l0-timeframe').value,
            start_date: document.getElementById('l0-start-date').value,
            end_date: document.getElementById('l0-end-date').value,
            missing_method: document.getElementById('l0-missing-method').value,
            outlier_method: document.getElementById('l0-outlier-method').value,
            outlier_threshold: parseFloat(document.getElementById('l0-outlier-threshold').value),
            n_trials: parseInt(document.getElementById('l0-n-trials').value),
            timeout: parseInt(document.getElementById('l0-timeout').value)
        };
    }
}

/**
 * 驗證 L0 配置
 */
function validateL0Config(config) {
    if (!config.symbols || config.symbols.length === 0) {
        showNotification('error', '請至少選擇一個交易對');
        return false;
    }
    
    if (config.symbols.length > 20) {
        showNotification('warning', '交易對過多可能導致處理時間過長');
    }
    
    return true;
}

/**
 * 監控優化進度
 */
async function monitorL0Progress() {
    // 模擬進度更新（實際應該通過 WebSocket 或輪詢）
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 10;
        if (progress > 100) progress = 100;
        
        document.getElementById('l0-progress-bar').style.width = `${progress}%`;
        document.getElementById('l0-progress-text').textContent = 
            `已完成 ${Math.floor(progress)}%`;
        
        if (progress >= 100) {
            clearInterval(interval);
            document.getElementById('l0-progress-text').textContent = '優化完成！';
            showNotification('success', 'Layer 0 優化已完成');
            
            // 恢復按鈕
            const btn = document.getElementById('l0-run-btn');
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-play"></i> 開始優化';
        }
    }, 500);
}

console.log('✅ Layer 0 模組載入完成');

