/**
 * Optuna v2.0 核心控制邏輯
 * 負責層級切換、狀態管理、API 通訊
 */

// 全局狀態管理
const OptunaState = {
    currentLayer: 0,
    layers: {},
    isRunning: false,
    apiBase: '/api/v1/optuna'
};

/**
 * 頁面初始化
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Optuna v2.0 初始化');
    
    // 載入預設層級 (L0)
    switchLayer(0);
});

/**
 * 切換層級
 * @param {number} layerNum - 層級編號 (0-9)
 */
function switchLayer(layerNum) {
    console.log(`切換到 Layer ${layerNum}`);
    
    // 更新側邊欄高亮
    document.querySelectorAll('.layer-item').forEach(item => {
        item.classList.remove('active');
    });
    document.querySelector(`.layer-item[data-layer="${layerNum}"]`).classList.add('active');
    
    // 更新當前層級
    OptunaState.currentLayer = layerNum;
    
    // 載入對應層級的內容
    loadLayerContent(layerNum);
}

/**
 * 載入層級內容
 * @param {number} layerNum - 層級編號
 */
function loadLayerContent(layerNum) {
    const contentDiv = document.getElementById('layer-content');
    
    // 根據層級調用對應的渲染函數
    switch(layerNum) {
        case 0:
            if (typeof renderLayer0 === 'function') {
                renderLayer0(contentDiv);
            } else {
                contentDiv.innerHTML = '<p style="color: var(--text-tertiary);">L0 模組載入中...</p>';
            }
            break;
        case 1:
            if (typeof renderLayer1 === 'function') {
                renderLayer1(contentDiv);
            } else {
                contentDiv.innerHTML = getPlaceholderContent(1);
            }
            break;
        case 2:
            if (typeof renderLayer2 === 'function') {
                renderLayer2(contentDiv);
            } else {
                contentDiv.innerHTML = getPlaceholderContent(2);
            }
            break;
        case 3:
            if (typeof renderLayer3 === 'function') {
                renderLayer3(contentDiv);
            } else {
                contentDiv.innerHTML = getPlaceholderContent(3);
            }
            break;
        case 4:
            contentDiv.innerHTML = getPlaceholderContent(4);
            break;
        case 5:
            contentDiv.innerHTML = getPlaceholderContent(5);
            break;
        case 6:
            contentDiv.innerHTML = getPlaceholderContent(6);
            break;
        case 7:
            contentDiv.innerHTML = getPlaceholderContent(7);
            break;
        case 8:
            contentDiv.innerHTML = getPlaceholderContent(8);
            break;
        case 9:
            contentDiv.innerHTML = getPlaceholderContent(9);
            break;
        default:
            contentDiv.innerHTML = '<p>無效的層級</p>';
    }
}

/**
 * 獲取佔位內容（暫未實現的層級）
 */
function getPlaceholderContent(layerNum) {
    const layerNames = [
        '數據基座',
        '標籤生成',
        '特徵工程',
        '模型訓練',
        '模型集成',
        '風控模塊',
        '信號生成',
        '回測引擎',
        '報告生成',
        '實盤部署'
    ];
    
    return `
        <div class="layer-header">
            <div class="layer-header-top">
                <div class="layer-title-group">
                    <div class="layer-badge" data-layer="${layerNum}">
                        <span>Layer ${layerNum}</span>
                    </div>
                    <h1 class="layer-title">${layerNames[layerNum]}</h1>
                </div>
            </div>
        </div>
        
        <div class="alert alert-info">
            <i class="fas fa-info-circle"></i>
            <div>
                <strong>功能開發中</strong><br>
                此層級的參數配置介面正在開發中。<br>
                <span style="color: var(--text-tertiary); font-size: 12px;">
                    ✅ L0（數據基座）已完成，可立即使用<br>
                    ⏳ L1-L9 預計 2-3 天完成
                </span>
            </div>
        </div>
        
        <div class="card" style="margin-top: 20px;">
            <div class="card-header">
                <i class="fas fa-lightbulb"></i> 使用建議
            </div>
            <div class="card-body">
                <p style="margin-bottom: 12px;">在此層級開發完成前，您可以：</p>
                <ul style="margin-left: 20px; color: var(--text-secondary);">
                    <li style="margin-bottom: 8px;">✅ 使用 <strong>L0 數據基座</strong> 進行數據配置和優化</li>
                    <li style="margin-bottom: 8px;">📊 返回<a href="/" style="color: var(--brand-secondary); text-decoration: none;"> 主儀表盤 </a>查看交易狀態和監控</li>
                    <li style="margin-bottom: 8px;">📖 查閱文檔了解此層級的功能說明</li>
                </ul>
            </div>
        </div>
    `;
}

/**
 * API 請求封裝
 */
const OptunaAPI = {
    /**
     * 啟動層級優化
     */
    async startOptimization(layerNum, config) {
        try {
            const response = await fetch(`${OptunaState.apiBase}/layer${layerNum}/optimize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('啟動優化失敗:', error);
            showNotification('error', `啟動失敗: ${error.message}`);
            throw error;
        }
    },
    
    /**
     * 獲取優化狀態
     */
    async getStatus(layerNum) {
        try {
            const response = await fetch(`${OptunaState.apiBase}/layer${layerNum}/status`);
            if (!response.ok) throw new Error(response.statusText);
            return await response.json();
        } catch (error) {
            console.error('獲取狀態失敗:', error);
            return null;
        }
    },
    
    /**
     * 列出可用檔案
     */
    async listFiles(directory, extension = null) {
        try {
            let url = `/api/v1/files/list?directory=${encodeURIComponent(directory)}`;
            if (extension) {
                url += `&extension=${extension}`;
            }
            
            const response = await fetch(url);
            if (!response.ok) {
                // 如果 API 不存在，返回空列表
                return { files: [] };
            }
            return await response.json();
        } catch (error) {
            console.error('列出檔案失敗:', error);
            return { files: [] };
        }
    }
};

/**
 * 通知系統
 */
function showNotification(type, message) {
    // 創建通知元素
    const notification = document.createElement('div');
    notification.className = `alert alert-${type}`;
    notification.style.position = 'fixed';
    notification.style.top = '80px';
    notification.style.right = '24px';
    notification.style.zIndex = '9999';
    notification.style.minWidth = '300px';
    notification.style.animation = 'slideInRight 0.3s ease';
    
    const icons = {
        success: 'check-circle',
        error: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle'
    };
    
    notification.innerHTML = `
        <i class="fas fa-${icons[type]}"></i>
        <div>${message}</div>
    `;
    
    document.body.appendChild(notification);
    
    // 3秒後自動移除
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// 添加動畫 CSS
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

/**
 * 表單驗證工具
 */
const FormValidator = {
    /**
     * 驗證必填欄位
     */
    validateRequired(value) {
        return value !== null && value !== undefined && value !== '';
    },
    
    /**
     * 驗證數字範圍
     */
    validateRange(value, min, max) {
        const num = parseFloat(value);
        return !isNaN(num) && num >= min && num <= max;
    },
    
    /**
     * 驗證正整數
     */
    validatePositiveInteger(value) {
        const num = parseInt(value);
        return !isNaN(num) && num > 0 && Number.isInteger(num);
    },
    
    /**
     * 顯示錯誤訊息
     */
    showError(inputElement, message) {
        // 移除舊錯誤
        const oldError = inputElement.parentElement.querySelector('.form-error');
        if (oldError) oldError.remove();
        
        // 添加新錯誤
        const errorDiv = document.createElement('div');
        errorDiv.className = 'form-error';
        errorDiv.textContent = message;
        inputElement.parentElement.appendChild(errorDiv);
        inputElement.style.borderColor = 'var(--error)';
    },
    
    /**
     * 清除錯誤
     */
    clearError(inputElement) {
        const errorDiv = inputElement.parentElement.querySelector('.form-error');
        if (errorDiv) errorDiv.remove();
        inputElement.style.borderColor = '';
    }
};

console.log('✅ Optuna v2.0 核心模組載入完成');

