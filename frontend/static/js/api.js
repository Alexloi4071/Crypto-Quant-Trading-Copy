/**
 * API Client
 * 前端API调用封装，与后端API路由深度集成
 * 提供完整的API调用功能和错误处理
 */

class APIClient {
    constructor(baseURL = '') {
        this.baseURL = baseURL;
        this.authToken = null;
        this.defaultHeaders = {
            'Content-Type': 'application/json',
        };
        
        // 初始化axios实例
        this.http = axios.create({
            baseURL: this.baseURL,
            timeout: 30000,
            headers: this.defaultHeaders
        });
        
        // 请求拦截器
        this.http.interceptors.request.use(
            (config) => {
                if (this.authToken) {
                    config.headers.Authorization = `Bearer ${this.authToken}`;
                }
                return config;
            },
            (error) => {
                return Promise.reject(error);
            }
        );
        
        // 响应拦截器
        this.http.interceptors.response.use(
            (response) => {
                return response;
            },
            (error) => {
                this.handleError(error);
                return Promise.reject(error);
            }
        );
        
        console.log('API Client 初始化完成');
    }
    
    // 设置认证令牌
    setAuthToken(token) {
        this.authToken = token;
        if (token) {
            this.http.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        } else {
            delete this.http.defaults.headers.common['Authorization'];
        }
    }
    
    // 错误处理
    handleError(error) {
        if (error.response) {
            const status = error.response.status;
            const message = error.response.data?.error?.message || error.response.data?.message || 'Unknown error';
            
            console.error(`API Error ${status}:`, message);
            
            if (status === 401) {
                this.onUnauthorized();
            } else if (status === 403) {
                this.onForbidden();
            }
        } else if (error.request) {
            console.error('Network Error:', error.message);
        } else {
            console.error('Error:', error.message);
        }
    }
    
    // 未授权处理
    onUnauthorized() {
        console.log('未授权，需要重新登录');
        // 这里可以重定向到登录页面或刷新token
    }
    
    // 权限不足处理
    onForbidden() {
        console.log('权限不足');
        // 显示权限不足消息
    }
    
    // ========== 健康检查 API ==========
    
    async getHealth() {
        try {
            const response = await this.http.get('/api/v1/health');
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getSystemStatus() {
        try {
            const response = await this.http.get('/api/v1/status');
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    // ========== 组合管理 API ==========
    
    async getPortfolioSummary(includeHistory = false) {
        try {
            const response = await this.http.get('/api/v1/portfolio/summary', {
                params: { include_history: includeHistory }
            });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getPositions(params = {}) {
        try {
            const response = await this.http.get('/api/v1/portfolio/positions', { params });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getPositionDetail(positionId) {
        try {
            const response = await this.http.get(`/api/v1/portfolio/positions/${positionId}`);
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async closePosition(positionId, reason = 'Manual close') {
        try {
            const response = await this.http.post(`/api/v1/portfolio/positions/${positionId}/close`, null, {
                params: { reason }
            });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getPerformanceMetrics(period = '30d', benchmark = null) {
        try {
            const params = { period };
            if (benchmark) params.benchmark = benchmark;
            
            const response = await this.http.get('/api/v1/portfolio/performance', { params });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getRiskAssessment() {
        try {
            const response = await this.http.get('/api/v1/portfolio/risk-assessment');
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getPortfolioHistory(days = 30, granularity = '1h') {
        try {
            const response = await this.http.get('/api/v1/portfolio/history', {
                params: { days, granularity }
            });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    // ========== 交易控制 API ==========
    
    async getTradingStatus() {
        try {
            const response = await this.http.get('/api/v1/trading/status');
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async controlTrading(action, symbols = null, reason = null) {
        try {
            const data = { action };
            if (symbols) data.symbols = symbols;
            if (reason) data.reason = reason;
            
            const response = await this.http.post('/api/v1/trading/control', data);
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async executeManualTrade(tradeData) {
        try {
            const response = await this.http.post('/api/v1/trading/manual-trade', tradeData);
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getActiveStrategies() {
        try {
            const response = await this.http.get('/api/v1/trading/strategies');
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async updateStrategy(strategyName, updateData) {
        try {
            const response = await this.http.put(`/api/v1/trading/strategies/${strategyName}`, updateData);
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getTradeHistory(params = {}) {
        try {
            const response = await this.http.get('/api/v1/trading/trades/history', { params });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    // ========== 交易信号 API ==========
    
    async getCurrentSignals(params = {}) {
        try {
            const response = await this.http.get('/api/v1/signals/current', { params });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getLatestSignals(params = {}) {
        try {
            const response = await this.http.get('/api/v1/trading/signals/latest', { params });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getSignalHistory(params = {}) {
        try {
            const response = await this.http.get('/api/v1/signals/history', { params });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getSignalAnalysis(symbol, params = {}) {
        try {
            const response = await this.http.get(`/api/v1/signals/analysis/${symbol}`, { params });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async createManualSignal(signalData) {
        try {
            const response = await this.http.post('/api/v1/trading/signals/manual', signalData);
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getSignalPerformance(params = {}) {
        try {
            const response = await this.http.get('/api/v1/signals/performance', { params });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    // ========== 数据查询 API ==========
    
    async getAvailableSymbols(exchange = null, activeOnly = true) {
        try {
            const params = { active_only: activeOnly };
            if (exchange) params.exchange = exchange;
            
            const response = await this.http.get('/api/v1/data/symbols', { params });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getOHLCVData(symbol, params = {}) {
        try {
            const response = await this.http.get(`/api/v1/data/ohlcv/${symbol}`, { params });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getFeatureData(symbol, params = {}) {
        try {
            const response = await this.http.get(`/api/v1/data/features/${symbol}`, { params });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getMarketInfo(symbol) {
        try {
            const response = await this.http.get(`/api/v1/data/market-info/${symbol}`);
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async checkDataQuality(symbol, params = {}) {
        try {
            const response = await this.http.get(`/api/v1/data/quality-check/${symbol}`, { params });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async refreshData(symbol, params = {}) {
        try {
            const response = await this.http.post(`/api/v1/data/refresh/${symbol}`, null, { params });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    // ========== 系统监控 API ==========
    
    async getCurrentMetrics(params = {}) {
        try {
            const response = await this.http.get('/api/v1/monitoring/metrics/current', { params });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getMetricHistory(metricName, params = {}) {
        try {
            const response = await this.http.get('/api/v1/monitoring/metrics/history', {
                params: { metric_name: metricName, ...params }
            });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getAlerts(params = {}) {
        try {
            const response = await this.http.get('/api/v1/monitoring/alerts', { params });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async acknowledgeAlert(alertId, notes = null) {
        try {
            const params = {};
            if (notes) params.notes = notes;
            
            const response = await this.http.post(`/api/v1/monitoring/alerts/${alertId}/acknowledge`, null, { params });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async resolveAlert(alertId, resolutionNotes = null) {
        try {
            const params = {};
            if (resolutionNotes) params.resolution_notes = resolutionNotes;
            
            const response = await this.http.post(`/api/v1/monitoring/alerts/${alertId}/resolve`, null, { params });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getSystemHealth(detailed = false) {
        try {
            const response = await this.http.get('/api/v1/monitoring/health', {
                params: { detailed }
            });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getNotificationStatus() {
        try {
            const response = await this.http.get('/api/v1/monitoring/notifications/status');
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async testNotification(channel, message = 'This is a test notification') {
        try {
            const response = await this.http.post('/api/v1/monitoring/notifications/test', null, {
                params: { channel, message }
            });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    async getDashboardData(timeRange = '1h') {
        try {
            const response = await this.http.get('/api/v1/monitoring/dashboard-data', {
                params: { time_range: timeRange }
            });
            return response.data;
        } catch (error) {
            throw error;
        }
    }
    
    // ========== WebSocket 连接管理 ==========
    
    connectWebSocket(onMessage = null, onError = null) {
        const wsUrl = `ws://${window.location.host}/ws`;
        const websocket = new WebSocket(wsUrl);
        
        websocket.onopen = (event) => {
            console.log('WebSocket 连接已建立');
        };
        
        websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (onMessage) {
                    onMessage(data);
                }
            } catch (error) {
                console.error('WebSocket 消息解析错误:', error);
            }
        };
        
        websocket.onclose = (event) => {
            console.log('WebSocket 连接已关闭');
        };
        
        websocket.onerror = (error) => {
            console.error('WebSocket 错误:', error);
            if (onError) {
                onError(error);
            }
        };
        
        return websocket;
    }
    
    // ========== 工具方法 ==========
    
    // 格式化金额
    formatCurrency(amount, currency = 'USD') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency,
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(amount);
    }
    
    // 格式化百分比
    formatPercentage(value, decimals = 2) {
        return `${(value * 100).toFixed(decimals)}%`;
    }
    
    // 格式化数字
    formatNumber(value, decimals = 2) {
        return value.toLocaleString(undefined, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    }
    
    // 格式化时间
    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleString();
    }
    
    // 获取相对时间
    getRelativeTime(timestamp) {
        const now = new Date();
        const past = new Date(timestamp);
        const diff = now - past;
        
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        const days = Math.floor(diff / 86400000);
        
        if (minutes < 1) return '刚刚';
        if (minutes < 60) return `${minutes}分钟前`;
        if (hours < 24) return `${hours}小时前`;
        return `${days}天前`;
    }
    
    // 批量请求
    async batchRequest(requests) {
        try {
            const promises = requests.map(request => this.http(request));
            const responses = await Promise.allSettled(promises);
            
            return responses.map((response, index) => {
                if (response.status === 'fulfilled') {
                    return { success: true, data: response.value.data };
                } else {
                    return { success: false, error: response.reason };
                }
            });
        } catch (error) {
            throw error;
        }
    }
    
    // 带重试的请求
    async requestWithRetry(requestConfig, maxRetries = 3, delay = 1000) {
        for (let i = 0; i < maxRetries; i++) {
            try {
                const response = await this.http(requestConfig);
                return response.data;
            } catch (error) {
                if (i === maxRetries - 1) throw error;
                
                // 等待后重试
                await new Promise(resolve => setTimeout(resolve, delay * (i + 1)));
            }
        }
    }
    
    // 缓存请求结果
    cacheRequest(key, requestFn, ttl = 60000) {
        const cache = this.cache || (this.cache = new Map());
        
        return async (...args) => {
            const cacheKey = `${key}_${JSON.stringify(args)}`;
            const cached = cache.get(cacheKey);
            
            if (cached && Date.now() - cached.timestamp < ttl) {
                return cached.data;
            }
            
            const data = await requestFn(...args);
            cache.set(cacheKey, { data, timestamp: Date.now() });
            
            return data;
        };
    }
    
    // 清除缓存
    clearCache(key = null) {
        if (!this.cache) return;
        
        if (key) {
            // 清除特定缓存
            for (const cacheKey of this.cache.keys()) {
                if (cacheKey.startsWith(key)) {
                    this.cache.delete(cacheKey);
                }
            }
        } else {
            // 清除所有缓存
            this.cache.clear();
        }
    }
}

// 全局API客户端实例
const apiClient = new APIClient();

// 导出到全局作用域
if (typeof window !== 'undefined') {
    window.APIClient = APIClient;
    window.apiClient = apiClient;
}

// 如果是Node.js环境，导出模块
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { APIClient, apiClient };
}

console.log('API Client 加载完成');