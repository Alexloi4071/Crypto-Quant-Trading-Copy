/**
 * Realtime Data Processing
 * 实时数据处理模块，负责WebSocket连接和实时数据流管理
 * 与后端API和现有系统组件深度集成
 */

class RealtimeManager {
    constructor() {
        this.websocket = null;
        this.isConnected = false;
        this.subscriptions = new Map();
        this.messageQueue = [];
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.heartbeatInterval = null;
        this.lastHeartbeat = null;
        
        // 事件监听器
        this.eventListeners = {
            'connect': [],
            'disconnect': [],
            'message': [],
            'error': [],
            'portfolio_update': [],
            'signal_new': [],
            'trading_status_change': [],
            'system_alert': [],
            'market_data': []
        };
        
        // 数据缓存
        this.dataCache = {
            portfolio: null,
            positions: [],
            signals: [],
            systemMetrics: {},
            marketData: {}
        };
        
        console.log('RealtimeManager 初始化');
    }
    
    // ========== 连接管理 ==========
    
    async connect(url = null) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            console.log('WebSocket 已连接');
            return;
        }
        
        const wsUrl = url || this.getWebSocketURL();
        
        try {
            console.log('正在连接 WebSocket:', wsUrl);
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = this.handleOpen.bind(this);
            this.websocket.onmessage = this.handleMessage.bind(this);
            this.websocket.onclose = this.handleClose.bind(this);
            this.websocket.onerror = this.handleError.bind(this);
            
        } catch (error) {
            console.error('WebSocket 连接失败:', error);
            this.handleError(error);
        }
    }
    
    disconnect() {
        if (this.websocket) {
            console.log('断开 WebSocket 连接');
            this.websocket.close(1000, 'Manual disconnect');
            this.websocket = null;
        }
        
        this.isConnected = false;
        this.clearHeartbeat();
        this.emit('disconnect', { reason: 'manual' });
    }
    
    getWebSocketURL() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${protocol}//${host}/ws`;
    }
    
    // ========== 事件处理 ==========
    
    handleOpen(event) {
        console.log('WebSocket 连接已建立');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.lastHeartbeat = Date.now();
        
        // 开始心跳检测
        this.startHeartbeat();
        
        // 恢复订阅
        this.restoreSubscriptions();
        
        // 发送排队的消息
        this.flushMessageQueue();
        
        this.emit('connect', { timestamp: new Date() });
    }
    
    handleMessage(event) {
        try {
            const data = JSON.parse(event.data);
            this.processMessage(data);
        } catch (error) {
            console.error('解析WebSocket消息失败:', error, event.data);
        }
    }
    
    handleClose(event) {
        console.log('WebSocket 连接已关闭', event.code, event.reason);
        this.isConnected = false;
        this.websocket = null;
        this.clearHeartbeat();
        
        this.emit('disconnect', { 
            code: event.code, 
            reason: event.reason,
            timestamp: new Date()
        });
        
        // 自动重连
        if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
        }
    }
    
    handleError(error) {
        console.error('WebSocket 错误:', error);
        this.emit('error', { error, timestamp: new Date() });
    }
    
    // ========== 消息处理 ==========
    
    processMessage(data) {
        const { type, channel, payload, timestamp } = data;
        
        // 更新数据缓存
        this.updateCache(type, payload);
        
        // 触发通用消息事件
        this.emit('message', data);
        
        // 处理特定类型的消息
        switch (type) {
            case 'heartbeat':
                this.handleHeartbeat(payload);
                break;
                
            case 'portfolio_update':
                this.handlePortfolioUpdate(payload);
                break;
                
            case 'position_update':
                this.handlePositionUpdate(payload);
                break;
                
            case 'signal_new':
                this.handleNewSignal(payload);
                break;
                
            case 'signal_update':
                this.handleSignalUpdate(payload);
                break;
                
            case 'trading_status_change':
                this.handleTradingStatusChange(payload);
                break;
                
            case 'system_metrics':
                this.handleSystemMetrics(payload);
                break;
                
            case 'system_alert':
                this.handleSystemAlert(payload);
                break;
                
            case 'market_data':
                this.handleMarketData(payload);
                break;
                
            case 'trade_executed':
                this.handleTradeExecuted(payload);
                break;
                
            default:
                console.log('未处理的消息类型:', type, payload);
        }
    }
    
    // ========== 特定消息处理器 ==========
    
    handleHeartbeat(payload) {
        this.lastHeartbeat = Date.now();
        // 可以在这里更新系统状态显示
    }
    
    handlePortfolioUpdate(payload) {
        this.dataCache.portfolio = payload;
        this.emit('portfolio_update', payload);
        
        // 如果有仪表盘实例，直接更新
        if (window.dashboard) {
            window.dashboard.updatePortfolioSummary(payload);
        }
    }
    
    handlePositionUpdate(payload) {
        const { position_id, action, position_data } = payload;
        
        if (action === 'open') {
            this.dataCache.positions.push(position_data);
        } else if (action === 'close' || action === 'update') {
            const index = this.dataCache.positions.findIndex(p => p.id === position_id);
            if (index !== -1) {
                if (action === 'close') {
                    this.dataCache.positions.splice(index, 1);
                } else {
                    this.dataCache.positions[index] = { ...this.dataCache.positions[index], ...position_data };
                }
            }
        }
        
        this.emit('position_update', payload);
    }
    
    handleNewSignal(payload) {
        const signal = payload.signal;
        
        // 添加到信号缓存
        this.dataCache.signals.unshift(signal);
        
        // 保持缓存大小
        if (this.dataCache.signals.length > 100) {
            this.dataCache.signals = this.dataCache.signals.slice(0, 100);
        }
        
        this.emit('signal_new', signal);
        
        // 显示新信号通知
        if (window.dashboard) {
            const signalText = `${signal.symbol} ${signal.signal_type}`;
            window.dashboard.showNotification(`新信号: ${signalText}`, 'info');
            
            // 刷新信号显示
            window.dashboard.loadLatestSignals();
        }
    }
    
    handleSignalUpdate(payload) {
        const { signal_id, update_data } = payload;
        
        // 更新缓存中的信号
        const signalIndex = this.dataCache.signals.findIndex(s => s.id === signal_id);
        if (signalIndex !== -1) {
            this.dataCache.signals[signalIndex] = { 
                ...this.dataCache.signals[signalIndex], 
                ...update_data 
            };
        }
        
        this.emit('signal_update', payload);
    }
    
    handleTradingStatusChange(payload) {
        const { old_status, new_status, reason } = payload;
        
        this.emit('trading_status_change', payload);
        
        // 更新仪表盘显示
        if (window.dashboard) {
            window.dashboard.updateTradingStatus(payload);
            
            // 显示状态变化通知
            const statusText = {
                'running': '运行中',
                'paused': '已暂停',
                'stopped': '已停止',
                'not_started': '未启动'
            };
            
            const message = `交易状态: ${statusText[old_status] || old_status} → ${statusText[new_status] || new_status}`;
            window.dashboard.showNotification(message, 'info');
        }
    }
    
    handleSystemMetrics(payload) {
        this.dataCache.systemMetrics = { ...this.dataCache.systemMetrics, ...payload };
        this.emit('system_metrics', payload);
        
        // 更新系统监控显示
        if (window.dashboard) {
            window.dashboard.updateSystemMetrics({ metrics: payload });
        }
    }
    
    handleSystemAlert(payload) {
        const alert = payload.alert;
        
        this.emit('system_alert', alert);
        
        // 显示告警通知
        if (window.dashboard) {
            const notificationType = {
                'critical': 'error',
                'high': 'error',
                'medium': 'warning',
                'low': 'info'
            }[alert.severity] || 'warning';
            
            window.dashboard.showNotification(alert.message, notificationType);
            
            // 刷新告警列表
            window.dashboard.loadLatestAlerts();
        }
    }
    
    handleMarketData(payload) {
        const { symbol, data } = payload;
        this.dataCache.marketData[symbol] = data;
        
        this.emit('market_data', payload);
    }
    
    handleTradeExecuted(payload) {
        const trade = payload.trade;
        
        // 显示交易执行通知
        if (window.dashboard) {
            const message = `交易执行: ${trade.symbol} ${trade.side} ${trade.size}`;
            window.dashboard.showNotification(message, 'success');
            
            // 刷新组合数据
            setTimeout(() => {
                window.dashboard.loadPortfolioSummary();
            }, 1000);
        }
        
        this.emit('trade_executed', trade);
    }
    
    // ========== 订阅管理 ==========
    
    subscribe(channel, callback = null) {
        if (!this.subscriptions.has(channel)) {
            this.subscriptions.set(channel, {
                subscribed: false,
                callbacks: []
            });
        }
        
        if (callback) {
            this.subscriptions.get(channel).callbacks.push(callback);
        }
        
        if (this.isConnected) {
            this.sendSubscribe(channel);
        }
    }
    
    unsubscribe(channel, callback = null) {
        const subscription = this.subscriptions.get(channel);
        if (!subscription) return;
        
        if (callback) {
            const index = subscription.callbacks.indexOf(callback);
            if (index !== -1) {
                subscription.callbacks.splice(index, 1);
            }
        }
        
        if (subscription.callbacks.length === 0) {
            this.subscriptions.delete(channel);
            
            if (this.isConnected) {
                this.sendUnsubscribe(channel);
            }
        }
    }
    
    sendSubscribe(channel) {
        const message = {
            type: 'subscribe',
            channel: channel,
            timestamp: Date.now()
        };
        
        this.send(message);
        
        const subscription = this.subscriptions.get(channel);
        if (subscription) {
            subscription.subscribed = true;
        }
    }
    
    sendUnsubscribe(channel) {
        const message = {
            type: 'unsubscribe',
            channel: channel,
            timestamp: Date.now()
        };
        
        this.send(message);
    }
    
    restoreSubscriptions() {
        for (const [channel, subscription] of this.subscriptions.entries()) {
            if (!subscription.subscribed) {
                this.sendSubscribe(channel);
            }
        }
    }
    
    // ========== 消息发送 ==========
    
    send(message) {
        if (this.isConnected && this.websocket.readyState === WebSocket.OPEN) {
            try {
                this.websocket.send(JSON.stringify(message));
                return true;
            } catch (error) {
                console.error('发送WebSocket消息失败:', error);
                this.messageQueue.push(message);
                return false;
            }
        } else {
            this.messageQueue.push(message);
            return false;
        }
    }
    
    flushMessageQueue() {
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            if (!this.send(message)) {
                // 如果发送失败，放回队列开头
                this.messageQueue.unshift(message);
                break;
            }
        }
    }
    
    // ========== 心跳检测 ==========
    
    startHeartbeat() {
        this.clearHeartbeat();
        
        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected) {
                // 检查是否超时
                const now = Date.now();
                const timeSinceLastHeartbeat = now - (this.lastHeartbeat || now);
                
                if (timeSinceLastHeartbeat > 60000) { // 60秒超时
                    console.warn('心跳超时，重连WebSocket');
                    this.websocket.close(1001, 'Heartbeat timeout');
                    return;
                }
                
                // 发送心跳
                this.send({
                    type: 'heartbeat',
                    timestamp: now
                });
            }
        }, 30000); // 每30秒发送心跳
    }
    
    clearHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }
    
    // ========== 重连机制 ==========
    
    scheduleReconnect() {
        const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
        console.log(`将在 ${delay}ms 后尝试重连 (第${this.reconnectAttempts + 1}次)`);
        
        setTimeout(() => {
            this.reconnectAttempts++;
            this.connect();
        }, delay);
    }
    
    // ========== 事件系统 ==========
    
    on(event, callback) {
        if (this.eventListeners[event]) {
            this.eventListeners[event].push(callback);
        }
    }
    
    off(event, callback) {
        if (this.eventListeners[event]) {
            const index = this.eventListeners[event].indexOf(callback);
            if (index !== -1) {
                this.eventListeners[event].splice(index, 1);
            }
        }
    }
    
    emit(event, data) {
        if (this.eventListeners[event]) {
            this.eventListeners[event].forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`事件处理器错误 (${event}):`, error);
                }
            });
        }
    }
    
    // ========== 数据缓存管理 ==========
    
    updateCache(type, payload) {
        switch (type) {
            case 'portfolio_update':
                this.dataCache.portfolio = payload;
                break;
                
            case 'system_metrics':
                this.dataCache.systemMetrics = { ...this.dataCache.systemMetrics, ...payload };
                break;
                
            case 'market_data':
                if (payload.symbol) {
                    this.dataCache.marketData[payload.symbol] = payload.data;
                }
                break;
        }
    }
    
    getFromCache(type, key = null) {
        switch (type) {
            case 'portfolio':
                return this.dataCache.portfolio;
                
            case 'positions':
                return this.dataCache.positions;
                
            case 'signals':
                return this.dataCache.signals;
                
            case 'system_metrics':
                return key ? this.dataCache.systemMetrics[key] : this.dataCache.systemMetrics;
                
            case 'market_data':
                return key ? this.dataCache.marketData[key] : this.dataCache.marketData;
                
            default:
                return null;
        }
    }
    
    clearCache() {
        this.dataCache = {
            portfolio: null,
            positions: [],
            signals: [],
            systemMetrics: {},
            marketData: {}
        };
    }
    
    // ========== 实用方法 ==========
    
    getConnectionStatus() {
        return {
            connected: this.isConnected,
            readyState: this.websocket ? this.websocket.readyState : WebSocket.CLOSED,
            url: this.websocket ? this.websocket.url : null,
            lastHeartbeat: this.lastHeartbeat,
            reconnectAttempts: this.reconnectAttempts,
            subscriptions: Array.from(this.subscriptions.keys()),
            queuedMessages: this.messageQueue.length
        };
    }
    
    // 获取统计信息
    getStats() {
        return {
            connection: this.getConnectionStatus(),
            cache: {
                portfolio: !!this.dataCache.portfolio,
                positions: this.dataCache.positions.length,
                signals: this.dataCache.signals.length,
                systemMetrics: Object.keys(this.dataCache.systemMetrics).length,
                marketData: Object.keys(this.dataCache.marketData).length
            },
            events: Object.fromEntries(
                Object.entries(this.eventListeners).map(([event, listeners]) => [event, listeners.length])
            )
        };
    }
}

// ========== 高级功能扩展 ==========

class RealtimeDataProcessor {
    constructor(realtimeManager) {
        this.rtm = realtimeManager;
        this.processors = new Map();
        this.filters = new Map();
    }
    
    // 注册数据处理器
    registerProcessor(type, processor) {
        if (!this.processors.has(type)) {
            this.processors.set(type, []);
        }
        this.processors.get(type).push(processor);
    }
    
    // 注册数据过滤器
    registerFilter(type, filter) {
        if (!this.filters.has(type)) {
            this.filters.set(type, []);
        }
        this.filters.get(type).push(filter);
    }
    
    // 处理数据
    process(type, data) {
        // 先应用过滤器
        let filteredData = data;
        const filters = this.filters.get(type) || [];
        for (const filter of filters) {
            filteredData = filter(filteredData);
            if (!filteredData) return; // 被过滤掉
        }
        
        // 应用处理器
        const processors = this.processors.get(type) || [];
        for (const processor of processors) {
            try {
                processor(filteredData);
            } catch (error) {
                console.error(`数据处理器错误 (${type}):`, error);
            }
        }
    }
}

// ========== 全局实例和初始化 ==========

// 创建全局实时管理器实例
const realtimeManager = new RealtimeManager();
const dataProcessor = new RealtimeDataProcessor(realtimeManager);

// 导出到全局作用域
if (typeof window !== 'undefined') {
    window.realtimeManager = realtimeManager;
    window.dataProcessor = dataProcessor;
    
    // 自动连接
    document.addEventListener('DOMContentLoaded', () => {
        // 延迟连接，确保其他脚本已加载
        setTimeout(() => {
            realtimeManager.connect();
        }, 1000);
    });
    
    // 页面卸载前断开连接
    window.addEventListener('beforeunload', () => {
        realtimeManager.disconnect();
    });
}

// Node.js环境导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        RealtimeManager,
        RealtimeDataProcessor,
        realtimeManager,
        dataProcessor
    };
}

// ========== 预定义订阅和处理器 ==========

// 页面加载完成后设置默认订阅
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        // 订阅关键数据频道
        realtimeManager.subscribe('portfolio');
        realtimeManager.subscribe('system_metrics');
        realtimeManager.subscribe('signals');
        realtimeManager.subscribe('alerts');
        realtimeManager.subscribe('trading_status');
        
        // 注册数据处理器
        dataProcessor.registerProcessor('portfolio_update', (data) => {
            console.log('Portfolio updated:', data);
        });
        
        dataProcessor.registerProcessor('signal_new', (data) => {
            console.log('New signal:', data);
        });
        
        dataProcessor.registerProcessor('system_alert', (data) => {
            console.log('System alert:', data);
        });
        
        console.log('Realtime subscriptions initialized');
    }, 2000);
});

console.log('Realtime data processing module loaded');