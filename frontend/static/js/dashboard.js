/**
 * Dashboard JavaScript
 * 仪表盘前端逻辑，与API深度集成
 * 提供实时数据展示和交互功能
 */

class TradingDashboard {
    constructor() {
        this.apiClient = window.apiClient || new APIClient();
        this.websocket = null;
        this.charts = {};
        this.updateIntervals = {};
        this.isInitialized = false;
        
        // 配置参数
        this.config = {
            updateInterval: 30000,      // 数据更新间隔（毫秒）
            chartUpdateInterval: 60000, // 图表更新间隔
            websocketReconnectDelay: 5000, // WebSocket重连延迟
            maxRetries: 3,             // 最大重试次数
            notificationDuration: 5000  // 通知显示时长
        };
        
        console.log('Trading Dashboard 初始化');
    }
    
    // ========== 初始化方法 ==========
    
    async initialize() {
        try {
            console.log('开始初始化仪表盘...');
            
            // 初始化事件监听器
            this.setupEventListeners();
            
            // 加载初始数据
            await this.loadInitialData();
            
            // 初始化图表
            this.initializeCharts();
            
            // 连接WebSocket
            this.connectWebSocket();
            
            // 设置定时更新
            this.setupPeriodicUpdates();
            
            this.isInitialized = true;
            console.log('仪表盘初始化完成');
            
        } catch (error) {
            console.error('仪表盘初始化失败:', error);
            this.showNotification('仪表盘初始化失败，请刷新页面重试', 'error');
        }
    }
    
    // 设置事件监听器
    setupEventListeners() {
        // 页面可见性变化
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseUpdates();
            } else {
                this.resumeUpdates();
            }
        });
        
        // 窗口大小变化
        window.addEventListener('resize', () => {
            this.resizeCharts();
        });
        
        // 浏览器关闭前清理
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
    }
    
    // 加载初始数据
    async loadInitialData() {
        const loadingTasks = [
            this.loadSystemHealth(),
            this.loadPortfolioSummary(),
            this.loadTradingStatus(),
            this.loadCurrentMetrics()
        ];
        
        // 并行加载，允许部分失败
        const results = await Promise.allSettled(loadingTasks);
        
        results.forEach((result, index) => {
            if (result.status === 'rejected') {
                console.warn(`初始数据加载失败 (任务${index + 1}):`, result.reason);
            }
        });
    }
    
    // ========== 数据加载方法 ==========
    
    async loadSystemHealth() {
        try {
            const health = await this.apiClient.getHealth();
            this.updateSystemStatus(health.data || health);
        } catch (error) {
            console.warn('加载系统健康状态失败:', error);
            this.updateSystemStatus({ status: 'unhealthy' });
        }
    }
    
    async loadPortfolioSummary() {
        try {
            const portfolio = await this.apiClient.getPortfolioSummary(true);
            this.updatePortfolioSummary(portfolio.data || portfolio);
        } catch (error) {
            console.warn('加载组合概览失败:', error);
            this.updatePortfolioSummary(this.getEmptyPortfolio());
        }
    }
    
    async loadTradingStatus() {
        try {
            const trading = await this.apiClient.getTradingStatus();
            this.updateTradingStatus(trading.data || trading);
        } catch (error) {
            console.warn('加载交易状态失败:', error);
            this.updateTradingStatus(this.getEmptyTradingStatus());
        }
    }
    
    async loadCurrentMetrics() {
        try {
            const metrics = await this.apiClient.getCurrentMetrics();
            this.updateSystemMetrics(metrics.data || metrics);
        } catch (error) {
            console.warn('加载系统指标失败:', error);
        }
    }
    
    async loadLatestSignals() {
        try {
            const signals = await this.apiClient.getLatestSignals({ limit: 5 });
            this.updateLatestSignals(signals.data || signals);
        } catch (error) {
            console.warn('加载最新信号失败:', error);
        }
    }
    
    async loadLatestAlerts() {
        try {
            const alerts = await this.apiClient.getAlerts({ 
                status: 'active', 
                limit: 5 
            });
            this.updateLatestAlerts(alerts.data || alerts);
        } catch (error) {
            console.warn('加载最新告警失败:', error);
        }
    }
    
    // ========== UI更新方法 ==========
    
    updateSystemStatus(health) {
        const indicator = document.getElementById('system-status-indicator');
        const text = document.getElementById('system-status-text');
        
        if (!indicator || !text) return;
        
        const status = health.status || 'unknown';
        
        // 更新指示器颜色和动画
        const statusConfig = {
            'healthy': { 
                color: 'bg-green-400', 
                text: '系统正常',
                pulse: true 
            },
            'degraded': { 
                color: 'bg-yellow-400', 
                text: '系统降级',
                pulse: true 
            },
            'unhealthy': { 
                color: 'bg-red-400', 
                text: '系统异常',
                pulse: true 
            },
            'unknown': { 
                color: 'bg-gray-400', 
                text: '状态未知',
                pulse: false 
            }
        };
        
        const config = statusConfig[status] || statusConfig['unknown'];
        
        indicator.className = `w-3 h-3 ${config.color} rounded-full ${config.pulse ? 'pulse-dot' : ''}`;
        text.textContent = config.text;
        
        // 存储状态用于其他组件
        this.systemStatus = health;
    }
    
    updatePortfolioSummary(portfolio) {
        // 更新组合价值
        const portfolioValue = portfolio.portfolio_value || 0;
        this.updateElement('portfolio-value', this.formatCurrency(portfolioValue));
        
        // 更新日盈亏
        const dailyPnl = portfolio.daily_pnl || 0;
        const dailyPnlPct = portfolio.daily_return_pct || 0;
        
        this.updateElement('portfolio-change', 
            `今日: ${this.formatCurrency(dailyPnl)} (${this.formatPercentage(dailyPnlPct)})`);
        
        // 更新持仓数量
        const positionsCount = (portfolio.positions || []).length;
        this.updateElement('positions-count', positionsCount.toString());
        
        // 更新日盈亏显示和颜色
        const dailyPnlElement = document.getElementById('daily-pnl');
        if (dailyPnlElement) {
            dailyPnlElement.textContent = this.formatCurrency(dailyPnl);
            dailyPnlElement.className = `text-2xl font-bold ${
                dailyPnl >= 0 ? 'text-green-600' : 'text-red-600'
            }`;
        }
        
        this.updateElement('daily-pnl-pct', this.formatPercentage(dailyPnlPct));
        
        // 更新组合图表
        if (portfolio.history) {
            this.updatePortfolioChart(portfolio.history);
        }
        
        // 存储组合数据
        this.portfolioData = portfolio;
    }
    
    updateTradingStatus(trading) {
        const systemStatus = trading.system_status || 'not_started';
        const tradingMode = trading.trading_mode || 'none';
        
        // 状态文本映射
        const statusText = {
            'running': '运行中',
            'paused': '已暂停', 
            'stopped': '已停止',
            'not_started': '未启动'
        };
        
        const modeText = {
            'live': '实盘交易',
            'paper': '纸交易',
            'simulation': '模拟模式',
            'none': '未设置'
        };
        
        this.updateElement('trading-status', statusText[systemStatus] || '未知');
        this.updateElement('trading-mode', modeText[tradingMode] || '未知');
        
        // 更新状态图标
        const iconElement = document.getElementById('trading-status-icon');
        if (iconElement) {
            const iconConfig = {
                'running': 'fas fa-play text-3xl text-green-500',
                'paused': 'fas fa-pause text-3xl text-yellow-500',
                'stopped': 'fas fa-stop text-3xl text-red-500',
                'not_started': 'fas fa-pause text-3xl text-gray-400'
            };
            
            iconElement.className = iconConfig[systemStatus] || iconConfig['not_started'];
        }
        
        // 更新交易控制按钮状态
        this.updateTradingControlButtons(systemStatus);
        
        // 存储交易状态
        this.tradingStatus = trading;
    }
    
    updateTradingControlButtons(systemStatus) {
        const startBtn = document.getElementById('start-trading-btn');
        const stopBtn = document.getElementById('stop-trading-btn');
        const pauseBtn = document.getElementById('pause-trading-btn');
        
        if (!startBtn || !stopBtn || !pauseBtn) return;
        
        // 重置所有按钮
        [startBtn, stopBtn, pauseBtn].forEach(btn => {
            btn.disabled = false;
            btn.classList.remove('opacity-50', 'cursor-not-allowed');
        });
        
        // 根据状态禁用相应按钮
        switch (systemStatus) {
            case 'running':
                startBtn.disabled = true;
                startBtn.classList.add('opacity-50', 'cursor-not-allowed');
                break;
            case 'stopped':
            case 'not_started':
                stopBtn.disabled = true;
                pauseBtn.disabled = true;
                stopBtn.classList.add('opacity-50', 'cursor-not-allowed');
                pauseBtn.classList.add('opacity-50', 'cursor-not-allowed');
                break;
        }
    }
    
    updateSystemMetrics(metricsData) {
        if (!metricsData || !metricsData.metrics) return;
        
        const metrics = metricsData.metrics;
        
        // 更新CPU使用率
        if (metrics.cpu_usage) {
            const cpuUsage = Math.round(metrics.cpu_usage.value || 0);
            this.updateElement('cpu-usage', `${cpuUsage}%`);
            this.updateProgressBar('cpu-bar', cpuUsage);
        }
        
        // 更新内存使用率
        if (metrics.memory_usage) {
            const memoryUsage = Math.round(metrics.memory_usage.value || 0);
            this.updateElement('memory-usage', `${memoryUsage}%`);
            this.updateProgressBar('memory-bar', memoryUsage);
        }
        
        // 更新磁盘使用率
        if (metrics.disk_usage) {
            const diskUsage = Math.round(metrics.disk_usage.value || 0);
            this.updateElement('disk-usage', `${diskUsage}%`);
            this.updateProgressBar('disk-bar', diskUsage);
        }
        
        // 更新系统监控图表
        this.updateSystemChart(metrics);
    }
    
    updateLatestSignals(signalsData) {
        const container = document.getElementById('latest-signals');
        if (!container) return;
        
        const signals = signalsData.signals || [];
        
        if (signals.length === 0) {
            container.innerHTML = `
                <div class="flex items-center justify-between text-sm text-gray-500">
                    <span>暂无最新信号</span>
                </div>
            `;
            return;
        }
        
        container.innerHTML = signals.map(signal => `
            <div class="flex items-center justify-between py-2 border-b border-gray-100 last:border-b-0">
                <div class="flex items-center space-x-3">
                    <div class="w-2 h-2 rounded-full ${this.getSignalColor(signal.signal_type)}"></div>
                    <div>
                        <div class="text-sm font-medium text-gray-900">${signal.symbol}</div>
                        <div class="text-xs text-gray-500">${signal.source}</div>
                    </div>
                </div>
                <div class="text-right">
                    <div class="text-sm font-medium ${this.getSignalTextColor(signal.signal_type)}">
                        ${this.formatSignalType(signal.signal_type)}
                    </div>
                    <div class="text-xs text-gray-500">
                        ${this.formatPercentage(signal.confidence)} 信心度
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    updateLatestAlerts(alertsData) {
        const container = document.getElementById('latest-alerts');
        if (!container) return;
        
        const alerts = alertsData.alerts || [];
        
        if (alerts.length === 0) {
            container.innerHTML = `
                <div class="flex items-center justify-between text-sm text-gray-500">
                    <span>系统运行正常，无告警</span>
                    <i class="fas fa-check-circle text-green-500"></i>
                </div>
            `;
            return;
        }
        
        container.innerHTML = alerts.map(alert => `
            <div class="flex items-center justify-between py-2 border-b border-gray-100 last:border-b-0">
                <div class="flex items-center space-x-3">
                    <i class="fas fa-exclamation-triangle ${this.getAlertColor(alert.severity)}"></i>
                    <div>
                        <div class="text-sm font-medium text-gray-900">${alert.title || alert.message}</div>
                        <div class="text-xs text-gray-500">${this.getRelativeTime(alert.created_at)}</div>
                    </div>
                </div>
                <div class="text-right">
                    <span class="text-xs px-2 py-1 rounded-full ${this.getAlertBadgeClass(alert.severity)}">
                        ${alert.severity.toUpperCase()}
                    </span>
                </div>
            </div>
        `).join('');
    }
    
    // ========== 图表管理 ==========
    
    initializeCharts() {
        // 初始化组合价值趋势图
        this.initializePortfolioChart();
        
        // 初始化系统监控图
        this.initializeSystemChart();
    }
    
    initializePortfolioChart() {
        const canvas = document.getElementById('portfolio-chart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        this.charts.portfolio = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: '组合价值',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: '#3b82f620',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: {
                            color: '#f3f4f6'
                        }
                    },
                    y: {
                        display: true,
                        grid: {
                            color: '#f3f4f6'
                        },
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    }
                },
                elements: {
                    point: {
                        radius: 0,
                        hoverRadius: 4
                    }
                }
            }
        });
    }
    
    initializeSystemChart() {
        const canvas = document.getElementById('system-chart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        this.charts.system = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['CPU', '内存', '磁盘'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: [
                        '#3b82f6',
                        '#10b981',
                        '#8b5cf6'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    updatePortfolioChart(historyData) {
        if (!this.charts.portfolio || !historyData) return;
        
        const values = historyData.values || [];
        const dates = historyData.dates || [];
        
        // 格式化日期标签
        const labels = dates.map(date => {
            return new Date(date).toLocaleDateString();
        });
        
        this.charts.portfolio.data.labels = labels;
        this.charts.portfolio.data.datasets[0].data = values;
        this.charts.portfolio.update('none');
    }
    
    updateSystemChart(metrics) {
        if (!this.charts.system) return;
        
        const cpuUsage = metrics.cpu_usage?.value || 0;
        const memoryUsage = metrics.memory_usage?.value || 0;
        const diskUsage = metrics.disk_usage?.value || 0;
        
        this.charts.system.data.datasets[0].data = [cpuUsage, memoryUsage, diskUsage];
        this.charts.system.update('none');
    }
    
    resizeCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart) {
                chart.resize();
            }
        });
    }
    
    // ========== WebSocket 管理 ==========
    
    connectWebSocket() {
        try {
            const wsUrl = `ws://${window.location.host}/ws`;
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = (event) => {
                console.log('WebSocket 连接已建立');
                this.showNotification('实时连接已建立', 'success');
                
                // 订阅数据频道
                this.subscribeToChannels();
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('WebSocket 消息解析错误:', error);
                }
            };
            
            this.websocket.onclose = (event) => {
                console.log('WebSocket 连接已关闭');
                
                // 自动重连
                setTimeout(() => {
                    this.connectWebSocket();
                }, this.config.websocketReconnectDelay);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket 错误:', error);
            };
            
        } catch (error) {
            console.error('WebSocket 连接失败:', error);
        }
    }
    
    subscribeToChannels() {
        if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) return;
        
        // 订阅组合更新
        this.websocket.send(JSON.stringify({
            type: 'subscribe',
            channel: 'portfolio'
        }));
        
        // 订阅系统指标
        this.websocket.send(JSON.stringify({
            type: 'subscribe',
            channel: 'system_metrics'
        }));
        
        // 订阅信号更新
        this.websocket.send(JSON.stringify({
            type: 'subscribe',
            channel: 'signals'
        }));
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'portfolio_update':
                this.updatePortfolioSummary(data.data);
                break;
                
            case 'system_metrics':
                this.updateSystemMetrics(data.data);
                break;
                
            case 'new_signal':
                this.handleNewSignal(data.data);
                break;
                
            case 'trading_status_update':
                this.updateTradingStatus(data.data);
                break;
                
            case 'alert':
                this.handleAlert(data.data);
                break;
                
            default:
                console.log('未处理的WebSocket消息:', data);
        }
    }
    
    handleNewSignal(signal) {
        // 显示新信号通知
        const signalText = `${signal.symbol} ${this.formatSignalType(signal.signal_type)}`;
        this.showNotification(`新信号: ${signalText}`, 'info');
        
        // 刷新信号列表
        this.loadLatestSignals();
    }
    
    handleAlert(alert) {
        // 显示告警通知
        const severity = alert.severity || 'info';
        const notificationType = severity === 'critical' ? 'error' : 'warning';
        
        this.showNotification(alert.message, notificationType);
        
        // 刷新告警列表
        this.loadLatestAlerts();
    }
    
    // ========== 定时更新管理 ==========
    
    setupPeriodicUpdates() {
        // 主仪表盘数据更新
        this.updateIntervals.main = setInterval(() => {
            this.updateMainDashboard();
        }, this.config.updateInterval);
        
        // 图表数据更新
        this.updateIntervals.charts = setInterval(() => {
            this.updateChartData();
        }, this.config.chartUpdateInterval);
    }
    
    async updateMainDashboard() {
        if (document.hidden) return; // 页面隐藏时跳过更新
        
        try {
            // 并行更新多个数据源
            await Promise.allSettled([
                this.loadPortfolioSummary(),
                this.loadTradingStatus(),
                this.loadLatestSignals(),
                this.loadLatestAlerts()
            ]);
        } catch (error) {
            console.error('定时更新失败:', error);
        }
    }
    
    async updateChartData() {
        if (document.hidden) return;
        
        try {
            await this.loadCurrentMetrics();
        } catch (error) {
            console.error('图表更新失败:', error);
        }
    }
    
    pauseUpdates() {
        Object.values(this.updateIntervals).forEach(interval => {
            if (interval) clearInterval(interval);
        });
        this.updateIntervals = {};
    }
    
    resumeUpdates() {
        this.setupPeriodicUpdates();
        // 立即执行一次更新
        this.updateMainDashboard();
    }
    
    // ========== 工具方法 ==========
    
    updateElement(id, content) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = content;
        }
    }
    
    updateProgressBar(id, percentage) {
        const bar = document.getElementById(id);
        if (bar) {
            bar.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
        }
    }
    
    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(amount);
    }
    
    formatPercentage(value) {
        return `${(value * 100).toFixed(2)}%`;
    }
    
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
    
    getSignalColor(signalType) {
        const colors = {
            'buy': 'bg-green-400',
            'sell': 'bg-red-400',
            'neutral': 'bg-gray-400',
            'hold': 'bg-blue-400'
        };
        return colors[signalType?.toLowerCase()] || 'bg-gray-400';
    }
    
    getSignalTextColor(signalType) {
        const colors = {
            'buy': 'text-green-600',
            'sell': 'text-red-600',
            'neutral': 'text-gray-600',
            'hold': 'text-blue-600'
        };
        return colors[signalType?.toLowerCase()] || 'text-gray-600';
    }
    
    formatSignalType(signalType) {
        const types = {
            'buy': '买入',
            'sell': '卖出',
            'neutral': '中性',
            'hold': '持有'
        };
        return types[signalType?.toLowerCase()] || signalType;
    }
    
    getAlertColor(severity) {
        const colors = {
            'low': 'text-blue-500',
            'medium': 'text-yellow-500',
            'high': 'text-orange-500',
            'critical': 'text-red-500'
        };
        return colors[severity] || 'text-gray-500';
    }
    
    getAlertBadgeClass(severity) {
        const classes = {
            'low': 'bg-blue-100 text-blue-800',
            'medium': 'bg-yellow-100 text-yellow-800',
            'high': 'bg-orange-100 text-orange-800',
            'critical': 'bg-red-100 text-red-800'
        };
        return classes[severity] || 'bg-gray-100 text-gray-800';
    }
    
    showNotification(message, type = 'info', duration = null) {
        const notificationArea = document.getElementById('notification-area');
        if (!notificationArea) return;
        
        const notification = document.createElement('div');
        const id = Date.now().toString();
        
        const bgColors = {
            'success': 'bg-green-500',
            'error': 'bg-red-500',
            'warning': 'bg-yellow-500',
            'info': 'bg-blue-500'
        };
        
        const icons = {
            'success': 'fas fa-check-circle',
            'error': 'fas fa-exclamation-triangle',
            'warning': 'fas fa-exclamation-circle',
            'info': 'fas fa-info-circle'
        };
        
        notification.className = `${bgColors[type] || bgColors.info} text-white px-4 py-3 rounded-md shadow-lg max-w-sm mb-2 flex items-center space-x-2`;
        notification.innerHTML = `
            <i class="${icons[type] || icons.info}"></i>
            <span class="flex-1">${message}</span>
            <button class="text-white hover:text-gray-200" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        notificationArea.appendChild(notification);
        
        // 自动移除
        const autoRemoveTime = duration || this.config.notificationDuration;
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, autoRemoveTime);
    }
    
    // 获取空数据对象
    getEmptyPortfolio() {
        return {
            portfolio_value: 0,
            initial_capital: 0,
            total_pnl: 0,
            total_return_pct: 0,
            daily_pnl: 0,
            daily_return_pct: 0,
            cash: 0,
            positions: [],
            risk_metrics: {},
            status: 'not_trading'
        };
    }
    
    getEmptyTradingStatus() {
        return {
            system_status: 'not_started',
            trading_mode: 'none',
            active_symbols: [],
            strategies_running: 0,
            auto_trading: false,
            config: {}
        };
    }
    
    // 清理资源
    cleanup() {
        // 清理定时器
        Object.values(this.updateIntervals).forEach(interval => {
            if (interval) clearInterval(interval);
        });
        
        // 关闭WebSocket
        if (this.websocket) {
            this.websocket.close();
        }
        
        // 销毁图表
        Object.values(this.charts).forEach(chart => {
            if (chart) chart.destroy();
        });
        
        console.log('仪表盘资源已清理');
    }
}

// ========== 全局函数 ==========

// 页面切换函数
function showSection(sectionName) {
    // 隐藏所有部分
    document.querySelectorAll('.section').forEach(section => {
        section.classList.add('hidden');
    });
    
    // 显示目标部分
    const targetSection = document.getElementById(`${sectionName}-section`);
    if (targetSection) {
        targetSection.classList.remove('hidden');
    }
    
    // 更新导航状态
    document.querySelectorAll('nav a').forEach(link => {
        link.classList.remove('text-blue-200', 'font-medium');
        link.classList.add('text-white');
    });
    
    // 如果是通过导航点击触发的，高亮当前导航
    if (event && event.target) {
        event.target.classList.add('text-blue-200', 'font-medium');
    }
}

// 交易控制函数
async function controlTrading(action) {
    if (!window.dashboard || !window.dashboard.isInitialized) {
        dashboard.showNotification('系统尚未初始化完成', 'warning');
        return;
    }
    
    try {
        const result = await window.dashboard.apiClient.controlTrading(action, null, `Manual ${action} from dashboard`);
        
        if (result.success) {
            dashboard.showNotification(`交易${action}成功`, 'success');
            // 1秒后更新状态
            setTimeout(async () => {
                await dashboard.loadTradingStatus();
            }, 1000);
        } else {
            const errorMsg = result.error?.message || result.message || '操作失败';
            dashboard.showNotification(`操作失败: ${errorMsg}`, 'error');
        }
    } catch (error) {
        console.error('交易控制失败:', error);
        dashboard.showNotification('操作失败，请检查网络连接', 'error');
    }
}

// 手动交易函数
async function executeManualTrade() {
    if (!window.dashboard || !window.dashboard.isInitialized) {
        dashboard.showNotification('系统尚未初始化完成', 'warning');
        return;
    }
    
    const symbol = document.getElementById('manual-symbol').value.trim().toUpperCase();
    const side = document.getElementById('manual-side').value;
    const size = parseFloat(document.getElementById('manual-size').value) || null;
    const stopLoss = parseFloat(document.getElementById('manual-stop-loss').value) || 2;
    
    if (!symbol) {
        dashboard.showNotification('请输入交易对', 'warning');
        return;
    }
    
    try {
        const tradeData = {
            symbol: symbol,
            side: side,
            size: size,
            stop_loss_pct: stopLoss / 100,
            order_type: 'market'
        };
        
        const result = await window.dashboard.apiClient.executeManualTrade(tradeData);
        
        if (result.success) {
            dashboard.showNotification(`手动交易执行成功: ${symbol} ${side}`, 'success');
            
            // 清空数量输入
            document.getElementById('manual-size').value = '';
            
            // 刷新组合数据
            setTimeout(async () => {
                await dashboard.loadPortfolioSummary();
            }, 1000);
        } else {
            const errorMsg = result.error?.message || result.message || '交易执行失败';
            dashboard.showNotification(`交易失败: ${errorMsg}`, 'error');
        }
    } catch (error) {
        console.error('手动交易失败:', error);
        dashboard.showNotification('交易执行失败，请检查网络连接', 'error');
    }
}

// 图表更新函数
function updateChart(timeRange) {
    console.log('更新图表时间范围:', timeRange);
    // 这里可以根据时间范围更新图表数据
    // 暂时只更新按钮样式
    document.querySelectorAll('[onclick*="updateChart"]').forEach(btn => {
        btn.classList.remove('text-blue-500', 'font-medium');
        btn.classList.add('text-gray-500');
    });
    
    event.target.classList.add('text-blue-500', 'font-medium');
    event.target.classList.remove('text-gray-500');
}

// 全局仪表盘实例
let dashboard = null;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', async function() {
    console.log('开始初始化Trading Dashboard...');
    
    try {
        // 创建仪表盘实例
        dashboard = new TradingDashboard();
        window.dashboard = dashboard;
        
        // 初始化仪表盘
        await dashboard.initialize();
        
        console.log('Trading Dashboard 初始化完成');
        
    } catch (error) {
        console.error('Trading Dashboard 初始化失败:', error);
        
        // 显示错误通知
        const notificationArea = document.getElementById('notification-area');
        if (notificationArea) {
            const errorNotification = document.createElement('div');
            errorNotification.className = 'bg-red-500 text-white px-4 py-3 rounded-md shadow-lg max-w-sm';
            errorNotification.textContent = '系统初始化失败，请刷新页面重试';
            notificationArea.appendChild(errorNotification);
        }
    }
});

console.log('Dashboard JavaScript 加载完成');