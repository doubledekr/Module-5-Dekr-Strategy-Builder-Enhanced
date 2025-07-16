// Dekr Strategy Builder Enhanced - Frontend Application
class DekrApp {
    constructor() {
        this.apiBase = window.location.origin + '/api';
        this.wsBase = window.location.origin.replace('http', 'ws') + '/ws';
        this.signalSocket = null;
        this.marketSocket = null;
        this.currentUser = { id: 1, tier: 3 }; // Mock user for demo
        this.strategies = [];
        this.signals = [];
        this.charts = {};
        
        this.init();
    }

    async init() {
        // Initialize the application
        await this.loadStrategies();
        this.setupWebSockets();
        this.setupEventListeners();
        this.updateUI();
    }

    // API Methods
    async apiCall(endpoint, method = 'GET', data = null) {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            }
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(`${this.apiBase}${endpoint}`, options);
            if (!response.ok) {
                throw new Error(`API call failed: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            this.showNotification('API Error: ' + error.message, 'danger');
            throw error;
        }
    }

    async loadStrategies() {
        try {
            const response = await this.apiCall(`/strategies/?user_id=${this.currentUser.id}`);
            this.strategies = response.strategies;
        } catch (error) {
            console.error('Error loading strategies:', error);
        }
    }

    async loadSignals(strategyId = null) {
        try {
            const endpoint = strategyId ? `/signals/?strategy_id=${strategyId}` : '/signals/';
            const response = await this.apiCall(endpoint);
            this.signals = response.signals;
        } catch (error) {
            console.error('Error loading signals:', error);
        }
    }

    async createStrategy(strategyData) {
        try {
            const response = await this.apiCall('/strategies/', 'POST', strategyData);
            await this.loadStrategies();
            this.showNotification('Strategy created successfully!', 'success');
            return response;
        } catch (error) {
            console.error('Error creating strategy:', error);
            throw error;
        }
    }

    async updateStrategy(strategyId, strategyData) {
        try {
            const response = await this.apiCall(`/strategies/${strategyId}`, 'PUT', strategyData);
            await this.loadStrategies();
            this.showNotification('Strategy updated successfully!', 'success');
            return response;
        } catch (error) {
            console.error('Error updating strategy:', error);
            throw error;
        }
    }

    async deleteStrategy(strategyId) {
        try {
            await this.apiCall(`/strategies/${strategyId}`, 'DELETE');
            await this.loadStrategies();
            this.showNotification('Strategy deleted successfully!', 'success');
        } catch (error) {
            console.error('Error deleting strategy:', error);
            throw error;
        }
    }

    async runBacktest(strategyId, symbol, startDate, endDate) {
        try {
            const backtestData = {
                symbol: symbol,
                start_date: startDate,
                end_date: endDate
            };
            const response = await this.apiCall(`/strategies/${strategyId}/backtest`, 'POST', backtestData);
            this.showNotification('Backtest completed successfully!', 'success');
            return response.backtest_result;
        } catch (error) {
            console.error('Error running backtest:', error);
            throw error;
        }
    }

    // WebSocket Methods
    setupWebSockets() {
        this.connectSignalSocket();
        this.connectMarketSocket();
    }

    connectSignalSocket() {
        try {
            this.signalSocket = new WebSocket(`${this.wsBase}/signals`);
            
            this.signalSocket.onopen = () => {
                console.log('Signal WebSocket connected');
                this.updateConnectionStatus('signals', true);
            };

            this.signalSocket.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleSignalMessage(message);
            };

            this.signalSocket.onclose = () => {
                console.log('Signal WebSocket disconnected');
                this.updateConnectionStatus('signals', false);
                // Reconnect after 5 seconds
                setTimeout(() => this.connectSignalSocket(), 5000);
            };

            this.signalSocket.onerror = (error) => {
                console.error('Signal WebSocket error:', error);
                this.updateConnectionStatus('signals', false);
            };
        } catch (error) {
            console.error('Error connecting to signal WebSocket:', error);
        }
    }

    connectMarketSocket() {
        try {
            this.marketSocket = new WebSocket(`${this.wsBase}/market-data`);
            
            this.marketSocket.onopen = () => {
                console.log('Market WebSocket connected');
                this.updateConnectionStatus('market', true);
            };

            this.marketSocket.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleMarketMessage(message);
            };

            this.marketSocket.onclose = () => {
                console.log('Market WebSocket disconnected');
                this.updateConnectionStatus('market', false);
                // Reconnect after 5 seconds
                setTimeout(() => this.connectMarketSocket(), 5000);
            };

            this.marketSocket.onerror = (error) => {
                console.error('Market WebSocket error:', error);
                this.updateConnectionStatus('market', false);
            };
        } catch (error) {
            console.error('Error connecting to market WebSocket:', error);
        }
    }

    handleSignalMessage(message) {
        switch (message.type) {
            case 'signal':
                this.addNewSignal(message.data);
                break;
            case 'subscription_confirmed':
                console.log('Subscribed to strategy:', message.strategy_id);
                break;
            case 'pong':
                // Handle ping/pong
                break;
            default:
                console.log('Unknown signal message:', message);
        }
    }

    handleMarketMessage(message) {
        switch (message.type) {
            case 'market_status':
                this.updateMarketStatus(message.data);
                break;
            default:
                console.log('Unknown market message:', message);
        }
    }

    subscribeToStrategy(strategyId) {
        if (this.signalSocket && this.signalSocket.readyState === WebSocket.OPEN) {
            this.signalSocket.send(JSON.stringify({
                type: 'subscribe',
                strategy_id: strategyId
            }));
        }
    }

    unsubscribeFromStrategy(strategyId) {
        if (this.signalSocket && this.signalSocket.readyState === WebSocket.OPEN) {
            this.signalSocket.send(JSON.stringify({
                type: 'unsubscribe',
                strategy_id: strategyId
            }));
        }
    }

    // UI Methods
    updateUI() {
        this.updateStrategiesDisplay();
        this.updateSignalsDisplay();
        this.updateDashboard();
    }

    updateStrategiesDisplay() {
        const container = document.getElementById('strategies-container');
        if (!container) return;

        if (this.strategies.length === 0) {
            container.innerHTML = `
                <div class="text-center py-5">
                    <i data-feather="trending-up" class="mb-3" style="width: 64px; height: 64px; color: #6c757d;"></i>
                    <h5 class="text-muted">No strategies yet</h5>
                    <p class="text-muted">Create your first trading strategy to get started</p>
                    <a href="/strategy-builder" class="btn btn-primary">Create Strategy</a>
                </div>
            `;
            feather.replace();
            return;
        }

        const strategiesHtml = this.strategies.map(strategy => `
            <div class="col-md-6 col-lg-4 mb-4">
                <div class="card strategy-card h-100">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start mb-3">
                            <h5 class="card-title">${strategy.name}</h5>
                            <span class="badge ${strategy.is_active ? 'bg-success' : 'bg-secondary'}">
                                ${strategy.is_active ? 'Active' : 'Inactive'}
                            </span>
                        </div>
                        <p class="card-text text-muted">${strategy.description}</p>
                        <div class="mb-3">
                            <small class="text-muted">Type: ${strategy.strategy_type}</small><br>
                            <small class="text-muted">Symbols: ${strategy.symbols.join(', ')}</small>
                        </div>
                        <div class="d-flex justify-content-between">
                            <button class="btn btn-sm btn-outline-primary" onclick="dekr.editStrategy('${strategy.id}')">
                                <i data-feather="edit-2" class="me-1"></i>Edit
                            </button>
                            <button class="btn btn-sm btn-outline-info" onclick="dekr.viewStrategy('${strategy.id}')">
                                <i data-feather="eye" class="me-1"></i>View
                            </button>
                            <button class="btn btn-sm btn-outline-danger" onclick="dekr.confirmDeleteStrategy('${strategy.id}')">
                                <i data-feather="trash-2" class="me-1"></i>Delete
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');

        container.innerHTML = strategiesHtml;
        feather.replace();
    }

    updateSignalsDisplay() {
        const container = document.getElementById('signals-container');
        if (!container) return;

        if (this.signals.length === 0) {
            container.innerHTML = `
                <div class="text-center py-5">
                    <i data-feather="activity" class="mb-3" style="width: 64px; height: 64px; color: #6c757d;"></i>
                    <h5 class="text-muted">No signals yet</h5>
                    <p class="text-muted">Signals will appear here when your active strategies generate them</p>
                </div>
            `;
            feather.replace();
            return;
        }

        const signalsHtml = this.signals.map(signal => `
            <div class="col-md-6 col-lg-4 mb-4">
                <div class="card signal-card h-100">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start mb-3">
                            <h6 class="card-title">${signal.symbol}</h6>
                            <span class="badge ${this.getSignalBadgeClass(signal.signal_type)}">
                                ${signal.signal_type.toUpperCase()}
                            </span>
                        </div>
                        <p class="card-text">
                            <strong>Price:</strong> $${signal.price.toFixed(2)}<br>
                            <strong>Confidence:</strong> ${(signal.confidence * 100).toFixed(1)}%
                        </p>
                        <div class="confidence-meter mb-3">
                            <div class="confidence-fill" style="width: ${signal.confidence * 100}%"></div>
                        </div>
                        <small class="text-muted">
                            ${new Date(signal.timestamp).toLocaleString()}
                        </small>
                    </div>
                </div>
            </div>
        `).join('');

        container.innerHTML = signalsHtml;
    }

    updateDashboard() {
        // Update dashboard statistics
        const activeStrategies = this.strategies.filter(s => s.is_active).length;
        const totalSignals = this.signals.length;
        const buySignals = this.signals.filter(s => s.signal_type === 'buy').length;
        const sellSignals = this.signals.filter(s => s.signal_type === 'sell').length;

        const statsContainer = document.getElementById('dashboard-stats');
        if (statsContainer) {
            statsContainer.innerHTML = `
                <div class="col-md-3 mb-4">
                    <div class="card">
                        <div class="card-body text-center">
                            <i data-feather="trending-up" class="mb-2" style="width: 32px; height: 32px; color: #28a745;"></i>
                            <h5 class="card-title">${activeStrategies}</h5>
                            <p class="card-text text-muted">Active Strategies</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-4">
                    <div class="card">
                        <div class="card-body text-center">
                            <i data-feather="activity" class="mb-2" style="width: 32px; height: 32px; color: #007bff;"></i>
                            <h5 class="card-title">${totalSignals}</h5>
                            <p class="card-text text-muted">Total Signals</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-4">
                    <div class="card">
                        <div class="card-body text-center">
                            <i data-feather="arrow-up" class="mb-2" style="width: 32px; height: 32px; color: #28a745;"></i>
                            <h5 class="card-title">${buySignals}</h5>
                            <p class="card-text text-muted">Buy Signals</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-4">
                    <div class="card">
                        <div class="card-body text-center">
                            <i data-feather="arrow-down" class="mb-2" style="width: 32px; height: 32px; color: #dc3545;"></i>
                            <h5 class="card-title">${sellSignals}</h5>
                            <p class="card-text text-muted">Sell Signals</p>
                        </div>
                    </div>
                </div>
            `;
            feather.replace();
        }
    }

    addNewSignal(signal) {
        this.signals.unshift(signal);
        this.updateSignalsDisplay();
        this.updateDashboard();
        
        // Show notification
        this.showNotification(`New ${signal.signal_type.toUpperCase()} signal for ${signal.symbol} at $${signal.price.toFixed(2)}`, 'info');
        
        // Play sound if enabled
        this.playNotificationSound();
    }

    updateConnectionStatus(type, connected) {
        const indicator = document.getElementById(`${type}-status`);
        if (indicator) {
            indicator.className = `real-time-indicator ${connected ? 'text-success' : 'text-danger'}`;
            indicator.innerHTML = `
                <span class="real-time-dot ${connected ? 'bg-success' : 'bg-danger'}"></span>
                ${connected ? 'Connected' : 'Disconnected'}
            `;
        }
    }

    updateMarketStatus(data) {
        const container = document.getElementById('market-status');
        if (container && data.market) {
            const isOpen = data.market === 'open';
            container.innerHTML = `
                <span class="badge ${isOpen ? 'bg-success' : 'bg-secondary'}">
                    Market ${isOpen ? 'Open' : 'Closed'}
                </span>
            `;
        }
    }

    // Strategy Management
    editStrategy(strategyId) {
        const strategy = this.strategies.find(s => s.id === strategyId);
        if (strategy) {
            // Store strategy in session storage and redirect to builder
            sessionStorage.setItem('editStrategy', JSON.stringify(strategy));
            window.location.href = '/strategy-builder';
        }
    }

    viewStrategy(strategyId) {
        const strategy = this.strategies.find(s => s.id === strategyId);
        if (strategy) {
            this.showStrategyModal(strategy);
        }
    }

    confirmDeleteStrategy(strategyId) {
        const strategy = this.strategies.find(s => s.id === strategyId);
        if (strategy) {
            if (confirm(`Are you sure you want to delete the strategy "${strategy.name}"?`)) {
                this.deleteStrategy(strategyId);
            }
        }
    }

    // Chart Methods
    createChart(containerId, type, data, options = {}) {
        const ctx = document.getElementById(containerId);
        if (!ctx) return null;

        // Destroy existing chart if it exists
        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        const defaultOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        };

        const chart = new Chart(ctx, {
            type: type,
            data: data,
            options: { ...defaultOptions, ...options }
        });

        this.charts[containerId] = chart;
        return chart;
    }

    // Utility Methods
    getSignalBadgeClass(signalType) {
        switch (signalType) {
            case 'buy':
            case 'strong_buy':
                return 'bg-success';
            case 'sell':
            case 'strong_sell':
                return 'bg-danger';
            default:
                return 'bg-warning';
        }
    }

    showNotification(message, type = 'info') {
        const alertClass = `alert-${type}`;
        const alertHtml = `
            <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;

        const container = document.getElementById('notifications') || document.body;
        const alertElement = document.createElement('div');
        alertElement.innerHTML = alertHtml;
        container.appendChild(alertElement);

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alert = alertElement.querySelector('.alert');
            if (alert) {
                new bootstrap.Alert(alert).close();
            }
        }, 5000);
    }

    showStrategyModal(strategy) {
        const modalHtml = `
            <div class="modal fade" id="strategyModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">${strategy.name}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <p><strong>Description:</strong> ${strategy.description}</p>
                            <p><strong>Type:</strong> ${strategy.strategy_type}</p>
                            <p><strong>Symbols:</strong> ${strategy.symbols.join(', ')}</p>
                            <p><strong>Status:</strong> ${strategy.is_active ? 'Active' : 'Inactive'}</p>
                            
                            <h6>Buy Conditions:</h6>
                            <ul>
                                ${strategy.buy_conditions.map(c => `<li>${c.indicator} ${c.operator} ${c.value}</li>`).join('')}
                            </ul>
                            
                            <h6>Sell Conditions:</h6>
                            <ul>
                                ${strategy.sell_conditions.map(c => `<li>${c.indicator} ${c.operator} ${c.value}</li>`).join('')}
                            </ul>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            <button type="button" class="btn btn-primary" onclick="dekr.editStrategy('${strategy.id}')">Edit</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Remove existing modal
        const existingModal = document.getElementById('strategyModal');
        if (existingModal) {
            existingModal.remove();
        }

        // Add new modal
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        const modal = new bootstrap.Modal(document.getElementById('strategyModal'));
        modal.show();
    }

    playNotificationSound() {
        // Simple notification sound
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        oscillator.frequency.value = 800;
        oscillator.type = 'sine';
        gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);

        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.1);
    }

    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    }

    formatPercentage(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'percent',
            minimumFractionDigits: 2
        }).format(value);
    }

    // Event Listeners
    setupEventListeners() {
        // Keep WebSocket connections alive
        setInterval(() => {
            if (this.signalSocket && this.signalSocket.readyState === WebSocket.OPEN) {
                this.signalSocket.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000); // Ping every 30 seconds

        // Handle page visibility change
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                // Page is hidden, reduce activity
            } else {
                // Page is visible, resume normal activity
                this.loadStrategies();
                this.loadSignals();
            }
        });
    }
}

// Initialize the application
let dekr;
document.addEventListener('DOMContentLoaded', () => {
    dekr = new DekrApp();
    
    // Replace feather icons
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
});

// Strategy Builder specific functions
class StrategyBuilder {
    constructor() {
        this.conditions = [];
        this.currentStrategy = null;
        this.init();
    }

    init() {
        this.loadEditStrategy();
        this.setupEventListeners();
    }

    loadEditStrategy() {
        const editStrategy = sessionStorage.getItem('editStrategy');
        if (editStrategy) {
            this.currentStrategy = JSON.parse(editStrategy);
            this.populateForm(this.currentStrategy);
            sessionStorage.removeItem('editStrategy');
        }
    }

    populateForm(strategy) {
        document.getElementById('strategy-name').value = strategy.name;
        document.getElementById('strategy-description').value = strategy.description;
        document.getElementById('strategy-type').value = strategy.strategy_type;
        document.getElementById('symbols').value = strategy.symbols.join(', ');
        
        // Populate conditions
        this.conditions = [...strategy.buy_conditions, ...strategy.sell_conditions];
        this.updateConditionsDisplay();
    }

    addCondition(type) {
        const condition = {
            type: type,
            indicator: 'sma',
            operator: '>',
            value: 0,
            timeframe: '1D',
            parameters: {}
        };
        
        this.conditions.push(condition);
        this.updateConditionsDisplay();
    }

    removeCondition(index) {
        this.conditions.splice(index, 1);
        this.updateConditionsDisplay();
    }

    updateConditionsDisplay() {
        const container = document.getElementById('conditions-container');
        if (!container) return;

        const conditionsHtml = this.conditions.map((condition, index) => `
            <div class="card mb-3">
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-2">
                            <select class="form-select" onchange="strategyBuilder.updateCondition(${index}, 'type', this.value)">
                                <option value="buy" ${condition.type === 'buy' ? 'selected' : ''}>Buy</option>
                                <option value="sell" ${condition.type === 'sell' ? 'selected' : ''}>Sell</option>
                            </select>
                        </div>
                        <div class="col-md-2">
                            <select class="form-select" onchange="strategyBuilder.updateCondition(${index}, 'indicator', this.value)">
                                <option value="sma" ${condition.indicator === 'sma' ? 'selected' : ''}>SMA</option>
                                <option value="ema" ${condition.indicator === 'ema' ? 'selected' : ''}>EMA</option>
                                <option value="rsi" ${condition.indicator === 'rsi' ? 'selected' : ''}>RSI</option>
                                <option value="macd" ${condition.indicator === 'macd' ? 'selected' : ''}>MACD</option>
                                <option value="bollinger" ${condition.indicator === 'bollinger' ? 'selected' : ''}>Bollinger Bands</option>
                            </select>
                        </div>
                        <div class="col-md-2">
                            <select class="form-select" onchange="strategyBuilder.updateCondition(${index}, 'operator', this.value)">
                                <option value=">" ${condition.operator === '>' ? 'selected' : ''}>></option>
                                <option value="<" ${condition.operator === '<' ? 'selected' : ''}><</option>
                                <option value=">=" ${condition.operator === '>=' ? 'selected' : ''}>>=</option>
                                <option value="<=" ${condition.operator === '<=' ? 'selected' : ''}><=</option>
                                <option value="crosses_above" ${condition.operator === 'crosses_above' ? 'selected' : ''}>Crosses Above</option>
                                <option value="crosses_below" ${condition.operator === 'crosses_below' ? 'selected' : ''}>Crosses Below</option>
                            </select>
                        </div>
                        <div class="col-md-2">
                            <input type="number" class="form-control" value="${condition.value}" 
                                   onchange="strategyBuilder.updateCondition(${index}, 'value', parseFloat(this.value))">
                        </div>
                        <div class="col-md-2">
                            <select class="form-select" onchange="strategyBuilder.updateCondition(${index}, 'timeframe', this.value)">
                                <option value="1D" ${condition.timeframe === '1D' ? 'selected' : ''}>1 Day</option>
                                <option value="1H" ${condition.timeframe === '1H' ? 'selected' : ''}>1 Hour</option>
                                <option value="15min" ${condition.timeframe === '15min' ? 'selected' : ''}>15 Min</option>
                            </select>
                        </div>
                        <div class="col-md-2">
                            <button class="btn btn-danger btn-sm" onclick="strategyBuilder.removeCondition(${index})">
                                <i data-feather="x"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');

        container.innerHTML = conditionsHtml;
        feather.replace();
    }

    updateCondition(index, field, value) {
        if (this.conditions[index]) {
            this.conditions[index][field] = value;
        }
    }

    async saveStrategy() {
        const name = document.getElementById('strategy-name').value;
        const description = document.getElementById('strategy-description').value;
        const strategyType = document.getElementById('strategy-type').value;
        const symbols = document.getElementById('symbols').value.split(',').map(s => s.trim());

        const buyConditions = this.conditions.filter(c => c.type === 'buy');
        const sellConditions = this.conditions.filter(c => c.type === 'sell');

        const strategyData = {
            user_id: dekr.currentUser.id,
            name: name,
            description: description,
            strategy_type: strategyType,
            symbols: symbols,
            buy_conditions: buyConditions,
            sell_conditions: sellConditions,
            risk_management: {
                stop_loss: 0.05,
                take_profit: 0.10
            }
        };

        try {
            if (this.currentStrategy) {
                await dekr.updateStrategy(this.currentStrategy.id, strategyData);
            } else {
                await dekr.createStrategy(strategyData);
            }
            
            window.location.href = '/';
        } catch (error) {
            console.error('Error saving strategy:', error);
        }
    }

    setupEventListeners() {
        const saveButton = document.getElementById('save-strategy');
        if (saveButton) {
            saveButton.addEventListener('click', () => this.saveStrategy());
        }
    }
}

// Backtesting specific functions
class BacktestingInterface {
    constructor() {
        this.currentResults = null;
        this.init();
    }

    init() {
        this.loadStrategies();
        this.setupEventListeners();
    }

    async loadStrategies() {
        if (dekr && dekr.strategies) {
            const select = document.getElementById('strategy-select');
            if (select) {
                select.innerHTML = '<option value="">Select a strategy</option>' +
                    dekr.strategies.map(s => `<option value="${s.id}">${s.name}</option>`).join('');
            }
        }
    }

    async runBacktest() {
        const strategyId = document.getElementById('strategy-select').value;
        const symbol = document.getElementById('symbol').value;
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;

        if (!strategyId || !symbol || !startDate || !endDate) {
            dekr.showNotification('Please fill in all fields', 'warning');
            return;
        }

        try {
            const loadingButton = document.getElementById('run-backtest');
            loadingButton.innerHTML = '<span class="loading-spinner"></span> Running...';
            loadingButton.disabled = true;

            const results = await dekr.runBacktest(strategyId, symbol, startDate, endDate);
            this.currentResults = results;
            this.displayResults(results);

            loadingButton.innerHTML = 'Run Backtest';
            loadingButton.disabled = false;
        } catch (error) {
            console.error('Error running backtest:', error);
            const loadingButton = document.getElementById('run-backtest');
            loadingButton.innerHTML = 'Run Backtest';
            loadingButton.disabled = false;
        }
    }

    displayResults(results) {
        const container = document.getElementById('backtest-results');
        if (!container) return;

        const resultsHtml = `
            <div class="card">
                <div class="card-header">
                    <h5>Backtest Results</h5>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <h6>Performance Metrics</h6>
                            <table class="table table-sm">
                                <tbody>
                                    <tr><td>Total Return</td><td>${dekr.formatPercentage(results.total_return)}</td></tr>
                                    <tr><td>Annualized Return</td><td>${dekr.formatPercentage(results.annualized_return)}</td></tr>
                                    <tr><td>Sharpe Ratio</td><td>${results.sharpe_ratio.toFixed(2)}</td></tr>
                                    <tr><td>Max Drawdown</td><td>${dekr.formatPercentage(results.max_drawdown)}</td></tr>
                                    <tr><td>Win Rate</td><td>${dekr.formatPercentage(results.win_rate)}</td></tr>
                                </tbody>
                            </table>
                        </div>
                        <div class="col-md-6">
                            <h6>Trade Statistics</h6>
                            <table class="table table-sm">
                                <tbody>
                                    <tr><td>Total Trades</td><td>${results.total_trades}</td></tr>
                                    <tr><td>Avg Trade Duration</td><td>${results.avg_trade_duration.toFixed(1)} days</td></tr>
                                    <tr><td>Profit Factor</td><td>${results.profit_factor.toFixed(2)}</td></tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                    
                    <div class="mt-4">
                        <h6>Recent Trades</h6>
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Entry Date</th>
                                        <th>Exit Date</th>
                                        <th>Entry Price</th>
                                        <th>Exit Price</th>
                                        <th>Return</th>
                                        <th>Duration</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${results.trades.slice(-10).map(trade => `
                                        <tr>
                                            <td>${new Date(trade.entry_date).toLocaleDateString()}</td>
                                            <td>${new Date(trade.exit_date).toLocaleDateString()}</td>
                                            <td>${dekr.formatCurrency(trade.entry_price)}</td>
                                            <td>${dekr.formatCurrency(trade.exit_price)}</td>
                                            <td class="${trade.return_pct > 0 ? 'text-success' : 'text-danger'}">
                                                ${dekr.formatPercentage(trade.return_pct)}
                                            </td>
                                            <td>${trade.duration_days} days</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        `;

        container.innerHTML = resultsHtml;
    }

    setupEventListeners() {
        const runButton = document.getElementById('run-backtest');
        if (runButton) {
            runButton.addEventListener('click', () => this.runBacktest());
        }
    }
}

// Initialize page-specific functionality
document.addEventListener('DOMContentLoaded', () => {
    const path = window.location.pathname;
    
    if (path.includes('strategy-builder')) {
        window.strategyBuilder = new StrategyBuilder();
    } else if (path.includes('backtesting')) {
        window.backtestingInterface = new BacktestingInterface();
    }
});
