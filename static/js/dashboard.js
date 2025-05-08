/**
 * Dashboard JavaScript
 * This file contains all the functionality for the Solana Trading Bot dashboard
 */

// Global variables
let priceChart;
let priceData = {
    labels: [],
    datasets: [{
        label: 'SOL-USDT Price',
        backgroundColor: 'rgba(0, 255, 163, 0.1)',
        borderColor: '#00FFA3',
        data: [],
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 2,
        fill: true
    }]
};

// Initialize everything when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initPriceChart();
    setupEventHandlers();
    fetchBotStatus();
    
    // Set up periodic updates
    setInterval(fetchBotStatus, 10000);  // Update status every 10 seconds
    setInterval(() => fetchMarketPrice('SOL-USDT'), 5000);  // Update price every 5 seconds
    
    // Initial market analysis
    refreshMarketAnalysis('SOL-USDT', '15m');
});

/**
 * Initialize the price chart
 */
function initPriceChart() {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    priceChart = new Chart(ctx, {
        type: 'line',
        data: priceData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 500
            },
            scales: {
                x: {
                    ticks: {
                        maxTicksLimit: 8,
                        maxRotation: 0
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                },
                y: {
                    position: 'right',
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return `$${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Set up event handlers for buttons and forms
 */
function setupEventHandlers() {
    // Start bot button
    const startBotBtn = document.getElementById('startBotBtn');
    if (startBotBtn) {
        startBotBtn.addEventListener('click', function() {
            // Populate confirmation modal with current settings
            const symbol = document.getElementById('symbolSelect').value;
            const interval = document.getElementById('intervalSelect').value;
            const mode = document.getElementById('modeSelect').value;
            const notify = document.getElementById('notifyCheck').checked;
            
            document.getElementById('confirmSymbol').textContent = symbol;
            document.getElementById('confirmInterval').textContent = interval;
            document.getElementById('confirmMode').textContent = mode === 'paper' ? 'Paper Trading' : 'Live Trading';
            document.getElementById('confirmNotify').textContent = notify ? 'Enabled' : 'Disabled';
            
            // Show modal
            const modal = new bootstrap.Modal(document.getElementById('startBotModal'));
            modal.show();
        });
    }
    
    // Confirm start button in modal
    const confirmStartBtn = document.getElementById('confirmStartBtn');
    if (confirmStartBtn) {
        confirmStartBtn.addEventListener('click', function() {
            startBot();
            const modal = bootstrap.Modal.getInstance(document.getElementById('startBotModal'));
            modal.hide();
        });
    }
    
    // Stop bot button
    const stopBotBtn = document.getElementById('stopBotBtn');
    if (stopBotBtn) {
        stopBotBtn.addEventListener('click', function() {
            const modal = new bootstrap.Modal(document.getElementById('stopBotModal'));
            modal.show();
        });
    }
    
    // Confirm stop button in modal
    const confirmStopBtn = document.getElementById('confirmStopBtn');
    if (confirmStopBtn) {
        confirmStopBtn.addEventListener('click', function() {
            stopBot();
            const modal = bootstrap.Modal.getInstance(document.getElementById('stopBotModal'));
            modal.hide();
        });
    }
    
    // Interval buttons for chart
    document.querySelectorAll('[data-interval]').forEach(button => {
        button.addEventListener('click', function() {
            // Update active class
            document.querySelectorAll('[data-interval]').forEach(btn => {
                btn.classList.remove('active');
            });
            this.classList.add('active');
            
            // Get current symbol and new interval
            const symbol = document.getElementById('symbolSelect').value;
            const interval = this.getAttribute('data-interval');
            
            // Refresh analysis with new interval
            refreshMarketAnalysis(symbol, interval);
        });
    });
}

/**
 * Fetch bot status from the API
 */
function fetchBotStatus() {
    fetch('/api/bot/status')
        .then(response => response.json())
        .then(data => {
            updateBotStatusUI(data);
        })
        .catch(error => {
            console.error('Error fetching bot status:', error);
            showAlert('Error connecting to server', 'danger');
        });
}

/**
 * Update UI elements with bot status data
 */
function updateBotStatusUI(data) {
    // Update running status
    const isRunning = data.is_running === true;
    
    // Update status indicator
    const statusElement = document.querySelector('.trading-active-dot');
    if (statusElement) {
        if (isRunning) {
            statusElement.classList.add('pulse');
        } else {
            statusElement.classList.remove('pulse');
        }
    }
    
    // Update buttons
    const startBotBtn = document.getElementById('startBotBtn');
    const stopBotBtn = document.getElementById('stopBotBtn');
    
    if (startBotBtn && stopBotBtn) {
        startBotBtn.disabled = isRunning;
        stopBotBtn.disabled = !isRunning;
    }
    
    // Update trading signals if available
    if (data.strategy_signals) {
        updateIndicatorSignal('rsi', data.strategy_signals.rsi);
        updateIndicatorSignal('trend', data.strategy_signals.trend);
        updateIndicatorSignal('macd', data.strategy_signals.macd);
        updateIndicatorSignal('bb', data.strategy_signals.bollinger);
        
        // Update integrated signal
        const integratedSignal = document.getElementById('integratedSignal');
        if (integratedSignal) {
            let signalClass = 'bg-secondary';
            
            switch (data.integrated_signal) {
                case 'strong_buy':
                    signalClass = 'bg-success';
                    break;
                case 'buy':
                    signalClass = 'bg-info';
                    break;
                case 'strong_sell':
                    signalClass = 'bg-danger';
                    break;
                case 'sell':
                    signalClass = 'bg-warning';
                    break;
                default:
                    signalClass = 'bg-secondary';
            }
            
            integratedSignal.className = `signal-badge ${signalClass}`;
            integratedSignal.textContent = (data.integrated_signal || 'neutral').toUpperCase().replace('_', ' ');
        }
    }
    
    // Update ROI if available
    if (data.roi !== undefined) {
        const roiElement = document.querySelector('.stats-card p:nth-child(2)');
        if (roiElement) {
            const roiValue = parseFloat(data.roi);
            roiElement.className = `fw-bold fs-5 mb-0 ${roiValue > 0 ? 'positive-roi' : roiValue < 0 ? 'negative-roi' : ''}`;
            roiElement.textContent = `${roiValue.toFixed(2)}%`;
        }
    }
    
    // Update current position if available
    if (data.position) {
        // TODO: Update position display with actual data
    }
    
    // Update last update timestamp
    const lastUpdated = document.getElementById('lastUpdated');
    if (lastUpdated) {
        const now = new Date();
        lastUpdated.textContent = now.toLocaleTimeString();
    }
}

/**
 * Update an indicator signal display
 */
function updateIndicatorSignal(indicator, signal) {
    const element = document.getElementById(`${indicator}Signal`);
    if (element) {
        let signalClass = 'bg-secondary';
        
        switch (signal) {
            case 'buy':
                signalClass = 'bg-success';
                break;
            case 'sell':
                signalClass = 'bg-danger';
                break;
            default:
                signalClass = 'bg-secondary';
        }
        
        element.className = `signal-badge ${signalClass}`;
        element.textContent = (signal || 'neutral').toUpperCase();
    }
}

/**
 * Start the trading bot
 */
function changeTradingMode(newMode, confirmed = false) {
    // Función para cambiar entre modo papel y real (con confirmación para real)
    const data = {
        mode: newMode,
        confirm: confirmed
    };
    
    fetch('/api/change_mode', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Modo cambiado con éxito
            showAlert(data.message, newMode === 'live' ? 'warning' : 'success');
            
            // Actualizar UI para reflejar el nuevo modo
            const modeSelect = document.getElementById('modeSelect');
            if (modeSelect) {
                modeSelect.value = newMode;
            }
            
            // Si estamos en la página de dashboard, actualizar el estado del bot
            fetchBotStatus();
        } else if (data.needs_confirmation) {
            // Se requiere confirmación adicional para modo real
            if (confirm(data.message + "\n\n¿Está SEGURO de activar el trading con fondos REALES?")) {
                // Usuario confirmó, intentar de nuevo con el flag de confirmación
                changeTradingMode(newMode, true);
            }
        } else {
            // Error al cambiar el modo
            showAlert(data.message, 'danger');
        }
    })
    .catch(error => {
        console.error('Error changing trading mode:', error);
        showAlert('Error al cambiar el modo de trading', 'danger');
    });
}

// Función para actualizar el modo cuando cambia el select
function updateTradingMode() {
    const modeSelect = document.getElementById('modeSelect');
    if (modeSelect) {
        const newMode = modeSelect.value;
        if (newMode !== getCurrentTradingMode()) {
            changeTradingMode(newMode);
        }
    }
}

// Función para obtener el modo actual
function getCurrentTradingMode() {
    const modeSelect = document.getElementById('modeSelect');
    return modeSelect ? modeSelect.value : 'paper';
}

function startBot() {
    const symbol = document.getElementById('symbolSelect').value;
    const interval = document.getElementById('intervalSelect').value;
    const mode = document.getElementById('modeSelect').value;
    const notify = document.getElementById('notifyCheck').checked;
    
    // Si está intentando iniciar en modo real, primero verificar que el modo real está activo
    if (mode === 'live' && getCurrentTradingMode() !== 'live') {
        showAlert('Primero debe cambiar al modo de trading real antes de iniciar el bot en modo real', 'warning');
        return;
    }
    
    const data = {
        symbol: symbol,
        interval: interval,
        mode: mode,
        notify: notify
    };
    
    fetch('/api/bot/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('Bot started successfully', 'success');
            // Update UI immediately without waiting for the next status refresh
            document.getElementById('startBotBtn').disabled = true;
            document.getElementById('stopBotBtn').disabled = false;
            const statusDot = document.querySelector('.trading-inactive-dot');
            if (statusDot) {
                statusDot.className = 'trading-active-dot pulse';
            }
        } else {
            showAlert(`Failed to start bot: ${data.message}`, 'danger');
        }
    })
    .catch(error => {
        console.error('Error starting bot:', error);
        showAlert('Error connecting to server', 'danger');
    });
}

/**
 * Stop the trading bot
 */
function stopBot() {
    fetch('/api/bot/stop', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('Bot is stopping...', 'info');
            // Don't update UI immediately, wait for status refresh to confirm
        } else {
            showAlert(`Failed to stop bot: ${data.message}`, 'danger');
        }
    })
    .catch(error => {
        console.error('Error stopping bot:', error);
        showAlert('Error connecting to server', 'danger');
    });
}

/**
 * Fetch the current market price
 */
function fetchMarketPrice(symbol) {
    fetch(`/api/market/price?symbol=${symbol}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                addPriceDataPoint(data.price);
            }
        })
        .catch(error => {
            console.error('Error fetching price:', error);
        });
}

/**
 * Add a new price data point to the chart
 */
function addPriceDataPoint(price) {
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    
    // Add new data point
    priceData.labels.push(timeString);
    priceData.datasets[0].data.push(price);
    
    // Keep only the last 30 data points for a cleaner chart
    if (priceData.labels.length > 30) {
        priceData.labels.shift();
        priceData.datasets[0].data.shift();
    }
    
    // Update chart
    priceChart.update();
}

/**
 * Refresh market analysis
 */
function refreshMarketAnalysis(symbol, interval) {
    fetch(`/api/market/analyze?symbol=${symbol}&interval=${interval}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update signals
                updateIndicatorSignal('rsi', data.details.signals.rsi);
                updateIndicatorSignal('trend', data.details.signals.trend);
                updateIndicatorSignal('macd', data.details.signals.macd);
                updateIndicatorSignal('bb', data.details.signals.bollinger);
                
                // Update integrated signal
                const integratedSignal = document.getElementById('integratedSignal');
                if (integratedSignal) {
                    let signalClass = 'bg-secondary';
                    
                    switch (data.signal) {
                        case 'strong_buy':
                            signalClass = 'bg-success';
                            break;
                        case 'buy':
                            signalClass = 'bg-info';
                            break;
                        case 'strong_sell':
                            signalClass = 'bg-danger';
                            break;
                        case 'sell':
                            signalClass = 'bg-warning';
                            break;
                        default:
                            signalClass = 'bg-secondary';
                    }
                    
                    integratedSignal.className = `signal-badge ${signalClass}`;
                    integratedSignal.textContent = (data.signal || 'neutral').toUpperCase().replace('_', ' ');
                }
            }
        })
        .catch(error => {
            console.error('Error analyzing market:', error);
        });
}

/**
 * Show a bootstrap alert
 */
function showAlert(message, type = 'info') {
    const alertContainer = document.getElementById('alertContainer');
    const alertElement = document.createElement('div');
    alertElement.className = `alert alert-${type} alert-dismissible fade show`;
    alertElement.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertContainer.appendChild(alertElement);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertElement.classList.remove('show');
        setTimeout(() => {
            alertContainer.removeChild(alertElement);
        }, 500);
    }, 5000);
}