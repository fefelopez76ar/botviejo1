/**
 * Dashboard JavaScript for Solana Trading Bot
 * 
 * Controls dashboard functionality, including:
 * - Real-time updates of trading data
 * - Starting/stopping the bot
 * - Price chart visualization
 * - Market analysis refreshing
 */

// Global variables
let priceChart = null;
let priceData = [];
let timeLabels = [];
const maxDataPoints = 30;

// DOM ready handler
document.addEventListener('DOMContentLoaded', function() {
    // Initialize price chart
    initPriceChart();
    
    // Set up button event handlers
    setupEventHandlers();
    
    // Initial data fetch
    fetchBotStatus();
    fetchMarketPrice('SOL-USDT');
    
    // Set up periodic updates
    setInterval(fetchBotStatus, 30000); // Update status every 30 seconds
    setInterval(() => fetchMarketPrice('SOL-USDT'), 10000); // Update price every 10 seconds
});

/**
 * Initialize the price chart
 */
function initPriceChart() {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    // Create initial empty chart
    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timeLabels,
            datasets: [{
                label: 'SOL Price (USDT)',
                data: priceData,
                borderColor: '#4C9AFF',
                backgroundColor: 'rgba(76, 154, 255, 0.1)',
                borderWidth: 2,
                pointRadius: 2,
                pointHoverRadius: 5,
                fill: true,
                tension: 0.2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: false
                }
            },
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false,
                },
                legend: {
                    display: true,
                    position: 'top',
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
            const startBotModal = new bootstrap.Modal(document.getElementById('startBotModal'));
            startBotModal.show();
        });
    }
    
    // Stop bot button
    const stopBotBtn = document.getElementById('stopBotBtn');
    if (stopBotBtn) {
        stopBotBtn.addEventListener('click', stopBot);
    }
    
    // Confirm start bot button in modal
    const confirmStartBot = document.getElementById('confirmStartBot');
    if (confirmStartBot) {
        confirmStartBot.addEventListener('click', startBot);
    }
    
    // Refresh analysis button
    const refreshAnalysisBtn = document.getElementById('refreshAnalysisBtn');
    if (refreshAnalysisBtn) {
        refreshAnalysisBtn.addEventListener('click', function() {
            const symbol = document.getElementById('tradingSymbol') ? 
                           document.getElementById('tradingSymbol').value : 'SOL-USDT';
            const interval = document.getElementById('tradingInterval') ?
                             document.getElementById('tradingInterval').value : '15m';
            
            refreshMarketAnalysis(symbol, interval);
        });
    }
}

/**
 * Fetch bot status from the API
 */
function fetchBotStatus() {
    fetch('/api/bot/status')
        .then(response => response.json())
        .then(data => {
            if (data) {
                updateBotStatusUI(data);
            }
        })
        .catch(error => console.error('Error fetching bot status:', error));
}

/**
 * Update UI elements with bot status data
 */
function updateBotStatusUI(data) {
    // Update current price if available
    if (data.current_price) {
        const priceDisplay = `$${parseFloat(data.current_price).toFixed(2)}`;
        
        // Update price in stats box (create it if it doesn't exist)
        const priceElements = document.querySelectorAll('h3, h5, span, div').forEach(el => {
            if (el.textContent.includes('$0.00') || 
                el.textContent.includes('$1.00') || 
                (el.textContent.startsWith('$') && el.textContent.length < 8)) {
                el.textContent = priceDisplay;
            }
        });
        
        // Add to price chart
        addPriceDataPoint(parseFloat(data.current_price));
    }
    
    // Update balance if available
    if (data.current_balance) {
        document.querySelectorAll('h3, span').forEach(el => {
            if (el.textContent.includes('$1000.00') || 
                (el.textContent.startsWith('$') && 
                 el.textContent.includes('.') && 
                 !el.textContent.includes('%'))) {
                el.textContent = `$${parseFloat(data.current_balance).toFixed(2)}`;
            }
        });
    }
    
    // Update ROI
    if (data.roi !== undefined) {
        const roiValue = document.getElementById('roiValue');
        if (roiValue) {
            const roiText = `${parseFloat(data.roi).toFixed(2)}%`;
            roiValue.textContent = roiText;
            
            // Change color based on value
            if (parseFloat(data.roi) > 0) {
                roiValue.classList.add('text-success');
                roiValue.classList.remove('text-danger');
            } else if (parseFloat(data.roi) < 0) {
                roiValue.classList.add('text-danger');
                roiValue.classList.remove('text-success');
            } else {
                roiValue.classList.remove('text-success', 'text-danger');
            }
        }
    }
    
    // Update trade statistics
    if (data.trade_count !== undefined) {
        document.querySelectorAll('h3, span').forEach(el => {
            if (el.textContent === '0' && el.closest('div').textContent.toLowerCase().includes('operaciones')) {
                el.textContent = data.trade_count;
            }
        });
    }
    
    // Update wins/losses
    if (data.wins !== undefined && data.losses !== undefined) {
        document.querySelectorAll('span').forEach(el => {
            if (el.textContent.includes('✓')) {
                el.textContent = `${data.wins} ✓`;
            } else if (el.textContent.includes('✗')) {
                el.textContent = `${data.losses} ✗`;
            }
        });
    }
    
    // Update trading signal
    if (data.integrated_signal) {
        const signalIndicator = document.getElementById('signalIndicator');
        if (signalIndicator) {
            // Clear previous content
            signalIndicator.innerHTML = '';
            
            // Create new badge based on signal
            const badge = document.createElement('span');
            badge.classList.add('badge', 'py-2', 'px-3');
            
            switch (data.integrated_signal) {
                case 'strong_buy':
                    badge.classList.add('bg-success');
                    badge.textContent = 'COMPRA FUERTE';
                    break;
                case 'buy':
                    badge.classList.add('bg-success');
                    badge.textContent = 'COMPRA';
                    break;
                case 'strong_sell':
                    badge.classList.add('bg-danger');
                    badge.textContent = 'VENTA FUERTE';
                    break;
                case 'sell':
                    badge.classList.add('bg-danger');
                    badge.textContent = 'VENTA';
                    break;
                default:
                    badge.classList.add('bg-secondary');
                    badge.textContent = 'NEUTRAL';
            }
            
            signalIndicator.appendChild(badge);
        }
    }
    
    // Update indicator signals
    if (data.strategy_signals) {
        updateIndicatorSignal('rsi', data.strategy_signals.rsi);
        updateIndicatorSignal('macd', data.strategy_signals.macd);
        updateIndicatorSignal('trend', data.strategy_signals.trend);
        updateIndicatorSignal('bb', data.strategy_signals.bollinger);
    }
    
    // Update running status buttons
    if (data.is_running !== undefined) {
        const startBotBtn = document.getElementById('startBotBtn');
        const stopBotBtn = document.getElementById('stopBotBtn');
        
        if (startBotBtn && stopBotBtn) {
            if (data.is_running) {
                startBotBtn.classList.add('d-none');
                stopBotBtn.classList.remove('d-none');
            } else {
                startBotBtn.classList.remove('d-none');
                stopBotBtn.classList.add('d-none');
            }
        }
    }
}

/**
 * Update an indicator signal display
 */
function updateIndicatorSignal(indicator, signal) {
    const element = document.getElementById(`${indicator}Value`);
    if (!element) return;
    
    // Clear current content
    element.innerHTML = '';
    
    // Create span based on signal
    const span = document.createElement('span');
    
    if (!signal || signal === 'neutral') {
        span.classList.add('text-muted');
        span.textContent = 'Neutral';
    } else if (signal === 'buy') {
        span.classList.add('text-success');
        
        switch (indicator) {
            case 'rsi':
                span.textContent = 'Sobrevendido';
                break;
            case 'macd':
            case 'trend':
                span.textContent = 'Alcista';
                break;
            case 'bb':
                span.textContent = 'Soporte';
                break;
            default:
                span.textContent = 'Compra';
        }
    } else if (signal === 'sell') {
        span.classList.add('text-danger');
        
        switch (indicator) {
            case 'rsi':
                span.textContent = 'Sobrecomprado';
                break;
            case 'macd':
            case 'trend':
                span.textContent = 'Bajista';
                break;
            case 'bb':
                span.textContent = 'Resistencia';
                break;
            default:
                span.textContent = 'Venta';
        }
    } else {
        span.classList.add('text-muted');
        span.textContent = '--';
    }
    
    element.appendChild(span);
}

/**
 * Start the trading bot
 */
function startBot() {
    const mode = document.getElementById('tradingMode').value;
    const symbol = document.getElementById('tradingSymbol').value;
    const interval = document.getElementById('tradingInterval').value;
    const notify = document.getElementById('notifyEnabled').checked;
    
    // Create request data
    const data = {
        mode: mode,
        symbol: symbol,
        interval: interval,
        notify: notify
    };
    
    // Make API request
    fetch('/api/bot/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        // Hide modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('startBotModal'));
        modal.hide();
        
        // Display result
        if (result.success) {
            showAlert('Bot iniciado correctamente', 'success');
            // Update UI immediately
            fetchBotStatus();
        } else {
            showAlert(`Error: ${result.message}`, 'danger');
        }
    })
    .catch(error => {
        console.error('Error starting bot:', error);
        showAlert('Error al iniciar el bot', 'danger');
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
    .then(result => {
        if (result.success) {
            showAlert('Bot detenido correctamente', 'warning');
            // Update UI immediately
            fetchBotStatus();
        } else {
            showAlert(`Error: ${result.message}`, 'danger');
        }
    })
    .catch(error => {
        console.error('Error stopping bot:', error);
        showAlert('Error al detener el bot', 'danger');
    });
}

/**
 * Fetch the current market price
 */
function fetchMarketPrice(symbol) {
    fetch(`/api/market/price?symbol=${symbol}`)
        .then(response => response.json())
        .then(data => {
            if (data.success && data.price) {
                // Add to chart
                addPriceDataPoint(data.price);
            }
        })
        .catch(error => console.error('Error fetching market price:', error));
}

/**
 * Add a new price data point to the chart
 */
function addPriceDataPoint(price) {
    const now = new Date();
    const timeStr = now.toLocaleTimeString();
    
    // Add new data
    timeLabels.push(timeStr);
    priceData.push(price);
    
    // Limit the number of data points
    if (timeLabels.length > maxDataPoints) {
        timeLabels.shift();
        priceData.shift();
    }
    
    // Update chart
    if (priceChart) {
        priceChart.data.labels = timeLabels;
        priceChart.data.datasets[0].data = priceData;
        priceChart.update();
    }
}

/**
 * Refresh market analysis
 */
function refreshMarketAnalysis(symbol, interval) {
    fetch(`/api/market/analyze?symbol=${symbol}&interval=${interval}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update UI with analysis results
                const signalIndicator = document.getElementById('signalIndicator');
                if (signalIndicator) {
                    // Clear previous content
                    signalIndicator.innerHTML = '';
                    
                    // Create new badge based on signal
                    const badge = document.createElement('span');
                    badge.classList.add('badge', 'py-2', 'px-3');
                    
                    switch (data.signal) {
                        case 'strong_buy':
                            badge.classList.add('bg-success');
                            badge.textContent = 'COMPRA FUERTE';
                            break;
                        case 'buy':
                            badge.classList.add('bg-success');
                            badge.textContent = 'COMPRA';
                            break;
                        case 'strong_sell':
                            badge.classList.add('bg-danger');
                            badge.textContent = 'VENTA FUERTE';
                            break;
                        case 'sell':
                            badge.classList.add('bg-danger');
                            badge.textContent = 'VENTA';
                            break;
                        default:
                            badge.classList.add('bg-secondary');
                            badge.textContent = 'NEUTRAL';
                    }
                    
                    signalIndicator.appendChild(badge);
                }
                
                // Update indicator signals
                if (data.details && data.details.signals) {
                    updateIndicatorSignal('rsi', data.details.signals.rsi);
                    updateIndicatorSignal('macd', data.details.signals.macd);
                    updateIndicatorSignal('trend', data.details.signals.trend);
                    updateIndicatorSignal('bb', data.details.signals.bollinger);
                }
                
                showAlert('Análisis de mercado actualizado', 'info');
            } else {
                showAlert(`Error: ${data.message}`, 'danger');
            }
        })
        .catch(error => {
            console.error('Error refreshing market analysis:', error);
            showAlert('Error al actualizar el análisis de mercado', 'danger');
        });
}

/**
 * Show a bootstrap alert
 */
function showAlert(message, type = 'info') {
    // Create alert container if it doesn't exist
    let alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) {
        alertContainer = document.createElement('div');
        alertContainer.id = 'alertContainer';
        alertContainer.style.position = 'fixed';
        alertContainer.style.top = '20px';
        alertContainer.style.right = '20px';
        alertContainer.style.zIndex = '9999';
        document.body.appendChild(alertContainer);
    }
    
    // Create alert
    const alert = document.createElement('div');
    alert.classList.add('alert', `alert-${type}`, 'alert-dismissible', 'fade', 'show');
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add to container
    alertContainer.appendChild(alert);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alert) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }
    }, 5000);
}
