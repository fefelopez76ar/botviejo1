import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from datetime import datetime, timedelta
import json
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the trading bot
from trading_bot import TradingBot, load_config, STATE_FILE

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "solana_trading_bot_secret")

# Global bot instance
trading_bot = None
bot_thread = None
is_bot_running = False

def load_bot_state():
    """Load the bot's state from the state file"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading bot state: {e}")
        return {}

def initialize_bot():
    """Initialize the trading bot with configuration"""
    global trading_bot
    
    # Load configuration
    config = load_config()
    
    api_key = config.get('OKX_API_KEY', '')
    api_secret = config.get('OKX_API_SECRET', '')
    passphrase = config.get('OKX_PASSPHRASE', '')
    
    # Default to paper trading mode
    mode = 'paper'
    
    if api_key and api_secret and passphrase:
        # Create bot instance
        trading_bot = TradingBot(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            mode=mode
        )
        logger.info(f"Trading bot initialized in {mode} mode")
        return True
    else:
        logger.error("Missing API credentials")
        return False

def bot_runner(symbol, interval, notify):
    """Function to run the bot in a separate thread"""
    global trading_bot, is_bot_running
    
    logger.info(f"Starting bot thread with {symbol} {interval}")
    is_bot_running = True
    
    try:
        trading_bot.run(
            symbol=symbol,
            interval=interval,
            notify=notify,
            continuous=True
        )
    except Exception as e:
        logger.error(f"Bot thread error: {e}")
    finally:
        is_bot_running = False
        logger.info("Bot thread stopped")

@app.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page route"""
    bot_state = load_bot_state()
    
    # Add running status
    bot_state['is_running'] = is_bot_running
    
    # Format some values for display
    if 'current_price' in bot_state:
        bot_state['current_price'] = f"${float(bot_state['current_price']):.2f}"
    
    if 'current_balance' in bot_state:
        bot_state['current_balance'] = f"${float(bot_state['current_balance']):.2f}"
    
    if 'roi' in bot_state:
        bot_state['roi'] = f"{float(bot_state['roi']):.2f}%"
    
    return render_template('dashboard.html', bot_state=bot_state)

@app.route('/settings')
def settings():
    """Settings page route"""
    # Load current config
    config = load_config()
    
    # Check if bot is initialized
    bot_initialized = trading_bot is not None
    
    return render_template('settings.html', 
                          config=config, 
                          bot_initialized=bot_initialized,
                          is_running=is_bot_running)

@app.route('/battle-arena')
def battle_arena():
    """Battle Arena page route (Desactivada)"""
    # La batalla de bots ha sido desactivada, redirigir al dashboard
    from flask import redirect, url_for
    return redirect(url_for('dashboard'))

@app.route('/error-management')
def error_management():
    """Error Management page route"""
    # Importar error reporter
    from core.error_reporter import get_error_reporter
    reporter = get_error_reporter()
    
    # Obtener errores recientes y organizarlos por estado
    recent_errors = reporter.get_recent_errors()
    
    # Obtener errores fijos y soluciones disponibles
    fixed_errors = [err for err in recent_errors if err.get('status') == 'fixed']
    available_fixes = []
    
    # En un sistema real, obtendríamos las soluciones disponibles del servidor
    # Para esta implementación, simulamos algunas soluciones
    for err in recent_errors:
        if err.get('status') == 'fixed':
            available_fixes.append({
                'error_id': err.get('error_id'),
                'error_type': err.get('error_type'),
                'fix_type': 'code_update',
                'description': 'Actualización de código para solucionar este error'
            })
    
    return render_template('error_management.html', 
                         recent_errors=recent_errors,
                         fixed_errors=fixed_errors,
                         available_fixes=available_fixes)

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    """API endpoint to start the bot"""
    global bot_thread, trading_bot, is_bot_running
    
    if is_bot_running:
        return jsonify({
            'success': False,
            'message': 'Bot is already running'
        })
    
    if not trading_bot:
        if not initialize_bot():
            return jsonify({
                'success': False,
                'message': 'Failed to initialize bot. Check API credentials.'
            })
    
    data = request.json
    symbol = data.get('symbol', 'SOL-USDT')
    interval = data.get('interval', '15m')
    notify = data.get('notify', False)
    mode = data.get('mode', 'paper')
    
    # Update bot mode if needed
    if trading_bot.mode != mode:
        trading_bot.mode = mode
        logger.info(f"Updated bot mode to {mode}")
    
    # Start bot in a new thread
    bot_thread = threading.Thread(
        target=bot_runner,
        args=(symbol, interval, notify),
        daemon=True
    )
    bot_thread.start()
    
    return jsonify({
        'success': True,
        'message': f'Bot started with {symbol} in {interval} interval'
    })

@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    """API endpoint to stop the bot"""
    global is_bot_running
    
    if not is_bot_running:
        return jsonify({
            'success': False,
            'message': 'Bot is not running'
        })
    
    # Set running flag to False, the thread will exit in the next iteration
    is_bot_running = False
    
    return jsonify({
        'success': True,
        'message': 'Bot stopping...'
    })

@app.route('/api/bot/status', methods=['GET'])
def bot_status():
    """API endpoint to get bot status"""
    bot_state = load_bot_state()
    
    # Add running status
    bot_state['is_running'] = is_bot_running
    
    return jsonify(bot_state)

@app.route('/api/market/price', methods=['GET'])
def get_market_price():
    """API endpoint to get current market price"""
    global trading_bot
    
    symbol = request.args.get('symbol', 'SOL-USDT')
    
    if not trading_bot:
        if not initialize_bot():
            return jsonify({
                'success': False,
                'message': 'Bot not initialized'
            })
    
    try:
        price = trading_bot.get_market_price(symbol)
        return jsonify({
            'success': True,
            'price': price,
            'symbol': symbol
        })
    except Exception as e:
        logger.error(f"Error getting market price: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/market/analyze', methods=['GET'])
def analyze_market():
    """API endpoint to analyze market"""
    global trading_bot
    
    symbol = request.args.get('symbol', 'SOL-USDT')
    interval = request.args.get('interval', '15m')
    
    if not trading_bot:
        if not initialize_bot():
            return jsonify({
                'success': False,
                'message': 'Bot not initialized'
            })
    
    try:
        signal, details = trading_bot.analyze_market(symbol, interval)
        
        # Convert numpy/pandas values to Python native types for JSON serialization
        cleaned_details = {}
        for key, value in details.items():
            if hasattr(value, 'item'):
                # Convert numpy values
                cleaned_details[key] = value.item()
            elif value is not None:
                cleaned_details[key] = float(value) if isinstance(value, (int, float)) else value
            else:
                cleaned_details[key] = None
        
        return jsonify({
            'success': True,
            'signal': signal,
            'details': cleaned_details
        })
    except Exception as e:
        logger.error(f"Error analyzing market: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/account/balance', methods=['GET'])
def get_balance():
    """API endpoint to get account balance"""
    global trading_bot
    
    if not trading_bot:
        if not initialize_bot():
            return jsonify({
                'success': False,
                'message': 'Bot not initialized'
            })
    
    try:
        balance = trading_bot.get_account_balance()
        return jsonify({
            'success': True,
            'balance': balance
        })
    except Exception as e:
        logger.error(f"Error getting account balance: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/account/positions', methods=['GET'])
def get_positions():
    """API endpoint to get account positions"""
    global trading_bot
    
    symbol = request.args.get('symbol', None)
    
    if not trading_bot:
        if not initialize_bot():
            return jsonify({
                'success': False,
                'message': 'Bot not initialized'
            })
    
    try:
        positions = trading_bot.get_positions(symbol)
        return jsonify({
            'success': True,
            'positions': positions
        })
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/change_mode', methods=['POST'])
def change_trading_mode():
    """API endpoint para cambiar entre modo papel y real (restringido a paper)"""
    global trading_bot
    
    data = request.json
    new_mode = data.get('mode', 'paper')
    confirmed = data.get('confirm', False)
    
    # RESTRICCIÓN: Forzar siempre a paper trading
    if new_mode == 'live':
        logger.warning("⚠️ ADVERTENCIA: Se intentó activar el modo live pero está deshabilitado por seguridad.")
        new_mode = 'paper'
        
    # Solo permitir 'paper' (live deshabilitado)
    if new_mode != 'paper':
        return jsonify({
            "success": False, 
            "message": "Solo se permite el modo 'paper'. El modo live está deshabilitado por seguridad."
        })
    
    if not trading_bot:
        if not initialize_bot():
            return jsonify({
                'success': False,
                'message': 'Bot not initialized'
            })
    
    # Cambiar modo (siempre a paper)
    try:
        # Si existe el método set_trading_mode
        if hasattr(trading_bot, 'set_trading_mode'):
            trading_bot.set_trading_mode(new_mode)
        else:
            # Si no existe, simplemente asignar la propiedad
            trading_bot.mode = new_mode
            
        return jsonify({
            "success": True, 
            "message": "Modo establecido a Paper Trading (simulación)"
        })
    except Exception as e:
        logger.error(f"Error al cambiar modo de trading: {e}")
        return jsonify({
            "success": False, 
            "message": f"Error al cambiar modo: {str(e)}"
        })
        
# API endpoints para el sistema de reporte de errores
@app.route('/api/errors/recent', methods=['GET'])
def get_recent_errors():
    """API endpoint para obtener errores recientes"""
    try:
        from core.error_reporter import get_error_reporter
        reporter = get_error_reporter()
        
        # Limitar a 20 errores recientes
        recent_errors = reporter.get_recent_errors(20)
        
        # Si no hay errores, crear algunos de ejemplo para probar la interfaz
        if not recent_errors:
            # Errores simulados para probar la interfaz
            recent_errors = [
                {
                    "error_id": "test_error_1",
                    "timestamp": datetime.now().isoformat(),
                    "error_type": "ConnectionError",
                    "error_message": "Failed to connect to exchange API",
                    "module": "api_client.exchange_client",
                    "status": "pending"
                },
                {
                    "error_id": "test_error_2",
                    "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "error_type": "ValueError",
                    "error_message": "Invalid parameter in trading strategy",
                    "module": "strategies.scalping_strategy",
                    "status": "reported"
                },
                {
                    "error_id": "test_error_3",
                    "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                    "error_type": "DataError",
                    "error_message": "Invalid market data format",
                    "module": "data_management.market_data",
                    "status": "fixed"
                }
            ]
        
        return jsonify({
            'success': True,
            'errors': recent_errors
        })
    except Exception as e:
        logger.error(f"Error getting recent errors: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/errors/<error_id>', methods=['GET'])
def get_error_details(error_id):
    """API endpoint para obtener detalles de un error específico"""
    try:
        from core.error_reporter import get_error_reporter
        reporter = get_error_reporter()
        
        error_details = reporter.get_error_details(error_id)
        
        # Si no se encuentra el error, crear uno de ejemplo para probar
        if not error_details:
            # Error simulado para probar la interfaz
            if error_id == "test_error_1":
                error_details = {
                    "error_id": "test_error_1",
                    "timestamp": datetime.now().isoformat(),
                    "error_type": "ConnectionError",
                    "error_message": "Failed to connect to exchange API",
                    "module": "api_client.exchange_client",
                    "traceback": "Traceback (most recent call last):\n  File \"api_client/exchange_client.py\", line 125, in connect\n    response = requests.get(self.api_url, timeout=5)\n  File \"requests/api.py\", line 76, in get\n    return request('get', url, params=params, **kwargs)\n  File \"requests/api.py\", line 61, in request\n    return session.request(method=method, url=url, **kwargs)\n  File \"requests/sessions.py\", line 530, in request\n    resp = self.send(prep, **send_kwargs)\n  File \"requests/sessions.py\", line 643, in send\n    r = adapter.send(request, **kwargs)\n  File \"requests/adapters.py\", line 516, in send\n    raise ConnectionError(e, request=request)\nConnectionError: HTTPConnectionPool(host='api.exchange.com', port=80): Max retries exceeded with url: /v1/market/prices (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f8b1c3a8d90>: Failed to establish a new connection: [Errno 111] Connection refused'))",
                    "system_info": {
                        "platform": "linux",
                        "python_version": "3.9.7",
                        "bot_version": "1.0.0"
                    },
                    "status": "pending"
                }
            elif error_id == "test_error_2":
                error_details = {
                    "error_id": "test_error_2",
                    "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "error_type": "ValueError",
                    "error_message": "Invalid parameter in trading strategy",
                    "module": "strategies.scalping_strategy",
                    "traceback": "Traceback (most recent call last):\n  File \"strategies/scalping_strategy.py\", line 87, in calculate_signal\n    rsi_value = calculate_rsi(data, period=-14)  # Invalid negative period\n  File \"indicators/momentum.py\", line 45, in calculate_rsi\n    if period <= 0:\n        raise ValueError(\"RSI period must be positive\")\nValueError: RSI period must be positive",
                    "system_info": {
                        "platform": "linux",
                        "python_version": "3.9.7",
                        "bot_version": "1.0.0"
                    },
                    "status": "reported",
                    "report_time": (datetime.now() - timedelta(hours=1)).isoformat()
                }
            elif error_id == "test_error_3":
                error_details = {
                    "error_id": "test_error_3",
                    "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                    "error_type": "DataError",
                    "error_message": "Invalid market data format",
                    "module": "data_management.market_data",
                    "traceback": "Traceback (most recent call last):\n  File \"data_management/market_data.py\", line 132, in process_candles\n    open_price = float(candle['open'])\n  File \"json/decoder.py\", line 355, in raw_decode\n    raise JSONDecodeError(\"Expecting value\", s, err.value) from None\njson.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)",
                    "system_info": {
                        "platform": "linux",
                        "python_version": "3.9.7",
                        "bot_version": "1.0.0"
                    },
                    "status": "fixed",
                    "fix_info": {
                        "fix_type": "code_update",
                        "description": "Actualización para corregir el manejo de datos JSON inválidos",
                        "fix_date": datetime.now().isoformat()
                    }
                }
            else:
                return jsonify({
                    'success': False,
                    'message': f'Error no encontrado: {error_id}'
                }), 404
        
        return jsonify({
            'success': True,
            'error': error_details
        })
    except Exception as e:
        logger.error(f"Error getting error details: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/errors/report', methods=['POST'])
def report_error():
    """API endpoint para reportar un error al servidor central"""
    try:
        from core.error_reporter import get_error_reporter
        reporter = get_error_reporter()
        
        data = request.json
        error_id = data.get('error_id')
        comments = data.get('comments')
        
        if not error_id:
            return jsonify({
                'success': False,
                'message': 'Error ID no proporcionado'
            }), 400
        
        # Simulamos la respuesta para pruebas
        # En un sistema real, esto enviaría el error al servidor central
        if error_id in ["test_error_1", "test_error_2", "test_error_3"]:
            return jsonify({
                'status': 'success',
                'message': 'Error reportado exitosamente',
                'ticket_id': f'TICKET-{error_id}'
            })
        
        # Intentar reportar el error real
        result = reporter.report_error(error_id, None, comments)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error reporting error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error al reportar: {str(e)}'
        })

@app.route('/api/errors/check-fixes', methods=['GET'])
def check_error_fixes():
    """API endpoint para verificar si hay soluciones disponibles"""
    try:
        from core.error_reporter import get_error_reporter
        reporter = get_error_reporter()
        
        # Simulamos la respuesta para pruebas
        # En un sistema real, esto consultaría al servidor central
        return jsonify({
            'status': 'success',
            'fixes_available': 1,
            'fixes': [
                {
                    'error_id': 'test_error_2',
                    'fix_type': 'code_update',
                    'description': 'Corrección para validar parámetros en estrategias de trading'
                }
            ]
        })
    except Exception as e:
        logger.error(f"Error checking fixes: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error al verificar soluciones: {str(e)}'
        })

@app.route('/api/errors/apply-fix', methods=['POST'])
def apply_error_fix():
    """API endpoint para aplicar una solución a un error específico"""
    try:
        from core.error_reporter import get_error_reporter
        reporter = get_error_reporter()
        
        data = request.json
        error_id = data.get('error_id')
        
        if not error_id:
            return jsonify({
                'status': 'error',
                'message': 'Error ID no proporcionado'
            }), 400
        
        # Simulamos la respuesta para pruebas
        # En un sistema real, esto aplicaría la solución desde el servidor
        if error_id in ["test_error_1", "test_error_2", "test_error_3"]:
            return jsonify({
                'status': 'success',
                'message': 'Solución aplicada correctamente',
                'error_id': error_id,
                'fix_type': 'code_update'
            })
        
        # Intentar aplicar la solución real
        result = reporter.apply_fix(error_id)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error applying fix: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error al aplicar solución: {str(e)}'
        })

@app.route('/api/errors/apply-all-fixes', methods=['POST'])
def apply_all_error_fixes():
    """API endpoint para aplicar todas las soluciones disponibles"""
    try:
        from core.error_reporter import get_error_reporter
        reporter = get_error_reporter()
        
        # Simulamos la respuesta para pruebas
        # En un sistema real, esto aplicaría todas las soluciones disponibles
        return jsonify({
            'status': 'success',
            'total_fixes': 1,
            'applied_fixes': 1,
            'failed_fixes': 0,
            'applied': [
                {
                    'error_id': 'test_error_2',
                    'fix_type': 'code_update'
                }
            ],
            'failed': []
        })
    except Exception as e:
        logger.error(f"Error applying all fixes: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error al aplicar soluciones: {str(e)}'
        })

@app.route('/api/sync/status', methods=['GET'])
def get_sync_status():
    """API endpoint para obtener el estado de sincronización"""
    try:
        # Simulamos la respuesta para pruebas
        return jsonify({
            'connected': True,
            'last_sync': datetime.now().isoformat(),
            'sync_server': 'https://sync.solanabot.com'
        })
    except Exception as e:
        logger.error(f"Error getting sync status: {e}")
        return jsonify({
            'connected': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Try to initialize bot at startup
    initialize_bot()
    app.run(host='0.0.0.0', port=5000, debug=True)
