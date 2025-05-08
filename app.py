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
    """Battle Arena page route"""
    return render_template('battle_arena.html')

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

if __name__ == '__main__':
    # Try to initialize bot at startup
    initialize_bot()
    app.run(host='0.0.0.0', port=5000, debug=True)
