
from flask import Flask, jsonify, render_template, request, redirect, url_for
import logging
import time
import threading
import os
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "solana_trading_bot_secret")

class ScalpingBot:
    def __init__(self):
        self.active = False
        self.balance = 1000
        self.current_position = None
        self.last_trade_price = 0
        self.total_trades = 0
        self.mode = "paper"  # Modo por defecto: paper trading
        
    def start(self):
        self.active = True
        logger.info(f"Bot iniciado en modo {self.mode}")
        
    def stop(self):
        self.active = False
        logger.info("Bot detenido")
        
    def get_status(self):
        return {
            "active": self.active,
            "balance": self.balance,
            "position": self.current_position,
            "total_trades": self.total_trades,
            "mode": self.mode
        }
    
    def set_mode(self, mode):
        """Establece el modo de trading (paper o live)"""
        if mode not in ["paper", "live"]:
            logger.warning(f"Modo no válido: {mode}. Usando paper por defecto")
            mode = "paper"
        
        # Si intenta cambiar a live, mostrar advertencia
        if mode == "live" and self.mode == "paper":
            logger.warning("⚠️ CAMBIANDO A MODO LIVE TRADING - USAR CON PRECAUCIÓN")
        
        self.mode = mode
        logger.info(f"Modo de trading cambiado a: {mode}")
        return True

bot = ScalpingBot()

def run_bot():
    while True:
        if bot.active:
            try:
                # Aquí irá la lógica de trading real
                current_price = 170 + (time.time() % 10 - 5)  # Simular precio de SOL
                logger.info(f"Precio actual: ${current_price:.2f}")
                
                # Lógica simple de simulación
                if bot.current_position is None and time.time() % 30 < 15:
                    # Abrir posición cada 30 segundos
                    bot.current_position = {"type": "long", "entry_price": current_price}
                    logger.info(f"Posición abierta a ${current_price:.2f}")
                elif bot.current_position is not None and time.time() % 30 >= 15:
                    # Cerrar posición
                    profit = (current_price - bot.current_position["entry_price"]) if bot.current_position["type"] == "long" else (bot.current_position["entry_price"] - current_price)
                    bot.balance += profit
                    bot.total_trades += 1
                    logger.info(f"Posición cerrada a ${current_price:.2f}. Profit: ${profit:.2f}")
                    bot.current_position = None
                
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error en la ejecución del bot: {e}")
                time.sleep(5)
        else:
            time.sleep(1)

# Rutas para API
@app.route('/api/start', methods=['POST'])
def api_start_bot():
    data = request.json or {}
    mode = data.get('mode', 'paper')
    
    # Asegurar que siempre se use paper trading por defecto
    if mode == 'live':
        logger.warning("⚠️ Se intentó iniciar en modo live. Cambiando a paper trading por seguridad.")
        mode = 'paper'
    
    bot.set_mode(mode)
    bot.start()
    return jsonify({"status": "Bot iniciado", "mode": bot.mode})

@app.route('/api/stop', methods=['POST'])
def api_stop_bot():
    bot.stop()
    return jsonify({"status": "Bot detenido"})

@app.route('/api/status')
def api_get_status():
    return jsonify(bot.get_status())

# Rutas para interfaz web
@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Panel de control"""
    return render_template('dashboard.html', bot_status=bot.get_status())

# Exportar app para Gunicorn
application = app

if __name__ == '__main__':
    # Iniciar el thread del bot
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    # Iniciar el servidor web
    app.run(host='0.0.0.0', port=5000, debug=False)
