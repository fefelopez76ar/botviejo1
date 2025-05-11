
from flask import Flask, jsonify
import logging
import time
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ScalpingBot:
    def __init__(self):
        self.active = False
        self.balance = 1000
        self.current_position = None
        self.last_trade_price = 0
        self.total_trades = 0
        
    def start(self):
        self.active = True
        logger.info("Bot iniciado")
        
    def stop(self):
        self.active = False
        logger.info("Bot detenido")
        
    def get_status(self):
        return {
            "active": self.active,
            "balance": self.balance,
            "position": self.current_position,
            "total_trades": self.total_trades
        }

bot = ScalpingBot()

def run_bot():
    while True:
        if bot.active:
            try:
                # Aquí irá la lógica de trading real
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(5)
        else:
            time.sleep(1)

@app.route('/api/start')
def start_bot():
    bot.start()
    return jsonify({"status": "Bot iniciado"})

@app.route('/api/stop')
def stop_bot():
    bot.stop()
    return jsonify({"status": "Bot detenido"})

@app.route('/api/status')
def get_status():
    return jsonify(bot.get_status())

if __name__ == '__main__':
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=False)
