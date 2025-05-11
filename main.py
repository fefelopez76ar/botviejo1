
from flask import Flask, render_template, jsonify
from adaptive_system.bot_battle_arena import BotBattleArena
import logging
import threading
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
arena = None

def init_battle_arena():
    """Inicializa y ejecuta un solo bot de scalping"""
    global arena
    arena = BotBattleArena()

    # Crear un solo bot de scalping
    warrior_id = arena.add_warrior(
        strategy_name="breakout_scalping",
        timeframe="1m",
        params={
            "take_profit_pct": 0.8,
            "stop_loss_pct": 0.5,
            "trailing_stop_pct": 0.3,
            "max_position_size_pct": 50.0,
            "min_volume_threshold": 1.5,
            "min_rr_ratio": 1.5,
            "max_fee_impact_pct": 0.15
        }
    )
    
    # Activar el bot
    arena.activate_warrior(warrior_id)

    # Ciclo de evaluación más frecuente para aprendizaje rápido
    while True:
        try:
            arena.evaluate_arena()
            time.sleep(60)  # Evaluar cada minuto
        except Exception as e:
            logger.error(f"Error en ciclo de evaluación: {e}")
            time.sleep(5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/battle-arena')
def battle_arena():
    return render_template('battle_arena.html')

@app.route('/api/status')
def get_status():
    if not arena:
        return jsonify({"error": "Bot no inicializado"})
    return jsonify(arena.get_arena_status())

@app.route('/api/leaderboard')
def get_leaderboard():
    if not arena:
        return jsonify({"error": "Bot no inicializado"})
    return jsonify(arena.get_leaderboard())

if __name__ == '__main__':
    # Iniciar bot en un hilo separado
    arena_thread = threading.Thread(target=init_battle_arena, daemon=True)
    arena_thread.start()

    # Iniciar servidor web
    app.run(host='0.0.0.0', port=5000, debug=False)
