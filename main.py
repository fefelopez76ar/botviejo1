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
    """Inicializa y ejecuta la arena de batalla"""
    global arena
    arena = BotBattleArena()

    # Crear arena estándar con bots predefinidos
    arena.create_standard_arena([
        "breakout_scalping",
        "momentum_scalping", 
        "mean_reversion",
        "ml_adaptive"
    ])

    # Ciclo de evaluación continua
    while True:
        try:
            # Evaluar arena cada 15 minutos
            arena.evaluate_arena()
            time.sleep(900)  # 15 minutos
        except Exception as e:
            logger.error(f"Error en ciclo de evaluación: {e}")
            time.sleep(60)  # Esperar 1 minuto en caso de error

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/battle-arena')
def battle_arena():
    """Página de la arena de batalla."""
    return render_template('battle_arena.html')

@app.route('/api/status')
def get_status():
    if not arena:
        return jsonify({"error": "Arena no inicializada"})
    return jsonify(arena.get_arena_status())

@app.route('/api/leaderboard')
def get_leaderboard():
    if not arena:
        return jsonify({"error": "Arena no inicializada"})
    return jsonify(arena.get_leaderboard())

if __name__ == '__main__':
    # Iniciar arena en un hilo separado
    arena_thread = threading.Thread(target=init_battle_arena, daemon=True)
    arena_thread.start()

    # Iniciar servidor web con debug desactivado y host accesible
    app.run(host='0.0.0.0', port=5000, debug=False)