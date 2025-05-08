from flask import Flask, render_template, request, jsonify
import os
import json
import logging
from datetime import datetime

from adaptive_system.bot_battle_arena import get_bot_battle_arena, BotBattleArena
from core.multi_bot_manager import get_multi_bot_manager
from risk_management.adaptive_position_manager import get_position_manager

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('app')

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "bot_battle_secret_key")

# Rutas principales
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/battle-arena')
def battle_arena():
    """Página de la arena de batalla de bots."""
    return render_template('battle_arena.html')

# API para la arena de batalla
@app.route('/api/battle-arena/status')
def api_battle_arena_status():
    """Obtiene el estado de la arena de batalla."""
    arena = get_bot_battle_arena()
    return jsonify(arena.get_arena_status())

@app.route('/api/battle-arena/warriors')
def api_battle_arena_warriors():
    """Obtiene los guerreros de la arena."""
    arena = get_bot_battle_arena()
    return jsonify(arena.get_leaderboard())

@app.route('/api/battle-arena/evaluate', methods=['POST'])
def api_battle_arena_evaluate():
    """Realiza una evaluación de la arena."""
    arena = get_bot_battle_arena()
    result = arena.evaluate_arena()
    return jsonify(result)

@app.route('/api/battle-arena/generate-charts', methods=['POST'])
def api_battle_arena_generate_charts():
    """Genera gráficos de la arena de batalla."""
    arena = get_bot_battle_arena()
    
    # Generar gráficos
    evolution_result = arena.generate_evolution_chart()
    leaderboard_result = arena.generate_leaderboard_chart()
    
    return jsonify({
        "evolution_chart": evolution_result,
        "leaderboard_chart": leaderboard_result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

"""
Solana Trading Bot - Main Application
"""

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
