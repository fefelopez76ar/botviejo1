#!/usr/bin/env python3
"""
Simula operaciones de aprendizaje para generar datos inmediatos
"""
import json
import requests
from datetime import datetime

def create_learning_data():
    """Crea datos de aprendizaje de ejemplo basados en condiciones reales de mercado"""
    
    # Obtener precio actual de SOL
    try:
        response = requests.get('https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT')
        current_price = float(response.json()['price'])
    except:
        current_price = 245.0  # Precio de respaldo
    
    # Generar operaciones de aprendizaje realistas
    learning_data = []
    
    # Operación 1: Compra exitosa
    learning_data.append({
        "timestamp": datetime.now().isoformat(),
        "success": True,
        "profit": 15.23,
        "indicators": {
            "sma_5": current_price - 0.5,
            "sma_20": current_price - 1.2,
            "rsi": 62.4
        }
    })
    
    # Operación 2: Venta con pérdida
    learning_data.append({
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "profit": -8.76,
        "indicators": {
            "sma_5": current_price + 0.3,
            "sma_20": current_price - 0.8,
            "rsi": 71.2
        }
    })
    
    # Operación 3: Compra exitosa
    learning_data.append({
        "timestamp": datetime.now().isoformat(),
        "success": True,
        "profit": 22.15,
        "indicators": {
            "sma_5": current_price - 0.2,
            "sma_20": current_price - 1.5,
            "rsi": 58.9
        }
    })
    
    # Guardar datos de aprendizaje
    with open('learning_data.json', 'w') as f:
        json.dump(learning_data, f, indent=2)
    
    print(f"Archivo learning_data.json creado con {len(learning_data)} operaciones")
    print(f"Precio SOL usado: ${current_price:.2f}")
    
    # Mostrar estadísticas
    successful = [t for t in learning_data if t['success']]
    total_profit = sum(t['profit'] for t in learning_data)
    win_rate = len(successful) / len(learning_data) * 100
    
    print(f"Operaciones exitosas: {len(successful)}/{len(learning_data)}")
    print(f"Tasa de éxito: {win_rate:.1f}%")
    print(f"Ganancia total: {total_profit:+.2f} USDT")

if __name__ == "__main__":
    create_learning_data()