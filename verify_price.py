#!/usr/bin/env python3
"""
Verificador de datos de mercado en tiempo real
"""
import requests
import json
import time
from datetime import datetime

def check_market_data():
    """Verifica datos de mercado y estado del bot"""
    print("Verificando datos de mercado y estado del bot...")
    print("-" * 50)
    
    # Verificar precio actual de SOL
    try:
        response = requests.get('https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT', timeout=5)
        if response.status_code == 200:
            price_data = response.json()
            sol_price = float(price_data['price'])
            print(f"SOL precio actual: ${sol_price:.4f}")
        else:
            print("Error obteniendo precio de mercado")
    except Exception as e:
        print(f"Error conectando a datos de mercado: {e}")
    
    # Verificar estado del bot
    try:
        response = requests.get('http://localhost:5000/stats', timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"Estado del bot:")
            print(f"  Balance: ${stats['balance']:.2f}")
            print(f"  Operaciones: {stats['total_trades']}")
            print(f"  ROI: {stats['roi']:+.2f}%")
            
            if stats['total_trades'] > 0:
                print(f"  Tasa de éxito: {stats['win_rate']:.1f}%")
                print(f"  Ganancia total: {stats['total_profit']:+.2f} USDT")
        else:
            print("Bot API no responde")
    except Exception as e:
        print(f"Error conectando al bot: {e}")
    
    # Verificar archivo de aprendizaje
    try:
        with open('learning_data.json', 'r') as f:
            learning_data = json.load(f)
        print(f"Archivo de aprendizaje: {len(learning_data)} registros")
        
        if learning_data:
            last_trade = learning_data[-1]
            print(f"Última operación: {last_trade['timestamp'][:19]}")
            print(f"Resultado: {'ÉXITO' if last_trade['success'] else 'PÉRDIDA'}")
            print(f"Ganancia: {last_trade['profit']:+.2f} USDT")
    except FileNotFoundError:
        print("Archivo de aprendizaje: No encontrado")
    except Exception as e:
        print(f"Error leyendo archivo de aprendizaje: {e}")
    
    print(f"\nVerificación completada: {datetime.now().strftime('%H:%M:%S')}")

def monitor_continuous():
    """Monitoreo continuo cada 30 segundos"""
    try:
        while True:
            check_market_data()
            print("\nEsperando 30 segundos... (Ctrl+C para salir)")
            time.sleep(30)
            print("\n" + "="*60)
    except KeyboardInterrupt:
        print("\nMonitoreo detenido")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        monitor_continuous()
    else:
        check_market_data()