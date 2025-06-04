#!/usr/bin/env python3
"""
Visor de logs y estadísticas del bot de aprendizaje
"""
import os
import json
import time
from datetime import datetime

def show_bot_status():
    """Muestra el estado actual del bot"""
    print("=" * 60)
    print("SOLANA SCALPING BOT - ESTADO DE APRENDIZAJE")
    print("=" * 60)
    
    # Verificar si hay datos de aprendizaje
    if os.path.exists('learning_data.json'):
        try:
            with open('learning_data.json', 'r') as f:
                learning_data = json.load(f)
            
            if learning_data:
                successful_trades = [t for t in learning_data if t['success']]
                total_profit = sum(t['profit'] for t in learning_data)
                win_rate = len(successful_trades) / len(learning_data) * 100
                
                print(f"📊 ESTADÍSTICAS DE APRENDIZAJE:")
                print(f"   Operaciones totales: {len(learning_data)}")
                print(f"   Operaciones exitosas: {len(successful_trades)}")
                print(f"   Tasa de éxito: {win_rate:.1f}%")
                print(f"   Ganancia total: {total_profit:+.2f} USDT")
                print()
                
                # Mostrar últimas 3 operaciones
                print("🔄 ÚLTIMAS OPERACIONES:")
                for trade in learning_data[-3:]:
                    status = "✅ ÉXITO" if trade['success'] else "❌ PÉRDIDA"
                    timestamp = trade['timestamp'][:19]
                    profit = trade['profit']
                    print(f"   {timestamp} | {status} | {profit:+.2f} USDT")
            else:
                print("📈 El bot está recopilando datos de mercado...")
                print("   Esperando al menos 20 velas para comenzar análisis")
        except Exception as e:
            print(f"Error leyendo datos: {e}")
    else:
        print("🔄 Bot iniciando - Aún no hay datos de operaciones")
    
    print()
    
    # Mostrar logs recientes
    if os.path.exists('trading_bot.log'):
        print("📝 ACTIVIDAD RECIENTE:")
        try:
            with open('trading_bot.log', 'r') as f:
                lines = f.readlines()
                for line in lines[-5:]:
                    if 'SOL:' in line or 'COMPRA' in line or 'VENTA' in line or 'cerrada' in line:
                        print(f"   {line.strip()}")
        except Exception as e:
            print(f"Error leyendo logs: {e}")
    
    print()
    print("💰 MODO: Paper Trading (Sin dinero real)")
    print("🎯 PAR: SOL/USDT")
    print("⏱️  TIMEFRAME: 1 minuto")
    print()
    print("El bot está aprendiendo automáticamente de los movimientos del mercado.")

def monitor_live():
    """Monitoreo en vivo del bot"""
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            show_bot_status()
            print("\n⏳ Actualizando en 30 segundos... (Ctrl+C para salir)")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n\n👋 Monitoreo detenido")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        monitor_live()
    else:
        show_bot_status()