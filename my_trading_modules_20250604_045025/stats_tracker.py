#!/usr/bin/env python3
"""
Monitor de estadísticas del bot en tiempo real
"""
import time
import requests
import json
from datetime import datetime

def get_bot_stats():
    """Obtiene estadísticas del bot via API"""
    try:
        response = requests.get('http://localhost:5000/stats', timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def show_live_stats():
    """Muestra estadísticas en vivo del bot"""
    print("Monitoreando bot de aprendizaje automático...")
    print("=" * 60)
    
    data_count = 0
    start_time = datetime.now()
    
    while True:
        try:
            stats = get_bot_stats()
            current_time = datetime.now()
            runtime = (current_time - start_time).total_seconds() / 60
            
            print(f"\r[{current_time.strftime('%H:%M:%S')}] "
                  f"Balance: ${stats['balance']:.2f} | "
                  f"Operaciones: {stats['total_trades']} | "
                  f"ROI: {stats['roi']:+.2f}% | "
                  f"Runtime: {runtime:.1f}min", end="", flush=True)
            
            if stats['total_trades'] > 0:
                print(f" | Win Rate: {stats['win_rate']:.1f}%", end="")
            
            # Verificar si hay archivo de aprendizaje
            try:
                with open('learning_data.json', 'r') as f:
                    learning_data = json.load(f)
                    if learning_data:
                        print(f"\n\n📊 DATOS DE APRENDIZAJE ENCONTRADOS:")
                        print(f"Archivo: learning_data.json")
                        print(f"Registros: {len(learning_data)}")
                        
                        if len(learning_data) >= 3:
                            print("\nÚltimas operaciones:")
                            for trade in learning_data[-3:]:
                                status = "ÉXITO" if trade['success'] else "PÉRDIDA"
                                profit = trade['profit']
                                timestamp = trade['timestamp'][:16]
                                print(f"  {timestamp} | {status} | {profit:+.2f} USDT")
                        break
            except FileNotFoundError:
                pass
            
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\n\nMonitoreo detenido")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(5)

if __name__ == "__main__":
    show_live_stats()