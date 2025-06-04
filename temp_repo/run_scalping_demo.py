"""
Simulación de scalping para Solana con datos reales
"""
import time
import random
import os
import sys
from datetime import datetime
import ccxt

print("=== SIMULACIÓN DE SCALPING DE SOLANA CON DATOS REALES ===")
print(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Modo: PAPER TRADING (Simulación con datos reales)")

# Obtener precio real de Solana
try:
    # Usar API de OKX
    exchange = ccxt.okx({
        'apiKey': 'abc0a2f7-4b02-4f60-a4b9-fd575598e4e9',
        'secret': '2D78D8359A4873449E832B37BABC33E6',
        'password': 'Daeco1212@',
        'enableRateLimit': True
    })
    
    # Obtener precio de Solana
    ticker = exchange.fetch_ticker('SOL/USDT')
    current_price = ticker['last']
    
    print(f"✅ Conexión con OKX exitosa")
    print(f"📊 Precio actual de Solana (SOL/USDT): ${current_price}")
    print(f"📈 Variación 24h: {ticker['percentage']}%")
    print(f"💹 Volumen 24h: {ticker['quoteVolume']}")
    
except Exception as e:
    print(f"❌ Error al obtener datos reales: {e}")
    print("Usando datos simulados como respaldo")
    current_price = 179.31

# Configuración de la simulación
balance = 10000.0  # Balance inicial en USDT
position_size_pct = 5.0  # Tamaño de posición como % del balance
position_size = balance * position_size_pct / 100
print(f"\n💰 Balance inicial: ${balance:.2f}")
print(f"🔄 Tamaño de operación: {position_size_pct}% (${position_size:.2f})")

# Datos de simulación
open_position = False
entry_price = 0.0
position_type = ""
profit_loss = 0.0
trades_made = 0
winning_trades = 0
total_volume = 0.0
equity = balance

print("\n⏱️ Iniciando simulación de scalping...")
print("Presiona Ctrl+C para detener la simulación en cualquier momento")

# Simulación de indicadores técnicos
rsi_value = 45.0
rsi_direction = 1
macd_histogram = 0.0
price_history = [current_price]

try:
    iteration = 0
    while iteration < 100:  # Limitado a 100 iteraciones para la demo
        iteration += 1
        
        # Simular pequeños cambios en el precio (volatilidad realista para scalping)
        price_change = (random.random() - 0.5) * 0.001 * current_price
        current_price += price_change
        price_history.append(current_price)
        
        # Actualizar indicadores simulados
        # RSI
        rsi_change = (random.random() - 0.5) * 3
        rsi_value += rsi_change
        if rsi_value > 70 or rsi_value < 30:
            rsi_direction *= -1
        rsi_value = max(5, min(95, rsi_value))
        
        # MACD
        macd_histogram += (random.random() - 0.5) * 0.02
        if abs(macd_histogram) > 0.5:
            macd_histogram *= 0.9
        
        # Bandas de Bollinger
        bb_middle = sum(price_history[-20:]) / min(20, len(price_history))
        bb_std = 0.001 * current_price * (1 + random.random())
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        
        # Generar señal
        signal = 0  # -1: vender, 0: neutral, 1: comprar
        signal_components = {
            "RSI": -1 if rsi_value > 70 else 1 if rsi_value < 30 else 0,
            "MACD": 1 if macd_histogram > 0.1 else -1 if macd_histogram < -0.1 else 0,
            "BB": -1 if current_price > bb_upper else 1 if current_price < bb_lower else 0,
        }
        
        # Pesos de los componentes
        weights = {"RSI": 0.4, "MACD": 0.35, "BB": 0.25}
        weighted_signal = sum(signal_components[k] * weights[k] for k in signal_components)
        signal = 1 if weighted_signal > 0.2 else -1 if weighted_signal < -0.2 else 0
        
        # Simular operaciones
        if not open_position and signal == 1:
            # Abrir posición larga
            entry_price = current_price
            position_size_usd = position_size
            position_type = "LONG"
            open_position = True
            trades_made += 1
            total_volume += position_size_usd
            trade_reason = "RSI Sobrevendido + MACD Alcista" if rsi_value < 30 else "Soporte en banda inferior"
            print(f"\n🟢 COMPRA: ${entry_price:.2f} | Razón: {trade_reason}")
            
        elif not open_position and signal == -1:
            # Abrir posición corta
            entry_price = current_price
            position_size_usd = position_size
            position_type = "SHORT"
            open_position = True
            trades_made += 1
            total_volume += position_size_usd
            trade_reason = "RSI Sobrecomprado + MACD Bajista" if rsi_value > 70 else "Resistencia en banda superior"
            print(f"\n🔴 VENTA CORTA: ${entry_price:.2f} | Razón: {trade_reason}")
            
        elif open_position:
            # Calcular P/L actual
            if position_type == "LONG":
                profit_loss = ((current_price - entry_price) / entry_price) * position_size_usd
                # Cerrar posición si hay suficiente ganancia o si la señal es vender
                if (profit_loss > position_size_usd * 0.003) or (signal == -1) or (iteration % 30 == 0):
                    balance += profit_loss
                    open_position = False
                    if profit_loss > 0:
                        winning_trades += 1
                        print(f"✅ CIERRE GANANCIA: ${current_price:.2f} | Beneficio: ${profit_loss:.2f} ({profit_loss/position_size_usd*100:.2f}%)")
                    else:
                        print(f"❌ CIERRE PÉRDIDA: ${current_price:.2f} | Pérdida: ${profit_loss:.2f} ({profit_loss/position_size_usd*100:.2f}%)")
            
            elif position_type == "SHORT":
                profit_loss = ((entry_price - current_price) / entry_price) * position_size_usd
                # Cerrar posición si hay suficiente ganancia o si la señal es comprar
                if (profit_loss > position_size_usd * 0.003) or (signal == 1) or (iteration % 30 == 0):
                    balance += profit_loss
                    open_position = False
                    if profit_loss > 0:
                        winning_trades += 1
                        print(f"✅ CIERRE GANANCIA CORTO: ${current_price:.2f} | Beneficio: ${profit_loss:.2f} ({profit_loss/position_size_usd*100:.2f}%)")
                    else:
                        print(f"❌ CIERRE PÉRDIDA CORTO: ${current_price:.2f} | Pérdida: ${profit_loss:.2f} ({profit_loss/position_size_usd*100:.2f}%)")
        
        # Calcular equity actual
        if open_position:
            if position_type == "LONG":
                current_profit = ((current_price - entry_price) / entry_price) * position_size_usd
            else:
                current_profit = ((entry_price - current_price) / entry_price) * position_size_usd
            equity = balance + current_profit
        else:
            equity = balance
        
        # Mostrar información cada 10 iteraciones
        if iteration % 10 == 0 or iteration == 1:
            print(f"\n--- Iteración {iteration} ---")
            print(f"Precio: ${current_price:.2f} | RSI: {rsi_value:.1f} | MACD: {macd_histogram:.4f}")
            print(f"Balance: ${balance:.2f} | Equity: ${equity:.2f}")
            if open_position:
                print(f"Posición abierta: {position_type} | Entrada: ${entry_price:.2f} | P/L: ${current_profit:.2f}")
            
        # Pequeña pausa para simular el paso del tiempo
        time.sleep(0.2)
    
    # Mostrar resumen al finalizar
    print("\n\n=== RESUMEN DE LA SIMULACIÓN ===")
    print(f"Operaciones realizadas: {trades_made}")
    win_rate = (winning_trades / trades_made * 100) if trades_made > 0 else 0
    print(f"Operaciones ganadoras: {winning_trades} ({win_rate:.1f}%)")
    print(f"Balance final: ${balance:.2f} (inicial: $10,000.00)")
    print(f"Retorno: {(balance/10000-1)*100:.2f}%")
    print(f"Volumen operado: ${total_volume:.2f}")
    
except KeyboardInterrupt:
    print("\n\nSimulación interrumpida por el usuario")
    # Mostrar resumen parcial
    print("\n=== RESUMEN PARCIAL ===")
    print(f"Operaciones realizadas: {trades_made}")
    win_rate = (winning_trades / trades_made * 100) if trades_made > 0 else 0
    print(f"Operaciones ganadoras: {winning_trades} ({win_rate:.1f}%)")
    print(f"Balance final: ${balance:.2f} (inicial: $10,000.00)")
    print(f"Retorno: {(balance/10000-1)*100:.2f}%")

print("\n¡Simulación finalizada!")