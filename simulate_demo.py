"""
Demo simplificada para mostrar las capacidades del bot de trading
"""

import random
import time
from datetime import datetime, timedelta

# Colores para la terminal
class Colors:
    GREEN = '\033[32m'
    RED = '\033[31m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title):
    """Imprime un encabezado con formato"""
    width = 80
    print(Colors.CYAN + Colors.BOLD + "=" * width + Colors.END)
    print(Colors.CYAN + Colors.BOLD + title.center(width) + Colors.END)
    print(Colors.CYAN + Colors.BOLD + "=" * width + Colors.END)

def progress_bar(progress, width=40):
    """Muestra una barra de progreso"""
    filled_width = int(width * progress)
    empty_width = width - filled_width
    
    bar = f"[{Colors.GREEN}{'█' * filled_width}{Colors.END}{'░' * empty_width}] {progress * 100:.1f}%"
    print(bar)

def simulate_trade(market_condition, pattern_type, initial_price, strategy_weights):
    """Simula una operación de trading"""
    # Determinar dirección basada en pesos
    direction = "LONG" if sum(strategy_weights.values()) > 0 else "SHORT"
    
    # Simular precio de entrada
    entry_price = initial_price
    
    # Determinar stop loss y take profit
    sl_pct = 0.015  # 1.5%
    tp_pct = 0.025  # 2.5%
    
    if direction == "LONG":
        stop_loss = entry_price * (1 - sl_pct)
        take_profit = entry_price * (1 + tp_pct)
    else:
        stop_loss = entry_price * (1 + sl_pct)
        take_profit = entry_price * (1 - tp_pct)
    
    # Simular movimiento de precio
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {Colors.BOLD}Abriendo posición {direction}{Colors.END}")
    print(f"Condición del mercado: {market_condition}")
    print(f"Patrón detectado: {pattern_type}")
    print(f"Precio de entrada: ${entry_price:.2f}")
    print(f"Stop Loss: ${stop_loss:.2f} | Take Profit: ${take_profit:.2f}")
    
    # Simular operación
    current_price = entry_price
    position_size = 100  # Unidades
    position_value = entry_price * position_size
    
    # Mostrar pesos de estrategia
    print("\nPesos de indicadores:")
    for indicator, weight in strategy_weights.items():
        color = Colors.GREEN if weight > 0 else Colors.RED if weight < 0 else Colors.WHITE
        print(f"• {indicator}: {color}{weight:.4f}{Colors.END}")
    
    # Simular paso del tiempo
    steps = random.randint(5, 15)
    outcome = random.choice(["win", "loss", "breakeven"])
    
    print("\nSimulando movimiento de precio...")
    
    for i in range(steps):
        # Calcular nuevo precio
        if outcome == "win":
            # Mover hacia take profit
            if direction == "LONG":
                target = take_profit
            else:
                target = take_profit
        elif outcome == "loss":
            # Mover hacia stop loss
            if direction == "LONG":
                target = stop_loss
            else:
                target = stop_loss
        else:
            # Oscilar cerca del precio de entrada
            target = entry_price * (1 + random.uniform(-0.005, 0.005))
        
        # Añadir algo de ruido
        noise = random.uniform(-0.003, 0.003)
        
        # Calcular próximo precio
        step_pct = (i + 1) / steps
        current_price = entry_price + (target - entry_price) * step_pct + (entry_price * noise)
        
        # Mostrar estado actual
        if direction == "LONG":
            pnl = (current_price - entry_price) * position_size
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - current_price) * position_size
            pnl_pct = (entry_price - current_price) / entry_price * 100
        
        pnl_color = Colors.GREEN if pnl > 0 else Colors.RED
        
        print(f"[{i+1}/{steps}] Precio: ${current_price:.2f} | P/L: {pnl_color}${pnl:.2f} ({pnl_pct:.2f}%){Colors.END}")
        
        # Simular progreso hacia take profit/stop loss
        if direction == "LONG":
            if outcome == "win":
                progress = (current_price - entry_price) / (take_profit - entry_price)
            else:
                progress = (entry_price - current_price) / (entry_price - stop_loss)
        else:
            if outcome == "win":
                progress = (entry_price - current_price) / (entry_price - take_profit)
            else:
                progress = (current_price - entry_price) / (stop_loss - entry_price)
        
        progress = max(0, min(1, progress))
        print(f"Progreso hacia {'TP' if outcome == 'win' else 'SL'}: ", end="")
        progress_bar(progress)
        
        # Pausa
        time.sleep(0.5)
    
    # Resultado final
    if direction == "LONG":
        final_pnl = (current_price - entry_price) * position_size
        pnl_pct = (current_price - entry_price) / entry_price * 100
    else:
        final_pnl = (entry_price - current_price) * position_size
        pnl_pct = (entry_price - current_price) / entry_price * 100
    
    result_color = Colors.GREEN if final_pnl > 0 else Colors.RED
    
    # Verificar circuit breaker
    circuit_breaker_triggered = False
    if final_pnl < -position_value * 0.02:  # Si pierde más del 2% del valor
        circuit_breaker_triggered = True
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {Colors.BOLD}Posición cerrada{Colors.END}")
    print(f"Precio de salida: ${current_price:.2f}")
    print(f"Resultado: {result_color}{'+' if final_pnl > 0 else ''}{final_pnl:.2f} ({pnl_pct:.2f}%){Colors.END}")
    
    if circuit_breaker_triggered:
        print(f"\n{Colors.RED}🚨 CIRCUIT BREAKER ACTIVADO: Pérdida excesiva detectada.{Colors.END}")
        print(f"{Colors.RED}Trading pausado por seguridad. Reinicio automático en 6 horas.{Colors.END}")
    
    # Actualizar pesos de estrategia basados en resultado
    learning_rate = 0.05
    updated_weights = {}
    
    print("\nActualizando pesos de estrategia basado en resultados:")
    
    for indicator, weight in strategy_weights.items():
        # Si la señal fue en la dirección correcta
        signal_correct = (direction == "LONG" and weight > 0 and final_pnl > 0) or \
                         (direction == "LONG" and weight < 0 and final_pnl < 0) or \
                         (direction == "SHORT" and weight < 0 and final_pnl > 0) or \
                         (direction == "SHORT" and weight > 0 and final_pnl < 0)
        
        if signal_correct:
            # Reforzar el peso
            adjustment = abs(weight) * learning_rate
            if weight > 0:
                updated_weights[indicator] = weight + adjustment
            else:
                updated_weights[indicator] = weight - adjustment
        else:
            # Disminuir el peso
            adjustment = abs(weight) * learning_rate
            if weight > 0:
                updated_weights[indicator] = weight - adjustment
            else:
                updated_weights[indicator] = weight + adjustment
    
    # Normalizar pesos
    total = sum(abs(w) for w in updated_weights.values())
    if total > 0:
        for k in updated_weights:
            updated_weights[k] /= total
    
    # Mostrar pesos actualizados
    for indicator in strategy_weights:
        old_weight = strategy_weights[indicator]
        new_weight = updated_weights[indicator]
        change = new_weight - old_weight
        
        change_str = f"+{change:.4f}" if change > 0 else f"{change:.4f}"
        change_color = Colors.GREEN if change > 0 else Colors.RED if change < 0 else Colors.WHITE
        
        print(f"• {indicator}: {old_weight:.4f} → {new_weight:.4f} ({change_color}{change_str}{Colors.END})")
    
    return final_pnl, updated_weights, circuit_breaker_triggered

def run_demo():
    """Ejecuta la demostración del bot"""
    print_header("DEMO DEL BOT DE TRADING ALGORÍTMICO PARA SOLANA")
    
    # Estado inicial
    balance = 10000.0  # $10,000 USDT
    initial_price = 100.0  # $100 por SOL
    total_trades = 0
    winning_trades = 0
    
    # Pesos iniciales de estrategia
    strategy_weights = {
        "RSI": 0.30,
        "MACD": 0.25,
        "Bollinger": 0.15,
        "Volume": 0.10,
        "Pattern": 0.20
    }
    
    # Condiciones de mercado posibles
    market_conditions = [
        "STRONG_UPTREND",
        "MODERATE_UPTREND",
        "LATERAL_LOW_VOL",
        "LATERAL_HIGH_VOL",
        "MODERATE_DOWNTREND", 
        "STRONG_DOWNTREND",
        "EXTREME_VOLATILITY"
    ]
    
    # Patrones de velas posibles
    patterns = [
        "ENGULFING_BULLISH",
        "ENGULFING_BEARISH",
        "DOJI",
        "HAMMER",
        "SHOOTING_STAR",
        "MORNING_STAR",
        "EVENING_STAR"
    ]
    
    # Simulación de trades
    trade_count = 3
    print(f"\nSimulando {trade_count} operaciones con el bot...")
    
    for i in range(trade_count):
        print(f"\n{Colors.BOLD}--- OPERACIÓN #{i+1} ---{Colors.END}")
        print(f"Balance actual: ${balance:.2f}")
        
        # Seleccionar condición de mercado aleatoria
        market_condition = random.choice(market_conditions)
        # Seleccionar patrón aleatorio
        pattern = random.choice(patterns)
        
        # Simular trade
        pnl, new_weights, circuit_broken = simulate_trade(
            market_condition, 
            pattern, 
            initial_price * (1 + random.uniform(-0.05, 0.05)),
            strategy_weights
        )
        
        # Actualizar balance
        balance += pnl
        
        # Actualizar estadísticas
        total_trades += 1
        if pnl > 0:
            winning_trades += 1
        
        # Actualizar pesos
        strategy_weights = new_weights
        
        # Si se activó el circuit breaker, pausar simulación
        if circuit_broken:
            print(f"\n{Colors.YELLOW}Simulación pausada por circuit breaker. Reanudando después de la pausa...{Colors.END}")
            time.sleep(2)
        
        # Pausa entre trades
        time.sleep(1)
    
    # Mostrar resultados finales
    print_header("RESULTADOS DE LA SIMULACIÓN")
    
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    profit = balance - 10000.0
    roi = profit / 10000.0 * 100
    
    result_color = Colors.GREEN if profit > 0 else Colors.RED
    
    print(f"\nBalance inicial: $10,000.00")
    print(f"Balance final: ${balance:.2f}")
    print(f"Rendimiento: {result_color}{'+' if profit > 0 else ''}{profit:.2f} ({roi:.2f}%){Colors.END}")
    print(f"Operaciones: {total_trades} (ganadas: {winning_trades})")
    print(f"Win rate: {win_rate:.1f}%")
    
    # Mostrar pesos finales de la estrategia
    print("\nPesos finales de la estrategia (adaptados mediante aprendizaje):")
    for indicator, weight in sorted(strategy_weights.items(), key=lambda x: x[1], reverse=True):
        color = Colors.GREEN if weight >= 0.25 else Colors.YELLOW if weight >= 0.15 else Colors.WHITE
        print(f"• {indicator}: {color}{weight:.4f}{Colors.END}")
    
    print("\nOtras características implementadas:")
    print("• Monitoreo de drawdown con circuit breakers automáticos")
    print("• Simulación realista con slippage y fees")
    print("• Gestión dinámica de tamaños de posición según volatilidad")
    print("• Detección de patrones y análisis de order flow")
    print("• Generación de reportes detallados con métricas clave")
    
    print(f"\n{Colors.BOLD}Simulación completada. El bot está listo para su uso en papel trading.{Colors.END}")
    print(f"{Colors.BOLD}Para ejecución real, deben cumplirse los requisitos mínimos de seguridad.{Colors.END}")

if __name__ == "__main__":
    run_demo()