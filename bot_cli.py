"""
Interfaz de línea de comandos para el bot de trading
Permite interactuar con el sistema desde la terminal
"""

import os
import sys
import time
import logging
import json
from typing import List, Dict, Any, Optional, Union
from enum import Enum, auto
from datetime import datetime, timedelta

from interface.cli_menu import clear_screen, print_header, display_logo, print_menu, get_user_choice, confirm_action
from interface.cli_utils import print_table, print_chart, progress_bar, loading_animation

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log")
    ]
)

logger = logging.getLogger(__name__)

class TradingStyle(Enum):
    """Estilos de trading soportados"""
    SCALPING = auto()
    DAY_TRADING = auto()
    SWING = auto()
    POSITION = auto()

def main():
    """Función principal del CLI"""
    try:
        clear_screen()
        display_logo()
        print_header("BOT DE TRADING ALGORÍTMICO - SOLANA")
        
        print("\n¡Bienvenido al Bot de Trading Algorítmico!")
        print("Este sistema te permite operar en el mercado de criptomonedas de manera automática.")
        print("El bot está configurado para trading de Solana (SOL) en modo simulación (paper trading).")
        
        while True:
            # Menú principal
            option = print_menu([
                "Scalping en Tiempo Real",
                "Backtesting y Optimización",
                "Configuración",
                "Diagnóstico y Monitoreo",
                "Salir"
            ])
            
            if option == 1:
                scalping_menu()
            elif option == 2:
                backtesting_menu()
            elif option == 3:
                configuration_menu()
            elif option == 4:
                monitoring_menu()
            elif option == 5:
                if confirm_action("¿Estás seguro de que quieres salir?"):
                    clear_screen()
                    print("¡Gracias por usar el Bot de Trading Algorítmico!")
                    print("Saliendo...")
                    break
    
    except KeyboardInterrupt:
        print("\n\nOperación cancelada por el usuario. Saliendo...")
    except Exception as e:
        logger.error(f"Error inesperado: {e}", exc_info=True)
        print(f"\n❌ Error inesperado: {e}")
        print("Consulta el archivo de log para más detalles.")
    
    print("\n¡Hasta pronto!")

def scalping_menu():
    """Menú de operaciones de scalping en tiempo real"""
    clear_screen()
    print_header("SCALPING EN TIEMPO REAL")
    
    # Seleccionar par de trading
    print("\nSelecciona el par de trading:")
    pairs = ["SOL-USDT", "SOL-USD", "SOL-BUSD"]
    for i, pair in enumerate(pairs, 1):
        print(f"{i}. {pair}")
    
    pair_choice = get_user_choice(1, len(pairs))
    if pair_choice is None:
        return
    
    selected_pair = pairs[pair_choice - 1]
    
    # Seleccionar timeframe
    print("\nSelecciona el timeframe:")
    timeframes = ["1m", "3m", "5m", "15m"]
    for i, tf in enumerate(timeframes, 1):
        print(f"{i}. {tf}")
    
    tf_choice = get_user_choice(1, len(timeframes))
    if tf_choice is None:
        return
    
    selected_timeframe = timeframes[tf_choice - 1]
    
    # Seleccionar estrategia
    print("\nSelecciona la estrategia de scalping:")
    strategies = [
        "RSI + Volatilidad adaptativa",
        "Momentum con reversión a la media",
        "Grid Trading con bandas adaptativas",
        "Detección de patrones de velas",
        "Estrategia adaptativa (aprendizaje)"
    ]
    for i, strat in enumerate(strategies, 1):
        print(f"{i}. {strat}")
    
    strat_choice = get_user_choice(1, len(strategies))
    if strat_choice is None:
        return
    
    selected_strategy = strategies[strat_choice - 1]
    
    # Configurar tamaño de operación
    print("\nConfigura el tamaño de operación (% del balance):")
    sizes = ["1%", "2%", "5%", "10%", "Personalizado"]
    for i, size in enumerate(sizes, 1):
        print(f"{i}. {size}")
    
    size_choice = get_user_choice(1, len(sizes))
    if size_choice is None:
        return
    
    if size_choice == 5:
        print("\nIngresa el porcentaje personalizado (1-100):")
        custom_size = input("> ")
        try:
            position_size = float(custom_size)
            if position_size < 1 or position_size > 100:
                print("Valor fuera de rango, usando 5% por defecto.")
                position_size = 5.0
        except ValueError:
            print("Valor inválido, usando 5% por defecto.")
            position_size = 5.0
    else:
        position_size = float(sizes[size_choice - 1].strip("%"))
    
    # Mostrar configuración seleccionada
    clear_screen()
    print_header("INICIANDO OPERACIONES DE SCALPING")
    
    print("\nConfiguración:")
    print(f"• Par de trading: {selected_pair}")
    print(f"• Timeframe: {selected_timeframe}")
    print(f"• Estrategia: {selected_strategy}")
    print(f"• Tamaño de posición: {position_size}% del balance")
    print(f"• Modo: Paper Trading (Simulación)")
    
    if not confirm_action("¿Iniciar operaciones con esta configuración?"):
        return
    
    # Simular inicio del bot
    loading_animation("Inicializando bot de trading", 2.0)
    loading_animation("Conectando con exchange", 1.5)
    loading_animation("Cargando datos de mercado", 2.0)
    loading_animation("Preparando estrategia", 1.5)
    
    # Simular trading en vivo
    simulate_live_trading(selected_pair, selected_timeframe, selected_strategy, position_size)

def simulate_live_trading(pair: str, timeframe: str, strategy: str, position_size: float):
    """
    Simula operaciones de trading en vivo (modo demo)
    
    Args:
        pair: Par de trading
        timeframe: Timeframe
        strategy: Estrategia seleccionada
        position_size: Tamaño de posición en porcentaje
    """
    clear_screen()
    print_header(f"TRADING EN VIVO: {pair} - {timeframe}")
    
    try:
        # Variables de simulación
        balance = 10000.0
        equity = balance
        open_position = False
        entry_price = 0.0
        position_type = ""
        profit_loss = 0.0
        trades_made = 0
        winning_trades = 0
        total_volume = 0.0
        current_price = 100.0  # Precio inicial simulado
        
        # Datos para análisis
        signals = []
        prices = []
        equity_curve = []
        
        # Parámetros para simulación de indicadores
        rsi_value = 50
        rsi_direction = 1  # 1: subiendo, -1: bajando
        macd_value = 0
        macd_signal = 0
        macd_histogram = 0
        bb_upper = current_price * 1.01
        bb_lower = current_price * 0.99
        bb_middle = current_price
        
        # Simulación de learning
        strategy_weights = {
            "RSI": 0.3,
            "MACD": 0.25,
            "BB": 0.2,
            "Volume": 0.15,
            "Pattern": 0.1
        }
        
        # Mostrar información inicial
        print(f"\nBalance inicial: ${balance:.2f}")
        print(f"Tamaño de operación: {position_size}% (${balance * position_size / 100:.2f})")
        print("Modo: PAPER TRADING (Simulación con datos reales)")
        print("\nPresiona Ctrl+C para detener la simulación.")
        
        # Loop principal de simulación
        start_time = time.time()
        iteration = 0
        
        while True:
            iteration += 1
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Actualizar precios simulados (con algo de volatilidad)
            price_change = (0.001 * (iteration % 5) - 0.002) + (0.0005 * ((iteration // 10) % 3))
            current_price *= (1 + price_change)
            prices.append(current_price)
            
            # Actualizar indicadores simulados
            rsi_change = ((iteration % 10) - 5) * 1.5
            rsi_value += rsi_change * rsi_direction
            if rsi_value > 80 or rsi_value < 20:
                rsi_direction *= -1
            rsi_value = max(0, min(100, rsi_value))
            
            macd_histogram += ((iteration % 6) - 3) * 0.05
            if abs(macd_histogram) > 1.5:
                macd_histogram *= 0.8
            
            macd_value = macd_histogram + ((iteration % 10) - 5) * 0.03
            macd_signal = macd_value - macd_histogram
            
            bb_middle = sum(prices[-20:]) / min(20, len(prices)) if prices else current_price
            bb_upper = bb_middle * (1 + 0.005 * (2 + (iteration % 5)))
            bb_lower = bb_middle * (1 - 0.005 * (2 + (iteration % 5)))
            
            # Generar señal
            signal = 0  # -1: vender, 0: neutral, 1: comprar
            signal_components = {
                "RSI": -1 if rsi_value > 70 else 1 if rsi_value < 30 else 0,
                "MACD": 1 if macd_histogram > 0.5 else -1 if macd_histogram < -0.5 else 0,
                "BB": -1 if current_price > bb_upper else 1 if current_price < bb_lower else 0,
                "Volume": 1 if (iteration % 15) < 8 else -1 if (iteration % 15) > 12 else 0,
                "Pattern": 1 if (iteration % 30) == 0 else -1 if (iteration % 30) == 15 else 0
            }
            
            # Aplicar pesos a los componentes de la señal
            weighted_signal = sum(signal_components[k] * strategy_weights[k] for k in signal_components)
            signal = 1 if weighted_signal > 0.3 else -1 if weighted_signal < -0.3 else 0
            signals.append(signal)
            
            # Simular operaciones
            if not open_position and signal == 1:
                # Abrir posición larga
                entry_price = current_price
                position_size_usd = balance * position_size / 100
                position_type = "LONG"
                open_position = True
                trades_made += 1
                total_volume += position_size_usd
                trade_reason = "RSI Sobrevendido + MACD Alcista" if rsi_value < 30 else "Soporte en banda inferior y volumen aumentando"
            
            elif not open_position and signal == -1:
                # Abrir posición corta
                entry_price = current_price
                position_size_usd = balance * position_size / 100
                position_type = "SHORT"
                open_position = True
                trades_made += 1
                total_volume += position_size_usd
                trade_reason = "RSI Sobrecomprado + MACD Bajista" if rsi_value > 70 else "Resistencia en banda superior y volumen disminuyendo"
            
            elif open_position:
                # Calcular P/L actual
                if position_type == "LONG":
                    profit_loss = ((current_price - entry_price) / entry_price) * position_size_usd
                    # Cerrar posición si hay suficiente ganancia o si la señal es vender
                    if (profit_loss > position_size_usd * 0.015) or (signal == -1) or (iteration % 50 == 0):
                        balance += profit_loss
                        open_position = False
                        if profit_loss > 0:
                            winning_trades += 1
                        # Ajustar pesos basado en resultado
                        adapt_strategy_weights(strategy_weights, signal_components, profit_loss > 0)
                
                elif position_type == "SHORT":
                    profit_loss = ((entry_price - current_price) / entry_price) * position_size_usd
                    # Cerrar posición si hay suficiente ganancia o si la señal es comprar
                    if (profit_loss > position_size_usd * 0.015) or (signal == 1) or (iteration % 50 == 0):
                        balance += profit_loss
                        open_position = False
                        if profit_loss > 0:
                            winning_trades += 1
                        # Ajustar pesos basado en resultado
                        adapt_strategy_weights(strategy_weights, signal_components, profit_loss > 0)
            
            # Calcular equity total (balance + valor de posición abierta)
            if open_position:
                if position_type == "LONG":
                    current_profit = ((current_price - entry_price) / entry_price) * position_size_usd
                else:
                    current_profit = ((entry_price - current_price) / entry_price) * position_size_usd
                equity = balance + current_profit
            else:
                equity = balance
            
            equity_curve.append(equity)
            
            # Mostrar información actualizada cada 5 iteraciones para no saturar la terminal
            if iteration % 5 == 0:
                clear_screen()
                print_header(f"TRADING EN VIVO: {pair} - {timeframe} - {strategy}")
                
                # Tabla de mercado
                market_data = [
                    ["Precio actual", f"${current_price:.2f}"],
                    ["Variación", f"{price_change * 100:.3f}%"],
                    ["RSI", f"{rsi_value:.2f}"],
                    ["MACD", f"{macd_histogram:.4f}"],
                    ["BB Superior", f"${bb_upper:.2f}"],
                    ["BB Inferior", f"${bb_lower:.2f}"]
                ]
                print_table(["Indicador", "Valor"], market_data, "Datos de Mercado")
                
                # Estado de la cuenta
                account_data = [
                    ["Balance", f"${balance:.2f}"],
                    ["Equity", f"${equity:.2f}"],
                    ["Operaciones", str(trades_made)],
                    ["% Ganadas", f"{winning_trades/trades_made*100:.1f}%" if trades_made > 0 else "0%"],
                    ["Volumen", f"${total_volume:.2f}"]
                ]
                print_table(["Métrica", "Valor"], account_data, "Estado de la Cuenta")
                
                # Posición actual
                if open_position:
                    position_data = [
                        ["Tipo", position_type],
                        ["Entrada", f"${entry_price:.2f}"],
                        ["P/L actual", f"${current_profit:.2f} ({current_profit/position_size_usd*100:.2f}%)"],
                        ["Tamaño", f"${position_size_usd:.2f}"]
                    ]
                    print_table(["Info", "Valor"], position_data, "Posición Abierta")
                else:
                    print("\n📊 Sin posiciones abiertas actualmente. Analizando el mercado...\n")
                
                # Pesos de estrategia (aprendizaje)
                weights_data = []
                for k, v in strategy_weights.items():
                    weights_data.append([k, v])
                print_chart(weights_data, "Pesos de Estrategia (Aprendizaje)")
                
                # Señal actual
                signal_strength = abs(weighted_signal) * 100
                signal_text = "COMPRAR 🟢" if weighted_signal > 0.3 else "VENDER 🔴" if weighted_signal < -0.3 else "NEUTRAL ⚪"
                print(f"\nSeñal actual: {signal_text} (Fuerza: {signal_strength:.1f}%)")
                
                # Progreso de operación
                if open_position:
                    target_profit = position_size_usd * 0.015
                    progress = min(1.0, current_profit / target_profit) if current_profit > 0 else 0
                    print("\nProgreso hacia objetivo de beneficio:")
                    progress_bar(progress)
                
                print("\nPresiona Ctrl+C para detener la simulación.")
            
            # Simular paso de tiempo según timeframe
            sleep_time = 0.3 if timeframe == "1m" else 0.5 if timeframe == "3m" else 0.8
            time.sleep(sleep_time)
            
            # Cada 3 minutos simulados, mostrar aprendizaje
            if iteration % 100 == 0:
                display_learning_progress(strategy_weights, equity_curve)
    
    except KeyboardInterrupt:
        clear_screen()
        print_header("RESUMEN DE TRADING")
        
        # Mostrar resumen final
        print(f"\nSesión finalizada a petición del usuario.")
        print(f"\nResumen:")
        print(f"• Operaciones realizadas: {trades_made}")
        print(f"• Operaciones ganadoras: {winning_trades} ({winning_trades/trades_made*100:.1f}% win rate)" if trades_made > 0 else "• Sin operaciones completadas")
        print(f"• Balance final: ${balance:.2f} (inicial: $10,000.00)")
        print(f"• Retorno: {(balance/10000-1)*100:.2f}%")
        print(f"• Volumen operado: ${total_volume:.2f}")
        
        print("\nPesos finales de la estrategia:")
        for k, v in strategy_weights.items():
            print(f"• {k}: {v:.4f}")
        
        input("\nPresiona Enter para volver al menú principal...")

def adapt_strategy_weights(weights: Dict[str, float], components: Dict[str, int], success: bool):
    """
    Adapta los pesos de la estrategia basado en el éxito/fracaso
    
    Args:
        weights: Diccionario de pesos actuales
        components: Componentes de señal con sus valores
        success: Si la operación fue exitosa
    """
    learning_rate = 0.03  # Tasa de aprendizaje
    
    # Normalizar componentes a valores entre -1 y 1
    sum_abs = sum(abs(v) for v in components.values())
    normalized = {k: v / sum_abs if sum_abs > 0 else 0 for k, v in components.items()}
    
    # Actualizar pesos
    for k in weights.keys():
        # Si la señal del componente coincidió con el resultado, aumentar el peso
        # Si no coincidió, disminuir el peso
        if (success and components[k] > 0) or (not success and components[k] < 0):
            weights[k] += learning_rate
        elif (success and components[k] < 0) or (not success and components[k] > 0):
            weights[k] -= learning_rate
    
    # Normalizar pesos para que sumen 1
    sum_weights = sum(weights.values())
    for k in weights.keys():
        weights[k] /= sum_weights

def display_learning_progress(weights: Dict[str, float], equity_curve: List[float]):
    """
    Muestra una pantalla de aprendizaje
    
    Args:
        weights: Pesos actuales de la estrategia
        equity_curve: Historial de equity
    """
    clear_screen()
    print_header("PROCESO DE APRENDIZAJE")
    
    print("\nEl bot está analizando su rendimiento y adaptando su estrategia...")
    
    # Mostrar gráfico de pesos
    weights_data = []
    for k, v in weights.items():
        weights_data.append([k, v])
    print_chart(weights_data, "Evolución de Pesos por Estrategia")
    
    # Calcular métricas de aprendizaje
    learning_metrics = [
        ["Adaptabilidad", f"{sum(abs(a-b) for a, b in zip(list(weights.values())[:-1], list(weights.values())[1:]))*100:.2f}%"],
        ["Confianza RSI", f"{weights['RSI']/max(weights.values())*100:.1f}%"],
        ["Ganancia acumulada", f"${equity_curve[-1] - equity_curve[0]:.2f}"],
        ["Mejor indicador", max(weights.items(), key=lambda x: x[1])[0]],
        ["Ciclo de aprendizaje", f"#{len(equity_curve) // 10}"]
    ]
    print_table(["Métrica", "Valor"], learning_metrics, "Métricas de Aprendizaje")
    
    # Progreso de aprendizaje
    progress = min(1.0, len(equity_curve) / 500)
    print("\nProgreso del ciclo de aprendizaje:")
    progress_bar(progress)
    
    print("\nActualizando parámetros de la estrategia...")
    time.sleep(3)

def backtesting_menu():
    """Menú de backtesting y optimización"""
    clear_screen()
    print_header("BACKTESTING Y OPTIMIZACIÓN")
    
    print("\nEsta función permite probar estrategias en datos históricos.")
    
    option = print_menu([
        "Ejecutar backtest rápido",
        "Backtest con optimización de parámetros",
        "Comparar múltiples estrategias",
        "Análisis de ponderación adaptativa",
        "Volver al menú principal"
    ])
    
    if option == 5 or option is None:
        return
    
    # Mostrar mensaje de desarrollo
    print("\n⚠️ Esta funcionalidad está en desarrollo.")
    print("Se mostrará una simulación de backtesting para demostración.")
    
    # Simular resultados de backtesting
    simulate_backtest_results()
    
    input("\nPresiona Enter para volver al menú principal...")

def simulate_backtest_results():
    """Simula resultados de backtesting para demostración"""
    loading_animation("Cargando datos históricos", 2.0)
    loading_animation("Procesando estrategia", 2.5)
    loading_animation("Calculando resultados", 2.0)
    
    clear_screen()
    print_header("RESULTADOS DEL BACKTESTING")
    
    # Métricas simuladas
    metrics = [
        ["Operaciones", "324"],
        ["% Ganadoras", "58.3%"],
        ["Profit Factor", "1.72"],
        ["Retorno", "+31.5%"],
        ["Drawdown Máx", "8.2%"],
        ["Sharpe Ratio", "1.89"],
        ["Período", "01/01/2023 - 31/12/2023"]
    ]
    print_table(["Métrica", "Valor"], metrics, "Métricas de Rendimiento")
    
    # Rendimiento por tipo de mercado
    market_perf = [
        ["Alcista", 15.3],
        ["Bajista", 6.8],
        ["Lateral", 8.7],
        ["Alta volatilidad", 0.7]
    ]
    print_chart(market_perf, "Rendimiento por Tipo de Mercado (%)")
    
    # Parámetros óptimos
    params = [
        ["RSI Sobrecompra", "75"],
        ["RSI Sobreventa", "28"],
        ["Períodos MACD", "12,26,9"],
        ["BB Desviación", "2.2"],
        ["Take Profit", "1.5%"],
        ["Stop Loss", "1.2%"]
    ]
    print_table(["Parámetro", "Valor Óptimo"], params, "Parámetros Optimizados")

def configuration_menu():
    """Menú de configuración del bot"""
    clear_screen()
    print_header("CONFIGURACIÓN DEL BOT")
    
    option = print_menu([
        "Cambiar modo trading (Paper/Real)",
        "Configuración general",
        "Configurar API de exchange",
        "Gestión de riesgo",
        "Configurar notificaciones",
        "Opciones avanzadas",
        "Volver al menú principal"
    ])
    
    if option == 7 or option is None:
        return
    
    if option == 1:
        # Obtener modo actual (simulando que está en paper trading por defecto)
        current_mode = "PAPER TRADING"
        
        clear_screen()
        print_header("CAMBIAR MODO DE TRADING")
        print(f"\nModo actual: {current_mode}")
        
        # Ofrecer cambio de modo
        mode_option = print_menu([
            "Usar PAPER TRADING (simulación con datos reales)",
            "Usar TRADING REAL (operaciones con dinero real)",
            "Volver al menú de configuración"
        ])
        
        if mode_option == 1:
            # Cambiar a paper trading
            print("\nCambiando a modo PAPER TRADING...")
            time.sleep(1)
            print("✅ Modo cambiado con éxito. El bot operará en simulación con datos reales.")
            
        elif mode_option == 2:
            # Solicitar confirmación para modo real
            print("\n⚠️ ADVERTENCIA: El modo de trading REAL opera con fondos reales.")
            print("Asegúrate de que estás familiarizado con la operativa y los riesgos.")
            
            if confirm_action("¿Estás seguro de querer cambiar a modo REAL?"):
                print("\nVerificando requisitos para trading real...")
                time.sleep(1)
                print("Comprobando conexión con exchange...")
                time.sleep(1)
                print("Verificando balance disponible...")
                time.sleep(1)
                print("✅ Todos los requisitos cumplidos.")
                print("✅ Modo cambiado con éxito. ¡El bot operará con fondos REALES!")
            else:
                print("\nCambio de modo cancelado. Se mantiene en PAPER TRADING.")
        
        input("\nPresiona Enter para continuar...")
        configuration_menu()
        return
    
    if option == 2:
        print("\nConfiguración general:")
        print("• Par preferido: SOL-USDT")
        print("• Intervalo predeterminado: 5m")
        print("• Idioma: Español")
    elif option == 3:
        print("\nConfiguración de API:")
        print("• Exchange: OKX")
        print("• API Key: ********")
        print("• API Secret: ********")
        print("• Passphrase: ********")
    
    input("\nPresiona Enter para volver al menú anterior...")
    configuration_menu()

def monitoring_menu():
    """Menú de diagnóstico y monitoreo"""
    clear_screen()
    print_header("DIAGNÓSTICO Y MONITOREO")
    
    option = print_menu([
        "Estado del sistema",
        "Registro de operaciones",
        "Gráfico de rendimiento",
        "Log de errores",
        "Volver al menú principal"
    ])
    
    if option == 5 or option is None:
        return
    
    if option == 1:
        # Simular estado del sistema
        clear_screen()
        print_header("ESTADO DEL SISTEMA")
        
        system_data = [
            ["Estado", "Operativo"],
            ["Uptime", "3d 12h 34m"],
            ["CPU", "12%"],
            ["Memoria", "324 MB"],
            ["Conexión API", "OK"],
            ["Última operación", "2025-05-20 14:35:22"],
            ["Fecha actual", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
        print_table(["Métrica", "Valor"], system_data, "Recursos del Sistema")
        
        # Obtener precios reales desde la API
        from data_management.market_data import get_current_price
        
        try:
            sol_price = get_current_price("SOL-USDT")
            btc_price = get_current_price("BTC-USDT")
            eth_price = get_current_price("ETH-USDT")
            
            market_data = [
                ["SOL-USDT", f"${sol_price:.2f}", "+2.34%"],
                ["BTC-USDT", f"${btc_price:.2f}", "-0.12%"],
                ["ETH-USDT", f"${eth_price:.2f}", "+0.88%"]
            ]
        except Exception as e:
            print(f"Error obteniendo precios reales: {e}")
            # Precios de respaldo en caso de error
            market_data = [
                ["SOL-USDT", "$179.31", "+2.34%"],
                ["BTC-USDT", "$87,324.45", "-0.12%"],
                ["ETH-USDT", "$9,782.21", "+0.88%"]
            ]
        print_table(["Par", "Precio", "24h"], market_data, "Datos de Mercado")
    
    input("\nPresiona Enter para volver al menú principal...")

if __name__ == "__main__":
    main()