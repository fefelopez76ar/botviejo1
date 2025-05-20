"""
Demostración del bot de scalping en modo simulación
"""

import time
import random
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Colores para la terminal
class Colors:
    GREEN = '\033[32m'
    RED = '\033[31m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title: str):
    """Imprime un encabezado con formato"""
    width = 80
    print(Colors.CYAN + Colors.BOLD + "=" * width + Colors.END)
    print(Colors.CYAN + Colors.BOLD + title.center(width) + Colors.END)
    print(Colors.CYAN + Colors.BOLD + "=" * width + Colors.END)

def print_table(headers: List[str], data: List[List[Any]], title: str = ""):
    """Imprime una tabla formateada"""
    if not data:
        print("No hay datos para mostrar.")
        return
    
    # Determinar ancho de cada columna
    col_widths = [len(str(h)) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Calcular ancho total de la tabla
    total_width = sum(col_widths) + (3 * len(headers)) + 1
    
    # Mostrar título si se proporciona
    if title:
        print("\n" + Colors.BOLD + title.center(total_width) + Colors.END)
    
    # Imprimir encabezados
    header_row = "| "
    for i, header in enumerate(headers):
        header_row += f"{Colors.BOLD}{str(header).center(col_widths[i])}{Colors.END} | "
    print("\n" + header_row)
    
    # Imprimir separador
    separator = "+" + "+".join(["-" * (width + 2) for width in col_widths]) + "+"
    print(separator)
    
    # Imprimir filas de datos
    for row in data:
        row_str = "| "
        for i, cell in enumerate(row):
            if i < len(col_widths):
                # Añadir color según el tipo de dato o contenido
                if isinstance(cell, (int, float)) and str(cell).startswith(("-", "+")) and cell != 0:
                    # Números positivos/negativos
                    color = Colors.GREEN if str(cell).startswith("+") or float(cell) > 0 else Colors.RED
                    row_str += f"{color}{str(cell).center(col_widths[i])}{Colors.END} | "
                elif str(cell).startswith(("↑", "▲")):
                    # Tendencia alcista
                    row_str += f"{Colors.GREEN}{str(cell).center(col_widths[i])}{Colors.END} | "
                elif str(cell).startswith(("↓", "▼")):
                    # Tendencia bajista
                    row_str += f"{Colors.RED}{str(cell).center(col_widths[i])}{Colors.END} | "
                else:
                    row_str += f"{str(cell).center(col_widths[i])} | "
        print(row_str)
    
    print(separator + "\n")

def print_chart(data: List[Tuple[str, float]], title: str = "", max_height: int = 10):
    """Imprime un gráfico de barras simple usando caracteres ASCII"""
    if not data:
        print("No hay datos para mostrar.")
        return
    
    # Calcular el valor máximo para escalar el gráfico
    max_value = max(abs(val) for _, val in data)
    
    # Determinar factor de escala
    scale_factor = max_height / max_value if max_value > 0 else 1
    
    # Mostrar título
    if title:
        print("\n" + Colors.BOLD + title + Colors.END)
    
    # Determinar ancho de etiquetas para alinear el gráfico
    max_label_width = max(len(label) for label, _ in data)
    
    # Imprimir gráfico
    for label, value in data:
        # Calcular altura de barra
        bar_height = int(abs(value) * scale_factor)
        
        # Determinar color según valor
        color = Colors.GREEN if value >= 0 else Colors.RED
        
        # Imprimir barra con etiqueta alineada
        print(f"{label.ljust(max_label_width)} | ", end="")
        print(f"{color}{'█' * bar_height}{Colors.END} {value:.2f}")
    
    print("")

def progress_bar(progress: float, width: int = 40):
    """Muestra una barra de progreso"""
    filled_width = int(width * progress)
    empty_width = width - filled_width
    
    bar = f"[{Colors.GREEN}{'█' * filled_width}{Colors.END}{'░' * empty_width}] {progress * 100:.1f}%"
    print(bar)

def adapt_strategy_weights(weights: Dict[str, float], components: Dict[str, int], success: bool):
    """Adapta los pesos de la estrategia basado en el éxito/fracaso"""
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

def main():
    """Función principal de demostración"""
    print_header("DEMO DE SCALPING DE SOLANA")
    
    print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d')}")
    print("Esta es una demostración del bot de trading algorítmico para Solana.")
    print("Los datos son simulados para mostrar las capacidades del bot.")
    
    # Configuración inicial
    pair = "SOL-USDT"
    timeframe = "1m"
    strategy = "Estrategia adaptativa (aprendizaje)"
    position_size = 5.0
    
    print(f"\nConfiguración:")
    print(f"• Par: {pair}")
    print(f"• Timeframe: {timeframe}")
    print(f"• Estrategia: {strategy}")
    print(f"• Tamaño de posición: {position_size}% del balance")
    print(f"• Modo: Paper Trading (simulación)")
    
    print("\nIniciando simulación de trading...")
    time.sleep(2)
    
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
    current_profit = 0.0
    
    # Datos para análisis
    signals = []
    prices = []
    equity_curve = []
    
    # Inicializar RSI, MACD y BB
    rsi_value = 50
    rsi_direction = 1  # 1: subiendo, -1: bajando
    macd_histogram = 0
    bb_upper = current_price * 1.01
    bb_lower = current_price * 0.99
    
    # Pesos de estrategia
    strategy_weights = {
        "RSI": 0.3,
        "MACD": 0.25,
        "BB": 0.2,
        "Volume": 0.15,
        "Pattern": 0.1
    }
    
    # Loop de simulación
    for iteration in range(1, 50):
        # Actualizar precio
        price_change = (0.001 * (iteration % 5) - 0.002) + (0.0005 * ((iteration // 10) % 3))
        current_price *= (1 + price_change)
        prices.append(current_price)
        
        # Actualizar RSI y MACD
        rsi_change = ((iteration % 10) - 5) * 1.5
        rsi_value += rsi_change * rsi_direction
        if rsi_value > 80 or rsi_value < 20:
            rsi_direction *= -1
        rsi_value = max(0, min(100, rsi_value))
        
        macd_histogram += ((iteration % 6) - 3) * 0.05
        if abs(macd_histogram) > 1.5:
            macd_histogram *= 0.8
        
        # Actualizar BB
        bb_middle = sum(prices[-20:]) / min(20, len(prices)) if prices else current_price
        bb_upper = bb_middle * (1 + 0.005 * (2 + (iteration % 5)))
        bb_lower = bb_middle * (1 - 0.005 * (2 + (iteration % 5)))
        
        # Generar señal
        signal_components = {
            "RSI": -1 if rsi_value > 70 else 1 if rsi_value < 30 else 0,
            "MACD": 1 if macd_histogram > 0.5 else -1 if macd_histogram < -0.5 else 0,
            "BB": -1 if current_price > bb_upper else 1 if current_price < bb_lower else 0,
            "Volume": 1 if (iteration % 15) < 8 else -1 if (iteration % 15) > 12 else 0,
            "Pattern": 1 if (iteration % 30) == 0 else -1 if (iteration % 30) == 15 else 0
        }
        
        # Aplicar pesos a la señal
        weighted_signal = sum(signal_components[k] * strategy_weights[k] for k in signal_components)
        signal = 1 if weighted_signal > 0.3 else -1 if weighted_signal < -0.3 else 0
        signals.append(signal)
        
        # Procesar operación
        position_size_usd = balance * position_size / 100
        
        if not open_position and signal == 1:
            # Abrir long
            entry_price = current_price
            position_type = "LONG"
            open_position = True
            trades_made += 1
            total_volume += position_size_usd
            print(f"\n[Iteración {iteration}] {Colors.GREEN}Abriendo LONG a ${entry_price:.2f}{Colors.END}")
        
        elif not open_position and signal == -1:
            # Abrir short
            entry_price = current_price
            position_type = "SHORT"
            open_position = True
            trades_made += 1
            total_volume += position_size_usd
            print(f"\n[Iteración {iteration}] {Colors.RED}Abriendo SHORT a ${entry_price:.2f}{Colors.END}")
        
        elif open_position:
            # Actualizar P/L
            if position_type == "LONG":
                current_profit = ((current_price - entry_price) / entry_price) * position_size_usd
                if current_profit > position_size_usd * 0.015 or signal == -1 or iteration % 20 == 0:
                    print(f"\n[Iteración {iteration}] Cerrando LONG a ${current_price:.2f}, P/L: {Colors.GREEN if current_profit > 0 else Colors.RED}${current_profit:.2f}{Colors.END}")
                    balance += current_profit
                    open_position = False
                    if current_profit > 0:
                        winning_trades += 1
                    adapt_strategy_weights(strategy_weights, signal_components, current_profit > 0)
            else:
                current_profit = ((entry_price - current_price) / entry_price) * position_size_usd
                if current_profit > position_size_usd * 0.015 or signal == 1 or iteration % 20 == 0:
                    print(f"\n[Iteración {iteration}] Cerrando SHORT a ${current_price:.2f}, P/L: {Colors.GREEN if current_profit > 0 else Colors.RED}${current_profit:.2f}{Colors.END}")
                    balance += current_profit
                    open_position = False
                    if current_profit > 0:
                        winning_trades += 1
                    adapt_strategy_weights(strategy_weights, signal_components, current_profit > 0)
        
        # Actualizar equity
        if open_position:
            if position_type == "LONG":
                current_profit = ((current_price - entry_price) / entry_price) * position_size_usd
            else:
                current_profit = ((entry_price - current_price) / entry_price) * position_size_usd
            equity = balance + current_profit
        else:
            equity = balance
        
        equity_curve.append(equity)
        
        # Mostrar información cada 10 iteraciones
        if iteration % 10 == 0 or iteration == 1:
            # Datos de mercado
            market_data = [
                ["Precio actual", f"${current_price:.2f}"],
                ["Variación", f"{price_change * 100:.3f}%"],
                ["RSI", f"{rsi_value:.2f}"],
                ["MACD Hist", f"{macd_histogram:.4f}"],
                ["BB Superior", f"${bb_upper:.2f}"],
                ["BB Inferior", f"${bb_lower:.2f}"]
            ]
            print_table(["Indicador", "Valor"], market_data, "Datos de Mercado")
            
            # Estado de cuenta
            account_data = [
                ["Balance", f"${balance:.2f}"],
                ["Equity", f"${equity:.2f}"],
                ["Operaciones", str(trades_made)],
                ["% Ganadas", f"{winning_trades/trades_made*100:.1f}%" if trades_made > 0 else "0%"],
                ["Volumen", f"${total_volume:.2f}"]
            ]
            print_table(["Métrica", "Valor"], account_data, "Estado de la Cuenta")
            
            # Pesos de estrategia
            weights_data = []
            for k, v in strategy_weights.items():
                weights_data.append([k, v])
            print_chart(weights_data, "Pesos de Estrategia (Aprendizaje)")
            
            # Posición actual
            if open_position:
                print(f"\nPosición actual: {position_type} | Entrada: ${entry_price:.2f} | P/L: ${current_profit:.2f}")
                print(f"Progreso hacia objetivo de beneficio:")
                target_profit = position_size_usd * 0.015
                progress = min(1.0, current_profit / target_profit) if current_profit > 0 else 0
                progress_bar(progress)
            else:
                print(f"\nSin posiciones abiertas. Analizando el mercado...")
            
            time.sleep(1)
    
    # Mostrar resultados finales
    print_header("RESULTADOS DE LA SIMULACIÓN")
    print(f"\nBalance inicial: $10,000.00")
    print(f"Balance final: ${balance:.2f}")
    print(f"Retorno: {(balance/10000-1)*100:.2f}%")
    print(f"Operaciones: {trades_made} ({winning_trades} ganadoras)")
    print(f"Win rate: {winning_trades/trades_made*100:.1f}%" if trades_made > 0 else "Win rate: 0%")
    
    print("\nPesos finales de la estrategia:")
    for k, v in strategy_weights.items():
        print(f"• {k}: {v:.4f}")
    
    print("\nDemostración completa. El bot está listo para ejecutarse en tu PC local.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulación interrumpida por el usuario.")
    except Exception as e:
        print(f"\nError en la simulación: {e}")