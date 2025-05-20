"""
Utilidades para la interfaz de línea de comandos del bot

Proporciona funciones para mostrar información en formato de tabla y gráficos simples
basados en ASCII para la terminal.
"""

import os
import time
from typing import List, Dict, Any, Tuple

class Colors:
    """Colores para texto en terminal"""
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def confirm_action(message: str) -> bool:
    """
    Solicita confirmación al usuario
    
    Args:
        message: Mensaje a mostrar
        
    Returns:
        bool: True si se confirma, False si no
    """
    print(f"\n{message} (s/n)")
    choice = input("> ").strip().lower()
    
    return choice in ['s', 'si', 'y', 'yes']

def print_table(headers: List[str], data: List[List[Any]], title: str = ""):
    """
    Imprime una tabla formateada
    
    Args:
        headers: Lista de encabezados
        data: Lista de filas con datos
        title: Título opcional de la tabla
    """
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
                    color = Colors.GREEN if str(cell).startswith("+") or float(cell) >.0 else Colors.RED
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
    """
    Imprime un gráfico de barras simple usando caracteres ASCII
    
    Args:
        data: Lista de tuplas (etiqueta, valor)
        title: Título del gráfico
        max_height: Altura máxima del gráfico
    """
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
    """
    Muestra una barra de progreso
    
    Args:
        progress: Progreso entre 0 y 1
        width: Ancho de la barra
    """
    filled_width = int(width * progress)
    empty_width = width - filled_width
    
    bar = f"[{Colors.GREEN}{'█' * filled_width}{Colors.END}{'░' * empty_width}] {progress * 100:.1f}%"
    print(bar, end="\r")

def loading_animation(message: str, duration: float = 3.0):
    """
    Muestra una animación de carga
    
    Args:
        message: Mensaje a mostrar
        duration: Duración en segundos
    """
    symbols = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    start_time = time.time()
    
    i = 0
    while time.time() - start_time < duration:
        print(f"\r{symbols[i % len(symbols)]} {message}", end="")
        time.sleep(0.1)
        i += 1
    
    print("")  # Nueva línea al finalizar