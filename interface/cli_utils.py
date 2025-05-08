"""
Utilidades para la interfaz de línea de comandos del bot de trading
"""

import os
import sys
import platform
import time
from typing import List, Dict, Any, Tuple, Optional
import shutil

# Colores ANSI para la terminal
class Colors:
    # Colores básicos
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    BRIGHT_GREEN = '\033[92;1m'
    BRIGHT_RED = '\033[91;1m'
    BRIGHT_YELLOW = '\033[93;1m'
    BRIGHT_CYAN = '\033[96;1m'
    BRIGHT_MAGENTA = '\033[95;1m'
    BRIGHT_WHITE = '\033[97;1m'
    
    # Estilos
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Fondos (evitamos fondos oscuros)
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_CYAN = '\033[106m'
    BG_WHITE = '\033[107m'
    
    # Reset
    END = '\033[0m'

def clear_screen():
    """Limpia la pantalla de la terminal"""
    os_name = platform.system().lower()
    if os_name == 'windows':
        os.system('cls')
    else:
        os.system('clear')

def print_header(title: str, width: int = 80):
    """
    Imprime un encabezado con el título proporcionado
    
    Args:
        title: Título a mostrar
        width: Ancho del encabezado
    """
    clear_screen()
    
    # Determinar ancho disponible en la terminal
    terminal_width = shutil.get_terminal_size().columns
    width = min(width, terminal_width)
    
    # Crear la línea de separación
    separator = "=" * width
    
    print(f"{Colors.BRIGHT_YELLOW}{separator}{Colors.END}")
    title_centered = title.center(width)
    print(f"{Colors.BRIGHT_YELLOW}{title_centered}{Colors.END}")
    print(f"{Colors.BRIGHT_YELLOW}{separator}{Colors.END}")
    print()

def print_subheader(title: str, width: int = 80):
    """
    Imprime un subencabezado con el título proporcionado
    
    Args:
        title: Título a mostrar
        width: Ancho del subencabezado
    """
    # Determinar ancho disponible en la terminal
    terminal_width = shutil.get_terminal_size().columns
    width = min(width, terminal_width)
    
    # Crear la línea de separación
    separator = "-" * width
    
    print(f"{Colors.YELLOW}{separator}{Colors.END}")
    print(f"{Colors.YELLOW}{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.YELLOW}{separator}{Colors.END}")
    print()

def print_menu(options: List[str]) -> int:
    """
    Imprime un menú con las opciones proporcionadas y obtiene la elección del usuario
    
    Args:
        options: Lista de opciones a mostrar
        
    Returns:
        int: Opción elegida (1-based)
    """
    print()
    for i, option in enumerate(options, 1):
        print(f"{Colors.CYAN}{i}. {option}{Colors.END}")
    
    print()
    return get_user_choice(1, len(options))

def get_user_choice(min_value: int, max_value: int) -> int:
    """
    Obtiene una elección numérica del usuario
    
    Args:
        min_value: Valor mínimo aceptable
        max_value: Valor máximo aceptable
        
    Returns:
        int: Elección del usuario o 0 si es inválida
    """
    prompt = f"Ingresa tu opción ({min_value}-{max_value}): "
    
    try:
        choice = int(input(f"{Colors.BRIGHT_GREEN}{prompt}{Colors.END}"))
        if min_value <= choice <= max_value:
            return choice
        else:
            print(f"{Colors.RED}Opción no válida. Por favor, elige entre {min_value} y {max_value}.{Colors.END}")
            time.sleep(1)
            return 0
    except ValueError:
        print(f"{Colors.RED}Por favor, ingresa un número válido.{Colors.END}")
        time.sleep(1)
        return 0

def confirm_action(message: str) -> bool:
    """
    Solicita confirmación al usuario
    
    Args:
        message: Mensaje de confirmación
        
    Returns:
        bool: True si el usuario confirma, False de lo contrario
    """
    response = input(f"{Colors.YELLOW}{message} (s/n): {Colors.END}").strip().lower()
    return response == 's' or response == 'si' or response == 'sí' or response == 'y' or response == 'yes'

def print_table(headers: List[str], data: List[List[Any]], colors: List[str] = None):
    """
    Imprime una tabla con los datos proporcionados
    
    Args:
        headers: Lista de encabezados
        data: Lista de filas (cada fila es una lista de valores)
        colors: Lista de colores para cada columna (opcional)
    """
    if not data:
        return
    
    # Determinar el ancho de cada columna
    num_columns = len(headers)
    col_widths = [len(header) for header in headers]
    
    for row in data:
        for i, cell in enumerate(row[:num_columns]):
            cell_str = str(cell)
            col_widths[i] = max(col_widths[i], len(cell_str))
    
    # Agregar padding
    col_widths = [width + 2 for width in col_widths]
    
    # Crear la línea de separación
    separator = "+" + "+".join("-" * width for width in col_widths) + "+"
    
    # Imprimir encabezados
    print(separator)
    header_str = "|"
    for i, header in enumerate(headers):
        header_str += f" {Colors.BRIGHT_CYAN}{header:{col_widths[i]-2}}{Colors.END} |"
    print(header_str)
    print(separator)
    
    # Imprimir datos
    for row in data:
        row_str = "|"
        for i, cell in enumerate(row[:num_columns]):
            cell_str = str(cell)
            
            # Aplicar color si se especifica
            cell_color = ""
            if colors and i < len(colors) and colors[i]:
                cell_color = colors[i]
            
            # Colorear celdas específicas basado en su contenido
            if "COMPRA" in cell_str or "LONG" in cell_str:
                cell_color = Colors.GREEN
            elif "VENTA" in cell_str or "SHORT" in cell_str:
                cell_color = Colors.RED
            elif "Activo" in cell_str or "🟢" in cell_str:
                cell_color = Colors.GREEN
            elif "Inactivo" in cell_str or "⚪" in cell_str:
                cell_color = Colors.YELLOW
            
            # Colorear ROI, P&L y valores positivos/negativos
            if i > 0 and (headers[i] == "ROI" or headers[i] == "P&L"):
                if "+" in cell_str or (cell_str.replace("%", "").replace("$", "").strip() and float(cell_str.replace("%", "").replace("$", "").strip()) > 0):
                    cell_color = Colors.GREEN
                elif "-" in cell_str or (cell_str.replace("%", "").replace("$", "").strip() and float(cell_str.replace("%", "").replace("$", "").strip()) < 0):
                    cell_color = Colors.RED
            
            row_str += f" {cell_color}{cell_str:{col_widths[i]-2}}{Colors.END} |"
        print(row_str)
    
    print(separator)

def print_chart(data, value_key: str = 'close', width: int = 60, height: int = 15, title: str = None):
    """
    Imprime un gráfico ASCII simple con los datos proporcionados
    
    Args:
        data: DataFrame o diccionario con datos
        value_key: Clave para extraer los valores a graficar
        width: Ancho del gráfico
        height: Altura del gráfico
        title: Título del gráfico
    """
    if title:
        print(f"\n{Colors.CYAN}{title}{Colors.END}")
    
    # Extraer valores
    values = []
    if hasattr(data, value_key):
        values = data[value_key].tolist()
    elif isinstance(data, list) and all(isinstance(d, dict) and value_key in d for d in data):
        values = [d[value_key] for d in data]
    elif isinstance(data, dict) and value_key in data:
        values = data[value_key]
    else:
        print("No se pudieron extraer valores para el gráfico")
        return
    
    # Asegurar que tenemos suficientes datos
    if len(values) < 2:
        print("No hay suficientes datos para mostrar el gráfico")
        return
    
    # Calcular los límites
    min_val = min(values)
    max_val = max(values)
    
    # Añadir un pequeño margen
    range_val = max_val - min_val
    margin = range_val * 0.05
    min_val -= margin
    max_val += margin
    
    # Crear matriz para el gráfico
    chart = []
    for y in range(height):
        chart.append([' ' for _ in range(width)])
    
    # Mapear valores al gráfico
    for x in range(min(width, len(values))):
        idx = -width + x if len(values) > width else x
        val = values[idx]
        y = int((height - 1) * (1 - (val - min_val) / (max_val - min_val)))
        y = max(0, min(height - 1, y))
        chart[y][x] = '*'
    
    # Imprimir el gráfico
    y_labels = []
    for i in range(height):
        y_val = max_val - i * (max_val - min_val) / (height - 1)
        y_labels.append(f"{y_val:.1f}")
    
    # Determinar el ancho máximo de las etiquetas Y
    y_width = max(len(label) for label in y_labels)
    
    # Imprimir el gráfico con eje Y
    for i in range(height):
        print(f"{Colors.YELLOW}{y_labels[i]:>{y_width}} {Colors.END}+", end='')
        for x in range(width):
            if chart[i][x] == '*':
                print(f"{Colors.GREEN}*{Colors.END}", end='')
            else:
                print(chart[i][x], end='')
        print()
    
    # Imprimir eje X
    print(f"{' ' * (y_width + 1)}+", end='')
    print('-' * width)
    
    print(f"{' ' * (y_width + 1)}", end='')
    step = max(1, width // 10)
    for i in range(0, width, step):
        print(f"{i:^{step}}", end='')
    print()

def print_status(message: str, status: str = 'info'):
    """
    Imprime un mensaje de estado con el color correspondiente
    
    Args:
        message: Mensaje a mostrar
        status: Tipo de mensaje (info, success, warning, error)
    """
    if status == 'success':
        print(f"{Colors.GREEN}✅ {message}{Colors.END}")
    elif status == 'warning':
        print(f"{Colors.YELLOW}⚠️ {message}{Colors.END}")
    elif status == 'error':
        print(f"{Colors.RED}❌ {message}{Colors.END}")
    else:  # info
        print(f"{Colors.CYAN}ℹ️ {message}{Colors.END}")

def print_section(title: str, content: Dict[str, str] = None, data: List[List] = None, headers: List[str] = None):
    """
    Imprime una sección con título y contenido
    
    Args:
        title: Título de la sección
        content: Diccionario con pares clave-valor
        data: Lista de filas para tabla
        headers: Encabezados de tabla
    """
    print(f"\n{Colors.MAGENTA}{Colors.BOLD}{title}{Colors.END}")
    
    if content:
        for key, value in content.items():
            # Colorear valores positivos/negativos
            value_color = Colors.END
            if isinstance(value, str):
                if "+" in value or (value.replace("%", "").replace("$", "").strip() and float(value.replace("%", "").replace("$", "").strip()) > 0):
                    value_color = Colors.GREEN
                elif "-" in value or (value.replace("%", "").replace("$", "").strip() and float(value.replace("%", "").replace("$", "").strip()) < 0):
                    value_color = Colors.RED
            
            print(f"{key}: {value_color}{value}{Colors.END}")
    
    if data and headers:
        print_table(headers, data)

def loading_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """
    Muestra una barra de progreso
    
    Args:
        iteration: Iteración actual
        total: Total de iteraciones
        prefix: Prefijo
        suffix: Sufijo
        length: Longitud de la barra
        fill: Carácter de relleno
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{Colors.GREEN}{bar}{Colors.END}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()