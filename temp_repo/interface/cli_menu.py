"""
Módulo para la interfaz de línea de comandos del bot

Proporciona funciones para mostrar menús, solicitar input y mostrar información
de manera clara en la terminal.
"""

import os
import sys
import time
from typing import List, Any, Optional

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

def clear_screen():
    """Limpia la pantalla de la terminal"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title: str):
    """
    Muestra un encabezado con formato
    
    Args:
        title: Título del encabezado
    """
    width = min(os.get_terminal_size().columns, 80)
    print(Colors.CYAN + Colors.BOLD + "=" * width + Colors.END)
    print(Colors.CYAN + Colors.BOLD + title.center(width) + Colors.END)
    print(Colors.CYAN + Colors.BOLD + "=" * width + Colors.END)

def display_logo():
    """Muestra el logo del bot"""
    logo = """
  _____       _                     _______           _           
 / ____|     | |                   |__   __|         | |          
| (___   ___ | | __ _ _ __   __ _     | |_ __ __ _  __| | ___ _ __ 
 \___ \ / _ \| |/ _` | '_ \ / _` |    | | '__/ _` |/ _` |/ _ \ '__|
 ____) | (_) | | (_| | | | | (_| |    | | | | (_| | (_| |  __/ |   
|_____/ \___/|_|\__,_|_| |_|\__,_|    |_|_|  \__,_|\__,_|\___|_|   
                                                                  
    """
    print(Colors.GREEN + logo + Colors.END)

def print_menu(options: List[str]) -> int:
    """
    Muestra un menú con opciones numeradas y solicita una elección
    
    Args:
        options: Lista de opciones a mostrar
        
    Returns:
        int: Número de la opción elegida
    """
    print("\nSelecciona una opción:")
    for i, option in enumerate(options, 1):
        print(f"{Colors.CYAN}{i}{Colors.END}. {option}")
    
    return get_user_choice(1, len(options))

def get_user_choice(min_val: int, max_val: int) -> Optional[int]:
    """
    Solicita al usuario elegir un número dentro de un rango
    
    Args:
        min_val: Valor mínimo permitido
        max_val: Valor máximo permitido
        
    Returns:
        Optional[int]: Número elegido o None si se cancela
    """
    while True:
        try:
            choice = input(f"\n> ")
            
            # Opción para cancelar
            if choice.lower() in ['q', 'exit', 'quit', 'cancelar', 'back', 'volver']:
                return None
            
            choice_int = int(choice)
            if min_val <= choice_int <= max_val:
                return choice_int
            else:
                print(f"Elige un número entre {min_val} y {max_val}.")
        except ValueError:
            print("Por favor ingresa un número válido.")

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