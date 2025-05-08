"""
Módulo para la interfaz de menú CLI del bot de trading
Proporciona funciones para mostrar menús y navegar por la aplicación
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional

from .cli_utils import (
    clear_screen, print_header, print_subheader, print_menu, 
    get_user_choice, confirm_action, print_table, print_chart,
    print_status, print_section, loading_bar, Colors
)

def clear_screen():
    """Limpia la pantalla de la terminal"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title: str):
    """
    Imprime un encabezado con el título proporcionado
    
    Args:
        title: Título a mostrar
    """
    terminal_width = 80
    separator = "=" * terminal_width
    
    clear_screen()
    print(f"{Colors.BRIGHT_YELLOW}{separator}{Colors.END}")
    print(f"{Colors.BRIGHT_YELLOW}{title.center(terminal_width)}{Colors.END}")
    print(f"{Colors.BRIGHT_YELLOW}{separator}{Colors.END}")
    print()

def display_logo():
    """Muestra el logo ASCII del bot"""
    print(f"{Colors.BRIGHT_YELLOW}")
    print(r"   _____       _                    _____              _ _             ____        _   ")
    print(r"  / ____|     | |                  |_   _|            | (_)           |  _ \      | |  ")
    print(r" | (___   ___ | | __ _ _ __   __ _   | |_ __ __ _  __| |_ _ __   __ _| |_) | ___ | |_ ")
    print(r"  \___ \ / _ \| |/ _` | '_ \ / _` |  | | '__/ _` |/ _` | | '_ \ / _` |  _ < / _ \| __|")
    print(r"  ____) | (_) | | (_| | | | | (_| |  | | | | (_| | (_| | | | | | (_| | |_) | (_) | |_ ")
    print(r" |_____/ \___/|_|\__,_|_| |_|\__,_|  |_|_|  \__,_|\__,_|_|_| |_|\__, |____/ \___/ \__|")
    print(r"                                                                  __/ |                ")
    print(r"                                                                 |___/                 ")
    print(f"{Colors.END}")

def welcome_screen():
    """Muestra la pantalla de bienvenida"""
    clear_screen()
    display_logo()
    print(f"\n{Colors.CYAN}Bienvenido al Solana Trading Bot - Un sistema avanzado de trading algorítmico{Colors.END}")
    print(f"\n{Colors.GREEN}El bot está diseñado para operar en el mercado de Solana (SOL) utilizando estrategias adaptativas y aprendizaje automático.{Colors.END}")
    print(f"\n{Colors.YELLOW}Características principales:{Colors.END}")
    print(f"{Colors.YELLOW}• Múltiples estrategias de trading configurables{Colors.END}")
    print(f"{Colors.YELLOW}• Modos de operación simulada (paper) y real{Colors.END}")
    print(f"{Colors.YELLOW}• Backtesting y optimización de parámetros{Colors.END}")
    print(f"{Colors.YELLOW}• Sistema adaptativo que aprende y mejora con el tiempo{Colors.END}")
    print(f"{Colors.YELLOW}• Soporte para múltiples bots simultáneos{Colors.END}")
    print(f"{Colors.YELLOW}• Gestión avanzada de riesgos{Colors.END}")
    print(f"{Colors.YELLOW}• Notificaciones por Telegram{Colors.END}")
    
    print(f"\n{Colors.BRIGHT_CYAN}Presiona Enter para continuar...{Colors.END}")
    input()