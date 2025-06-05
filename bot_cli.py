"""
Interfaz de línea de comandos para el bot de trading
Permite interactuar con el sistema desde la terminal
"""

import os
import sys
import time
import logging
import json
import subprocess
from typing import List, Dict, Any, Optional, Union
from enum import Enum, auto
from datetime import datetime, timedelta

# --- Configuración de Logging para bot_cli ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BotCLI')

# --- Función para verificar e instalar dependencias ---
def check_and_install_dependencies():
    required_packages = [
        "ccxt", "pandas", "numpy", "matplotlib",
        "scikit-learn", "statsmodels", "tabulate",
        "websocket-client", "python-dotenv"
    ]

    try:
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            print("Instalando dependencias necesarias...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                *missing_packages
            ])
            print("Dependencias instaladas correctamente.")
        else:
            print("Todas las dependencias ya están satisfechas.")

        return True
    except Exception as e:
        print(f"Error al instalar dependencias: {e}")
        print("Por favor, instala manualmente: pip install ccxt pandas numpy matplotlib scikit-learn statsmodels tabulate websocket-client python-dotenv")
        return False

# --- Definición de la Enumeración para Opciones del Menú ---
class BotMenuOption(Enum):
    STATUS = auto()
    MARKET_DATA = auto()
    WALLET = auto()
    POSITIONS = auto()
    TRADES = auto()
    EXIT = auto()

# --- Funciones de Utilidad y Display ---
def print_table(headers: List[str], data: List[List[str]], title: str):
    try:
        from tabulate import tabulate
        print(f"\n--- {title} ---")
        print(tabulate(data, headers=headers, tablefmt="grid"))
        print("----------------------------------")
    except ImportError:
        logger.error("La librería 'tabulate' no está instalada. No se pueden imprimir tablas.")
        print(f"\n--- {title} ---")
        for row in data:
            print(row)
        print("----------------------------------")

def display_main_menu():
    print("\n--- Menú Principal del Bot de Trading ---")
    print("1. Estado del Sistema")
    print("2. Datos de Mercado (SOL-USDT)")
    print("3. Balance de Billetera")
    print("4. Posiciones Abiertas")
    print("5. Historial de Trades")
    print("q. Salir")
    print("------------------------------------------")

def display_system_status():
    logger.info("Mostrando estado del sistema...")
    system_data = [
        ["Estado", "Operativo"],
        ["Uptime", "Simulado: 3d 12h 34m"],
        ["CPU Uso", "Simulado: 12%"],
        ["Memoria Uso", "Simulado: 324 MB"],
        ["Conexión API", "OK"],
        ["Última operación", "Simulado: 2025-05-20 14:35:22"],
        ["Fecha actual", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    ]
    print_table(["Métrica", "Valor"], system_data, "Recursos del Sistema")

def display_market_data():
    logger.info("Mostrando datos de mercado...")
    market_data = [
        ["Último Precio", "$179.31"],
        ["Cambio 24h", "+2.34%"],
        ["Volumen 24h (SOL)", "5.2M"],
        ["Volumen 24h (USDT)", "930M"],
        ["High 24h", "$185.00"],
        ["Low 24h", "$170.00"]
    ]
    print_table(["Métrica", "Valor"], market_data, "Datos de Mercado SOL-USDT")

def display_wallet_balances():
    logger.info("Mostrando balances de billetera...")
    balances = [
        ["SOL", "5.75 SOL"],
        ["USDT", "1250.60 USDT"],
        ["Total (USD)", "$2280.00 USD (simulado)"]
    ]
    print_table(["Activo", "Balance"], balances, "Balances de Billetera")

def display_open_positions():
    logger.info("Mostrando posiciones abiertas...")
    positions = [
        ["SOL-USDT", "Long", "0.5 SOL", "$178.50", "N/A", "+$0.40"],
        ["BTC-USDT", "Short", "0.001 BTC", "$60100.00", "N/A", "-$0.10"]
    ]
    print_table(["Par", "Tipo", "Cantidad", "Precio Entrada", "Precio Actual", "P&L No Realizado"],
                 positions, "Posiciones Abiertas")

def display_historical_trades():
    logger.info("Mostrando historial de trades...")
    trades = [
        ["2025-06-03 10:30", "SOL-USDT", "Buy", "0.1 SOL", "$178.20", "$17.82"],
        ["2025-06-03 10:25", "SOL-USDT", "Sell", "0.1 SOL", "$179.10", "$17.91"],
        ["2025-06-02 15:00", "BTC-USDT", "Buy", "0.0005 BTC", "$60200.00", "$30.10"]
    ]
    print_table(["Fecha/Hora", "Par", "Tipo", "Cantidad", "Precio", "Total"],
                 trades, "Historial de Trades Recientes")

def handle_menu_selection(choice: str):
    if choice == '1':
        display_system_status()
    elif choice == '2':
        display_market_data()
    elif choice == '3':
        display_wallet_balances()
    elif choice == '4':
        display_open_positions()
    elif choice == '5':
        display_historical_trades()
    elif choice == 'q':
        pass
    else:
        print("Opción no válida. Por favor, intenta de nuevo.")
        logger.warning(f"Opción de menú no válida ingresada: '{choice}'")

def cli_main_loop():
    while True:
        display_main_menu()
        choice = input("Selecciona una opción: ").strip().lower()
        if choice == 'q':
            logger.info("Usuario selecciono 'q'. Saliendo del CLI.")
            break
        handle_menu_selection(choice)

if __name__ == "__main__":
    if check_and_install_dependencies():
        print("\n--- Iniciando CLI del bot. Presiona 'q' para salir. ---")
        cli_main_loop()
        print("\nCLI del bot finalizado. ¡Hasta pronto!")
    else:
        logger.critical("No se pudieron instalar las dependencias. El CLI no puede iniciar.")