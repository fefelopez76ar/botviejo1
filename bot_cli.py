"""
Interfaz de línea de comandos para el Solana Trading Bot
Este script proporciona un menú interactivo para gestionar el bot de trading
"""

import os
import sys
import time
import signal
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log")
    ]
)
logger = logging.getLogger("TradingBotCLI")

# Intentar importar módulos necesarios
try:
    # Módulos del core
    from core.bot import TradingBot
    from core.bot_manager import BotManager
    from core.config import load_config, save_config
    
    # Módulos de interfaz
    from interface.cli_menu import clear_screen, print_header, print_menu, get_user_choice
    from interface.cli_utils import print_table, print_chart, confirm_action
    
    # Módulos de datos
    from data_management.market_data import get_available_symbols, get_market_data
    
    # Módulos de backtesting
    from backtesting.engine import run_backtest
    
    # Módulos de base de datos
    from db.operations import get_bots, get_bot_history
    
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    print(f"Error: {e}")
    print("Make sure all dependencies are installed.")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Variables globales
bot_manager = None
running = True

def init_bot_manager():
    """Inicializa el gestor de bots"""
    global bot_manager
    try:
        config = load_config()
        bot_manager = BotManager(config)
        logger.info("Bot manager initialized")
        return True
    except Exception as e:
        logger.error(f"Error initializing bot manager: {e}")
        print(f"Error initializing bot manager: {e}")
        return False

def main_menu():
    """Muestra el menú principal"""
    clear_screen()
    
    # Mostrar logo y encabezado con colores
    from interface.cli_menu import display_logo, print_header
    display_logo()
    print_header("SOLANA TRADING BOT - MENÚ PRINCIPAL")
    
    from interface.cli_utils import Colors
    
    # Estado del sistema
    try:
        from core.bot_manager import get_system_status
        status = get_system_status()
        
        print(f"\n{Colors.CYAN}Estado del sistema:{Colors.END}")
        print(f"Bots activos: {Colors.GREEN}{status.get('active_bots', 0)}{Colors.END}")
        print(f"Última operación: {status.get('last_trade_time', 'N/A')}")
        print(f"Precio actual SOL: {Colors.GREEN}{status.get('current_price', 0.0):.2f} USDT{Colors.END}")
    except:
        # Si hay error, mostrar mensajes simples
        print(f"\n{Colors.CYAN}Estado del sistema:{Colors.END}")
        print(f"Bot listo para operar")
    
    options = [
        "Gestionar Bots",
        "Configuración",
        "Ver Mercados",
        "Backtesting",
        "Panel de Control",
        "Configuración de IA",
        "Salir"
    ]
    
    choice = print_menu(options)
    
    if choice == 1:
        bot_management_menu()
    elif choice == 2:
        configuration_menu()
    elif choice == 3:
        market_view_menu()
    elif choice == 4:
        backtesting_menu()
    elif choice == 5:
        dashboard_menu()
    elif choice == 6:
        from interface.ai_menu import ai_configuration_menu
        ai_configuration_menu()
    elif choice == 7:
        confirm_exit()
    else:
        print("Opción no válida. Inténtalo de nuevo.")
        time.sleep(1)

def bot_management_menu():
    """Menú de gestión de bots"""
    while True:
        clear_screen()
        print_header("GESTIÓN DE BOTS")
        
        # Obtener lista de bots
        bots = get_bots()
        
        # Mostrar bots activos
        if bots:
            print("\nBots configurados:")
            bot_data = []
            for i, bot in enumerate(bots, 1):
                status = "🟢 Activo" if bot.get("active", False) else "⚪ Inactivo"
                mode = "📝 Simulado" if bot.get("paper_trading", True) else "💰 Real"
                bot_data.append([
                    i,
                    bot.get("name", "Bot sin nombre"),
                    bot.get("symbol", "SOL-USDT"),
                    status,
                    mode,
                    bot.get("strategy", "N/A"),
                    f"{bot.get('balance', 0):.2f} USDT"
                ])
            
            headers = ["#", "Nombre", "Par", "Estado", "Modo", "Estrategia", "Balance"]
            print_table(headers, bot_data)
        else:
            print("\nNo hay bots configurados.")
        
        options = [
            "Crear nuevo bot",
            "Iniciar/detener bot",
            "Editar configuración de bot",
            "Ver detalles de bot",
            "Eliminar bot",
            "Volver al menú principal"
        ]
        
        choice = print_menu(options)
        
        if choice == 1:
            create_new_bot()
        elif choice == 2:
            toggle_bot_status()
        elif choice == 3:
            edit_bot_config()
        elif choice == 4:
            view_bot_details()
        elif choice == 5:
            delete_bot()
        elif choice == 6:
            break
        else:
            print("Opción no válida. Inténtalo de nuevo.")
            time.sleep(1)

def create_new_bot():
    """Crea un nuevo bot de trading"""
    clear_screen()
    print_header("CREAR NUEVO BOT")
    
    # Solicitar nombre del bot
    print("\nIntroduce un nombre para el bot:")
    bot_name = input("> ").strip()
    if not bot_name:
        print("El nombre no puede estar vacío.")
        time.sleep(2)
        return
    
    # Seleccionar par de trading
    symbols = get_available_symbols()
    print("\nSelecciona el par de trading:")
    for i, symbol in enumerate(symbols, 1):
        print(f"{i}. {symbol}")
    
    symbol_choice = get_user_choice(1, len(symbols))
    selected_symbol = symbols[symbol_choice - 1] if symbol_choice else "SOL-USDT"
    
    # Seleccionar modo de trading
    print("\nSelecciona el modo de trading:")
    print("1. Paper Trading (Simulado)")
    print("2. Live Trading (Real)")
    
    mode_choice = get_user_choice(1, 2)
    paper_trading = mode_choice == 1
    
    # Seleccionar estrategia
    print("\nSelecciona la estrategia de trading:")
    strategies = [
        "Cruce de Medias Móviles",
        "RSI + Bollinger Bands",
        "MACD + Tendencia",
        "Estadística (Mean Reversion)",
        "Adaptativa (Múltiples indicadores)",
        "Machine Learning (Experimental)"
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. {strategy}")
    
    strategy_choice = get_user_choice(1, len(strategies))
    selected_strategy = strategies[strategy_choice - 1] if strategy_choice else strategies[0]
    
    # Configurar balance inicial (para paper trading)
    initial_balance = 1000.0
    if paper_trading:
        print("\nIntroduce el balance inicial (USDT):")
        try:
            initial_balance = float(input("> ").strip())
        except ValueError:
            print("Valor no válido. Usando 1000 USDT por defecto.")
            initial_balance = 1000.0
    
    # Crear configuración del bot
    bot_config = {
        "name": bot_name,
        "symbol": selected_symbol,
        "paper_trading": paper_trading,
        "strategy": selected_strategy,
        "balance": initial_balance,
        "active": False,
        "created_at": datetime.now().isoformat(),
        "config": {
            "risk_per_trade": 1.0,  # Porcentaje del balance por operación
            "leverage": 1.0,        # Sin apalancamiento por defecto
            "stop_loss": 2.0,       # ATR multiplier para stop loss
            "take_profit": 3.0,     # ATR multiplier para take profit
            "time_interval": "15m"  # Intervalo de tiempo predeterminado
        }
    }
    
    # Guardar bot en la base de datos
    try:
        bot_manager.create_bot(bot_config)
        print(f"\n✅ Bot '{bot_name}' creado exitosamente!")
    except Exception as e:
        logger.error(f"Error creating bot: {e}")
        print(f"\n❌ Error al crear el bot: {e}")
    
    input("\nPresiona Enter para continuar...")

def toggle_bot_status():
    """Inicia o detiene un bot"""
    bots = get_bots()
    if not bots:
        print("No hay bots configurados.")
        time.sleep(2)
        return
    
    clear_screen()
    print_header("INICIAR/DETENER BOT")
    
    print("\nSelecciona un bot:")
    for i, bot in enumerate(bots, 1):
        status = "🟢 Activo" if bot.get("active", False) else "⚪ Inactivo"
        print(f"{i}. {bot.get('name')} - {status}")
    
    bot_choice = get_user_choice(1, len(bots))
    if not bot_choice:
        return
    
    selected_bot = bots[bot_choice - 1]
    current_status = selected_bot.get("active", False)
    
    action = "detener" if current_status else "iniciar"
    if confirm_action(f"¿Estás seguro de que deseas {action} el bot '{selected_bot.get('name')}'?"):
        try:
            if current_status:
                bot_manager.stop_bot(selected_bot["id"])
                print(f"\n✅ Bot '{selected_bot.get('name')}' detenido exitosamente!")
            else:
                bot_manager.start_bot(selected_bot["id"])
                print(f"\n✅ Bot '{selected_bot.get('name')}' iniciado exitosamente!")
        except Exception as e:
            logger.error(f"Error toggling bot status: {e}")
            print(f"\n❌ Error: {e}")
    
    input("\nPresiona Enter para continuar...")

def edit_bot_config():
    """Edita la configuración de un bot"""
    bots = get_bots()
    if not bots:
        print("No hay bots configurados.")
        time.sleep(2)
        return
    
    clear_screen()
    print_header("EDITAR CONFIGURACIÓN DE BOT")
    
    print("\nSelecciona un bot:")
    for i, bot in enumerate(bots, 1):
        print(f"{i}. {bot.get('name')} - {bot.get('symbol')}")
    
    bot_choice = get_user_choice(1, len(bots))
    if not bot_choice:
        return
    
    selected_bot = bots[bot_choice - 1]
    
    while True:
        clear_screen()
        print_header(f"CONFIGURACIÓN: {selected_bot.get('name')}")
        
        print("\nConfig actual:")
        for key, value in selected_bot.get("config", {}).items():
            print(f"{key}: {value}")
        
        print("\nSelecciona qué modificar:")
        config_options = [
            "Riesgo por operación (%)",
            "Apalancamiento",
            "Stop Loss (multiplicador ATR)",
            "Take Profit (multiplicador ATR)",
            "Intervalo de tiempo",
            "Guardar y volver"
        ]
        
        for i, option in enumerate(config_options, 1):
            print(f"{i}. {option}")
        
        config_choice = get_user_choice(1, len(config_options))
        if config_choice == 6:
            break
        
        # Modificar configuración
        if config_choice == 1:
            print("\nIngresa el nuevo valor de riesgo por operación (%):")
            try:
                risk = float(input("> ").strip())
                if 0.1 <= risk <= 5.0:
                    selected_bot["config"]["risk_per_trade"] = risk
                else:
                    print("El valor debe estar entre 0.1% y 5%")
            except ValueError:
                print("Valor no válido")
        
        elif config_choice == 2:
            print("\nIngresa el nuevo valor de apalancamiento:")
            try:
                leverage = float(input("> ").strip())
                if 1.0 <= leverage <= 10.0:
                    selected_bot["config"]["leverage"] = leverage
                else:
                    print("El valor debe estar entre 1.0 y 10.0")
            except ValueError:
                print("Valor no válido")
        
        elif config_choice == 3:
            print("\nIngresa el nuevo valor de Stop Loss (multiplicador ATR):")
            try:
                stop_loss = float(input("> ").strip())
                if 0.5 <= stop_loss <= 5.0:
                    selected_bot["config"]["stop_loss"] = stop_loss
                else:
                    print("El valor debe estar entre 0.5 y 5.0")
            except ValueError:
                print("Valor no válido")
        
        elif config_choice == 4:
            print("\nIngresa el nuevo valor de Take Profit (multiplicador ATR):")
            try:
                take_profit = float(input("> ").strip())
                if 1.0 <= take_profit <= 10.0:
                    selected_bot["config"]["take_profit"] = take_profit
                else:
                    print("El valor debe estar entre 1.0 y 10.0")
            except ValueError:
                print("Valor no válido")
        
        elif config_choice == 5:
            print("\nSelecciona el nuevo intervalo de tiempo:")
            intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            for i, interval in enumerate(intervals, 1):
                print(f"{i}. {interval}")
            
            interval_choice = get_user_choice(1, len(intervals))
            if interval_choice:
                selected_bot["config"]["time_interval"] = intervals[interval_choice - 1]
        
        time.sleep(1)
    
    # Guardar cambios
    try:
        bot_manager.update_bot(selected_bot["id"], selected_bot)
        print("\n✅ Configuración guardada exitosamente!")
    except Exception as e:
        logger.error(f"Error updating bot config: {e}")
        print(f"\n❌ Error al guardar configuración: {e}")
    
    input("\nPresiona Enter para continuar...")

def view_bot_details():
    """Muestra detalles de un bot específico"""
    bots = get_bots()
    if not bots:
        print("No hay bots configurados.")
        time.sleep(2)
        return
    
    clear_screen()
    print_header("DETALLES DEL BOT")
    
    print("\nSelecciona un bot:")
    for i, bot in enumerate(bots, 1):
        status = "🟢 Activo" if bot.get("active", False) else "⚪ Inactivo"
        print(f"{i}. {bot.get('name')} - {status}")
    
    bot_choice = get_user_choice(1, len(bots))
    if not bot_choice:
        return
    
    selected_bot = bots[bot_choice - 1]
    
    while True:
        clear_screen()
        print_header(f"DETALLES: {selected_bot.get('name')}")
        
        # Información general
        print("\n📊 INFORMACIÓN GENERAL:")
        print(f"ID: {selected_bot.get('id')}")
        print(f"Par: {selected_bot.get('symbol')}")
        print(f"Modo: {'📝 Simulado' if selected_bot.get('paper_trading', True) else '💰 Real'}")
        print(f"Estado: {'🟢 Activo' if selected_bot.get('active', False) else '⚪ Inactivo'}")
        print(f"Estrategia: {selected_bot.get('strategy')}")
        print(f"Balance: {selected_bot.get('balance', 0):.2f} USDT")
        print(f"Creado: {selected_bot.get('created_at')}")
        
        # Configuración
        print("\n⚙️ CONFIGURACIÓN:")
        for key, value in selected_bot.get("config", {}).items():
            print(f"{key}: {value}")
        
        # Estado actual
        print("\n🔄 ESTADO ACTUAL:")
        if selected_bot.get("active", False):
            try:
                status = bot_manager.get_bot_status(selected_bot["id"])
                print(f"Posición abierta: {'Sí' if status.get('has_position') else 'No'}")
                print(f"Tipo de posición: {status.get('position_type', 'N/A')}")
                print(f"Tamaño: {status.get('position_size', 0)}")
                print(f"Precio de entrada: {status.get('entry_price', 0):.2f}")
                print(f"Precio actual: {status.get('current_price', 0):.2f}")
                print(f"P&L: {status.get('pnl', 0):.2f} USDT")
                print(f"ROI: {status.get('roi', 0):.2f}%")
            except Exception as e:
                print(f"Error al obtener estado: {e}")
        else:
            print("Bot inactivo")
        
        # Historial reciente
        print("\n📜 HISTORIAL RECIENTE:")
        try:
            history = get_bot_history(selected_bot["id"], limit=5)
            if history:
                for entry in history:
                    trade_type = "🟢 COMPRA" if entry.get("type") == "buy" else "🔴 VENTA"
                    print(f"{entry.get('timestamp')} - {trade_type} - {entry.get('price'):.2f} - {entry.get('size'):.4f} - {entry.get('pnl', 0):.2f} USDT")
            else:
                print("No hay historial disponible")
        except Exception as e:
            print(f"Error al obtener historial: {e}")
        
        # Menú de opciones
        print("\nOpciones:")
        options = [
            "Actualizar",
            "Ver gráfico",
            "Ver historial completo",
            "Volver"
        ]
        
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        
        choice = get_user_choice(1, len(options))
        
        if choice == 1:
            continue  # Actualizar (simplemente refrescar la pantalla)
        elif choice == 2:
            view_bot_chart(selected_bot)
        elif choice == 3:
            view_bot_history(selected_bot)
        elif choice == 4:
            break
        else:
            print("Opción no válida")
            time.sleep(1)

def view_bot_chart(bot):
    """Muestra el gráfico de un bot"""
    clear_screen()
    print_header(f"GRÁFICO: {bot.get('name')}")
    
    try:
        # Obtener datos para el gráfico
        market_data = get_market_data(bot.get("symbol"), bot.get("config", {}).get("time_interval", "15m"))
        
        # Mostrar un gráfico ASCII simple
        if market_data:
            print_chart(market_data, title=f"{bot.get('symbol')} - {bot.get('config', {}).get('time_interval', '15m')}")
        else:
            print("No hay datos disponibles para mostrar el gráfico")
    except Exception as e:
        logger.error(f"Error displaying chart: {e}")
        print(f"Error al mostrar el gráfico: {e}")
    
    input("\nPresiona Enter para continuar...")

def view_bot_history(bot):
    """Muestra el historial completo de un bot"""
    clear_screen()
    print_header(f"HISTORIAL: {bot.get('name')}")
    
    try:
        history = get_bot_history(bot["id"])
        if history:
            print(f"\nHistorial de operaciones ({len(history)} operaciones):")
            
            history_data = []
            for entry in history:
                trade_type = "COMPRA" if entry.get("type") == "buy" else "VENTA"
                history_data.append([
                    entry.get('timestamp'),
                    trade_type,
                    f"{entry.get('price'):.2f}",
                    f"{entry.get('size'):.4f}",
                    f"{entry.get('pnl', 0):.2f} USDT",
                    entry.get('reason', 'N/A')
                ])
            
            headers = ["Fecha", "Tipo", "Precio", "Tamaño", "P&L", "Razón"]
            print_table(headers, history_data)
            
            # Mostrar resumen
            total_trades = len(history)
            winning_trades = sum(1 for entry in history if entry.get('pnl', 0) > 0)
            losing_trades = sum(1 for entry in history if entry.get('pnl', 0) <= 0)
            
            total_profit = sum(entry.get('pnl', 0) for entry in history if entry.get('pnl', 0) > 0)
            total_loss = sum(entry.get('pnl', 0) for entry in history if entry.get('pnl', 0) <= 0)
            
            print("\nResumen:")
            print(f"Total operaciones: {total_trades}")
            print(f"Operaciones ganadoras: {winning_trades} ({winning_trades/total_trades*100:.1f}%)")
            print(f"Operaciones perdedoras: {losing_trades} ({losing_trades/total_trades*100:.1f}%)")
            print(f"Ganancia total: {total_profit:.2f} USDT")
            print(f"Pérdida total: {total_loss:.2f} USDT")
            print(f"Resultado neto: {(total_profit + total_loss):.2f} USDT")
        else:
            print("No hay historial disponible")
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        print(f"Error al obtener historial: {e}")
    
    input("\nPresiona Enter para continuar...")

def delete_bot():
    """Elimina un bot"""
    bots = get_bots()
    if not bots:
        print("No hay bots configurados.")
        time.sleep(2)
        return
    
    clear_screen()
    print_header("ELIMINAR BOT")
    
    print("\nSelecciona un bot para eliminar:")
    for i, bot in enumerate(bots, 1):
        status = "🟢 Activo" if bot.get("active", False) else "⚪ Inactivo"
        print(f"{i}. {bot.get('name')} - {status}")
    
    bot_choice = get_user_choice(1, len(bots))
    if not bot_choice:
        return
    
    selected_bot = bots[bot_choice - 1]
    
    if confirm_action(f"¿Estás SEGURO de que deseas ELIMINAR el bot '{selected_bot.get('name')}'? Esta acción no se puede deshacer."):
        try:
            bot_manager.delete_bot(selected_bot["id"])
            print(f"\n✅ Bot '{selected_bot.get('name')}' eliminado exitosamente!")
        except Exception as e:
            logger.error(f"Error deleting bot: {e}")
            print(f"\n❌ Error al eliminar el bot: {e}")
    
    input("\nPresiona Enter para continuar...")

def configuration_menu():
    """Menú de configuración"""
    while True:
        clear_screen()
        print_header("CONFIGURACIÓN")
        
        # Cargar configuración actual
        config = load_config()
        
        print("\nConfiguración actual:")
        # Mostrar configuración principal (sin API keys completas)
        sanitized_config = config.copy()
        if "api_key" in sanitized_config:
            sanitized_config["api_key"] = sanitized_config["api_key"][:5] + "****"
        if "api_secret" in sanitized_config:
            sanitized_config["api_secret"] = "****"
        if "api_passphrase" in sanitized_config:
            sanitized_config["api_passphrase"] = "****"
        
        for key, value in sanitized_config.items():
            print(f"{key}: {value}")
        
        options = [
            "Configurar API de OKX",
            "Configurar Telegram",
            "Configurar parámetros generales",
            "Volver al menú principal"
        ]
        
        choice = print_menu(options)
        
        if choice == 1:
            configure_okx_api()
        elif choice == 2:
            configure_telegram()
        elif choice == 3:
            configure_general_params()
        elif choice == 4:
            break
        else:
            print("Opción no válida. Inténtalo de nuevo.")
            time.sleep(1)

def configure_okx_api():
    """Configura API de OKX"""
    clear_screen()
    print_header("CONFIGURACIÓN DE API OKX")
    
    config = load_config()
    
    print("\nConfigura tus credenciales de API de OKX.")
    print("Puedes obtenerlas en https://www.okx.com/account/my-api")
    print("\nDeja vacío si no deseas cambiar el valor actual.")
    
    # API Key
    current_api_key = config.get("api_key", "")
    masked_key = current_api_key[:5] + "****" if current_api_key else ""
    print(f"\nAPI Key actual: {masked_key}")
    print("Nueva API Key:")
    new_api_key = input("> ").strip()
    
    # API Secret
    print("\nAPI Secret actual: ****")
    print("Nuevo API Secret:")
    new_api_secret = input("> ").strip()
    
    # API Passphrase
    print("\nAPI Passphrase actual: ****")
    print("Nuevo API Passphrase:")
    new_api_passphrase = input("> ").strip()
    
    # Actualizar configuración
    if new_api_key:
        config["api_key"] = new_api_key
    if new_api_secret:
        config["api_secret"] = new_api_secret
    if new_api_passphrase:
        config["api_passphrase"] = new_api_passphrase
    
    # Guardar configuración
    save_config(config)
    print("\n✅ Configuración de API guardada exitosamente!")
    
    input("\nPresiona Enter para continuar...")

def configure_telegram():
    """Configura notificaciones de Telegram"""
    clear_screen()
    print_header("CONFIGURACIÓN DE TELEGRAM")
    
    config = load_config()
    
    print("\nConfigura tus credenciales de Telegram para recibir notificaciones.")
    print("Para obtener un token de bot, contacta a @BotFather en Telegram.")
    print("Para obtener tu chat ID, contacta a @userinfobot en Telegram.")
    print("\nDeja vacío si no deseas cambiar el valor actual.")
    
    # Bot Token
    current_token = config.get("telegram_token", "")
    masked_token = current_token[:5] + "****" if current_token else ""
    print(f"\nBot Token actual: {masked_token}")
    print("Nuevo Bot Token:")
    new_token = input("> ").strip()
    
    # Chat ID
    current_chat_id = config.get("telegram_chat_id", "")
    print(f"\nChat ID actual: {current_chat_id}")
    print("Nuevo Chat ID:")
    new_chat_id = input("> ").strip()
    
    # Actualizar configuración
    if new_token:
        config["telegram_token"] = new_token
    if new_chat_id:
        config["telegram_chat_id"] = new_chat_id
    
    # Guardar configuración
    save_config(config)
    print("\n✅ Configuración de Telegram guardada exitosamente!")
    
    input("\nPresiona Enter para continuar...")

def configure_general_params():
    """Configura parámetros generales"""
    clear_screen()
    print_header("PARÁMETROS GENERALES")
    
    config = load_config()
    
    print("\nConfigura parámetros generales de la aplicación.")
    
    # Default Symbol
    current_symbol = config.get("default_symbol", "SOL-USDT")
    print(f"\nPar predeterminado actual: {current_symbol}")
    print("Nuevo par predeterminado:")
    new_symbol = input("> ").strip()
    
    # Default Interval
    current_interval = config.get("default_interval", "15m")
    print(f"\nIntervalo predeterminado actual: {current_interval}")
    print("Nuevo intervalo predeterminado (1m, 5m, 15m, 30m, 1h, 4h, 1d):")
    new_interval = input("> ").strip()
    
    # Default Mode
    current_mode = "paper" if config.get("default_paper_trading", True) else "live"
    print(f"\nModo predeterminado actual: {current_mode}")
    print("Nuevo modo predeterminado (paper/live):")
    new_mode = input("> ").strip().lower()
    
    # Actualizar configuración
    if new_symbol:
        config["default_symbol"] = new_symbol
    if new_interval:
        if new_interval in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]:
            config["default_interval"] = new_interval
        else:
            print("Intervalo no válido. No se ha actualizado.")
    if new_mode in ["paper", "live"]:
        config["default_paper_trading"] = (new_mode == "paper")
    elif new_mode:
        print("Modo no válido. No se ha actualizado.")
    
    # Guardar configuración
    save_config(config)
    print("\n✅ Parámetros generales guardados exitosamente!")
    
    input("\nPresiona Enter para continuar...")

def market_view_menu():
    """Menú de vista de mercado"""
    while True:
        clear_screen()
        print_header("VISTA DE MERCADO")
        
        # Obtener símbolos disponibles
        symbols = get_available_symbols()
        
        print("\nSelecciona un par de trading:")
        for i, symbol in enumerate(symbols, 1):
            print(f"{i}. {symbol}")
        print(f"{len(symbols) + 1}. Volver al menú principal")
        
        choice = get_user_choice(1, len(symbols) + 1)
        if choice == len(symbols) + 1:
            break
        
        if 1 <= choice <= len(symbols):
            selected_symbol = symbols[choice - 1]
            view_market_data(selected_symbol)
        else:
            print("Opción no válida. Inténtalo de nuevo.")
            time.sleep(1)

def view_market_data(symbol):
    """Muestra datos de mercado para un símbolo"""
    while True:
        clear_screen()
        print_header(f"DATOS DE MERCADO: {symbol}")
        
        # Seleccionar intervalo
        print("\nSelecciona un intervalo de tiempo:")
        intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        for i, interval in enumerate(intervals, 1):
            print(f"{i}. {interval}")
        print(f"{len(intervals) + 1}. Volver")
        
        choice = get_user_choice(1, len(intervals) + 1)
        if choice == len(intervals) + 1:
            break
        
        if 1 <= choice <= len(intervals):
            selected_interval = intervals[choice - 1]
            
            # Obtener datos de mercado
            try:
                market_data = get_market_data(symbol, selected_interval)
                
                clear_screen()
                print_header(f"DATOS DE MERCADO: {symbol} - {selected_interval}")
                
                # Mostrar precio actual
                print(f"\nPrecio actual: {market_data['close'].iloc[-1]:.2f} USDT")
                print(f"Cambio 24h: {((market_data['close'].iloc[-1] / market_data['close'].iloc[-24] - 1) * 100):.2f}%")
                
                # Mostrar gráfico
                print_chart(market_data, title=f"{symbol} - {selected_interval}")
                
                # Mostrar análisis técnico básico
                print("\nAnálisis Técnico:")
                
                # RSI
                from indicators.technical import calculate_rsi
                rsi = calculate_rsi(market_data['close'])
                print(f"RSI (14): {rsi.iloc[-1]:.2f} - {'Sobrecomprado' if rsi.iloc[-1] > 70 else 'Sobreventa' if rsi.iloc[-1] < 30 else 'Neutral'}")
                
                # MACD
                from indicators.technical import calculate_macd
                macd, signal, hist = calculate_macd(market_data['close'])
                macd_signal = "Alcista" if macd.iloc[-1] > signal.iloc[-1] else "Bajista"
                print(f"MACD: {macd.iloc[-1]:.4f}, Signal: {signal.iloc[-1]:.4f} - {macd_signal}")
                
                # Bandas de Bollinger
                from indicators.technical import calculate_bollinger_bands
                mid, upper, lower = calculate_bollinger_bands(market_data['close'])
                bb_position = (market_data['close'].iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
                bb_status = "Sobrecomprado" if bb_position > 1 else "Sobreventa" if bb_position < 0 else "Neutral"
                print(f"Bollinger: {bb_position:.2f} - {bb_status}")
                
                # Tendencia
                sma20 = market_data['close'].rolling(20).mean().iloc[-1]
                sma50 = market_data['close'].rolling(50).mean().iloc[-1]
                trend = "Alcista" if sma20 > sma50 else "Bajista"
                print(f"Tendencia (SMA20 vs SMA50): {trend}")
                
                print("\nOpciones:")
                print("1. Actualizar datos")
                print("2. Cambiar intervalo")
                print("3. Volver")
                
                subchoice = get_user_choice(1, 3)
                if subchoice == 2:
                    break  # Volver a selección de intervalo
                elif subchoice == 3:
                    return  # Salir completamente
                # Si es 1, simplemente se repite el bucle (actualizar)
                
            except Exception as e:
                logger.error(f"Error retrieving market data: {e}")
                print(f"Error al obtener datos de mercado: {e}")
                input("\nPresiona Enter para continuar...")
                break
        else:
            print("Opción no válida. Inténtalo de nuevo.")
            time.sleep(1)

def backtesting_menu():
    """Menú de backtesting"""
    while True:
        clear_screen()
        print_header("BACKTESTING")
        
        options = [
            "Ejecutar backtest",
            "Ejecutar backtest de todas las estrategias",
            "Ver resultados anteriores",
            "Optimizar parámetros",
            "Análisis de tendencia del mercado",
            "Sistema de aprendizaje automático",
            "Volver al menú principal"
        ]
        
        choice = print_menu(options)
        
        if choice == 1:
            run_new_backtest()
        elif choice == 2:
            run_multi_strategy_backtest_menu()
        elif choice == 3:
            view_backtest_results()
        elif choice == 4:
            optimize_parameters()
        elif choice == 5:
            market_trend_analysis()
        elif choice == 6:
            automated_learning_menu()
        elif choice == 7:
            break
        else:
            print("Opción no válida. Inténtalo de nuevo.")
            time.sleep(1)

def run_multi_strategy_backtest_menu():
    """Menú para ejecutar backtesting de todas las estrategias"""
    clear_screen()
    print_header("BACKTEST DE TODAS LAS ESTRATEGIAS")
    
    # Seleccionar símbolo
    symbols = get_available_symbols()
    
    print("\nSelecciona el par de trading:")
    for i, symbol in enumerate(symbols, 1):
        print(f"{i}. {symbol}")
    
    symbol_choice = get_user_choice(1, len(symbols))
    if not symbol_choice:
        return
    
    selected_symbol = symbols[symbol_choice - 1]
    
    # Seleccionar periodo
    print("\nSelecciona el periodo de backtesting:")
    periods = [
        "Últimos 30 días",
        "Últimos 60 días",
        "Últimos 90 días",
        "Últimos 180 días"
    ]
    
    for i, period in enumerate(periods, 1):
        print(f"{i}. {period}")
    
    period_choice = get_user_choice(1, len(periods))
    if not period_choice:
        return
    
    # Mapear elección a días
    days_map = {1: 30, 2: 60, 3: 90, 4: 180}
    days = days_map.get(period_choice, 90)
    
    # Seleccionar intervalo
    print("\nSelecciona el intervalo de tiempo:")
    intervals = ["15m", "1h", "4h", "1d"]
    
    for i, interval in enumerate(intervals, 1):
        print(f"{i}. {interval}")
    
    interval_choice = get_user_choice(1, len(intervals))
    if not interval_choice:
        return
    
    selected_interval = intervals[interval_choice - 1]
    
    # Ejecutar backtesting
    print(f"\n⚙️ Ejecutando backtesting de todas las estrategias para {selected_symbol} en {selected_interval} (últimos {days} días)")
    print("Este proceso puede tardar varios minutos.")
    
    try:
        # Importar solo cuando se necesita para evitar errores de inicio
        from backtesting.advanced_optimizer import run_multi_strategy_backtest
        
        # Mostrar progreso
        print("\nRecopilando datos históricos...")
        time.sleep(1)
        
        # Simular progreso
        print("\nEjecutando estrategias:")
        strategies = [
            "SMA Crossover", "RSI Strategy", "MACD Strategy", 
            "Bollinger Bands", "Mean Reversion", "RSI+MACD Combined", 
            "SMA Crossover (Fast)", "SMA Crossover (Slow)", 
            "RSI (Aggressive)", "RSI (Conservative)"
        ]
        
        for i, strategy in enumerate(strategies):
            progress = (i + 1) / len(strategies) * 100
            print(f"Procesando {strategy:<20} [{progress:>6.2f}%]", end="\r")
            time.sleep(0.5)
        
        print("\nFinalizando análisis..." + " " * 30)
        time.sleep(1)
        
        # Ejecutar backtesting
        results = run_multi_strategy_backtest(selected_symbol, selected_interval, days)
        
        # Mostrar resultados
        clear_screen()
        print_header("RESULTADOS DE BACKTESTING MÚLTIPLE")
        
        print(f"\nPar: {selected_symbol}")
        print(f"Intervalo: {selected_interval}")
        print(f"Periodo: Últimos {days} días")
        
        # Detectar tendencia
        trend = results.get("trend_report", {}).get("current_trend", "No disponible")
        print(f"\nTendencia detectada: {trend}")
        
        # Mostrar mejores estrategias
        print("\nMejores estrategias:")
        
        best_strategies = results.get("best_strategies", [])
        
        if best_strategies:
            strategy_data = []
            for i, strategy in enumerate(best_strategies, 1):
                strategy_data.append([
                    i,
                    strategy["name"],
                    f"{strategy['return_pct']:.2f}%",
                    f"{strategy['win_rate']:.2f}%",
                    f"{strategy['profit_factor']:.2f}",
                    f"{strategy['sharpe_ratio']:.2f}",
                    strategy["total_trades"]
                ])
            
            headers = ["#", "Estrategia", "Retorno", "Win Rate", "Profit Factor", "Sharpe", "Trades"]
            print_table(headers, strategy_data)
            
            # Mostrar mejores estrategias por tendencia
            trend_report = results.get("trend_report", {})
            
            if trend_report:
                print("\nMejores estrategias por tendencia:")
                
                for trend_type, trend_data in trend_report.items():
                    if trend_type == "current_trend":
                        continue
                    
                    symbol_data = trend_data.get(selected_symbol, [])
                    if symbol_data:
                        best_for_trend = symbol_data[0]["name"]
                        print(f"{trend_type}: {best_for_trend}")
        else:
            print("No se encontraron resultados.")
        
        # Preguntar si crear bot con la mejor estrategia
        if best_strategies:
            best_strategy = best_strategies[0]["name"]
            
            print(f"\n¿Deseas crear un nuevo bot con la estrategia '{best_strategy}'?")
            print("1. Sí, crear bot en modo simulado")
            print("2. No")
            
            create_choice = get_user_choice(1, 2)
            
            if create_choice == 1:
                # Crear bot con la mejor estrategia
                create_bot_from_strategy(selected_symbol, best_strategy)
    
    except ImportError as e:
        print(f"\n❌ Error: Funcionalidad no disponible. {e}")
    except Exception as e:
        print(f"\n❌ Error al ejecutar backtesting múltiple: {e}")
    
    input("\nPresiona Enter para continuar...")

def create_bot_from_strategy(symbol: str, strategy_name: str):
    """Crea un bot a partir de una estrategia"""
    clear_screen()
    print_header("CREAR BOT DESDE ESTRATEGIA")
    
    # Solicitar nombre del bot
    print(f"\nCreando bot con estrategia '{strategy_name}' para {symbol}")
    print("\nIntroduce un nombre para el bot:")
    bot_name = input("> ").strip()
    if not bot_name:
        print("El nombre no puede estar vacío.")
        time.sleep(2)
        return
    
    # Modo simulado
    paper_trading = True
    
    # Configurar balance inicial
    print("\nIntroduce el balance inicial (USDT):")
    try:
        initial_balance = float(input("> ").strip())
    except ValueError:
        print("Valor no válido. Usando 1000 USDT por defecto.")
        initial_balance = 1000.0
    
    # Crear configuración del bot
    bot_config = {
        "name": bot_name,
        "symbol": symbol,
        "paper_trading": paper_trading,
        "strategy": strategy_name,
        "balance": initial_balance,
        "active": False,
        "created_at": datetime.now().isoformat(),
        "config": {
            "risk_per_trade": 1.0,
            "leverage": 1.0,
            "stop_loss": 2.0,
            "take_profit": 3.0,
            "time_interval": "15m"
        }
    }
    
    # Guardar bot en la base de datos
    try:
        bot_manager.create_bot(bot_config)
        print(f"\n✅ Bot '{bot_name}' creado exitosamente con la estrategia '{strategy_name}'!")
    except Exception as e:
        logger.error(f"Error creating bot: {e}")
        print(f"\n❌ Error al crear el bot: {e}")

def market_trend_analysis():
    """Analiza la tendencia actual del mercado"""
    clear_screen()
    print_header("ANÁLISIS DE TENDENCIA DEL MERCADO")
    
    # Seleccionar símbolo
    symbols = get_available_symbols()
    
    print("\nSelecciona el par de trading:")
    for i, symbol in enumerate(symbols, 1):
        print(f"{i}. {symbol}")
    
    symbol_choice = get_user_choice(1, len(symbols))
    if not symbol_choice:
        return
    
    selected_symbol = symbols[symbol_choice - 1]
    
    # Seleccionar intervalo
    print("\nSelecciona el intervalo de tiempo:")
    intervals = ["15m", "1h", "4h", "1d"]
    
    for i, interval in enumerate(intervals, 1):
        print(f"{i}. {interval}")
    
    interval_choice = get_user_choice(1, len(intervals))
    if not interval_choice:
        return
    
    selected_interval = intervals[interval_choice - 1]
    
    # Ejecutar análisis
    print(f"\n⚙️ Analizando tendencia de mercado para {selected_symbol} en {selected_interval}...")
    
    try:
        # Importar solo cuando se necesita para evitar errores de inicio
        from backtesting.advanced_optimizer import analyze_current_market
        
        # Ejecutar análisis
        analysis = analyze_current_market(selected_symbol, selected_interval)
        
        # Mostrar resultados
        clear_screen()
        print_header("RESULTADOS DE ANÁLISIS DE TENDENCIA")
        
        print(f"\nPar: {selected_symbol}")
        print(f"Intervalo: {selected_interval}")
        print(f"Fecha de análisis: {analysis.get('timestamp', 'N/A')}")
        
        # Mostrar tendencia detectada
        trend = analysis.get("trend", "No disponible")
        market_condition = analysis.get("market_condition", "No disponible")
        
        print(f"\nTendencia detectada: {trend}")
        print(f"Condición de mercado: {market_condition}")
        print(f"Volatilidad: {analysis.get('volatility', 0) * 100:.2f}%")
        print(f"Días en tendencia actual: {analysis.get('days_in_trend', 0)}")
        
        # Mostrar precios y medias
        print(f"\nPrecio actual: {analysis.get('current_price', 0):.2f}")
        print(f"SMA 20: {analysis.get('sma20', 0):.2f}")
        print(f"SMA 50: {analysis.get('sma50', 0):.2f}")
        print(f"SMA 100: {analysis.get('sma100', 0):.2f}")
        
        # Recomendaciones de estrategias
        print("\nRecomendaciones de estrategias para la tendencia actual:")
        
        trend_recommendations = {
            "strong_uptrend": ["SMA Crossover", "MACD Strategy", "RSI (Aggressive)"],
            "weak_uptrend": ["SMA Crossover", "MACD Strategy", "Bollinger Bands"],
            "neutral": ["Mean Reversion", "Bollinger Bands", "RSI Strategy"],
            "weak_downtrend": ["Mean Reversion", "RSI Strategy", "Bollinger Bands"],
            "strong_downtrend": ["Mean Reversion", "RSI (Conservative)", "MACD Strategy"],
            "choppy": ["Bollinger Bands", "Mean Reversion", "RSI+MACD Combined"],
            "volatile": ["RSI+MACD Combined", "Bollinger Bands", "RSI (Conservative)"]
        }
        
        if trend in trend_recommendations:
            for i, strategy in enumerate(trend_recommendations[trend], 1):
                print(f"{i}. {strategy}")
        else:
            print("No hay recomendaciones disponibles para esta tendencia.")
        
        # Preguntar si crear bot con estrategia recomendada
        if trend in trend_recommendations:
            best_strategy = trend_recommendations[trend][0]
            
            print(f"\n¿Deseas crear un nuevo bot con la estrategia recomendada '{best_strategy}'?")
            print("1. Sí, crear bot en modo simulado")
            print("2. No")
            
            create_choice = get_user_choice(1, 2)
            
            if create_choice == 1:
                # Crear bot con la estrategia recomendada
                create_bot_from_strategy(selected_symbol, best_strategy)
    
    except ImportError as e:
        print(f"\n❌ Error: Funcionalidad no disponible. {e}")
    except Exception as e:
        print(f"\n❌ Error al analizar tendencia: {e}")
    
    input("\nPresiona Enter para continuar...")

def automated_learning_menu():
    """Menú del sistema de aprendizaje automático"""
    while True:
        clear_screen()
        print_header("SISTEMA DE APRENDIZAJE AUTOMÁTICO")
        
        from interface.cli_utils import Colors
        
        print(f"\n{Colors.CYAN}El sistema de aprendizaje automático permite:{Colors.END}")
        print(f"{Colors.YELLOW}• Detectar automáticamente tendencias y condiciones de mercado{Colors.END}")
        print(f"{Colors.YELLOW}• Ejecutar backtesting de todas las estrategias disponibles{Colors.END}")
        print(f"{Colors.YELLOW}• Optimizar parámetros para cada estrategia{Colors.END}")
        print(f"{Colors.YELLOW}• Crear y ejecutar bots en modo simulado para aprendizaje{Colors.END}")
        print(f"{Colors.YELLOW}• Mejorar continuamente basándose en resultados{Colors.END}")
        
        options = [
            "Iniciar ciclo de aprendizaje",
            "Ver estado del sistema de aprendizaje",
            "Crear bot optimizado",
            "Crear bot con Machine Learning",
            "Volver al menú de backtesting"
        ]
        
        choice = print_menu(options)
        
        if choice == 1:
            run_learning_cycle()
        elif choice == 2:
            view_learning_system_status()
        elif choice == 3:
            create_optimized_bot()
        elif choice == 4:
            create_ml_bot()
        elif choice == 5:
            break
        else:
            print("Opción no válida. Inténtalo de nuevo.")
            time.sleep(1)

def run_learning_cycle():
    """Ejecuta un ciclo completo de aprendizaje"""
    clear_screen()
    print_header("INICIAR CICLO DE APRENDIZAJE")
    
    # Seleccionar símbolos
    symbols = get_available_symbols()
    
    print("\nSelecciona los pares de trading a incluir en el ciclo de aprendizaje:")
    for i, symbol in enumerate(symbols, 1):
        print(f"{i}. {symbol}")
    
    print("\nIngresa los números separados por comas (ej: 1,3,5) o 'a' para todos:")
    selection = input("> ").strip().lower()
    
    selected_symbols = []
    if selection == 'a':
        selected_symbols = symbols
    else:
        try:
            indices = [int(idx.strip()) for idx in selection.split(',') if idx.strip()]
            selected_symbols = [symbols[idx-1] for idx in indices if 1 <= idx <= len(symbols)]
        except:
            print("Selección inválida. Usando el primer símbolo por defecto.")
            selected_symbols = [symbols[0]] if symbols else []
    
    if not selected_symbols:
        print("No se seleccionaron símbolos. Operación cancelada.")
        time.sleep(2)
        return
    
    # Seleccionar intervalo
    print("\nSelecciona el intervalo de tiempo:")
    intervals = ["15m", "1h", "4h", "1d"]
    
    for i, interval in enumerate(intervals, 1):
        print(f"{i}. {interval}")
    
    interval_choice = get_user_choice(1, len(intervals))
    if not interval_choice:
        return
    
    selected_interval = intervals[interval_choice - 1]
    
    # Confirmar inicio
    symbols_str = ", ".join(selected_symbols)
    if not confirm_action(f"¿Confirmas iniciar un ciclo de aprendizaje para {symbols_str} en {selected_interval}? Este proceso puede tardar varios minutos."):
        return
    
    # Ejecutar ciclo de aprendizaje
    print(f"\n⚙️ Iniciando ciclo de aprendizaje para {symbols_str} en {selected_interval}...")
    print("Este proceso puede tardar varios minutos.")
    
    try:
        # Importar solo cuando se necesita para evitar errores de inicio
        from backtesting.advanced_optimizer import create_auto_learning_bots
        
        # Ejecutar ciclo
        results = create_auto_learning_bots(selected_symbols, selected_interval)
        
        # Mostrar resultados
        clear_screen()
        print_header("RESULTADOS DEL CICLO DE APRENDIZAJE")
        
        print(f"\nSímbolos: {symbols_str}")
        print(f"Intervalo: {selected_interval}")
        print(f"Inicio: {results.get('started_at', 'N/A')}")
        print(f"Fin: {results.get('completed_at', 'N/A')}")
        
        # Mostrar bots creados
        created_bots = results.get("created_bots", [])
        
        if created_bots:
            print("\nBots creados durante el ciclo:")
            
            bot_data = []
            for bot in created_bots:
                bot_results = bot.get("results", {})
                bot_data.append([
                    bot.get("bot_id", "N/A"),
                    bot.get("type", "N/A"),
                    bot_results.get("symbol", "N/A"),
                    bot_results.get("strategy", "N/A"),
                    f"{bot_results.get('return_pct', 0):.2f}%",
                    bot_results.get("trades", 0),
                    f"{bot_results.get('win_rate', 0):.2f}%"
                ])
            
            headers = ["Bot ID", "Tipo", "Par", "Estrategia", "Retorno", "Trades", "Win Rate"]
            print_table(headers, bot_data)
        else:
            print("\nNo se crearon bots durante el ciclo.")
    
    except ImportError as e:
        print(f"\n❌ Error: Funcionalidad no disponible. {e}")
    except Exception as e:
        print(f"\n❌ Error al ejecutar ciclo de aprendizaje: {e}")
    
    input("\nPresiona Enter para continuar...")

def view_learning_system_status():
    """Muestra el estado del sistema de aprendizaje"""
    clear_screen()
    print_header("ESTADO DEL SISTEMA DE APRENDIZAJE")
    
    try:
        # Importar solo cuando se necesita para evitar errores de inicio
        from backtesting.advanced_optimizer import get_learning_system_status
        
        # Obtener estado
        status = get_learning_system_status()
        
        # Mostrar resumen
        print("\nResumen del sistema:")
        print(f"Total de bots: {status.get('total_bots', 0)}")
        
        bots_by_type = status.get("bots_by_type", {})
        for bot_type, count in bots_by_type.items():
            print(f"Bots de tipo {bot_type}: {count}")
        
        # Mostrar rendimiento por tipo
        performance_by_type = status.get("performance_by_type", {})
        
        if performance_by_type:
            print("\nRendimiento por tipo de bot:")
            
            for bot_type, perf in performance_by_type.items():
                print(f"\nTipo: {bot_type}")
                print(f"Cantidad: {perf.get('count', 0)}")
                print(f"Retorno promedio: {perf.get('avg_return', 0):.2f}%")
                
                best_bot = perf.get("best_bot", {})
                if best_bot:
                    print(f"Mejor bot: {best_bot.get('name', 'N/A')} ({best_bot.get('return_pct', 0):.2f}%)")
        
        # Mostrar insights de aprendizaje
        insights = status.get("learning_insights", {})
        
        if "error" not in insights:
            print("\nInsights de aprendizaje:")
            
            best_strategies = insights.get("best_strategies_by_condition", {})
            
            for condition, strategies in best_strategies.items():
                print(f"\nCondición: {condition}")
                
                for i, strategy in enumerate(strategies[:3], 1):
                    print(f"{i}. {strategy.get('name', 'N/A')} - Retorno: {strategy.get('avg_return', 0):.2f}% - " + 
                          f"Win Rate: {strategy.get('avg_win_rate', 0):.2f}%")
        else:
            print("\nNo hay insights de aprendizaje disponibles.")
    
    except ImportError as e:
        print(f"\n❌ Error: Funcionalidad no disponible. {e}")
    except Exception as e:
        print(f"\n❌ Error al obtener estado: {e}")
    
    input("\nPresiona Enter para continuar...")

def create_optimized_bot():
    """Crea un bot optimizado basado en aprendizaje"""
    clear_screen()
    print_header("CREAR BOT OPTIMIZADO")
    
    # Seleccionar símbolo
    symbols = get_available_symbols()
    
    print("\nSelecciona el par de trading:")
    for i, symbol in enumerate(symbols, 1):
        print(f"{i}. {symbol}")
    
    symbol_choice = get_user_choice(1, len(symbols))
    if not symbol_choice:
        return
    
    selected_symbol = symbols[symbol_choice - 1]
    
    # Seleccionar intervalo
    print("\nSelecciona el intervalo de tiempo:")
    intervals = ["15m", "1h", "4h", "1d"]
    
    for i, interval in enumerate(intervals, 1):
        print(f"{i}. {interval}")
    
    interval_choice = get_user_choice(1, len(intervals))
    if not interval_choice:
        return
    
    selected_interval = intervals[interval_choice - 1]
    
    # Seleccionar días para optimización
    print("\nSelecciona el periodo para optimización:")
    periods = ["30 días", "60 días", "90 días", "180 días"]
    
    for i, period in enumerate(periods, 1):
        print(f"{i}. {period}")
    
    period_choice = get_user_choice(1, len(periods))
    if not period_choice:
        return
    
    days_map = {1: 30, 2: 60, 3: 90, 4: 180}
    selected_days = days_map.get(period_choice, 60)
    
    # Solicitar nombre del bot
    print("\nIntroduce un nombre para el bot:")
    bot_name = input("> ").strip()
    if not bot_name:
        print("El nombre no puede estar vacío.")
        time.sleep(2)
        return
    
    # Ejecutar creación de bot optimizado
    print(f"\n⚙️ Creando bot optimizado para {selected_symbol} en {selected_interval}...")
    print("Este proceso incluye análisis de mercado, backtesting y optimización de parámetros.")
    print("Puede tardar varios minutos.")
    
    try:
        # Importar solo cuando se necesita para evitar errores de inicio
        from backtesting.advanced_optimizer import AutomatedLearningSystem
        
        # Crear sistema de aprendizaje
        learning_system = AutomatedLearningSystem()
        
        # Crear bot optimizado
        print("\nAnalizando condiciones de mercado...")
        time.sleep(1)
        
        print("\nEjecutando backtesting de estrategias...")
        time.sleep(2)
        
        print("\nOptimizando parámetros...")
        time.sleep(2)
        
        # Crear bot
        bot_id = learning_system.create_optimized_bot(selected_symbol, selected_interval, selected_days)
        
        if bot_id:
            # Actualizar nombre
            bot_config = learning_system._load_bot_config(bot_id)
            if bot_config:
                bot_config["name"] = bot_name
                learning_system._save_bot_config(bot_id, bot_config)
            
            # Iniciar simulación
            print("\nIniciando simulación con bot optimizado...")
            sim_result = learning_system.start_simulation_bot(bot_id, days_to_simulate=30)
            
            # Mostrar resultados
            clear_screen()
            print_header("BOT OPTIMIZADO CREADO")
            
            print(f"\nBot '{bot_name}' creado exitosamente!")
            print(f"ID: {bot_id}")
            print(f"Par: {selected_symbol}")
            print(f"Intervalo: {selected_interval}")
            
            if "error" not in sim_result:
                print(f"\nResultados de simulación inicial (30 días):")
                print(f"Estrategia: {sim_result.get('strategy', 'N/A')}")
                print(f"Balance inicial: {sim_result.get('initial_balance', 0):.2f} USDT")
                print(f"Balance final: {sim_result.get('final_balance', 0):.2f} USDT")
                print(f"Retorno: {sim_result.get('return_pct', 0):.2f}%")
                print(f"Operaciones: {sim_result.get('trades', 0)}")
                print(f"Win Rate: {sim_result.get('win_rate', 0):.2f}%")
            else:
                print(f"\nBot creado pero la simulación inicial falló: {sim_result.get('error', 'Error desconocido')}")
        else:
            print("\n❌ Error al crear bot optimizado.")
    
    except ImportError as e:
        print(f"\n❌ Error: Funcionalidad no disponible. {e}")
    except Exception as e:
        print(f"\n❌ Error al crear bot optimizado: {e}")
    
    input("\nPresiona Enter para continuar...")

def create_ml_bot():
    """Crea un bot basado en Machine Learning"""
    clear_screen()
    print_header("CREAR BOT CON MACHINE LEARNING")
    
    # Seleccionar símbolo
    symbols = get_available_symbols()
    
    print("\nSelecciona el par de trading:")
    for i, symbol in enumerate(symbols, 1):
        print(f"{i}. {symbol}")
    
    symbol_choice = get_user_choice(1, len(symbols))
    if not symbol_choice:
        return
    
    selected_symbol = symbols[symbol_choice - 1]
    
    # Seleccionar intervalo
    print("\nSelecciona el intervalo de tiempo:")
    intervals = ["15m", "1h", "4h", "1d"]
    
    for i, interval in enumerate(intervals, 1):
        print(f"{i}. {interval}")
    
    interval_choice = get_user_choice(1, len(intervals))
    if not interval_choice:
        return
    
    selected_interval = intervals[interval_choice - 1]
    
    # Seleccionar tipo de modelo
    print("\nSelecciona el tipo de modelo de ML:")
    models = [
        "Random Forest (más rápido, buena precisión)",
        "Gradient Boosting (balance velocidad/precisión)",
        "MLP Neural Network (más lento, alta capacidad)"
    ]
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    model_choice = get_user_choice(1, len(models))
    if not model_choice:
        return
    
    model_types = ["random_forest", "gradient_boosting", "mlp"]
    selected_model = model_types[model_choice - 1]
    
    # Solicitar nombre del bot
    print("\nIntroduce un nombre para el bot:")
    bot_name = input("> ").strip()
    if not bot_name:
        print("El nombre no puede estar vacío.")
        time.sleep(2)
        return
    
    # Ejecutar creación de bot ML
    print(f"\n⚙️ Creando bot ML con modelo {selected_model} para {selected_symbol} en {selected_interval}...")
    print("Este proceso incluye entrenamiento de modelo, que puede tardar varios minutos.")
    
    try:
        # Importar solo cuando se necesita para evitar errores de inicio
        from backtesting.advanced_optimizer import AutomatedLearningSystem
        
        # Crear sistema de aprendizaje
        learning_system = AutomatedLearningSystem()
        
        # Crear bot ML
        print("\nObteniendo datos históricos...")
        time.sleep(1)
        
        print("\nPreparando características...")
        time.sleep(1)
        
        print("\nEntrenando modelo de ML...")
        time.sleep(3)
        
        # Crear bot
        bot_id = learning_system.create_ml_bot(selected_symbol, selected_interval, selected_model)
        
        if bot_id:
            # Actualizar nombre
            bot_config = learning_system._load_bot_config(bot_id)
            if bot_config:
                bot_config["name"] = bot_name
                learning_system._save_bot_config(bot_id, bot_config)
            
            # Iniciar simulación
            print("\nIniciando simulación con bot ML...")
            sim_result = learning_system.start_simulation_bot(bot_id, days_to_simulate=30)
            
            # Mostrar resultados
            clear_screen()
            print_header("BOT ML CREADO")
            
            print(f"\nBot '{bot_name}' creado exitosamente!")
            print(f"ID: {bot_id}")
            print(f"Par: {selected_symbol}")
            print(f"Intervalo: {selected_interval}")
            print(f"Modelo: {selected_model}")
            
            if "error" not in sim_result:
                print(f"\nResultados de simulación inicial (30 días):")
                print(f"Balance inicial: {sim_result.get('initial_balance', 0):.2f} USDT")
                print(f"Balance final: {sim_result.get('final_balance', 0):.2f} USDT")
                print(f"Retorno: {sim_result.get('return_pct', 0):.2f}%")
                print(f"Operaciones: {sim_result.get('trades', 0)}")
                print(f"Win Rate: {sim_result.get('win_rate', 0):.2f}%")
            else:
                print(f"\nBot creado pero la simulación inicial falló: {sim_result.get('error', 'Error desconocido')}")
        else:
            print("\n❌ Error al crear bot ML.")
    
    except ImportError as e:
        print(f"\n❌ Error: Funcionalidad no disponible. {e}")
    except Exception as e:
        print(f"\n❌ Error al crear bot ML: {e}")
    
    input("\nPresiona Enter para continuar...")

def run_new_backtest():
    """Ejecuta un nuevo backtest"""
    clear_screen()
    print_header("NUEVO BACKTEST")
    
    # Seleccionar símbolo
    symbols = get_available_symbols()
    
    print("\nSelecciona el par de trading:")
    for i, symbol in enumerate(symbols, 1):
        print(f"{i}. {symbol}")
    
    symbol_choice = get_user_choice(1, len(symbols))
    if not symbol_choice:
        return
    
    selected_symbol = symbols[symbol_choice - 1]
    
    # Seleccionar estrategia
    print("\nSelecciona la estrategia de trading:")
    strategies = [
        "Cruce de Medias Móviles",
        "RSI + Bollinger Bands",
        "MACD + Tendencia",
        "Estadística (Mean Reversion)",
        "Adaptativa (Múltiples indicadores)"
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. {strategy}")
    
    strategy_choice = get_user_choice(1, len(strategies))
    if not strategy_choice:
        return
    
    selected_strategy = strategies[strategy_choice - 1]
    
    # Seleccionar periodo
    print("\nSelecciona el periodo de backtesting:")
    periods = [
        "Último mes",
        "Últimos 3 meses",
        "Últimos 6 meses",
        "Último año",
        "Personalizado"
    ]
    
    for i, period in enumerate(periods, 1):
        print(f"{i}. {period}")
    
    period_choice = get_user_choice(1, len(periods))
    if not period_choice:
        return
    
    # Calcular fechas
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    
    if period_choice == 1:
        start_date = end_date - timedelta(days=30)
    elif period_choice == 2:
        start_date = end_date - timedelta(days=90)
    elif period_choice == 3:
        start_date = end_date - timedelta(days=180)
    elif period_choice == 4:
        start_date = end_date - timedelta(days=365)
    elif period_choice == 5:
        print("\nIntroduce la fecha de inicio (YYYY-MM-DD):")
        start_date_str = input("> ").strip()
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        except ValueError:
            print("Formato de fecha incorrecto. Usando último mes por defecto.")
            start_date = end_date - timedelta(days=30)
    
    # Ejecutar backtest
    try:
        print("\nEjecutando backtest...")
        
        strategy_name_to_func = {
            "Cruce de Medias Móviles": "moving_average_crossover",
            "RSI + Bollinger Bands": "rsi_bollinger",
            "MACD + Tendencia": "macd_trend",
            "Estadística (Mean Reversion)": "mean_reversion",
            "Adaptativa (Múltiples indicadores)": "adaptive"
        }
        
        strategy_func = strategy_name_to_func.get(selected_strategy)
        
        results = run_backtest(
            symbol=selected_symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            strategy=strategy_func
        )
        
        clear_screen()
        print_header("RESULTADOS DEL BACKTEST")
        
        print(f"\nPar: {selected_symbol}")
        print(f"Estrategia: {selected_strategy}")
        print(f"Periodo: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
        
        print("\nResultados:")
        print(f"Balance inicial: {results['initial_balance']:.2f} USDT")
        print(f"Balance final: {results['final_balance']:.2f} USDT")
        print(f"Retorno: {results['return_pct']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {results['sortino_ratio']:.2f}")
        print(f"Drawdown máximo: {results['max_drawdown_pct']:.2f}%")
        print(f"Trades totales: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        
        # Mostrar gráfico de equity curve
        print("\nCurva de Equity:")
        print_chart(results['equity_curve'], value_key='equity', title="Equity Curve")
        
        # Guardar resultados
        backtest_id = save_backtest_results(results)
        print(f"\n✅ Resultados guardados con ID: {backtest_id}")
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        print(f"\n❌ Error al ejecutar backtest: {e}")
    
    input("\nPresiona Enter para continuar...")

def view_backtest_results():
    """Ver resultados de backtests anteriores"""
    clear_screen()
    print_header("RESULTADOS DE BACKTESTS")
    
    try:
        from db.operations import get_backtest_results
        
        # Obtener resultados
        results = get_backtest_results()
        
        if not results:
            print("\nNo hay resultados de backtests guardados.")
            input("\nPresiona Enter para continuar...")
            return
        
        print("\nSelecciona un backtest para ver detalles:")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['symbol']} - {result['strategy']} - {result['date']} - {result['return_pct']:.2f}%")
        
        print(f"{len(results) + 1}. Volver")
        
        choice = get_user_choice(1, len(results) + 1)
        if choice == len(results) + 1:
            return
        
        if 1 <= choice <= len(results):
            selected_result = results[choice - 1]
            
            clear_screen()
            print_header(f"DETALLES DEL BACKTEST: {selected_result['id']}")
            
            print(f"\nPar: {selected_result['symbol']}")
            print(f"Estrategia: {selected_result['strategy']}")
            print(f"Periodo: {selected_result['start_date']} a {selected_result['end_date']}")
            
            print("\nResultados:")
            print(f"Balance inicial: {selected_result['initial_balance']:.2f} USDT")
            print(f"Balance final: {selected_result['final_balance']:.2f} USDT")
            print(f"Retorno: {selected_result['return_pct']:.2f}%")
            print(f"Sharpe Ratio: {selected_result['sharpe_ratio']:.2f}")
            print(f"Sortino Ratio: {selected_result['sortino_ratio']:.2f}")
            print(f"Drawdown máximo: {selected_result['max_drawdown_pct']:.2f}%")
            print(f"Trades totales: {selected_result['total_trades']}")
            print(f"Win Rate: {selected_result['win_rate']:.2f}%")
            print(f"Profit Factor: {selected_result['profit_factor']:.2f}")
            
            # Mostrar gráfico si hay datos
            if 'equity_curve' in selected_result:
                print("\nCurva de Equity:")
                print_chart(selected_result['equity_curve'], value_key='equity', title="Equity Curve")
            
            print("\nOpciones:")
            print("1. Crear bot con esta configuración")
            print("2. Volver")
            
            subchoice = get_user_choice(1, 2)
            if subchoice == 1:
                # Crear bot basado en backtest
                create_bot_from_backtest(selected_result)
        
    except Exception as e:
        logger.error(f"Error viewing backtest results: {e}")
        print(f"Error al ver resultados de backtests: {e}")
    
    input("\nPresiona Enter para continuar...")

def create_bot_from_backtest(backtest_result):
    """Crea un bot basado en un backtest"""
    clear_screen()
    print_header("CREAR BOT DESDE BACKTEST")
    
    # Solicitar nombre del bot
    print("\nIntroduce un nombre para el bot:")
    bot_name = input("> ").strip()
    if not bot_name:
        print("El nombre no puede estar vacío.")
        time.sleep(2)
        return
    
    # Solicitar modo (paper/live)
    print("\nSelecciona el modo de trading:")
    print("1. Paper Trading (Simulado)")
    print("2. Live Trading (Real)")
    
    mode_choice = get_user_choice(1, 2)
    if not mode_choice:
        return
    
    paper_trading = mode_choice == 1
    
    # Configurar balance inicial (para paper trading)
    initial_balance = 1000.0
    if paper_trading:
        print("\nIntroduce el balance inicial (USDT):")
        try:
            initial_balance = float(input("> ").strip())
        except ValueError:
            print("Valor no válido. Usando 1000 USDT por defecto.")
            initial_balance = 1000.0
    
    # Crear configuración del bot
    bot_config = {
        "name": bot_name,
        "symbol": backtest_result['symbol'],
        "paper_trading": paper_trading,
        "strategy": backtest_result['strategy'],
        "balance": initial_balance,
        "active": False,
        "created_at": datetime.now().isoformat(),
        "config": backtest_result.get('config', {
            "risk_per_trade": 1.0,
            "leverage": 1.0,
            "stop_loss": 2.0,
            "take_profit": 3.0,
            "time_interval": "15m"
        })
    }
    
    # Guardar bot en la base de datos
    try:
        bot_manager.create_bot(bot_config)
        print(f"\n✅ Bot '{bot_name}' creado exitosamente!")
    except Exception as e:
        logger.error(f"Error creating bot: {e}")
        print(f"\n❌ Error al crear el bot: {e}")

def optimize_parameters():
    """Optimiza parámetros de una estrategia"""
    clear_screen()
    print_header("OPTIMIZACIÓN DE PARÁMETROS")
    
    # Seleccionar símbolo
    symbols = get_available_symbols()
    
    print("\nSelecciona el par de trading:")
    for i, symbol in enumerate(symbols, 1):
        print(f"{i}. {symbol}")
    
    symbol_choice = get_user_choice(1, len(symbols))
    if not symbol_choice:
        return
    
    selected_symbol = symbols[symbol_choice - 1]
    
    # Seleccionar estrategia
    print("\nSelecciona la estrategia de trading:")
    strategies = [
        "Cruce de Medias Móviles",
        "RSI + Bollinger Bands",
        "MACD + Tendencia"
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. {strategy}")
    
    strategy_choice = get_user_choice(1, len(strategies))
    if not strategy_choice:
        return
    
    selected_strategy = strategies[strategy_choice - 1]
    
    # Definir parámetros a optimizar según estrategia
    params_to_optimize = {}
    
    if selected_strategy == "Cruce de Medias Móviles":
        params_to_optimize = {
            "fast_period": [5, 10, 15, 20, 25],
            "slow_period": [20, 30, 40, 50, 60]
        }
    elif selected_strategy == "RSI + Bollinger Bands":
        params_to_optimize = {
            "rsi_period": [7, 14, 21],
            "rsi_overbought": [70, 75, 80],
            "rsi_oversold": [20, 25, 30],
            "bb_period": [15, 20, 25],
            "bb_dev": [1.8, 2.0, 2.2]
        }
    elif selected_strategy == "MACD + Tendencia":
        params_to_optimize = {
            "fast_ema": [8, 12, 16],
            "slow_ema": [21, 26, 30],
            "signal_period": [7, 9, 12]
        }
    
    # Período de optimización
    print("\nSelecciona el periodo de optimización:")
    periods = [
        "Último mes",
        "Últimos 3 meses"
    ]
    
    for i, period in enumerate(periods, 1):
        print(f"{i}. {period}")
    
    period_choice = get_user_choice(1, len(periods))
    if not period_choice:
        return
    
    # Calcular fechas
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    
    if period_choice == 1:
        start_date = end_date - timedelta(days=30)
    elif period_choice == 2:
        start_date = end_date - timedelta(days=90)
    
    # Ejecutar optimización
    try:
        print("\n⚙️ Ejecutando optimización de parámetros...")
        print("Este proceso puede tardar varios minutos.")
        print("Se probarán todas las combinaciones de parámetros.")
        
        # Mapeo de nombres de estrategia a funciones
        strategy_name_to_func = {
            "Cruce de Medias Móviles": "moving_average_crossover",
            "RSI + Bollinger Bands": "rsi_bollinger",
            "MACD + Tendencia": "macd_trend"
        }
        
        strategy_func = strategy_name_to_func.get(selected_strategy)
        
        from backtesting.optimization import optimize_strategy
        
        results = optimize_strategy(
            symbol=selected_symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            strategy=strategy_func,
            params=params_to_optimize
        )
        
        clear_screen()
        print_header("RESULTADOS DE OPTIMIZACIÓN")
        
        print(f"\nPar: {selected_symbol}")
        print(f"Estrategia: {selected_strategy}")
        print(f"Periodo: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
        
        print("\nMejores parámetros:")
        for param, value in results['best_params'].items():
            print(f"{param}: {value}")
        
        print("\nResultados con mejores parámetros:")
        print(f"Retorno: {results['best_result']['return_pct']:.2f}%")
        print(f"Sharpe Ratio: {results['best_result']['sharpe_ratio']:.2f}")
        print(f"Trades totales: {results['best_result']['total_trades']}")
        print(f"Win Rate: {results['best_result']['win_rate']:.2f}%")
        
        # Mostrar comparativa de los mejores resultados
        print("\nTop 5 combinaciones de parámetros:")
        top_results = results['all_results'][:5]
        
        for i, result in enumerate(top_results, 1):
            print(f"\n{i}. Retorno: {result['return_pct']:.2f}%, Sharpe: {result['sharpe_ratio']:.2f}")
            for param, value in result['params'].items():
                print(f"   {param}: {value}")
        
        print("\nOpciones:")
        print("1. Crear bot con los mejores parámetros")
        print("2. Volver")
        
        choice = get_user_choice(1, 2)
        if choice == 1:
            # Crear bot con los mejores parámetros
            create_bot_with_optimized_params(selected_symbol, selected_strategy, results['best_params'])
        
    except Exception as e:
        logger.error(f"Error optimizing parameters: {e}")
        print(f"\n❌ Error en la optimización: {e}")
    
    input("\nPresiona Enter para continuar...")

def create_bot_with_optimized_params(symbol, strategy, params):
    """Crea un bot con parámetros optimizados"""
    clear_screen()
    print_header("CREAR BOT CON PARÁMETROS OPTIMIZADOS")
    
    # Solicitar nombre del bot
    print("\nIntroduce un nombre para el bot:")
    bot_name = input("> ").strip()
    if not bot_name:
        print("El nombre no puede estar vacío.")
        time.sleep(2)
        return
    
    # Solicitar modo (paper/live)
    print("\nSelecciona el modo de trading:")
    print("1. Paper Trading (Simulado)")
    print("2. Live Trading (Real)")
    
    mode_choice = get_user_choice(1, 2)
    if not mode_choice:
        return
    
    paper_trading = mode_choice == 1
    
    # Configurar balance inicial (para paper trading)
    initial_balance = 1000.0
    if paper_trading:
        print("\nIntroduce el balance inicial (USDT):")
        try:
            initial_balance = float(input("> ").strip())
        except ValueError:
            print("Valor no válido. Usando 1000 USDT por defecto.")
            initial_balance = 1000.0
    
    # Crear configuración del bot
    config = {
        "risk_per_trade": 1.0,
        "leverage": 1.0,
        "stop_loss": 2.0,
        "take_profit": 3.0,
        "time_interval": "15m"
    }
    
    # Añadir parámetros optimizados
    for param, value in params.items():
        config[param] = value
    
    bot_config = {
        "name": bot_name,
        "symbol": symbol,
        "paper_trading": paper_trading,
        "strategy": strategy,
        "balance": initial_balance,
        "active": False,
        "created_at": datetime.now().isoformat(),
        "config": config
    }
    
    # Guardar bot en la base de datos
    try:
        bot_manager.create_bot(bot_config)
        print(f"\n✅ Bot '{bot_name}' creado exitosamente!")
    except Exception as e:
        logger.error(f"Error creating bot: {e}")
        print(f"\n❌ Error al crear el bot: {e}")

def dashboard_menu():
    """Menú de panel de control"""
    while True:
        clear_screen()
        print_header("PANEL DE CONTROL")
        
        # Mostrar resumen
        print("\n📊 RESUMEN GLOBAL:")
        
        try:
            # Obtener datos de resumen
            bots = get_bots()
            active_bots = sum(1 for bot in bots if bot.get('active', False))
            total_balance = sum(bot.get('balance', 0) for bot in bots)
            total_pnl = sum(bot.get('pnl', 0) for bot in bots)
            
            # Mostrar datos
            print(f"Bots totales: {len(bots)}")
            print(f"Bots activos: {active_bots}")
            print(f"Balance total: {total_balance:.2f} USDT")
            print(f"P&L total: {total_pnl:.2f} USDT")
            
            # Mostrar bots activos
            if active_bots > 0:
                print("\n🤖 BOTS ACTIVOS:")
                active_bot_data = []
                for bot in bots:
                    if bot.get('active', False):
                        active_bot_data.append([
                            bot.get('name', 'Bot sin nombre'),
                            bot.get('symbol', 'SOL-USDT'),
                            f"{bot.get('balance', 0):.2f} USDT",
                            f"{bot.get('pnl', 0):.2f} USDT",
                            f"{bot.get('roi', 0):.2f}%"
                        ])
                
                headers = ["Nombre", "Par", "Balance", "P&L", "ROI"]
                print_table(headers, active_bot_data)
            
            # Mostrar operaciones recientes
            print("\n📜 OPERACIONES RECIENTES:")
            
            # Obtener operaciones recientes
            from db.operations import get_recent_trades
            
            recent_trades = get_recent_trades(limit=10)
            
            if recent_trades:
                trade_data = []
                for trade in recent_trades:
                    trade_data.append([
                        trade.get('bot_name', 'N/A'),
                        trade.get('timestamp', 'N/A'),
                        trade.get('symbol', 'N/A'),
                        "COMPRA" if trade.get('type') == 'buy' else "VENTA",
                        f"{trade.get('price', 0):.2f}",
                        f"{trade.get('size', 0):.4f}",
                        f"{trade.get('pnl', 0):.2f} USDT"
                    ])
                
                headers = ["Bot", "Fecha", "Par", "Tipo", "Precio", "Tamaño", "P&L"]
                print_table(headers, trade_data)
            else:
                print("No hay operaciones recientes")
            
            # Obtener precio actual de SOL
            from data_management.market_data import get_current_price
            
            sol_price = get_current_price("SOL-USDT")
            print(f"\nPrecio actual de SOL: {sol_price:.2f} USDT")
            
        except Exception as e:
            logger.error(f"Error displaying dashboard: {e}")
            print(f"Error al mostrar panel de control: {e}")
        
        print("\nOpciones:")
        print("1. Actualizar datos")
        print("2. Ver estadísticas detalladas")
        print("3. Volver al menú principal")
        
        choice = get_user_choice(1, 3)
        
        if choice == 1:
            continue  # Simplemente refrescar
        elif choice == 2:
            view_detailed_stats()
        elif choice == 3:
            break
        else:
            print("Opción no válida. Inténtalo de nuevo.")
            time.sleep(1)

def view_detailed_stats():
    """Muestra estadísticas detalladas"""
    clear_screen()
    print_header("ESTADÍSTICAS DETALLADAS")
    
    try:
        # Obtener estadísticas
        from db.operations import get_performance_stats
        
        stats = get_performance_stats()
        
        print("\n📈 RENDIMIENTO POR ESTRATEGIA:")
        strategy_stats = stats.get('strategy_stats', [])
        
        if strategy_stats:
            strategy_data = []
            for stat in strategy_stats:
                strategy_data.append([
                    stat.get('strategy', 'N/A'),
                    str(stat.get('trade_count', 0)),
                    f"{stat.get('win_rate', 0):.2f}%",
                    f"{stat.get('avg_profit', 0):.2f} USDT",
                    f"{stat.get('avg_loss', 0):.2f} USDT",
                    f"{stat.get('profit_factor', 0):.2f}",
                    f"{stat.get('avg_duration', 'N/A')}"
                ])
            
            headers = ["Estrategia", "Trades", "Win Rate", "Profit Avg", "Loss Avg", "Profit Factor", "Duración Avg"]
            print_table(headers, strategy_data)
        else:
            print("No hay estadísticas disponibles")
        
        print("\n📊 RENDIMIENTO POR SÍMBOLO:")
        symbol_stats = stats.get('symbol_stats', [])
        
        if symbol_stats:
            symbol_data = []
            for stat in symbol_stats:
                symbol_data.append([
                    stat.get('symbol', 'N/A'),
                    str(stat.get('trade_count', 0)),
                    f"{stat.get('win_rate', 0):.2f}%",
                    f"{stat.get('avg_profit', 0):.2f} USDT",
                    f"{stat.get('avg_loss', 0):.2f} USDT",
                    f"{stat.get('profit_factor', 0):.2f}"
                ])
            
            headers = ["Símbolo", "Trades", "Win Rate", "Profit Avg", "Loss Avg", "Profit Factor"]
            print_table(headers, symbol_data)
        else:
            print("No hay estadísticas disponibles")
        
        print("\n📆 RENDIMIENTO POR MES:")
        monthly_stats = stats.get('monthly_stats', [])
        
        if monthly_stats:
            monthly_data = []
            for stat in monthly_stats:
                monthly_data.append([
                    stat.get('month', 'N/A'),
                    str(stat.get('trade_count', 0)),
                    f"{stat.get('win_rate', 0):.2f}%",
                    f"{stat.get('total_profit', 0):.2f} USDT",
                    f"{stat.get('total_loss', 0):.2f} USDT",
                    f"{stat.get('net_pnl', 0):.2f} USDT"
                ])
            
            headers = ["Mes", "Trades", "Win Rate", "Profit Total", "Loss Total", "P&L Neto"]
            print_table(headers, monthly_data)
        else:
            print("No hay estadísticas disponibles")
        
    except Exception as e:
        logger.error(f"Error displaying detailed stats: {e}")
        print(f"Error al mostrar estadísticas detalladas: {e}")
    
    input("\nPresiona Enter para continuar...")

def save_backtest_results(results):
    """Guarda resultados de backtest en la base de datos"""
    try:
        from db.operations import save_backtest
        
        return save_backtest(results)
    except Exception as e:
        logger.error(f"Error saving backtest results: {e}")
        return None

def confirm_exit():
    """Confirma salida del programa"""
    if confirm_action("¿Estás seguro de que deseas salir? Todos los bots activos seguirán ejecutándose."):
        global running
        running = False

def signal_handler(sig, frame):
    """Manejador de señales para salida segura"""
    global running
    print("\nSeñal de interrupción recibida. Saliendo...")
    running = False

def main():
    """Función principal"""
    # Registrar manejador de señales
    signal.signal(signal.SIGINT, signal_handler)
    
    # Inicializar el gestor de bots
    if not init_bot_manager():
        print("Error al inicializar el gestor de bots. Saliendo.")
        return
    
    # Bucle principal
    while running:
        try:
            main_menu()
        except Exception as e:
            logger.error(f"Error en menú principal: {e}")
            print(f"Error: {e}")
            print("Reiniciando menú...")
            time.sleep(2)
    
    # Mensaje de salida
    print("Gracias por usar el Solana Trading Bot. ¡Hasta pronto!")

if __name__ == "__main__":
    main()