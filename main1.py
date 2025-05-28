#!/usr/bin/env python3
"""
SolanaScalper - Bot de Trading para Solana
Versión CLI optimizada para operaciones de scalping

Uso: python main1.py (Esta es la versión MODIFICADA para pruebas)
"""

import os
import sys
import time
import logging
import threading
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from tabulate import tabulate
import signal

# --- NUEVAS IMPORTACIONES PARA EL CLIENTE WEBSOCKET ---
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from api_client.modulocola import data_queue
from api_client.modulo2 import OKXWebSocketClient
# ---------------------------------------------------
from data_management.historical_data_saver import HistoricalDataSaver
# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_bot.log") # Este log es para el bot general
    ]
)
logger = logging.getLogger("SolanaScalper")

# --- Cargar variables de entorno (al inicio, para que el WebSocket las tenga) ---
config_path = Path(__file__).parent / "config.env"
load_dotenv(dotenv_path=config_path)
# ---------------------------------------------------------------------------------

class ScalpingBot:
    """Bot de scalping para trading de Solana en OKX"""

    def __init__(self, historical_data_saver: HistoricalDataSaver): # MODIFICADO: Ahora recibe el saver
        self.active = False
        self.exchange = None
        self.symbol = "SOL/USDT"
        self.timeframe = "1m"  # Timeframe para scalping
        self.balance = 0.0
        self.current_position = None
        self.last_trade_price = 0.0
        self.total_trades = 0
        self.profitable_trades = 0
        self.mode = "paper"  # paper o real
        self.trades_history = []
        self.price_history = []
        self.stop_event = threading.Event()
        self.status_thread = None
        self.trading_thread = None
        # --- NUEVO: Atributo para la cola de datos en el bot ---
        self.data_queue = data_queue # Referencia a la cola global de modulocola.py
        # --- NUEVO: Atributo para HistoricalDataSaver ---
        self.historical_data_saver = historical_data_saver
        # ------------------------------------------------------

    def initialize(self):
        logger.info(f"Inicializando bot en modo {self.mode.upper()}...")
        # Llama al método de inicialización del HistoricalDataSaver
        self.historical_data_saver.connect()
        logger.info("Bot inicializado.")

    def start(self):
        if not self.active:
            logger.info("Iniciando operaciones del bot...")
            self.active = True
            self.stop_event.clear()
            logger.info("ScalpingBot se ejecutará en el bucle principal asíncrono.")
        else:
            logger.warning("El bot ya está activo.")

    def stop(self):
        if self.active:
            logger.info("Deteniendo operaciones del bot...")
            self.active = False
            self.stop_event.set()
            logger.info("Operaciones del bot detenidas.")
        else:
            logger.warning("El bot no está activo.")

    def shutdown(self):
        logger.info("Apagando el bot...")
        self.stop()
        # Cierra la conexión de la base de datos al apagar
        self.historical_data_saver.disconnect()
        logger.info("Bot apagado y recursos liberados.")

    async def run_data_consumer(self):
        logger.info("[ScalpingBot]: Iniciando consumidor de datos de la cola...")
        while not self.stop_event.is_set():
            try:
                data = await asyncio.wait_for(self.data_queue.get(), timeout=1.0)
                # logger.debug(f"[ScalpingBot]: Datos recibidos de la cola: {data.get('type', 'N/A')}")

                if data.get('type') == 'ticker':
                    inst_id = data.get('instrument')
                    price_data = data.get('data')
                    if price_data:
                        last_price = price_data[0].get('last')
                        logger.info(f"[ScalpingBot - Ticker]: {inst_id} - Último precio: {last_price}")
                        # NUEVO: Guardar datos de ticker
                        self.historical_data_saver.save_ticker_data(data) # <--- ¡Aquí está la adición!
                elif data.get('type') == 'books-l2-tbt':
                    logger.info(f"[ScalpingBot - OrderBook]: {data.get('instrument')} - Bid: {data.get('best_bid')}, Ask: {data.get('best_ask')}")
                    # NUEVO: Guardar datos de order book
                    self.historical_data_saver.save_order_book_data(data) # <--- ¡Aquí está la adición!
                elif data.get('type') == 'trades':
                    logger.info(f"[ScalpingBot - Trade]: {data.get('instrument')} - Trades recibidos.")

                # Asegurarse de que task_done() se llama para cada elemento 'get'
                self.data_queue.task_done()
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                logger.info("[ScalpingBot]: Consumidor de cola cancelado.")
                break
            except Exception as e:
                logger.error(f"[ScalpingBot ERROR]: Error al consumir de la cola: {e}")
                if not self.data_queue.empty():
                    try:
                        self.data_queue.task_done()
                    except ValueError:
                        pass
                await asyncio.sleep(1)

        logger.info("[ScalpingBot]: Consumidor de datos de la cola detenido.")

    def set_mode(self, new_mode):
        if new_mode in ["paper", "real"]:
            self.mode = new_mode
            logger.info(f"Modo de operación cambiado a: {new_mode.upper()}")
        else:
            logger.warning(f"Modo '{new_mode}' no válido. Use 'paper' o 'real'.")

    def display_status(self):
        print("\n" + "="*60)
        print(f"ESTADO DEL BOT: {'ACTIVO' if self.active else 'INACTIVO'} | MODO: {self.mode.upper()}")
        print(f"Par de Trading: {self.symbol} | Timeframe: {self.timeframe}")
        print(f"Balance Actual: {self.balance:.2f} USDT")
        print(f"Posición Actual: {self.current_position if self.current_position else 'Ninguna'}")
        print(f"Total de Trades: {self.total_trades} | Rentables: {self.profitable_trades} ({self.profitable_trades/self.total_trades*100:.2f}% profitable)" if self.total_trades > 0 else "Total de Trades: 0")
        print("="*60 + "\n")

async def main_cli_interface_async():
    logger.info(f"{'='*50}\n{'SolanaScalper - Bot de Trading v2.0 (MODO ASÍNCRONO)':^50}\n{'='*50}\n")
    logger.info("Iniciando componentes asíncronos y síncronos...")

    # --- 1. Verificar variables de entorno OKX ---
    required_env_vars = ['OKX_API_KEY', 'OKX_API_SECRET', 'OKX_PASSPHRASE']
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]

    if missing_vars:
        logger.error("\n⚠️  FALTAN VARIABLES DE ENTORNO NECESARIAS PARA OKX WEBSOCKET")
        logger.error("Por favor, configura las siguientes variables en config.env:")
        for var in missing_vars:
            logger.error(f"  • {var}")
        logger.error("\nEl bot NO PUEDE iniciar sin estas credenciales. Saliendo.")
        return

    # --- 2. Inicializar el Cliente WebSocket (OKXWebSocketClient de modulo2.py) ---
    api_key = os.getenv("OKX_API_KEY")
    secret_key = os.getenv("OKX_API_SECRET")
    passphrase = os.getenv("OKX_PASSPHRASE")

    okx_client = OKXWebSocketClient(api_key, secret_key, passphrase, data_queue)

    logger.info("Conectando y suscribiendo al WebSocket de OKX...")
    await okx_client.connect()
    auth_success = await okx_client.authenticate() # Captura el resultado de la autenticación

    if not auth_success: # Si la autenticación falló, mostramos mensaje y salimos
        logger.error("La autenticación con OKX falló. Por favor, revisa tus credenciales.")
        logger.error("No se pueden iniciar las suscripciones sin autenticación exitosa.")
        # Intentar cerrar la conexión del websocket si existe
        # El atributo .closed es de websockets.client.WebSocketClientProtocol
        # La verificación es necesaria para evitar AttributeError si el objeto no se inicializó bien
        if okx_client.ws and hasattr(okx_client.ws, 'closed') and not okx_client.ws.closed:
            await okx_client.ws.close()
        return # Salir de la función principal

    # SUSCRIPCIONES INICIALES
    # MODIFICADO: ELIMINADA LA SUSCRIPCIÓN A "trades"
    await okx_client.subscribe([
        {"channel": "tickers", "instId": "SOL-USDT"},
        {"channel": "books-l2-tbt", "instId": "SOL-USDT"}
    ])
    logger.info("Suscripciones iniciales a tickers y order book enviadas.")

    # --- NUEVO: Inicializar HistoricalDataSaver ---
    historical_data_saver = HistoricalDataSaver()
    # --------------------------------------------

    # --- 3. Inicializar tu ScalpingBot (síncrono) ---
    # MODIFICADO: Pasar la instancia de historical_data_saver al ScalpingBot
    bot = ScalpingBot(historical_data_saver)
    bot.initialize()

    # --- 4. Crear un hilo para manejar la entrada del usuario de forma síncrona ---
    user_input_queue = asyncio.Queue()
    def _get_user_input_thread(loop): # Pasar el loop de asyncio al hilo
        while True:
            try:
                user_input = input("> ").lower() # <--- ¡AQUÍ ESTÁ EL PROMPT!
                asyncio.run_coroutine_threadsafe(user_input_queue.put(user_input), loop)
                if user_input == 'q':
                    break
            except EOFError:
                asyncio.run_coroutine_threadsafe(user_input_queue.put('q'), loop)
                break
            except Exception as e:
                logger.error(f"Error en hilo de input: {e}")
                asyncio.run_coroutine_threadsafe(user_input_queue.put('q'), loop)
                break

    input_thread = threading.Thread(target=_get_user_input_thread, args=(asyncio.get_event_loop(),), daemon=True) # Pasar el loop
    input_thread.start()
    logger.info("Hilo de entrada de usuario iniciado. Presiona 'q' para salir.")


    # --- 5. Ejecutar todas las tareas en paralelo con asyncio.gather ---
    tasks = [
        okx_client.receive_messages(okx_client.process_message),
        bot.run_data_consumer()
    ]

    async def cli_command_handler():
        while True:
            try:
                cmd = await user_input_queue.get()
                if cmd == 'q':
                    logger.info("Comando 'q' recibido: Iniciando cierre del bot.")
                    bot.shutdown()
                    if okx_client.ws and hasattr(okx_client.ws, 'closed') and not okx_client.ws.closed:
                        await okx_client.ws.close()
                    break
                elif cmd == 's':
                    bot.stop()
                elif cmd == 'm':
                    new_mode = "real" if bot.mode == "paper" else "paper"
                    bot.set_mode(new_mode)
                elif cmd == 'status':
                    bot.display_status()
                else:
                    logger.warning(f"Comando desconocido: {cmd}")
            except asyncio.CancelledError:
                logger.info("Manejador de comandos CLI cancelado.")
                break
            except Exception as e:
                logger.error(f"Error en manejador de comandos CLI: {e}")
            finally:
                try:
                    if not user_input_queue.empty():
                        user_input_queue.task_done()
                except ValueError as ve:
                    logger.warning(f"task_done() llamado innecesariamente: {ve}")
            await asyncio.sleep(0.1)

    tasks.append(cli_command_handler())

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Tareas principales canceladas.")
    finally:
        logger.info("Cerrando recursos del bot...")
        bot.shutdown()
        # También aquí se verifica si ws existe y tiene el atributo .closed
        if okx_client.ws and hasattr(okx_client.ws, 'closed') and not okx_client.ws.closed:
            await okx_client.ws.close()

    logger.info("Bot apagado completamente.")


# --- Punto de entrada principal ---
if __name__ == "__main__":
    try:
        asyncio.run(main_cli_interface_async())
    except KeyboardInterrupt:
        logger.info("Proceso principal interrumpido por el usuario (Ctrl+C).")
    except Exception as e:
        logger.critical(f"Un error inesperado detuvo el bot: {e}", exc_info=True)