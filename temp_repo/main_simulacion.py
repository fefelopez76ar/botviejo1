import asyncio
import logging
import threading
import time
from collections import deque
import sys
import os
import random

# Asegurarse de que el directorio del proyecto esté en el PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Asume que el proyecto está un nivel arriba
sys.path.insert(0, project_root)

# Importar los módulos que quieres probar en simulación
from core.signal_engine import SignalEngine
from core.trade_executor import TradeExecutor
from data_management.historical_data_saver_async import HistoricalDataSaver # Importa el guardador de datos

# --- Configuración del Logger ---
def setup_logging():
    log_dir = os.path.join(current_dir, 'info') # Puedes ajustar esta ruta si 'info' está en la raíz
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'bot_simulacion.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("SignalEngine").setLevel(logging.INFO)
    logging.getLogger("TradeExecutor").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger("MainBotSimulacion")

running = True # Bandera global para controlar el bucle

# --- Función para simular y procesar datos ---
# Ahora recibe la instancia del guardador de datos
async def simulate_data(signal_engine_processor, historical_data_saver_instance):
    logger.info("Iniciando simulación de datos para SOL-USDT...")
    while running:
        # Simular datos de ticker
        price = round(random.uniform(170.0, 180.0), 2)
        timestamp_ms = int(time.time() * 1000) # Timestamp en milisegundos

        # Formato corregido para datos de ticker (mimic OKX public ticker data)
        # 'last' es el campo que SignalEngine debería esperar como precio de cierre/último precio.
        ticker_data_for_processing = {
            "arg": {"channel": "tickers", "instId": "SOL-USDT"},
            "data": [{
                "instId": "SOL-USDT",
                "last": str(price), # Usar 'last' para el precio
                "ts": str(timestamp_ms),
                "askPx": str(round(price + 0.01, 2)),
                "bidPx": str(round(price - 0.01, 2)),
                "high24h": str(round(price + 5, 2)),
                "low24h": str(round(price - 5, 2)),
                "volCcy24h": str(round(random.uniform(100000, 500000), 2)),
            }]
        }

        # Simular datos de order book
        # Asegúrate de que bids y asks sean listas de listas de strings, como espera SQLite y JSON
        bids = [[str(round(price - 0.05 - random.uniform(0, 0.02), 2)), str(round(10 + random.uniform(0, 5), 3))]]
        asks = [[str(round(price + 0.05 + random.uniform(0, 0.02), 2)), str(round(10 + random.uniform(0, 5), 3))]]

        # Mimic OKX order book data structure
        order_book_data_for_processing = {
            "arg": {"channel": "books-l2-tbt", "instId": "SOL-USDT"},
            "data": [{
                "asks": asks,
                "bids": bids,
                "instId": "SOL-USDT",
                "ts": str(timestamp_ms),
                "seqNum": str(random.randint(1000000, 9999999)) # Número de secuencia para order book
            }]
        }

        # Procesar datos de ticker con SignalEngine
        await signal_engine_processor(ticker_data_for_processing)

        # Guardar ambos tipos de datos usando HistoricalDataSaver
        # HistoricalDataSaver.save_data debería manejar diferentes tipos de datos
        # basándose en el canal en 'arg'.
        await historical_data_saver_instance.save_data(ticker_data_for_processing)
        await historical_data_saver_instance.save_data(order_book_data_for_processing)

        await asyncio.sleep(0.5) # Simular datos cada 0.5 segundos

# --- Punto de entrada principal ---
async def main():
    global running
    logger.info("==================================================")
    logger.info("SolanaScalper - Bot de Trading v2.0 (MODO SIMULACIÓN)")
    logger.info("==================================================")
    logger.info("Iniciando componentes de simulación...")

    trade_executor = TradeExecutor(initial_balance=10000.0, risk_per_trade_pct=0.001)
    signal_engine = SignalEngine(
        instrument_id="SOL-USDT",
        data_buffer_size=50,
        on_signal_generated=trade_executor.execute_order
    )

    # Inicializar HistoricalDataSaver
    # Asumimos que db_path="info/market_data.db" se maneja correctamente dentro de HistoricalDataSaver
    historical_data_saver = HistoricalDataSaver() # No es necesario pasar db_path si el default es info/

    # Iniciar la simulación de datos en segundo plano
    simulation_task = asyncio.create_task(
        simulate_data(signal_engine.process_data, historical_data_saver) # Pasa ambas instancias
    )

    # Manejador de comandos CLI (simplificado para modo simulación)
    def cli_input_listener():
        global running
        while True:
            try:
                cmd = input("> ").strip().lower()
                if cmd == 'q':
                    logger.info("Interrupción por teclado detectada (q). Apagando bot...")
                    running = False
                    break
                else:
                    logger.warning(f"Comando desconocido en simulación: {cmd}")
            except EOFError:
                logger.info("EOF detectado, apagando bot...")
                running = False
                break
            except Exception as e:
                logger.error(f"Error en manejador de comandos CLI: {e}")
                running = False # Salir ante un error inesperado

    # Iniciar la escucha de entrada CLI en un hilo separado (es bloqueante)
    cli_thread = threading.Thread(target=cli_input_listener, daemon=True)
    cli_thread.start()

    # Bucle principal del bot en simulación
    while running:
        await asyncio.sleep(1)

    # Cierre limpio
    if simulation_task:
        simulation_task.cancel()
        try:
            await simulation_task
        except asyncio.CancelledError:
            logger.info("Simulación de datos cancelada.")

    # Asegurarse de que el guardador de datos se cierre limpiamente
    await historical_data_saver.close_connection() # Asumiendo que HistoricalDataSaver tiene un método close_connection

    logger.info("Bot de trading en modo simulación detenido.")
    logger.info("==================================================")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupción por teclado (Ctrl+C) detectada. Apagando bot...")
    except Exception as e:
        logger.error(f"Error inesperado en el main: {e}")