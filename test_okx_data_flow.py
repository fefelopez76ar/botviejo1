import asyncio
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# --- Importaciones de tus módulos ---
# Asegúrate de que modulo2.py y modulocola.py estén en api_client/
# y que modulo2.py esté ajustado para encontrar config.env.
from api_client.modulocola import data_queue
from api_client.modulo2 import OKXWebSocketClient

# --- Configuración de Logging para esta prueba ---
# Crear carpeta 'info' si no existe
info_dir = Path(__file__).parent / "info"
info_dir.mkdir(exist_ok=True)

log_file_path = info_dir / "test_okx_data_flow.log" # Log separado para la prueba

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG) # Nivel general para todos los handlers

# Limpiar handlers existentes (importante si el script se ejecuta varias veces en un entorno de desarrollo)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Handler para el archivo de logs (todo lo que sea DEBUG o superior)
file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG) # Guarda todo en el archivo
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
root_logger.addHandler(file_handler)

# Handler para la consola (solo INFO o superior para que no desborde)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # Muestra INFO y WARNING/ERROR en consola
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
root_logger.addHandler(console_handler)

# --- Cargar variables de entorno (para este script de prueba) ---
config_path = Path(__file__).parent / "config.env"
load_dotenv(dotenv_path=config_path)

# --- Función de prueba para consumir datos de la cola ---
async def test_queue_consumer():
    logging.info("\n[TEST CONSUMIDOR]: Activo. Esperando datos de la cola...")
    while True:
        try:
            data = await data_queue.get() # Espera a que haya un elemento en la cola
            logging.info(f"[TEST CONSUMIDOR]: Datos recibidos: Tipo={data.get('type', 'N/A')}, Instrumento={data.get('instrument', 'N/A')}")
            # Imprime más detalles solo si son relevantes y para depuración
            if data.get('type') == 'ticker' and 'last_price' in data:
                logging.info(f"       Precio: {data['last_price']}")
            elif data.get('type') == 'books-l2-tbt' and 'best_bid' in data:
                logging.info(f"       Mejor Bid: {data['best_bid']}, Mejor Ask: {data['best_ask']}")
            data_queue.task_done() # Marca la tarea como completada para esta entrada de cola
        except Exception as e:
            logging.error(f"[TEST CONSUMIDOR ERROR]: Error al procesar datos de la cola: {e}")
            await asyncio.sleep(1) # Esperar un poco antes de reintentar para evitar bucles de error rápidos


# --- Función principal de ejecución para la prueba ---
async def main_test_flow():
    logging.info(f"{'='*50}\n{'Test Flujo de Datos OKX WebSocket':^50}\n{'='*50}\n")

    # Verificar variables de entorno OKX
    required_env_vars = ['OKX_API_KEY', 'OKX_API_SECRET', 'OKX_PASSPHRASE']
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]

    if missing_vars:
        logging.error("⚠️ FALTAN VARIABLES DE ENTORNO NECESARIAS PARA OKX WEBSOCKET")
        for var in missing_vars:
            logging.error(f"  • {var}")
        logging.error("Por favor, configura config.env en la raíz de tu proyecto.")
        return # Salir si faltan credenciales

    # Inicializar el cliente WebSocket
    api_key = os.getenv("OKX_API_KEY")
    secret_key = os.getenv("OKX_API_SECRET")
    passphrase = os.getenv("OKX_PASSPHRASE")

    okx_client = OKXWebSocketClient(api_key, secret_key, passphrase, data_queue)

    # Conectar y suscribir
    logging.info("Conectando y suscribiendo al WebSocket de OKX...")
    await okx_client.connect()
    await okx_client.authenticate()
    await okx_client.subscribe([
        {"channel": "tickers", "instId": "SOL-USDT"},
        {"channel": "books-l2-tbt", "instId": "SOL-USDT"},
        {"channel": "trades", "instId": "SOL-USDT"}
    ])
    logging.info("Suscripciones enviadas. Iniciando recepción de mensajes del WebSocket.")

    # Ejecutar ambas tareas en paralelo (llenar la cola y consumir la cola)
    await asyncio.gather(
        okx_client.receive_messages(okx_client.process_message), # Este mantiene el WS vivo y llena la cola
        test_queue_consumer() # Este consume de la cola y lo imprime
    )

# --- Punto de entrada del script ---
if __name__ == "__main__":
    try:
        asyncio.run(main_test_flow())
    except KeyboardInterrupt:
        logging.info("Prueba de flujo de datos detenida por el usuario (Ctrl+C).")
    except Exception as e:
        logging.critical(f"Un error inesperado detuvo el test: {e}", exc_info=True)