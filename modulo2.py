import asyncio
import websockets
import json
import os
from dotenv import load_dotenv
from pathlib import Path
import logging
import hmac
import hashlib
import base64
import time

# Cargar las variables de entorno desde config.env con ruta completa
config_path = Path(__file__).parent / "config.env"
load_dotenv(dotenv_path=config_path)

# Crear carpeta 'info' si no existe
info_dir = Path(__file__).parent / "info"
info_dir.mkdir(exist_ok=True)

# Configurar logging para guardar en la carpeta 'info'
log_file_path = info_dir / "bot.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Verificar y crear la carpeta 'info' al inicio del script
if not info_dir.exists():
    try:
        info_dir.mkdir()
        logging.info("Carpeta 'info' creada exitosamente.")
    except Exception as e:
        logging.error(f"Error al crear la carpeta 'info': {e}")

# Depuración: Imprimir las variables cargadas
print("Depuración: Variables de entorno cargadas")
print(f"OKX_API_KEY: {os.getenv('OKX_API_KEY')}")
print(f"OKX_API_SECRET: {os.getenv('OKX_API_SECRET')}")
print(f"OKX_PASSPHRASE: {os.getenv('OKX_PASSPHRASE')}")

# ==============================================================================
# CAMBIOS AÑADIDOS AQUI para la cola
# Importar la cola desde el archivo modulocola.py
from modulocola import data_queue
# ==============================================================================

class OKXWebSocketClient:
    # ==============================================================================
    # CAMBIOS AÑADIDOS AQUI para la cola
    # El constructor ahora recibe la cola como argumento
    def __init__(self, api_key, secret_key, passphrase, data_queue: asyncio.Queue):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.url = "wss://ws.okx.com:8443/ws/v5/public"
        self.connection = None
        self.data_queue = data_queue # Guardamos la referencia a la cola
    # ==============================================================================

    async def connect(self):
        while True:
            try:
                self.connection = await websockets.connect(self.url)
                logging.info("Conexión establecida con la API de OKX WebSocket.")
                break
            except Exception as e:
                logging.error(f"Error al conectar con la API de OKX WebSocket: {e}")
                logging.info("Reintentando conexión en 5 segundos...")
                await asyncio.sleep(5)

    async def authenticate(self):
        if not self.connection:
            raise Exception("Primero debes establecer una conexión.")

        # Generar timestamp
        timestamp = str(int(time.time()))

        # Crear firma
        message = timestamp + 'GET' + '/users/self/verify'
        signature = base64.b64encode(hmac.new(self.secret_key.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).digest()).decode('utf-8')

        # Mensaje de autenticación
        auth_message = {
            "op": "login",
            "args": [
                {
                    "apiKey": self.api_key,
                    "passphrase": self.passphrase,
                    "timestamp": timestamp,
                    "sign": signature
                }
            ]
        }

        try:
            await self.connection.send(json.dumps(auth_message))
            response = await self.connection.recv()
            logging.info(f"Respuesta de autenticación: {response}")
        except Exception as e:
            logging.error(f"Error durante la autenticación: {e}")

    async def measure_latency(self):
        """Mide la latencia de la conexión con el servidor de OKX."""
        if not self.connection:
            raise Exception("Primero debes establecer una conexión.")

        try:
            start_time = time.time()
            await self.connection.ping()
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convertir a milisegundos
            logging.info(f"Latencia medida: {latency:.2f} ms")
        except Exception as e:
            logging.error(f"Error al medir la latencia: {e}")

    async def subscribe(self, channels):
        if not self.connection:
            raise Exception("Primero debes establecer una conexión.")

        # Mensaje de suscripción
        subscribe_message = {
            "op": "subscribe",
            "args": channels
        }

        try:
            await self.connection.send(json.dumps(subscribe_message))
            response = await self.connection.recv()
            logging.info(f"Respuesta de suscripción: {response}")

            # Medir latencia después de la suscripción
            await self.measure_latency()
        except Exception as e:
            logging.error(f"Error durante la suscripción: {e}")

    async def subscribe_to_deep_books(self, instrument_id):
        """Suscribirse a un canal de mayor profundidad del libro de órdenes."""
        if not self.connection:
            raise Exception("Primero debes establecer una conexión.")

        deep_books_channel = {
            "channel": "books-l2-tbt",  # Canal de mayor profundidad
            "instId": instrument_id
        }

        try:
            await self.subscribe([deep_books_channel])
            logging.info(f"Suscripción exitosa al canal de mayor profundidad para {instrument_id}.")
        except Exception as e:
            logging.error(f"Error al suscribirse al canal de mayor profundidad: {e}")

    async def fetch_historical_data(self, instrument_id):
        """Obtener datos históricos granulares para análisis."""
        # Este método puede usar una API REST separada para obtener datos históricos
        logging.info(f"Obteniendo datos históricos para {instrument_id}...")
        # Aquí se implementaría la lógica para llamar a la API REST y procesar los datos
        pass

    async def process_ticker_data(self, data):
        """Procesar múltiples niveles de bid/ask del ticker si están disponibles."""
        best_bid = data.get("bidPx", "N/A")
        best_ask = data.get("askPx", "N/A")
        bid_size = data.get("bidSz", "N/A")
        ask_size = data.get("askSz", "N/A")

        logging.info(f"Mejor Bid: {best_bid}, Tamaño: {bid_size} | Mejor Ask: {best_ask}, Tamaño: {ask_size}")

        # Si hay más niveles disponibles, procesarlos
        if "additionalBids" in data and "additionalAsks" in data:
            logging.info(f"Niveles adicionales de Bid/Ask disponibles para {data.get('instId', 'N/A')}.")
        
        # ==============================================================================
        # CAMBIOS AÑADIDOS AQUI para la cola
        # Poner datos de tickers en la cola
        if "last" in data: # Aseguramos que tenemos el precio final
            processed_ticker_data = {
                "type": "ticker",
                "instrument": data.get("instId", "N/A"),
                "last_price": float(data["last"]),
                "timestamp": data.get("ts") # OKX suele enviar timestamp
            }
            await self.data_queue.put(processed_ticker_data)
            logging.debug(f"Ticker para {data.get('instId', 'N/A')} puesto en la cola. Tamaño: {self.data_queue.qsize()}")
        # ==============================================================================

    async def process_books_data(self, data):
        """Procesa datos del libro de órdenes con mayor profundidad."""
        asks = data.get("asks", [])
        bids = data.get("bids", [])

        if asks and bids:
            try:
                best_ask = float(asks[0][0])  # Mejor precio de venta
                best_bid = float(bids[0][0])  # Mejor precio de compra
                spread = best_ask - best_bid  # Calcular el spread

                logging.info(f"Spread actual: {spread:.2f}")

                # Procesar niveles adicionales si están disponibles
                if len(asks) > 1 and len(bids) > 1:
                    logging.info(f"Segundo mejor ask: {asks[1][0]}, Segundo mejor bid: {bids[1][0]}")
            except (ValueError, IndexError) as e:
                logging.warning(f"Error al procesar datos del libro de órdenes: {e}")
        else:
            logging.warning("Datos incompletos o inválidos en 'asks' o 'bids'.")
        
        # ==============================================================================
        # CAMBIOS AÑADIDOS AQUI para la cola
        # Poner datos de books en la cola
        if asks and bids:
            try:
                processed_book_data = {
                    "type": "books-l2-tbt",
                    "instrument": data.get("instId", "N/A"),
                    "best_bid": float(bids[0][0]),
                    "best_ask": float(asks[0][0]),
                    "full_bids": bids, # Incluimos los datos completos del libro
                    "full_asks": asks, # para que el bot tenga toda la información
                    "timestamp": data.get("ts")
                }
                await self.data_queue.put(processed_book_data)
                logging.debug(f"Libro de órdenes para {data.get('instId', 'N/A')} puesto en la cola. Tamaño: {self.data_queue.qsize()}")
            except (ValueError, IndexError) as e:
                logging.warning(f"Error al preparar datos del libro para la cola: {e}")
        # ==============================================================================

    async def process_message(self, message):
        """Procesa mensajes recibidos de manera eficiente y controla la salida."""
        if not hasattr(self, "last_print_time"):
            self.last_print_time = 0  # Inicializar el tiempo de la última impresión

        try:
            data = json.loads(message)
            if "arg" in data and "data" in data:
                channel = data["arg"].get("channel", "N/A")
                instrument = data["arg"].get("instId", "N/A")
                payload = data["data"]

                # Procesar datos según el canal
                if channel == "books-l2-tbt":
                    await self.process_books_data(payload[0])
                elif channel == "tickers":
                    await self.process_ticker_data(payload[0])
                elif channel == "trades":
                    for trade in payload:
                        logging.debug(f"Trade ejecutado - Precio: {trade.get('px', 'N/A')}, Tamaño: {trade.get('sz', 'N/A')}, Dirección: {trade.get('side', 'N/A')}")
                    # ==============================================================================
                    # CAMBIOS AÑADIDOS AQUI para la cola
                    # Poner datos de trades en la cola
                    if payload:
                        processed_trades_data = {
                            "type": "trades",
                            "instrument": instrument,
                            "trades": payload, # Lista de trades
                            "timestamp": time.time() * 1000 # Añadir timestamp si no viene en los datos
                        }
                        await self.data_queue.put(processed_trades_data)
                        logging.debug(f"Trades para {instrument} puestos en la cola. Tamaño: {self.data_queue.qsize()}")
                    # ==============================================================================

                # Controlar la frecuencia de impresión
                current_time = time.time()
                if current_time - self.last_print_time >= 10:  # Intervalo de 10 segundos
                    logging.info(f"[Resumen] Canal: {channel}, Instrumento: {instrument}, Datos: {payload[:1]}")
                    self.last_print_time = current_time
            else:
                logging.debug("Mensaje recibido sin datos relevantes.")
        except json.JSONDecodeError:
            logging.warning("No se pudo decodificar el mensaje recibido.")
        except Exception as e:
            logging.error(f"Error al procesar el mensaje: {e}")

    async def receive_messages(self, process_message_callback=None):
        """Recibe mensajes del WebSocket y los procesa si se proporciona un callback."""
        if not self.connection:
            raise Exception("Primero debes establecer una conexión.")

        try:
            while True:
                message = await self.connection.recv()
                if process_message_callback:
                    await process_message_callback(message)
                else:
                    logging.info(f"Mensaje recibido: {message}")
        except websockets.ConnectionClosed as e:
            logging.warning(f"Conexión cerrada: {e}. Intentando reconectar...")
            # ==============================================================================
            # CAMBIOS AÑADIDOS AQUI para la cola
            # Guardar la lista de canales suscritos para re-suscribir
            logging.info("Reconectando y re-suscribiendo a los canales activos.")
            # En un bot real, necesitarías una lista de canales a los que estabas suscrito
            # y re-enviar las suscripciones aquí después de connect y authenticate.
            # Por ahora, el ejemplo de uso en main() volverá a suscribir si main() se reinicia.
            # ==============================================================================
            await self.connect()
            await self.authenticate()
        except Exception as e:
            logging.error(f"Error al recibir mensajes: {e}")

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar credenciales desde el archivo de configuración
    api_key = os.getenv("OKX_API_KEY")
    secret_key = os.getenv("OKX_API_SECRET")
    passphrase = os.getenv("OKX_PASSPHRASE")

    if not all([api_key, secret_key, passphrase]):
        logging.error("Por favor, asegúrate de que las credenciales están configuradas en el archivo config.env.")
    else:
        # ==============================================================================
        # CAMBIOS AÑADIDOS AQUI para la cola
        # Pasamos la cola al cliente al crearlo
        client = OKXWebSocketClient(api_key, secret_key, passphrase, data_queue)
        # ==============================================================================

        async def main():
            await client.connect()
            await client.authenticate()
            await client.subscribe([
                {"channel": "tickers", "instId": "SOL-USDT"},
                {"channel": "books", "instId": "SOL-USDT"},
                {"channel": "trades", "instId": "SOL-USDT"}
            ])
            await client.subscribe_to_deep_books("SOL-USDT")  # Suscribirse al canal de mayor profundidad

            # ==============================================================================
            # CAMBIOS AÑADIDOS AQUI para la cola
            # Definimos un "consumidor" de la cola para fines de prueba
            async def queue_consumer():
                while True:
                    # Espera a que haya un elemento en la cola
                    data = await data_queue.get()
                    logging.info(f"CONSUMIDOR DE COLA: Datos recibidos: {data.get('type', 'desconocido')} para {data.get('instrument', 'N/A')}. Tamaño de cola actual: {data_queue.qsize()}")
                    # Aquí es donde tu "otro bot" (de trading) procesaría estos datos
                    # Por ejemplo, si es un ticker: if data["type"] == "ticker": print(f"Precio actual de {data['instrument']}: {data['last_price']}")
                    data_queue.task_done() # Marca la tarea como completada para esta entrada de cola
            # ==============================================================================


            # NOTA: La función process_message que estaba aquí en el __main__ original
            # ha sido eliminada porque la función process_message del cliente (self.process_message)
            # es la que usaremos y ya la hemos modificado para poner en la cola.

            # ==============================================================================
            # CAMBIOS AÑADIDOS AQUI para la cola
            # Ejecutamos el cliente y el consumidor de la cola en paralelo
            await asyncio.gather(
                client.receive_messages(client.process_message), # El cliente recibe y procesa mensajes
                queue_consumer() # El consumidor toma mensajes de la cola
            )
            # ==============================================================================

        asyncio.run(main())