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
from datetime import datetime

# Cargar las variables de entorno desde config.env con ruta completa
# Nota: Este load_dotenv se hace aquí por si el módulo se ejecuta de forma independiente.
# En main.py también se cargan.
config_path = Path(__file__).parent.parent / "config.env"
load_dotenv(dotenv_path=config_path)

# Crear carpeta 'info' si no existe y configurar logging en un archivo txt llamado 'pendientes.txt'
info_dir = Path(__file__).parent.parent / "info"
info_dir.mkdir(exist_ok=True) # Asegura que la carpeta 'info' exista

# Ruta del archivo pendientes.txt
log_txt_path = info_dir / "pendientes.txt"

# Escribir contenido inicial en pendientes.txt
contenido_inicial = """Pendientes del proyecto:

1. **Final Testing:**
   - Run the bot to confirm that it initializes correctly, processes data, and saves it to the database without errors.
   - Verify that the database (`market_data.db`) contains the expected data using `verificar_db.py`.

2. **Error Handling:**
   - Ensure robust error handling for any remaining edge cases during bot execution.

3. **Folder and File Verification:**
   - Confirm that the `info/` folder exists and contains the necessary logs.
   - Verify that the new `market_data.db` file is created with the correct schema.

4. **Data Validation:**
   - Compress and share the `info/` folder for further analysis of saved data.

"""

with open(log_txt_path, "w", encoding="utf-8") as f:
    f.write(contenido_inicial)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_txt_path, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class OKXWebSocketClient:
    def __init__(self, api_key, secret_key, passphrase, data_queue):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.data_queue = data_queue
        self.ws = None
        self.is_connected = False
        # Usar la URL pública para tickers/books/trades
        self.base_url = "wss://ws.okx.com:8443/ws/v5/public"
        # Para canales privados (autenticados) como orders, balance, etc., se usaría:
        # self.base_url = "wss://ws.okx.com:8443/ws/v5/private"


    async def connect(self):
        """Establece la conexión WebSocket con OKX."""
        try:
            self.ws = await websockets.connect(self.base_url)
            self.is_connected = True
            logging.info("Conexión establecida con la API de OKX WebSocket.")
        except Exception as e:
            logging.error(f"Error al conectar con OKX WebSocket: {e}")
            self.is_connected = False
            raise # Re-lanzar la excepción para que el llamador la maneje

    async def authenticate(self):
        """Realiza la autenticación con la API privada de OKX."""
        if not self.is_connected:
            logging.error("No se puede autenticar: el WebSocket no está conectado.")
            return False

        timestamp = str(int(time.time()))  # Usar tiempo en segundos
        signature_payload = timestamp + "GET" + "/users/self/verify" # Para login, el body es vacío
        sign = base64.b64encode(
            hmac.new(
                self.secret_key.encode('utf-8'),
                signature_payload.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode()

        login_message = {
            "op": "login",
            "args": [{
                "apiKey": self.api_key,
                "passphrase": self.passphrase,
                "timestamp": timestamp,
                "sign": sign
            }]
        }
        logging.info("Enviando solicitud de autenticación a OKX WebSocket...")
        await self.ws.send(json.dumps(login_message))

        try:
            auth_response_str = await asyncio.wait_for(self.ws.recv(), timeout=10)
            auth_response = json.loads(auth_response_str)
            logging.info(f"Respuesta de autenticación: {auth_response}")

            if "event" in auth_response and auth_response["event"] == "login":
                if "code" in auth_response and auth_response["code"] == "0":
                    logging.info("Autenticación con OKX WebSocket exitosa.")
                    return True
                else:
                    error_msg = auth_response.get("msg", "Error desconocido")
                    error_code = auth_response.get("code", "N/A")
                    logging.error(f"Error de autenticación con OKX: {error_msg} (Código: {error_code})")
                    return False
            elif "event" in auth_response and auth_response["event"] == "error":
                error_msg = auth_response.get("msg", "Error desconocido")
                error_code = auth_response.get("code", "N/A")
                logging.error(f"Respuesta de autenticación con error: {error_msg} (Código: {error_code})")
                return False
            else:
                logging.warning(f"Respuesta de autenticación inesperada: {auth_response}")
                return False
        except asyncio.TimeoutError:
            logging.error("Timeout al esperar respuesta de autenticación de OKX.")
            return False
        except json.JSONDecodeError:
            logging.error("No se pudo decodificar la respuesta JSON de autenticación.")
            return False
        except Exception as e:
            logging.error(f"Error durante la autenticación de OKX: {e}")
            return False


    async def subscribe(self, channels: list):
        """
        Suscribe a los canales especificados.
        Channels es una lista de diccionarios, ej:
        [{"channel": "tickers", "instId": "SOL-USDT"}, {"channel": "books", "instId": "SOL-USDT"}]
        """
        if not self.is_connected:
            logging.error("No se puede suscribir: el WebSocket no está conectado.")
            return

        # Filtrar el canal 'trades' de la lista de canales
        channels = [channel for channel in channels if channel.get("channel") != "trades"]

        subscribe_message = {
            "op": "subscribe",
            "args": channels
        }
        logging.info(f"Enviando suscripción: {subscribe_message}")
        await self.ws.send(json.dumps(subscribe_message))
        # Opcional: Esperar confirmación de suscripción si es necesario
        try:
            sub_response_str = await asyncio.wait_for(self.ws.recv(), timeout=5)
            sub_response = json.loads(sub_response_str)
            logging.info(f"Respuesta de suscripción: {sub_response}")
            if "event" in sub_response and sub_response["event"] == "subscribe" and sub_response.get("success", False):
                logging.info(f"Suscripción exitosa para {sub_response['arg']['channel']} - {sub_response['arg']['instId']}")
            elif "event" in sub_response and sub_response["event"] == "error":
                logging.error(f"Error al suscribirse: {sub_response.get('msg', 'Error desconocido')}")
        except asyncio.TimeoutError:
            logging.warning("Timeout esperando confirmación de suscripción. La suscripción pudo haber sido exitosa de todos modos.")
        except Exception as e:
            logging.error(f"Error al procesar respuesta de suscripción: {e}")


    async def receive_messages(self, message_processor):
        """
        Recibe mensajes del WebSocket y los pasa a un procesador.
        El procesador debe ser una función/método que acepta un mensaje JSON decodificado.
        """
        logging.info("Iniciando recepción de mensajes del WebSocket.")
        try:
            while self.is_connected:
                message = await self.ws.recv()
                # logging.debug(f"Mensaje crudo recibido: {message[:100]}...") # Evitar imprimir mensajes muy largos
                await message_processor(message) # Pasa el mensaje al procesador (process_message en este caso)
        except websockets.exceptions.ConnectionClosedOK:
            logging.info("Conexión WebSocket cerrada limpiamente.")
        except websockets.exceptions.ConnectionClosedError as e:
            logging.error(f"Conexión WebSocket cerrada con error: {e.code} - {e.reason}")
            self.is_connected = False
        except Exception as e:
            logging.error(f"Error inesperado al recibir mensajes WebSocket: {e}")
            self.is_connected = False
        finally:
            if self.ws:
                await self.ws.close()
            logging.info("Receptor de mensajes del WebSocket detenido.")


    async def process_message(self, message):
        """Procesa un mensaje JSON recibido del WebSocket y lo pone en la cola."""
        try:
            data = json.loads(message)

            if "event" in data:
                # Mensajes de evento como 'login', 'subscribe', 'error', 'pong'
                if data["event"] == "pong":
                    logging.debug("Ping/Pong: Latencia medida: %s ms", data.get("latency", "N/A"))
                elif data["event"] == "error":
                    logging.error(f"Error del servidor OKX: {data.get('msg', 'N/A')} (Código: {data.get('code', 'N/A')})")
                elif data["event"] == "login" or data["event"] == "subscribe":
                    # Estos ya se logean en authenticate/subscribe. No necesitamos procesarlos aquí de nuevo.
                    pass
                else:
                    logging.debug(f"Evento inesperado: {data}")
                return # No enviar eventos a la cola de datos de trading

            # Si no es un evento, debería ser un mensaje de datos
            if "arg" in data and "data" in data:
                channel = data["arg"].get("channel")
                inst_id = data["arg"].get("instId")
                payload_data = data["data"] # Lista de diccionarios

                processed_item = None
                if channel == "tickers" and payload_data:
                    ticker_info = payload_data[0]
                    # Simplificamos la estructura para la cola
                    processed_item = {
                        "type": "ticker",
                        "instrument": inst_id,
                        "timestamp": ticker_info.get('ts'),
                        "last_price": float(ticker_info.get('last')),
                        "best_bid": float(ticker_info.get('bidPx')),
                        "best_ask": float(ticker_info.get('askPx')),
                        "data": payload_data # Opcional: mantener la data original completa
                    }
                    # logging.info(f"Mejor Bid: {processed_item['best_bid']}, Tamaño: {ticker_info.get('bidSz')} | Mejor Ask: {processed_item['best_ask']}, Tamaño: {ticker_info.get('askSz')}")

                elif channel == "books-l2-tbt" and payload_data:
                    book_info = payload_data[0]
                    bids = book_info.get('bids', [])
                    asks = book_info.get('asks', [])
                    # Simplificamos para la cola
                    processed_item = {
                        "type": "books-l2-tbt",
                        "instrument": inst_id,
                        "timestamp": book_info.get('ts'),
                        "best_bid": float(bids[0][0]) if bids else None,
                        "best_ask": float(asks[0][0]) if asks else None,
                        "data": payload_data # Opcional: mantener la data original completa
                    }

                # Si hemos procesado algo, lo ponemos en la cola
                if processed_item:
                    await self.data_queue.put(processed_item)
                    logging.info(f"Datos '{processed_item['type']}' de '{processed_item['instrument']}' puestos en la cola.")
                else:
                    logging.debug(f"Datos de canal '{channel}' no procesados o vacíos: {data}")

        except json.JSONDecodeError:
            logging.warning(f"No se pudo decodificar el mensaje JSON recibido: {message[:100]}...")
        except Exception as e:
            logging.error(f"Error al procesar mensaje WebSocket: {e} - Mensaje: {message[:200]}...")