import asyncio
import websockets
import json
import os
from dotenv import load_dotenv
from pathlib import Path
import logging
import hmac
import hashlib
import time

# Cargar las variables de entorno desde config.env con ruta completa
config_path = Path(__file__).parent / "config.env"
load_dotenv(dotenv_path=config_path)

# Configurar logging para manejar errores y eventos
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Depuración: Imprimir las variables cargadas
print("Depuración: Variables de entorno cargadas")
print(f"OKX_API_KEY: {os.getenv('OKX_API_KEY')}")
print(f"OKX_API_SECRET: {os.getenv('OKX_API_SECRET')}")
print(f"OKX_PASSPHRASE: {os.getenv('OKX_PASSPHRASE')}")

class OKXWebSocketClient:
    def __init__(self, api_key, secret_key, passphrase):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.url = "wss://ws.okx.com:8443/ws/v5/public"
        self.connection = None

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

        # Generar timestamp en milisegundos
        timestamp = str(int(time.time() * 1000))

        # Crear firma
        message = timestamp + 'GET' + '/users/self/verify'
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

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
        except Exception as e:
            logging.error(f"Error durante la suscripción: {e}")

    async def receive_messages(self, process_message_callback):
        if not self.connection:
            raise Exception("Primero debes establecer una conexión.")

        try:
            while True:
                message = await self.connection.recv()
                process_message_callback(message)
        except websockets.ConnectionClosed as e:
            logging.warning(f"Conexión cerrada: {e}. Intentando reconectar...")
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
        client = OKXWebSocketClient(api_key, secret_key, passphrase)

        async def main():
            await client.connect()
            await client.authenticate()
            await client.subscribe([
                {"channel": "tickers", "instId": "SOL-USDT"},
                {"channel": "books", "instId": "SOL-USDT"},
                {"channel": "trades", "instId": "SOL-USDT"}
            ])

            def process_message(message):
                if not hasattr(process_message, "last_print_time"):
                    process_message.last_print_time = 0  # Inicializar el atributo

                try:
                    data = json.loads(message)
                    if "data" in data:
                        current_time = asyncio.get_event_loop().time()

                        if current_time - process_message.last_print_time >= 15:  # Intervalo de 15 segundos
                            resumen = {
                                "Canal": data.get("arg", {}).get("channel", "N/A"),
                                "Instrumento": data.get("arg", {}).get("instId", "N/A"),
                                "Datos": data.get("data", "N/A")
                            }

                            # Procesar datos del libro de órdenes
                            if resumen["Canal"] == "books":
                                asks = resumen["Datos"][0].get("asks", [])
                                bids = resumen["Datos"][0].get("bids", [])

                                if asks and bids and len(asks[0]) > 0 and len(bids[0]) > 0:
                                    try:
                                        best_ask = float(asks[0][0])  # Mejor precio de venta
                                        best_bid = float(bids[0][0])  # Mejor precio de compra
                                        spread = best_ask - best_bid  # Calcular el spread

                                        logging.info(f"Spread actual: {spread:.2f}")
                                    except (ValueError, IndexError) as e:
                                        logging.warning(f"Error al procesar datos del libro de órdenes: {e}")
                                else:
                                    logging.warning("Datos incompletos o inválidos en 'asks' o 'bids'.")

                            # Procesar datos de trades
                            if resumen["Canal"] == "trades":
                                for trade in resumen["Datos"]:
                                    price = trade.get("px", "N/A")
                                    size = trade.get("sz", "N/A")
                                    side = trade.get("side", "N/A")
                                    logging.info(f"Trade ejecutado - Precio: {price}, Tamaño: {size}, Dirección: {side}")

                            # Formatear la salida para mayor legibilidad
                            logging.info("\n[Resumen de Datos]\n" + "\n".join(f"{key}: {value}" for key, value in resumen.items()) + "\n-------------------")

                            process_message.last_print_time = current_time
                    else:
                        logging.debug("Mensaje recibido sin datos relevantes.")
                except json.JSONDecodeError:
                    logging.warning("No se pudo decodificar el mensaje recibido.")

            await client.receive_messages(process_message)

        asyncio.run(main())
