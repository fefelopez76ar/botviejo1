import subprocess
import asyncio
import websockets
import json
import time
from prettytable import PrettyTable

def start_modulo1():
    """Inicia el script modulo1.py como un proceso independiente."""
    subprocess.Popen(["python", "modulo1.py"], cwd="c:\\proyectos\\mibot2\\CryptoTradingBot")

async def fetch_data():
    """Conecta al WebSocket de modulo1 y recibe datos."""
    url = "ws://localhost:8765"  # Asegúrate de que modulo1 exponga datos en este WebSocket
    try:
        async with websockets.connect(url) as websocket:
            while True:
                try:
                    data = await websocket.recv()
                    display_data(data)
                except json.JSONDecodeError:
                    print("[ERROR] No se pudo decodificar el mensaje recibido.")
                except Exception as e:
                    print(f"[ERROR] Error al procesar los datos: {e}")

                print("\n[INFO] Los datos se refrescarán nuevamente en 30 segundos...")
                await asyncio.sleep(30)  # Refrescar cada 30 segundos
    except Exception as e:
        print(f"[ERROR] Error al conectar con el WebSocket: {e}")


def display_data(data):
    """Muestra los datos en un cuadro con explicaciones."""
    try:
        parsed_data = json.loads(data)

        # Verificar si los datos tienen la estructura esperada
        if not isinstance(parsed_data, dict):
            print("[ERROR] Estructura de datos inesperada. Se esperaba un diccionario.")
            return

        # Crear tabla para mostrar datos
        table = PrettyTable()
        table.field_names = ["Canal", "Instrumento", "Detalle"]

        for channel, details in parsed_data.items():
            if not isinstance(details, list):
                print(f"[ERROR] Estructura inesperada en el canal {channel}. Se esperaba una lista.")
                continue

            for detail in details:
                # Agregar explicaciones para cada canal
                explanation = ""
                if channel == "tickers":
                    explanation = "Información de precios en tiempo real."
                elif channel == "books":
                    explanation = "Libro de órdenes con precios de compra y venta."
                elif channel == "trades":
                    explanation = "Historial de transacciones recientes."

                table.add_row([channel, detail.get("instId", "N/A"), f"{json.dumps(detail)}\n{explanation}"])

        print("\n[Datos Recibidos - Refrescados cada 30 segundos]")
        print(table)
    except json.JSONDecodeError:
        print("[ERROR] No se pudo decodificar los datos recibidos.")
    except Exception as e:
        print(f"[ERROR] Error al mostrar los datos: {e}")

if __name__ == "__main__":
    start_modulo1()  # Inicia modulo1.py
    asyncio.run(fetch_data())
