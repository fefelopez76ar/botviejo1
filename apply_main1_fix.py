import os
import re

def apply_main1_changes():
    main1_path = "main1.py"
    try:
        with open(main1_path, 'r') as f:
            content = f.read()

        print(f"--> Procesando {main1_path}...")

        # 1. Modificar las importaciones de OKXWebSocketClient
        old_import = r"from api_client\.modulo2 import OKXWebSocketClient"
        new_imports = (
            "from api_client.modulo2 import OKXWebSocketClient as PublicOKXWebSocketClient\n"
            "from api_client.modulo2 import OKXWebSocketClient as BusinessOKXWebSocketClient"
        )
        content = re.sub(old_import, new_imports, content, count=1)
        print("  - Importaciones de OKXWebSocketClient actualizadas.")

        # 2. Reemplazar el bloque de instanciación y suscripción
        old_block_pattern = re.compile(
            r"# Instanciar el cliente WebSocket de OKX \(con la URL por defecto\)"  # Start marker
            r"[\s\S]*?"  # Match any character (including newline) non-greedily
            r"logger\.info\(\"Suscripciones iniciales a tickers y candles enviadas\. Suscripción a Order Book Nivel 2 TBT deshabilitada por restricción VIP\.\"\)", # End marker
            re.DOTALL  # Allows . to match newlines
        )

        new_block = """
     # Instanciar el cliente WebSocket para CANALES PÚBLICOS (tickers)
     public_ws_client = PublicOKXWebSocketClient(api_key, secret_key, passphrase, data_queue)
     public_ws_client.ws_url = "wss://ws.okx.com:8443/ws/v5/public" # Asegurar la URL pública

     # Instanciar el cliente WebSocket para CANALES DE NEGOCIO (candles)
     business_ws_client = BusinessOKXWebSocketClient(api_key, secret_key, passphrase, data_queue)
     business_ws_client.ws_url = "wss://ws.okx.com:8443/ws/v5/business" # Asegurar la URL de negocio

     logger.info("Conectando y suscribiendo a los WebSockets de OKX (Público y Negocio)...")

     # Conectar y suscribir cliente público (para tickers)
     await public_ws_client.connect()
     logger.info("Enviando suscripción a Tickers (Público): {'op': 'subscribe', 'args': [{'channel': 'tickers', 'instId': 'SOL-USDT'}]}")
     await public_ws_client.subscribe([
         {"channel": "tickers", "instId": "SOL-USDT"}
     ])
     logger.info("Suscripción a Tickers SOL-USDT enviada.")

     # Conectar y suscribir cliente de negocio (para candles)
     await business_ws_client.connect()
     logger.info("Enviando suscripción a Candles (Negocio): {'op': 'subscribe', 'args': [{'channel': 'candles', 'instId': 'SOL-USDT', 'bar': '1m'}]}")
     await business_ws_client.subscribe([
         {"channel": "candles", "instId": "SOL-USDT", "bar": "1m"}
     ])
     logger.info("Suscripción a Candles SOL-USDT (1m) enviada.")

     logger.info("Suscripciones iniciales enviadas. Suscripción a Order Book Nivel 2 TBT deshabilitada por restricción VIP.")
        """
        new_block_lines = new_block.strip().splitlines()
        indented_new_block = "\n".join(["    " + line for line in new_block_lines])
        
        content = old_block_pattern.sub(indented_new_block, content, count=1)
        print("  - Bloque de instanciación y suscripción de clientes reemplazado.")


        # 3. Modificar las tareas de recepción de mensajes
        old_tasks_append = r"tasks\.append\(okx_client\.receive_messages\(\)\)"
        new_tasks_append = (
            "tasks.append(public_ws_client.receive_messages())\n"
            "        tasks.append(business_ws_client.receive_messages())"
        )
        content = re.sub(old_tasks_append, new_tasks_append, content, count=1)
        print("  - Tareas de recepción de mensajes actualizadas.")

        # 4. Modificar el apagado del cliente WebSocket
        old_close_block = re.compile(
            r"if okx_client\.ws and hasattr\(okx_client\.ws, 'closed'\) and not okx_client\.ws\.closed:\s*"
            r"await okx_client\.ws\.close\(\)",
            re.DOTALL
        )
        new_close_block = """
         if public_ws_client.ws and hasattr(public_ws_client.ws, 'closed') and not public_ws_client.ws.closed:
             await public_ws_client.ws.close()
         if business_ws_client.ws and hasattr(business_ws_client.ws, 'closed') and not business_ws_client.ws.closed:
             await business_ws_client.ws.close()
        """
        new_close_block_indented = "\n".join(["            " + line.strip() for line in new_close_block.strip().splitlines()])
        content = old_close_block.sub(new_close_block_indented, content, count=1)
        print("  - Lógica de cierre de conexiones WebSocket actualizada.")


        with open(main1_path, 'w') as f:
            f.write(content)
        print(f"--> {main1_path}: Todos los cambios aplicados correctamente.")

    except FileNotFoundError:
        print(f"Error: {main1_path} no encontrado. Asegúrate de estar en el directorio raíz del proyecto.")
    except Exception as e:
        print(f"Error al modificar {main1_path}: {e}")

if __name__ == "__main__":
    apply_main1_changes()
