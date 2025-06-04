#!/usr/bin/env python3
import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configurar logging simple solo a consola
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SolanaScalper")

# Cargar variables de entorno
config_path = Path("/home/runner/workspace/config.env")
load_dotenv(dotenv_path=config_path)

# Importar módulos del bot
sys.path.insert(0, '/home/runner/workspace')
from api_client.modulocola import data_queue
from api_client.modulo2 import OKXWebSocketClient

async def test_okx_connection():
    """Prueba rápida de conexión OKX con las correcciones aplicadas"""
    
    logger.info("==================================================")
    logger.info("SolanaScalper - Prueba de Conexión OKX")
    logger.info("==================================================")
    
    # Obtener credenciales
    api_key = os.getenv("OKX_API_KEY")
    secret_key = os.getenv("OKX_API_SECRET")
    passphrase = os.getenv("OKX_PASSPHRASE")
    
    logger.info(f"Credenciales cargadas: API_KEY={api_key[:8]}...")
    
    # Cliente público para tickers
    public_ws = OKXWebSocketClient(api_key, secret_key, passphrase, data_queue)
    public_ws.ws_url = "wss://ws.okx.com:8443/ws/v5/public"
    
    # Cliente privado para candles
    business_ws = OKXWebSocketClient(api_key, secret_key, passphrase, data_queue)
    business_ws.ws_url = "wss://ws.okx.com:8443/ws/v5/business"
    
    try:
        # Conectar público
        await public_ws.connect()
        logger.info("Conectado a WebSocket público")
        
        # Suscribirse a tickers sin instType
        await public_ws.subscribe([
            {"channel": "tickers", "instId": "SOL-USDT"}
        ])
        logger.info("Suscripción a tickers enviada")
        
        # Esperar respuesta
        await asyncio.sleep(2)
        
        # Conectar negocio
        await business_ws.connect()
        logger.info("Conectado a WebSocket de negocio")
        
        # Suscribirse a candles con nombre correcto
        await business_ws.subscribe([
            {"channel": "candle1m", "instId": "SOL-USDT"}
        ])
        logger.info("Suscripción a candles enviada")
        
        # Esperar respuestas
        await asyncio.sleep(5)
        
    except Exception as e:
        logger.error(f"Error en prueba: {e}")
    
    logger.info("Prueba completada")

if __name__ == "__main__":
    asyncio.run(test_okx_connection())