#!/usr/bin/env python3
import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Cargar configuración
sys.path.insert(0, '/home/runner/workspace')
config_path = Path("/home/runner/workspace/config.env")
load_dotenv(dotenv_path=config_path)

from api_client.modulocola import data_queue
from api_client.modulo2 import OKXWebSocketClient

async def test_final():
    logger.info("=== PRUEBA FINAL BOT SOLANA OKX ===")
    
    # Credenciales
    api_key = os.getenv("OKX_API_KEY")
    secret_key = os.getenv("OKX_API_SECRET")
    passphrase = os.getenv("OKX_PASSPHRASE")
    
    # Cliente público para ticker
    public_ws = OKXWebSocketClient(api_key, secret_key, passphrase, data_queue)
    public_ws.ws_url = "wss://ws.okx.com:8443/ws/v5/public"
    
    # Cliente de negocio para candles
    business_ws = OKXWebSocketClient(api_key, secret_key, passphrase, data_queue)
    business_ws.ws_url = "wss://ws.okx.com:8443/ws/v5/business"
    
    try:
        # TICKER (público)
        await public_ws.connect()
        logger.info("✓ Conectado a WebSocket público")
        
        await public_ws.subscribe([{"channel": "ticker", "instId": "SOL-USDT"}])
        logger.info("→ Suscripción a ticker enviada")
        await asyncio.sleep(2)
        
        # CANDLES (negocio)
        await business_ws.connect()
        logger.info("✓ Conectado a WebSocket de negocio")
        
        await business_ws.subscribe([{"channel": "candle1m", "instId": "SOL-USDT"}])
        logger.info("→ Suscripción a candles enviada")
        await asyncio.sleep(3)
        
        logger.info("=== PRUEBA COMPLETADA ===")
        
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_final())