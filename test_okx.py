#!/usr/bin/env python3
"""
Prueba los canales correctos para cuentas básicas de OKX
Basado en documentación oficial de OKX API WebSocket
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

sys.path.insert(0, '/home/runner/workspace')
config_path = Path("/home/runner/workspace/config.env")
load_dotenv(dotenv_path=config_path)

from api_client.modulocola import data_queue
from api_client.modulo2 import OKXWebSocketClient

async def test_correct_channels():
    """Prueba los canales correctos según documentación OKX"""
    
    logger.info("=== PRUEBA CANALES CORRECTOS OKX BÁSICO ===")
    
    api_key = os.getenv("OKX_API_KEY")
    secret_key = os.getenv("OKX_API_SECRET")
    passphrase = os.getenv("OKX_PASSPHRASE")
    
    # Configuraciones correctas según documentación OKX
    test_configs = [
        {
            "name": "Public - books",
            "url": "wss://ws.okx.com:8443/ws/v5/public",
            "channel": "books",
            "params": {"instId": "SOL-USDT"}
        },
        {
            "name": "Public - trades", 
            "url": "wss://ws.okx.com:8443/ws/v5/public",
            "channel": "trades",
            "params": {"instId": "SOL-USDT"}
        },
        {
            "name": "Business - tickers (en business)",
            "url": "wss://ws.okx.com:8443/ws/v5/business",
            "channel": "tickers",
            "params": {"instId": "SOL-USDT"}
        },
        {
            "name": "Business - candle1m (confirmado)",
            "url": "wss://ws.okx.com:8443/ws/v5/business",
            "channel": "candle1m", 
            "params": {"instId": "SOL-USDT"}
        },
        {
            "name": "Business - index-tickers",
            "url": "wss://ws.okx.com:8443/ws/v5/business",
            "channel": "index-tickers",
            "params": {"instId": "SOL-USDT"}
        }
    ]
    
    for config in test_configs:
        logger.info(f"\n--- Probando: {config['name']} ---")
        
        try:
            client = OKXWebSocketClient(api_key, secret_key, passphrase, data_queue)
            client.ws_url = config['url']
            
            await client.connect()
            logger.info(f"✓ Conectado a {config['url'].split('/')[-1]}")
            
            subscription = {
                "channel": config['channel'],
                **config['params']
            }
            
            await client.subscribe([subscription])
            logger.info(f"→ Enviado: {subscription}")
            
            # Esperar respuesta
            await asyncio.sleep(3)
            
            if client.ws:
                await client.ws.close()
                
        except Exception as e:
            logger.error(f"✗ Error: {e}")
            
        await asyncio.sleep(1)
    
    logger.info("\n=== PRUEBA COMPLETADA ===")

if __name__ == "__main__":
    asyncio.run(test_correct_channels())