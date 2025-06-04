#!/usr/bin/env python3
"""
Script principal para ejecutar el bot de trading Solana
Configurado para cuentas básicas de OKX (solo candles)
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SolanaScalper")

# Cargar configuración
config_path = Path("config.env")
load_dotenv(dotenv_path=config_path)

# Importar módulos del bot
from api_client.modulocola import data_queue
from api_client.modulo2 import OKXWebSocketClient

async def start_trading_bot():
    """Inicia el bot de trading con configuración simplificada"""
    
    logger.info("Iniciando SolanaScalper Bot...")
    logger.info("Configuración: Cuenta básica OKX - Solo datos de velas")
    
    # Verificar credenciales
    api_key = os.getenv("OKX_API_KEY")
    secret_key = os.getenv("OKX_API_SECRET")
    passphrase = os.getenv("OKX_PASSPHRASE")
    
    if not all([api_key, secret_key, passphrase]):
        logger.error("Credenciales OKX no encontradas en config.env")
        return
    
    # Crear cliente WebSocket
    ws_client = OKXWebSocketClient(api_key, secret_key, passphrase, data_queue)
    ws_client.ws_url = "wss://ws.okx.com:8443/ws/v5/business"
    
    try:
        # Conectar y suscribir
        await ws_client.connect()
        logger.info("Conexión establecida con OKX")
        
        await ws_client.subscribe([
            {"channel": "candle1m", "instId": "SOL-USDT"}
        ])
        logger.info("Suscripción a candles SOL/USDT activa")
        
        # Procesar datos en bucle
        logger.info("Bot iniciado - presiona Ctrl+C para detener")
        
        while True:
            try:
                if not data_queue.empty():
                    data = data_queue.get_nowait()
                    logger.info(f"Datos recibidos: {str(data)[:100]}...")
                    
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Deteniendo bot...")
                break
                
    except Exception as e:
        logger.error(f"Error en el bot: {e}")
        
    finally:
        if ws_client.ws:
            await ws_client.ws.close()
        logger.info("Bot detenido")

if __name__ == "__main__":
    try:
        asyncio.run(start_trading_bot())
    except KeyboardInterrupt:
        logger.info("Bot detenido por el usuario")