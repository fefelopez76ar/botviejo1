#!/usr/bin/env python3
"""
Demo del bot de scalping funcionando solo con datos de candles
Configurado para cuentas básicas de OKX
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
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SolanaScalper")

# Cargar configuración
sys.path.insert(0, '/home/runner/workspace')
config_path = Path("/home/runner/workspace/config.env")
load_dotenv(dotenv_path=config_path)

# Importar módulos del bot
from api_client.modulocola import data_queue
from api_client.modulo2 import OKXWebSocketClient

async def demo_scalping_bot():
    """Demo del bot de scalping con datos reales de OKX"""
    
    logger.info("="*60)
    logger.info("SOLANA SCALPING BOT - DEMO PARA CUENTA BÁSICA OKX")
    logger.info("="*60)
    
    # Obtener credenciales
    api_key = os.getenv("OKX_API_KEY")
    secret_key = os.getenv("OKX_API_SECRET")
    passphrase = os.getenv("OKX_PASSPHRASE")
    
    if not all([api_key, secret_key, passphrase]):
        logger.error("❌ Credenciales OKX no encontradas en config.env")
        return
    
    logger.info(f"📊 Configuración:")
    logger.info(f"   • Par de trading: SOL/USDT")
    logger.info(f"   • Timeframe: 1 minuto")
    logger.info(f"   • Modo: Paper Trading")
    logger.info(f"   • API Key: {api_key[:8]}...")
    
    # Crear cliente WebSocket para candles
    ws_client = OKXWebSocketClient(api_key, secret_key, passphrase, data_queue)
    ws_client.ws_url = "wss://ws.okx.com:8443/ws/v5/business"
    
    try:
        # Conectar
        logger.info("🔗 Conectando a OKX WebSocket...")
        await ws_client.connect()
        logger.info("✅ Conexión establecida exitosamente")
        
        # Suscribirse a candles de 1 minuto
        logger.info("📈 Suscribiendo a datos de velas SOL/USDT...")
        await ws_client.subscribe([
            {"channel": "candle1m", "instId": "SOL-USDT"}
        ])
        logger.info("✅ Suscripción a candles enviada")
        
        # Esperar datos
        logger.info("⏳ Esperando datos de mercado...")
        data_received = False
        
        for i in range(30):  # Esperar hasta 30 segundos
            await asyncio.sleep(1)
            
            # Verificar si hay datos en la cola
            if not data_queue.empty():
                data_count = data_queue.qsize()
                if not data_received:
                    logger.info(f"🎯 ¡Datos recibidos! Cola tiene {data_count} elementos")
                    data_received = True
                
                # Procesar algunos datos de ejemplo
                if data_count > 5:
                    logger.info("📊 Procesando datos de mercado...")
                    for _ in range(min(3, data_count)):
                        try:
                            data = data_queue.get_nowait()
                            logger.info(f"   📈 Datos: {str(data)[:100]}...")
                        except:
                            break
                    break
            else:
                if i % 10 == 0:
                    logger.info(f"   ⏱️  Esperando... ({i}/30)")
        
        if data_received:
            logger.info("✅ Bot funcionando correctamente con datos reales de OKX")
            logger.info("🤖 El bot está listo para operaciones de scalping")
        else:
            logger.warning("⚠️  No se recibieron datos en 30 segundos")
            
    except Exception as e:
        logger.error(f"❌ Error en el demo: {e}")
        
    finally:
        # Cerrar conexión
        if ws_client.ws:
            await ws_client.ws.close()
            logger.info("🔌 Conexión WebSocket cerrada")
    
    logger.info("="*60)
    logger.info("DEMO COMPLETADO")
    logger.info("="*60)

if __name__ == "__main__":
    asyncio.run(demo_scalping_bot())