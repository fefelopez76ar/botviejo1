#!/usr/bin/env python3
"""
Test script to verify OKX WebSocket data flow and channel names
This will help identify which channels work on which endpoints
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Load configuration
sys.path.insert(0, '/home/runner/workspace')
config_path = Path("/home/runner/workspace/config.env")
load_dotenv(dotenv_path=config_path)

from api_client.modulocola import data_queue
from api_client.modulo2 import OKXWebSocketClient

async def test_channel_variations():
    """Test different channel names and combinations"""
    
    logger.info("=== TESTING OKX CHANNEL VARIATIONS ===")
    
    api_key = os.getenv("OKX_API_KEY")
    secret_key = os.getenv("OKX_API_SECRET")
    passphrase = os.getenv("OKX_PASSPHRASE")
    
    # Test configurations
    test_configs = [
        {
            "name": "Public - tickers",
            "url": "wss://ws.okx.com:8443/ws/v5/public",
            "channel": "tickers",
            "params": {"instId": "SOL-USDT"}
        },
        {
            "name": "Public - ticker", 
            "url": "wss://ws.okx.com:8443/ws/v5/public",
            "channel": "ticker",
            "params": {"instId": "SOL-USDT"}
        },
        {
            "name": "Public - books5",
            "url": "wss://ws.okx.com:8443/ws/v5/public", 
            "channel": "books5",
            "params": {"instId": "SOL-USDT"}
        },
        {
            "name": "Business - candle1m",
            "url": "wss://ws.okx.com:8443/ws/v5/business",
            "channel": "candle1m", 
            "params": {"instId": "SOL-USDT"}
        },
        {
            "name": "Business - mark-price",
            "url": "wss://ws.okx.com:8443/ws/v5/business",
            "channel": "mark-price",
            "params": {"instId": "SOL-USDT"}
        }
    ]
    
    for config in test_configs:
        logger.info(f"\n--- Testing: {config['name']} ---")
        
        try:
            # Create client
            client = OKXWebSocketClient(api_key, secret_key, passphrase, data_queue)
            client.ws_url = config['url']
            
            # Connect
            await client.connect()
            logger.info(f"✓ Connected to {config['url']}")
            
            # Subscribe
            subscription = {
                "channel": config['channel'],
                **config['params']
            }
            
            await client.subscribe([subscription])
            logger.info(f"→ Subscription sent: {subscription}")
            
            # Wait for response
            await asyncio.sleep(2)
            
            # Close connection
            if client.ws:
                await client.ws.close()
                
        except Exception as e:
            logger.error(f"✗ Error with {config['name']}: {e}")
            
        await asyncio.sleep(1)
    
    logger.info("\n=== CHANNEL TESTING COMPLETED ===")

if __name__ == "__main__":
    asyncio.run(test_channel_variations())