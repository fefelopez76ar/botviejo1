#!/usr/bin/env python3
"""
Script para probar el modo de simulación en OKX
Este script verifica la conexión y funcionalidad del modo demo trading en OKX.
"""

import os
import sys
import time
import ccxt
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("TestDemoMode")

# Cargar variables de entorno
load_dotenv('config.env')

def main():
    """Función principal para probar el modo demo"""
    print("\n=== PRUEBA DE MODO SIMULACIÓN EN OKX ===\n")
    
    # Obtener credenciales
    api_key = os.environ.get('OKX_API_KEY', '')
    api_secret = os.environ.get('OKX_API_SECRET', '')
    passphrase = os.environ.get('OKX_PASSPHRASE', '')
    use_demo = os.environ.get('USE_DEMO_MODE', 'true').lower() == 'true'
    
    print(f"Modo de simulación activado: {use_demo}")
    
    # 1. Probar conexión con API pública
    print("\n1. Probando conexión con API pública...")
    try:
        # Configurar cliente OKX público
        public_exchange = ccxt.okx({
            'enableRateLimit': True
        })
        
        # Obtener precio de Solana
        ticker = public_exchange.fetch_ticker('SOL/USDT')
        print(f"✓ Conexión con API pública exitosa")
        print(f"Precio actual de Solana (SOL/USDT): ${ticker['last']}")
        print(f"Volumen 24h: {ticker['quoteVolume']}")
        print(f"Variación 24h: {ticker['percentage']}%")
        
    except Exception as e:
        print(f"✗ Error al conectar con API pública: {e}")
    
    # 2. Probar conexión con API en modo demo
    print("\n2. Probando conexión con API en modo demo...")
    try:
        # Configuración para modo demo
        config = {
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'warnOnFetchOpenOrdersWithoutSymbol': False
            }
        }
        
        # Ajustar para modo demo si está activado
        if use_demo:
            config['hostname'] = 'wspap.okx.com'  # Hostname para simulación
            print("Configurado para usar OKX Demo Trading API")
        
        # Crear exchange
        exchange = ccxt.okx(config)
        
        # Verificar conexión
        print("\nVerificando estado de la cuenta...")
        balance = exchange.fetch_balance()
        
        if balance:
            print("✓ Conexión exitosa con cuenta en modo simulación")
            print("\nSaldos disponibles:")
            
            # Mostrar saldos de las principales monedas
            for currency in ['USDT', 'BTC', 'ETH', 'SOL']:
                if currency in balance['free']:
                    amount = balance['free'][currency]
                    print(f"  • {currency}: {amount}")
            
            # Verificar mercados disponibles
            print("\nMercados disponibles para SOL:")
            markets = exchange.load_markets()
            sol_markets = [market for market in markets.keys() if 'SOL/' in market]
            for market in sol_markets[:5]:  # Mostrar solo los primeros 5
                print(f"  • {market}")
            
            # Intentar crear una orden de prueba en modo simulación
            if use_demo:
                print("\nIntentando crear orden de prueba (simulada)...")
                try:
                    order = exchange.create_limit_buy_order(
                        symbol='SOL/USDT',
                        amount=0.1,  # Cantidad pequeña
                        price=150.0  # Precio por debajo del mercado
                    )
                    print("✓ Orden creada exitosamente:")
                    print(f"  • ID: {order.get('id', 'N/A')}")
                    print(f"  • Tipo: {order.get('type', 'N/A')}")
                    print(f"  • Lado: {order.get('side', 'N/A')}")
                    print(f"  • Precio: {order.get('price', 'N/A')}")
                    print(f"  • Cantidad: {order.get('amount', 'N/A')}")
                    print(f"  • Estado: {order.get('status', 'N/A')}")
                    
                    # Cancelar la orden
                    print("\nCancelando orden...")
                    result = exchange.cancel_order(order['id'], 'SOL/USDT')
                    print("✓ Orden cancelada exitosamente")
                    
                except Exception as order_error:
                    print(f"✗ Error al crear orden: {order_error}")
                    if "insufficient balance" in str(order_error).lower():
                        print("  • La cuenta demo no tiene suficiente balance, esto es normal")
                        print("  • Puedes solicitar fondos demo en la plataforma OKX")
        else:
            print("✗ No se pudieron obtener los saldos de la cuenta")
        
    except Exception as e:
        print(f"✗ Error al conectar con API en modo demo: {e}")
        if "IP whitelist" in str(e):
            print("\nProblema con la lista blanca de IP:")
            print("1. Inicia sesión en OKX")
            print("2. Ve a 'API Management'")
            print("3. Selecciona tu API key")
            print("4. Agrega la IP de este servidor a la lista blanca o usa '0.0.0.0/0' para pruebas")
    
    print("\n=== PRUEBA COMPLETADA ===")

if __name__ == "__main__":
    main()