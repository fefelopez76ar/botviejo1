import os
import ccxt
import json
import time

print("Test de conexión OKX para el bot de trading de Solana")
print("------------------------------------------------------")

# 1. Probar conexión con API pública (no requiere autenticación)
print("\n1. Intentando obtener precios con API pública de OKX...")

try:
    # Configurar cliente OKX público (sin autenticación)
    public_exchange = ccxt.okx({
        'enableRateLimit': True
    })
    
    # Obtener precio de Solana
    ticker = public_exchange.fetch_ticker('SOL/USDT')
    print(f"✓ ¡Conexión pública exitosa!")
    print(f"Precio actual de Solana (SOL/USDT): ${ticker['last']}")
    print(f"Volumen 24h: {ticker['quoteVolume']}")
    print(f"Variación 24h: {ticker['percentage']}%")
    
except Exception as e:
    print(f"✗ Error al conectar con API pública: {e}")

# 2. Probar conexión con API autenticada
print("\n2. Intentando conectar con API autenticada de OKX...")
time.sleep(1)

# Usar las credenciales desde variables de entorno o archivo config.env
api_key = "abc0a2f7-4b02-4f60-a4b9-fd575598e4e9"
api_secret = "2D78D8359A4873449E832B37BABC33E6"
passphrase = "Daeco1212@"

try:
    # Configurar cliente OKX en modo simulación
    exchange = ccxt.okx({
        'apiKey': api_key,
        'secret': api_secret,
        'password': passphrase,  # OKX usa 'password' en lugar de 'passphrase'
        'enableRateLimit': True,
        'hostname': 'wspap.okx.com',  # Usar el hostname de la API de simulación
        'options': {
            'defaultType': 'spot',
            'warnOnFetchOpenOrdersWithoutSymbol': False
        }
    })
    print("Modo de simulación activado - usando API Demo Trading de OKX")
    
    # Obtener precio de Solana
    ticker = exchange.fetch_ticker('SOL/USDT')
    print(f"✓ ¡Conexión con autenticación exitosa!")
    print(f"Precio actual de Solana (SOL/USDT): ${ticker['last']}")
    print(f"Volumen 24h: {ticker['quoteVolume']}")
    print(f"Variación 24h: {ticker['percentage']}%")
    
except Exception as e:
    print(f"✗ Error al conectar con API autenticada: {e}")
    
    if "IP whitelist" in str(e):
        print("\nSolución: Necesitas agregar la IP de este servidor a la lista blanca de tu API OKX:")
        print("1. Inicia sesión en OKX")
        print("2. Ve a 'API Management'")
        print("3. Selecciona tu API key que termina en 'e4e9'")
        print("4. Agrega la IP '34.53.73.60' a la lista blanca")
        print("5. O para pruebas, puedes usar '0.0.0.0/0' (permite todas las IPs)")
    
print("\n------------------------------------------------------")