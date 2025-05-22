import os
import ccxt
import json

# Usar las credenciales desde variables de entorno o archivo config.env
api_key = "abc0a2f7-4b02-4f60-a4b9-fd575598e4e9"
api_secret = "2D78D8359A4873449E832B37BABC33E6"
passphrase = "Daeco1212@"

print("Intentando conectar a OKX...")

try:
    # Configurar cliente OKX
    exchange = ccxt.okx({
        'apiKey': api_key,
        'secret': api_secret,
        'password': passphrase,  # OKX usa 'password' en lugar de 'passphrase'
        'enableRateLimit': True
    })
    
    # Obtener precio de Solana
    ticker = exchange.fetch_ticker('SOL/USDT')
    print(f"Conexión exitosa!")
    print(f"Precio actual de Solana (SOL/USDT): ${ticker['last']}")
    
    # Mostrar otros datos útiles
    print(f"Volumen 24h: {ticker['quoteVolume']}")
    print(f"Variación 24h: {ticker['percentage']}%")
    
except Exception as e:
    print(f"Error al conectar con OKX: {e}")