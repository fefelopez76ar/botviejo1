"""
Verificador de precios para Solana desde OKX
"""
import ccxt
import time

print("Obteniendo precio de Solana desde OKX (API pública)...")

# Usar API pública (sin autenticación)
try:
    exchange = ccxt.okx({'enableRateLimit': True})
    ticker = exchange.fetch_ticker('SOL/USDT')
    
    print(f"¡Conexión exitosa!")
    print(f"Precio actual de Solana (SOL/USDT): ${ticker['last']}")
    print(f"Volumen 24h: {ticker['quoteVolume']}")
    print(f"Variación 24h: {ticker['percentage']}%")
    
except Exception as e:
    print(f"Error al conectar con OKX: {e}")

# Intentar ahora con las credenciales proporcionadas
print("\nIntentando con API key (si aún no has agregado la IP a la lista blanca, esto fallará)...")
time.sleep(1)

try:
    authenticated_exchange = ccxt.okx({
        'apiKey': "abc0a2f7-4b02-4f60-a4b9-fd575598e4e9",
        'secret': "2D78D8359A4873449E832B37BABC33E6",
        'password': "Daeco1212@",
        'enableRateLimit': True
    })
    
    ticker = authenticated_exchange.fetch_ticker('SOL/USDT')
    print(f"¡Conexión con autenticación exitosa!")
    print(f"Precio actual de Solana (SOL/USDT): ${ticker['last']}")
    
except Exception as e:
    print(f"Error al conectar con API autenticada: {e}")
    print("Si ves un error de lista blanca de IP, necesitas agregar la IP de Replit a tu configuración OKX")