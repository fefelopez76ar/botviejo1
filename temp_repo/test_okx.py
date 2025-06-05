import ccxt
import os

print("Verificando conexión con OKX...")

try:
    # Usar las credenciales de config.env
    api_key = "abc0a2f7-4b02-4f60-a4b9-fd575598e4e9"
    api_secret = "2D78D8359A4873449E832B37BABC33E6"
    password = "Daeco1212@"
    
    # Configurar conexión
    exchange = ccxt.okx({
        'apiKey': api_key,
        'secret': api_secret,
        'password': password,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'test': True  # Modo DEMO
        }
    })
    
    print("Conexión establecida con OKX")
    
    # Verificar precio de Solana
    ticker = exchange.fetch_ticker('SOL/USDT')
    price = ticker['last']
    print(f"Precio actual de Solana (SOL/USDT): ${price}")
    
    # Verificar balance
    balance = exchange.fetch_balance()
    usdt_balance = balance.get('total', {}).get('USDT', 0)
    print(f"Balance de USDT: ${usdt_balance}")
    
    print("✅ Conexión con OKX funcionando correctamente")
    
except Exception as e:
    print(f"❌ Error conectando con OKX: {e}")
    print("Por favor verifica tus credenciales API y conexión a internet")