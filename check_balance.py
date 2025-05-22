import os
import ccxt
import dotenv

# Cargar variables de entorno
dotenv.load_dotenv('config.env')

print("Consultando saldo en OKX...")

# Configurar exchange
try:
    exchange = ccxt.okx({
        'apiKey': os.environ.get('OKX_API_KEY'),
        'secret': os.environ.get('OKX_API_SECRET'),
        'password': os.environ.get('OKX_PASSPHRASE'),
        'enableRateLimit': True
    })
    
    # Configurar modo demo si está habilitado
    if os.environ.get('USE_DEMO_MODE') == 'true':
        print("Modo de simulación (Demo Trading) activado")
        exchange.set_sandbox_mode(True)
    
    # Obtener balance
    balance = exchange.fetch_balance()
    
    # Mostrar saldo total
    print("\nSALDO EN CUENTA OKX:")
    if 'total' in balance:
        for currency, amount in balance['total'].items():
            if amount > 0:
                print(f"{currency}: {amount}")
    else:
        print("No se encontró información de saldo")
        
except Exception as e:
    print(f"Error al consultar saldo: {e}")
    print("\nPosibles causas:")
    print("1. Credenciales API incorrectas o no configuradas")
    print("2. IP no autorizada en la lista blanca de OKX")
    print("3. Conexión a internet fallida")
