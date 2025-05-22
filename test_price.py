"""
Script para probar la conexión con OKX y obtener precios reales
"""
import logging
import sys
from data_management.market_data import get_current_price, initialize_exchange

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("PriceTest")

def test_okx_connection():
    """Prueba la conexión directa con OKX usando ccxt"""
    print("\nProbando conexión directa con OKX...")
    
    exchange = initialize_exchange(test=False)
    
    if exchange:
        try:
            # Probar obtener ticker directamente
            ticker = exchange.fetch_ticker('SOL/USDT')
            print(f"Conexión exitosa a OKX!")
            print(f"Precio actual de Solana (SOL/USDT): ${ticker['last']}")
            print(f"Volumen 24h: {ticker['quoteVolume']}")
            print(f"Variación 24h: {ticker['percentage']}%")
            return True
        except Exception as e:
            print(f"Error al obtener datos de OKX: {e}")
            return False
    else:
        print("No se pudo inicializar la conexión con OKX")
        return False

def test_price_function():
    """Prueba la función get_current_price del sistema"""
    print("\nProbando función get_current_price...")
    
    try:
        # Probar obtener precio a través de la función del sistema
        price = get_current_price("SOL-USDT")
        print(f"Precio de Solana obtenido: ${price}")
        return True
    except Exception as e:
        print(f"Error al obtener precio con get_current_price: {e}")
        return False

if __name__ == "__main__":
    print("====== PRUEBA DE PRECIOS DE SOLANA ======")
    
    # Probar conexión directa
    test_okx_connection()
    
    # Probar función del sistema
    test_price_function()
    
    print("\n======= FIN DE PRUEBA ========")