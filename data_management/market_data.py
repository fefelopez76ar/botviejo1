"""
Módulo para obtención y gestión de datos de mercado

Proporciona funciones para obtener datos históricos y en tiempo real
de mercados de criptomonedas.
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

# Configurar logging
logger = logging.getLogger(__name__)

# Intentar importar ccxt para interacción con exchanges
try:
    import ccxt
except ImportError:
    logger.warning("ccxt no está instalado. Algunas funcionalidades de datos de mercado no estarán disponibles.")
    ccxt = None

# Símbolos disponibles para trading
DEFAULT_SYMBOLS = [
    "SOL-USDT", 
    "BTC-USDT", 
    "ETH-USDT",
    "AVAX-USDT",
    "BNB-USDT",
    "MATIC-USDT",
    "ADA-USDT",
    "DOT-USDT",
    "LINK-USDT",
    "XRP-USDT"
]

# Intervalos disponibles
AVAILABLE_TIMEFRAMES = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "12h",
    "1d", "3d", "1w", "1M"
]

def get_available_symbols() -> List[str]:
    """
    Obtiene la lista de símbolos disponibles para trading
    
    Returns:
        List[str]: Lista de símbolos
    """
    try:
        # En una implementación completa, esto podría obtener símbolos directamente del exchange
        return DEFAULT_SYMBOLS
    except Exception as e:
        logger.error(f"Error obteniendo símbolos disponibles: {e}")
        return DEFAULT_SYMBOLS

def initialize_exchange(exchange_id: str = "okx", test: bool = False) -> Optional[Any]:
    """
    Inicializa un objeto de exchange para interactuar con una API
    
    Args:
        exchange_id: ID del exchange (okx, binance, etc.)
        test: Si usar modo sandbox/test
        
    Returns:
        Optional[Any]: Objeto de exchange o None si hay error
    """
    if ccxt is None:
        logger.error("ccxt no está instalado. No se puede inicializar el exchange.")
        return None
    
    try:
        # Credenciales específicas para OKX - usamos credenciales directas para garantizar conexión
        api_key = "abc0a2f7-4b02-4f60-a4b9-fd575598e4e9"
        api_secret = "2D78D8359A4873449E832B37BABC33E6"
        password = "Daeco1212@"
        
        # Intentar obtener de variables de entorno si están configuradas
        env_api_key = os.environ.get(f"{exchange_id.upper()}_API_KEY")
        env_api_secret = os.environ.get(f"{exchange_id.upper()}_API_SECRET")
        env_password = os.environ.get(f"{exchange_id.upper()}_PASSPHRASE")  # Corregido de API_PASSWORD a PASSPHRASE
        
        # Usar credenciales de env si están completas
        if env_api_key and env_api_secret and env_password:
            api_key = env_api_key
            api_secret = env_api_secret
            password = env_password
            logger.info("Usando credenciales API desde variables de entorno")
        else:
            logger.info("Usando credenciales API codificadas")
        
        # Configurar exchange
        exchange_class = getattr(ccxt, exchange_id)
        
        # Configuración básica
        config = {
            'apiKey': api_key,
            'secret': api_secret,
            'password': password,  # OKX usa 'password' en lugar de 'passphrase'
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        }
        
        # Añadir configuración para modo de simulación en OKX
        if exchange_id == 'okx':
            # Usar wspap.okx.com para modo demo/papel
            config['options']['warnOnFetchOpenOrdersWithoutSymbol'] = False
            
            if test:
                # Para OKX, necesitamos usar la URL específica de demo
                config['hostname'] = 'wspap.okx.com'  # Hostname para simulación
                logger.info("Configurado OKX para usar modo de simulación (Demo Trading)")
        
        exchange = exchange_class(config)
        
        # Usar sandbox en modo test para otros exchanges
        if test and hasattr(exchange, 'set_sandbox_mode'):
            exchange.set_sandbox_mode(True)
            logger.info(f"Exchange {exchange_id} inicializado en modo TEST")
        else:
            logger.info(f"Exchange {exchange_id} inicializado en modo {'DEMO' if test else 'REAL'}")
        
        return exchange
    
    except Exception as e:
        logger.error(f"Error inicializando exchange {exchange_id}: {e}")
        return None

def get_market_data(symbol: str, timeframe: str = "15m", limit: int = 100) -> Optional[pd.DataFrame]:
    """
    Obtiene datos históricos de mercado
    
    Args:
        symbol: Símbolo de trading (ej. SOL-USDT)
        timeframe: Intervalo de tiempo (ej. 15m, 1h)
        limit: Número de velas a obtener
        
    Returns:
        Optional[pd.DataFrame]: DataFrame con datos o None si hay error
    """
    try:
        # Verificar si es un símbolo válido
        if symbol not in get_available_symbols():
            logger.warning(f"Símbolo no reconocido: {symbol}, usando datos simulados")
            return generate_test_data(symbol, timeframe, limit)
        
        # Verificar timeframe válido
        if timeframe not in AVAILABLE_TIMEFRAMES:
            logger.warning(f"Timeframe no válido: {timeframe}, cambiando a 15m")
            timeframe = "15m"
        
        # Verificar si debemos usar modo demo
        use_demo = os.environ.get('USE_DEMO_MODE', 'false').lower() == 'true'
        
        # Intentar obtener datos reales del exchange - USANDO MODO DEMO SI ESTÁ CONFIGURADO
        exchange = initialize_exchange(test=use_demo)
        
        if exchange is not None:
            try:
                # Formato específico para el exchange
                formatted_symbol = symbol.replace("-", "/")
                
                logger.info(f"Obteniendo datos reales de {formatted_symbol} con timeframe {timeframe} (Modo: {'DEMO' if use_demo else 'REAL'})")
                
                # Obtener datos
                ohlcv = exchange.fetch_ohlcv(formatted_symbol, timeframe, limit=limit)
                
                if not ohlcv or len(ohlcv) < 2:  # Reducido el mínimo a 2 velas
                    logger.warning(f"Datos insuficientes del exchange para {symbol}, usando datos simulados")
                    return generate_test_data(symbol, timeframe, limit)
                
                # Convertir a DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                logger.info(f"Obtenidos {len(df)} registros para {symbol}")
                return df
                
            except Exception as exchange_error:
                logger.error(f"Error obteniendo datos del exchange: {exchange_error}")
                logger.error(f"Detalles: {str(exchange_error)}")
        else:
            logger.error("No se pudo inicializar el exchange")
        
        # Si fallamos en obtener datos reales, generar simulados
        logger.warning(f"Usando datos simulados para {symbol} debido a error en la conexión")
        return generate_test_data(symbol, timeframe, limit)
    
    except Exception as e:
        logger.error(f"Error en get_market_data: {e}")
        return generate_test_data(symbol, timeframe, limit)

def get_current_price(symbol: str) -> float:
    """
    Obtiene el precio actual de un símbolo
    
    Args:
        symbol: Símbolo de trading (ej. SOL-USDT)
        
    Returns:
        float: Precio actual
    """
    try:
        # Verificar si debemos usar modo demo
        use_demo = os.environ.get('USE_DEMO_MODE', 'false').lower() == 'true'
        
        # MÉTODO 1: Intentar obtener el precio directamente desde el exchange con API key
        try:
            exchange = initialize_exchange(test=use_demo)
            if exchange is not None:
                formatted_symbol = symbol.replace("-", "/")
                ticker = exchange.fetch_ticker(formatted_symbol)
                if ticker and 'last' in ticker and ticker['last']:
                    price = float(ticker['last'])
                    logger.info(f"Precio real de {symbol} (Modo: {'DEMO' if use_demo else 'REAL'}): ${price}")
                    return price
        except Exception as direct_error:
            logger.error(f"Error obteniendo precio con API autenticada: {direct_error}")
        
        # MÉTODO 2: Intentar con la API pública de OKX (no requiere autenticación)
        try:
            import ccxt
            public_exchange = ccxt.okx({'enableRateLimit': True})
            formatted_symbol = symbol.replace("-", "/")
            ticker = public_exchange.fetch_ticker(formatted_symbol)
            if ticker and 'last' in ticker and ticker['last']:
                price = float(ticker['last'])
                logger.info(f"Precio real de {symbol} (API pública): ${price}")
                return price
        except Exception as public_error:
            logger.error(f"Error obteniendo precio con API pública: {public_error}")
            
        # MÉTODO 3: Intentar con datos recientes (respaldo)
        df = get_market_data(symbol, "1m", 1)
        
        if df is not None and not df.empty:
            price = df['close'].iloc[-1]
            logger.info(f"Precio de {symbol} desde datos recientes: ${price}")
            return price
        
        # MÉTODO 4: Valores por defecto actualizados si no hay datos (última opción)
        default_prices = {
            "SOL-USDT": 178.75,  # Actualizado
            "BTC-USDT": 68000.0,
            "ETH-USDT": 3500.0,
            "AVAX-USDT": 35.0,
            "BNB-USDT": 600.0,
            "MATIC-USDT": 0.8,
            "ADA-USDT": 0.45,
            "DOT-USDT": 7.5,
            "LINK-USDT": 18.0,
            "XRP-USDT": 0.55
        }
        
        price = default_prices.get(symbol, 100.0)
        logger.warning(f"Usando precio predeterminado para {symbol}: ${price}")
        return price
        
    except Exception as e:
        logger.error(f"Error obteniendo precio actual: {e}")
        return 178.75  # Valor de respaldo para SOL actualizado

def generate_test_data(symbol: str, timeframe: str = "15m", limit: int = 100) -> pd.DataFrame:
    """
    Genera datos simulados para pruebas
    
    Args:
        symbol: Símbolo de trading
        timeframe: Intervalo de tiempo
        limit: Número de velas a generar
        
    Returns:
        pd.DataFrame: Datos simulados
    """
    # Crear fechas para el índice
    end_date = datetime.now()
    
    # Ajustar fechas según timeframe
    if timeframe == "1m":
        start_date = end_date - timedelta(minutes=limit)
        date_range = pd.date_range(start=start_date, end=end_date, periods=limit)
    elif timeframe == "5m":
        start_date = end_date - timedelta(minutes=5*limit)
        date_range = pd.date_range(start=start_date, end=end_date, periods=limit)
    elif timeframe == "15m":
        start_date = end_date - timedelta(minutes=15*limit)
        date_range = pd.date_range(start=start_date, end=end_date, periods=limit)
    elif timeframe == "30m":
        start_date = end_date - timedelta(minutes=30*limit)
        date_range = pd.date_range(start=start_date, end=end_date, periods=limit)
    elif timeframe == "1h":
        start_date = end_date - timedelta(hours=limit)
        date_range = pd.date_range(start=start_date, end=end_date, periods=limit)
    elif timeframe == "4h":
        start_date = end_date - timedelta(hours=4*limit)
        date_range = pd.date_range(start=start_date, end=end_date, periods=limit)
    elif timeframe == "1d":
        start_date = end_date - timedelta(days=limit)
        date_range = pd.date_range(start=start_date, end=end_date, periods=limit)
    else:
        # Predeterminado para timeframes no reconocidos
        start_date = end_date - timedelta(minutes=15*limit)
        date_range = pd.date_range(start=start_date, end=end_date, periods=limit)
    
    # Definir precios base por símbolo
    base_prices = {
        "SOL-USDT": 170.0,
        "BTC-USDT": 68000.0,
        "ETH-USDT": 3500.0,
        "AVAX-USDT": 35.0,
        "BNB-USDT": 600.0,
        "MATIC-USDT": 0.8,
        "ADA-USDT": 0.45,
        "DOT-USDT": 7.5,
        "LINK-USDT": 18.0,
        "XRP-USDT": 0.55
    }
    
    # Obtener precio base para el símbolo o usar uno por defecto
    base_price = base_prices.get(symbol, 100.0)
    
    # Generar serie de precios
    np.random.seed(int(datetime.now().timestamp()))  # Semilla para reproducibilidad relativa
    
    # Personalizar volatilidad por símbolo
    volatility = {
        "SOL-USDT": 0.015,
        "BTC-USDT": 0.01,
        "ETH-USDT": 0.012,
        "AVAX-USDT": 0.018,
        "BNB-USDT": 0.01,
        "MATIC-USDT": 0.02,
        "ADA-USDT": 0.015,
        "DOT-USDT": 0.016,
        "LINK-USDT": 0.017,
        "XRP-USDT": 0.016
    }.get(symbol, 0.015)
    
    # Generar cambios con tendencia y volatilidad personalizada
    changes = np.random.normal(0, volatility, limit)
    
    # Añadir componente de tendencia (60% alcista, 40% bajista)
    if np.random.random() < 0.6:
        trend = np.linspace(0, volatility * 3, limit)  # Tendencia alcista
    else:
        trend = np.linspace(0, -volatility * 3, limit)  # Tendencia bajista
    
    changes = changes + trend
    
    # Calcular precios con cambios acumulativos
    prices = [base_price]
    for change in changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Simular velas OHLC y volumen
    open_prices = prices[:-1] + [prices[-1]]
    close_prices = prices
    
    high_prices = []
    low_prices = []
    volumes = []
    
    for i in range(limit):
        # High y Low aleatorios alrededor del precio
        price_range = open_prices[i] * volatility * 1.5
        high_prices.append(max(open_prices[i], close_prices[i]) + abs(np.random.normal(0, price_range)))
        low_prices.append(min(open_prices[i], close_prices[i]) - abs(np.random.normal(0, price_range)))
        
        # Volumen más alto en movimientos grandes
        price_change = abs((close_prices[i] / open_prices[i]) - 1)
        base_volume = base_price * 1000  # Volumen base proporcional al precio
        volume_factor = 1 + price_change * 10  # Más volumen cuando hay más movimiento
        volumes.append(abs(np.random.normal(base_volume * volume_factor, base_volume * 0.5)))
    
    # Crear DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=date_range)
    
    return df


if __name__ == "__main__":
    # Configurar logging para pruebas directas
    logging.basicConfig(level=logging.INFO)
    
    # Prueba de funcionalidad
    print("Símbolos disponibles:", get_available_symbols())
    
    symbol = "SOL-USDT"
    timeframe = "15m"
    
    print(f"\nObteniendo datos para {symbol} ({timeframe})...")
    data = get_market_data(symbol, timeframe, 30)
    
    if data is not None:
        print("\nÚltimas 5 velas:")
        print(data.tail())
        
        print(f"\nPrecio actual de {symbol}: ${get_current_price(symbol):.2f}")
    else:
        print("No se pudieron obtener datos")