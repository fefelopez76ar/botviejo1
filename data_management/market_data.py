"""
Módulo para gestión de datos de mercado
Proporciona funciones para obtener y procesar datos de mercado de diferentes exchanges
"""

import os
import json
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

logger = logging.getLogger("MarketData")

def get_available_symbols() -> List[str]:
    """
    Obtiene los símbolos disponibles para operar
    
    Returns:
        List[str]: Lista de símbolos disponibles
    """
    # En una implementación real, esto obtendría los pares desde el exchange
    # Para demostración, retornamos una lista predefinida
    return [
        "SOL-USDT",
        "BTC-USDT",
        "ETH-USDT",
        "AVAX-USDT",
        "ADA-USDT"
    ]

def get_market_data(symbol: str, interval: str = "15m", limit: int = 100) -> pd.DataFrame:
    """
    Obtiene datos históricos del mercado
    
    Args:
        symbol: Par de trading
        interval: Intervalo de tiempo
        limit: Número de velas a obtener
        
    Returns:
        pd.DataFrame: DataFrame con datos históricos (OHLCV)
    """
    # En una implementación real, esto obtendría datos desde el exchange
    # Para demostración, generamos datos aleatorios
    
    # Fechas para el rango (desde ahora hacia atrás)
    end_time = datetime.now()
    
    # Determinar duración del intervalo en minutos
    interval_minutes = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440
    }.get(interval, 15)
    
    # Calcular inicio
    start_time = end_time - timedelta(minutes=interval_minutes * limit)
    
    # Generar fechas
    dates = []
    current = start_time
    while current <= end_time:
        dates.append(current)
        current += timedelta(minutes=interval_minutes)
    
    # Limitar a 'limit' elementos
    dates = dates[:limit]
    
    # Generar datos aleatorios basados en un precio base según el símbolo
    base_prices = {
        "SOL-USDT": 150.0,
        "BTC-USDT": 65000.0,
        "ETH-USDT": 3500.0,
        "AVAX-USDT": 35.0,
        "ADA-USDT": 0.45
    }
    
    base_price = base_prices.get(symbol, 100.0)
    volatility = base_price * 0.01  # 1% de volatilidad
    
    # Generar precios con tendencia (browniano)
    prices = [base_price]
    for i in range(1, limit):
        # Movimiento browniano con tendencia
        change = np.random.normal(0, 1) * volatility
        prices.append(prices[-1] + change)
    
    # Genera OHLCV
    data = []
    for i, date in enumerate(dates):
        price = prices[i]
        
        # Generar OHLC con dispersión alrededor del precio de cierre
        high = price * (1 + abs(np.random.normal(0, 0.003)))
        low = price * (1 - abs(np.random.normal(0, 0.003)))
        open_price = low + (high - low) * np.random.random()
        close = price
        
        # Volumen aleatorio
        volume = base_price * 1000 * (0.5 + np.random.random())
        
        data.append([date, open_price, high, low, close, volume])
    
    # Crear DataFrame
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.set_index("timestamp", inplace=True)
    
    return df

def get_current_price(symbol: str) -> float:
    """
    Obtiene el precio actual del mercado
    
    Args:
        symbol: Par de trading
        
    Returns:
        float: Precio actual
    """
    # En una implementación real, esto obtendría el precio desde el exchange
    try:
        # Para demo, usar endpoint público de OKX
        import requests
        
        # Extraer ticker base
        if "-" in symbol:
            base_currency = symbol.split("-")[0]
        else:
            base_currency = "SOL"
            
        ticker = f"{base_currency}-USDT"
        
        url = f"https://www.okx.com/api/v5/market/ticker?instId={ticker}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == "0" and "data" in data:
                price = float(data["data"][0]["last"])
                logger.info(f"Precio obtenido: {price}")
                return price
        
        # Si falla, retornar precio simulado
        return simulate_price(symbol)
    
    except Exception as e:
        logger.error(f"Error al obtener precio de {symbol}: {e}")
        return simulate_price(symbol)

def simulate_price(symbol: str) -> float:
    """
    Simula un precio para demostración
    
    Args:
        symbol: Par de trading
        
    Returns:
        float: Precio simulado
    """
    base_prices = {
        "SOL-USDT": 150.0,
        "BTC-USDT": 65000.0,
        "ETH-USDT": 3500.0,
        "AVAX-USDT": 35.0,
        "ADA-USDT": 0.45
    }
    
    base_price = base_prices.get(symbol, 100.0)
    variation = np.random.uniform(-0.01, 0.01) * base_price  # ±1%
    
    return base_price + variation

def get_order_book(symbol: str, depth: int = 10) -> Dict[str, List]:
    """
    Obtiene el libro de órdenes actual
    
    Args:
        symbol: Par de trading
        depth: Profundidad del libro
        
    Returns:
        Dict: Libro de órdenes (bids y asks)
    """
    # En una implementación real, esto obtendría el libro desde el exchange
    # Para demostración, generamos datos aleatorios
    
    # Obtener precio actual (simulado)
    current_price = get_current_price(symbol)
    
    # Generar bids (ordenes de compra, por debajo del precio)
    bids = []
    for i in range(depth):
        price = current_price * (1 - 0.0005 * (i + 1))
        size = np.random.uniform(0.1, 10.0) * 10
        bids.append([price, size])
    
    # Generar asks (ordenes de venta, por encima del precio)
    asks = []
    for i in range(depth):
        price = current_price * (1 + 0.0005 * (i + 1))
        size = np.random.uniform(0.1, 10.0) * 10
        asks.append([price, size])
    
    return {
        "bids": bids,
        "asks": asks,
        "timestamp": datetime.now().timestamp()
    }

def get_ticker(symbol: str) -> Dict[str, Any]:
    """
    Obtiene información resumida del ticker
    
    Args:
        symbol: Par de trading
        
    Returns:
        Dict: Información del ticker
    """
    # En una implementación real, esto obtendría datos desde el exchange
    # Para demostración, generamos datos aleatorios
    
    current_price = get_current_price(symbol)
    
    return {
        "symbol": symbol,
        "last": current_price,
        "bid": current_price * 0.9998,
        "ask": current_price * 1.0002,
        "high": current_price * 1.01,
        "low": current_price * 0.99,
        "volume": current_price * 10000,
        "change": np.random.uniform(-1.0, 1.0),
        "timestamp": datetime.now().timestamp()
    }

def save_market_data(df: pd.DataFrame, symbol: str, interval: str) -> bool:
    """
    Guarda datos de mercado en archivo
    
    Args:
        df: DataFrame con datos
        symbol: Par de trading
        interval: Intervalo de tiempo
        
    Returns:
        bool: True si se guardó correctamente, False en caso contrario
    """
    try:
        # Crear directorio si no existe
        os.makedirs("data/market", exist_ok=True)
        
        # Nombre de archivo
        file_name = f"data/market/{symbol}_{interval}.csv"
        
        # Guardar
        df.to_csv(file_name)
        
        logger.info(f"Datos guardados en {file_name}")
        return True
    except Exception as e:
        logger.error(f"Error al guardar datos: {e}")
        return False

def load_market_data(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """
    Carga datos de mercado desde archivo
    
    Args:
        symbol: Par de trading
        interval: Intervalo de tiempo
        
    Returns:
        Optional[pd.DataFrame]: DataFrame con datos o None si no existe
    """
    file_name = f"data/market/{symbol}_{interval}.csv"
    
    if os.path.exists(file_name):
        try:
            df = pd.read_csv(file_name, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
    
    return None

def update_market_data(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """
    Actualiza datos de mercado (carga existentes y añade nuevos)
    
    Args:
        symbol: Par de trading
        interval: Intervalo de tiempo
        limit: Número máximo de velas a mantener
        
    Returns:
        pd.DataFrame: DataFrame con datos actualizados
    """
    # Cargar datos existentes
    existing_data = load_market_data(symbol, interval)
    
    # Obtener datos nuevos
    new_data = get_market_data(symbol, interval, 100)  # Últimas 100 velas
    
    if existing_data is not None:
        # Combinar datos
        combined_data = pd.concat([existing_data, new_data])
        
        # Eliminar duplicados
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        
        # Ordenar por fecha
        combined_data.sort_index(inplace=True)
        
        # Limitar a 'limit' filas
        if len(combined_data) > limit:
            combined_data = combined_data.iloc[-limit:]
    else:
        combined_data = new_data
    
    # Guardar datos actualizados
    save_market_data(combined_data, symbol, interval)
    
    return combined_data