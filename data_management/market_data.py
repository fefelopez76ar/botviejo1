#!/usr/bin/env python3
"""
Módulo para gestión de datos de mercado, incluyendo:
- Obtención de datos en tiempo real de varios exchanges
- Almacenamiento y caché de datos históricos
- Preprocesamiento de datos para análisis
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MarketData')

class MarketData:
    """
    Gestor de datos de mercado para obtener y procesar datos en tiempo real
    e históricos para el análisis y trading.
    """
    
    def __init__(self, 
               exchange: str = 'okx',
               cache_dir: str = 'data/market_cache',
               cache_expiry_hours: Dict[str, int] = None):
        """
        Inicializa el gestor de datos de mercado.
        
        Args:
            exchange: Exchange a utilizar ('okx', 'binance', etc.)
            cache_dir: Directorio para caché de datos
            cache_expiry_hours: Horas de expiración de caché por timeframe
        """
        self.exchange = exchange.lower()
        self.cache_dir = cache_dir
        
        # Configurar expiración de caché por timeframe
        self.cache_expiry_hours = cache_expiry_hours or {
            "1m": 24,    # Datos de 1 minuto expiran en 24 horas
            "5m": 48,    # Datos de 5 minutos expiran en 48 horas
            "15m": 72,   # Datos de 15 minutos expiran en 72 horas
            "30m": 96,   # Datos de 30 minutos expiran en 96 horas
            "1h": 168,   # Datos de 1 hora expiran en 1 semana
            "4h": 336,   # Datos de 4 horas expiran en 2 semanas
            "1d": 720,   # Datos diarios expiran en 30 días
            "1w": 2160   # Datos semanales expiran en 90 días
        }
        
        # Mapeo de timeframes entre exchanges
        self.timeframe_map = {
            'okx': {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1h": "1H", "4h": "4H", "1d": "1D", "1w": "1W"
            },
            'binance': {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"
            }
        }
        
        # Credenciales API (si se necesitan)
        self.api_key = os.environ.get(f"{exchange.upper()}_API_KEY")
        self.api_secret = os.environ.get(f"{exchange.upper()}_API_SECRET")
        
        # Crear directorio de caché si no existe
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_current_price(self, symbol: str) -> float:
        """
        Obtiene el precio actual de un par.
        
        Args:
            symbol: Símbolo del par (ej. "SOL-USDT")
            
        Returns:
            float: Precio actual o 0 si hay error
        """
        try:
            # Obtener ticker según el exchange
            if self.exchange == 'okx':
                logger.info(f"Obteniendo precio via endpoint público: https://www.okx.com/api/v5/market/ticker")
                response = requests.get(
                    f"https://www.okx.com/api/v5/market/ticker?instId={symbol}"
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("code") == "0" and "data" in data and data["data"]:
                        price = float(data["data"][0].get("last", 0))
                        logger.info(f"Precio obtenido: {price}")
                        return price
            
            elif self.exchange == 'binance':
                response = requests.get(
                    f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.replace('-', '')}"
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return float(data.get("price", 0))
            
            logger.warning(f"No se pudo obtener precio para {symbol}")
            return 0
            
        except Exception as e:
            logger.error(f"Error al obtener precio actual para {symbol}: {e}")
            return 0
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Obtiene datos completos del ticker para un par.
        
        Args:
            symbol: Símbolo del par
            
        Returns:
            Dict[str, Any]: Datos del ticker o diccionario vacío si hay error
        """
        try:
            # Obtener ticker según el exchange
            if self.exchange == 'okx':
                response = requests.get(
                    f"https://www.okx.com/api/v5/market/ticker?instId={symbol}"
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("code") == "0" and "data" in data and data["data"]:
                        ticker_data = data["data"][0]
                        
                        return {
                            "symbol": symbol,
                            "last_price": float(ticker_data.get("last", 0)),
                            "bid_price": float(ticker_data.get("bidPx", 0)),
                            "ask_price": float(ticker_data.get("askPx", 0)),
                            "volume_24h": float(ticker_data.get("vol24h", 0)),
                            "volume_ccy_24h": float(ticker_data.get("volCcy24h", 0)),
                            "open_24h": float(ticker_data.get("open24h", 0)),
                            "high_24h": float(ticker_data.get("high24h", 0)),
                            "low_24h": float(ticker_data.get("low24h", 0)),
                            "timestamp": datetime.now().isoformat()
                        }
            
            elif self.exchange == 'binance':
                response = requests.get(
                    f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol.replace('-', '')}"
                )
                
                if response.status_code == 200:
                    ticker_data = response.json()
                    
                    return {
                        "symbol": symbol,
                        "last_price": float(ticker_data.get("lastPrice", 0)),
                        "bid_price": float(ticker_data.get("bidPrice", 0)),
                        "ask_price": float(ticker_data.get("askPrice", 0)),
                        "volume_24h": float(ticker_data.get("volume", 0)),
                        "quote_volume_24h": float(ticker_data.get("quoteVolume", 0)),
                        "open_24h": float(ticker_data.get("openPrice", 0)),
                        "high_24h": float(ticker_data.get("highPrice", 0)),
                        "low_24h": float(ticker_data.get("lowPrice", 0)),
                        "timestamp": datetime.now().isoformat()
                    }
            
            logger.warning(f"No se pudieron obtener datos del ticker para {symbol}")
            return {}
            
        except Exception as e:
            logger.error(f"Error al obtener ticker para {symbol}: {e}")
            return {}
    
    def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """
        Obtiene el libro de órdenes para un par.
        
        Args:
            symbol: Símbolo del par
            depth: Profundidad del libro de órdenes
            
        Returns:
            Dict[str, Any]: Libro de órdenes o diccionario vacío si hay error
        """
        try:
            # Obtener orderbook según el exchange
            if self.exchange == 'okx':
                response = requests.get(
                    f"https://www.okx.com/api/v5/market/books?instId={symbol}&sz={depth}"
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("code") == "0" and "data" in data and data["data"]:
                        book_data = data["data"][0]
                        
                        return {
                            "symbol": symbol,
                            "bids": book_data.get("bids", []),  # [[precio, cantidad], ...]
                            "asks": book_data.get("asks", []),  # [[precio, cantidad], ...]
                            "timestamp": datetime.now().isoformat()
                        }
            
            elif self.exchange == 'binance':
                response = requests.get(
                    f"https://api.binance.com/api/v3/depth?symbol={symbol.replace('-', '')}&limit={depth}"
                )
                
                if response.status_code == 200:
                    book_data = response.json()
                    
                    return {
                        "symbol": symbol,
                        "bids": book_data.get("bids", []),  # [[precio, cantidad], ...]
                        "asks": book_data.get("asks", []),  # [[precio, cantidad], ...]
                        "timestamp": datetime.now().isoformat()
                    }
            
            logger.warning(f"No se pudo obtener orderbook para {symbol}")
            return {}
            
        except Exception as e:
            logger.error(f"Error al obtener orderbook para {symbol}: {e}")
            return {}
    
    def get_historical_data(self, 
                          symbol: str, 
                          timeframe: str = "1h", 
                          limit: int = 100,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos para un par.
        
        Args:
            symbol: Símbolo del par
            timeframe: Marco temporal ("1m", "5m", "15m", "1h", "4h", "1d", "1w")
            limit: Número máximo de velas a obtener
            start_time: Tiempo de inicio (opcional)
            end_time: Tiempo de fin (opcional)
            use_cache: Si se debe usar la caché
            
        Returns:
            Optional[pd.DataFrame]: DataFrame con datos o None si hay error
        """
        # Verificar si hay datos en caché
        if use_cache:
            cached_data = self._get_cached_data(symbol, timeframe)
            if cached_data is not None:
                # Filtrar según start_time y end_time si se proporcionan
                if start_time:
                    cached_data = cached_data[cached_data.index >= pd.Timestamp(start_time)]
                if end_time:
                    cached_data = cached_data[cached_data.index <= pd.Timestamp(end_time)]
                
                # Si hay suficientes datos en caché, devolver
                if len(cached_data) >= limit:
                    return cached_data.iloc[-limit:]
        
        # Obtener datos desde la API
        try:
            # Convertir timeframe al formato del exchange
            exchange_timeframe = self.timeframe_map.get(self.exchange, {}).get(timeframe, timeframe)
            
            # Obtener datos según el exchange
            if self.exchange == 'okx':
                params = {
                    "instId": symbol,
                    "bar": exchange_timeframe,
                    "limit": str(limit)
                }
                
                # Añadir parámetros de tiempo si se proporcionan
                if start_time:
                    params["after"] = int(start_time.timestamp() * 1000)
                if end_time:
                    params["before"] = int(end_time.timestamp() * 1000)
                
                response = requests.get(
                    "https://www.okx.com/api/v5/market/candles",
                    params=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("code") == "0" and "data" in data:
                        # Procesar datos
                        candles = data["data"]
                        
                        # Formato OKX: [timestamp, open, high, low, close, vol, ...]
                        df = pd.DataFrame(candles, columns=[
                            "timestamp", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote", "confirm"
                        ])
                        
                        # Convertir tipos de datos
                        for col in ["open", "high", "low", "close", "volume"]:
                            df[col] = pd.to_numeric(df[col])
                        
                        # Configurar índice
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
                        df.set_index("timestamp", inplace=True)
                        df.sort_index(inplace=True)
                        
                        # Guardar en caché
                        self._cache_data(df, symbol, timeframe)
                        
                        return df
            
            elif self.exchange == 'binance':
                params = {
                    "symbol": symbol.replace('-', ''),
                    "interval": exchange_timeframe,
                    "limit": limit
                }
                
                # Añadir parámetros de tiempo si se proporcionan
                if start_time:
                    params["startTime"] = int(start_time.timestamp() * 1000)
                if end_time:
                    params["endTime"] = int(end_time.timestamp() * 1000)
                
                response = requests.get(
                    "https://api.binance.com/api/v3/klines",
                    params=params
                )
                
                if response.status_code == 200:
                    candles = response.json()
                    
                    # Formato Binance: [open_time, open, high, low, close, volume, ...]
                    df = pd.DataFrame(candles, columns=[
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trades", "taker_buy_base",
                        "taker_buy_quote", "ignore"
                    ])
                    
                    # Convertir tipos de datos
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = pd.to_numeric(df[col])
                    
                    # Configurar índice
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
                    df.set_index("timestamp", inplace=True)
                    df.sort_index(inplace=True)
                    
                    # Guardar en caché
                    self._cache_data(df, symbol, timeframe)
                    
                    return df
            
            logger.warning(f"No se pudieron obtener datos históricos para {symbol} ({timeframe})")
            return None
            
        except Exception as e:
            logger.error(f"Error al obtener datos históricos para {symbol} ({timeframe}): {e}")
            return None
    
    def _get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Obtiene datos desde la caché.
        
        Args:
            symbol: Símbolo del par
            timeframe: Marco temporal
            
        Returns:
            Optional[pd.DataFrame]: DataFrame con datos o None si no hay caché válida
        """
        # Comprobar si existe el archivo de caché
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{timeframe}.csv")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            # Verificar si la caché está vencida
            file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            cache_age = datetime.now() - file_mtime
            
            # Obtener tiempo de expiración para este timeframe
            expiry_hours = self.cache_expiry_hours.get(timeframe, 24)
            
            if cache_age > timedelta(hours=expiry_hours):
                # Caché vencida
                return None
            
            # Cargar datos desde caché
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error al cargar caché para {symbol} ({timeframe}): {e}")
            return None
    
    def _cache_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        Guarda datos en caché.
        
        Args:
            df: DataFrame con datos
            symbol: Símbolo del par
            timeframe: Marco temporal
        """
        try:
            # Comprobar que hay datos para guardar
            if df is None or df.empty:
                return
            
            # Guardar en formato CSV
            cache_file = os.path.join(self.cache_dir, f"{symbol}_{timeframe}.csv")
            df.to_csv(cache_file)
            
            logger.debug(f"Datos cacheados para {symbol} ({timeframe}): {len(df)} registros")
            
        except Exception as e:
            logger.error(f"Error al cachear datos para {symbol} ({timeframe}): {e}")
    
    def calculate_indicators(self, 
                           df: pd.DataFrame, 
                           indicators: Dict[str, Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Calcula indicadores técnicos para un DataFrame.
        
        Args:
            df: DataFrame con datos OHLCV
            indicators: Diccionario de indicadores a calcular con parámetros
            
        Returns:
            pd.DataFrame: DataFrame con indicadores calculados
        """
        if df is None or df.empty:
            logger.warning("No se pueden calcular indicadores en DataFrame vacío")
            return df
        
        # Hacer copia para no modificar el original
        result_df = df.copy()
        
        # Indicadores predeterminados si no se proporcionan
        default_indicators = {
            "sma": {"periods": [5, 10, 20, 50, 200]},
            "ema": {"periods": [9, 21, 55, 200]},
            "rsi": {"period": 14},
            "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            "bollinger": {"period": 20, "std_dev": 2.0},
            "atr": {"period": 14}
        }
        
        # Usar indicadores proporcionados o predeterminados
        indicators = indicators or default_indicators
        
        try:
            # Calcular cada indicador según configuración
            for indicator, params in indicators.items():
                if indicator == "sma":
                    # Media móvil simple para cada período
                    for period in params.get("periods", [20]):
                        col_name = f"sma_{period}"
                        result_df[col_name] = result_df["close"].rolling(window=period).mean()
                
                elif indicator == "ema":
                    # Media móvil exponencial para cada período
                    for period in params.get("periods", [20]):
                        col_name = f"ema_{period}"
                        result_df[col_name] = result_df["close"].ewm(span=period, adjust=False).mean()
                
                elif indicator == "rsi":
                    # RSI (Relative Strength Index)
                    period = params.get("period", 14)
                    delta = result_df["close"].diff()
                    
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    
                    avg_gain = gain.rolling(window=period).mean()
                    avg_loss = loss.rolling(window=period).mean()
                    
                    # Calcular para el resto de períodos usando valores suavizados
                    for i in range(period, len(result_df)):
                        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
                        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
                    
                    rs = avg_gain / avg_loss
                    result_df["rsi"] = 100 - (100 / (1 + rs))
                
                elif indicator == "macd":
                    # MACD (Moving Average Convergence Divergence)
                    fast_period = params.get("fast_period", 12)
                    slow_period = params.get("slow_period", 26)
                    signal_period = params.get("signal_period", 9)
                    
                    # Calcular EMAs
                    fast_ema = result_df["close"].ewm(span=fast_period, adjust=False).mean()
                    slow_ema = result_df["close"].ewm(span=slow_period, adjust=False).mean()
                    
                    # Calcular MACD y señal
                    result_df["macd_line"] = fast_ema - slow_ema
                    result_df["macd_signal"] = result_df["macd_line"].ewm(span=signal_period, adjust=False).mean()
                    result_df["macd_histogram"] = result_df["macd_line"] - result_df["macd_signal"]
                
                elif indicator == "bollinger":
                    # Bandas de Bollinger
                    period = params.get("period", 20)
                    std_dev = params.get("std_dev", 2.0)
                    
                    result_df["bb_middle"] = result_df["close"].rolling(window=period).mean()
                    result_df["bb_std"] = result_df["close"].rolling(window=period).std()
                    
                    result_df["bb_upper"] = result_df["bb_middle"] + (result_df["bb_std"] * std_dev)
                    result_df["bb_lower"] = result_df["bb_middle"] - (result_df["bb_std"] * std_dev)
                    result_df["bb_width"] = (result_df["bb_upper"] - result_df["bb_lower"]) / result_df["bb_middle"]
                    result_df["bb_pct"] = (result_df["close"] - result_df["bb_lower"]) / (result_df["bb_upper"] - result_df["bb_lower"])
                
                elif indicator == "atr":
                    # ATR (Average True Range)
                    period = params.get("period", 14)
                    
                    # Calcular True Range
                    tr1 = abs(result_df["high"] - result_df["low"])
                    tr2 = abs(result_df["high"] - result_df["close"].shift())
                    tr3 = abs(result_df["low"] - result_df["close"].shift())
                    
                    result_df["tr"] = pd.DataFrame([tr1, tr2, tr3]).max()
                    
                    # Calcular ATR
                    result_df["atr"] = result_df["tr"].rolling(window=period).mean()
                    
                    # ATR como porcentaje del precio
                    result_df["atr_pct"] = result_df["atr"] / result_df["close"] * 100
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error al calcular indicadores: {e}")
            # Devolver DataFrame original si hay error
            return df
    
    def resample_data(self, 
                     df: pd.DataFrame, 
                     new_timeframe: str) -> pd.DataFrame:
        """
        Cambia la temporalidad de los datos.
        
        Args:
            df: DataFrame con datos OHLCV indexados por timestamp
            new_timeframe: Nueva temporalidad (ej. '1h', '4h', '1d')
            
        Returns:
            pd.DataFrame: DataFrame con nueva temporalidad
        """
        if df is None or df.empty:
            logger.warning("No se pueden resamplear datos vacíos")
            return df
        
        try:
            # Mapeo de timeframes a formatos de pandas resample
            timeframe_map = {
                "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                "1h": "1H", "4h": "4H", "1d": "1D", "1w": "1W"
            }
            
            # Obtener formato para resample
            resample_rule = timeframe_map.get(new_timeframe)
            
            if not resample_rule:
                logger.warning(f"Timeframe no soportado para resampleo: {new_timeframe}")
                return df
            
            # Realizar resampleo
            resampled = df.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            return resampled.dropna()
            
        except Exception as e:
            logger.error(f"Error al resamplear datos: {e}")
            return df

def get_market_data(symbol: str, 
                   timeframe: str = "1h", 
                   limit: int = 100,
                   exchange: str = "okx",
                   with_indicators: bool = False) -> Optional[pd.DataFrame]:
    """
    Función de conveniencia para obtener datos de mercado.
    
    Args:
        symbol: Símbolo del par
        timeframe: Marco temporal
        limit: Número de velas a obtener
        exchange: Exchange a utilizar
        with_indicators: Si se deben calcular indicadores
        
    Returns:
        Optional[pd.DataFrame]: DataFrame con datos o None si hay error
    """
    # Crear instancia de MarketData
    market_data = MarketData(exchange=exchange)
    
    # Obtener datos históricos
    df = market_data.get_historical_data(symbol, timeframe, limit)
    
    # Calcular indicadores si se solicita
    if df is not None and with_indicators:
        df = market_data.calculate_indicators(df)
    
    return df

def multi_timeframe_analysis(symbol: str, 
                            timeframes: List[str] = None,
                            exchange: str = "okx") -> Dict[str, Any]:
    """
    Realiza análisis en múltiples timeframes.
    
    Args:
        symbol: Símbolo del par
        timeframes: Lista de timeframes a analizar
        exchange: Exchange a utilizar
        
    Returns:
        Dict[str, Any]: Resultados del análisis multi-timeframe
    """
    # Timeframes predeterminados si no se proporcionan
    timeframes = timeframes or ["5m", "15m", "1h", "4h", "1d"]
    
    # Crear instancia de MarketData
    market_data = MarketData(exchange=exchange)
    
    results = {}
    
    # Obtener datos para cada timeframe
    for tf in timeframes:
        df = market_data.get_historical_data(symbol, tf, limit=100)
        
        if df is not None and not df.empty:
            # Calcular indicadores básicos
            df = market_data.calculate_indicators(df)
            
            # Guardar en resultados
            results[tf] = df
    
    # Análisis de tendencia combinado
    if results:
        trend_votes = {
            "bullish": 0,
            "bearish": 0,
            "neutral": 0
        }
        
        # Análisis simple por timeframe
        for tf, df in results.items():
            # Verificar tendencia por EMAs
            if "ema_9" in df.columns and "ema_21" in df.columns:
                last_ema9 = df["ema_9"].iloc[-1]
                last_ema21 = df["ema_21"].iloc[-1]
                
                if last_ema9 > last_ema21:
                    trend_votes["bullish"] += 1
                elif last_ema9 < last_ema21:
                    trend_votes["bearish"] += 1
                else:
                    trend_votes["neutral"] += 1
            
            # Verificar RSI
            if "rsi" in df.columns:
                last_rsi = df["rsi"].iloc[-1]
                
                if last_rsi > 60:
                    trend_votes["bullish"] += 0.5
                elif last_rsi < 40:
                    trend_votes["bearish"] += 0.5
                
            # Verificar MACD
            if "macd_histogram" in df.columns:
                last_macd_hist = df["macd_histogram"].iloc[-1]
                
                if last_macd_hist > 0:
                    trend_votes["bullish"] += 0.5
                elif last_macd_hist < 0:
                    trend_votes["bearish"] += 0.5
        
        # Determinar tendencia dominante
        max_trend = max(trend_votes, key=trend_votes.get)
        
        return {
            "symbol": symbol,
            "timeframes_analyzed": timeframes,
            "trend_analysis": trend_votes,
            "dominant_trend": max_trend,
            "timestamp": datetime.now().isoformat()
        }
    
    return {"error": f"No se pudieron obtener datos para {symbol}"}

def demo_market_data():
    """Demostración del módulo de datos de mercado."""
    print("\n📊 DATOS DE MERCADO 📊")
    print("Este módulo permite obtener y analizar datos de mercado en tiempo real.")
    
    # Usar SOL-USDT para la demostración
    symbol = "SOL-USDT"
    
    # 1. Obtener precio actual
    print("\n1. Precio actual:")
    market_data = MarketData(exchange='okx')
    current_price = market_data.get_current_price(symbol)
    
    if current_price > 0:
        print(f"  El precio actual de {symbol} es: ${current_price:.2f}")
    else:
        print(f"  No se pudo obtener el precio de {symbol}")
    
    # 2. Obtener datos históricos
    print("\n2. Datos históricos (últimas 5 velas):")
    df = market_data.get_historical_data(symbol, "1h", limit=5)
    
    if df is not None and not df.empty:
        print(df[["open", "high", "low", "close", "volume"]].to_string())
    else:
        print("  No se pudieron obtener datos históricos")
    
    # 3. Calcular indicadores
    print("\n3. Indicadores técnicos:")
    
    if df is not None and not df.empty:
        # Calcular indicadores
        df_with_indicators = market_data.calculate_indicators(df)
        
        # Mostrar algunos indicadores
        if "rsi" in df_with_indicators.columns:
            print(f"  RSI actual: {df_with_indicators['rsi'].iloc[-1]:.2f}")
        
        if "ema_9" in df_with_indicators.columns and "ema_21" in df_with_indicators.columns:
            ema9 = df_with_indicators["ema_9"].iloc[-1]
            ema21 = df_with_indicators["ema_21"].iloc[-1]
            print(f"  EMA 9: {ema9:.2f}")
            print(f"  EMA 21: {ema21:.2f}")
            print(f"  Señal: {'Alcista' if ema9 > ema21 else 'Bajista'}")
        
        if "bb_upper" in df_with_indicators.columns and "bb_lower" in df_with_indicators.columns:
            bb_upper = df_with_indicators["bb_upper"].iloc[-1]
            bb_lower = df_with_indicators["bb_lower"].iloc[-1]
            current_close = df_with_indicators["close"].iloc[-1]
            
            print(f"  Bandas de Bollinger:")
            print(f"    - Superior: ${bb_upper:.2f}")
            print(f"    - Precio: ${current_close:.2f}")
            print(f"    - Inferior: ${bb_lower:.2f}")
    
    # 4. Análisis multi-timeframe
    print("\n4. Análisis multi-timeframe:")
    mtf_result = multi_timeframe_analysis(symbol, timeframes=["15m", "1h", "4h"])
    
    if "error" not in mtf_result:
        print(f"  Tendencia dominante: {mtf_result['dominant_trend']}")
        print(f"  Análisis de tendencia:")
        for trend, votes in mtf_result["trend_analysis"].items():
            print(f"    - {trend}: {votes}")
    else:
        print(f"  Error: {mtf_result.get('error')}")
    
    print("\n✅ Demostración completada.")
    return True

def get_available_symbols(exchange: str = "okx") -> List[str]:
    """
    Obtiene la lista de símbolos disponibles para trading.
    
    Args:
        exchange: Exchange a consultar
        
    Returns:
        List[str]: Lista de símbolos disponibles
    """
    # Lista de símbolos más populares
    popular_symbols = [
        'SOL-USDT', 'BTC-USDT', 'ETH-USDT', 'ADA-USDT', 'XRP-USDT',
        'DOGE-USDT', 'DOT-USDT', 'AVAX-USDT', 'MATIC-USDT', 'LINK-USDT'
    ]
    
    try:
        # Intentar obtener lista de símbolos desde el exchange
        if exchange.lower() == 'okx':
            response = requests.get("https://www.okx.com/api/v5/market/tickers?instType=SPOT")
            
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == "0" and "data" in data:
                    # Filtrar solo pares con USDT
                    symbols = [item["instId"] for item in data["data"] if "-USDT" in item["instId"]]
                    return symbols
                    
        elif exchange.lower() == 'binance':
            response = requests.get("https://api.binance.com/api/v3/exchangeInfo")
            
            if response.status_code == 200:
                data = response.json()
                # Filtrar solo pares con USDT
                symbols = [symbol["symbol"].replace("USDT", "-USDT") 
                          for symbol in data["symbols"] 
                          if symbol["symbol"].endswith("USDT") and symbol["status"] == "TRADING"]
                return symbols
        
        # Si no se pudieron obtener, devolver la lista predefinida
        logger.warning(f"No se pudieron obtener símbolos de {exchange}, usando lista predefinida")
        return popular_symbols
        
    except Exception as e:
        logger.error(f"Error al obtener símbolos disponibles: {e}")
        return popular_symbols

if __name__ == "__main__":
    try:
        demo_market_data()
    except Exception as e:
        print(f"Error en la demostración: {e}")