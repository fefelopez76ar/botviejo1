#!/usr/bin/env python3
"""
Módulo de selección de pares de trading con búsqueda dinámica y autocompletado.

Este módulo permite:
1. Buscar y seleccionar cualquier par de trading disponible
2. Adaptar las estrategias del bot a las características específicas de cada par
3. Guardar configuraciones personalizadas por par de trading
"""

import os
import sys
import json
import logging
import time
import pandas as pd
import requests
from typing import Dict, List, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PairSelector')

class PairSelector:
    """
    Sistema de selección y adaptación de pares de trading.
    
    Esta clase permite buscar, seleccionar y adaptar el bot a cualquier par
    de trading disponible en los exchanges soportados.
    """
    
    def __init__(self, 
               exchange: str = 'okx',
               cache_file: str = 'data/trading_pairs_cache.json',
               cache_duration_hours: int = 24):
        """
        Inicializa el selector de pares de trading.
        
        Args:
            exchange: Exchange a utilizar ('okx', 'binance', etc.)
            cache_file: Archivo para cachear información de pares
            cache_duration_hours: Duración de la caché en horas
        """
        self.exchange = exchange
        self.cache_file = cache_file
        self.cache_duration_hours = cache_duration_hours
        
        # Estructura para almacenar pares de trading
        self.trading_pairs = {}
        
        # Mapa de características por par
        self.pair_characteristics = {}
        
        # Cargar caché o datos iniciales
        self._ensure_cache_directory()
        self._load_or_update_pairs()
        
        # Cargar configuraciones personalizadas por par
        self.pair_configs = self._load_pair_configs()
    
    def _ensure_cache_directory(self):
        """Asegura que el directorio de caché exista."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
    
    def _load_or_update_pairs(self):
        """
        Carga pares desde la caché o actualiza si es necesario.
        """
        try:
            # Verificar si existe la caché y no está vencida
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Verificar si la caché es válida
                cache_timestamp = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01T00:00:00'))
                cache_age = datetime.now() - cache_timestamp
                
                if cache_age < timedelta(hours=self.cache_duration_hours):
                    self.trading_pairs = cache_data.get('pairs', {})
                    logger.info(f"Cargados {len(self.trading_pairs)} pares de trading desde caché")
                    return
            
            # Si la caché no existe o está vencida, actualizar
            self._update_trading_pairs()
            
        except Exception as e:
            logger.error(f"Error al cargar pares de trading: {e}")
            # Intentar actualizar si hay error
            self._update_trading_pairs()
    
    def _update_trading_pairs(self):
        """
        Actualiza la lista de pares de trading disponibles desde el exchange.
        """
        try:
            # Llamar al método específico del exchange
            if self.exchange.lower() == 'okx':
                pairs = self._fetch_okx_pairs()
            elif self.exchange.lower() == 'binance':
                pairs = self._fetch_binance_pairs()
            else:
                pairs = {}
                logger.warning(f"Exchange no soportado: {self.exchange}")
            
            if pairs:
                self.trading_pairs = pairs
                
                # Guardar en caché
                cache_data = {
                    'timestamp': datetime.now().isoformat(),
                    'pairs': pairs
                }
                
                with open(self.cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=4)
                
                logger.info(f"Actualizados {len(pairs)} pares de trading")
                
                # Iniciar proceso de análisis de características en segundo plano
                self._analyze_pair_characteristics_async()
            
        except Exception as e:
            logger.error(f"Error al actualizar pares de trading: {e}")
    
    def _fetch_okx_pairs(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene pares de trading de OKX.
        
        Returns:
            Dict[str, Dict[str, Any]]: Pares de trading con sus características
        """
        pairs = {}
        
        try:
            # Endpoint público de OKX para obtener instrumentos
            url = "https://www.okx.com/api/v5/public/instruments"
            
            # Obtener diferentes tipos de instrumentos (spot, futures, etc.)
            instrument_types = ["SPOT", "SWAP", "FUTURES"]
            
            for inst_type in instrument_types:
                params = {"instType": inst_type}
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("code") == "0" and "data" in data:
                        for instrument in data["data"]:
                            symbol = instrument.get("instId", "")
                            
                            # Solo procesar si tiene un símbolo válido
                            if symbol:
                                base_ccy = instrument.get("baseCcy", "")
                                quote_ccy = instrument.get("quoteCcy", "")
                                
                                # Crear entrada estructurada
                                pairs[symbol] = {
                                    "symbol": symbol,
                                    "base_asset": base_ccy,
                                    "quote_asset": quote_ccy,
                                    "type": inst_type,
                                    "status": instrument.get("state", ""),
                                    "min_size": float(instrument.get("minSz", "0")),
                                    "tick_size": float(instrument.get("tickSz", "0")),
                                    "lot_size": float(instrument.get("lotSz", "0")),
                                    "metadata": instrument
                                }
            
            logger.info(f"Obtenidos {len(pairs)} pares de OKX")
            
        except Exception as e:
            logger.error(f"Error al obtener pares de OKX: {e}")
        
        return pairs
    
    def _fetch_binance_pairs(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene pares de trading de Binance.
        
        Returns:
            Dict[str, Dict[str, Any]]: Pares de trading con sus características
        """
        pairs = {}
        
        try:
            # Endpoint público de Binance para obtener información de exchange
            url = "https://api.binance.com/api/v3/exchangeInfo"
            
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                if "symbols" in data:
                    for symbol_data in data["symbols"]:
                        status = symbol_data.get("status", "")
                        
                        # Solo procesar símbolos activos
                        if status == "TRADING":
                            symbol = symbol_data.get("symbol", "")
                            
                            if symbol:
                                base_asset = symbol_data.get("baseAsset", "")
                                quote_asset = symbol_data.get("quoteAsset", "")
                                
                                # Crear entrada estructurada
                                pairs[symbol] = {
                                    "symbol": symbol,
                                    "base_asset": base_asset,
                                    "quote_asset": quote_asset,
                                    "type": "SPOT",
                                    "status": status,
                                    "metadata": symbol_data
                                }
                                
                                # Extraer información de filtros
                                for filter_data in symbol_data.get("filters", []):
                                    filter_type = filter_data.get("filterType", "")
                                    
                                    if filter_type == "LOT_SIZE":
                                        pairs[symbol]["min_size"] = float(filter_data.get("minQty", "0"))
                                        pairs[symbol]["lot_size"] = float(filter_data.get("stepSize", "0"))
                                    
                                    elif filter_type == "PRICE_FILTER":
                                        pairs[symbol]["tick_size"] = float(filter_data.get("tickSize", "0"))
            
            logger.info(f"Obtenidos {len(pairs)} pares de Binance")
            
        except Exception as e:
            logger.error(f"Error al obtener pares de Binance: {e}")
        
        return pairs
    
    def _analyze_pair_characteristics_async(self):
        """
        Inicia análisis asincrónico de características de pares.
        """
        # Se ejecuta en un hilo separado para no bloquear
        def analyze_worker():
            try:
                # Seleccionar pares populares para análisis prioritario
                popular_assets = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOT", "DOGE", "AVAX", "MATIC"]
                usdt_pairs = {k: v for k, v in self.trading_pairs.items() if v.get("quote_asset") == "USDT"}
                
                # Priorizar pares populares
                priority_pairs = {}
                for asset in popular_assets:
                    for symbol, data in usdt_pairs.items():
                        if data.get("base_asset") == asset:
                            priority_pairs[symbol] = data
                
                # Añadir el resto de pares USDT
                other_pairs = {k: v for k, v in usdt_pairs.items() if k not in priority_pairs}
                
                # Analizar primero los prioritarios
                self._analyze_pairs_batch(list(priority_pairs.keys()))
                
                # Luego el resto (limitando a 50 para no sobrecargar)
                other_keys = list(other_pairs.keys())[:50]
                self._analyze_pairs_batch(other_keys)
                
                logger.info(f"Análisis de características completado para {len(priority_pairs) + len(other_keys)} pares")
                
            except Exception as e:
                logger.error(f"Error en análisis de características: {e}")
        
        # Iniciar en hilo separado
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(analyze_worker)
    
    def _analyze_pairs_batch(self, symbols: List[str]):
        """
        Analiza un lote de pares para determinar sus características.
        
        Args:
            symbols: Lista de símbolos a analizar
        """
        for symbol in symbols:
            try:
                # Obtener datos históricos
                hist_data = self._fetch_historical_data(symbol, "1d", 30)
                
                if hist_data is not None and not hist_data.empty:
                    # Calcular métricas básicas
                    volatility = hist_data["close"].pct_change().std() * 100  # Volatilidad diaria en %
                    avg_volume = hist_data["volume"].mean()
                    volume_volatility = hist_data["volume"].std() / avg_volume
                    
                    # Calcular rangos de precio
                    avg_range = (hist_data["high"] - hist_data["low"]).mean()
                    avg_range_pct = avg_range / hist_data["close"].mean() * 100
                    
                    # Calcular tendencia
                    start_price = hist_data["close"].iloc[0]
                    end_price = hist_data["close"].iloc[-1]
                    trend = (end_price / start_price - 1) * 100  # % de cambio
                    
                    # Guardar características
                    self.pair_characteristics[symbol] = {
                        "volatility_daily": volatility,
                        "avg_volume": avg_volume,
                        "volume_volatility": volume_volatility,
                        "avg_range_pct": avg_range_pct,
                        "trend_30d": trend,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Pequeña pausa para no sobrecargar API
                    time.sleep(0.2)
            
            except Exception as e:
                logger.error(f"Error al analizar par {symbol}: {e}")
    
    def _fetch_historical_data(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos para un par.
        
        Args:
            symbol: Símbolo del par
            timeframe: Marco temporal
            limit: Cantidad de velas
            
        Returns:
            Optional[pd.DataFrame]: DataFrame con datos o None si hay error
        """
        try:
            # Usar la API del exchange correspondiente
            if self.exchange.lower() == 'okx':
                # Endpoint para datos de velas de OKX
                url = "https://www.okx.com/api/v5/market/candles"
                
                params = {
                    "instId": symbol,
                    "bar": timeframe,
                    "limit": limit
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("code") == "0" and "data" in data:
                        # Formato: [timestamp, open, high, low, close, volume, ...]
                        candles = data["data"]
                        
                        df = pd.DataFrame(candles, columns=[
                            "timestamp", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote", "confirm"
                        ])
                        
                        # Convertir tipos de datos
                        for col in ["open", "high", "low", "close", "volume"]:
                            df[col] = pd.to_numeric(df[col])
                        
                        # Ordenar por timestamp
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
                        df.sort_values("timestamp", inplace=True)
                        
                        return df
            
            elif self.exchange.lower() == 'binance':
                # Endpoint para datos de velas de Binance
                url = "https://api.binance.com/api/v3/klines"
                
                # Convertir timeframe a formato de Binance
                interval_map = {
                    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                    "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"
                }
                binance_interval = interval_map.get(timeframe, "1d")
                
                params = {
                    "symbol": symbol,
                    "interval": binance_interval,
                    "limit": limit
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    # Formato: [timestamp, open, high, low, close, volume, ...]
                    candles = response.json()
                    
                    df = pd.DataFrame(candles, columns=[
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_asset_volume", "number_of_trades",
                        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                    ])
                    
                    # Convertir tipos de datos
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = pd.to_numeric(df[col])
                    
                    # Ordenar por timestamp
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
                    df.sort_values("timestamp", inplace=True)
                    
                    return df
            
            return None
                
        except Exception as e:
            logger.error(f"Error al obtener datos históricos para {symbol}: {e}")
            return None
    
    def _load_pair_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Carga configuraciones personalizadas por par.
        
        Returns:
            Dict[str, Dict[str, Any]]: Configuraciones por par
        """
        config_file = "data/pair_configs.json"
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error al cargar configuraciones de pares: {e}")
        
        return {}
    
    def _save_pair_configs(self):
        """Guarda configuraciones personalizadas por par."""
        config_file = "data/pair_configs.json"
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.pair_configs, f, indent=4)
            
            logger.info(f"Guardadas configuraciones para {len(self.pair_configs)} pares")
            
        except Exception as e:
            logger.error(f"Error al guardar configuraciones de pares: {e}")
    
    def search_pairs(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Busca pares de trading que coincidan con la consulta.
        
        Args:
            query: Texto de búsqueda (símbolo, nombre, etc.)
            max_results: Número máximo de resultados
            
        Returns:
            List[Dict[str, Any]]: Lista de pares coincidentes
        """
        results = []
        query = query.upper()
        
        # Primero buscar coincidencias exactas de símbolo
        if query in self.trading_pairs:
            results.append(self.trading_pairs[query])
        
        # Luego buscar coincidencias parciales
        for symbol, data in self.trading_pairs.items():
            # Si ya tenemos suficientes resultados, parar
            if len(results) >= max_results:
                break
                
            # Evitar duplicados (coincidencias exactas ya añadidas)
            if data in results:
                continue
                
            # Buscar en símbolo, base_asset y quote_asset
            if (query in symbol or 
                query in data.get("base_asset", "") or 
                query in data.get("quote_asset", "")):
                results.append(data)
        
        return results
    
    def get_pair_info(self, symbol: str) -> Dict[str, Any]:
        """
        Obtiene información detallada de un par específico.
        
        Args:
            symbol: Símbolo del par
            
        Returns:
            Dict[str, Any]: Información del par
        """
        # Información básica del par
        pair_info = self.trading_pairs.get(symbol, {})
        
        if not pair_info:
            return {"error": f"Par no encontrado: {symbol}"}
        
        # Añadir características si están disponibles
        if symbol in self.pair_characteristics:
            pair_info["characteristics"] = self.pair_characteristics[symbol]
        
        # Añadir configuración personalizada si existe
        if symbol in self.pair_configs:
            pair_info["custom_config"] = self.pair_configs[symbol]
        
        return pair_info
    
    def configure_pair(self, symbol: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configura parámetros personalizados para un par específico.
        
        Args:
            symbol: Símbolo del par
            config: Configuración personalizada
            
        Returns:
            Dict[str, Any]: Resultado de la operación
        """
        if symbol not in self.trading_pairs:
            return {"error": f"Par no encontrado: {symbol}"}
        
        # Validar y guardar configuración
        try:
            # Actualizar configuración existente o crear nueva
            if symbol in self.pair_configs:
                self.pair_configs[symbol].update(config)
            else:
                self.pair_configs[symbol] = config
            
            # Añadir metadatos
            self.pair_configs[symbol]["last_updated"] = datetime.now().isoformat()
            
            # Guardar configuraciones
            self._save_pair_configs()
            
            return {
                "success": True,
                "message": f"Configuración actualizada para {symbol}",
                "config": self.pair_configs[symbol]
            }
            
        except Exception as e:
            logger.error(f"Error al configurar par {symbol}: {e}")
            return {"error": f"Error al configurar par: {e}"}
    
    def get_optimized_parameters(self, symbol: str) -> Dict[str, Any]:
        """
        Obtiene parámetros optimizados para trading en este par.
        
        Args:
            symbol: Símbolo del par
            
        Returns:
            Dict[str, Any]: Parámetros optimizados
        """
        # Verificar si el par existe
        if symbol not in self.trading_pairs:
            return {"error": f"Par no encontrado: {symbol}"}
        
        # Comprobar si hay configuración personalizada
        if symbol in self.pair_configs:
            return self.pair_configs[symbol]
        
        # Comprobar si tenemos características analizadas
        if symbol in self.pair_characteristics:
            char = self.pair_characteristics[symbol]
            
            # Ajustar parámetros según características
            volatility = char.get("volatility_daily", 5.0)
            avg_range_pct = char.get("avg_range_pct", 3.0)
            
            # Adaptar parámetros según volatilidad
            if volatility > 10.0:  # Alta volatilidad
                params = {
                    "take_profit_pct": min(avg_range_pct * 0.8, 1.5),
                    "stop_loss_pct": min(avg_range_pct * 0.5, 1.0),
                    "trailing_stop_pct": min(avg_range_pct * 0.4, 0.8),
                    "max_position_size_pct": 1.0,  # Menor exposición en pares volátiles
                    "volatility_category": "high"
                }
            elif volatility > 5.0:  # Media volatilidad
                params = {
                    "take_profit_pct": min(avg_range_pct * 0.7, 1.0),
                    "stop_loss_pct": min(avg_range_pct * 0.4, 0.7),
                    "trailing_stop_pct": min(avg_range_pct * 0.3, 0.5),
                    "max_position_size_pct": 2.0,
                    "volatility_category": "medium"
                }
            else:  # Baja volatilidad
                params = {
                    "take_profit_pct": min(avg_range_pct * 0.6, 0.7),
                    "stop_loss_pct": min(avg_range_pct * 0.3, 0.5),
                    "trailing_stop_pct": min(avg_range_pct * 0.2, 0.3),
                    "max_position_size_pct": 3.0,  # Mayor exposición en pares estables
                    "volatility_category": "low"
                }
            
            # Añadir parámetros adicionales
            params.update({
                "min_volume": char.get("avg_volume", 1000) * 0.1,  # Mínimo 10% del volumen promedio
                "auto_generated": True,
                "generation_timestamp": datetime.now().isoformat()
            })
            
            return params
        
        # Parámetros por defecto si no hay datos
        return {
            "take_profit_pct": 0.8,
            "stop_loss_pct": 0.5,
            "trailing_stop_pct": 0.3,
            "max_position_size_pct": 2.0,
            "min_volume": 1000,
            "auto_generated": True,
            "default_params": True,
            "generation_timestamp": datetime.now().isoformat()
        }
    
    def get_current_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Obtiene datos actuales de mercado para un par.
        
        Args:
            symbol: Símbolo del par
            
        Returns:
            Dict[str, Any]: Datos de mercado actuales
        """
        try:
            # Verificar si el par existe
            if symbol not in self.trading_pairs:
                return {"error": f"Par no encontrado: {symbol}"}
            
            # Obtener datos según el exchange
            if self.exchange.lower() == 'okx':
                # Endpoint para ticker de OKX
                url = "https://www.okx.com/api/v5/market/ticker"
                
                params = {
                    "instId": symbol
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("code") == "0" and "data" in data and data["data"]:
                        ticker = data["data"][0]
                        
                        return {
                            "symbol": symbol,
                            "price": float(ticker.get("last", 0)),
                            "bid": float(ticker.get("bidPx", 0)),
                            "ask": float(ticker.get("askPx", 0)),
                            "volume_24h": float(ticker.get("volCcy24h", 0)),
                            "change_24h_pct": float(ticker.get("sodUtc0", 0)),
                            "high_24h": float(ticker.get("high24h", 0)),
                            "low_24h": float(ticker.get("low24h", 0)),
                            "timestamp": datetime.now().isoformat()
                        }
            
            elif self.exchange.lower() == 'binance':
                # Endpoint para ticker de Binance
                url = "https://api.binance.com/api/v3/ticker/24hr"
                
                params = {
                    "symbol": symbol
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    ticker = response.json()
                    
                    return {
                        "symbol": symbol,
                        "price": float(ticker.get("lastPrice", 0)),
                        "bid": float(ticker.get("bidPrice", 0)),
                        "ask": float(ticker.get("askPrice", 0)),
                        "volume_24h": float(ticker.get("volume", 0)),
                        "change_24h_pct": float(ticker.get("priceChangePercent", 0)),
                        "high_24h": float(ticker.get("highPrice", 0)),
                        "low_24h": float(ticker.get("lowPrice", 0)),
                        "timestamp": datetime.now().isoformat()
                    }
            
            return {"error": f"No se pudieron obtener datos para {symbol}"}
            
        except Exception as e:
            logger.error(f"Error al obtener datos de mercado para {symbol}: {e}")
            return {"error": f"Error: {e}"}
    
    def get_recommended_pairs(self, strategy_type: str = 'scalping') -> List[Dict[str, Any]]:
        """
        Obtiene pares recomendados para un tipo de estrategia.
        
        Args:
            strategy_type: Tipo de estrategia ('scalping', 'swing', etc.)
            
        Returns:
            List[Dict[str, Any]]: Lista de pares recomendados
        """
        recommendations = []
        
        # Solo considerar pares con características analizadas
        pairs_with_data = {k: v for k, v in self.pair_characteristics.items() if k in self.trading_pairs}
        
        if not pairs_with_data:
            return []
        
        # Filtrar según el tipo de estrategia
        if strategy_type == 'scalping':
            # Para scalping queremos volatilidad moderada-alta y alto volumen
            for symbol, char in pairs_with_data.items():
                pair_data = self.trading_pairs[symbol]
                
                # Solo considerar pares con USDT
                if pair_data.get("quote_asset") != "USDT":
                    continue
                
                volatility = char.get("volatility_daily", 0)
                volume = char.get("avg_volume", 0)
                
                # Criterios para scalping
                if 3.0 <= volatility <= 15.0 and volume > 1000000:
                    score = (volatility * 0.6) + (volume / 1000000 * 0.4)
                    
                    recommendations.append({
                        "symbol": symbol,
                        "base_asset": pair_data.get("base_asset", ""),
                        "quote_asset": pair_data.get("quote_asset", ""),
                        "volatility": volatility,
                        "volume": volume,
                        "score": score
                    })
        
        elif strategy_type == 'swing':
            # Para swing trading buscamos tendencias claras
            for symbol, char in pairs_with_data.items():
                pair_data = self.trading_pairs[symbol]
                
                # Solo considerar pares con USDT
                if pair_data.get("quote_asset") != "USDT":
                    continue
                
                trend = abs(char.get("trend_30d", 0))
                volume = char.get("avg_volume", 0)
                
                # Criterios para swing trading
                if trend > 5.0 and volume > 500000:
                    score = (trend * 0.7) + (volume / 1000000 * 0.3)
                    
                    recommendations.append({
                        "symbol": symbol,
                        "base_asset": pair_data.get("base_asset", ""),
                        "quote_asset": pair_data.get("quote_asset", ""),
                        "trend": char.get("trend_30d", 0),
                        "volume": volume,
                        "score": score
                    })
        
        else:
            # Estrategia genérica, priorizar volumen y volatilidad moderada
            for symbol, char in pairs_with_data.items():
                pair_data = self.trading_pairs[symbol]
                
                # Solo considerar pares con USDT
                if pair_data.get("quote_asset") != "USDT":
                    continue
                
                volatility = char.get("volatility_daily", 0)
                volume = char.get("avg_volume", 0)
                
                # Criterios generales
                if volatility > 0 and volume > 100000:
                    score = (min(volatility, 10) * 0.5) + (volume / 1000000 * 0.5)
                    
                    recommendations.append({
                        "symbol": symbol,
                        "base_asset": pair_data.get("base_asset", ""),
                        "quote_asset": pair_data.get("quote_asset", ""),
                        "volatility": volatility,
                        "volume": volume,
                        "score": score
                    })
        
        # Ordenar por puntuación (score) descendente
        recommendations.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Limitar a los 10 mejores
        return recommendations[:10]
    
    def adapt_strategy_to_pair(self, symbol: str, strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapta una estrategia a las características específicas del par.
        
        Args:
            symbol: Símbolo del par
            strategy_params: Parámetros base de la estrategia
            
        Returns:
            Dict[str, Any]: Parámetros adaptados
        """
        # Si el par no existe, devolver los parámetros originales
        if symbol not in self.trading_pairs:
            return strategy_params
        
        # Obtener características del par
        if symbol in self.pair_characteristics:
            char = self.pair_characteristics[symbol]
            
            # Copiar parámetros para no modificar el original
            adapted_params = strategy_params.copy()
            
            # Adaptaciones según volatilidad
            volatility = char.get("volatility_daily", 5.0)
            
            # Ajustar take profit y stop loss según volatilidad
            if "take_profit_pct" in strategy_params:
                # Para pares más volátiles, aumentar take profit proporcionalmente
                volatility_factor = volatility / 5.0  # normalizado a volatilidad base de 5%
                adapted_params["take_profit_pct"] = strategy_params["take_profit_pct"] * min(volatility_factor, 2.0)
            
            if "stop_loss_pct" in strategy_params:
                # Para pares más volátiles, aumentar stop loss proporcionalmente pero con un factor menor
                volatility_factor = volatility / 5.0
                adapted_params["stop_loss_pct"] = strategy_params["stop_loss_pct"] * min(volatility_factor, 1.8)
            
            # Ajustar tamaño de posición inversamente a la volatilidad
            if "position_size_pct" in strategy_params:
                volatility_factor = volatility / 5.0
                adapted_params["position_size_pct"] = strategy_params["position_size_pct"] / min(volatility_factor, 2.0)
            
            # Añadir metadatos de adaptación
            adapted_params["adapted_to"] = symbol
            adapted_params["volatility_factor"] = volatility / 5.0
            adapted_params["adaptation_timestamp"] = datetime.now().isoformat()
            
            return adapted_params
        
        # Si no hay características, devolver los parámetros originales
        return strategy_params

def get_trading_pair_selector(exchange: str = 'okx') -> PairSelector:
    """
    Función de conveniencia para obtener un selector de pares.
    
    Args:
        exchange: Exchange a utilizar
        
    Returns:
        PairSelector: Selector de pares inicializado
    """
    return PairSelector(exchange=exchange)

def search_trading_pairs(query: str, exchange: str = 'okx', max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Función de conveniencia para buscar pares de trading.
    
    Args:
        query: Texto de búsqueda
        exchange: Exchange a utilizar
        max_results: Número máximo de resultados
        
    Returns:
        List[Dict[str, Any]]: Lista de pares coincidentes
    """
    selector = get_trading_pair_selector(exchange)
    return selector.search_pairs(query, max_results)

def demo_pair_selection():
    """Demostración del selector de pares de trading."""
    print("\n🔍 SELECTOR DE PARES DE TRADING 🔍")
    print("Este módulo permite buscar y adaptar estrategias a cualquier par de trading.")
    
    # Crear selector de pares
    selector = PairSelector(exchange='okx')
    
    # Ejemplos de búsqueda
    print("\n1. Búsqueda de pares:")
    
    # Buscar SOL
    sol_results = selector.search_pairs("SOL", 5)
    if sol_results:
        print("\n  Resultados para 'SOL':")
        for i, pair in enumerate(sol_results):
            print(f"   {i+1}. {pair['symbol']} ({pair['base_asset']}/{pair['quote_asset']})")
    
    # Buscar BTC
    btc_results = selector.search_pairs("BTC", 5)
    if btc_results:
        print("\n  Resultados para 'BTC':")
        for i, pair in enumerate(btc_results):
            print(f"   {i+1}. {pair['symbol']} ({pair['base_asset']}/{pair['quote_asset']})")
    
    # Buscar algo menos común
    alt_results = selector.search_pairs("ETH", 5)
    if alt_results:
        print("\n  Resultados para 'ETH':")
        for i, pair in enumerate(alt_results):
            print(f"   {i+1}. {pair['symbol']} ({pair['base_asset']}/{pair['quote_asset']})")
    
    # Mostrar cómo se adaptan estrategias
    print("\n2. Adaptación de estrategias:")
    
    # Estrategia base para scalping
    base_strategy = {
        "name": "Scalping Momentum",
        "take_profit_pct": 0.8,
        "stop_loss_pct": 0.5,
        "position_size_pct": 2.0,
        "timeframe": "5m"
    }
    
    # Adaptar para diferentes pares
    symbols_to_adapt = ["SOL-USDT", "BTC-USDT", "ETH-USDT"]
    
    print("\n  Estrategia base:")
    for k, v in base_strategy.items():
        print(f"   - {k}: {v}")
    
    for symbol in symbols_to_adapt:
        adapted = selector.adapt_strategy_to_pair(symbol, base_strategy)
        
        print(f"\n  Adaptada para {symbol}:")
        for k, v in adapted.items():
            if k in ['take_profit_pct', 'stop_loss_pct', 'position_size_pct']:
                print(f"   - {k}: {v}")
    
    # Mostrar pares recomendados
    print("\n3. Pares recomendados para scalping:")
    
    recommended = selector.get_recommended_pairs('scalping')
    if recommended:
        for i, pair in enumerate(recommended[:5]):
            print(f"   {i+1}. {pair['symbol']} - Volatilidad: {pair['volatility']:.2f}%, "
                  f"Volumen: {pair['volume']:.0f}, Score: {pair['score']:.2f}")
    
    print("\n✅ Demostración completada. Ya puedes usar el selector de pares.")
    print("   Para buscar pares, usa: search_trading_pairs('SIMBOLO')")
    print("   Para obtener información detallada: selector.get_pair_info('SIMBOLO-USDT')")
    
    return True

if __name__ == "__main__":
    try:
        demo_pair_selection()
    except Exception as e:
        print(f"Error en la demostración: {e}")