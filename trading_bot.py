"""
Bot de Trading para Solana - Versión Autónoma

Este script está diseñado para ejecutarse como un workflow dedicado en Replit
y proporciona funcionalidad 24/7 para trading automatizado de Solana.

Características:
- Monitoreo continuo del mercado
- Ejecución automática de estrategias
- Sistema de recuperación ante fallos
- Notificaciones en tiempo real vía Telegram
- Modo paper trading (simulación) y live trading

Uso:
    python solana_trading_bot.py --mode paper --interval 15m --notify
"""

import os
import sys
import time
import json
import random
import logging
import argparse
import threading
import traceback
from datetime import datetime, timedelta
import hmac
import base64
import hashlib
import requests
from typing import Dict, List, Union, Optional, Any, Tuple
import pandas as pd
import numpy as np

# Intentar importar librería para notificaciones
try:
    import telebot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    
# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SolanaTradingBot")

# Constantes
DEFAULT_SYMBOL = "SOL-USDT"
DEFAULT_INTERVAL = "15m"
DEFAULT_MODE = "paper"  # 'paper' para simulación, 'live' para trading real
DEFAULT_INITIAL_BALANCE = 1000.0  # Balance inicial para paper trading
DEFAULT_LEVERAGE = 3.0  # Apalancamiento predeterminado

# Archivo de estado para seguimiento entre reinicios
STATE_FILE = "bot_state.json"
CONFIG_FILE = "config.env"

class TradingBot:
    """
    Bot de trading para criptomonedas con OKX
    """
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str, mode: str = 'paper'):
        """
        Inicializa el bot de trading
        
        Args:
            api_key: Clave API de OKX
            api_secret: Secret API de OKX
            passphrase: Passphrase de API de OKX
            mode: Modo de operación ('live' o 'paper')
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        
        # Validar modo
        if mode not in ['live', 'paper']:
            logger.warning(f"Modo no válido: {mode}. Usando modo paper por defecto")
            mode = 'paper'
        
        self.mode = mode
        self.base_url = "https://www.okx.com"
        
        # Estado del trading
        self.position = None
        self.orders = []
        self.balance = DEFAULT_INITIAL_BALANCE
        self.leverage = DEFAULT_LEVERAGE
        
        # Inicializar objetos de caché
        self._server_time_cache = {}
        self._positions_cache = {}
        self.time_offset = 10000  # Offset por defecto de 10s
        self.server_time = 0
        
        # Sincronizar tiempo con el servidor
        self._sync_time()
        
        # Inicializar información de trading
        self.trading_data = {
            'symbol': '',
            'interval': '',
            'position': None,
            'entry_price': 0.0,
            'current_price': 0.0,
            'strategy_signals': {},
            'integrated_signal': 'neutral',
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'roi': 0.0,
            'pnl': 0.0,
            'trade_count': 0,
            'wins': 0,
            'losses': 0,
            'start_balance': DEFAULT_INITIAL_BALANCE,
            'current_balance': DEFAULT_INITIAL_BALANCE
        }
        
        logger.info(f"Bot inicializado en modo {mode}")
    
    def _sign_request(self, method: str, endpoint: str, params: dict) -> Dict[str, str]:
        """
        Firma una petición para autenticación con OKX
        
        Args:
            method: Método HTTP (GET, POST, etc.)
            endpoint: Ruta de la API
            params: Parámetros de la petición
            
        Returns:
            Dict: Headers firmados para la autenticación
        """
        # SOLUCIÓN DIRECTA: Obtener el timestamp directamente del servidor OKX
        try:
            server_response = requests.get(f"{self.base_url}/api/v5/public/time")
            if server_response.status_code == 200:
                # Usar directamente el timestamp del servidor + un offset extra
                timestamp_ms = int(server_response.json()['data'][0]['ts']) 
                
                # Añadir buffer EXTREMO para resolver problema persistente
                if self.mode == 'paper':
                    timestamp_ms += 120000  # 2 MINUTOS adicionales para paper trading
                else:
                    timestamp_ms += 60000   # 1 MINUTO adicional para trading real
                
                logger.warning(f"SOLUCIÓN EXTREMA: Añadiendo offset de {120 if self.mode == 'paper' else 60} segundos al timestamp")
                
                timestamp = str(timestamp_ms)
                logger.info(f"Usando timestamp OKX + buffer: {timestamp}")
            else:
                # Fallback a método anterior
                current_time_ms = int(time.time() * 1000)
                timestamp_ms = current_time_ms + self.time_offset
                timestamp = str(timestamp_ms)
                logger.warning(f"Fallback a timestamp local + offset: {timestamp}")
        except Exception as e:
            # Fallback a método anterior en caso de error
            current_time_ms = int(time.time() * 1000)
            timestamp_ms = current_time_ms + self.time_offset
            timestamp = str(timestamp_ms)
            logger.warning(f"Error obteniendo timestamp del servidor: {e}. Usando fallback: {timestamp}")
        
        # Ordenar parámetros si es necesario
        if method == 'GET' and params:
            query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
            endpoint = f"{endpoint}?{query_string}"
            params = {}
        
        # Preparar la cadena a firmar
        if params:
            json_params = json.dumps(params)
            to_sign = timestamp + method + endpoint + json_params
        else:
            to_sign = timestamp + method + endpoint
        
        # Generar firma HMAC
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                to_sign.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        # Construir headers
        headers = {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': signature,
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
        
        # Añadir flag de demo si estamos en modo paper
        if self.mode == 'paper':
            headers['x-simulated-trading'] = '1'  # Mantener como string '1'
            logger.info(f"Usando modo paper trading con flag: {headers['x-simulated-trading']}")
        
        return headers
    
    def _sync_time(self) -> bool:
        """
        Sincroniza el tiempo con el servidor de OKX
        
        Returns:
            bool: True si la sincronización fue exitosa, False en caso contrario
        """
        try:
            # Obtener tiempo directamente del servidor OKX
            server_response = requests.get(f"{self.base_url}/api/v5/public/time")
            if server_response.status_code == 200:
                server_time = int(server_response.json()['data'][0]['ts'])
                local_time = int(time.time() * 1000)
                
                # Para paper trading, usar un offset aún mayor (15 segundos)
                time_buffer = 15000 if self.mode == 'paper' else 10000
                
                # Calcular offset total: diferencia + buffer
                self.time_offset = server_time - local_time + time_buffer
                self.server_time = server_time
                
                logger.info(f"✅ Sincronización de tiempo correcta. Offset: {self.time_offset}ms (diferencia + {time_buffer}ms)")
                return True
            else:
                logger.error(f"Error al obtener tiempo del servidor: {server_response.text}")
                # Establecer un offset por defecto según el modo
                self.time_offset = 15000 if self.mode == 'paper' else 10000
                return False
        except Exception as e:
            logger.error(f"Error sincronizando tiempo: {e}")
            # Establecer un offset por defecto según el modo
            self.time_offset = 15000 if self.mode == 'paper' else 10000
            return False
    
    def get_server_time(self, cache_seconds: int = 30, trading_type: str = None) -> str:
        """
        Obtiene el tiempo del servidor con offset fijo para resolver problemas de sincronización
        """
        try:
            # Obtener tiempo directamente del servidor OKX
            server_response = requests.get(f"{self.base_url}/api/v5/public/time")
            if server_response.status_code == 200:
                server_time = server_response.json()['data'][0]['ts']
                # No añadir ningún offset, solo devolver el tiempo exacto del servidor
                logger.debug(f"Usando tiempo exacto del servidor OKX: {server_time}")
                return server_time
        except Exception as e:
            logger.error(f"Error al obtener tiempo del servidor: {e}")
        
        # Fallback: Usar tiempo local con offset
        current_time_ms = int(time.time() * 1000) + self.time_offset
        logger.debug(f"Fallback a tiempo local: {current_time_ms}, offset: +{self.time_offset}ms")
        return str(current_time_ms)
    
    def _request(self, method: str, endpoint: str, params: dict = None) -> Dict:
        """
        Realiza una petición a la API de OKX
        
        Args:
            method: Método HTTP (GET, POST, etc.)
            endpoint: Ruta de la API
            params: Parámetros de la petición
            
        Returns:
            Dict: Respuesta de la API
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._sign_request(method, endpoint, params or {})
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=params)
            else:
                raise ValueError(f"Método HTTP no soportado: {method}")
            
            # Verificar si la petición fue exitosa
            if response.status_code != 200:
                logger.error(f"Error en la API: {response.status_code} - {response.text}")
                return {"code": str(response.status_code), "msg": response.text, "data": []}
            
            # Parsear respuesta
            result = response.json()
            
            # Verificar si hay error en la respuesta
            if result.get('code') != '0':
                logger.error(f"Error en la API: {result.get('code')} - {result.get('msg')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en petición a API: {str(e)}")
            return {"code": "9999", "msg": str(e), "data": []}
    
    def get_account_balance(self) -> Dict:
        """
        Obtiene el balance de la cuenta
        
        Returns:
            Dict: Información del balance de la cuenta
        """
        # Si estamos en modo paper, usar balance simulado
        if self.mode == 'paper':
            # Generar balance simulado con 1000 USDT
            logger.info("🔄 MODO PAPER: Usando balance simulado")
            return {
                "totalEq": "1000",
                "isoEq": "1000",
                "adjEq": "1000",
                "ordFroz": "0",
                "imr": "0",
                "mmr": "0",
                "details": [
                    {
                        "ccy": "USDT",
                        "eq": "1000",
                        "cashBal": "1000",
                        "uTime": str(int(time.time() * 1000)),
                        "isoEq": "0",
                        "availEq": "1000",
                        "disEq": "1000",
                        "availBal": "1000",
                        "frozenBal": "0",
                    }
                ]
            }
        
        # Si no estamos en modo paper, realizar petición real
        endpoint = '/api/v5/account/balance'
        response = self._request('GET', endpoint)
        
        if response.get('code') == '0':
            return response['data'][0]
        else:
            logger.error(f"No se pudo obtener el balance: {response.get('msg')}")
            # Fallback a balance vacío
            return {}
    
    def get_positions(self, symbol: str = None) -> List[Dict]:
        """
        Obtiene las posiciones abiertas
        
        Args:
            symbol (str, optional): Símbolo específico a consultar
            
        Returns:
            List[Dict]: Lista de posiciones abiertas
        """
        # Si estamos en modo paper, devolver posiciones simuladas
        if self.mode == 'paper':
            logger.info("🔄 MODO PAPER: Usando posiciones simuladas")
            
            # Devolver posición actual si existe
            if hasattr(self, 'position') and self.position and (not symbol or symbol == self.position.get('instId')):
                position_data = {
                    'adl': '1',
                    'availPos': str(self.position.get('pos', '0')),
                    'avgPx': str(self.position.get('avgPx', '0')),
                    'cTime': str(int(time.time() * 1000)),
                    'ccy': 'USDT',
                    'instId': self.position.get('instId', symbol or 'SOL-USDT'),
                    'instType': 'SPOT',
                    'lever': '1',
                    'pos': str(self.position.get('pos', '0')),
                    'posSide': 'long',
                    'uTime': str(int(time.time() * 1000)),
                    'upl': str(self.position.get('upl', '0'))
                }
                return [position_data]
            else:
                return []
        
        # Solicitud real a la API
        endpoint = '/api/v5/account/positions'
        params = {}
        if symbol:
            params['instId'] = symbol
        
        response = self._request('GET', endpoint, params)
        
        if response.get('code') == '0':
            return response['data']
        else:
            logger.error(f"No se pudieron obtener las posiciones: {response.get('msg')}")
            return []
    
    def get_market_price(self, symbol: str) -> float:
        """
        Obtiene el precio actual del mercado
        
        Args:
            symbol (str): Símbolo del instrumento
            
        Returns:
            float: Precio actual o 0 si hay error
        """
        # SOLUCIÓN: Usar endpoint público sin autenticación
        try:
            endpoint = '/api/v5/market/ticker'
            params = {'instId': symbol}
            url = f"{self.base_url}{endpoint}"
            
            # Petición directa sin autenticación
            logger.info(f"Obteniendo precio via endpoint público: {url}")
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == '0' and result['data']:
                    price = float(result['data'][0]['last'])
                    logger.info(f"Precio obtenido: {price}")
                    return price
            
            logger.error(f"No se pudo obtener el precio: {response.text}")
            return 0.0
        except Exception as e:
            logger.error(f"Error al obtener precio: {e}")
            return 0.0
    
    def get_historical_data(self, symbol: str, interval: str = '15m', 
                         limit: int = 100) -> pd.DataFrame:
        """
        Obtiene datos históricos de precios
        
        Args:
            symbol (str): Símbolo del instrumento
            interval (str): Intervalo de tiempo (1m, 5m, 15m, 1h, 4h, 1d)
            limit (int): Número de velas a obtener
            
        Returns:
            pd.DataFrame: DataFrame con datos históricos
        """
        # SOLUCIÓN: Usar endpoint público sin autenticación
        try:
            # Mapear intervalos a formato OKX
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1H',
                '4h': '4H',
                '1d': '1D'
            }
            
            okx_interval = interval_map.get(interval, '15m')
            
            endpoint = '/api/v5/market/candles'
            params = {
                'instId': symbol,
                'bar': okx_interval,
                'limit': str(limit)
            }
            
            # Petición directa sin autenticación
            url = f"{self.base_url}{endpoint}"
            logger.info(f"Obteniendo datos históricos via endpoint público: {url}")
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == '0' and result['data']:
                    # Formatear datos
                    data = []
                    for candle in result['data']:
                        data.append({
                            'timestamp': int(candle[0]),
                            'open': float(candle[1]),
                            'high': float(candle[2]),
                            'low': float(candle[3]),
                            'close': float(candle[4]),
                            'volume': float(candle[5])
                        })
                    
                    # Crear DataFrame
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.sort_values('date')
                    df.set_index('date', inplace=True)
                    
                    logger.info(f"Datos históricos obtenidos: {len(df)} velas")
                    return df
            
            logger.error(f"No se pudieron obtener datos históricos: {response.text}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error al obtener datos históricos: {e}")
            return pd.DataFrame()
    
    def place_order(self, symbol: str, side: str, size: float, 
                  order_type: str = 'market', price: float = None) -> Dict:
        """
        Coloca una orden en el mercado
        
        Args:
            symbol (str): Símbolo del instrumento
            side (str): Dirección de la orden ('buy' o 'sell')
            size (float): Tamaño de la orden
            order_type (str): Tipo de orden ('market' o 'limit')
            price (float, optional): Precio para órdenes limit
            
        Returns:
            Dict: Resultado de la orden
        """
        if self.mode == 'paper':
            # Simulación de orden en paper trading
            logger.info(f"[PAPER] Orden simulada: {side} {size} {symbol} a precio {'market' if not price else price}")
            
            current_price = self.get_market_price(symbol)
            if not current_price:
                return {"code": "9999", "msg": "No se pudo obtener el precio actual", "data": []}
            
            # Simular orden exitosa
            order_id = f"paper_{int(time.time() * 1000)}"
            executed_price = price if price and order_type == 'limit' else current_price
            
            # Actualizar balance simulado
            if side == 'buy':
                self.position = {
                    'posId': order_id,
                    'instId': symbol,
                    'pos': size,
                    'avgPx': executed_price,
                    'upl': 0.0
                }
            elif side == 'sell' and self.position:
                # Calcular P&L
                entry_price = self.position.get('avgPx', executed_price)
                position_size = self.position.get('pos', 0)
                pnl = (executed_price - entry_price) * position_size if position_size > 0 else 0
                
                # Actualizar balance
                self.balance += pnl
                self.position = None
            
            # Orden simulada exitosa
            result = {
                "code": "0",
                "msg": "",
                "data": [{
                    "ordId": order_id,
                    "clOrdId": f"client_{order_id}",
                    "tag": "",
                    "sCode": "0",
                    "sMsg": ""
                }]
            }
            
            logger.info(f"[PAPER] Orden ejecutada: ID {order_id}, Precio {executed_price}")
            return result
        else:
            # Trading real
            endpoint = '/api/v5/trade/order'
            
            # Preparar parámetros
            params = {
                'instId': symbol,
                'tdMode': 'cross',  # Usar margen cruzado
                'side': side,
                'ordType': order_type,
                'sz': str(size)
            }
            
            if order_type == 'limit' and price:
                params['px'] = str(price)
            
            # Enviar orden
            response = self._request('POST', endpoint, params)
            
            if response.get('code') == '0':
                logger.info(f"Orden colocada: {side} {size} {symbol}")
                return response
            else:
                logger.error(f"Error al colocar orden: {response.get('msg')}")
                return response
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """
        Cancela una orden existente
        
        Args:
            symbol (str): Símbolo del instrumento
            order_id (str): ID de la orden a cancelar
            
        Returns:
            Dict: Resultado de la cancelación
        """
        if self.mode == 'paper':
            # Simulación de cancelación en paper trading
            logger.info(f"[PAPER] Cancelación simulada de orden {order_id} para {symbol}")
            
            # Simular cancelación exitosa
            result = {
                "code": "0", 
                "msg": "", 
                "data": [{
                    "ordId": order_id,
                    "clOrdId": f"client_{order_id}",
                    "sCode": "0",
                    "sMsg": ""
                }]
            }
            
            return result
        else:
            # Cancelación real
            endpoint = '/api/v5/trade/cancel-order'
            
            params = {
                'instId': symbol,
                'ordId': order_id
            }
            
            response = self._request('POST', endpoint, params)
            
            if response.get('code') == '0':
                logger.info(f"Orden {order_id} cancelada")
                return response
            else:
                logger.error(f"Error al cancelar orden: {response.get('msg')}")
                return response
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calcula indicadores técnicos para tomar decisiones
        
        Args:
            df (pd.DataFrame): DataFrame con datos históricos
            
        Returns:
            Dict[str, Any]: Diccionario con indicadores calculados
        """
        if df.empty:
            return {}
        
        # Asegurar que tenemos suficientes datos
        if len(df) < 30:
            logger.warning(f"Datos insuficientes para calcular indicadores: {len(df)} puntos")
            return {}
        
        # Copiar DataFrame para evitar warnings
        df = df.copy()
        
        # Calcular indicadores
        indicators = {}
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Medias móviles
        indicators['sma_20'] = df['close'].rolling(window=20).mean()
        indicators['sma_50'] = df['close'].rolling(window=50).mean()
        indicators['sma_200'] = df['close'].rolling(window=200).mean()
        
        # MACD
        indicators['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        indicators['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
        indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']
        
        # Bollinger Bands
        indicators['sma_20'] = df['close'].rolling(window=20).mean()
        indicators['upper_band'] = indicators['sma_20'] + (df['close'].rolling(window=20).std() * 2)
        indicators['lower_band'] = indicators['sma_20'] - (df['close'].rolling(window=20).std() * 2)
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators['atr'] = true_range.rolling(window=14).mean()
        
        # Retornar solo los indicadores más recientes
        result = {}
        for key, value in indicators.items():
            if isinstance(value, pd.Series):
                result[key] = value.iloc[-1] if not pd.isna(value.iloc[-1]) else None
            else:
                result[key] = value
        
        return result
    
    def analyze_market(self, symbol: str, interval: str = '15m') -> Tuple[str, Dict]:
        """
        Analiza el mercado y genera señales de trading
        
        Args:
            symbol (str): Símbolo del instrumento
            interval (str): Intervalo de tiempo
            
        Returns:
            Tuple[str, Dict]: Señal de trading y detalles
        """
        # Obtener datos históricos
        df = self.get_historical_data(symbol, interval)
        
        if df.empty:
            logger.error("No se pudieron obtener datos para análisis")
            return "neutral", {}
        
        # Calcular indicadores
        indicators = self.calculate_indicators(df)
        
        if not indicators:
            logger.error("No se pudieron calcular indicadores")
            return "neutral", {}
        
        # Analizar RSI
        rsi = indicators.get('rsi')
        rsi_signal = "neutral"
        if rsi is not None:
            if rsi < 30:
                rsi_signal = "buy"  # Sobrevendido
            elif rsi > 70:
                rsi_signal = "sell"  # Sobrecomprado
        
        # Analizar tendencia con medias móviles
        sma_20 = indicators.get('sma_20')
        sma_50 = indicators.get('sma_50')
        trend_signal = "neutral"
        
        if sma_20 is not None and sma_50 is not None:
            if sma_20 > sma_50:
                trend_signal = "buy"  # Tendencia alcista
            elif sma_20 < sma_50:
                trend_signal = "sell"  # Tendencia bajista
        
        # Analizar MACD
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        macd_hist = indicators.get('macd_hist')
        macd_signal_result = "neutral"
        
        if macd is not None and macd_signal is not None and macd_hist is not None:
            if macd > macd_signal and macd_hist > 0:
                macd_signal_result = "buy"  # Señal alcista
            elif macd < macd_signal and macd_hist < 0:
                macd_signal_result = "sell"  # Señal bajista
        
        # Analizar Bollinger Bands
        close = df['close'].iloc[-1] if not df.empty else None
        upper_band = indicators.get('upper_band')
        lower_band = indicators.get('lower_band')
        bb_signal = "neutral"
        
        if close is not None and upper_band is not None and lower_band is not None:
            if close > upper_band:
                bb_signal = "sell"  # Precio por encima de banda superior
            elif close < lower_band:
                bb_signal = "buy"  # Precio por debajo de banda inferior
        
        # Combinar señales
        signals = {
            'rsi': rsi_signal,
            'trend': trend_signal,
            'macd': macd_signal_result,
            'bollinger': bb_signal
        }
        
        # Contar señales
        buy_count = sum(1 for s in signals.values() if s == "buy")
        sell_count = sum(1 for s in signals.values() if s == "sell")
        
        # Determinar señal final
        final_signal = "neutral"
        if buy_count >= 3:
            final_signal = "strong_buy"
        elif buy_count >= 2:
            final_signal = "buy"
        elif sell_count >= 3:
            final_signal = "strong_sell"
        elif sell_count >= 2:
            final_signal = "sell"
        
        # Guardar en estado para análisis
        self.trading_data['strategy_signals'] = signals
        self.trading_data['integrated_signal'] = final_signal
        
        # Preparar detalles para retorno
        details = {
            'price': close,
            'rsi': rsi,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'signals': signals,
            'final_signal': final_signal
        }
        
        logger.info(f"Análisis de mercado: {final_signal} (RSI: {rsi:.2f}, Señales: {signals})")
        return final_signal, details
    
    def execute_strategy(self, symbol: str, interval: str = '15m') -> Dict:
        """
        Ejecuta la estrategia de trading
        
        Args:
            symbol (str): Símbolo del instrumento
            interval (str): Intervalo de tiempo
            
        Returns:
            Dict: Resultado de la ejecución
        """
        # Actualizar precio actual
        current_price = self.get_market_price(symbol)
        if not current_price:
            return {"status": "error", "message": "No se pudo obtener el precio actual"}
        
        # Guardar precio actual en el estado
        self.trading_data['current_price'] = current_price
        self.trading_data['symbol'] = symbol
        self.trading_data['interval'] = interval
        
        # Verificar si ya tenemos una posición abierta
        positions = self.get_positions(symbol)
        
        has_position = False
        position_side = None
        position_size = 0
        entry_price = 0
        
        for position in positions:
            if position['instId'] == symbol and float(position['pos']) != 0:
                has_position = True
                position_side = "buy" if float(position['pos']) > 0 else "sell"
                position_size = abs(float(position['pos']))
                entry_price = float(position['avgPx'])
                # Guardar en estado
                self.position = position
                self.trading_data['position'] = {
                    'side': position_side,
                    'size': position_size,
                    'entry_price': entry_price
                }
                self.trading_data['entry_price'] = entry_price
                break
        
        # Analizar mercado
        signal, details = self.analyze_market(symbol, interval)
        
        # Determinar acción basada en señal y posición
        action = "hold"
        reason = "Señal neutral o insuficiente"
        
        if not has_position:
            # Sin posición, evaluamos entrar
            if signal in ["strong_buy", "buy"]:
                action = "buy"
                reason = f"Señal de compra {signal}"
            elif signal in ["strong_sell", "sell"]:
                action = "sell"
                reason = f"Señal de venta {signal}"
        else:
            # Con posición, evaluamos salir
            if position_side == "buy" and signal in ["strong_sell", "sell"]:
                action = "close_long"
                reason = f"Señal de venta {signal} con posición larga"
            elif position_side == "sell" and signal in ["strong_buy", "buy"]:
                action = "close_short"
                reason = f"Señal de compra {signal} con posición corta"
            
            # Evaluar también Take Profit y Stop Loss
            if position_side == "buy":
                # Calcular P&L actual
                pnl_pct = (current_price - entry_price) / entry_price * 100
                
                # Take Profit (10%)
                if pnl_pct >= 10.0:
                    action = "close_long"
                    reason = f"Take Profit alcanzado: {pnl_pct:.2f}%"
                
                # Stop Loss (-5%)
                elif pnl_pct <= -5.0:
                    action = "close_long"
                    reason = f"Stop Loss alcanzado: {pnl_pct:.2f}%"
            
            elif position_side == "sell":
                # Para posiciones cortas, el P&L es inverso
                pnl_pct = (entry_price - current_price) / entry_price * 100
                
                # Take Profit (10%)
                if pnl_pct >= 10.0:
                    action = "close_short"
                    reason = f"Take Profit alcanzado: {pnl_pct:.2f}%"
                
                # Stop Loss (-5%)
                elif pnl_pct <= -5.0:
                    action = "close_short"
                    reason = f"Stop Loss alcanzado: {pnl_pct:.2f}%"
        
        # Ejecutar acción
        result = {"status": "success", "action": action, "reason": reason, "price": current_price}
        
        if action == "buy":
            # Calcular tamaño de orden (10% del balance)
            account_info = self.get_account_balance()
            
            if account_info:
                # Para trading real
                if self.mode == 'live':
                    available_balance = float(account_info.get('details', [{}])[0].get('availBal', 0))
                    order_size = available_balance * 0.1 / current_price
                else:
                    # Para paper trading
                    order_size = self.balance * 0.1 / current_price
                    logger.info(f"🔄 MODO PAPER: Calculando orden con balance simulado: {self.balance:.2f} USDT")
                
                order_size = round(order_size, 4)  # Redondear a 4 decimales
                
                # Colocar orden
                order_result = self.place_order(symbol, "buy", order_size)
                result["order_result"] = order_result
                
                # Actualizar estado en paper trading
                if self.mode == 'paper' and order_result.get('code') == '0':
                    cost = order_size * current_price
                    self.balance -= cost
                    logger.info(f"🔄 MODO PAPER: Compra simulada: {order_size:.4f} SOL a ${current_price:.2f}")
                    logger.info(f"🔄 MODO PAPER: Reducido balance en {cost:.2f} USDT. Nuevo balance: {self.balance:.2f} USDT")
                
                # Notificar
                notification_msg = f"🟢 COMPRA: {order_size:.4f} {symbol} a ${current_price:.2f}\nRazón: {reason}"
                self.send_notification(notification_msg)
            else:
                result["status"] = "error"
                result["message"] = "No se pudo obtener información de la cuenta"
        
        elif action == "sell":
            # Calcular tamaño de orden (10% del balance)
            account_info = self.get_account_balance()
            
            if account_info:
                # Para trading real
                if self.mode == 'live':
                    available_balance = float(account_info.get('details', [{}])[0].get('availBal', 0))
                    order_size = available_balance * 0.1 / current_price
                else:
                    # Para paper trading
                    order_size = self.balance * 0.1 / current_price
                    logger.info(f"🔄 MODO PAPER: Calculando orden con balance simulado: {self.balance:.2f} USDT")
                
                order_size = round(order_size, 4)  # Redondear a 4 decimales
                
                # Colocar orden
                order_result = self.place_order(symbol, "sell", order_size)
                result["order_result"] = order_result
                
                # Actualizar estado en paper trading (las ventas no reducen el balance hasta que se cierra la posición)
                if self.mode == 'paper' and order_result.get('code') == '0':
                    logger.info(f"🔄 MODO PAPER: Venta simulada: {order_size:.4f} SOL a ${current_price:.2f}")
                    logger.info(f"🔄 MODO PAPER: Posición corta abierta. Balance actual: {self.balance:.2f} USDT")
                
                # Notificar
                notification_msg = f"🔴 VENTA: {order_size:.4f} {symbol} a ${current_price:.2f}\nRazón: {reason}"
                self.send_notification(notification_msg)
            else:
                result["status"] = "error"
                result["message"] = "No se pudo obtener información de la cuenta"
        
        elif action == "close_long":
            # Cerrar posición larga
            if has_position and position_side == "buy":
                order_result = self.place_order(symbol, "sell", position_size)
                result["order_result"] = order_result
                
                # Calcular P&L
                pnl = (current_price - entry_price) * position_size
                pnl_pct = (current_price - entry_price) / entry_price * 100
                
                # Actualizar estadísticas
                self.trading_data['trade_count'] += 1
                if pnl > 0:
                    self.trading_data['wins'] += 1
                else:
                    self.trading_data['losses'] += 1
                
                # Notificar
                notification_msg = f"🔵 CIERRE LARGO: {position_size:.4f} {symbol} a ${current_price:.2f}\nP&L: ${pnl:.2f} ({pnl_pct:.2f}%)\nRazón: {reason}"
                self.send_notification(notification_msg)
            else:
                result["status"] = "error"
                result["message"] = "No hay posición larga para cerrar"
        
        elif action == "close_short":
            # Cerrar posición corta
            if has_position and position_side == "sell":
                order_result = self.place_order(symbol, "buy", position_size)
                result["order_result"] = order_result
                
                # Calcular P&L (inverso para cortos)
                pnl = (entry_price - current_price) * position_size
                pnl_pct = (entry_price - current_price) / entry_price * 100
                
                # Actualizar estadísticas
                self.trading_data['trade_count'] += 1
                if pnl > 0:
                    self.trading_data['wins'] += 1
                else:
                    self.trading_data['losses'] += 1
                
                # Notificar
                notification_msg = f"🟣 CIERRE CORTO: {position_size:.4f} {symbol} a ${current_price:.2f}\nP&L: ${pnl:.2f} ({pnl_pct:.2f}%)\nRazón: {reason}"
                self.send_notification(notification_msg)
            else:
                result["status"] = "error"
                result["message"] = "No hay posición corta para cerrar"
        
        # Log de la acción
        logger.info(f"Acción ejecutada: {action} - {reason} - Precio: {current_price}")
        
        # Guardar el resultado de la estrategia
        self.save_state()
        
        return result
    
    def save_state(self) -> None:
        """
        Guarda el estado actual del bot
        """
        # Actualizar balance y ROI en paper trading
        if self.mode == 'paper':
            self.trading_data['current_balance'] = self.balance
            self.trading_data['roi'] = (self.balance - DEFAULT_INITIAL_BALANCE) / DEFAULT_INITIAL_BALANCE * 100
        
        # Guardar estado en archivo
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(self.trading_data, f, indent=2)
            logger.debug("Estado guardado correctamente")
        except Exception as e:
            logger.error(f"Error al guardar estado: {str(e)}")
    
    def load_state(self) -> None:
        """
        Carga el estado anterior del bot
        """
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    self.trading_data = json.load(f)
                logger.info("Estado anterior cargado correctamente")
                
                # Restaurar balance en paper trading
                if self.mode == 'paper':
                    self.balance = self.trading_data.get('current_balance', DEFAULT_INITIAL_BALANCE)
        except Exception as e:
            logger.error(f"Error al cargar estado anterior: {str(e)}")
    
    def send_notification(self, message: str) -> None:
        """
        Envía notificación vía Telegram
        
        Args:
            message (str): Mensaje a enviar
        """
        if not TELEGRAM_AVAILABLE:
            logger.debug("Librería Telebot no disponible, no se enviarán notificaciones")
            return
        
        try:
            # Cargar configuración
            config = load_config()
            bot_token = config.get('TELEGRAM_BOT_TOKEN')
            chat_id = config.get('TELEGRAM_CHAT_ID')
            
            if not bot_token or not chat_id:
                logger.debug("Token de Telegram o Chat ID no configurados")
                return
            
            # Enviar mensaje
            bot = telebot.TeleBot(bot_token)
            bot.send_message(chat_id, message, parse_mode='Markdown')
            logger.debug("Notificación enviada correctamente")
        except Exception as e:
            logger.error(f"Error al enviar notificación: {str(e)}")
    
    def run(self, symbol: str = DEFAULT_SYMBOL, interval: str = DEFAULT_INTERVAL, 
          notify: bool = False, continuous: bool = False) -> None:
        """
        Ejecuta el bot de trading
        
        Args:
            symbol (str): Símbolo del instrumento
            interval (str): Intervalo de tiempo
            notify (bool): Activar notificaciones
            continuous (bool): Ejecutar en modo continuo
        """
        logger.info(f"Iniciando bot para {symbol} en intervalo {interval} (Modo: {self.mode})")
        
        # Validar credenciales
        if not self.api_key or not self.api_secret or not self.passphrase:
            logger.error("Credenciales API incompletas")
            if notify:
                self.send_notification("⚠️ Error: Credenciales API incompletas")
            return
        
        # Cargar estado anterior
        self.load_state()
        
        # Enviar notificación de inicio
        if notify:
            mode_str = "PAPER TRADING" if self.mode == 'paper' else "TRADING REAL"
            start_msg = f"""🤖 *BOT DE TRADING SOLANA - INICIADO* 🤖
            
⏰ Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
💰 Modo: {mode_str}
📊 Par: {symbol}
⏱️ Intervalo: {interval}

*Estado: ACTIVO y monitoreando mercado* ✅

_El bot enviará notificaciones de operaciones..._"""
            self.send_notification(start_msg)
        
        # Ciclo principal
        try:
            if continuous:
                # Modo continuo con intervalo adaptativo
                while True:
                    start_time = time.time()
                    
                    try:
                        # Ejecutar estrategia
                        self.execute_strategy(symbol, interval)
                    except Exception as e:
                        logger.error(f"Error en ciclo de ejecución: {str(e)}", exc_info=True)
                        if notify:
                            self.send_notification(f"⚠️ Error en bot: {str(e)}")
                    
                    # Calcular tiempo de espera según intervalo
                    wait_time = self._get_wait_time(interval)
                    elapsed = time.time() - start_time
                    sleep_time = max(5, wait_time - elapsed)  # Al menos 5 segundos
                    
                    logger.info(f"Esperando {sleep_time:.1f} segundos hasta próxima ejecución...")
                    time.sleep(sleep_time)
            else:
                # Modo única ejecución
                self.execute_strategy(symbol, interval)
                logger.info("Ejecución única completada")
        
        except KeyboardInterrupt:
            logger.info("Bot detenido manualmente")
            if notify:
                self.send_notification("🛑 Bot detenido manualmente")
        
        except Exception as e:
            logger.error(f"Error crítico en el bot: {str(e)}", exc_info=True)
            if notify:
                self.send_notification(f"❌ Error crítico: {str(e)}")
    
    def _get_wait_time(self, interval: str) -> int:
        """
        Calcula tiempo de espera según intervalo
        
        Args:
            interval (str): Intervalo de tiempo
            
        Returns:
            int: Tiempo de espera en segundos
        """
        # Mapear intervalos a segundos
        interval_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        
        # Usar tiempo predeterminado si el intervalo no es válido
        return interval_map.get(interval, 900)

def load_config(config_file: str = CONFIG_FILE) -> Dict[str, str]:
    """
    Carga configuración desde archivo .env
    
    Args:
        config_file (str): Ruta al archivo de configuración
        
    Returns:
        Dict[str, str]: Configuración cargada
    """
    config = {}
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip().strip('"\'')
            logger.info(f"Configuración cargada desde {config_file}")
        else:
            logger.warning(f"Archivo de configuración {config_file} no encontrado")
    except Exception as e:
        logger.error(f"Error al cargar configuración: {str(e)}")
    
    return config

def parse_args():
    """
    Parsea argumentos de línea de comandos
    
    Returns:
        argparse.Namespace: Argumentos parseados
    """
    parser = argparse.ArgumentParser(description='Bot de Trading de Solana')
    parser.add_argument('--mode', type=str, default=DEFAULT_MODE, 
                      choices=['paper', 'live'], 
                      help='Modo de trading (paper o live)')
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL,
                      help='Símbolo de trading (e.j., SOL-USDT)')
    parser.add_argument('--interval', type=str, default=DEFAULT_INTERVAL,
                      choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                      help='Intervalo de tiempo para el análisis')
    parser.add_argument('--notify', action='store_true',
                      help='Activar notificaciones')
    parser.add_argument('--continuous', action='store_true',
                      help='Ejecutar en modo continuo')
    parser.add_argument('--interactive', action='store_true',
                      help='Iniciar en modo interactivo')
    
    return parser.parse_args()

def interactive_mode():
    """
    Inicia el bot en modo interactivo con menú
    """
    try:
        from enhanced_bot import main_menu
        main_menu()
    except ImportError:
        try:
            from bot_interactivo import main_menu
            main_menu()
        except ImportError:
            logger.error("No se encontró módulo para modo interactivo")
            print("Error: No se encontró módulo para modo interactivo")
            print("Por favor, asegúrate de tener enhanced_bot.py o bot_interactivo.py")

def main():
    """
    Función principal
    """
    # Parsear argumentos
    args = parse_args()
    
    # Si está en modo interactivo, abrir ese modo
    if args.interactive:
        interactive_mode()
        return
    
    # Cargar configuración
    config = load_config()
    
    # Verificar credenciales
    api_key = config.get('OKX_API_KEY', '')
    api_secret = config.get('OKX_API_SECRET', '')
    passphrase = config.get('OKX_PASSPHRASE', '')
    
    if not api_key or not api_secret or not passphrase:
        logger.error("Credenciales de API no encontradas en config.env")
        print("Error: Credenciales de API no encontradas en config.env")
        print("Por favor, configura OKX_API_KEY, OKX_API_SECRET y OKX_PASSPHRASE")
        return
    
    # Inicializar bot
    bot = TradingBot(
        api_key=api_key,
        api_secret=api_secret,
        passphrase=passphrase,
        mode=args.mode
    )
    
    # Ejecutar bot
    bot.run(
        symbol=args.symbol,
        interval=args.interval,
        notify=args.notify,
        continuous=args.continuous
    )

if __name__ == "__main__":
    main()