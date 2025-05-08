"""
Cliente WebSocket para OKX
Este módulo implementa una conexión WebSocket con OKX para datos en tiempo real.
"""

import json
import time
import hmac
import base64
import hashlib
import threading
import logging
from datetime import datetime
from typing import Dict, List, Callable, Optional, Any
import websocket

# Configurar logging
logger = logging.getLogger("WebSocketClient")

class OKXWebSocketClient:
    """
    Cliente WebSocket para OKX que maneja la conexión y los mensajes.
    """
    
    # URLs de WebSocket para OKX
    PUBLIC_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
    PRIVATE_WS_URL = "wss://ws.okx.com:8443/ws/v5/private"
    
    def __init__(self, api_key: str = "", api_secret: str = "", passphrase: str = "", 
                 is_paper_trading: bool = True):
        """
        Inicializa el cliente WebSocket
        
        Args:
            api_key: API key de OKX
            api_secret: API secret de OKX
            passphrase: Passphrase de la API de OKX
            is_paper_trading: Si es True, se conecta al entorno de pruebas
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.is_paper_trading = is_paper_trading
        
        # Websocket connections
        self.public_ws = None
        self.private_ws = None
        
        # Callback functions
        self.on_ticker_callback = None
        self.on_kline_callback = None
        self.on_orderbook_callback = None
        self.on_trade_callback = None
        self.on_account_callback = None
        self.on_position_callback = None
        self.on_order_callback = None
        
        # Control flags
        self.keep_running = False
        self.connected_public = False
        self.connected_private = False
        
        # Management threads
        self.public_thread = None
        self.private_thread = None
        self.ping_thread = None
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, 
                           body: str = "") -> str:
        """
        Genera la firma para la autenticación con OKX
        
        Args:
            timestamp: Timestamp en formato ISO
            method: Método HTTP (GET, POST, etc.)
            request_path: Ruta de la petición
            body: Cuerpo de la petición
            
        Returns:
            str: Firma codificada en base64
        """
        if not body:
            message = timestamp + method + request_path
        else:
            message = timestamp + method + request_path + body
            
        mac = hmac.new(
            bytes(self.api_secret, encoding='utf-8'),
            bytes(message, encoding='utf-8'),
            digestmod=hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode('utf-8')
    
    def _get_login_params(self) -> Dict:
        """
        Obtiene los parámetros para la autenticación WebSocket
        
        Returns:
            Dict: Parámetros de login para WebSocket
        """
        timestamp = datetime.utcnow().isoformat()[:-3] + 'Z'
        sign = self._generate_signature(timestamp, 'GET', '/users/self/verify')
        
        return {
            "op": "login",
            "args": [
                {
                    "apiKey": self.api_key,
                    "passphrase": self.passphrase,
                    "timestamp": timestamp,
                    "sign": sign
                }
            ]
        }
    
    def _on_public_message(self, ws, message):
        """Callback para mensajes públicos"""
        try:
            data = json.loads(message)
            
            # Manejar mensaje de ping
            if 'event' in data and data['event'] == 'ping':
                self._send_pong(ws)
                return
                
            # Manejar datos de canales
            if 'data' in data:
                channel = data.get('arg', {}).get('channel')
                
                if channel == 'tickers':
                    if self.on_ticker_callback:
                        self.on_ticker_callback(data['data'])
                
                elif channel == 'candle1m' or channel == 'candle5m' or channel == 'candle15m' or channel == 'candle1H':
                    if self.on_kline_callback:
                        self.on_kline_callback(data['data'], channel)
                
                elif channel == 'books' or channel == 'books5' or channel == 'books-l2-tbt':
                    if self.on_orderbook_callback:
                        self.on_orderbook_callback(data['data'])
                
                elif channel == 'trades':
                    if self.on_trade_callback:
                        self.on_trade_callback(data['data'])
        
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON message: {message}")
        except Exception as e:
            logger.error(f"Error processing public message: {e}")
    
    def _on_private_message(self, ws, message):
        """Callback para mensajes privados"""
        try:
            data = json.loads(message)
            
            # Manejar mensaje de login
            if 'event' in data and data['event'] == 'login':
                if data.get('code') == '0':
                    logger.info("WebSocket private login successful")
                    self.connected_private = True
                else:
                    logger.error(f"WebSocket private login failed: {data}")
                return
            
            # Manejar mensaje de ping
            if 'event' in data and data['event'] == 'ping':
                self._send_pong(ws)
                return
            
            # Manejar datos de canales
            if 'data' in data:
                channel = data.get('arg', {}).get('channel')
                
                if channel == 'account':
                    if self.on_account_callback:
                        self.on_account_callback(data['data'])
                
                elif channel == 'positions':
                    if self.on_position_callback:
                        self.on_position_callback(data['data'])
                
                elif channel == 'orders':
                    if self.on_order_callback:
                        self.on_order_callback(data['data'])
        
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON message: {message}")
        except Exception as e:
            logger.error(f"Error processing private message: {e}")
    
    def _on_public_error(self, ws, error):
        """Callback para errores públicos"""
        logger.error(f"Public WebSocket error: {error}")
    
    def _on_private_error(self, ws, error):
        """Callback para errores privados"""
        logger.error(f"Private WebSocket error: {error}")
    
    def _on_public_close(self, ws, close_status_code, close_msg):
        """Callback para cierre de conexión pública"""
        logger.info(f"Public WebSocket connection closed: {close_msg}")
        self.connected_public = False
        
        # Reconectar si es necesario
        if self.keep_running:
            logger.info("Attempting to reconnect public WebSocket...")
            self._connect_public()
    
    def _on_private_close(self, ws, close_status_code, close_msg):
        """Callback para cierre de conexión privada"""
        logger.info(f"Private WebSocket connection closed: {close_msg}")
        self.connected_private = False
        
        # Reconectar si es necesario
        if self.keep_running:
            logger.info("Attempting to reconnect private WebSocket...")
            self._connect_private()
    
    def _on_public_open(self, ws):
        """Callback para apertura de conexión pública"""
        logger.info("Public WebSocket connection opened")
        self.connected_public = True
    
    def _on_private_open(self, ws):
        """Callback para apertura de conexión privada"""
        logger.info("Private WebSocket connection opened")
        # Enviar login para conexión privada
        login_params = self._get_login_params()
        ws.send(json.dumps(login_params))
    
    def _send_pong(self, ws):
        """Envía mensaje de pong"""
        pong_msg = {"op": "pong"}
        ws.send(json.dumps(pong_msg))
    
    def _connect_public(self):
        """Establece conexión WebSocket pública"""
        url = self.PUBLIC_WS_URL
        
        # Agregar flag de paper trading si es necesario
        if self.is_paper_trading:
            url = url.replace('ws.okx.com', 'wspap.okx.com')
        
        self.public_ws = websocket.WebSocketApp(
            url,
            on_open=self._on_public_open,
            on_message=self._on_public_message,
            on_error=self._on_public_error,
            on_close=self._on_public_close
        )
    
    def _connect_private(self):
        """Establece conexión WebSocket privada"""
        if not self.api_key or not self.api_secret or not self.passphrase:
            logger.warning("API credentials not provided, skipping private WebSocket connection")
            return
        
        url = self.PRIVATE_WS_URL
        
        # Agregar flag de paper trading si es necesario
        if self.is_paper_trading:
            url = url.replace('ws.okx.com', 'wspap.okx.com')
        
        self.private_ws = websocket.WebSocketApp(
            url,
            on_open=self._on_private_open,
            on_message=self._on_private_message,
            on_error=self._on_private_error,
            on_close=self._on_private_close
        )
    
    def _run_public_ws(self):
        """Ejecuta el WebSocket público en un loop"""
        while self.keep_running:
            try:
                self.public_ws.run_forever()
            except Exception as e:
                logger.error(f"Public WebSocket error: {e}")
            
            # Si debe seguir ejecutándose, esperar y reconectar
            if self.keep_running:
                time.sleep(5)  # Esperar 5 segundos antes de reconectar
                self._connect_public()
    
    def _run_private_ws(self):
        """Ejecuta el WebSocket privado en un loop"""
        if not self.private_ws:
            return
        
        while self.keep_running:
            try:
                self.private_ws.run_forever()
            except Exception as e:
                logger.error(f"Private WebSocket error: {e}")
            
            # Si debe seguir ejecutándose, esperar y reconectar
            if self.keep_running:
                time.sleep(5)  # Esperar 5 segundos antes de reconectar
                self._connect_private()
    
    def _ping_loop(self):
        """Envía ping periódico para mantener conexiones activas"""
        while self.keep_running:
            try:
                if self.connected_public:
                    ping_msg = {"op": "ping"}
                    self.public_ws.send(json.dumps(ping_msg))
                
                if self.connected_private and self.private_ws:
                    ping_msg = {"op": "ping"}
                    self.private_ws.send(json.dumps(ping_msg))
            except Exception as e:
                logger.error(f"Error sending ping: {e}")
            
            time.sleep(20)  # Ping cada 20 segundos
    
    def start(self):
        """Inicia las conexiones WebSocket"""
        if self.keep_running:
            logger.warning("WebSocket client already running")
            return
        
        self.keep_running = True
        
        # Conexión pública
        self._connect_public()
        self.public_thread = threading.Thread(target=self._run_public_ws)
        self.public_thread.daemon = True
        self.public_thread.start()
        
        # Conexión privada (si hay credenciales)
        if self.api_key and self.api_secret and self.passphrase:
            self._connect_private()
            self.private_thread = threading.Thread(target=self._run_private_ws)
            self.private_thread.daemon = True
            self.private_thread.start()
        
        # Thread de ping
        self.ping_thread = threading.Thread(target=self._ping_loop)
        self.ping_thread.daemon = True
        self.ping_thread.start()
        
        logger.info("WebSocket client started")
    
    def stop(self):
        """Detiene las conexiones WebSocket"""
        self.keep_running = False
        
        if self.public_ws:
            self.public_ws.close()
        
        if self.private_ws:
            self.private_ws.close()
        
        logger.info("WebSocket client stopped")
    
    def subscribe_tickers(self, symbols: List[str]):
        """Suscribe a tickers (precios en tiempo real)"""
        if not self.connected_public:
            logger.warning("Public WebSocket not connected, can't subscribe to tickers")
            return
        
        args = []
        for symbol in symbols:
            args.append({
                "channel": "tickers",
                "instId": symbol
            })
        
        sub_msg = {
            "op": "subscribe",
            "args": args
        }
        
        self.public_ws.send(json.dumps(sub_msg))
        logger.info(f"Subscribed to tickers for {symbols}")
    
    def subscribe_klines(self, symbols: List[str], intervals: List[str]):
        """
        Suscribe a velas (klines)
        
        Args:
            symbols: Lista de pares (ej. ["BTC-USDT", "ETH-USDT"])
            intervals: Lista de intervalos (ej. ["1m", "5m", "15m", "1H", "4H", "1D"])
        """
        if not self.connected_public:
            logger.warning("Public WebSocket not connected, can't subscribe to klines")
            return
        
        # Mapear intervalos al formato de OKX
        interval_map = {
            "1m": "candle1m",
            "5m": "candle5m",
            "15m": "candle15m",
            "30m": "candle30m",
            "1h": "candle1H",
            "4h": "candle4H",
            "1d": "candle1D"
        }
        
        args = []
        for symbol in symbols:
            for interval in intervals:
                channel = interval_map.get(interval.lower())
                if channel:
                    args.append({
                        "channel": channel,
                        "instId": symbol
                    })
        
        sub_msg = {
            "op": "subscribe",
            "args": args
        }
        
        self.public_ws.send(json.dumps(sub_msg))
        logger.info(f"Subscribed to klines for {symbols} with intervals {intervals}")
    
    def subscribe_orderbooks(self, symbols: List[str], depth: str = "5"):
        """
        Suscribe a libros de órdenes
        
        Args:
            symbols: Lista de pares (ej. ["BTC-USDT", "ETH-USDT"])
            depth: Profundidad de libro ("5", "400", "full")
        """
        if not self.connected_public:
            logger.warning("Public WebSocket not connected, can't subscribe to orderbooks")
            return
        
        # Mapear profundidad al canal correcto
        if depth == "5":
            channel = "books5"  # 5 mejores niveles
        elif depth == "full":
            channel = "books-l2-tbt"  # Libro completo tick by tick
        else:
            channel = "books"  # Profundidad estándar (400 niveles)
        
        args = []
        for symbol in symbols:
            args.append({
                "channel": channel,
                "instId": symbol
            })
        
        sub_msg = {
            "op": "subscribe",
            "args": args
        }
        
        self.public_ws.send(json.dumps(sub_msg))
        logger.info(f"Subscribed to orderbooks for {symbols} with depth {depth}")
    
    def subscribe_trades(self, symbols: List[str]):
        """Suscribe a trades en tiempo real"""
        if not self.connected_public:
            logger.warning("Public WebSocket not connected, can't subscribe to trades")
            return
        
        args = []
        for symbol in symbols:
            args.append({
                "channel": "trades",
                "instId": symbol
            })
        
        sub_msg = {
            "op": "subscribe",
            "args": args
        }
        
        self.public_ws.send(json.dumps(sub_msg))
        logger.info(f"Subscribed to trades for {symbols}")
    
    def subscribe_account(self):
        """Suscribe a actualizaciones de cuenta"""
        if not self.connected_private or not self.private_ws:
            logger.warning("Private WebSocket not connected, can't subscribe to account")
            return
        
        sub_msg = {
            "op": "subscribe",
            "args": [
                {
                    "channel": "account"
                }
            ]
        }
        
        self.private_ws.send(json.dumps(sub_msg))
        logger.info("Subscribed to account updates")
    
    def subscribe_positions(self, symbols: Optional[List[str]] = None):
        """
        Suscribe a actualizaciones de posiciones
        
        Args:
            symbols: Lista de pares o None para todas las posiciones
        """
        if not self.connected_private or not self.private_ws:
            logger.warning("Private WebSocket not connected, can't subscribe to positions")
            return
        
        args = []
        if symbols:
            for symbol in symbols:
                args.append({
                    "channel": "positions",
                    "instId": symbol
                })
        else:
            args.append({
                "channel": "positions"
            })
        
        sub_msg = {
            "op": "subscribe",
            "args": args
        }
        
        self.private_ws.send(json.dumps(sub_msg))
        logger.info("Subscribed to position updates")
    
    def subscribe_orders(self, symbols: Optional[List[str]] = None):
        """
        Suscribe a actualizaciones de órdenes
        
        Args:
            symbols: Lista de pares o None para todas las órdenes
        """
        if not self.connected_private or not self.private_ws:
            logger.warning("Private WebSocket not connected, can't subscribe to orders")
            return
        
        args = []
        if symbols:
            for symbol in symbols:
                args.append({
                    "channel": "orders",
                    "instId": symbol
                })
        else:
            args.append({
                "channel": "orders"
            })
        
        sub_msg = {
            "op": "subscribe",
            "args": args
        }
        
        self.private_ws.send(json.dumps(sub_msg))
        logger.info("Subscribed to order updates")
    
    def register_ticker_callback(self, callback: Callable[[List[Dict]], None]):
        """Registra callback para tickers"""
        self.on_ticker_callback = callback
    
    def register_kline_callback(self, callback: Callable[[List[Dict], str], None]):
        """Registra callback para velas"""
        self.on_kline_callback = callback
    
    def register_orderbook_callback(self, callback: Callable[[List[Dict]], None]):
        """Registra callback para libros de órdenes"""
        self.on_orderbook_callback = callback
    
    def register_trade_callback(self, callback: Callable[[List[Dict]], None]):
        """Registra callback para trades"""
        self.on_trade_callback = callback
    
    def register_account_callback(self, callback: Callable[[List[Dict]], None]):
        """Registra callback para cuenta"""
        self.on_account_callback = callback
    
    def register_position_callback(self, callback: Callable[[List[Dict]], None]):
        """Registra callback para posiciones"""
        self.on_position_callback = callback
    
    def register_order_callback(self, callback: Callable[[List[Dict]], None]):
        """Registra callback para órdenes"""
        self.on_order_callback = callback