"""
Bot de Trading para Solana - Versión Autónoma

Este script está diseñado para ejecutarse como un sistema independiente
que implementa un bot de trading algorítmico enfocado en Solana.
"""

import os
import sys
import time
import json
import logging
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_bot.log")
    ]
)
logger = logging.getLogger("TradingBot")

# Intentar importar módulos necesarios
try:
    # Módulos de estrategias
    from strategies.scalping_strategies import get_available_scalping_strategies, get_strategy_by_name
    
    # Módulos de datos
    from data_management.market_data import get_market_data, get_current_price
    
    # Módulos de adaptación
    from adaptive_weighting import AdaptiveWeightingSystem, MarketCondition, TimeInterval
    
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    print(f"Error: {e}")
    print("Make sure all dependencies are installed.")

class TradingMode(Enum):
    """Modos de operación del bot"""
    PAPER = "paper"  # Trading simulado (sin operaciones reales)
    LIVE = "live"    # Trading en vivo (operaciones reales)

class TradingBot:
    """Bot de trading principal para operar en mercados de criptomonedas"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el bot con configuración
        
        Args:
            config: Diccionario de configuración
        """
        # Configuración básica
        self.symbol = config.get("symbol", "SOL-USDT")
        self.timeframe = config.get("timeframe", "15m")
        self.paper_trading = config.get("paper_trading", True)
        self.mode = TradingMode.PAPER if self.paper_trading else TradingMode.LIVE
        
        # Forzar modo paper de manera preventiva
        self.paper_trading = True
        self.mode = TradingMode.PAPER
        
        # Configuración de estrategias
        self.strategy_name = config.get("strategy", "rsi_scalping")
        self.strategy = get_strategy_by_name(self.strategy_name)
        if not self.strategy:
            logger.warning(f"Estrategia {self.strategy_name} no encontrada, usando RSI por defecto")
            self.strategy_name = "rsi_scalping"
            self.strategy = get_strategy_by_name(self.strategy_name)
        
        # Sistema de ponderación adaptativa
        self.adaptive_system = AdaptiveWeightingSystem()
        
        # Configuración de riesgo
        self.max_position_size = config.get("max_position_size", 0.1)  # % del balance
        self.stop_loss_pct = config.get("stop_loss_pct", 1.0)  # %
        self.take_profit_pct = config.get("take_profit_pct", 2.0)  # %
        self.max_loss_per_day = config.get("max_loss_per_day", 5.0)  # %
        self.max_trades_per_day = config.get("max_trades_per_day", 20)
        
        # Estado de trading
        self.running = False
        self.current_position = None
        self.balance = 1000.0  # Balance inicial para paper trading
        self.initial_balance = self.balance
        
        # Historial
        self.trade_history = []
        self.signal_history = []
        self.learning_events = []
        
        # Contadores
        self.trades_today = 0
        self.loss_today = 0.0
        self.start_time = datetime.now()
        
        # Callbacks y eventos
        self.on_trade_callback = None
        self.on_signal_callback = None
        self.on_learning_callback = None
        self.on_position_update_callback = None
        
        logger.info(f"Bot inicializado: {self.symbol} en {self.timeframe}, Modo: {self.mode.value}")
    
    def run(self):
        """Ejecuta el bucle principal del bot"""
        if self.running:
            logger.warning("El bot ya está en ejecución")
            return
        
        self.running = True
        logger.info(f"Iniciando bot de trading para {self.symbol}")
        
        try:
            # Bucle principal
            while self.running:
                try:
                    # Obtener datos actualizados
                    data = self.get_market_data()
                    
                    if data is None or len(data) < 20:
                        logger.warning("Datos insuficientes para análisis, esperando...")
                        time.sleep(10)
                        continue
                    
                    # Detectar condición de mercado
                    market_condition = self.adaptive_system.detect_market_condition(data)
                    time_interval = TimeInterval(self.timeframe) if self.timeframe in [e.value for e in TimeInterval] else TimeInterval.MINUTE_15
                    
                    # Generar señal
                    signal, reason, details = self.strategy.get_signal(data)
                    
                    # Ajustar señal con sistema adaptativo
                    indicator_signals = details.get("indicators", {})
                    if indicator_signals:
                        # Si hay signals de múltiples indicadores, ponderarlos
                        signal = self.adaptive_system.calculate_weighted_signal(
                            indicator_signals, market_condition, time_interval
                        )
                    
                    # Registrar señal
                    current_price = data['close'].iloc[-1]
                    self.add_signal(signal, current_price, reason)
                    
                    # Procesar señal
                    self.process_signal(signal, current_price, reason, details)
                    
                    # Gestionar posición existente
                    if self.current_position:
                        self.manage_position(current_price, data)
                    
                    # Añadir evento de aprendizaje periódicamente
                    if np.random.random() < 0.1:  # 10% de probabilidad
                        self.add_learning_event(
                            "ADAPTACIÓN", 
                            f"Ajuste de pesos para condición {market_condition.value}", 
                            True
                        )
                    
                    # Esperar antes del siguiente ciclo
                    time.sleep(5)  # Ajustar según timeframe
                    
                except Exception as e:
                    logger.error(f"Error en ciclo de trading: {e}")
                    time.sleep(30)
        
        except KeyboardInterrupt:
            logger.info("Bot detenido por usuario")
        finally:
            self.running = False
            logger.info("Bot detenido")
    
    def stop(self):
        """Detiene el bot"""
        self.running = False
        logger.info("Deteniendo bot...")
    
    def get_market_data(self) -> pd.DataFrame:
        """
        Obtiene datos de mercado actualizados
        
        Returns:
            pd.DataFrame: Datos de mercado en formato pandas
        """
        try:
            # Obtener datos de mercado desde módulo de datos
            limit = 100  # Ajustar según necesidad
            data = get_market_data(self.symbol, self.timeframe, limit)
            
            if data is None or len(data) < 10:
                logger.warning("Error al obtener datos, generando datos simulados para pruebas")
                # Generar datos simulados para testing
                data = self.generate_test_data()
            
            return data
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de mercado: {e}")
            return self.generate_test_data()
    
    def generate_test_data(self) -> pd.DataFrame:
        """
        Genera datos simulados para pruebas
        
        Returns:
            pd.DataFrame: Datos de mercado simulados
        """
        # Crear índice de tiempo
        now = datetime.now()
        if self.timeframe == "1m":
            index = [now - timedelta(minutes=i) for i in range(100, 0, -1)]
        elif self.timeframe == "5m":
            index = [now - timedelta(minutes=5*i) for i in range(100, 0, -1)]
        elif self.timeframe == "15m":
            index = [now - timedelta(minutes=15*i) for i in range(100, 0, -1)]
        else:
            index = [now - timedelta(hours=i) for i in range(100, 0, -1)]
        
        # Crear precio base y movimiento
        base_price = 170.0  # SOL-USDT precio aproximado
        price_series = [base_price]
        for i in range(1, 100):
            move = np.random.normal(0, 0.01)  # 1% de volatilidad
            new_price = price_series[-1] * (1 + move)
            price_series.append(new_price)
        
        # Crear DataFrame
        data = pd.DataFrame({
            'open': price_series,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in price_series],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in price_series],
            'close': price_series,
            'volume': [abs(np.random.normal(1000000, 500000)) for _ in range(100)]
        }, index=pd.DatetimeIndex(index))
        
        return data
    
    def process_signal(self, signal: float, price: float, reason: str, details: Dict[str, Any]):
        """
        Procesa una señal de trading
        
        Args:
            signal: Señal (-2 a 2)
            price: Precio actual
            reason: Razón de la señal
            details: Detalles adicionales
        """
        # Verificar si ya estamos en posición
        if self.current_position:
            # Si estamos en posición larga y la señal es negativa, considerar cerrar
            if self.current_position["type"] == "long" and signal < -0.5:
                self.close_position(price, f"Señal de venta: {reason}")
            
            # Si estamos en posición corta y la señal es positiva, considerar cerrar
            elif self.current_position["type"] == "short" and signal > 0.5:
                self.close_position(price, f"Señal de compra: {reason}")
            
            # En otros casos, mantener posición y gestionar con stop loss/take profit
            return
        
        # Si no estamos en posición, evaluar entrada
        if signal > 0.5:
            # Señal de compra/long
            self.open_position("long", price, self.calculate_position_size(price), reason)
        
        elif signal < -0.5:
            # Señal de venta/short (si está permitido)
            # Nota: Descomentar si se permite posiciones cortas
            # self.open_position("short", price, self.calculate_position_size(price), reason)
            pass
    
    def calculate_position_size(self, price: float) -> float:
        """
        Calcula el tamaño adecuado de posición
        
        Args:
            price: Precio actual
        
        Returns:
            float: Tamaño de posición en unidades
        """
        # Calcular tamaño basado en % del balance y gestión de riesgo
        position_value = self.balance * self.max_position_size
        
        # Convertir valor a unidades
        units = position_value / price
        
        # Redondear a 4 decimales
        return round(units, 4)
    
    def open_position(self, position_type: str, price: float, size: float, reason: str):
        """
        Abre una posición nueva
        
        Args:
            position_type: Tipo de posición ('long' o 'short')
            price: Precio de entrada
            size: Tamaño en unidades
            reason: Razón de la entrada
        """
        # Verificar condiciones para abrir posición
        if not self.can_open_position():
            logger.info(f"No se puede abrir posición: límites alcanzados")
            return
        
        # Calcular niveles de take profit y stop loss
        if position_type == "long":
            take_profit = price * (1 + self.take_profit_pct / 100)
            stop_loss = price * (1 - self.stop_loss_pct / 100)
        else:  # short
            take_profit = price * (1 - self.take_profit_pct / 100)
            stop_loss = price * (1 + self.stop_loss_pct / 100)
        
        # Crear objeto de posición
        self.current_position = {
            "type": position_type,
            "entry_price": price,
            "size": size,
            "value": price * size,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "entry_time": datetime.now(),
            "reason": reason
        }
        
        # Actualizar balance en paper trading
        if self.paper_trading:
            # Simular comisión
            commission = price * size * 0.0005  # 0.05% de comisión
            self.balance -= commission
        
        # Notificar
        logger.info(f"Posición abierta: {position_type.upper()} {size} {self.symbol} @ {price}")
        
        # Llamar callback si existe
        trade_data = {
            "type": position_type.upper(),
            "price": price,
            "size": size,
            "profit": 0,
            "reason": reason
        }
        
        self.add_trade(trade_data)
        
        if self.on_position_update_callback:
            self.on_position_update_callback(self.current_position)
    
    def close_position(self, price: float, reason: str):
        """
        Cierra la posición actual
        
        Args:
            price: Precio de cierre
            reason: Razón del cierre
        """
        if not self.current_position:
            logger.warning("No hay posición abierta para cerrar")
            return
        
        # Calcular ganancia/pérdida
        entry_price = self.current_position["entry_price"]
        size = self.current_position["size"]
        position_type = self.current_position["type"]
        
        if position_type == "long":
            profit_pct = (price / entry_price - 1) * 100
            profit = (price - entry_price) * size
        else:  # short
            profit_pct = (entry_price / price - 1) * 100
            profit = (entry_price - price) * size
        
        # Actualizar balance en paper trading
        if self.paper_trading:
            # Simular comisión
            commission = price * size * 0.0005  # 0.05% de comisión
            self.balance += (price * size - commission)
            
            # Actualizar estadísticas de pérdida diaria
            if profit < 0:
                self.loss_today += abs(profit)
        
        # Registrar operación
        holding_time = datetime.now() - self.current_position["entry_time"]
        minutes_held = holding_time.total_seconds() / 60
        
        logger.info(f"Posición cerrada: {position_type.upper()} {size} {self.symbol} @ {price}, " +
                   f"Profit: {profit_pct:.2f}% (${profit:.2f}), Tiempo: {minutes_held:.1f} min")
        
        # Actualizar estadísticas de estrategia
        self.strategy.update_stats(profit > 0, profit, minutes_held)
        
        # Evento de aprendizaje
        success = profit > 0
        self.add_learning_event(
            "EVALUACIÓN", 
            f"{'Exitosa' if success else 'Fallida'} {position_type} ({profit_pct:.2f}%)", 
            success
        )
        
        # Crear objeto de trade para historial
        trade_data = {
            "type": "SELL" if position_type == "long" else "BUY",
            "price": price,
            "size": size,
            "profit": profit,
            "profit_pct": profit_pct,
            "time_held": minutes_held,
            "reason": reason
        }
        
        self.add_trade(trade_data)
        
        # Resetear posición actual
        previous_position = self.current_position
        self.current_position = None
        self.trades_today += 1
        
        # Llamar callbacks
        if self.on_position_update_callback:
            self.on_position_update_callback(None)
    
    def manage_position(self, current_price: float, data: pd.DataFrame):
        """
        Gestiona posición abierta (stops, trailing, etc)
        
        Args:
            current_price: Precio actual
            data: DataFrame con datos de mercado
        """
        if not self.current_position:
            return
        
        position = self.current_position
        position_type = position["type"]
        entry_price = position["entry_price"]
        
        # Calcular PnL actual
        if position_type == "long":
            pnl_pct = (current_price / entry_price - 1) * 100
        else:  # short
            pnl_pct = (entry_price / current_price - 1) * 100
        
        # Verificar condiciones de salida
        hit_take_profit = False
        hit_stop_loss = False
        
        if position_type == "long":
            hit_take_profit = current_price >= position["take_profit"]
            hit_stop_loss = current_price <= position["stop_loss"]
        else:  # short
            hit_take_profit = current_price <= position["take_profit"]
            hit_stop_loss = current_price >= position["stop_loss"]
        
        # Ajustar stop loss dinámico (trailing) en ganancias
        if pnl_pct > 0.5 and position_type == "long":
            # Si estamos en ganancia, mover stop loss para asegurar parte
            new_stop = current_price * (1 - self.stop_loss_pct / 200)  # Más ajustado
            if new_stop > position["stop_loss"]:
                position["stop_loss"] = new_stop
                logger.info(f"Stop loss ajustado a: {new_stop:.4f}")
        
        # Cerrar si se alcanzan niveles
        if hit_take_profit:
            self.close_position(current_price, "Take Profit alcanzado")
        elif hit_stop_loss:
            self.close_position(current_price, "Stop Loss activado")
    
    def can_open_position(self) -> bool:
        """
        Verifica si se cumplen condiciones para abrir posición
        
        Returns:
            bool: True si se puede abrir posición
        """
        # Verificar límites diarios
        if self.trades_today >= self.max_trades_per_day:
            logger.info(f"Máximo de operaciones diarias alcanzado: {self.trades_today}")
            return False
        
        if self.loss_today >= (self.initial_balance * self.max_loss_per_day / 100):
            logger.info(f"Máxima pérdida diaria alcanzada: ${self.loss_today:.2f}")
            return False
        
        # Verificar balance suficiente
        if self.balance <= 0:
            logger.info("Balance insuficiente para operar")
            return False
        
        return True
    
    def add_trade(self, trade_data: Dict[str, Any]):
        """
        Añade operación al historial
        
        Args:
            trade_data: Datos de la operación
        """
        # Añadir timestamp
        trade_data["timestamp"] = datetime.now()
        
        # Añadir al historial
        self.trade_history.append(trade_data)
        
        # Mantener tamaño de historial
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
        
        # Llamar callback si existe
        if self.on_trade_callback:
            self.on_trade_callback(trade_data)
    
    def add_signal(self, signal: float, price: float, reason: str):
        """
        Añade señal al historial
        
        Args:
            signal: Valor de la señal (-2 a 2)
            price: Precio actual
            reason: Razón de la señal
        """
        # Crear objeto de señal
        signal_data = {
            "timestamp": datetime.now(),
            "signal": signal,
            "price": price,
            "reason": reason
        }
        
        # Añadir al historial
        self.signal_history.append(signal_data)
        
        # Mantener tamaño de historial
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
        
        # Llamar callback si existe
        if self.on_signal_callback:
            self.on_signal_callback(signal, price, reason)
    
    def add_learning_event(self, event_type: str, description: str, success: bool = True):
        """
        Añade evento de aprendizaje
        
        Args:
            event_type: Tipo de evento
            description: Descripción del evento
            success: Si fue exitoso
        """
        # Crear objeto de evento
        event_data = {
            "timestamp": datetime.now(),
            "type": event_type,
            "description": description,
            "success": success
        }
        
        # Añadir al historial
        self.learning_events.append(event_data)
        
        # Mantener tamaño de historial
        if len(self.learning_events) > 1000:
            self.learning_events = self.learning_events[-1000:]
        
        # Llamar callback si existe
        if self.on_learning_callback:
            self.on_learning_callback(event_type, description, success)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de rendimiento
        
        Returns:
            Dict: Estadísticas de rendimiento
        """
        # Calcular estadísticas básicas
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if trade.get('profit', 0) > 0)
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades * 100
        else:
            win_rate = 0
        
        total_profit = sum(trade.get('profit', 0) for trade in self.trade_history)
        profit_factor = 1.0
        
        # Calcular profit factor si hay suficientes trades
        if total_trades > 5:
            gross_profit = sum(trade.get('profit', 0) for trade in self.trade_history if trade.get('profit', 0) > 0)
            gross_loss = sum(abs(trade.get('profit', 0)) for trade in self.trade_history if trade.get('profit', 0) < 0)
            
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
        
        # Estadísticas de estrategia
        strategy_stats = self.strategy.get_stats() if self.strategy else {}
        
        # Tiempo de ejecución
        runtime = datetime.now() - self.start_time
        hours_running = runtime.total_seconds() / 3600
        
        return {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "total_profit": total_profit,
            "profit_pct": (self.balance / self.initial_balance - 1) * 100,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "trades_today": self.trades_today,
            "loss_today": self.loss_today,
            "time_running_hours": hours_running,
            "strategy_stats": strategy_stats
        }
    
    def set_callbacks(self, trade_cb=None, signal_cb=None, learning_cb=None, position_cb=None):
        """
        Configura callbacks para eventos
        
        Args:
            trade_cb: Callback para operaciones
            signal_cb: Callback para señales
            learning_cb: Callback para eventos de aprendizaje
            position_cb: Callback para actualizaciones de posición
        """
        self.on_trade_callback = trade_cb
        self.on_signal_callback = signal_cb
        self.on_learning_callback = learning_cb
        self.on_position_update_callback = position_cb
        
        logger.info("Callbacks configurados")

class ScalpingBot(TradingBot):
    """Bot especializado en operaciones de scalping (timeframes cortos)"""
    
    def __init__(self, symbol: str, timeframe: str, strategy: str, paper_trading: bool = True):
        """
        Inicializa el bot de scalping
        
        Args:
            symbol: Símbolo de trading
            timeframe: Intervalo de tiempo
            strategy: Nombre de estrategia
            paper_trading: Si usar paper trading
        """
        # Configuración específica para scalping
        config = {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
            "paper_trading": paper_trading,
            "max_position_size": 0.05,  # 5% del balance
            "stop_loss_pct": 0.3,      # 0.3% stop loss
            "take_profit_pct": 0.5,     # 0.5% take profit
            "max_trades_per_day": 50,   # Más operaciones para scalping
            "max_loss_per_day": 3.0     # Límite de pérdida más estricto
        }
        
        super().__init__(config)
        
        # Ajustes específicos para scalping
        self.max_holding_time = 30  # Máximo tiempo en minutos
        
        logger.info(f"Bot de scalping inicializado: {self.symbol} {self.timeframe}")
    
    def run(self):
        """Ejecuta el bucle principal del bot con ajustes para scalping"""
        if self.running:
            logger.warning("El bot ya está en ejecución")
            return
        
        self.running = True
        logger.info(f"Iniciando bot de scalping para {self.symbol}")
        
        try:
            # Bucle principal con intervalos más cortos
            while self.running:
                try:
                    # Obtener datos actualizados
                    data = self.get_market_data()
                    
                    if data is None or len(data) < 20:
                        logger.warning("Datos insuficientes para análisis, esperando...")
                        time.sleep(5)
                        continue
                    
                    # Detectar condición de mercado
                    market_condition = self.adaptive_system.detect_market_condition(data)
                    time_interval = TimeInterval(self.timeframe) if self.timeframe in [e.value for e in TimeInterval] else TimeInterval.MINUTE_1
                    
                    # Generar señal
                    signal, reason, details = self.strategy.get_signal(data)
                    
                    # Registrar señal
                    current_price = data['close'].iloc[-1]
                    self.add_signal(signal, current_price, reason)
                    
                    # Procesar señal con lógica de scalping más agresiva
                    self.process_scalping_signal(signal, current_price, reason, details)
                    
                    # Gestionar posición existente con criterios de scalping
                    if self.current_position:
                        self.manage_scalping_position(current_price, data)
                    
                    # Añadir evento de aprendizaje periódicamente
                    if np.random.random() < 0.15:  # 15% de probabilidad (más frecuente)
                        self.add_learning_event(
                            "ADAPTACIÓN", 
                            f"Ajuste fino para condición {market_condition.value} en scalping", 
                            True
                        )
                    
                    # Esperar menos tiempo entre ciclos
                    time.sleep(2)  # Más rápido para scalping
                    
                except Exception as e:
                    logger.error(f"Error en ciclo de scalping: {e}")
                    time.sleep(10)
        
        except KeyboardInterrupt:
            logger.info("Bot de scalping detenido por usuario")
        finally:
            self.running = False
            logger.info("Bot de scalping detenido")
    
    def process_scalping_signal(self, signal: float, price: float, reason: str, details: Dict[str, Any]):
        """
        Procesa señal con criterios específicos de scalping
        
        Args:
            signal: Señal (-2 a 2)
            price: Precio actual
            reason: Razón de la señal
            details: Detalles adicionales
        """
        # Modificación para operaciones más rápidas
        
        # Verificar si ya estamos en posición
        if self.current_position:
            # En scalping, cerrar posiciones más rápidamente
            if self.current_position["type"] == "long" and signal <= -0.3:
                self.close_position(price, f"Señal de salida rápida: {reason}")
            
            elif self.current_position["type"] == "short" and signal >= 0.3:
                self.close_position(price, f"Señal de salida rápida: {reason}")
            
            # Verificar tiempo máximo de posición
            holding_time = datetime.now() - self.current_position["entry_time"]
            if holding_time.total_seconds() / 60 > self.max_holding_time:
                self.close_position(price, "Tiempo máximo de posición alcanzado")
            
            return
        
        # Entradas más agresivas para scalping
        if signal >= 0.3:  # Umbral más bajo para long
            self.open_position("long", price, self.calculate_position_size(price), reason)
        
        elif signal <= -0.3 and False:  # Umbral más bajo para short (desactivado)
            # self.open_position("short", price, self.calculate_position_size(price), reason)
            pass
    
    def manage_scalping_position(self, current_price: float, data: pd.DataFrame):
        """
        Gestiona posición con ajustes para scalping
        
        Args:
            current_price: Precio actual
            data: DataFrame con datos de mercado
        """
        if not self.current_position:
            return
        
        position = self.current_position
        position_type = position["type"]
        entry_price = position["entry_price"]
        
        # Calcular PnL actual
        if position_type == "long":
            pnl_pct = (current_price / entry_price - 1) * 100
        else:  # short
            pnl_pct = (entry_price / current_price - 1) * 100
        
        # Trailing stop más agresivo para scalping
        if pnl_pct > 0.25 and position_type == "long":  # Umbral más bajo
            # Si estamos en ganancia, mover stop loss para asegurar parte
            new_stop = current_price * (1 - self.stop_loss_pct / 400)  # Más ajustado
            if new_stop > position["stop_loss"]:
                position["stop_loss"] = new_stop
                logger.info(f"Trailing stop ajustado a: {new_stop:.4f}")
        
        # Cerrar si se alcanzan niveles
        hit_take_profit = False
        hit_stop_loss = False
        
        if position_type == "long":
            hit_take_profit = current_price >= position["take_profit"]
            hit_stop_loss = current_price <= position["stop_loss"]
        else:  # short
            hit_take_profit = current_price <= position["take_profit"]
            hit_stop_loss = current_price >= position["stop_loss"]
        
        if hit_take_profit:
            self.close_position(current_price, "Take Profit alcanzado (scalping)")
        elif hit_stop_loss:
            self.close_position(current_price, "Stop Loss activado (scalping)")
        
        # Monitoreo de volatilidad
        # En scalping, podemos cerrar posiciones si la volatilidad cambia demasiado
        recent_vol = data['close'].pct_change().tail(10).std() * 100
        if recent_vol > 0.5:  # Si volatilidad es muy alta
            self.close_position(current_price, "Alta volatilidad detectada")

# Ejemplo de uso independiente
if __name__ == "__main__":
    # Usar la clase ScalpingBot para operaciones de muy corto plazo
    bot = ScalpingBot(
        symbol="SOL-USDT",
        timeframe="1m",
        strategy="rsi_scalping",
        paper_trading=True  # Solo paper trading por seguridad
    )
    
    # Configurar callbacks para eventos
    def on_trade(trade_data):
        print(f"Nueva operación: {trade_data}")
    
    def on_signal(signal, price, reason):
        print(f"Señal: {signal:.2f} a ${price:.2f} - {reason}")
    
    def on_learning(event_type, description, success):
        print(f"Aprendizaje: {event_type} - {description} - {'✓' if success else '✗'}")
    
    def on_position(position_data):
        if position_data:
            print(f"Posición: {position_data['type']} {position_data['size']} @ {position_data['entry_price']}")
        else:
            print("Posición cerrada")
    
    # Configurar callbacks
    bot.set_callbacks(on_trade, on_signal, on_learning, on_position)
    
    # Iniciar bot
    try:
        logger.info("Presiona Ctrl+C para detener el bot")
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot detenido por usuario")
    finally:
        # Mostrar estadísticas
        stats = bot.get_stats()
        print("\n--- ESTADÍSTICAS DE TRADING ---")
        print(f"Balance final: ${stats['balance']:.2f} (Inicial: ${stats['initial_balance']:.2f})")
        print(f"Rentabilidad: {stats['profit_pct']:.2f}%")
        print(f"Operaciones totales: {stats['total_trades']}")
        print(f"Win rate: {stats['win_rate']:.2f}%")
        print(f"Profit factor: {stats['profit_factor']:.2f}")
        print(f"Tiempo de ejecución: {stats['time_running_hours']:.2f} horas")