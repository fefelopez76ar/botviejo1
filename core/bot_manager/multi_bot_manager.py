#!/usr/bin/env python3
"""
Gestor de múltiples bots de trading en simulación.

Este módulo permite ejecutar y monitorear múltiples instancias de bots
con diferentes estrategias, timeframes y pares de trading, cada uno
con su propio capital inicial simulado.
"""

import os
import sys
import json
import logging
import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MultiBotManager')

class BotInstance:
    """
    Representa una instancia individual de un bot de trading.
    """
    
    def __init__(self, 
               bot_id: str,
               strategy_name: str,
               symbol: str = "SOL-USDT",
               timeframe: str = "1h",
               initial_balance: float = 100.0,
               leverage: int = 1,
               market_type: str = "spot",
               params: Dict[str, Any] = None):
        """
        Inicializa una instancia de bot.
        
        Args:
            bot_id: Identificador único del bot
            strategy_name: Nombre de la estrategia
            symbol: Par de trading (ej. "SOL-USDT")
            timeframe: Marco temporal (ej. "1h")
            initial_balance: Balance inicial en USDT
            leverage: Apalancamiento (solo para futuros)
            market_type: Tipo de mercado ("spot" o "futures")
            params: Parámetros específicos de la estrategia
        """
        self.bot_id = bot_id
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.leverage = leverage
        self.market_type = market_type
        self.params = params or {}
        
        # Estado del bot
        self.running = False
        self.start_time = None
        self.last_update_time = None
        self.error = None
        
        # Posición actual
        self.position = None
        
        # Historial de operaciones
        self.trades = []
        
        # Métricas de rendimiento
        self.metrics = {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_profit_per_trade": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "profitable_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "total_fees": 0.0,
            "roi": 0.0
        }
        
        # Thread para ejecución del bot
        self.thread = None
        self.stop_event = threading.Event()
    
    def start(self):
        """Inicia la ejecución del bot en un hilo separado."""
        if self.running:
            logger.warning(f"Bot {self.bot_id} ya está en ejecución")
            return False
        
        self.running = True
        self.start_time = datetime.now()
        self.last_update_time = self.start_time
        self.error = None
        
        # Iniciar hilo
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"Bot {self.bot_id} iniciado con estrategia {self.strategy_name} en {self.symbol} ({self.timeframe})")
        return True
    
    def stop(self):
        """Detiene la ejecución del bot."""
        if not self.running:
            logger.warning(f"Bot {self.bot_id} no está en ejecución")
            return False
        
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)
        
        self.running = False
        logger.info(f"Bot {self.bot_id} detenido")
        return True
    
    def _run(self):
        """Función principal del hilo del bot."""
        try:
            while not self.stop_event.is_set():
                # Obtener datos de mercado actualizados
                market_data = self._get_market_data()
                
                # Ejecutar estrategia
                signal = self._execute_strategy(market_data)
                
                # Procesar señal de trading
                if signal:
                    self._process_signal(signal, market_data)
                
                # Actualizar posiciones existentes
                self._update_positions(market_data)
                
                # Calcular métricas
                self._calculate_metrics()
                
                # Actualizar timestamp
                self.last_update_time = datetime.now()
                
                # Simular velocidad real
                if self.timeframe == "1m":
                    time.sleep(1.0)  # Más rápido para 1m
                elif self.timeframe == "5m":
                    time.sleep(2.0)  # Más rápido para 5m
                elif self.timeframe == "15m":
                    time.sleep(3.0)  # Más rápido para 15m
                else:
                    time.sleep(5.0)  # Timeframes más largos
                
        except Exception as e:
            self.error = str(e)
            self.running = False
            logger.error(f"Error en bot {self.bot_id}: {e}")
    
    def _get_market_data(self) -> Dict[str, Any]:
        """
        Obtiene datos de mercado para el símbolo y timeframe.
        
        Returns:
            Dict[str, Any]: Datos de mercado
        """
        try:
            # Importar aquí para evitar dependencias circulares
            from data_management.market_data import MarketData
            
            # Crear instancia de MarketData
            md = MarketData()
            
            # Obtener datos históricos
            df = md.get_historical_data(self.symbol, self.timeframe, limit=100)
            
            # Calcular indicadores
            if df is not None and not df.empty:
                df = md.calculate_indicators(df)
                
                # Obtener precio actual
                current_price = df['close'].iloc[-1]
                
                # Obtener datos del libro de órdenes
                orderbook = md.get_orderbook(self.symbol)
                
                return {
                    "dataframe": df,
                    "current_price": current_price,
                    "orderbook": orderbook,
                    "timestamp": datetime.now()
                }
            
            # Si no hay datos, usar último precio conocido
            current_price = md.get_current_price(self.symbol)
            
            return {
                "dataframe": None,
                "current_price": current_price,
                "orderbook": {},
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error al obtener datos de mercado para {self.bot_id}: {e}")
            
            # Valores por defecto si hay error
            return {
                "dataframe": None,
                "current_price": self._get_last_known_price(),
                "orderbook": {},
                "timestamp": datetime.now(),
                "error": str(e)
            }
    
    def _get_last_known_price(self) -> float:
        """
        Obtiene el último precio conocido del símbolo.
        
        Returns:
            float: Último precio conocido o precio estimado
        """
        # Si hay trades, usar el precio del último
        if self.trades:
            return self.trades[0].get("price", 0)
        
        # Valores por defecto para diferentes símbolos
        default_prices = {
            "SOL-USDT": 150.25,
            "BTC-USDT": 65000.0,
            "ETH-USDT": 3500.0,
            "XRP-USDT": 0.55,
            "ADA-USDT": 0.45
        }
        
        # Intentar encontrar una coincidencia parcial
        base_asset = self.symbol.split('-')[0]
        for symbol, price in default_prices.items():
            if base_asset in symbol:
                return price
        
        # Valor por defecto
        return 100.0
    
    def _execute_strategy(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Ejecuta la estrategia configurada.
        
        Args:
            market_data: Datos de mercado
            
        Returns:
            Optional[Dict[str, Any]]: Señal de trading o None
        """
        if market_data.get("error") or market_data.get("dataframe") is None:
            return None
        
        df = market_data["dataframe"]
        
        # Obtener la estrategia adecuada
        if self.strategy_name == "breakout_scalping":
            from scalping_strategies import ScalpingStrategies
            
            # Crear instancia de estrategia con parámetros
            scalper = ScalpingStrategies(**self.params)
            
            # Ejecutar estrategia
            return scalper.breakout_scalping_strategy(df, market_data.get("orderbook"))
            
        elif self.strategy_name == "momentum_scalping":
            from scalping_strategies import ScalpingStrategies
            
            # Crear instancia de estrategia con parámetros
            scalper = ScalpingStrategies(**self.params)
            
            # Ejecutar estrategia
            return scalper.momentum_scalping_strategy(df)
            
        elif self.strategy_name == "mean_reversion":
            from scalping_strategies import ScalpingStrategies
            
            # Crear instancia de estrategia con parámetros
            scalper = ScalpingStrategies(**self.params)
            
            # Ejecutar estrategia
            return scalper.mean_reversion_scalping(df)
            
        elif self.strategy_name == "ml_adaptive":
            # Importar aquí para evitar dependencias circulares
            from indicator_weighting import IndicatorWeighting
            
            # Crear instancia de ponderación adaptativa
            weighting = IndicatorWeighting()
            
            # Obtener predicción
            prediction = weighting.get_prediction_summary(df)
            
            # Convertir a formato de señal
            if prediction.get("recommendation") in ["strong_buy", "buy"]:
                return {
                    "signal": "buy",
                    "strategy": "ml_adaptive",
                    "confidence": prediction.get("confidence", 0.5),
                    "entry_price": market_data["current_price"],
                    "take_profit": market_data["current_price"] * (1 + self.params.get("take_profit_pct", 0.8) / 100),
                    "stop_loss": market_data["current_price"] * (1 - self.params.get("stop_loss_pct", 0.6) / 100),
                    "timestamp": datetime.now().isoformat()
                }
            elif prediction.get("recommendation") in ["strong_sell", "sell"]:
                return {
                    "signal": "sell",
                    "strategy": "ml_adaptive",
                    "confidence": prediction.get("confidence", 0.5),
                    "entry_price": market_data["current_price"],
                    "take_profit": market_data["current_price"] * (1 - self.params.get("take_profit_pct", 0.8) / 100),
                    "stop_loss": market_data["current_price"] * (1 + self.params.get("stop_loss_pct", 0.6) / 100),
                    "timestamp": datetime.now().isoformat()
                }
            
            return None
        
        # Estrategia por defecto: no hacer nada
        return None
    
    def _process_signal(self, signal: Dict[str, Any], market_data: Dict[str, Any]):
        """
        Procesa una señal de trading.
        
        Args:
            signal: Señal de trading
            market_data: Datos de mercado
        """
        if signal.get("signal") not in ["buy", "sell"]:
            return
        
        # Verificar si ya tenemos una posición
        if self.position:
            # Si la señal es en la dirección opuesta, considerar cerrar posición actual
            if (self.position["side"] == "long" and signal["signal"] == "sell") or \
               (self.position["side"] == "short" and signal["signal"] == "buy"):
                # Cerrar posición actual
                self._close_position(market_data["current_price"], "signal_reversal")
        
        # Abrir nueva posición si no tenemos una o si la anterior se cerró
        if not self.position:
            # Calcular tamaño de posición (% del balance)
            position_size_pct = signal.get("position_size_pct", self.params.get("position_size_pct", 50.0))
            position_value = self.current_balance * (position_size_pct / 100.0)
            
            # En futuros, aplicar apalancamiento
            if self.market_type == "futures":
                effective_position_value = position_value * self.leverage
            else:
                effective_position_value = position_value
            
            # Calcular cantidad en base al precio actual
            current_price = market_data["current_price"]
            quantity = effective_position_value / current_price
            
            # Abrir posición
            if signal["signal"] == "buy":
                self._open_long_position(current_price, quantity, signal)
            else:  # sell
                if self.market_type == "futures":  # solo permitir shorts en futuros
                    self._open_short_position(current_price, quantity, signal)
    
    def _open_long_position(self, price: float, quantity: float, signal: Dict[str, Any]):
        """
        Abre una posición larga.
        
        Args:
            price: Precio de entrada
            quantity: Cantidad a comprar
            signal: Señal de trading que generó la orden
        """
        # Calcular comisión
        commission_rate = 0.001  # 0.1% (fee típico de exchange)
        commission = price * quantity * commission_rate
        
        # Actualizar balance (restar comisión)
        position_cost = price * quantity
        
        # En spot, restar el costo de la posición del balance
        if self.market_type == "spot":
            # Verificar si hay suficiente balance
            if position_cost + commission > self.current_balance:
                # Ajustar cantidad según balance disponible
                max_quantity = (self.current_balance - commission) / price
                quantity = max_quantity * 0.99  # Margen de seguridad
                position_cost = price * quantity
                commission = price * quantity * commission_rate
            
            self.current_balance -= (position_cost + commission)
        else:
            # En futuros, solo restar la comisión y el margen requerido
            margin_required = position_cost / self.leverage
            if margin_required + commission > self.current_balance:
                # Ajustar cantidad según margen disponible
                max_quantity = (self.current_balance - commission) * self.leverage / price
                quantity = max_quantity * 0.99  # Margen de seguridad
                position_cost = price * quantity
                margin_required = position_cost / self.leverage
                commission = price * quantity * commission_rate
            
            self.current_balance -= (margin_required + commission)
        
        # Crear objeto de posición
        self.position = {
            "side": "long",
            "entry_price": price,
            "quantity": quantity,
            "entry_time": datetime.now(),
            "take_profit": signal.get("take_profit"),
            "stop_loss": signal.get("stop_loss"),
            "strategy": signal.get("strategy"),
            "timeframe": self.timeframe,
            "market_type": self.market_type,
            "leverage": self.leverage,
            "entry_commission": commission,
            "position_value": position_cost
        }
        
        # Registrar operación en el historial
        self.trades.append({
            "id": f"T{len(self.trades) + 1}",
            "symbol": self.symbol,
            "side": "BUY",
            "price": price,
            "quantity": quantity,
            "commission": commission,
            "timestamp": datetime.now().isoformat(),
            "order_type": "MARKET",
            "trade_type": "ENTRY",
            "position_id": id(self.position),
            "strategy": signal.get("strategy"),
            "timeframe": self.timeframe
        })
        
        logger.info(f"Bot {self.bot_id}: Posición LONG abierta a ${price:.2f} x {quantity:.4f} = ${position_cost:.2f}")
    
    def _open_short_position(self, price: float, quantity: float, signal: Dict[str, Any]):
        """
        Abre una posición corta (solo en futuros).
        
        Args:
            price: Precio de entrada
            quantity: Cantidad a vender
            signal: Señal de trading que generó la orden
        """
        if self.market_type != "futures":
            logger.warning(f"Bot {self.bot_id}: Intento de abrir posición SHORT en modo {self.market_type}")
            return
        
        # Calcular comisión
        commission_rate = 0.001  # 0.1% (fee típico de exchange)
        commission = price * quantity * commission_rate
        
        # Calcular margen requerido
        position_value = price * quantity
        margin_required = position_value / self.leverage
        
        # Verificar si hay suficiente balance
        if margin_required + commission > self.current_balance:
            # Ajustar cantidad según margen disponible
            max_quantity = (self.current_balance - commission) * self.leverage / price
            quantity = max_quantity * 0.99  # Margen de seguridad
            position_value = price * quantity
            margin_required = position_value / self.leverage
            commission = price * quantity * commission_rate
        
        # Actualizar balance (restar comisión y margen)
        self.current_balance -= (margin_required + commission)
        
        # Crear objeto de posición
        self.position = {
            "side": "short",
            "entry_price": price,
            "quantity": quantity,
            "entry_time": datetime.now(),
            "take_profit": signal.get("take_profit"),
            "stop_loss": signal.get("stop_loss"),
            "strategy": signal.get("strategy"),
            "timeframe": self.timeframe,
            "market_type": self.market_type,
            "leverage": self.leverage,
            "entry_commission": commission,
            "position_value": position_value
        }
        
        # Registrar operación en el historial
        self.trades.append({
            "id": f"T{len(self.trades) + 1}",
            "symbol": self.symbol,
            "side": "SELL",
            "price": price,
            "quantity": quantity,
            "commission": commission,
            "timestamp": datetime.now().isoformat(),
            "order_type": "MARKET",
            "trade_type": "ENTRY",
            "position_id": id(self.position),
            "strategy": signal.get("strategy"),
            "timeframe": self.timeframe
        })
        
        logger.info(f"Bot {self.bot_id}: Posición SHORT abierta a ${price:.2f} x {quantity:.4f} = ${position_value:.2f}")
    
    def _update_positions(self, market_data: Dict[str, Any]):
        """
        Actualiza posiciones existentes para comprobar TP/SL.
        
        Args:
            market_data: Datos de mercado actuales
        """
        if not self.position:
            return
        
        current_price = market_data["current_price"]
        
        # Evaluar posición larga
        if self.position["side"] == "long":
            # Comprobar Take Profit
            if self.position["take_profit"] and current_price >= self.position["take_profit"]:
                self._close_position(current_price, "take_profit")
                return
            
            # Comprobar Stop Loss
            if self.position["stop_loss"] and current_price <= self.position["stop_loss"]:
                self._close_position(current_price, "stop_loss")
                return
        
        # Evaluar posición corta
        elif self.position["side"] == "short":
            # Comprobar Take Profit
            if self.position["take_profit"] and current_price <= self.position["take_profit"]:
                self._close_position(current_price, "take_profit")
                return
            
            # Comprobar Stop Loss
            if self.position["stop_loss"] and current_price >= self.position["stop_loss"]:
                self._close_position(current_price, "stop_loss")
                return
    
    def _close_position(self, price: float, reason: str):
        """
        Cierra la posición actual.
        
        Args:
            price: Precio de cierre
            reason: Razón del cierre ("take_profit", "stop_loss", "signal_reversal", etc.)
        """
        if not self.position:
            return
        
        # Calcular comisión de salida
        commission_rate = 0.001  # 0.1% (fee típico de exchange)
        commission = price * self.position["quantity"] * commission_rate
        
        # Calcular PnL
        if self.position["side"] == "long":
            # Para posiciones largas: (precio_salida - precio_entrada) * cantidad
            pnl = (price - self.position["entry_price"]) * self.position["quantity"]
            if self.market_type == "futures":
                # En futuros, el PnL se multiplica por el apalancamiento
                pnl *= self.leverage
        else:  # short
            # Para posiciones cortas: (precio_entrada - precio_salida) * cantidad
            pnl = (self.position["entry_price"] - price) * self.position["quantity"]
            if self.market_type == "futures":
                # En futuros, el PnL se multiplica por el apalancamiento
                pnl *= self.leverage
        
        # Calcular duración de la posición
        duration = datetime.now() - self.position["entry_time"]
        duration_hours = duration.total_seconds() / 3600
        
        # Calcular estadísticas
        entry_value = self.position["entry_price"] * self.position["quantity"]
        exit_value = price * self.position["quantity"]
        
        if self.market_type == "spot":
            # En spot, agregar el valor de la posición al balance
            self.current_balance += (exit_value - commission)
        else:
            # En futuros, agregar el margen y el PnL al balance
            margin = self.position["position_value"] / self.leverage
            self.current_balance += (margin + pnl - commission)
        
        # Registrar operación en el historial
        self.trades.append({
            "id": f"T{len(self.trades) + 1}",
            "symbol": self.symbol,
            "side": "SELL" if self.position["side"] == "long" else "BUY",
            "price": price,
            "quantity": self.position["quantity"],
            "commission": commission,
            "timestamp": datetime.now().isoformat(),
            "order_type": "MARKET",
            "trade_type": "EXIT",
            "position_id": id(self.position),
            "strategy": self.position["strategy"],
            "timeframe": self.timeframe,
            "pnl": pnl,
            "pnl_percent": pnl / entry_value * 100 if entry_value > 0 else 0,
            "duration_hours": duration_hours,
            "exit_reason": reason
        })
        
        logger.info(f"Bot {self.bot_id}: Posición {self.position['side'].upper()} cerrada a ${price:.2f}, PnL: ${pnl:.2f}")
        
        # Limpiar posición actual
        self.position = None
    
    def _calculate_metrics(self):
        """Calcula métricas de rendimiento actualizadas."""
        # Si no hay trades, no hay métricas que calcular
        if not self.trades:
            return
        
        # Filtrar sólo operaciones de salida (que tienen PnL)
        exit_trades = [t for t in self.trades if t.get("trade_type") == "EXIT"]
        
        if not exit_trades:
            return
        
        # Calcular métricas básicas
        total_trades = len(exit_trades)
        profitable_trades = sum(1 for t in exit_trades if t.get("pnl", 0) > 0)
        losing_trades = total_trades - profitable_trades
        
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calcular ganancias y pérdidas totales
        total_profit = sum(t.get("pnl", 0) for t in exit_trades if t.get("pnl", 0) > 0)
        total_loss = sum(abs(t.get("pnl", 0)) for t in exit_trades if t.get("pnl", 0) < 0)
        
        # Calcular profit factor
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calcular ganancia/pérdida promedio por operación
        avg_profit_per_trade = sum(t.get("pnl", 0) for t in exit_trades) / total_trades if total_trades > 0 else 0
        
        # Calcular comisiones totales
        total_fees = sum(t.get("commission", 0) for t in self.trades)
        
        # Calcular ROI (Return On Investment)
        roi = (self.current_balance / self.initial_balance - 1) * 100
        
        # Calcular drawdown
        balances = [self.initial_balance]
        current_balance = self.initial_balance
        
        for trade in exit_trades:
            # Sumar el PnL y restar la comisión
            pnl = trade.get("pnl", 0)
            commission = trade.get("commission", 0)
            current_balance += pnl - commission
            balances.append(current_balance)
        
        # Calcular drawdown como la mayor caída desde un pico
        peak = self.initial_balance
        max_drawdown = 0
        
        for balance in balances:
            peak = max(peak, balance)
            drawdown = (peak - balance) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Actualizar diccionario de métricas
        self.metrics = {
            "win_rate": win_rate * 100,  # En porcentaje
            "profit_factor": profit_factor,
            "avg_profit_per_trade": avg_profit_per_trade,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "losing_trades": losing_trades,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "total_fees": total_fees,
            "roi": roi
        }
    
    def get_state(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del bot.
        
        Returns:
            Dict[str, Any]: Estado completo del bot
        """
        # Calcular métricas antes de retornar
        self._calculate_metrics()
        
        # Ordenar trades más recientes primero
        sorted_trades = sorted(self.trades, key=lambda t: t.get("timestamp", ""), reverse=True)
        
        # Calcular métricas diarias y mensuales
        daily_metrics = self._calculate_period_metrics(1)
        weekly_metrics = self._calculate_period_metrics(7)
        monthly_metrics = self._calculate_period_metrics(30)
        
        # Calcular comisión como porcentaje de la ganancia
        fee_to_profit_ratio = 0
        if self.metrics["total_profit"] > 0:
            fee_to_profit_ratio = self.metrics["total_fees"] / self.metrics["total_profit"] * 100
        
        return {
            "bot_id": self.bot_id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "market_type": self.market_type,
            "leverage": self.leverage,
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "current_position": self.position,
            "running": self.running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime": str(datetime.now() - self.start_time) if self.start_time else "00:00:00",
            "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
            "error": self.error,
            "metrics": self.metrics,
            "daily_metrics": daily_metrics,
            "weekly_metrics": weekly_metrics,
            "monthly_metrics": monthly_metrics,
            "recent_trades": sorted_trades[:10],  # Solo las 10 operaciones más recientes
            "fee_impact": {
                "fee_to_profit_ratio": fee_to_profit_ratio,
                "avg_fee_per_trade": self.metrics["total_fees"] / self.metrics["total_trades"] if self.metrics["total_trades"] > 0 else 0
            }
        }
    
    def _calculate_period_metrics(self, days: int) -> Dict[str, Any]:
        """
        Calcula métricas para un período específico (días).
        
        Args:
            days: Número de días a considerar
            
        Returns:
            Dict[str, Any]: Métricas del período
        """
        # Calcular la fecha de inicio del período
        start_date = datetime.now() - timedelta(days=days)
        
        # Filtrar trades del período
        period_exit_trades = [
            t for t in self.trades 
            if t.get("trade_type") == "EXIT" and 
               datetime.fromisoformat(t.get("timestamp", datetime.now().isoformat())) >= start_date
        ]
        
        # Si no hay trades en el período, retornar valores por defecto
        if not period_exit_trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit": 0,
                "profit_percent": 0,
                "fees": 0,
                "fees_percent": 0
            }
        
        # Calcular métricas del período
        total_trades = len(period_exit_trades)
        profitable_trades = sum(1 for t in period_exit_trades if t.get("pnl", 0) > 0)
        
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Sumar PnL y comisiones
        total_pnl = sum(t.get("pnl", 0) for t in period_exit_trades)
        
        # Incluir también comisiones de entrada
        entry_trade_ids = [t.get("position_id") for t in period_exit_trades]
        period_entry_trades = [
            t for t in self.trades 
            if t.get("trade_type") == "ENTRY" and t.get("position_id") in entry_trade_ids
        ]
        
        # Calcular comisiones totales
        entry_fees = sum(t.get("commission", 0) for t in period_entry_trades)
        exit_fees = sum(t.get("commission", 0) for t in period_exit_trades)
        total_fees = entry_fees + exit_fees
        
        # Calcular ganancia neta
        net_profit = total_pnl - total_fees
        
        # Calcular ROI del período
        # Asumimos que el balance inicial para el período es el balance actual - ganancia neta
        period_initial_balance = self.current_balance - net_profit
        period_roi = (net_profit / period_initial_balance) * 100 if period_initial_balance > 0 else 0
        
        # Calcular impacto de comisiones
        fee_impact = total_fees / abs(total_pnl) * 100 if total_pnl != 0 else 0
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate * 100,  # En porcentaje
            "profit": net_profit,
            "profit_percent": period_roi,
            "fees": total_fees,
            "fees_percent": fee_impact
        }

class MultiBotManager:
    """
    Gestor de múltiples instancias de bots de trading.
    """
    
    def __init__(self, config_file: str = None):
        """
        Inicializa el gestor de bots.
        
        Args:
            config_file: Archivo de configuración (opcional)
        """
        self.bots: Dict[str, BotInstance] = {}
        self.config_file = config_file
        
        # Cargar configuración si existe
        if config_file and os.path.exists(config_file):
            self._load_config()
        else:
            self._create_default_config()
    
    def _load_config(self):
        """Carga configuración desde archivo."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Crear bots a partir de la configuración
            for bot_config in config.get("bots", []):
                bot_id = bot_config.get("bot_id")
                if not bot_id:
                    continue
                
                self.create_bot(
                    bot_id=bot_id,
                    strategy_name=bot_config.get("strategy_name"),
                    symbol=bot_config.get("symbol", "SOL-USDT"),
                    timeframe=bot_config.get("timeframe", "1h"),
                    initial_balance=bot_config.get("initial_balance", 100.0),
                    leverage=bot_config.get("leverage", 1),
                    market_type=bot_config.get("market_type", "spot"),
                    params=bot_config.get("params", {})
                )
            
            logger.info(f"Configuración cargada desde {self.config_file}")
        except Exception as e:
            logger.error(f"Error al cargar configuración: {e}")
            # Crear configuración por defecto
            self._create_default_config()
    
    def _create_default_config(self):
        """Crea configuración por defecto."""
        # No crear bots automáticamente
        logger.info("Usando configuración por defecto")
    
    def _save_config(self):
        """Guarda configuración actual en archivo."""
        if not self.config_file:
            logger.warning("No se ha especificado archivo de configuración")
            return
        
        try:
            # Crear configuración con todos los bots
            config = {
                "bots": [
                    {
                        "bot_id": bot.bot_id,
                        "strategy_name": bot.strategy_name,
                        "symbol": bot.symbol,
                        "timeframe": bot.timeframe,
                        "initial_balance": bot.initial_balance,
                        "leverage": bot.leverage,
                        "market_type": bot.market_type,
                        "params": bot.params
                    }
                    for bot in self.bots.values()
                ]
            }
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # Guardar en archivo
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            logger.info(f"Configuración guardada en {self.config_file}")
        except Exception as e:
            logger.error(f"Error al guardar configuración: {e}")
    
    def create_bot(self, 
                 bot_id: str,
                 strategy_name: str,
                 symbol: str = "SOL-USDT",
                 timeframe: str = "1h",
                 initial_balance: float = 100.0,
                 leverage: int = 1,
                 market_type: str = "spot",
                 params: Dict[str, Any] = None) -> str:
        """
        Crea una nueva instancia de bot.
        
        Args:
            bot_id: Identificador único del bot
            strategy_name: Nombre de la estrategia
            symbol: Par de trading
            timeframe: Marco temporal
            initial_balance: Balance inicial
            leverage: Apalancamiento
            market_type: Tipo de mercado
            params: Parámetros específicos
            
        Returns:
            str: ID del bot creado
        """
        # Si no se proporciona ID, generar uno único
        if not bot_id:
            bot_id = f"Bot_{len(self.bots) + 1}"
        
        # Verificar si ya existe un bot con ese ID
        if bot_id in self.bots:
            logger.warning(f"Ya existe un bot con ID {bot_id}")
            return None
        
        # Crear nueva instancia
        bot = BotInstance(
            bot_id=bot_id,
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            initial_balance=initial_balance,
            leverage=leverage,
            market_type=market_type,
            params=params
        )
        
        # Agregar al diccionario de bots
        self.bots[bot_id] = bot
        
        # Guardar configuración actualizada
        self._save_config()
        
        logger.info(f"Bot creado: {bot_id} con estrategia {strategy_name} en {symbol} ({timeframe})")
        return bot_id
    
    def start_bot(self, bot_id: str) -> bool:
        """
        Inicia un bot específico.
        
        Args:
            bot_id: ID del bot a iniciar
            
        Returns:
            bool: True si se inició correctamente, False en caso contrario
        """
        if bot_id not in self.bots:
            logger.warning(f"Bot no encontrado: {bot_id}")
            return False
        
        return self.bots[bot_id].start()
    
    def stop_bot(self, bot_id: str) -> bool:
        """
        Detiene un bot específico.
        
        Args:
            bot_id: ID del bot a detener
            
        Returns:
            bool: True si se detuvo correctamente, False en caso contrario
        """
        if bot_id not in self.bots:
            logger.warning(f"Bot no encontrado: {bot_id}")
            return False
        
        return self.bots[bot_id].stop()
    
    def delete_bot(self, bot_id: str) -> bool:
        """
        Elimina un bot específico.
        
        Args:
            bot_id: ID del bot a eliminar
            
        Returns:
            bool: True si se eliminó correctamente, False en caso contrario
        """
        if bot_id not in self.bots:
            logger.warning(f"Bot no encontrado: {bot_id}")
            return False
        
        # Detener bot si está en ejecución
        if self.bots[bot_id].running:
            self.bots[bot_id].stop()
        
        # Eliminar del diccionario
        del self.bots[bot_id]
        
        # Guardar configuración actualizada
        self._save_config()
        
        logger.info(f"Bot eliminado: {bot_id}")
        return True
    
    def get_bot_status(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene el estado de un bot específico.
        
        Args:
            bot_id: ID del bot
            
        Returns:
            Optional[Dict[str, Any]]: Estado del bot o None si no existe
        """
        if bot_id not in self.bots:
            logger.warning(f"Bot no encontrado: {bot_id}")
            return None
        
        return self.bots[bot_id].get_state()
    
    def get_all_bots_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene el estado de todos los bots.
        
        Returns:
            Dict[str, Dict[str, Any]]: Estado de todos los bots
        """
        return {bot_id: bot.get_state() for bot_id, bot in self.bots.items()}
    
    def start_all_bots(self) -> Dict[str, bool]:
        """
        Inicia todos los bots.
        
        Returns:
            Dict[str, bool]: Resultado de inicio por bot
        """
        results = {}
        
        for bot_id, bot in self.bots.items():
            results[bot_id] = bot.start()
        
        return results
    
    def stop_all_bots(self) -> Dict[str, bool]:
        """
        Detiene todos los bots.
        
        Returns:
            Dict[str, bool]: Resultado de detención por bot
        """
        results = {}
        
        for bot_id, bot in self.bots.items():
            results[bot_id] = bot.stop()
        
        return results
    
    def get_combined_performance(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento combinadas de todos los bots.
        
        Returns:
            Dict[str, Any]: Métricas combinadas
        """
        # Si no hay bots, retornar métricas vacías
        if not self.bots:
            return {
                "total_bots": 0,
                "running_bots": 0,
                "total_balance": 0,
                "total_profit": 0,
                "total_roi": 0,
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "total_fees": 0,
                "fee_impact": 0,
                "best_bot": None,
                "worst_bot": None,
                "daily_performance": {
                    "profit": 0,
                    "profit_percent": 0,
                    "trades": 0,
                    "win_rate": 0,
                    "fees": 0,
                    "fees_percent": 0
                },
                "monthly_performance": {
                    "profit": 0,
                    "profit_percent": 0,
                    "trades": 0,
                    "win_rate": 0,
                    "fees": 0,
                    "fees_percent": 0
                }
            }
        
        # Obtener estado de todos los bots
        all_status = self.get_all_bots_status()
        
        # Calcular métricas combinadas
        total_bots = len(all_status)
        running_bots = sum(1 for status in all_status.values() if status.get("running", False))
        
        total_initial_balance = sum(status.get("initial_balance", 0) for status in all_status.values())
        total_current_balance = sum(status.get("current_balance", 0) for status in all_status.values())
        
        total_profit = total_current_balance - total_initial_balance
        total_roi = (total_current_balance / total_initial_balance - 1) * 100 if total_initial_balance > 0 else 0
        
        # Sumar métricas de todos los bots
        total_trades = sum(status.get("metrics", {}).get("total_trades", 0) for status in all_status.values())
        profitable_trades = sum(status.get("metrics", {}).get("profitable_trades", 0) for status in all_status.values())
        total_profit_sum = sum(status.get("metrics", {}).get("total_profit", 0) for status in all_status.values())
        total_loss_sum = sum(status.get("metrics", {}).get("total_loss", 0) for status in all_status.values())
        total_fees = sum(status.get("metrics", {}).get("total_fees", 0) for status in all_status.values())
        
        # Calcular win rate y profit factor combinados
        win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
        profit_factor = total_profit_sum / total_loss_sum if total_loss_sum > 0 else float('inf')
        
        # Encontrar máximo drawdown
        max_drawdown = max(
            (status.get("metrics", {}).get("max_drawdown", 0) for status in all_status.values()),
            default=0
        )
        
        # Calcular impacto de comisiones
        fee_impact = total_fees / total_profit_sum * 100 if total_profit_sum > 0 else 0
        
        # Identificar mejor y peor bot
        best_bot = max(
            all_status.items(),
            key=lambda x: x[1].get("metrics", {}).get("roi", 0),
            default=(None, {})
        )
        
        worst_bot = min(
            all_status.items(),
            key=lambda x: x[1].get("metrics", {}).get("roi", 0),
            default=(None, {})
        )
        
        # Calcular rendimiento diario y mensual combinado
        daily_profit = sum(status.get("daily_metrics", {}).get("profit", 0) for status in all_status.values())
        daily_trades = sum(status.get("daily_metrics", {}).get("total_trades", 0) for status in all_status.values())
        daily_roi = daily_profit / total_current_balance * 100 if total_current_balance > 0 else 0
        daily_fees = sum(status.get("daily_metrics", {}).get("fees", 0) for status in all_status.values())
        daily_fee_impact = daily_fees / daily_profit * 100 if daily_profit > 0 else 0
        daily_win_rate = 0
        
        if daily_trades > 0:
            daily_profitable_trades = sum(
                int(status.get("daily_metrics", {}).get("win_rate", 0) * status.get("daily_metrics", {}).get("total_trades", 0) / 100)
                for status in all_status.values()
            )
            daily_win_rate = daily_profitable_trades / daily_trades * 100
        
        # Calcular rendimiento mensual combinado
        monthly_profit = sum(status.get("monthly_metrics", {}).get("profit", 0) for status in all_status.values())
        monthly_trades = sum(status.get("monthly_metrics", {}).get("total_trades", 0) for status in all_status.values())
        monthly_roi = monthly_profit / total_current_balance * 100 if total_current_balance > 0 else 0
        monthly_fees = sum(status.get("monthly_metrics", {}).get("fees", 0) for status in all_status.values())
        monthly_fee_impact = monthly_fees / monthly_profit * 100 if monthly_profit > 0 else 0
        monthly_win_rate = 0
        
        if monthly_trades > 0:
            monthly_profitable_trades = sum(
                int(status.get("monthly_metrics", {}).get("win_rate", 0) * status.get("monthly_metrics", {}).get("total_trades", 0) / 100)
                for status in all_status.values()
            )
            monthly_win_rate = monthly_profitable_trades / monthly_trades * 100
        
        return {
            "total_bots": total_bots,
            "running_bots": running_bots,
            "total_initial_balance": total_initial_balance,
            "total_current_balance": total_current_balance,
            "total_profit": total_profit,
            "total_roi": total_roi,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "total_fees": total_fees,
            "fee_impact": fee_impact,
            "best_bot": {
                "id": best_bot[0],
                "roi": best_bot[1].get("metrics", {}).get("roi", 0) if best_bot[0] else 0,
                "profit": best_bot[1].get("metrics", {}).get("total_profit", 0) - best_bot[1].get("metrics", {}).get("total_loss", 0) if best_bot[0] else 0
            },
            "worst_bot": {
                "id": worst_bot[0],
                "roi": worst_bot[1].get("metrics", {}).get("roi", 0) if worst_bot[0] else 0,
                "profit": worst_bot[1].get("metrics", {}).get("total_profit", 0) - worst_bot[1].get("metrics", {}).get("total_loss", 0) if worst_bot[0] else 0
            },
            "daily_performance": {
                "profit": daily_profit,
                "profit_percent": daily_roi,
                "trades": daily_trades,
                "win_rate": daily_win_rate,
                "fees": daily_fees,
                "fees_percent": daily_fee_impact
            },
            "monthly_performance": {
                "profit": monthly_profit,
                "profit_percent": monthly_roi,
                "trades": monthly_trades,
                "win_rate": monthly_win_rate,
                "fees": monthly_fees,
                "fees_percent": monthly_fee_impact
            }
        }
    
    def create_standard_bot_set(self, symbol: str = "SOL-USDT") -> Dict[str, str]:
        """
        Crea un conjunto estándar de bots para diferentes estrategias y timeframes.
        
        Args:
            symbol: Par de trading a utilizar
            
        Returns:
            Dict[str, str]: IDs de los bots creados
        """
        created_bots = {}
        
        # Crear conjunto de bots para scalping
        scalping_timeframes = ["1m", "5m", "15m"]
        scalping_strategies = ["breakout_scalping", "momentum_scalping", "mean_reversion"]
        
        for strategy in scalping_strategies:
            for timeframe in scalping_timeframes:
                bot_id = f"{strategy}_{timeframe}_{symbol.replace('-', '')}"
                created_bots[bot_id] = self.create_bot(
                    bot_id=bot_id,
                    strategy_name=strategy,
                    symbol=symbol,
                    timeframe=timeframe,
                    initial_balance=100.0,
                    leverage=1,
                    market_type="spot",
                    params=self._get_default_params(strategy)
                )
        
        # Crear bots con ML Adaptativo en diferentes timeframes
        ml_timeframes = ["5m", "15m", "1h", "4h"]
        
        for timeframe in ml_timeframes:
            bot_id = f"ml_adaptive_{timeframe}_{symbol.replace('-', '')}"
            created_bots[bot_id] = self.create_bot(
                bot_id=bot_id,
                strategy_name="ml_adaptive",
                symbol=symbol,
                timeframe=timeframe,
                initial_balance=100.0,
                leverage=1,
                market_type="spot",
                params=self._get_default_params("ml_adaptive")
            )
        
        # Crear algunos bots de futuros
        futures_strategies = ["breakout_scalping", "momentum_scalping"]
        futures_timeframes = ["5m", "15m"]
        futures_leverages = [3, 5]
        
        for strategy, timeframe, leverage in zip(futures_strategies, futures_timeframes, futures_leverages):
            bot_id = f"{strategy}_futures_{leverage}x_{timeframe}_{symbol.replace('-', '')}"
            created_bots[bot_id] = self.create_bot(
                bot_id=bot_id,
                strategy_name=strategy,
                symbol=symbol,
                timeframe=timeframe,
                initial_balance=100.0,
                leverage=leverage,
                market_type="futures",
                params=self._get_default_params(strategy)
            )
        
        logger.info(f"Creado conjunto estándar de {len(created_bots)} bots para {symbol}")
        return created_bots
    
    def _get_default_params(self, strategy_name: str) -> Dict[str, Any]:
        """
        Obtiene parámetros por defecto para una estrategia.
        
        Args:
            strategy_name: Nombre de la estrategia
            
        Returns:
            Dict[str, Any]: Parámetros por defecto
        """
        if strategy_name == "breakout_scalping":
            return {
                "take_profit_pct": 0.8,
                "stop_loss_pct": 0.5,
                "trailing_stop_pct": 0.3,
                "max_position_size_pct": 50.0,
                "min_volume_threshold": 1.5,
                "min_rr_ratio": 1.5,
                "max_fee_impact_pct": 0.15
            }
        elif strategy_name == "momentum_scalping":
            return {
                "take_profit_pct": 0.6,
                "stop_loss_pct": 0.4,
                "trailing_stop_pct": 0.2,
                "max_position_size_pct": 50.0,
                "rsi_period": 7,
                "ema_fast": 5,
                "ema_slow": 8,
                "min_rr_ratio": 1.5,
                "max_fee_impact_pct": 0.15
            }
        elif strategy_name == "mean_reversion":
            return {
                "take_profit_pct": 0.7,
                "stop_loss_pct": 0.5,
                "trailing_stop_pct": 0.3,
                "max_position_size_pct": 50.0,
                "bb_period": 20,
                "bb_std": 2.0,
                "rsi_period": 7,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "min_rr_ratio": 1.5,
                "max_fee_impact_pct": 0.15
            }
        elif strategy_name == "ml_adaptive":
            return {
                "confidence_threshold": 0.65,
                "take_profit_pct": 0.8,
                "stop_loss_pct": 0.6,
                "trailing_stop_pct": 0.3,
                "max_position_size_pct": 50.0,
                "min_rr_ratio": 1.5,
                "max_fee_impact_pct": 0.15
            }
        
        # Valores por defecto genéricos
        return {
            "take_profit_pct": 0.7,
            "stop_loss_pct": 0.5,
            "max_position_size_pct": 50.0
        }

def get_multi_bot_manager(config_file: str = "data/multi_bot_config.json") -> MultiBotManager:
    """
    Función de conveniencia para obtener una instancia del gestor de bots.
    
    Args:
        config_file: Archivo de configuración
        
    Returns:
        MultiBotManager: Instancia del gestor de bots
    """
    return MultiBotManager(config_file)

def demo_multi_bot_manager():
    """Demostración del gestor de múltiples bots."""
    print("\n🤖 GESTOR DE MÚLTIPLES BOTS DE TRADING 🤖")
    print("Este sistema permite ejecutar y monitorear múltiples bots")
    print("con diferentes estrategias, timeframes y pares de trading.")
    
    # Crear gestor de bots
    manager = MultiBotManager("data/demo_multi_bot_config.json")
    
    # Crear conjunto estándar de bots para SOL-USDT
    created_bots = manager.create_standard_bot_set("SOL-USDT")
    
    print(f"\n1. Creados {len(created_bots)} bots para SOL-USDT")
    print(f"   Primeros 5 bots:")
    for i, (bot_id, created_id) in enumerate(list(created_bots.items())[:5]):
        print(f"   {i+1}. {bot_id}")
    
    # Iniciar todos los bots
    start_results = manager.start_all_bots()
    
    print(f"\n2. Iniciados {sum(1 for r in start_results.values() if r)} bots")
    
    # Esperar un poco para que los bots ejecuten algunas operaciones simuladas
    print("\n   Esperando 3 segundos para que los bots ejecuten operaciones simuladas...")
    time.sleep(3)
    
    # Obtener rendimiento combinado
    performance = manager.get_combined_performance()
    
    print("\n3. Rendimiento combinado:")
    print(f"   Total bots: {performance['total_bots']}")
    print(f"   Bots en ejecución: {performance['running_bots']}")
    print(f"   Balance total: ${performance['total_current_balance']:.2f}")
    print(f"   Operaciones totales: {performance['total_trades']}")
    
    print("\n4. Rendimiento por períodos:")
    print(f"   Diario:")
    print(f"     - Ganancia: ${performance['daily_performance']['profit']:.2f} ({performance['daily_performance']['profit_percent']:.2f}%)")
    print(f"     - Operaciones: {performance['daily_performance']['trades']}")
    print(f"     - Win rate: {performance['daily_performance']['win_rate']:.2f}%")
    print(f"     - Comisiones: ${performance['daily_performance']['fees']:.2f} ({performance['daily_performance']['fees_percent']:.2f}%)")
    
    print(f"   Mensual:")
    print(f"     - Ganancia: ${performance['monthly_performance']['profit']:.2f} ({performance['monthly_performance']['profit_percent']:.2f}%)")
    print(f"     - Operaciones: {performance['monthly_performance']['trades']}")
    print(f"     - Win rate: {performance['monthly_performance']['win_rate']:.2f}%")
    print(f"     - Comisiones: ${performance['monthly_performance']['fees']:.2f} ({performance['monthly_performance']['fees_percent']:.2f}%)")
    
    # Detener todos los bots
    stop_results = manager.stop_all_bots()
    
    print(f"\n5. Detenidos {sum(1 for r in stop_results.values() if r)} bots")
    
    print("\n✅ Demostración completada.")
    
    return manager

if __name__ == "__main__":
    try:
        manager = demo_multi_bot_manager()
    except Exception as e:
        print(f"Error en la demostración: {e}")