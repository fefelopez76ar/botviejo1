"""
Módulo de simulación realista para testing del bot de trading

Este módulo implementa:
1. Simulación de costos de trading (fees y slippage)
2. Simulación de latencia de red
3. Registro detallado de operaciones para evaluación de rendimiento
4. Generación de reportes de rendimiento
"""

import os
import json
import time
import random
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Callable

# Importar módulos del sistema
from stats_tracker import TradeStats
from risk_management.drawdown_monitor import DrawdownMonitor, CircuitBreaker
from risk_management.position_limits import PositionSizeManager
from adaptive_weighting import AdaptiveWeightingSystem, MarketCondition, TimeInterval
from pattern_recognition import PatternRecognition, PatternType

# Configurar logging
logger = logging.getLogger(__name__)

class TradingSimulator:
    """
    Simulador realista de trading para probar estrategias
    
    Esta clase simula condiciones de mercado realistas incluyendo:
    - Fees de exchange
    - Slippage (deslizamiento de precio)
    - Latencia de red
    - Fallos ocasionales en ejecución de órdenes
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el simulador de trading
        
        Args:
            config: Configuración del simulador
        """
        self.config = config or {
            # Parámetros de simulación
            'initial_balance': 10000.0,  # 10,000 USDT
            'asset': 'SOL',
            'quote': 'USDT',
            'fees': 0.001,               # 0.1% fee por operación
            'min_slippage': 0.0005,      # 0.05% mínimo de slippage
            'max_slippage': 0.002,       # 0.2% máximo de slippage
            'min_latency': 100,          # 100ms latencia mínima
            'max_latency': 500,          # 500ms latencia máxima
            'order_fail_probability': 0.01,  # 1% probabilidad de fallo de orden
            
            # Parámetros de posición
            'max_position_size': 0.2,    # 20% del balance como máximo
            'default_stop_loss': 0.02,   # 2% stop loss por defecto
            'default_take_profit': 0.03, # 3% take profit por defecto
            
            # Parámetros de gestión de riesgo
            'max_drawdown': 0.15,        # 15% drawdown máximo
            'max_consecutive_losses': 3,  # 3 pérdidas consecutivas máximo
            
            # Parámetros de almacenamiento
            'results_file': 'simulation_results.json',
            'trades_file': 'simulation_trades.json',
            
            # Mostrar progreso en tiempo real
            'show_progress': True,
            
            # Seed para reproducibilidad (None para aleatorio)
            'random_seed': None
        }
        
        # Establecer semilla para reproducibilidad si se proporciona
        if self.config['random_seed'] is not None:
            random.seed(self.config['random_seed'])
            np.random.seed(self.config['random_seed'])
        
        # Estado de simulación
        self.balance = self.config['initial_balance']
        self.equity = self.balance
        self.peak_equity = self.balance
        self.drawdown = 0.0
        self.max_drawdown = 0.0
        self.current_position = None
        self.open_positions = []
        self.closed_positions = []
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.consecutive_losses = 0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        
        # Inicializar sistemas de soporte
        self.stats = TradeStats(stats_file="simulation_stats.json")
        self.drawdown_monitor = DrawdownMonitor()
        self.circuit_breaker = CircuitBreaker()
        self.position_sizer = PositionSizeManager()
        self.pattern_recognizer = PatternRecognition()
        
        # Conectar sistemas
        self.circuit_breaker.connect_drawdown_monitor(self.drawdown_monitor)
        
        # Registrar callbacks para circuit breakers
        self.circuit_breaker.add_activation_callback(self._circuit_breaker_callback)
        
        # Flag de pausa para circuit breakers
        self.paused = False
        
        logger.info("Simulador de trading inicializado con balance: ${:.2f}".format(self.balance))
    
    def _circuit_breaker_callback(self, breaker_type: str, reason: str) -> None:
        """
        Callback cuando se activa un circuit breaker
        
        Args:
            breaker_type: Tipo de circuit breaker
            reason: Razón de activación
        """
        logger.warning(f"CIRCUIT BREAKER ACTIVADO: {breaker_type} - {reason}")
        self.paused = True
        
        # Cerrar posiciones actuales
        if self.current_position:
            logger.info("Cerrando posiciones abiertas debido a circuit breaker...")
            self.close_position(reason="circuit_breaker")
    
    def reset(self) -> None:
        """Reinicia el simulador para una nueva simulación"""
        self.balance = self.config['initial_balance']
        self.equity = self.balance
        self.peak_equity = self.balance
        self.drawdown = 0.0
        self.max_drawdown = 0.0
        self.current_position = None
        self.open_positions = []
        self.closed_positions = []
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.consecutive_losses = 0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.paused = False
        
        logger.info("Simulador reiniciado con balance: ${:.2f}".format(self.balance))
    
    def _simulate_latency(self) -> int:
        """
        Simula latencia de red
        
        Returns:
            int: Latencia simulada en ms
        """
        min_latency = self.config['min_latency']
        max_latency = self.config['max_latency']
        
        # Distribución triangular para latencia (más frecuente cerca del mínimo)
        latency = random.triangular(min_latency, max_latency, min_latency * 1.5)
        
        # Simular latencia real
        if self.config.get('apply_real_delay', False):
            time.sleep(latency / 1000)
        
        return int(latency)
    
    def _simulate_slippage(self, price: float, side: str) -> float:
        """
        Simula slippage en el precio
        
        Args:
            price: Precio base
            side: 'buy' o 'sell'
            
        Returns:
            float: Precio con slippage
        """
        min_slippage = self.config['min_slippage']
        max_slippage = self.config['max_slippage']
        
        # Distribución exponencial para slippage (más frecuente cerca del mínimo)
        slippage_pct = min_slippage + random.expovariate(1.0 / (max_slippage - min_slippage))
        slippage_pct = min(slippage_pct, max_slippage)
        
        # Slippage positivo (precio sube) para compras, negativo (precio baja) para ventas
        if side == 'buy':
            slippage_amount = price * slippage_pct
            new_price = price + slippage_amount
        else:
            slippage_amount = price * slippage_pct
            new_price = price - slippage_amount
        
        # Registrar slippage total
        self.total_slippage += abs(slippage_amount)
        
        return new_price
    
    def _calculate_fee(self, price: float, size: float) -> float:
        """
        Calcula fee de operación
        
        Args:
            price: Precio de operación
            size: Tamaño de operación en unidades
            
        Returns:
            float: Fee en quote currency
        """
        fee_pct = self.config['fees']
        fee_amount = price * size * fee_pct
        
        # Registrar fee total
        self.total_fees += fee_amount
        
        return fee_amount
    
    def _simulate_order_execution(self, side: str, price: float, size: float) -> Dict[str, Any]:
        """
        Simula ejecución de orden con condiciones realistas
        
        Args:
            side: 'buy' o 'sell'
            price: Precio solicitado
            size: Tamaño en unidades
            
        Returns:
            Dict: Resultado de la ejecución
        """
        # Simular latencia
        latency = self._simulate_latency()
        
        # Simular fallo ocasional
        if random.random() < self.config['order_fail_probability']:
            return {
                'success': False,
                'error': 'order_execution_failed',
                'latency': latency,
                'message': 'Fallo aleatorio de orden simulado'
            }
        
        # Simular slippage
        executed_price = self._simulate_slippage(price, side)
        
        # Calcular fee
        fee = self._calculate_fee(executed_price, size)
        
        # Calcular valor total
        value = executed_price * size
        
        return {
            'success': True,
            'side': side,
            'requested_price': price,
            'executed_price': executed_price,
            'size': size,
            'value': value,
            'fee': fee,
            'latency': latency,
            'slippage_pct': abs((executed_price - price) / price),
            'timestamp': datetime.now().isoformat()
        }
    
    def open_position(self, side: str, price: float, size: float, 
                    reason: str = "", stop_loss: Optional[float] = None, 
                    take_profit: Optional[float] = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Abre una posición
        
        Args:
            side: 'long' o 'short'
            price: Precio de entrada
            size: Tamaño en unidades
            reason: Razón de entrada
            stop_loss: Precio de stop loss (opcional)
            take_profit: Precio de take profit (opcional)
            metadata: Metadatos adicionales (opcional)
            
        Returns:
            Dict: Resultado de la operación
        """
        # Verificar si el simulador está pausado
        if self.paused:
            return {
                'success': False,
                'error': 'trading_paused',
                'message': 'Trading pausado por circuit breaker'
            }
        
        # Verificar si ya hay una posición abierta
        if self.current_position:
            return {
                'success': False,
                'error': 'position_exists',
                'message': 'Ya existe una posición abierta'
            }
        
        # Verificar si hay suficiente balance
        position_value = price * size
        if position_value > self.balance:
            return {
                'success': False,
                'error': 'insufficient_balance',
                'message': 'Balance insuficiente'
            }
        
        # Simular ejecución de orden
        order_side = 'buy' if side == 'long' else 'sell'
        execution = self._simulate_order_execution(order_side, price, size)
        
        if not execution['success']:
            return execution
        
        # Establecer stop loss y take profit
        if stop_loss is None:
            if side == 'long':
                stop_loss = execution['executed_price'] * (1 - self.config['default_stop_loss'])
            else:
                stop_loss = execution['executed_price'] * (1 + self.config['default_stop_loss'])
        
        if take_profit is None:
            if side == 'long':
                take_profit = execution['executed_price'] * (1 + self.config['default_take_profit'])
            else:
                take_profit = execution['executed_price'] * (1 - self.config['default_take_profit'])
        
        # Crear posición
        position = {
            'id': len(self.closed_positions) + len(self.open_positions) + 1,
            'side': side,
            'entry_price': execution['executed_price'],
            'current_price': execution['executed_price'],
            'size': size,
            'value': execution['value'],
            'fee': execution['fee'],
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now().isoformat(),
            'reason': reason,
            'metadata': metadata or {},
            'latency': execution['latency'],
            'slippage': execution['slippage_pct'],
            'status': 'open'
        }
        
        # Reducir balance
        self.balance -= execution['value'] + execution['fee']
        
        # Agregar posición
        self.current_position = position
        self.open_positions.append(position)
        
        # Registrar operación
        logger.info(f"Posición {position['id']} abierta: {side} a ${execution['executed_price']:.4f}, "
                   f"tamaño: {size}, valor: ${execution['value']:.2f}")
        
        return {
            'success': True,
            'position': position
        }
    
    def close_position(self, position_id: Optional[int] = None, 
                     price: Optional[float] = None, reason: str = "") -> Dict[str, Any]:
        """
        Cierra una posición
        
        Args:
            position_id: ID de la posición a cerrar (opcional, si None cierra la posición actual)
            price: Precio de salida (opcional, si None usa precio de mercado)
            reason: Razón de salida (opcional)
            
        Returns:
            Dict: Resultado de la operación
        """
        # Encontrar posición a cerrar
        position = None
        
        if position_id is not None:
            # Buscar por ID
            for pos in self.open_positions:
                if pos['id'] == position_id:
                    position = pos
                    break
            
            if not position:
                return {
                    'success': False,
                    'error': 'position_not_found',
                    'message': f'Posición con ID {position_id} no encontrada'
                }
        else:
            # Usar posición actual
            position = self.current_position
            
            if not position:
                return {
                    'success': False,
                    'error': 'no_open_position',
                    'message': 'No hay posición abierta para cerrar'
                }
        
        # Si no se proporciona precio, usar precio de mercado (entrada +/- 0.1%)
        if price is None:
            # Simular pequeño movimiento desde precio de entrada
            price_change = random.uniform(-0.001, 0.001)
            price = position['entry_price'] * (1 + price_change)
        
        # Simular ejecución de orden
        order_side = 'sell' if position['side'] == 'long' else 'buy'
        execution = self._simulate_order_execution(order_side, price, position['size'])
        
        if not execution['success']:
            return execution
        
        # Calcular PnL
        if position['side'] == 'long':
            pnl = (execution['executed_price'] - position['entry_price']) * position['size']
        else:
            pnl = (position['entry_price'] - execution['executed_price']) * position['size']
        
        # Restar fees
        pnl -= execution['fee'] + position['fee']
        
        # Actualizar posición
        position['exit_price'] = execution['executed_price']
        position['exit_time'] = datetime.now().isoformat()
        position['close_reason'] = reason
        position['pnl'] = pnl
        position['status'] = 'closed'
        position['exit_latency'] = execution['latency']
        position['exit_slippage'] = execution['slippage_pct']
        
        # Actualizar balance y estadísticas
        self.balance += execution['value'] + pnl
        
        # Actualizar contadores
        self.trade_count += 1
        
        if pnl > 0:
            self.win_count += 1
            self.consecutive_losses = 0
        else:
            self.loss_count += 1
            self.consecutive_losses += 1
        
        # Actualizar estadísticas
        self.update_stats(position)
        
        # Actualizar circuit breaker
        self.circuit_breaker.process_trade_result(pnl, self.balance)
        
        # Remover de posiciones abiertas
        if position in self.open_positions:
            self.open_positions.remove(position)
        
        # Agregar a posiciones cerradas
        self.closed_positions.append(position)
        
        # Resetear posición actual
        if self.current_position and self.current_position['id'] == position['id']:
            self.current_position = None
        
        # Registrar operación
        logger.info(f"Posición {position['id']} cerrada: {position['side']} a ${execution['executed_price']:.4f}, "
                   f"PnL: ${pnl:.2f}, Razón: {reason}")
        
        return {
            'success': True,
            'position': position,
            'pnl': pnl
        }
    
    def update_position(self, current_price: float) -> None:
        """
        Actualiza la posición actual con el precio actual
        
        Args:
            current_price: Precio actual
        """
        if not self.current_position:
            return
        
        position = self.current_position
        position['current_price'] = current_price
        
        # Calcular PnL no realizado
        if position['side'] == 'long':
            unrealized_pnl = (current_price - position['entry_price']) * position['size']
        else:
            unrealized_pnl = (position['entry_price'] - current_price) * position['size']
        
        position['unrealized_pnl'] = unrealized_pnl
        
        # Actualizar equity
        self.equity = self.balance + unrealized_pnl
        
        # Actualizar drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        self.drawdown = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        self.max_drawdown = max(self.max_drawdown, self.drawdown)
        
        # Actualizar monitor de drawdown
        self.drawdown_monitor.update_equity(self.equity)
        
        # Verificar stop loss y take profit
        if position['side'] == 'long':
            if current_price <= position['stop_loss']:
                self.close_position(reason="stop_loss", price=position['stop_loss'])
            elif current_price >= position['take_profit']:
                self.close_position(reason="take_profit", price=position['take_profit'])
        else:
            if current_price >= position['stop_loss']:
                self.close_position(reason="stop_loss", price=position['stop_loss'])
            elif current_price <= position['take_profit']:
                self.close_position(reason="take_profit", price=position['take_profit'])
    
    def update_stats(self, position: Dict[str, Any]) -> None:
        """
        Actualiza estadísticas del sistema
        
        Args:
            position: Posición cerrada
        """
        # Convertir a formato para stats_tracker
        trade_data = {
            'entry_time': position['entry_time'],
            'exit_time': position['exit_time'],
            'entry_price': position['entry_price'],
            'exit_price': position['exit_price'],
            'position_type': position['side'],
            'profit': position['pnl'],
            'reason': position.get('reason', ''),
            'close_reason': position.get('close_reason', ''),
            'size': position['size'],
            'metadata': position.get('metadata', {})
        }
        
        # Registrar operación en stats_tracker
        self.stats.record_trade(trade_data)
        
        # Actualizar performance del patrón si aplica
        if 'pattern' in position.get('metadata', {}):
            pattern_type = position['metadata']['pattern']
            market_condition = position['metadata'].get('market_condition', 'normal')
            
            try:
                # Convertir strings a enums
                pattern_enum = PatternType(pattern_type)
                market_enum = MarketCondition(market_condition)
                
                # Actualizar estadísticas del patrón
                self.pattern_recognizer.update_pattern_performance(
                    pattern_enum, 
                    position['pnl'] > 0, 
                    position['pnl'],
                    market_enum
                )
            except (ValueError, KeyError) as e:
                logger.warning(f"No se pudo actualizar rendimiento del patrón: {e}")
        
        # Actualizar estrategia adaptativa si aplica
        if 'signals' in position.get('metadata', {}):
            signals = position['metadata']['signals']
            market_condition = position['metadata'].get('market_condition', 'LATERAL_LOW_VOL')
            time_interval = position['metadata'].get('time_interval', 'MINUTE_5')
            
            try:
                # Convertir strings a enums si es necesario
                if isinstance(market_condition, str):
                    market_condition = MarketCondition[market_condition]
                
                if isinstance(time_interval, str):
                    time_interval = TimeInterval[time_interval]
                
                # Actualizar para cada indicador
                for indicator, signal in signals.items():
                    # Instanciar sistema si no existe
                    if not hasattr(self, 'adaptive_system'):
                        self.adaptive_system = AdaptiveWeightingSystem()
                    
                    # Actualizar rendimiento del indicador
                    self.adaptive_system.update_indicator_performance(
                        indicator,
                        position['pnl'] > 0,
                        position['pnl'],
                        market_condition,
                        time_interval
                    )
            except (ValueError, KeyError, AttributeError) as e:
                logger.warning(f"No se pudo actualizar pesos adaptativos: {e}")
        
        # Actualizar posición sizer
        self.position_sizer.record_trade_result(trade_data)
    
    def run_simulation(self, df: pd.DataFrame, strategy_fn: Callable) -> Dict[str, Any]:
        """
        Ejecuta simulación completa con una estrategia
        
        Args:
            df: DataFrame con datos OHLCV
            strategy_fn: Función que implementa la estrategia
            
        Returns:
            Dict: Resultados de la simulación
        """
        # Reiniciar simulador
        self.reset()
        
        # Verificar que hay suficientes datos
        if len(df) < 10:
            logger.error("Insuficientes datos para simulación (mínimo 10 velas)")
            return {
                'success': False,
                'error': 'insufficient_data',
                'message': 'Insuficientes datos para simulación'
            }
        
        # Columnas necesarias
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Columna requerida '{col}' no encontrada en DataFrame")
                return {
                    'success': False,
                    'error': 'missing_column',
                    'message': f"Columna requerida '{col}' no encontrada"
                }
        
        # Ejecutar simulación sobre cada vela
        for i in range(len(df)):
            # Si está pausado por circuit breaker, verificar si se puede reanudar
            if self.paused:
                can_resume, reason = self.circuit_breaker.can_resume_trading()
                if can_resume:
                    logger.info("Reanudando trading después de circuit breaker")
                    self.paused = False
                else:
                    logger.debug(f"Trading sigue pausado: {reason}")
                    continue
            
            # Obtener vela actual y slice de datos hasta el momento
            current_candle = df.iloc[i]
            data_slice = df.iloc[:i+1]
            
            # Determinar precio actual
            current_price = current_candle['close']
            
            # Actualizar posición existente
            self.update_position(current_price)
            
            # Saltear si hay posición abierta
            if self.current_position:
                continue
            
            # Llamar a la estrategia para obtener señal
            signal = strategy_fn(data_slice)
            
            # Si no hay señal, continuar
            if not signal or 'action' not in signal:
                continue
            
            # Procesar señal
            action = signal['action']
            
            if action == 'buy':
                # Calcular tamaño basado en gestión de riesgo
                size_in_quote = self.position_sizer.calculate_position_size(
                    self.balance,
                    signal.get('market_condition', 'normal'),
                    signal.get('risk_level', 1.0)
                )
                
                # Convertir a unidades
                size = size_in_quote / current_price
                
                # Abrir posición larga
                result = self.open_position(
                    side='long',
                    price=current_price,
                    size=size,
                    reason=signal.get('reason', 'strategy_signal'),
                    stop_loss=signal.get('stop_loss'),
                    take_profit=signal.get('take_profit'),
                    metadata=signal.get('metadata', {})
                )
                
                if not result['success']:
                    logger.warning(f"No se pudo abrir posición larga: {result.get('message')}")
            
            elif action == 'sell':
                # Calcular tamaño basado en gestión de riesgo
                size_in_quote = self.position_sizer.calculate_position_size(
                    self.balance,
                    signal.get('market_condition', 'normal'),
                    signal.get('risk_level', 1.0)
                )
                
                # Convertir a unidades
                size = size_in_quote / current_price
                
                # Abrir posición corta
                result = self.open_position(
                    side='short',
                    price=current_price,
                    size=size,
                    reason=signal.get('reason', 'strategy_signal'),
                    stop_loss=signal.get('stop_loss'),
                    take_profit=signal.get('take_profit'),
                    metadata=signal.get('metadata', {})
                )
                
                if not result['success']:
                    logger.warning(f"No se pudo abrir posición corta: {result.get('message')}")
            
            # Mostrar progreso si está habilitado
            if self.config['show_progress'] and i % 100 == 0:
                progress = (i + 1) / len(df) * 100
                logger.info(f"Simulación en progreso: {progress:.1f}% ({i+1}/{len(df)})")
        
        # Cerrar posición final si sigue abierta
        if self.current_position:
            self.close_position(reason="end_of_simulation")
        
        # Calcular y retornar resultados
        return self.get_results()
    
    def get_results(self) -> Dict[str, Any]:
        """
        Obtiene resultados de la simulación
        
        Returns:
            Dict: Resultados detallados
        """
        # Calcular métricas
        win_rate = self.win_count / self.trade_count if self.trade_count > 0 else 0
        
        # Total P&L
        total_pnl = self.balance - self.config['initial_balance']
        pnl_percentage = total_pnl / self.config['initial_balance'] * 100 if self.config['initial_balance'] > 0 else 0
        
        # Calcular ganancia/pérdida promedio
        avg_win = 0
        avg_loss = 0
        total_win = 0
        total_loss = 0
        
        for pos in self.closed_positions:
            if pos['pnl'] > 0:
                total_win += pos['pnl']
            else:
                total_loss += abs(pos['pnl'])
        
        avg_win = total_win / self.win_count if self.win_count > 0 else 0
        avg_loss = total_loss / self.loss_count if self.loss_count > 0 else 0
        
        # Calcular profit factor
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        
        # Recopilar resultados
        results = {
            'initial_balance': self.config['initial_balance'],
            'final_balance': self.balance,
            'total_pnl': total_pnl,
            'pnl_percentage': pnl_percentage,
            'max_drawdown': self.max_drawdown * 100,
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_fees': self.total_fees,
            'total_slippage': self.total_slippage,
            'closed_positions': len(self.closed_positions),
            'asset': self.config['asset'],
            'quote': self.config['quote'],
            'simulation_time': datetime.now().isoformat()
        }
        
        # Guardar resultados en archivo si está configurado
        if self.config.get('save_results', True):
            self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Guarda resultados en archivo JSON
        
        Args:
            results: Resultados a guardar
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.config['results_file']) or '.', exist_ok=True)
            
            # Guardar resultados
            with open(self.config['results_file'], 'w') as f:
                json.dump(results, f, indent=4)
            
            # Guardar historial de trades
            with open(self.config['trades_file'], 'w') as f:
                json.dump(self.closed_positions, f, indent=4)
            
            logger.info(f"Resultados guardados en {self.config['results_file']}")
            logger.info(f"Historial de trades guardado en {self.config['trades_file']}")
        except Exception as e:
            logger.error(f"Error al guardar resultados: {e}")
    
    def generate_report(self, output_file: str = "simulation_report.html") -> None:
        """
        Genera reporte HTML de la simulación
        
        Args:
            output_file: Archivo de salida
        """
        try:
            # Obtener resultados
            results = self.get_results()
            
            # Crear informe básico
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Reporte de Simulación de Trading</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
                    .header {{ background-color: #007bff; color: white; padding: 1em; border-radius: 5px; margin-bottom: 20px; }}
                    .metrics {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; margin-bottom: 30px; }}
                    .metric-card {{ background-color: #f8f9fa; border-radius: 5px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    .metric-title {{ font-size: 0.9em; color: #666; margin-bottom: 5px; }}
                    .metric-value {{ font-size: 1.4em; font-weight: bold; }}
                    .positive {{ color: #28a745; }}
                    .negative {{ color: #dc3545; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                    th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f8f9fa; }}
                    tr:hover {{ background-color: #f1f1f1; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Reporte de Simulación de Trading</h1>
                    <p>Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                </div>
                
                <h2>Resumen de Resultados</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-title">Balance Inicial</div>
                        <div class="metric-value">${results['initial_balance']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Balance Final</div>
                        <div class="metric-value">${results['final_balance']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Beneficio/Pérdida</div>
                        <div class="metric-value {('positive' if results['total_pnl'] >= 0 else 'negative')}">
                            ${results['total_pnl']:.2f} ({results['pnl_percentage']:.2f}%)
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Drawdown Máximo</div>
                        <div class="metric-value negative">{results['max_drawdown']:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Operaciones</div>
                        <div class="metric-value">{results['trade_count']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Win Rate</div>
                        <div class="metric-value {('positive' if results['win_rate'] >= 50 else 'negative')}">
                            {results['win_rate']:.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Profit Factor</div>
                        <div class="metric-value {('positive' if results['profit_factor'] >= 1 else 'negative')}">
                            {results['profit_factor']:.2f}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Ganancia Promedio</div>
                        <div class="metric-value positive">${results['avg_win']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Pérdida Promedio</div>
                        <div class="metric-value negative">${results['avg_loss']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Costos (Fees)</div>
                        <div class="metric-value">${results['total_fees']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Costos (Slippage)</div>
                        <div class="metric-value">${results['total_slippage']:.2f}</div>
                    </div>
                </div>
                
                <h2>Historial de Operaciones</h2>
                <table>
                    <tr>
                        <th>#</th>
                        <th>Tipo</th>
                        <th>Entrada</th>
                        <th>Salida</th>
                        <th>P/L</th>
                        <th>Razón</th>
                    </tr>
            """
            
            # Agregar historial de operaciones
            for i, pos in enumerate(self.closed_positions, 1):
                pnl_class = 'positive' if pos['pnl'] >= 0 else 'negative'
                
                # Formatear fechas
                entry_time = datetime.fromisoformat(pos['entry_time']).strftime('%Y-%m-%d %H:%M:%S')
                exit_time = datetime.fromisoformat(pos['exit_time']).strftime('%Y-%m-%d %H:%M:%S')
                
                html += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{pos['side'].upper()}</td>
                        <td>${pos['entry_price']:.4f}</td>
                        <td>${pos['exit_price']:.4f}</td>
                        <td class="{pnl_class}">${pos['pnl']:.2f}</td>
                        <td>{pos.get('close_reason', '')}</td>
                    </tr>
                """
            
            html += """
                </table>
            </body>
            </html>
            """
            
            # Escribir reporte a archivo
            with open(output_file, 'w') as f:
                f.write(html)
            
            logger.info(f"Reporte generado en: {output_file}")
            
        except Exception as e:
            logger.error(f"Error al generar reporte: {e}")