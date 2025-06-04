#!/usr/bin/env python3
"""
Estrategias de scalping para Solana optimizadas para operaciones de corto plazo.

Este módulo implementa estrategias de trading de alta frecuencia (scalping) 
diseñadas específicamente para Solana, considerando su volatilidad, rapidez
de ejecución y características únicas del mercado de criptomonedas.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
import time

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ScalpingStrategies')

class ScalpingStrategies:
    """
    Implementa estrategias de scalping para operaciones de alta frecuencia.
    
    Las estrategias de scalping están diseñadas para aprovechar movimientos
    pequeños de precio en intervalos muy cortos de tiempo, con una gestión
    estricta de riesgos y objetivos de ganancia modestos pero constantes.
    """
    
    def __init__(self, 
                max_position_size_pct: float = 2.0,
                take_profit_pct: float = 0.5,
                stop_loss_pct: float = 0.3,
                min_volume_threshold: float = 1.5,
                min_rr_ratio: float = 1.5,
                max_fee_impact_pct: float = 0.15):
        """
        Inicializa las estrategias de scalping con parámetros configurable.
        
        Args:
            max_position_size_pct: Tamaño máximo de posición como % del capital
            take_profit_pct: Objetivo de ganancia en % por operación
            stop_loss_pct: Nivel de stop loss en % por operación
            min_volume_threshold: Umbral mínimo de volumen (ratio vs promedio)
            min_rr_ratio: Ratio mínimo de riesgo/recompensa para entrar
            max_fee_impact_pct: Impacto máximo de comisiones como % de ganancia
        """
        self.max_position_size_pct = max_position_size_pct
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.min_volume_threshold = min_volume_threshold
        self.min_rr_ratio = min_rr_ratio
        self.max_fee_impact_pct = max_fee_impact_pct
        
        # Contador de operaciones
        self.trades_count = 0
        self.successful_trades = 0
        
        # Historial de operaciones
        self.trade_history = []
        
        # Parámetros dinámicos (se ajustan según resultados)
        self.dynamic_params = {
            'take_profit_pct': take_profit_pct,
            'stop_loss_pct': stop_loss_pct,
            'entry_threshold': 0.8  # Umbral para confirmar entrada
        }
    
    def analyze_orderbook(self, 
                         orderbook: Dict[str, Any], 
                         current_price: float,
                         depth_pct: float = 2.0) -> Dict[str, Any]:
        """
        Analiza el libro de órdenes para identificar oportunidades y riesgos.
        
        Args:
            orderbook: Datos del libro de órdenes (bids/asks)
            current_price: Precio actual del activo
            depth_pct: Profundidad del libro a analizar (% desde precio actual)
            
        Returns:
            Dict[str, Any]: Análisis del libro de órdenes
        """
        # Validar datos de entrada
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return {'error': 'Datos de orderbook inválidos'}
        
        bids = orderbook['bids']  # [[precio, cantidad], ...]
        asks = orderbook['asks']  # [[precio, cantidad], ...]
        
        # Calcular límites de profundidad
        lower_bound = current_price * (1 - depth_pct/100)
        upper_bound = current_price * (1 + depth_pct/100)
        
        # Filtrar órdenes dentro del rango de profundidad
        bids_in_range = [b for b in bids if float(b[0]) >= lower_bound]
        asks_in_range = [a for a in asks if float(a[0]) <= upper_bound]
        
        # Calcular volumen total en cada lado
        bid_volume = sum(float(b[1]) for b in bids_in_range)
        ask_volume = sum(float(a[1]) for a in asks_in_range)
        
        # Calcular presión de compra/venta
        buy_sell_ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')
        
        # Identificar paredes de órdenes (concentraciones grandes)
        bid_walls = []
        ask_walls = []
        
        # Umbral para considerar una pared (% del volumen total)
        wall_threshold = 0.15
        
        for bid in bids_in_range:
            bid_price, bid_size = float(bid[0]), float(bid[1])
            if bid_size / bid_volume > wall_threshold:
                bid_walls.append({
                    'price': bid_price,
                    'size': bid_size,
                    'pct_of_total': bid_size / bid_volume * 100
                })
        
        for ask in asks_in_range:
            ask_price, ask_size = float(ask[0]), float(ask[1])
            if ask_size / ask_volume > wall_threshold:
                ask_walls.append({
                    'price': ask_price,
                    'size': ask_size,
                    'pct_of_total': ask_size / ask_volume * 100
                })
        
        # Calcular distancia al soporte/resistencia más cercano (paredes)
        nearest_support = max([b['price'] for b in bid_walls], default=lower_bound)
        nearest_resistance = min([a['price'] for a in ask_walls], default=upper_bound)
        
        # Calcular desequilibrio en el libro
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        
        # Calcular liquidez promedio
        avg_liquidity = (bid_volume + ask_volume) / 2
        
        # Estimación de slippage para una orden de mercado (simulada)
        test_order_size = 10  # Simular orden de 10 unidades
        buy_slippage = self._estimate_slippage(asks_in_range, test_order_size, 'buy')
        sell_slippage = self._estimate_slippage(bids_in_range, test_order_size, 'sell')
        
        return {
            'buy_sell_ratio': buy_sell_ratio,
            'imbalance': imbalance,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_distance_pct': (current_price - nearest_support) / current_price * 100,
            'resistance_distance_pct': (nearest_resistance - current_price) / current_price * 100,
            'buy_slippage_pct': buy_slippage,
            'sell_slippage_pct': sell_slippage,
            'bid_walls': bid_walls,
            'ask_walls': ask_walls,
            'avg_liquidity': avg_liquidity,
            'timestamp': datetime.now().isoformat()
        }
    
    def _estimate_slippage(self, 
                         orders: List[List], 
                         size: float, 
                         side: str) -> float:
        """
        Estima el slippage para una orden de mercado.
        
        Args:
            orders: Lista de órdenes [[precio, cantidad], ...]
            size: Tamaño de la orden
            side: Lado de la orden ('buy' o 'sell')
            
        Returns:
            float: Slippage estimado en porcentaje
        """
        if not orders:
            return 0.0
            
        # Precio base (mejor precio disponible)
        base_price = float(orders[0][0])
        
        # Acumular tamaño hasta cubrir la orden
        remaining_size = size
        avg_price = 0
        total_cost = 0
        
        for order in orders:
            price, quantity = float(order[0]), float(order[1])
            
            if remaining_size <= 0:
                break
                
            executed_qty = min(remaining_size, quantity)
            total_cost += executed_qty * price
            remaining_size -= executed_qty
        
        if size - remaining_size > 0:
            avg_price = total_cost / (size - remaining_size)
        else:
            return 0.0  # No se pudo ejecutar nada
            
        # Calcular slippage
        if side == 'buy':
            slippage = (avg_price - base_price) / base_price * 100
        else:  # sell
            slippage = (base_price - avg_price) / base_price * 100
            
        return max(0, slippage)  # Slippage no puede ser negativo
    
    def breakout_scalping_strategy(self, 
                                 df: pd.DataFrame, 
                                 orderbook: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Estrategia de scalping basada en rupturas de niveles clave.
        
        Args:
            df: DataFrame con datos OHLCV recientes (timeframe corto)
            orderbook: Datos del libro de órdenes (opcional)
            
        Returns:
            Dict[str, Any]: Señal de trading con parámetros
        """
        if len(df) < 20:
            return {'signal': 'neutral', 'message': 'Datos insuficientes'}
        
        # Obtener precio actual y volumen
        current_price = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(10).mean().iloc[-1]
        
        # Identificar niveles clave (soportes y resistencias recientes)
        recent_high = df['high'].iloc[-10:].max()
        recent_low = df['low'].iloc[-10:].min()
        
        # Definir nivel de ruptura superior e inferior
        upper_breakout = recent_high
        lower_breakout = recent_low
        
        # Calcular bandas de Bollinger
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma20'] + (df['std20'] * 2)
        df['lower_band'] = df['sma20'] - (df['std20'] * 2)
        
        bb_upper = df['upper_band'].iloc[-1]
        bb_lower = df['lower_band'].iloc[-1]
        
        # Calcular volatilidad reciente (ATR simplificado)
        df['range'] = df['high'] - df['low']
        atr = df['range'].rolling(10).mean().iloc[-1]
        
        # Comprobar si hay suficiente volumen
        volume_sufficient = current_volume > avg_volume * self.min_volume_threshold
        
        # Señal de ruptura alcista
        bullish_breakout = (
            current_price > upper_breakout and 
            df['close'].iloc[-2] <= upper_breakout and
            volume_sufficient
        )
        
        # Señal de ruptura bajista
        bearish_breakout = (
            current_price < lower_breakout and 
            df['close'].iloc[-2] >= lower_breakout and
            volume_sufficient
        )
        
        # Considerar el libro de órdenes si está disponible
        orderbook_analysis = {}
        if orderbook:
            orderbook_analysis = self.analyze_orderbook(orderbook, current_price)
            
            # Ajustar señales según el libro de órdenes
            if 'imbalance' in orderbook_analysis:
                imbalance = orderbook_analysis['imbalance']
                
                # Fortalecer/debilitar señales según desequilibrio
                if bullish_breakout and imbalance < -0.2:  # Desequilibrio hacia ventas
                    bullish_breakout = False  # Descartar señal alcista
                
                if bearish_breakout and imbalance > 0.2:  # Desequilibrio hacia compras
                    bearish_breakout = False  # Descartar señal bajista
        
        # Calcular niveles de take profit y stop loss
        if bullish_breakout:
            # Estrategia alcista
            entry_price = current_price
            take_profit = entry_price * (1 + self.dynamic_params['take_profit_pct']/100)
            stop_loss = entry_price * (1 - self.dynamic_params['stop_loss_pct']/100)
            
            # Ajustar stop loss según ATR
            dynamic_stop = entry_price - (atr * 1.5)
            stop_loss = max(stop_loss, dynamic_stop)
            
            # Calcular ratio riesgo/recompensa
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < self.min_rr_ratio:
                return {
                    'signal': 'neutral',
                    'message': f'Ratio riesgo/recompensa insuficiente: {rr_ratio:.2f}'
                }
            
            # Calcular tamaño de posición recomendado
            position_size_pct = min(self.max_position_size_pct, rr_ratio)
            
            return {
                'signal': 'buy',
                'strategy': 'breakout_scalping',
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'risk_reward_ratio': rr_ratio,
                'position_size_pct': position_size_pct,
                'orderbook_analysis': orderbook_analysis,
                'confidence': 0.75 if volume_sufficient else 0.6,
                'timeframe': 'ultra_short',
                'timestamp': datetime.now().isoformat()
            }
            
        elif bearish_breakout:
            # Estrategia bajista
            entry_price = current_price
            take_profit = entry_price * (1 - self.dynamic_params['take_profit_pct']/100)
            stop_loss = entry_price * (1 + self.dynamic_params['stop_loss_pct']/100)
            
            # Ajustar stop loss según ATR
            dynamic_stop = entry_price + (atr * 1.5)
            stop_loss = min(stop_loss, dynamic_stop)
            
            # Calcular ratio riesgo/recompensa
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < self.min_rr_ratio:
                return {
                    'signal': 'neutral',
                    'message': f'Ratio riesgo/recompensa insuficiente: {rr_ratio:.2f}'
                }
            
            # Calcular tamaño de posición recomendado
            position_size_pct = min(self.max_position_size_pct, rr_ratio)
            
            return {
                'signal': 'sell',
                'strategy': 'breakout_scalping',
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'risk_reward_ratio': rr_ratio,
                'position_size_pct': position_size_pct,
                'orderbook_analysis': orderbook_analysis,
                'confidence': 0.75 if volume_sufficient else 0.6,
                'timeframe': 'ultra_short',
                'timestamp': datetime.now().isoformat()
            }
        
        # Sin señal clara
        return {
            'signal': 'neutral',
            'strategy': 'breakout_scalping',
            'message': 'No se detectó ruptura con suficiente volumen',
            'orderbook_analysis': orderbook_analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    def momentum_scalping_strategy(self, 
                                 df: pd.DataFrame,
                                 fast_period: int = 5,
                                 slow_period: int = 8,
                                 rsi_period: int = 7) -> Dict[str, Any]:
        """
        Estrategia de scalping basada en momento de precio.
        
        Args:
            df: DataFrame con datos OHLCV recientes
            fast_period: Período para media móvil rápida
            slow_period: Período para media móvil lenta
            rsi_period: Período para cálculo de RSI
            
        Returns:
            Dict[str, Any]: Señal de trading con parámetros
        """
        if len(df) < max(fast_period, slow_period, rsi_period) + 10:
            return {'signal': 'neutral', 'message': 'Datos insuficientes'}
        
        # Obtener precio actual y volumen
        current_price = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(10).mean().iloc[-1]
        
        # Calcular medias móviles rápidas (EMAs)
        df['fast_ema'] = df['close'].ewm(span=fast_period, adjust=False).mean()
        df['slow_ema'] = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calcular RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        current_rsi = df['rsi'].iloc[-1]
        
        # Calcular momento del precio
        df['momentum'] = df['close'].pct_change(3)
        current_momentum = df['momentum'].iloc[-1] * 100  # en porcentaje
        
        # Comprobar cruce de medias móviles
        current_fast = df['fast_ema'].iloc[-1]
        current_slow = df['slow_ema'].iloc[-1]
        prev_fast = df['fast_ema'].iloc[-2]
        prev_slow = df['slow_ema'].iloc[-2]
        
        # Señal de cruce alcista (golden cross)
        bullish_cross = current_fast > current_slow and prev_fast <= prev_slow
        
        # Señal de cruce bajista (death cross)
        bearish_cross = current_fast < current_slow and prev_fast >= prev_slow
        
        # Comprobar confirmación de volumen
        volume_confirmed = current_volume > avg_volume * 1.2
        
        # Umbral de momento (ajustable)
        momentum_threshold = 0.5  # 0.5% en 3 velas
        
        # Confirmar señales con RSI y momento
        strong_bullish = (
            bullish_cross and 
            current_rsi > 50 and 
            current_momentum > momentum_threshold and
            volume_confirmed
        )
        
        strong_bearish = (
            bearish_cross and 
            current_rsi < 50 and 
            current_momentum < -momentum_threshold and
            volume_confirmed
        )
        
        # Cálculo de ATR para stop loss dinámico
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        current_atr = df['atr'].iloc[-1]
        
        # Generar señal de trading
        if strong_bullish:
            # Estrategia alcista
            entry_price = current_price
            take_profit = entry_price * (1 + self.dynamic_params['take_profit_pct']/100)
            
            # Stop loss dinámico basado en ATR (1.5x ATR)
            stop_loss = entry_price - (current_atr * 1.5)
            
            # Asegurar mínimo de stop loss
            min_stop = entry_price * (1 - self.dynamic_params['stop_loss_pct']/100)
            stop_loss = max(stop_loss, min_stop)
            
            # Calcular ratio riesgo/recompensa
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < self.min_rr_ratio:
                return {
                    'signal': 'neutral',
                    'message': f'Ratio riesgo/recompensa insuficiente: {rr_ratio:.2f}'
                }
            
            # Calcular tamaño de posición recomendado
            position_size_pct = min(self.max_position_size_pct, rr_ratio * 0.7)
            
            return {
                'signal': 'buy',
                'strategy': 'momentum_scalping',
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'risk_reward_ratio': rr_ratio,
                'position_size_pct': position_size_pct,
                'rsi': current_rsi,
                'momentum': current_momentum,
                'confidence': 0.8 if volume_confirmed and rr_ratio > 2 else 0.65,
                'timeframe': 'ultra_short',
                'timestamp': datetime.now().isoformat()
            }
            
        elif strong_bearish:
            # Estrategia bajista
            entry_price = current_price
            take_profit = entry_price * (1 - self.dynamic_params['take_profit_pct']/100)
            
            # Stop loss dinámico basado en ATR (1.5x ATR)
            stop_loss = entry_price + (current_atr * 1.5)
            
            # Asegurar máximo de stop loss
            max_stop = entry_price * (1 + self.dynamic_params['stop_loss_pct']/100)
            stop_loss = min(stop_loss, max_stop)
            
            # Calcular ratio riesgo/recompensa
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < self.min_rr_ratio:
                return {
                    'signal': 'neutral',
                    'message': f'Ratio riesgo/recompensa insuficiente: {rr_ratio:.2f}'
                }
            
            # Calcular tamaño de posición recomendado
            position_size_pct = min(self.max_position_size_pct, rr_ratio * 0.7)
            
            return {
                'signal': 'sell',
                'strategy': 'momentum_scalping',
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'risk_reward_ratio': rr_ratio,
                'position_size_pct': position_size_pct,
                'rsi': current_rsi,
                'momentum': current_momentum,
                'confidence': 0.8 if volume_confirmed and rr_ratio > 2 else 0.65,
                'timeframe': 'ultra_short',
                'timestamp': datetime.now().isoformat()
            }
        
        # Sin señal clara
        return {
            'signal': 'neutral',
            'strategy': 'momentum_scalping',
            'message': 'No se detectó momento de precio convincente',
            'rsi': current_rsi,
            'momentum': current_momentum,
            'timestamp': datetime.now().isoformat()
        }
    
    def mean_reversion_scalping(self, 
                              df: pd.DataFrame,
                              bb_period: int = 20,
                              bb_std: float = 2.0) -> Dict[str, Any]:
        """
        Estrategia de scalping basada en reversión a la media.
        
        Args:
            df: DataFrame con datos OHLCV recientes
            bb_period: Período para bandas de Bollinger
            bb_std: Número de desviaciones estándar para bandas
            
        Returns:
            Dict[str, Any]: Señal de trading con parámetros
        """
        if len(df) < bb_period + 10:
            return {'signal': 'neutral', 'message': 'Datos insuficientes'}
        
        # Obtener precio actual
        current_price = df['close'].iloc[-1]
        
        # Calcular bandas de Bollinger
        df['sma'] = df['close'].rolling(window=bb_period).mean()
        df['std'] = df['close'].rolling(window=bb_period).std()
        df['upper_band'] = df['sma'] + (df['std'] * bb_std)
        df['lower_band'] = df['sma'] - (df['std'] * bb_std)
        df['bandwidth'] = (df['upper_band'] - df['lower_band']) / df['sma'] * 100
        
        current_upper = df['upper_band'].iloc[-1]
        current_lower = df['lower_band'].iloc[-1]
        current_sma = df['sma'].iloc[-1]
        current_bandwidth = df['bandwidth'].iloc[-1]
        avg_bandwidth = df['bandwidth'].rolling(10).mean().iloc[-1]
        
        # Calcular RSI para confirmar sobrecompra/sobreventa
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        current_rsi = df['rsi'].iloc[-1]
        
        # Verificar condiciones de compra/venta
        oversold = current_price <= current_lower and current_rsi < 30
        overbought = current_price >= current_upper and current_rsi > 70
        
        # Verificar estrechamiento de bandas (baja volatilidad)
        bands_squeezing = current_bandwidth < avg_bandwidth * 0.8
        
        # Calcular ATR para stop loss dinámico
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        current_atr = df['atr'].iloc[-1]
        
        # Generar señal de trading
        if oversold and not bands_squeezing:
            # Estrategia alcista (rebote desde sobreventa)
            entry_price = current_price
            take_profit = current_sma  # Objetivo: media móvil
            
            # Stop loss dinámico basado en ATR (1x ATR)
            stop_loss = entry_price - (current_atr * 1.0)
            
            # Asegurar mínimo de stop loss
            min_stop = entry_price * (1 - self.dynamic_params['stop_loss_pct']/100)
            stop_loss = max(stop_loss, min_stop)
            
            # Calcular ratio riesgo/recompensa
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < self.min_rr_ratio:
                return {
                    'signal': 'neutral',
                    'message': f'Ratio riesgo/recompensa insuficiente: {rr_ratio:.2f}'
                }
            
            # Calcular tamaño de posición recomendado
            position_size_pct = min(self.max_position_size_pct, rr_ratio * 0.5)
            
            return {
                'signal': 'buy',
                'strategy': 'mean_reversion',
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'risk_reward_ratio': rr_ratio,
                'position_size_pct': position_size_pct,
                'rsi': current_rsi,
                'bb_bandwidth': current_bandwidth,
                'confidence': 0.7 if current_rsi < 25 else 0.6,
                'timeframe': 'ultra_short',
                'timestamp': datetime.now().isoformat()
            }
            
        elif overbought and not bands_squeezing:
            # Estrategia bajista (caída desde sobrecompra)
            entry_price = current_price
            take_profit = current_sma  # Objetivo: media móvil
            
            # Stop loss dinámico basado en ATR (1x ATR)
            stop_loss = entry_price + (current_atr * 1.0)
            
            # Asegurar máximo de stop loss
            max_stop = entry_price * (1 + self.dynamic_params['stop_loss_pct']/100)
            stop_loss = min(stop_loss, max_stop)
            
            # Calcular ratio riesgo/recompensa
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < self.min_rr_ratio:
                return {
                    'signal': 'neutral',
                    'message': f'Ratio riesgo/recompensa insuficiente: {rr_ratio:.2f}'
                }
            
            # Calcular tamaño de posición recomendado
            position_size_pct = min(self.max_position_size_pct, rr_ratio * 0.5)
            
            return {
                'signal': 'sell',
                'strategy': 'mean_reversion',
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'risk_reward_ratio': rr_ratio,
                'position_size_pct': position_size_pct,
                'rsi': current_rsi,
                'bb_bandwidth': current_bandwidth,
                'confidence': 0.7 if current_rsi > 75 else 0.6,
                'timeframe': 'ultra_short',
                'timestamp': datetime.now().isoformat()
            }
        
        # Sin señal clara
        return {
            'signal': 'neutral',
            'strategy': 'mean_reversion',
            'message': 'No se detectaron condiciones de sobreventa/sobrecompra',
            'rsi': current_rsi,
            'bb_bandwidth': current_bandwidth,
            'timestamp': datetime.now().isoformat()
        }
    
    def arbitrage_opportunity(self, 
                            prices: Dict[str, float],
                            fees: Dict[str, float],
                            min_spread_pct: float = 0.5) -> Dict[str, Any]:
        """
        Detecta oportunidades de arbitraje entre exchanges.
        
        Args:
            prices: Diccionario de precios por exchange
            fees: Diccionario de comisiones por exchange
            min_spread_pct: Spread mínimo para considerar arbitraje
            
        Returns:
            Dict[str, Any]: Oportunidad de arbitraje si existe
        """
        if len(prices) < 2:
            return {'arbitrage': False, 'message': 'Se necesitan al menos 2 exchanges'}
        
        # Encontrar el precio más bajo para comprar
        buy_exchange = min(prices, key=prices.get)
        buy_price = prices[buy_exchange]
        buy_fee = fees.get(buy_exchange, 0.1) / 100  # 0.1% por defecto
        
        # Encontrar el precio más alto para vender
        sell_exchange = max(prices, key=prices.get)
        sell_price = prices[sell_exchange]
        sell_fee = fees.get(sell_exchange, 0.1) / 100  # 0.1% por defecto
        
        # Calcular spread y ganancia potencial
        spread_pct = (sell_price - buy_price) / buy_price * 100
        
        # Calcular impacto de comisiones
        fee_impact = (buy_price * buy_fee) + (sell_price * sell_fee)
        fee_impact_pct = fee_impact / buy_price * 100
        
        # Ganancia neta después de comisiones
        net_profit_pct = spread_pct - fee_impact_pct
        
        if net_profit_pct > min_spread_pct:
            # Oportunidad de arbitraje viable
            return {
                'arbitrage': True,
                'buy_exchange': buy_exchange,
                'buy_price': buy_price,
                'sell_exchange': sell_exchange,
                'sell_price': sell_price,
                'spread_pct': spread_pct,
                'fee_impact_pct': fee_impact_pct,
                'net_profit_pct': net_profit_pct,
                'recommended': True if net_profit_pct > min_spread_pct * 1.5 else False,
                'timestamp': datetime.now().isoformat()
            }
        
        return {
            'arbitrage': False,
            'message': f'Spread insuficiente: {net_profit_pct:.2f}% (min: {min_spread_pct}%)',
            'timestamp': datetime.now().isoformat()
        }
    
    def scalping_dashboard(self, timeframe: str = '1m') -> Dict[str, Any]:
        """
        Retorna un panel completo de estrategias de scalping y sus señales.
        
        Args:
            timeframe: Intervalo temporal para los datos
            
        Returns:
            Dict[str, Any]: Panel completo de estrategias de scalping
        """
        from data_management.market_data import get_market_data
        
        # Obtener datos de mercado recientes
        df = get_market_data("SOL-USDT", timeframe)
        if df is None or len(df) < 50:
            return {
                'error': 'Datos de mercado insuficientes',
                'timestamp': datetime.now().isoformat()
            }
        
        # Ejecutar todas las estrategias de scalping
        momentum_signal = self.momentum_scalping_strategy(df)
        breakout_signal = self.breakout_scalping_strategy(df)
        reversion_signal = self.mean_reversion_scalping(df)
        
        # Analizar y combinar señales
        signals = [
            momentum_signal,
            breakout_signal,
            reversion_signal
        ]
        
        buy_signals = [s for s in signals if s.get('signal') == 'buy']
        sell_signals = [s for s in signals if s.get('signal') == 'sell']
        neutral_signals = [s for s in signals if s.get('signal') == 'neutral']
        
        # Determinar señal combinada
        combined_signal = 'neutral'
        confidence = 0.0
        recommended_strategy = None
        
        if buy_signals and len(buy_signals) > len(sell_signals):
            # Señal alcista dominante
            best_signal = max(buy_signals, key=lambda x: x.get('confidence', 0))
            combined_signal = 'buy'
            confidence = best_signal.get('confidence', 0)
            recommended_strategy = best_signal.get('strategy')
            
        elif sell_signals and len(sell_signals) > len(buy_signals):
            # Señal bajista dominante
            best_signal = max(sell_signals, key=lambda x: x.get('confidence', 0))
            combined_signal = 'sell'
            confidence = best_signal.get('confidence', 0)
            recommended_strategy = best_signal.get('strategy')
        
        # Estadísticas de mercado actuales
        market_stats = {
            'price': df['close'].iloc[-1],
            'volume': df['volume'].iloc[-1],
            'avg_volume': df['volume'].rolling(10).mean().iloc[-1],
            'last_updated': datetime.now().isoformat()
        }
        
        return {
            'combined_signal': combined_signal,
            'confidence': confidence,
            'recommended_strategy': recommended_strategy,
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'neutral_signals': len(neutral_signals),
            'strategies': {
                'momentum': momentum_signal,
                'breakout': breakout_signal,
                'mean_reversion': reversion_signal
            },
            'market_stats': market_stats,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat()
        }

def demo_scalping_strategies():
    """Demostración de estrategias de scalping para Solana."""
    print("\n🔍 ESTRATEGIAS DE SCALPING PARA SOLANA 🔍")
    print("Estas estrategias están optimizadas para operaciones de")
    print("alta frecuencia en timeframes muy cortos (1m, 5m).")
    
    # Crear instancia de estrategias de scalping
    scalper = ScalpingStrategies(
        max_position_size_pct=2.0,
        take_profit_pct=0.5,
        stop_loss_pct=0.3,
        min_rr_ratio=1.5
    )
    
    # Simular datos de mercado para Solana
    dates = pd.date_range(start=datetime.now() - timedelta(days=1), 
                         periods=60, freq='1min')
    
    # Crear DataFrame simulado
    np.random.seed(42)  # Para reproducibilidad
    base_price = 150.0
    volatility = 0.001
    
    # Generar precios aleatorios con tendencia
    closes = [base_price]
    for i in range(1, 60):
        # Añadir tendencia ligeramente alcista
        close = closes[-1] * (1 + np.random.normal(0.0001, volatility))
        closes.append(close)
    
    # Generar OHLC basado en los precios de cierre
    highs = [c * (1 + abs(np.random.normal(0, volatility))) for c in closes]
    lows = [c * (1 - abs(np.random.normal(0, volatility))) for c in closes]
    opens = [lows[i] + (highs[i] - lows[i]) * np.random.random() for i in range(60)]
    
    # Generar volumen simulado
    volumes = [np.random.normal(1000, 200) for _ in range(60)]
    
    # Crear DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)
    
    # Ejecutar estrategias
    momentum_result = scalper.momentum_scalping_strategy(df)
    breakout_result = scalper.breakout_scalping_strategy(df)
    reversion_result = scalper.mean_reversion_scalping(df)
    
    # Mostrar resultados
    print("\n1. Estrategia de Scalping por Momento")
    print(f"   Señal: {momentum_result['signal']}")
    if momentum_result['signal'] != 'neutral':
        print(f"   Entrada: ${momentum_result['entry_price']:.2f}")
        print(f"   Take Profit: ${momentum_result['take_profit']:.2f}")
        print(f"   Stop Loss: ${momentum_result['stop_loss']:.2f}")
        print(f"   Ratio Riesgo/Recompensa: {momentum_result['risk_reward_ratio']:.2f}")
        print(f"   Confianza: {momentum_result['confidence']:.2f}")
    else:
        print(f"   Mensaje: {momentum_result.get('message', 'Sin señal')}")
    
    print("\n2. Estrategia de Scalping por Ruptura")
    print(f"   Señal: {breakout_result['signal']}")
    if breakout_result['signal'] != 'neutral':
        print(f"   Entrada: ${breakout_result['entry_price']:.2f}")
        print(f"   Take Profit: ${breakout_result['take_profit']:.2f}")
        print(f"   Stop Loss: ${breakout_result['stop_loss']:.2f}")
        print(f"   Ratio Riesgo/Recompensa: {breakout_result['risk_reward_ratio']:.2f}")
        print(f"   Confianza: {breakout_result['confidence']:.2f}")
    else:
        print(f"   Mensaje: {breakout_result.get('message', 'Sin señal')}")
    
    print("\n3. Estrategia de Scalping por Reversión a la Media")
    print(f"   Señal: {reversion_result['signal']}")
    if reversion_result['signal'] != 'neutral':
        print(f"   Entrada: ${reversion_result['entry_price']:.2f}")
        print(f"   Take Profit: ${reversion_result['take_profit']:.2f}")
        print(f"   Stop Loss: ${reversion_result['stop_loss']:.2f}")
        print(f"   Ratio Riesgo/Recompensa: {reversion_result['risk_reward_ratio']:.2f}")
        print(f"   Confianza: {reversion_result['confidence']:.2f}")
    else:
        print(f"   Mensaje: {reversion_result.get('message', 'Sin señal')}")
    
    # Simular análisis de libro de órdenes
    orderbook_analysis = {
        'buy_sell_ratio': 1.2,
        'imbalance': 0.15,
        'bid_volume': 15000,
        'ask_volume': 12500,
        'nearest_support': 149.5,
        'nearest_resistance': 150.8,
        'support_distance_pct': 0.33,
        'resistance_distance_pct': 0.53,
        'buy_slippage_pct': 0.02,
        'sell_slippage_pct': 0.03
    }
    
    print("\n4. Análisis de Libro de Órdenes")
    print(f"   Ratio compra/venta: {orderbook_analysis['buy_sell_ratio']:.2f}")
    print(f"   Desequilibrio: {orderbook_analysis['imbalance']:.2f}")
    print(f"   Soporte más cercano: ${orderbook_analysis['nearest_support']:.2f} ({orderbook_analysis['support_distance_pct']:.2f}%)")
    print(f"   Resistencia más cercana: ${orderbook_analysis['nearest_resistance']:.2f} ({orderbook_analysis['resistance_distance_pct']:.2f}%)")
    
    # Simular arbitraje
    prices = {
        'Binance': 150.25,
        'OKX': 150.10,
        'Bybit': 150.40
    }
    
    fees = {
        'Binance': 0.075,  # 0.075%
        'OKX': 0.08,       # 0.08%
        'Bybit': 0.06      # 0.06%
    }
    
    arb_result = scalper.arbitrage_opportunity(prices, fees, min_spread_pct=0.15)
    
    print("\n5. Oportunidad de Arbitraje")
    if arb_result['arbitrage']:
        print(f"   ✅ Arbitraje detectado:")
        print(f"   Comprar en: {arb_result['buy_exchange']} a ${arb_result['buy_price']:.2f}")
        print(f"   Vender en: {arb_result['sell_exchange']} a ${arb_result['sell_price']:.2f}")
        print(f"   Spread: {arb_result['spread_pct']:.2f}%")
        print(f"   Impacto de comisiones: {arb_result['fee_impact_pct']:.2f}%")
        print(f"   Ganancia neta: {arb_result['net_profit_pct']:.2f}%")
        print(f"   Recomendado: {'Sí' if arb_result['recommended'] else 'No'}")
    else:
        print(f"   ❌ {arb_result['message']}")
    
    print("\n✅ Demostración de estrategias de scalping completada.")
    return True

if __name__ == "__main__":
    try:
        demo_scalping_strategies()
    except Exception as e:
        print(f"Error en la demostración: {e}")