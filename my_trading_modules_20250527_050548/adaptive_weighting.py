"""
Módulo de ponderación adaptativa de indicadores según su rendimiento histórico

Este sistema implementa un mecanismo de aprendizaje que ajusta dinámicamente
el peso de cada indicador técnico basándose en su precisión histórica en
diferentes condiciones de mercado para Solana.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Any, Tuple, Optional

# Configurar logging
logger = logging.getLogger(__name__)

class MarketCondition(Enum):
    """Condiciones de mercado para contextualizar señales"""
    STRONG_UPTREND = "strong_uptrend"          # Tendencia alcista fuerte
    MODERATE_UPTREND = "moderate_uptrend"      # Tendencia alcista moderada
    LATERAL_LOW_VOL = "lateral_low_vol"        # Mercado lateral con baja volatilidad
    LATERAL_HIGH_VOL = "lateral_high_vol"      # Mercado lateral con alta volatilidad
    MODERATE_DOWNTREND = "moderate_downtrend"  # Tendencia bajista moderada
    STRONG_DOWNTREND = "strong_downtrend"      # Tendencia bajista fuerte
    EXTREME_VOLATILITY = "extreme_volatility"  # Volatilidad extrema

class TimeInterval(Enum):
    """Intervalos de tiempo para contextualizar señales"""
    MINUTE_1 = "1m"       # 1 minuto
    MINUTE_5 = "5m"       # 5 minutos
    MINUTE_15 = "15m"     # 15 minutos
    MINUTE_30 = "30m"     # 30 minutos
    HOUR_1 = "1h"         # 1 hora
    HOUR_4 = "4h"         # 4 horas
    DAY_1 = "1d"          # 1 día

class IndicatorPerformance:
    """Clase para almacenar y actualizar el rendimiento de los indicadores"""
    
    def __init__(self, indicator_name: str):
        """
        Inicializa el registro de rendimiento para un indicador
        
        Args:
            indicator_name: Nombre del indicador
        """
        self.name = indicator_name
        self.signals_count = 0
        self.correct_signals = 0
        self.incorrect_signals = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        
        # Rendimiento por condición de mercado
        self.market_conditions = {condition.value: {
            'signals': 0,
            'correct': 0,
            'incorrect': 0,
            'accuracy': 0.0,
            'profit': 0.0
        } for condition in MarketCondition}
        
        # Rendimiento por intervalo de tiempo
        self.time_intervals = {interval.value: {
            'signals': 0,
            'correct': 0,
            'incorrect': 0,
            'accuracy': 0.0,
            'profit': 0.0
        } for interval in TimeInterval}
        
        # Lista de señales recientes (últimas 20)
        self.recent_signals = []
    
    def update(self, correct: bool, profit: float, 
              market_condition: MarketCondition, 
              time_interval: TimeInterval):
        """
        Actualiza el rendimiento con una nueva señal
        
        Args:
            correct: Si la señal fue correcta
            profit: Ganancia/pérdida de la operación
            market_condition: Condición de mercado actual
            time_interval: Intervalo de tiempo usado
        """
        # Actualizar contadores generales
        self.signals_count += 1
        
        if correct:
            self.correct_signals += 1
            self.total_profit += profit
        else:
            self.incorrect_signals += 1
            self.total_loss += abs(profit)
        
        # Actualizar rendimiento por condición de mercado
        condition = market_condition.value
        
        self.market_conditions[condition]['signals'] += 1
        
        if correct:
            self.market_conditions[condition]['correct'] += 1
            self.market_conditions[condition]['profit'] += profit
        else:
            self.market_conditions[condition]['incorrect'] += 1
            self.market_conditions[condition]['profit'] -= abs(profit)
        
        # Actualizar accuracy por condición
        signals = self.market_conditions[condition]['signals']
        correct = self.market_conditions[condition]['correct']
        
        self.market_conditions[condition]['accuracy'] = correct / signals if signals > 0 else 0.0
        
        # Actualizar rendimiento por intervalo de tiempo
        interval = time_interval.value
        
        self.time_intervals[interval]['signals'] += 1
        
        if correct:
            self.time_intervals[interval]['correct'] += 1
            self.time_intervals[interval]['profit'] += profit
        else:
            self.time_intervals[interval]['incorrect'] += 1
            self.time_intervals[interval]['profit'] -= abs(profit)
        
        # Actualizar accuracy por intervalo
        signals = self.time_intervals[interval]['signals']
        correct = self.time_intervals[interval]['correct']
        
        self.time_intervals[interval]['accuracy'] = correct / signals if signals > 0 else 0.0
        
        # Actualizar señales recientes
        self.recent_signals.append({
            'correct': correct,
            'profit': profit,
            'market_condition': condition,
            'time_interval': interval
        })
        
        # Mantener solo las últimas 20 señales
        if len(self.recent_signals) > 20:
            self.recent_signals = self.recent_signals[-20:]
    
    def get_accuracy(self) -> float:
        """
        Obtiene la tasa de acierto global
        
        Returns:
            float: Tasa de acierto (0-1)
        """
        if self.signals_count == 0:
            return 0.0
        
        return self.correct_signals / self.signals_count
    
    def get_profit_factor(self) -> float:
        """
        Obtiene el factor de rentabilidad (profit factor)
        
        Returns:
            float: Factor de rentabilidad (ganancias/pérdidas)
        """
        if self.total_loss == 0:
            return float('inf') if self.total_profit > 0 else 0.0
        
        return self.total_profit / self.total_loss
    
    def get_market_condition_accuracy(self, condition: MarketCondition) -> float:
        """
        Obtiene la tasa de acierto para una condición de mercado específica
        
        Args:
            condition: Condición de mercado
            
        Returns:
            float: Tasa de acierto para esa condición
        """
        return self.market_conditions[condition.value]['accuracy']
    
    def get_time_interval_accuracy(self, interval: TimeInterval) -> float:
        """
        Obtiene la tasa de acierto para un intervalo de tiempo específico
        
        Args:
            interval: Intervalo de tiempo
            
        Returns:
            float: Tasa de acierto para ese intervalo
        """
        return self.time_intervals[interval.value]['accuracy']
    
    def get_recent_accuracy(self) -> float:
        """
        Obtiene la tasa de acierto de las señales recientes
        
        Returns:
            float: Tasa de acierto reciente
        """
        if not self.recent_signals:
            return 0.0
        
        correct_count = sum(1 for signal in self.recent_signals if signal['correct'])
        return correct_count / len(self.recent_signals)
    
    def to_dict(self) -> Dict:
        """
        Convierte el objeto a un diccionario para serialización
        
        Returns:
            Dict: Representación en diccionario
        """
        return {
            'name': self.name,
            'signals_count': self.signals_count,
            'correct_signals': self.correct_signals,
            'incorrect_signals': self.incorrect_signals,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'market_conditions': self.market_conditions,
            'time_intervals': self.time_intervals,
            'recent_signals': self.recent_signals
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IndicatorPerformance':
        """
        Crea un objeto desde un diccionario
        
        Args:
            data: Diccionario con datos
            
        Returns:
            IndicatorPerformance: Objeto reconstruido
        """
        perf = cls(data['name'])
        
        perf.signals_count = data['signals_count']
        perf.correct_signals = data['correct_signals']
        perf.incorrect_signals = data['incorrect_signals']
        perf.total_profit = data['total_profit']
        perf.total_loss = data['total_loss']
        perf.market_conditions = data['market_conditions']
        perf.time_intervals = data['time_intervals']
        perf.recent_signals = data['recent_signals']
        
        return perf

class AdaptiveWeightingSystem:
    """Sistema de ponderación adaptativa para indicadores técnicos"""
    
    def __init__(self, data_file: str = "indicator_performance.json"):
        """
        Inicializa el sistema de ponderación
        
        Args:
            data_file: Archivo para guardar/cargar datos de rendimiento
        """
        self.data_file = data_file
        self.indicators = {}
        self.base_weights = {}
        
        # Cargar datos si existen
        self._load_data()
        
        # Inicializar indicadores básicos si no hay datos
        if not self.indicators:
            self._initialize_indicators()
    
    def _load_data(self):
        """Carga datos de rendimiento desde archivo"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Cargar indicadores
                for name, indicator_data in data.get('indicators', {}).items():
                    self.indicators[name] = IndicatorPerformance.from_dict(indicator_data)
                
                # Cargar pesos base
                self.base_weights = data.get('base_weights', {})
                
                logger.info(f"Datos de rendimiento cargados desde {self.data_file}")
                
            except Exception as e:
                logger.error(f"Error al cargar datos: {e}")
                self._initialize_indicators()
        else:
            self._initialize_indicators()
    
    def _initialize_indicators(self):
        """Inicializa los indicadores con valores por defecto"""
        # Indicadores comunes
        default_indicators = [
            'RSI', 'MACD', 'Bollinger', 'EMA_Cross', 'VWAP',
            'Ichimoku', 'Stochastic', 'ADX', 'OBV', 'ATR',
            'Fractal', 'Support_Resistance', 'Pivot', 'Fibonacci',
            'Elliott_Wave'
        ]
        
        # Crear objetos de rendimiento
        for name in default_indicators:
            self.indicators[name] = IndicatorPerformance(name)
        
        # Pesos iniciales (equiponderación)
        weight = 1.0 / len(default_indicators)
        self.base_weights = {name: weight for name in default_indicators}
        
        logger.info("Indicadores inicializados con pesos equiponderados")
    
    def _save_data(self):
        """Guarda datos de rendimiento en archivo"""
        try:
            # Preparar datos para guardar
            indicators_data = {}
            
            for name, indicator in self.indicators.items():
                indicators_data[name] = indicator.to_dict()
            
            data = {
                'indicators': indicators_data,
                'base_weights': self.base_weights
            }
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.data_file) or '.', exist_ok=True)
            
            # Guardar datos
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=4)
            
            logger.info(f"Datos de rendimiento guardados en {self.data_file}")
            
        except Exception as e:
            logger.error(f"Error al guardar datos: {e}")
    
    def update_indicator_performance(self, indicator_name: str, correct: bool, 
                                    profit: float, market_condition: MarketCondition,
                                    time_interval: TimeInterval):
        """
        Actualiza el rendimiento de un indicador
        
        Args:
            indicator_name: Nombre del indicador
            correct: Si la señal fue correcta
            profit: Ganancia/pérdida de la operación
            market_condition: Condición de mercado actual
            time_interval: Intervalo de tiempo usado
        """
        # Asegurarse de que el indicador existe
        if indicator_name not in self.indicators:
            self.indicators[indicator_name] = IndicatorPerformance(indicator_name)
            
            # Añadir peso base (promedio de los existentes)
            if self.base_weights:
                avg_weight = sum(self.base_weights.values()) / len(self.base_weights)
                self.base_weights[indicator_name] = avg_weight
            else:
                self.base_weights[indicator_name] = 1.0
        
        # Actualizar rendimiento
        self.indicators[indicator_name].update(correct, profit, market_condition, time_interval)
        
        # Recalibrar pesos
        self._recalibrate_weights()
        
        # Guardar datos
        self._save_data()
        
        logger.info(f"Actualizado rendimiento de indicador: {indicator_name} (correcto: {correct}, profit: {profit:.2f})")
    
    def _recalibrate_weights(self):
        """Recalibra los pesos de los indicadores basándose en su rendimiento"""
        # Verificar que hay indicadores con señales
        active_indicators = {name: ind for name, ind in self.indicators.items() if ind.signals_count > 0}
        
        if not active_indicators:
            logger.info("No hay indicadores activos para recalibrar pesos")
            return
        
        # Calcular nuevos pesos basados en accuracy y profit factor
        new_weights = {}
        
        for name, indicator in active_indicators.items():
            # Combinar accuracy y profit factor (con más peso en accuracy)
            accuracy = indicator.get_accuracy()
            profit_factor = min(indicator.get_profit_factor(), 10.0)  # Cap profit factor
            
            # Combinar métricas (70% accuracy, 30% profit factor)
            combined_score = 0.7 * accuracy + 0.3 * (profit_factor / 10.0)
            
            # Añadir un factor para señales recientes (para adaptación rápida)
            recent_accuracy = indicator.get_recent_accuracy()
            recency_factor = 1.2 if recent_accuracy > accuracy else 0.8
            
            # Calcular peso final
            new_weights[name] = combined_score * recency_factor
        
        # Normalizar pesos
        total_weight = sum(new_weights.values())
        
        if total_weight > 0:
            for name in new_weights:
                new_weights[name] /= total_weight
            
            # Actualizar pesos base gradualmente (mezcla 70% nuevos, 30% antiguos)
            for name, weight in new_weights.items():
                if name in self.base_weights:
                    self.base_weights[name] = 0.7 * weight + 0.3 * self.base_weights[name]
                else:
                    self.base_weights[name] = weight
            
            # Normalizar pesos base
            total_base = sum(self.base_weights.values())
            for name in self.base_weights:
                self.base_weights[name] /= total_base
            
            logger.info("Pesos recalibrados basados en rendimiento")
    
    def get_indicator_weight(self, indicator_name: str, market_condition: MarketCondition, 
                           time_interval: TimeInterval) -> float:
        """
        Obtiene el peso adaptado de un indicador para condiciones específicas
        
        Args:
            indicator_name: Nombre del indicador
            market_condition: Condición actual del mercado
            time_interval: Intervalo de tiempo actual
            
        Returns:
            float: Peso ajustado del indicador
        """
        # Verificar que el indicador existe
        if indicator_name not in self.indicators:
            logger.warning(f"Indicador no encontrado: {indicator_name}")
            return 0.0
        
        indicator = self.indicators[indicator_name]
        
        # Obtener peso base
        base_weight = self.base_weights.get(indicator_name, 0.0)
        
        # Sin suficientes datos para ajustar, usar peso base
        if indicator.signals_count < 10:
            return base_weight
        
        # Ajustar según rendimiento en esta condición específica
        condition_accuracy = indicator.get_market_condition_accuracy(market_condition)
        interval_accuracy = indicator.get_time_interval_accuracy(time_interval)
        
        # Combinar factores de ajuste
        condition_factor = 1.5 if condition_accuracy > indicator.get_accuracy() else 0.7
        interval_factor = 1.3 if interval_accuracy > indicator.get_accuracy() else 0.8
        
        adjusted_weight = base_weight * condition_factor * interval_factor
        
        return adjusted_weight
    
    def get_all_weights(self, market_condition: MarketCondition, 
                      time_interval: TimeInterval) -> Dict[str, float]:
        """
        Obtiene todos los pesos ajustados para las condiciones actuales
        
        Args:
            market_condition: Condición actual del mercado
            time_interval: Intervalo de tiempo actual
            
        Returns:
            Dict[str, float]: Diccionario de pesos ajustados por indicador
        """
        weights = {}
        
        for name in self.indicators:
            weights[name] = self.get_indicator_weight(name, market_condition, time_interval)
        
        # Normalizar pesos
        total_weight = sum(weights.values())
        
        if total_weight > 0:
            for name in weights:
                weights[name] /= total_weight
        
        return weights
    
    def detect_market_condition(self, df: pd.DataFrame) -> MarketCondition:
        """
        Detecta la condición actual del mercado basándose en indicadores
        
        Args:
            df: DataFrame con datos de mercado (OHLCV)
            
        Returns:
            MarketCondition: Condición de mercado detectada
        """
        # Verificar que hay suficientes datos
        if len(df) < 20:
            return MarketCondition.LATERAL_LOW_VOL
        
        # Calcular indicadores básicos para detección
        
        # 1. Tendencia usando medias móviles
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        
        # 2. Volatilidad usando ATR
        true_range = pd.DataFrame()
        true_range['hl'] = df['high'] - df['low']
        true_range['hc'] = abs(df['high'] - df['close'].shift(1))
        true_range['lc'] = abs(df['low'] - df['close'].shift(1))
        df['atr14'] = true_range.max(axis=1).rolling(window=14).mean()
        
        # Obtener valores más recientes
        latest_sma20 = df['sma20'].iloc[-1]
        latest_sma50 = df['sma50'].iloc[-1]
        avg_price = df['close'].iloc[-20:].mean()
        latest_atr = df['atr14'].iloc[-1]
        avg_atr = df['atr14'].iloc[-20:].mean()
        
        # Calcular indicadores derivados
        sma_diff = (latest_sma20 - latest_sma50) / latest_sma50
        volatility_ratio = latest_atr / avg_price
        volatility_change = latest_atr / avg_atr
        
        # Decisión de condición de mercado
        
        # Comprobar volatilidad extrema primero
        if volatility_ratio > 0.03 or volatility_change > 2.0:
            return MarketCondition.EXTREME_VOLATILITY
        
        # Comprobar tendencias
        if sma_diff > 0.02:  # Tendencia alcista fuerte
            return MarketCondition.STRONG_UPTREND
        elif sma_diff > 0.005:  # Tendencia alcista moderada
            return MarketCondition.MODERATE_UPTREND
        elif sma_diff < -0.02:  # Tendencia bajista fuerte
            return MarketCondition.STRONG_DOWNTREND
        elif sma_diff < -0.005:  # Tendencia bajista moderada
            return MarketCondition.MODERATE_DOWNTREND
        
        # Si llegamos aquí, estamos en mercado lateral
        if volatility_ratio > 0.015:
            return MarketCondition.LATERAL_HIGH_VOL
        else:
            return MarketCondition.LATERAL_LOW_VOL
    
    def get_example_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Obtiene ejemplos de ponderaciones para diferentes condiciones de mercado
        
        Returns:
            Dict: Ejemplos de pesos por condición de mercado
        """
        example_weights = {}
        
        for condition in MarketCondition:
            # Usar intervalos de 1m para ejemplos
            example_weights[condition.value] = self.get_all_weights(
                condition, TimeInterval.MINUTE_1
            )
        
        return example_weights
    
    def calculate_weighted_signal(self, signals: Dict[str, float], 
                                 market_condition: MarketCondition,
                                 time_interval: TimeInterval) -> float:
        """
        Calcula la señal ponderada combinando múltiples indicadores
        
        Args:
            signals: Diccionario de señales por indicador (-1 a +1)
            market_condition: Condición actual del mercado
            time_interval: Intervalo de tiempo actual
            
        Returns:
            float: Señal ponderada combinada (-1 a +1)
        """
        # Obtener pesos adaptados
        weights = self.get_all_weights(market_condition, time_interval)
        
        # Filtrar pesos para indicadores presentes en la señal
        filtered_weights = {k: v for k, v in weights.items() if k in signals}
        
        # Si no hay pesos válidos, usar pesos iguales
        if not filtered_weights:
            num_signals = len(signals)
            if num_signals == 0:
                return 0.0
            filtered_weights = {k: 1.0 / num_signals for k in signals}
        
        # Normalizar pesos filtrados
        total_weight = sum(filtered_weights.values())
        normalized_weights = {k: v / total_weight for k, v in filtered_weights.items()}
        
        # Calcular señal ponderada
        weighted_signal = sum(signals[k] * normalized_weights[k] for k in normalized_weights)
        
        return weighted_signal