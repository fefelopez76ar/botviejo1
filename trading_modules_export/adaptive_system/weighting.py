"""
Módulo para el sistema adaptativo de ponderación de indicadores técnicos
Este sistema aprende del rendimiento histórico de los indicadores y ajusta sus pesos
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum

logger = logging.getLogger("AdaptiveSystem")

class MarketCondition(Enum):
    """Condiciones de mercado para evaluación adaptativa"""
    STRONG_UPTREND = "strong_uptrend"
    MODERATE_UPTREND = "moderate_uptrend"
    LATERAL_LOW_VOL = "lateral_low_vol"
    LATERAL_HIGH_VOL = "lateral_high_vol"
    MODERATE_DOWNTREND = "moderate_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    EXTREME_VOLATILITY = "extreme_volatility"
    
    @classmethod
    def from_string(cls, condition_str: str) -> 'MarketCondition':
        """Convierte string a enum"""
        for condition in cls:
            if condition.value == condition_str:
                return condition
        raise ValueError(f"Invalid market condition: {condition_str}")

class TimeInterval(Enum):
    """Intervalos de tiempo para análisis"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    
    @classmethod
    def from_string(cls, interval_str: str) -> 'TimeInterval':
        """Convierte string a enum"""
        for interval in cls:
            if interval.value == interval_str:
                return interval
        raise ValueError(f"Invalid time interval: {interval_str}")

class IndicatorPerformance:
    """Clase para almacenar y actualizar el rendimiento de los indicadores"""
    
    def __init__(self, indicator_name: str):
        """
        Inicializa el registro de rendimiento para un indicador
        
        Args:
            indicator_name: Nombre del indicador
        """
        self.indicator_name = indicator_name
        self.total_signals = 0
        self.correct_signals = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.recent_signals = []  # Lista de tuplas (correct, profit)
        self.recent_max_size = 50  # Máximo de señales recientes a almacenar
        
        # Rendimiento por condición de mercado
        self.market_condition_performance = {cond.value: {"signals": 0, "correct": 0} for cond in MarketCondition}
        
        # Rendimiento por intervalo de tiempo
        self.time_interval_performance = {interval.value: {"signals": 0, "correct": 0} for interval in TimeInterval}
    
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
        self.total_signals += 1
        if correct:
            self.correct_signals += 1
        
        # Actualizar ganancias/pérdidas
        if profit >= 0:
            self.total_profit += profit
        else:
            self.total_loss += abs(profit)
        
        # Actualizar señales recientes
        self.recent_signals.append((correct, profit))
        if len(self.recent_signals) > self.recent_max_size:
            self.recent_signals.pop(0)
        
        # Actualizar rendimiento por condición de mercado
        condition_key = market_condition.value
        self.market_condition_performance[condition_key]["signals"] += 1
        if correct:
            self.market_condition_performance[condition_key]["correct"] += 1
        
        # Actualizar rendimiento por intervalo de tiempo
        interval_key = time_interval.value
        self.time_interval_performance[interval_key]["signals"] += 1
        if correct:
            self.time_interval_performance[interval_key]["correct"] += 1
        
        # Log de actualización
        logger.info(f"Indicador '{self.indicator_name}' actualizado: correct={correct}, profit={profit:.2f}")
    
    def get_accuracy(self) -> float:
        """
        Obtiene la tasa de acierto global
        
        Returns:
            float: Tasa de acierto (0-1)
        """
        if self.total_signals == 0:
            return 0.5  # Valor por defecto si no hay señales
        return self.correct_signals / self.total_signals
    
    def get_profit_factor(self) -> float:
        """
        Obtiene el factor de rentabilidad (profit factor)
        
        Returns:
            float: Factor de rentabilidad (ganancias/pérdidas)
        """
        if self.total_loss == 0:
            return 1.0 if self.total_profit == 0 else 2.0  # Evitar división por cero
        return self.total_profit / self.total_loss
    
    def get_market_condition_accuracy(self, condition: MarketCondition) -> float:
        """
        Obtiene la tasa de acierto para una condición de mercado específica
        
        Args:
            condition: Condición de mercado
            
        Returns:
            float: Tasa de acierto para esa condición
        """
        condition_key = condition.value
        perf = self.market_condition_performance[condition_key]
        if perf["signals"] == 0:
            return 0.5  # Valor por defecto si no hay señales
        return perf["correct"] / perf["signals"]
    
    def get_time_interval_accuracy(self, interval: TimeInterval) -> float:
        """
        Obtiene la tasa de acierto para un intervalo de tiempo específico
        
        Args:
            interval: Intervalo de tiempo
            
        Returns:
            float: Tasa de acierto para ese intervalo
        """
        interval_key = interval.value
        perf = self.time_interval_performance[interval_key]
        if perf["signals"] == 0:
            return 0.5  # Valor por defecto si no hay señales
        return perf["correct"] / perf["signals"]
    
    def get_recent_accuracy(self) -> float:
        """
        Obtiene la tasa de acierto de las señales recientes
        
        Returns:
            float: Tasa de acierto reciente
        """
        if not self.recent_signals:
            return 0.5  # Valor por defecto si no hay señales recientes
        
        recent_correct = sum(1 for correct, _ in self.recent_signals if correct)
        return recent_correct / len(self.recent_signals)
    
    def to_dict(self) -> Dict:
        """
        Convierte el objeto a un diccionario para serialización
        
        Returns:
            Dict: Representación en diccionario
        """
        return {
            "indicator_name": self.indicator_name,
            "total_signals": self.total_signals,
            "correct_signals": self.correct_signals,
            "total_profit": self.total_profit,
            "total_loss": self.total_loss,
            "recent_signals": self.recent_signals,
            "market_condition_performance": self.market_condition_performance,
            "time_interval_performance": self.time_interval_performance
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
        obj = cls(data["indicator_name"])
        obj.total_signals = data["total_signals"]
        obj.correct_signals = data["correct_signals"]
        obj.total_profit = data["total_profit"]
        obj.total_loss = data["total_loss"]
        obj.recent_signals = data["recent_signals"]
        obj.market_condition_performance = data["market_condition_performance"]
        obj.time_interval_performance = data["time_interval_performance"]
        return obj

class AdaptiveWeightingSystem:
    """Sistema de ponderación adaptativa para indicadores técnicos"""
    
    def __init__(self, data_file: str = "indicator_performance.json"):
        """
        Inicializa el sistema de ponderación
        
        Args:
            data_file: Archivo para guardar/cargar datos de rendimiento
        """
        self.data_file = data_file
        self.indicators = {}  # Dict[str, IndicatorPerformance]
        self.recalibration_count = 0
        self.last_recalibration = datetime.now()
        
        # Inicializar o cargar datos
        if os.path.exists(data_file):
            self._load_data()
        else:
            self._initialize_indicators()
    
    def _load_data(self):
        """Carga datos de rendimiento desde archivo"""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            
            self.indicators = {}
            for indicator_name, indicator_data in data["indicators"].items():
                self.indicators[indicator_name] = IndicatorPerformance.from_dict(indicator_data)
            
            self.recalibration_count = data.get("recalibration_count", 0)
            self.last_recalibration = datetime.fromisoformat(data.get("last_recalibration", datetime.now().isoformat()))
            
            logger.info(f"Datos de rendimiento cargados desde {self.data_file}")
        except Exception as e:
            logger.error(f"Error al cargar datos de rendimiento: {e}")
            self._initialize_indicators()
    
    def _initialize_indicators(self):
        """Inicializa los indicadores con valores por defecto"""
        default_indicators = [
            "rsi",
            "macd",
            "bollinger_bands",
            "moving_averages",
            "adx",
            "ema_crossover",
            "volume_profile",
            "stochastic",
            "atr",
            "ichimoku"
        ]
        
        self.indicators = {name: IndicatorPerformance(name) for name in default_indicators}
        logger.info("Indicadores inicializados con valores por defecto")
    
    def _save_data(self):
        """Guarda datos de rendimiento en archivo"""
        try:
            data = {
                "indicators": {name: indicator.to_dict() for name, indicator in self.indicators.items()},
                "recalibration_count": self.recalibration_count,
                "last_recalibration": self.last_recalibration.isoformat()
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Datos de rendimiento guardados en {self.data_file}")
        except Exception as e:
            logger.error(f"Error al guardar datos de rendimiento: {e}")
    
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
        # Crear el indicador si no existe
        if indicator_name not in self.indicators:
            self.indicators[indicator_name] = IndicatorPerformance(indicator_name)
        
        # Actualizar rendimiento
        self.indicators[indicator_name].update(correct, profit, market_condition, time_interval)
        
        # Guardar después de cada actualización
        self._save_data()
        
        # Recalibrar pesos si es necesario
        self._recalibrate_weights()
    
    def _recalibrate_weights(self):
        """Recalibra los pesos de los indicadores basándose en su rendimiento"""
        # Incrementar contador y actualizar timestamp
        self.recalibration_count += 1
        self.last_recalibration = datetime.now()
        
        logger.info(f"Pesos recalibrados. Recalibración #{self.recalibration_count}")
    
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
        if indicator_name not in self.indicators:
            return 1.0  # Peso por defecto para indicadores desconocidos
        
        indicator = self.indicators[indicator_name]
        
        # Factores de rendimiento
        overall_accuracy = indicator.get_accuracy()
        profit_factor = min(indicator.get_profit_factor(), 3.0) / 3.0  # Normalizado a 0-1
        condition_accuracy = indicator.get_market_condition_accuracy(market_condition)
        interval_accuracy = indicator.get_time_interval_accuracy(time_interval)
        recent_accuracy = indicator.get_recent_accuracy()
        
        # Ponderación de factores (ajustar según necesidad)
        weight_factors = {
            "overall_accuracy": 0.15,
            "profit_factor": 0.20,
            "condition_accuracy": 0.30,
            "interval_accuracy": 0.15,
            "recent_accuracy": 0.20
        }
        
        # Calcular peso final
        weight = (
            overall_accuracy * weight_factors["overall_accuracy"] +
            profit_factor * weight_factors["profit_factor"] +
            condition_accuracy * weight_factors["condition_accuracy"] +
            interval_accuracy * weight_factors["interval_accuracy"] +
            recent_accuracy * weight_factors["recent_accuracy"]
        )
        
        # Normalizar
        min_weight = 0.5  # Peso mínimo para que todos tengan cierta influencia
        normalized_weight = min_weight + (1.0 - min_weight) * weight
        
        return normalized_weight
    
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
        return {
            name: self.get_indicator_weight(name, market_condition, time_interval)
            for name in self.indicators
        }
    
    def detect_market_condition(self, df: pd.DataFrame) -> MarketCondition:
        """
        Detecta la condición actual del mercado basándose en indicadores
        
        Args:
            df: DataFrame con datos de mercado (OHLCV)
            
        Returns:
            MarketCondition: Condición de mercado detectada
        """
        # 1. Calcular tendencia usando medias móviles
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        df['sma200'] = df['close'].rolling(window=200).mean()
        
        # Ángulo de tendencia (slope)
        recent_sma20 = df['sma20'].dropna().iloc[-20:]
        slope = (recent_sma20.iloc[-1] - recent_sma20.iloc[0]) / (recent_sma20.iloc[0] * 20)
        
        # 2. Calcular volatilidad
        df['returns'] = df['close'].pct_change()
        volatility = df['returns'].std() * np.sqrt(252)  # Anualizada
        
        # 3. Calcular rango diario promedio
        df['daily_range'] = (df['high'] - df['low']) / df['low']
        avg_range = df['daily_range'].rolling(window=14).mean().iloc[-1]
        
        # 4. Fuerza relativa (volumen ajustado al precio)
        df['volume_price_ratio'] = df['volume'] / df['close']
        avg_vol_price = df['volume_price_ratio'].rolling(window=20).mean().iloc[-1]
        
        # Determinar condición basada en métricas
        if slope > 0.01:  # Tendencia alcista fuerte
            if volatility > 0.8:
                return MarketCondition.EXTREME_VOLATILITY
            return MarketCondition.STRONG_UPTREND
        elif 0.003 < slope <= 0.01:  # Tendencia alcista moderada
            return MarketCondition.MODERATE_UPTREND
        elif -0.003 <= slope <= 0.003:  # Mercado lateral
            if volatility < 0.4:
                return MarketCondition.LATERAL_LOW_VOL
            else:
                return MarketCondition.LATERAL_HIGH_VOL
        elif -0.01 <= slope < -0.003:  # Tendencia bajista moderada
            return MarketCondition.MODERATE_DOWNTREND
        else:  # slope < -0.01, Tendencia bajista fuerte
            if volatility > 0.8:
                return MarketCondition.EXTREME_VOLATILITY
            return MarketCondition.STRONG_DOWNTREND
    
    def get_example_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Obtiene ejemplos de ponderaciones para diferentes condiciones de mercado
        
        Returns:
            Dict: Ejemplos de pesos por condición de mercado
        """
        examples = {}
        
        # Para cada tipo de condición de mercado
        for condition in MarketCondition:
            # Para un intervalo específico
            interval = TimeInterval.MINUTE_15
            
            # Obtener pesos para esta condición
            weights = self.get_all_weights(condition, interval)
            
            # Agregar al diccionario de ejemplos
            examples[condition.value] = weights
        
        return examples
    
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
        # Verificar que hay señales
        if not signals:
            return 0.0
        
        # Obtener pesos para las condiciones actuales
        weights = {}
        total_weight = 0.0
        
        for indicator, signal in signals.items():
            if indicator in self.indicators:
                weight = self.get_indicator_weight(indicator, market_condition, time_interval)
                weights[indicator] = weight
                total_weight += weight
        
        # Si no hay indicadores conocidos, usar pesos iguales
        if total_weight == 0:
            return sum(signals.values()) / len(signals)
        
        # Calcular señal ponderada
        weighted_signal = sum(signals[indicator] * weights.get(indicator, 1.0) 
                             for indicator in signals) / total_weight
        
        # Asegurar que está en el rango [-1, 1]
        return max(-1.0, min(1.0, weighted_signal))
    
    def get_indicator_performance_metrics(self, indicator_name: str) -> Dict:
        """
        Obtiene métricas detalladas de rendimiento para un indicador
        
        Args:
            indicator_name: Nombre del indicador
            
        Returns:
            Dict: Métricas de rendimiento
        """
        if indicator_name not in self.indicators:
            return {"error": "Indicador no encontrado"}
        
        indicator = self.indicators[indicator_name]
        
        # Métricas generales
        metrics = {
            "name": indicator.indicator_name,
            "total_signals": indicator.total_signals,
            "accuracy": indicator.get_accuracy(),
            "profit_factor": indicator.get_profit_factor(),
            "total_profit": indicator.total_profit,
            "total_loss": indicator.total_loss,
            "recent_accuracy": indicator.get_recent_accuracy()
        }
        
        # Rendimiento por condición de mercado
        market_conditions = {}
        for condition in MarketCondition:
            accuracy = indicator.get_market_condition_accuracy(condition)
            signals = indicator.market_condition_performance[condition.value]["signals"]
            market_conditions[condition.value] = {
                "accuracy": accuracy,
                "signals": signals
            }
        
        metrics["market_conditions"] = market_conditions
        
        # Rendimiento por intervalo de tiempo
        time_intervals = {}
        for interval in TimeInterval:
            accuracy = indicator.get_time_interval_accuracy(interval)
            signals = indicator.time_interval_performance[interval.value]["signals"]
            time_intervals[interval.value] = {
                "accuracy": accuracy,
                "signals": signals
            }
        
        metrics["time_intervals"] = time_intervals
        
        return metrics