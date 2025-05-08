"""
Módulo de ponderación adaptativa de indicadores según su rendimiento histórico

Este sistema implementa un mecanismo de aprendizaje que ajusta dinámicamente
el peso de cada indicador técnico basándose en su precisión histórica en
diferentes condiciones de mercado para Solana.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from datetime import datetime, timedelta

# Configurar logging
logger = logging.getLogger("AdaptiveWeighting")

# Definir condiciones de mercado
class MarketCondition(Enum):
    STRONG_UPTREND = "strong_uptrend"
    MODERATE_UPTREND = "moderate_uptrend"
    LATERAL_LOW_VOL = "lateral_low_vol"
    LATERAL_HIGH_VOL = "lateral_high_vol"
    MODERATE_DOWNTREND = "moderate_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    EXTREME_VOLATILITY = "extreme_volatility"

# Definir intervalos de tiempo
class TimeInterval(Enum):
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"


class IndicatorPerformance:
    """Clase para almacenar y actualizar el rendimiento de los indicadores"""
    
    def __init__(self, indicator_name: str):
        """
        Inicializa el registro de rendimiento para un indicador
        
        Args:
            indicator_name: Nombre del indicador
        """
        self.name = indicator_name
        self.total_signals = 0
        self.correct_signals = 0
        self.profit_sum = 0.0
        self.loss_sum = 0.0
        
        # Rendimiento por condición de mercado
        self.market_condition_performance = {
            condition.value: {
                "total": 0,
                "correct": 0,
                "profit_sum": 0.0,
                "loss_sum": 0.0
            } for condition in MarketCondition
        }
        
        # Rendimiento por intervalo de tiempo
        self.time_interval_performance = {
            interval.value: {
                "total": 0,
                "correct": 0,
                "profit_sum": 0.0,
                "loss_sum": 0.0
            } for interval in TimeInterval
        }
        
        # Rendimiento reciente (últimas 100 señales)
        self.recent_signals = []  # Lista de tuplas (correct: bool, profit: float)
    
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
        # Actualizar estadísticas globales
        self.total_signals += 1
        if correct:
            self.correct_signals += 1
            self.profit_sum += profit
        else:
            self.loss_sum += abs(profit)  # profit será negativo, lo convertimos a positivo
        
        # Actualizar estadísticas por condición de mercado
        mc_stats = self.market_condition_performance[market_condition.value]
        mc_stats["total"] += 1
        if correct:
            mc_stats["correct"] += 1
            mc_stats["profit_sum"] += profit
        else:
            mc_stats["loss_sum"] += abs(profit)
        
        # Actualizar estadísticas por intervalo de tiempo
        ti_stats = self.time_interval_performance[time_interval.value]
        ti_stats["total"] += 1
        if correct:
            ti_stats["correct"] += 1
            ti_stats["profit_sum"] += profit
        else:
            ti_stats["loss_sum"] += abs(profit)
        
        # Actualizar estadísticas recientes
        self.recent_signals.append((correct, profit))
        if len(self.recent_signals) > 100:
            self.recent_signals.pop(0)  # Eliminar la señal más antigua
    
    def get_accuracy(self) -> float:
        """
        Obtiene la tasa de acierto global
        
        Returns:
            float: Tasa de acierto (0-1)
        """
        if self.total_signals == 0:
            return 0.5  # Valor neutral si no hay señales
        return self.correct_signals / self.total_signals
    
    def get_profit_factor(self) -> float:
        """
        Obtiene el factor de rentabilidad (profit factor)
        
        Returns:
            float: Factor de rentabilidad (ganancias/pérdidas)
        """
        if self.loss_sum == 0:
            return 2.0  # Valor alto pero no infinito si no hay pérdidas
        return self.profit_sum / self.loss_sum
    
    def get_market_condition_accuracy(self, condition: MarketCondition) -> float:
        """
        Obtiene la tasa de acierto para una condición de mercado específica
        
        Args:
            condition: Condición de mercado
            
        Returns:
            float: Tasa de acierto para esa condición
        """
        stats = self.market_condition_performance[condition.value]
        if stats["total"] == 0:
            return 0.5  # Valor neutral
        return stats["correct"] / stats["total"]
    
    def get_time_interval_accuracy(self, interval: TimeInterval) -> float:
        """
        Obtiene la tasa de acierto para un intervalo de tiempo específico
        
        Args:
            interval: Intervalo de tiempo
            
        Returns:
            float: Tasa de acierto para ese intervalo
        """
        stats = self.time_interval_performance[interval.value]
        if stats["total"] == 0:
            return 0.5  # Valor neutral
        return stats["correct"] / stats["total"]
    
    def get_recent_accuracy(self) -> float:
        """
        Obtiene la tasa de acierto de las señales recientes
        
        Returns:
            float: Tasa de acierto reciente
        """
        if not self.recent_signals:
            return 0.5  # Valor neutral
        
        correct_count = sum(1 for correct, _ in self.recent_signals if correct)
        return correct_count / len(self.recent_signals)
    
    def to_dict(self) -> Dict:
        """
        Convierte el objeto a un diccionario para serialización
        
        Returns:
            Dict: Representación en diccionario
        """
        return {
            "name": self.name,
            "total_signals": self.total_signals,
            "correct_signals": self.correct_signals,
            "profit_sum": self.profit_sum,
            "loss_sum": self.loss_sum,
            "market_condition_performance": self.market_condition_performance,
            "time_interval_performance": self.time_interval_performance,
            "recent_signals": self.recent_signals
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
        instance = cls(data["name"])
        instance.total_signals = data["total_signals"]
        instance.correct_signals = data["correct_signals"]
        instance.profit_sum = data["profit_sum"]
        instance.loss_sum = data["loss_sum"]
        instance.market_condition_performance = data["market_condition_performance"]
        instance.time_interval_performance = data["time_interval_performance"]
        instance.recent_signals = data["recent_signals"]
        return instance


class AdaptiveWeightingSystem:
    """Sistema de ponderación adaptativa para indicadores técnicos"""
    
    def __init__(self, data_file: str = "indicator_performance.json"):
        """
        Inicializa el sistema de ponderación
        
        Args:
            data_file: Archivo para guardar/cargar datos de rendimiento
        """
        self.data_file = data_file
        self.indicators = {}  # Diccionario de IndicatorPerformance
        self.last_calibration = time.time()
        self.calibration_interval = 86400  # 24 horas en segundos
        
        # Pesos base para cada indicador (1.0 = peso neutral)
        self.base_weights = {
            "sma_crossover": 1.0,
            "ema_crossover": 1.0,
            "rsi": 1.0,
            "macd": 1.0,
            "bollinger_bands": 1.0,
            "atr": 1.0,
            "ichimoku": 1.0,
            "stochastic": 1.0,
            "adx": 1.0,
            "volume": 1.0,
            "price_action": 1.0
        }
        
        # Cargar datos previos si existen
        self._load_data()
    
    def _load_data(self):
        """Carga datos de rendimiento desde archivo"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                for indicator_name, indicator_data in data.items():
                    self.indicators[indicator_name] = IndicatorPerformance.from_dict(indicator_data)
                
                logger.info(f"Cargados datos de rendimiento para {len(self.indicators)} indicadores")
            except Exception as e:
                logger.error(f"Error al cargar datos de rendimiento: {e}")
                # Crear nuevos datos si hay error
                self._initialize_indicators()
        else:
            # Inicializar indicadores con valores por defecto
            self._initialize_indicators()
    
    def _initialize_indicators(self):
        """Inicializa los indicadores con valores por defecto"""
        for indicator_name in self.base_weights.keys():
            self.indicators[indicator_name] = IndicatorPerformance(indicator_name)
        
        logger.info(f"Inicializados {len(self.indicators)} indicadores con valores por defecto")
    
    def _save_data(self):
        """Guarda datos de rendimiento en archivo"""
        try:
            data = {name: indicator.to_dict() for name, indicator in self.indicators.items()}
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Guardados datos de rendimiento para {len(self.indicators)} indicadores")
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
        if indicator_name not in self.indicators:
            logger.warning(f"Indicador desconocido: {indicator_name}, inicializando")
            self.indicators[indicator_name] = IndicatorPerformance(indicator_name)
        
        self.indicators[indicator_name].update(
            correct=correct,
            profit=profit,
            market_condition=market_condition,
            time_interval=time_interval
        )
        
        # Verificar si es momento de recalibrar
        current_time = time.time()
        if current_time - self.last_calibration > self.calibration_interval:
            self._recalibrate_weights()
            self.last_calibration = current_time
            self._save_data()
    
    def _recalibrate_weights(self):
        """Recalibra los pesos de los indicadores basándose en su rendimiento"""
        logger.info("Recalibrando pesos de indicadores...")
        
        # Recalibrar pesos base según rendimiento global
        for name, indicator in self.indicators.items():
            if name in self.base_weights and indicator.total_signals > 50:
                accuracy = indicator.get_accuracy()
                profit_factor = indicator.get_profit_factor()
                recent_accuracy = indicator.get_recent_accuracy()
                
                # Fórmula de recalibración
                new_weight = (
                    accuracy * 0.5 +                  # 50% basado en precisión global
                    min(profit_factor / 2, 1.0) * 0.3 +  # 30% basado en factor de rentabilidad (max 1.0)
                    recent_accuracy * 0.2             # 20% basado en precisión reciente
                ) * 2.0  # Multiplicador para rango 0-2.0
                
                # Limitar rango
                new_weight = max(0.5, min(2.5, new_weight))
                
                # Actualizar peso base
                self.base_weights[name] = new_weight
                
                logger.info(f"Recalibrado {name}: Nuevo peso base = {new_weight:.2f} "
                          f"(precisión: {accuracy:.2f}, factor rentabilidad: {profit_factor:.2f}, "
                          f"precisión reciente: {recent_accuracy:.2f})")
    
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
        if indicator_name not in self.base_weights:
            return 1.0  # Peso neutral para indicadores desconocidos
        
        base_weight = self.base_weights.get(indicator_name, 1.0)
        
        # Consultar rendimiento específico por condición de mercado e intervalo
        condition_factor = 1.0
        interval_factor = 1.0
        
        if indicator_name in self.indicators:
            indicator = self.indicators[indicator_name]
            
            # Ajustar por rendimiento en condiciones de mercado específicas
            mc_accuracy = indicator.get_market_condition_accuracy(market_condition)
            # Calcular factor que aumenta/disminuye en base a lo bueno/malo que es en esta condición
            # 0.5 = neutral, <0.5 = malo, >0.5 = bueno
            condition_factor = 0.5 + (mc_accuracy - 0.5) * 1.5  # Rango aproximado: 0.5-1.5
            
            # Ajustar por rendimiento en intervalos específicos
            ti_accuracy = indicator.get_time_interval_accuracy(time_interval)
            interval_factor = 0.5 + (ti_accuracy - 0.5) * 1.5  # Rango aproximado: 0.5-1.5
        
        # Ajustes especiales para condiciones extremas
        if market_condition == MarketCondition.EXTREME_VOLATILITY:
            # Reducir osciladores en volatilidad extrema
            if indicator_name in ["rsi", "stochastic"]:
                condition_factor *= 0.7
            # Aumentar tendencia en volatilidad extrema
            elif indicator_name in ["sma_crossover", "ema_crossover", "adx"]:
                condition_factor *= 1.3
        
        # Calcular peso final
        final_weight = base_weight * condition_factor * interval_factor
        
        # Limitar rango final
        final_weight = max(0.3, min(3.0, final_weight))
        
        return final_weight
    
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
            for name in self.base_weights.keys()
        }
    
    def detect_market_condition(self, df: pd.DataFrame) -> MarketCondition:
        """
        Detecta la condición actual del mercado basándose en indicadores
        
        Args:
            df: DataFrame con datos de mercado (OHLCV)
            
        Returns:
            MarketCondition: Condición de mercado detectada
        """
        # Calcular volatilidad
        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(365)
        
        # Calcular ADX para fuerza de tendencia
        # Nota: Simplificado, en producción usar función completa
        high_minus_low = df['high'] - df['low']
        atr = high_minus_low.rolling(14).mean().iloc[-1]
        
        # Calcular direccionalidad (simplificado, usar función completa en producción)
        price_change_pct = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100
        
        # Detectar volatilidad extrema
        if volatility > 1.5:  # >150% anualizado
            return MarketCondition.EXTREME_VOLATILITY
        
        # Detectar tendencias
        if price_change_pct > 20:  # +20% en 20 periodos
            return MarketCondition.STRONG_UPTREND
        elif price_change_pct > 5:  # +5% en 20 periodos
            return MarketCondition.MODERATE_UPTREND
        elif price_change_pct < -20:  # -20% en 20 periodos
            return MarketCondition.STRONG_DOWNTREND
        elif price_change_pct < -5:  # -5% en 20 periodos
            return MarketCondition.MODERATE_DOWNTREND
        
        # Mercado lateral (sin tendencia clara)
        if volatility < 0.5:  # <50% anualizado
            return MarketCondition.LATERAL_LOW_VOL
        else:
            return MarketCondition.LATERAL_HIGH_VOL
    
    def get_example_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Obtiene ejemplos de ponderaciones para diferentes condiciones de mercado
        
        Returns:
            Dict: Ejemplos de pesos por condición de mercado
        """
        examples = {}
        
        # Para cada condición de mercado, obtener pesos con intervalo de 15m
        for condition in MarketCondition:
            weights = self.get_all_weights(condition, TimeInterval.MINUTE_15)
            examples[condition.value] = {
                k: round(v, 2) for k, v in weights.items()
            }
        
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
        if not signals:
            return 0.0  # Neutral si no hay señales
        
        # Obtener pesos adaptados
        weights = {
            indicator: self.get_indicator_weight(indicator, market_condition, time_interval)
            for indicator in signals.keys()
        }
        
        # Calcular señal ponderada
        weighted_sum = sum(signal * weights.get(indicator, 1.0) 
                         for indicator, signal in signals.items())
        total_weight = sum(weights.get(indicator, 1.0) 
                         for indicator in signals.keys())
        
        if total_weight == 0:
            return 0.0
        
        weighted_signal = weighted_sum / total_weight
        
        # Limitar al rango -1 a +1
        return max(-1.0, min(1.0, weighted_signal))


# Ejemplo de uso para documentación
if __name__ == "__main__":
    # Crear sistema
    aws = AdaptiveWeightingSystem()
    
    # Simular actualizaciones de rendimiento
    aws.update_indicator_performance(
        "rsi", 
        True, 
        0.05, 
        MarketCondition.LATERAL_LOW_VOL, 
        TimeInterval.MINUTE_15
    )
    
    # Obtener pesos
    weight = aws.get_indicator_weight(
        "rsi", 
        MarketCondition.LATERAL_LOW_VOL, 
        TimeInterval.MINUTE_15
    )
    
    print(f"Peso adaptado para RSI: {weight}")
    
    # Ejemplo de ponderaciones para diferentes condiciones
    examples = aws.get_example_weights()
    print("Ejemplos de ponderaciones por condición de mercado:")
    print(json.dumps(examples, indent=2))
    
    # Ejemplo de señal ponderada
    signals = {
        "rsi": -0.8,         # Fuerte señal de sobreventa
        "macd": -0.3,        # Débil señal de bajada
        "bollinger_bands": 0.9  # Fuerte señal de sobrecompra
    }
    
    weighted_signal = aws.calculate_weighted_signal(
        signals, 
        MarketCondition.LATERAL_LOW_VOL, 
        TimeInterval.MINUTE_15
    )
    
    print(f"Señal ponderada: {weighted_signal}")