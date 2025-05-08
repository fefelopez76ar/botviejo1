"""
Sistema de ponderación adaptativa de indicadores según confiabilidad histórica
Este módulo implementa un sistema que ajusta dinámicamente los pesos de los indicadores
técnicos basándose en su desempeño histórico en diferentes condiciones de mercado.
"""

import json
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Configurar logging
logger = logging.getLogger("IndicatorWeighting")

class IndicatorPerformanceTracker:
    """
    Clase que registra y analiza el rendimiento histórico de los indicadores técnicos
    para ajustar dinámicamente sus pesos en la toma de decisiones.
    """
    
    # Archivo para almacenar historial de rendimiento de indicadores
    PERFORMANCE_FILE = "indicator_performance.json"
    
    # Lista de indicadores técnicos soportados
    SUPPORTED_INDICATORS = [
        'sma_crossover', 'ema_crossover', 'rsi', 'macd', 'bollinger_bands',
        'stochastic', 'atr', 'adx', 'volume', 'candle_patterns'
    ]
    
    # Condiciones de mercado
    MARKET_CONDITIONS = [
        'uptrend_strong', 'uptrend_weak', 'downtrend_strong', 'downtrend_weak',
        'range_low_vol', 'range_high_vol', 'extreme_volatility'
    ]
    
    # Intervalos de tiempo
    TIME_INTERVALS = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    
    def __init__(self, symbol: str = 'SOL-USDT'):
        """
        Inicializa el tracker de rendimiento de indicadores
        
        Args:
            symbol: Símbolo del mercado a analizar
        """
        self.symbol = symbol
        self.performance_data = self._load_performance_data()
        self.current_weights = {}
        self.current_market_condition = 'range_low_vol'  # Condición por defecto
        self.current_interval = '15m'  # Intervalo por defecto
        
        # Inicializar pesos actuales
        self._calculate_current_weights()
    
    def _load_performance_data(self) -> Dict:
        """
        Carga los datos de rendimiento histórico desde archivo
        
        Returns:
            Dict: Datos de rendimiento histórico
        """
        if os.path.exists(self.PERFORMANCE_FILE):
            try:
                with open(self.PERFORMANCE_FILE, 'r') as file:
                    return json.load(file)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading performance data: {e}")
                return self._initialize_performance_data()
        else:
            logger.info("Performance file not found. Initializing with default values.")
            return self._initialize_performance_data()
    
    def _initialize_performance_data(self) -> Dict:
        """
        Inicializa la estructura de datos para el rendimiento de indicadores
        
        Returns:
            Dict: Estructura de datos inicializada
        """
        performance = {
            'metadata': {
                'symbol': self.symbol,
                'last_updated': datetime.now().isoformat(),
                'total_signals': 0
            },
            'indicators': {}
        }
        
        # Inicializar datos para cada indicador
        for indicator in self.SUPPORTED_INDICATORS:
            performance['indicators'][indicator] = {
                'overall': {
                    'total_signals': 0,
                    'correct_signals': 0,
                    'false_signals': 0,
                    'accuracy_rate': 0.5,  # Valor inicial neutral
                    'avg_profit': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 1.0,  # Valor inicial neutral
                    'base_weight': 1.0     # Peso base inicial
                },
                'market_conditions': {},
                'time_intervals': {}
            }
            
            # Inicializar para cada condición de mercado
            for condition in self.MARKET_CONDITIONS:
                performance['indicators'][indicator]['market_conditions'][condition] = {
                    'total_signals': 0,
                    'correct_signals': 0,
                    'accuracy_rate': 0.5,
                    'avg_profit': 0.0,
                    'avg_loss': 0.0,
                    'weight_modifier': 1.0  # Modificador de peso inicial
                }
            
            # Inicializar para cada intervalo de tiempo
            for interval in self.TIME_INTERVALS:
                performance['indicators'][indicator]['time_intervals'][interval] = {
                    'total_signals': 0,
                    'correct_signals': 0,
                    'accuracy_rate': 0.5,
                    'avg_profit': 0.0,
                    'avg_loss': 0.0,
                    'weight_modifier': 1.0  # Modificador de peso inicial
                }
        
        # Para Solana específicamente - ajustes iniciales según investigación previa
        # Estos valores son estimaciones iniciales basadas en el comportamiento típico de Solana
        performance['indicators']['bollinger_bands']['base_weight'] = 1.5
        performance['indicators']['rsi']['base_weight'] = 1.3
        performance['indicators']['sma_crossover']['base_weight'] = 1.8
        
        return performance
    
    def save_performance_data(self):
        """Guarda los datos de rendimiento en archivo"""
        try:
            # Actualizar metadata
            self.performance_data['metadata']['last_updated'] = datetime.now().isoformat()
            
            with open(self.PERFORMANCE_FILE, 'w') as file:
                json.dump(self.performance_data, file, indent=2)
            
            logger.info("Performance data saved successfully")
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def record_signal_result(self, indicator: str, signal_correct: bool, 
                          profit_loss: float, market_condition: str, time_interval: str):
        """
        Registra el resultado de una señal de trading para un indicador
        
        Args:
            indicator: Nombre del indicador
            signal_correct: Si la señal fue correcta o no
            profit_loss: Beneficio o pérdida resultante (positivo o negativo)
            market_condition: Condición de mercado al momento de la señal
            time_interval: Intervalo de tiempo utilizado
        """
        if indicator not in self.SUPPORTED_INDICATORS:
            logger.warning(f"Unsupported indicator: {indicator}")
            return
        
        if market_condition not in self.MARKET_CONDITIONS:
            logger.warning(f"Unsupported market condition: {market_condition}")
            market_condition = 'range_low_vol'  # Valor por defecto
        
        if time_interval not in self.TIME_INTERVALS:
            logger.warning(f"Unsupported time interval: {time_interval}")
            time_interval = '15m'  # Valor por defecto
        
        # Actualizar estadísticas generales
        overall = self.performance_data['indicators'][indicator]['overall']
        overall['total_signals'] += 1
        self.performance_data['metadata']['total_signals'] += 1
        
        if signal_correct:
            overall['correct_signals'] += 1
            if profit_loss > 0:
                # Actualizar promedio de ganancias (weighted average)
                prev_profit = overall['avg_profit']
                prev_count = overall['correct_signals'] - 1
                overall['avg_profit'] = (prev_profit * prev_count + profit_loss) / overall['correct_signals']
        else:
            overall['false_signals'] += 1
            if profit_loss < 0:
                # Actualizar promedio de pérdidas (weighted average)
                prev_loss = overall['avg_loss']
                prev_count = overall['false_signals'] - 1
                overall['avg_loss'] = (prev_loss * prev_count + abs(profit_loss)) / overall['false_signals']
        
        # Actualizar tasa de precisión
        if overall['total_signals'] > 0:
            overall['accuracy_rate'] = overall['correct_signals'] / overall['total_signals']
        
        # Actualizar profit factor
        if overall['avg_loss'] > 0:
            overall['profit_factor'] = overall['avg_profit'] / overall['avg_loss']
        else:
            overall['profit_factor'] = overall['avg_profit'] * 2  # Valor arbitrario alto si no hay pérdidas
        
        # Actualizar estadísticas para la condición de mercado
        market_stats = self.performance_data['indicators'][indicator]['market_conditions'][market_condition]
        market_stats['total_signals'] += 1
        
        if signal_correct:
            market_stats['correct_signals'] += 1
        
        if market_stats['total_signals'] > 0:
            market_stats['accuracy_rate'] = market_stats['correct_signals'] / market_stats['total_signals']
        
        # Actualizar estadísticas para el intervalo de tiempo
        interval_stats = self.performance_data['indicators'][indicator]['time_intervals'][time_interval]
        interval_stats['total_signals'] += 1
        
        if signal_correct:
            interval_stats['correct_signals'] += 1
        
        if interval_stats['total_signals'] > 0:
            interval_stats['accuracy_rate'] = interval_stats['correct_signals'] / interval_stats['total_signals']
        
        # Recalcular los modificadores de peso
        self._recalculate_weight_modifiers(indicator, market_condition, time_interval)
        
        # Recalcular pesos actuales
        self._calculate_current_weights()
        
        # Guardar datos
        self.save_performance_data()
    
    def _recalculate_weight_modifiers(self, indicator: str, market_condition: str, time_interval: str):
        """
        Recalcula los modificadores de peso basados en el rendimiento
        
        Args:
            indicator: Nombre del indicador
            market_condition: Condición de mercado
            time_interval: Intervalo de tiempo
        """
        # Recalcular modificador para condición de mercado
        market_stats = self.performance_data['indicators'][indicator]['market_conditions'][market_condition]
        if market_stats['total_signals'] >= 10:  # Solo recalcular si hay suficientes datos
            # Fórmula: base + (accuracy - 0.5) * 2
            # Esto da valores entre 0 (para accuracy 0%) y 2 (para accuracy 100%)
            new_modifier = 1.0 + (market_stats['accuracy_rate'] - 0.5) * 2
            
            # Limitar modificador entre 0.3 y 3.0
            new_modifier = max(0.3, min(3.0, new_modifier))
            
            # Actualizar gradualmente para evitar cambios bruscos
            market_stats['weight_modifier'] = (
                market_stats['weight_modifier'] * 0.7 + new_modifier * 0.3
            )
        
        # Recalcular modificador para intervalo de tiempo
        interval_stats = self.performance_data['indicators'][indicator]['time_intervals'][time_interval]
        if interval_stats['total_signals'] >= 10:  # Solo recalcular si hay suficientes datos
            new_modifier = 1.0 + (interval_stats['accuracy_rate'] - 0.5) * 2
            new_modifier = max(0.3, min(3.0, new_modifier))
            
            interval_stats['weight_modifier'] = (
                interval_stats['weight_modifier'] * 0.7 + new_modifier * 0.3
            )
        
        # Recalcular peso base del indicador
        overall = self.performance_data['indicators'][indicator]['overall']
        if overall['total_signals'] >= 30:  # Suficientes datos para ajuste significativo
            # Considerar tanto precisión como profit factor
            accuracy_component = (overall['accuracy_rate'] - 0.5) * 2  # Entre -1 y 1
            profit_component = min(1.0, (overall['profit_factor'] - 1) / 3)  # Entre 0 y 1
            
            new_base_weight = 1.0 + accuracy_component * 0.6 + profit_component * 0.4
            new_base_weight = max(0.5, min(2.5, new_base_weight))
            
            # Actualizar gradualmente
            overall['base_weight'] = overall['base_weight'] * 0.8 + new_base_weight * 0.2
    
    def set_current_context(self, market_condition: str, time_interval: str):
        """
        Establece el contexto actual para cálculo de pesos
        
        Args:
            market_condition: Condición actual del mercado
            time_interval: Intervalo de tiempo actual
        """
        if market_condition in self.MARKET_CONDITIONS:
            self.current_market_condition = market_condition
        
        if time_interval in self.TIME_INTERVALS:
            self.current_interval = time_interval
        
        # Recalcular pesos actuales con el nuevo contexto
        self._calculate_current_weights()
    
    def _calculate_current_weights(self):
        """Calcula los pesos actuales basados en el contexto actual"""
        self.current_weights = {}
        
        for indicator in self.SUPPORTED_INDICATORS:
            indicator_data = self.performance_data['indicators'][indicator]
            
            # Obtener peso base
            base_weight = indicator_data['overall']['base_weight']
            
            # Obtener modificadores para contexto actual
            market_modifier = indicator_data['market_conditions'][self.current_market_condition]['weight_modifier']
            interval_modifier = indicator_data['time_intervals'][self.current_interval]['weight_modifier']
            
            # Calcular peso final
            final_weight = base_weight * market_modifier * interval_modifier
            
            # Aplicar ajustes especiales para condiciones extremas
            if self.current_market_condition == 'extreme_volatility':
                # Reducir peso de osciladores
                if indicator in ['rsi', 'stochastic']:
                    final_weight *= 0.7
                # Aumentar peso de indicadores de tendencia
                elif indicator in ['sma_crossover', 'ema_crossover', 'adx']:
                    final_weight *= 1.3
                # Aumentar peso de patrones de velas para reversiones
                elif indicator == 'candle_patterns':
                    final_weight *= 1.2
            
            # Guardar peso final
            self.current_weights[indicator] = final_weight
        
        # Normalizar pesos (opcional, si se desea que sumen a un valor específico)
        # self._normalize_weights()
        
        logger.debug(f"Updated indicator weights: {self.current_weights}")
    
    def _normalize_weights(self, target_sum: float = 10.0):
        """
        Normaliza los pesos para que sumen un valor específico
        
        Args:
            target_sum: Valor objetivo para la suma de los pesos
        """
        weight_sum = sum(self.current_weights.values())
        
        if weight_sum > 0:
            scale_factor = target_sum / weight_sum
            
            for indicator in self.current_weights:
                self.current_weights[indicator] *= scale_factor
    
    def get_indicator_weight(self, indicator: str) -> float:
        """
        Obtiene el peso actual para un indicador específico
        
        Args:
            indicator: Nombre del indicador
            
        Returns:
            float: Peso actual del indicador
        """
        if indicator in self.current_weights:
            return self.current_weights[indicator]
        else:
            logger.warning(f"Weight requested for unknown indicator: {indicator}")
            return 1.0  # Valor por defecto
    
    def get_all_weights(self) -> Dict[str, float]:
        """
        Obtiene todos los pesos actuales
        
        Returns:
            Dict[str, float]: Diccionario con los pesos de todos los indicadores
        """
        return self.current_weights.copy()
    
    def get_performance_summary(self) -> Dict:
        """
        Obtiene un resumen del rendimiento de los indicadores
        
        Returns:
            Dict: Resumen de rendimiento
        """
        summary = {
            'total_signals': self.performance_data['metadata']['total_signals'],
            'last_updated': self.performance_data['metadata']['last_updated'],
            'indicators': {}
        }
        
        for indicator in self.SUPPORTED_INDICATORS:
            indicator_data = self.performance_data['indicators'][indicator]['overall']
            summary['indicators'][indicator] = {
                'accuracy_rate': indicator_data['accuracy_rate'],
                'profit_factor': indicator_data['profit_factor'],
                'current_weight': self.get_indicator_weight(indicator)
            }
        
        return summary
    
    def reset_performance_data(self):
        """Reinicia los datos de rendimiento a valores por defecto"""
        self.performance_data = self._initialize_performance_data()
        self._calculate_current_weights()
        self.save_performance_data()
        logger.info("Performance data has been reset to default values")
    
    def backtest_weight_adjustment(self, backtest_results: List[Dict]):
        """
        Ajusta los pesos basados en resultados de backtesting
        
        Args:
            backtest_results: Resultados del backtest con señales de indicadores
        """
        logger.info("Adjusting weights based on backtest results...")
        
        for result in backtest_results:
            indicator = result.get('indicator')
            is_correct = result.get('is_correct', False)
            profit_loss = result.get('profit_loss', 0.0)
            market_condition = result.get('market_condition', 'range_low_vol')
            time_interval = result.get('time_interval', '15m')
            
            if indicator in self.SUPPORTED_INDICATORS:
                self.record_signal_result(
                    indicator, is_correct, profit_loss, market_condition, time_interval
                )
        
        logger.info("Weight adjustment from backtest completed")


# Función para generar decisión ponderada basada en señales y pesos
def get_weighted_decision(signals: Dict[str, int], weights: Dict[str, float]) -> Tuple[int, float, Dict]:
    """
    Genera una decisión ponderada basada en las señales de los indicadores y sus pesos
    
    Args:
        signals: Diccionario con señales de indicadores (1=compra, -1=venta, 0=neutral)
        weights: Diccionario con pesos de los indicadores
        
    Returns:
        Tuple[int, float, Dict]: Decisión final, confianza y detalles
    """
    if not signals or not weights:
        return 0, 0.0, {}
    
    total_weight = 0.0
    weighted_sum = 0.0
    used_weights = {}
    
    # Calcular suma ponderada
    for indicator, signal in signals.items():
        if indicator in weights:
            weight = weights[indicator]
            weighted_sum += signal * weight
            total_weight += weight
            used_weights[indicator] = weight
    
    # Si no hay pesos válidos, retornar neutral
    if total_weight == 0:
        return 0, 0.0, {}
    
    # Calcular puntuación normalizada (-1 a 1)
    normalized_score = weighted_sum / total_weight
    
    # Calcular confianza (0 a 1) - qué tan fuerte es la señal
    confidence = abs(normalized_score)
    
    # Determinar decisión final
    if normalized_score > 0.3:  # Umbral para compra
        decision = 1  # Comprar
    elif normalized_score < -0.3:  # Umbral para venta
        decision = -1  # Vender
    else:
        decision = 0  # Neutral
    
    # Preparar detalles
    details = {
        'normalized_score': normalized_score,
        'confidence': confidence,
        'used_weights': used_weights,
        'total_weight': total_weight
    }
    
    return decision, confidence, details