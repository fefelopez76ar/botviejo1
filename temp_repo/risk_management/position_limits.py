"""
Sistema de gestión de límites de posición y control de riesgo

Este módulo proporciona funciones para:
1. Calcular tamaños óptimos de posición basados en riesgo
2. Establecer límites dinámicos según volatilidad del mercado
3. Ajustar tamaños de posición según rendimiento reciente
"""

import os
import json
import logging
import math
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np

# Configurar logging
logger = logging.getLogger(__name__)

class PositionSizeManager:
    """Sistema para calcular y gestionar tamaños de posición óptimos basados en riesgo"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el gestor de tamaños de posición
        
        Args:
            config: Configuración del gestor
        """
        self.config = config or {
            # Límites generales
            'max_position_pct': 0.20,      # 20% del balance como máximo absoluto
            'default_risk_pct': 0.01,      # 1% de riesgo por defecto
            
            # Ajustes dinámicos
            'reduce_after_loss': True,
            'increase_after_win': True,
            'loss_reduction_factor': 0.2,  # Reducir 20% después de pérdida
            'win_increase_factor': 0.1,    # Aumentar 10% después de ganancia
            'max_consecutive_factor': 0.5, # 50% máximo de reducción/aumento 
            
            # Volatilidad y ATR
            'use_volatility_scaling': True,
            'volatility_factor': 1.0,      # Factor de ajuste para volatilidad
            'min_position_pct': 0.001,     # 0.1% mínimo para evitar trades irrelevantes
            
            # Límites por condición de mercado
            'market_condition_limits': {
                'extreme_volatility': 0.01, # 1% en extrema volatilidad
                'strong_downtrend': 0.03,   # 3% en fuerte tendencia bajista
                'normal': 0.05              # 5% en condiciones normales
            },
            
            # Configuración para modo real (más conservador)
            'real_mode_factor': 0.5,       # 50% de los límites de paper trading
            
            # Archivo de configuración
            'config_file': 'position_limits.json'
        }
        
        # Historial de operaciones recientes y sus resultados
        self.recent_trades = []
        
        # Rendimiento de las últimas operaciones (1: ganancia, -1: pérdida)
        self.recent_performance = []
        
        # Volatilidad actual (relativa a la normal)
        self.current_volatility_ratio = 1.0
        
        # Modo actual (paper o real)
        self.trading_mode = 'paper'
        
        # Cargar configuración personalizada
        self._load_config()
        
        logger.info("Gestor de tamaños de posición inicializado")
    
    def _load_config(self) -> None:
        """Carga configuración personalizada desde archivo"""
        config_file = self.config['config_file']
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                
                # Actualizar configuración con valores guardados
                for key, value in data.items():
                    if key in self.config:
                        self.config[key] = value
                
                logger.info(f"Configuración de límites cargada desde {config_file}")
            except Exception as e:
                logger.error(f"Error al cargar configuración de límites: {e}")
    
    def _save_config(self) -> None:
        """Guarda configuración personalizada en archivo"""
        try:
            config_file = self.config['config_file']
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(config_file) or '.', exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            logger.debug(f"Configuración de límites guardada en {config_file}")
        except Exception as e:
            logger.error(f"Error al guardar configuración de límites: {e}")
    
    def set_trading_mode(self, mode: str) -> None:
        """
        Establece el modo de trading (paper o real)
        
        Args:
            mode: Modo de trading ('paper' o 'real')
        """
        if mode not in ['paper', 'real']:
            logger.error(f"Modo de trading inválido: {mode}")
            return
        
        self.trading_mode = mode
        logger.info(f"Modo de trading establecido: {mode}")
    
    def update_volatility(self, current_volatility: float, normal_volatility: float) -> None:
        """
        Actualiza la relación de volatilidad actual
        
        Args:
            current_volatility: Volatilidad actual del mercado
            normal_volatility: Volatilidad normal/promedio
        """
        if normal_volatility > 0:
            self.current_volatility_ratio = current_volatility / normal_volatility
            logger.debug(f"Volatilidad actualizada: {self.current_volatility_ratio:.2f}x normal")
    
    def record_trade_result(self, trade_data: Dict[str, Any]) -> None:
        """
        Registra el resultado de una operación
        
        Args:
            trade_data: Datos de la operación
        """
        # Añadir timestamp
        trade_data['timestamp'] = datetime.now().isoformat()
        
        # Añadir al historial
        self.recent_trades.append(trade_data)
        
        # Mantener solo las últimas 10 operaciones
        if len(self.recent_trades) > 10:
            self.recent_trades = self.recent_trades[-10:]
        
        # Actualizar rendimiento reciente
        profit = trade_data.get('profit', 0)
        self.recent_performance.append(1 if profit > 0 else -1)
        
        # Mantener solo los últimos 10 resultados
        if len(self.recent_performance) > 10:
            self.recent_performance = self.recent_performance[-10:]
        
        logger.debug(f"Resultado de operación registrado: {'ganancia' if profit > 0 else 'pérdida'} de {profit}")
    
    def _calculate_performance_factor(self) -> float:
        """
        Calcula un factor de ajuste basado en rendimiento reciente
        
        Returns:
            float: Factor de ajuste (0.5 - 1.5)
        """
        if not self.recent_performance:
            return 1.0
        
        # Contar operaciones recientes ganadoras y perdedoras
        recent_count = min(5, len(self.recent_performance))
        recent_results = self.recent_performance[-recent_count:]
        
        # Contar operaciones consecutivas del mismo tipo
        consecutive_count = 1
        for i in range(len(recent_results) - 1, 0, -1):
            if recent_results[i] == recent_results[i-1]:
                consecutive_count += 1
            else:
                break
        
        last_result = recent_results[-1]
        
        # Factor base
        factor = 1.0
        
        # Ajustar según configuración
        if last_result == -1 and self.config['reduce_after_loss']:
            # Reducir tamaño después de pérdida
            reduction = min(self.config['max_consecutive_factor'], 
                          consecutive_count * self.config['loss_reduction_factor'])
            factor = 1.0 - reduction
        
        elif last_result == 1 and self.config['increase_after_win']:
            # Aumentar tamaño después de ganancia
            increase = min(self.config['max_consecutive_factor'],
                         consecutive_count * self.config['win_increase_factor'])
            factor = 1.0 + increase
        
        logger.debug(f"Factor de rendimiento: {factor:.2f}")
        return factor
    
    def _calculate_volatility_factor(self) -> float:
        """
        Calcula un factor de ajuste basado en volatilidad
        
        Returns:
            float: Factor de ajuste (0.2 - 1.0)
        """
        if not self.config['use_volatility_scaling']:
            return 1.0
        
        # Más volatilidad = posiciones más pequeñas
        vol_ratio = self.current_volatility_ratio
        
        # Escalar inversamente con la volatilidad
        if vol_ratio <= 1.0:
            # Volatilidad normal o menor
            factor = 1.0
        else:
            # Volatilidad mayor a lo normal
            # Fórmula: factor = 1 / vol_ratio^volatility_factor
            # volatility_factor controla qué tan rápido se reducen los tamaños
            exponent = self.config['volatility_factor']
            factor = 1.0 / (vol_ratio ** exponent)
            
            # Asegurar un mínimo razonable
            factor = max(0.2, factor)
        
        logger.debug(f"Factor de volatilidad: {factor:.2f}")
        return factor
    
    def _get_market_condition_limit(self, market_condition: str) -> float:
        """
        Obtiene el límite de posición para una condición de mercado
        
        Args:
            market_condition: Condición del mercado
            
        Returns:
            float: Límite de posición para esa condición
        """
        limits = self.config['market_condition_limits']
        
        if market_condition in limits:
            return limits[market_condition]
        
        # Si no se especifica, usar el límite para condiciones normales
        return limits.get('normal', 0.05)
    
    def calculate_position_size(self, balance: float, market_condition: str = 'normal',
                              risk_level: float = 1.0) -> float:
        """
        Calcula el tamaño óptimo de posición
        
        Args:
            balance: Balance actual
            market_condition: Condición actual del mercado
            risk_level: Nivel de riesgo manual (0.1 - 2.0)
            
        Returns:
            float: Tamaño de posición recomendado
        """
        # 1. Obtener límite básico según condición de mercado
        base_limit = self._get_market_condition_limit(market_condition)
        
        # 2. Aplicar factor de rendimiento reciente
        performance_factor = self._calculate_performance_factor()
        
        # 3. Aplicar factor de volatilidad
        volatility_factor = self._calculate_volatility_factor()
        
        # 4. Aplicar factor según modo de trading
        mode_factor = self.config['real_mode_factor'] if self.trading_mode == 'real' else 1.0
        
        # 5. Aplicar nivel de riesgo manual (opcional)
        risk_level = max(0.1, min(2.0, risk_level))  # Limitar entre 0.1 y 2.0
        
        # Calcular porcentaje final
        size_pct = base_limit * performance_factor * volatility_factor * mode_factor * risk_level
        
        # Aplicar límites absolutos
        size_pct = max(self.config['min_position_pct'], min(self.config['max_position_pct'], size_pct))
        
        # Calcular valor monetario
        position_size = balance * size_pct
        
        logger.debug(f"Tamaño calculado: {size_pct:.2%} del balance (${position_size:.2f})")
        return position_size
    
    def get_max_positions(self, market_condition: str = 'normal') -> int:
        """
        Obtiene el número máximo de posiciones simultáneas recomendado
        
        Args:
            market_condition: Condición actual del mercado
            
        Returns:
            int: Número máximo de posiciones
        """
        # Lógica básica: más volatilidad o condiciones extremas = menos posiciones
        base_limit = self._get_market_condition_limit(market_condition)
        
        # En modo real ser más conservador
        mode_factor = self.config['real_mode_factor'] if self.trading_mode == 'real' else 1.0
        
        # Calcular máximo según porcentaje y factores
        if market_condition == 'extreme_volatility':
            max_positions = 1
        elif market_condition in ['strong_downtrend', 'strong_uptrend']:
            max_positions = 2
        else:
            # En condiciones normales, permitir más posiciones
            max_positions = int(0.5 / base_limit) if base_limit > 0 else 3
            
            # Aplicar factor de modo
            max_positions = max(1, int(max_positions * mode_factor))
        
        return max_positions
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Actualiza la configuración
        
        Args:
            new_config: Nueva configuración
        """
        # Actualizar solo los campos válidos
        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value
        
        # Guardar configuración
        self._save_config()
        
        logger.info("Configuración de límites actualizada")
    
    def get_recommended_stop_loss(self, entry_price: float, position_size: float, 
                                balance: float, market_condition: str = 'normal') -> float:
        """
        Calcula un stop loss recomendado según riesgo
        
        Args:
            entry_price: Precio de entrada
            position_size: Tamaño de la posición
            balance: Balance total
            market_condition: Condición de mercado
            
        Returns:
            float: Precio de stop loss recomendado
        """
        # Determinar riesgo máximo como porcentaje del balance
        risk_pct = self.config['default_risk_pct']
        
        # Ajustar según condición de mercado
        if market_condition == 'extreme_volatility':
            risk_pct *= 0.5  # Menos riesgo en alta volatilidad
        elif market_condition in ['strong_downtrend', 'moderate_downtrend']:
            risk_pct *= 0.7  # Menos riesgo en tendencia bajista
        
        # Calcular riesgo máximo en términos monetarios
        max_risk = balance * risk_pct
        
        # Calcular distancia del stop loss
        # stop_distance = max_risk / position_size * entry_price
        # Esto evita dividir por cero si position_size es muy pequeño
        if position_size > 0:
            max_loss_pct = max_risk / position_size
            stop_distance = entry_price * max_loss_pct
            
            # Limitar la distancia para evitar stops demasiado ajustados
            min_distance = entry_price * 0.005  # Mínimo 0.5% de distancia
            stop_distance = max(min_distance, stop_distance)
            
            # Calcular precio de stop loss
            stop_price = entry_price - stop_distance
            
            return max(0, stop_price)
        
        # Si no se puede calcular, usar un stop loss por defecto
        return entry_price * 0.95  # 5% por debajo del precio de entrada