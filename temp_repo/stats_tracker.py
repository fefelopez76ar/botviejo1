"""
Sistema de seguimiento de estadísticas de trading y evaluación de rendimiento.

Este módulo proporciona funciones para:
1. Registrar operaciones completadas y en curso
2. Calcular métricas de rendimiento (win rate, drawdown, etc.)
3. Evaluar el cumplimiento de requisitos para trading real
"""

import os
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta

# Configurar logging
logger = logging.getLogger(__name__)

class TradeStats:
    """Clase para seguimiento y análisis de estadísticas de trading"""
    
    def __init__(self, stats_file: str = "trade_stats.json"):
        """
        Inicializa el sistema de seguimiento de estadísticas
        
        Args:
            stats_file: Archivo para guardar estadísticas
        """
        self.stats_file = stats_file
        self.trades = []
        self.daily_results = {}
        self.equity_curve = []
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        
        # Cargar datos existentes
        self._load_stats()
    
    def _load_stats(self) -> None:
        """Carga estadísticas desde archivo"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)
                
                self.trades = data.get('trades', [])
                self.daily_results = data.get('daily_results', {})
                self.equity_curve = data.get('equity_curve', [])
                self.peak_equity = data.get('peak_equity', 0.0)
                self.max_drawdown = data.get('max_drawdown', 0.0)
                
                logger.info(f"Estadísticas cargadas desde {self.stats_file}")
            except Exception as e:
                logger.error(f"Error al cargar estadísticas: {e}")
    
    def _save_stats(self) -> None:
        """Guarda estadísticas en archivo"""
        try:
            data = {
                'trades': self.trades,
                'daily_results': self.daily_results,
                'equity_curve': self.equity_curve,
                'peak_equity': self.peak_equity,
                'max_drawdown': self.max_drawdown,
                'last_update': datetime.now().isoformat()
            }
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.stats_file) or '.', exist_ok=True)
            
            with open(self.stats_file, 'w') as f:
                json.dump(data, f, indent=4)
            
            logger.info(f"Estadísticas guardadas en {self.stats_file}")
        except Exception as e:
            logger.error(f"Error al guardar estadísticas: {e}")
    
    def record_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Registra una operación completada
        
        Args:
            trade_data: Datos de la operación (entrada, salida, profit, etc.)
        """
        # Validar datos mínimos
        required_fields = ['entry_time', 'exit_time', 'entry_price', 'exit_price', 'position_type', 'profit']
        for field in required_fields:
            if field not in trade_data:
                logger.error(f"Falta campo requerido '{field}' en datos de operación")
                return
        
        # Asegurar formato de fechas
        for time_field in ['entry_time', 'exit_time']:
            if isinstance(trade_data[time_field], datetime):
                trade_data[time_field] = trade_data[time_field].isoformat()
        
        # Añadir datos adicionales
        trade_data['id'] = len(self.trades) + 1
        trade_data['recorded_at'] = datetime.now().isoformat()
        
        # Registrar la operación
        self.trades.append(trade_data)
        
        # Actualizar resultados diarios
        exit_date = trade_data['exit_time'].split('T')[0]  # Formato YYYY-MM-DD
        if exit_date not in self.daily_results:
            self.daily_results[exit_date] = 0.0
        
        self.daily_results[exit_date] += trade_data['profit']
        
        # Actualizar equity curve
        current_equity = self.equity_curve[-1] if self.equity_curve else trade_data.get('initial_balance', 10000.0)
        new_equity = current_equity + trade_data['profit']
        self.equity_curve.append(new_equity)
        
        # Actualizar peak equity y drawdown
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
        else:
            current_drawdown = (self.peak_equity - new_equity) / self.peak_equity if self.peak_equity > 0 else 0
            self.current_drawdown = current_drawdown
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
        
        # Guardar datos actualizados
        self._save_stats()
        
        logger.info(f"Operación #{trade_data['id']} registrada: {trade_data['position_type']}, "
                   f"Profit: {trade_data['profit']:.2f}")
    
    def update_equity(self, current_equity: float) -> None:
        """
        Actualiza el equity actual sin registrar una operación
        
        Args:
            current_equity: Equity actual
        """
        # Añadir al equity curve
        self.equity_curve.append(current_equity)
        
        # Actualizar peak equity y drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        else:
            current_drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
            self.current_drawdown = current_drawdown
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
        
        # Guardar datos actualizados
        self._save_stats()
    
    def get_win_rate(self) -> float:
        """
        Calcula el win rate (porcentaje de operaciones ganadoras)
        
        Returns:
            float: Win rate (0.0 - 1.0)
        """
        if not self.trades:
            return 0.0
        
        winning_trades = sum(1 for trade in self.trades if trade.get('profit', 0) > 0)
        return winning_trades / len(self.trades)
    
    def get_consecutive_positive_days(self) -> int:
        """
        Calcula el número máximo de días consecutivos con resultados positivos
        
        Returns:
            int: Número máximo de días positivos consecutivos
        """
        if not self.daily_results:
            return 0
        
        # Ordenar fechas
        dates = sorted(self.daily_results.keys())
        
        # Buscar secuencia más larga de días positivos consecutivos
        max_consecutive = 0
        current_consecutive = 0
        
        for date in dates:
            if self.daily_results[date] > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def get_profit_factor(self) -> float:
        """
        Calcula el factor de rentabilidad (ganancia bruta / pérdida bruta)
        
        Returns:
            float: Factor de rentabilidad
        """
        if not self.trades:
            return 0.0
        
        gross_profit = sum(trade.get('profit', 0) for trade in self.trades if trade.get('profit', 0) > 0)
        gross_loss = abs(sum(trade.get('profit', 0) for trade in self.trades if trade.get('profit', 0) < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def get_max_drawdown(self) -> float:
        """
        Obtiene el drawdown máximo
        
        Returns:
            float: Drawdown máximo (0.0 - 1.0)
        """
        return self.max_drawdown
    
    def get_current_drawdown(self) -> float:
        """
        Obtiene el drawdown actual
        
        Returns:
            float: Drawdown actual (0.0 - 1.0)
        """
        return self.current_drawdown
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de las estadísticas de trading
        
        Returns:
            Dict: Resumen de estadísticas
        """
        return {
            'total_trades': len(self.trades),
            'win_rate': self.get_win_rate(),
            'profit_factor': self.get_profit_factor(),
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'consecutive_positive_days': self.get_consecutive_positive_days(),
            'total_profit': sum(trade.get('profit', 0) for trade in self.trades),
            'average_profit': sum(trade.get('profit', 0) for trade in self.trades) / len(self.trades) if self.trades else 0
        }
    
    def meets_real_trading_requirements(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Verifica si se cumplen los requisitos para trading real
        
        Returns:
            Tuple[bool, Dict]: (cumple requisitos, detalles)
        """
        # Definir requisitos (pueden ajustarse según necesidades)
        MIN_TRADES = 100
        MIN_WIN_RATE = 0.55  # 55%
        MIN_CONSECUTIVE_DAYS = 7
        MAX_ALLOWED_DRAWDOWN = 0.15  # 15%
        
        # Obtener métricas actuales
        total_trades = len(self.trades)
        win_rate = self.get_win_rate()
        consecutive_days = self.get_consecutive_positive_days()
        max_drawdown = self.max_drawdown
        
        # Verificar cada requisito
        requirements = {
            'min_trades': {
                'required': MIN_TRADES,
                'actual': total_trades,
                'passed': total_trades >= MIN_TRADES
            },
            'min_win_rate': {
                'required': MIN_WIN_RATE,
                'actual': win_rate,
                'passed': win_rate >= MIN_WIN_RATE
            },
            'min_consecutive_days': {
                'required': MIN_CONSECUTIVE_DAYS,
                'actual': consecutive_days,
                'passed': consecutive_days >= MIN_CONSECUTIVE_DAYS
            },
            'max_drawdown': {
                'required': MAX_ALLOWED_DRAWDOWN,
                'actual': max_drawdown,
                'passed': max_drawdown <= MAX_ALLOWED_DRAWDOWN
            }
        }
        
        # Verificar si se cumplen todos los requisitos
        all_passed = all(req['passed'] for req in requirements.values())
        
        return all_passed, requirements

# Clase para implementar circuit breakers
class CircuitBreakers:
    """Implementa mecanismos de seguridad para detener trading en condiciones adversas"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el sistema de circuit breakers
        
        Args:
            config: Configuración de circuit breakers
        """
        self.config = config or {
            'max_daily_loss': 0.05,  # 5% del balance
            'max_consecutive_losses': 3,
            'max_drawdown': 0.15,  # 15%
            'extreme_volatility': 3.0,  # 3x la volatilidad normal
            'cooldown_period': 24  # horas
        }
        
        self.breakers_triggered = {}
        self.consecutive_losses = 0
        
        logger.info("Sistema de circuit breakers inicializado")
    
    def check_daily_loss(self, daily_loss: float, balance: float) -> bool:
        """
        Verifica si se ha excedido la pérdida diaria máxima
        
        Args:
            daily_loss: Pérdida diaria acumulada (valor positivo)
            balance: Balance actual
            
        Returns:
            bool: True si se debe detener trading
        """
        max_loss = balance * self.config['max_daily_loss']
        
        if daily_loss >= max_loss:
            self._trigger_breaker('daily_loss')
            logger.warning(f"Circuit breaker activado: Pérdida diaria máxima excedida ({daily_loss:.2f} > {max_loss:.2f})")
            return True
        
        return False
    
    def check_consecutive_losses(self, trade_profit: float) -> bool:
        """
        Verifica si hay demasiadas pérdidas consecutivas
        
        Args:
            trade_profit: Ganancia/pérdida de la última operación
            
        Returns:
            bool: True si se debe detener trading
        """
        if trade_profit < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        if self.consecutive_losses >= self.config['max_consecutive_losses']:
            self._trigger_breaker('consecutive_losses')
            logger.warning(f"Circuit breaker activado: Máximo de pérdidas consecutivas excedido ({self.consecutive_losses})")
            return True
        
        return False
    
    def check_drawdown(self, current_drawdown: float) -> bool:
        """
        Verifica si el drawdown actual ha excedido el límite
        
        Args:
            current_drawdown: Drawdown actual (0.0 - 1.0)
            
        Returns:
            bool: True si se debe detener trading
        """
        if current_drawdown >= self.config['max_drawdown']:
            self._trigger_breaker('drawdown')
            logger.warning(f"Circuit breaker activado: Drawdown máximo excedido ({current_drawdown:.2%})")
            return True
        
        return False
    
    def check_volatility(self, current_volatility: float, avg_volatility: float) -> bool:
        """
        Verifica si la volatilidad actual excede los límites seguros
        
        Args:
            current_volatility: Volatilidad actual
            avg_volatility: Volatilidad promedio histórica
            
        Returns:
            bool: True si se debe detener trading
        """
        if current_volatility >= avg_volatility * self.config['extreme_volatility']:
            self._trigger_breaker('volatility')
            logger.warning(f"Circuit breaker activado: Volatilidad extrema ({current_volatility:.2f} > {avg_volatility * self.config['extreme_volatility']:.2f})")
            return True
        
        return False
    
    def _trigger_breaker(self, breaker_type: str) -> None:
        """
        Registra la activación de un circuit breaker
        
        Args:
            breaker_type: Tipo de circuit breaker
        """
        self.breakers_triggered[breaker_type] = datetime.now().isoformat()
    
    def can_resume_trading(self, breaker_type: str) -> bool:
        """
        Verifica si ya se puede reanudar trading después de un circuit breaker
        
        Args:
            breaker_type: Tipo de circuit breaker
            
        Returns:
            bool: True si se puede reanudar trading
        """
        if breaker_type not in self.breakers_triggered:
            return True
        
        trigger_time = datetime.fromisoformat(self.breakers_triggered[breaker_type])
        cooldown_hours = self.config['cooldown_period']
        
        if datetime.now() - trigger_time >= timedelta(hours=cooldown_hours):
            del self.breakers_triggered[breaker_type]
            logger.info(f"Circuit breaker desactivado: {breaker_type}")
            return True
        
        return False
    
    def reset_breaker(self, breaker_type: str) -> None:
        """
        Reinicia manualmente un circuit breaker
        
        Args:
            breaker_type: Tipo de circuit breaker
        """
        if breaker_type in self.breakers_triggered:
            del self.breakers_triggered[breaker_type]
            logger.info(f"Circuit breaker reiniciado manualmente: {breaker_type}")
    
    def is_trading_allowed(self) -> Tuple[bool, Optional[str]]:
        """
        Verifica si el trading está permitido (ningún circuit breaker activo)
        
        Returns:
            Tuple[bool, Optional[str]]: (trading permitido, razón si no está permitido)
        """
        if not self.breakers_triggered:
            return True, None
        
        # Encontrar el circuit breaker más reciente
        latest_breaker = max(self.breakers_triggered.items(), key=lambda x: x[1])
        
        trigger_time = datetime.fromisoformat(latest_breaker[1])
        cooldown_hours = self.config['cooldown_period']
        remaining_hours = cooldown_hours - (datetime.now() - trigger_time).total_seconds() / 3600
        
        if remaining_hours <= 0:
            del self.breakers_triggered[latest_breaker[0]]
            return self.is_trading_allowed()
        
        return False, f"Circuit breaker activo: {latest_breaker[0]}, tiempo restante: {remaining_hours:.1f}h"

# Clase para gestión de límites de posiciones
class PositionSizer:
    """Determina el tamaño apropiado de posiciones basado en riesgo y volatilidad"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el sistema de sizing de posiciones
        
        Args:
            config: Configuración de tamaños de posición
        """
        self.config = config or {
            'max_risk_per_trade': 0.01,  # 1% del balance
            'max_open_positions': 3,
            'volatility_scaling': True,
            'reduce_size_after_loss': True,
            'increase_size_after_win': True,
            'absolute_position_limit': 0.2  # 20% del balance como máximo
        }
        
        self.recent_results = []  # Lista de resultados recientes (1: ganancia, -1: pérdida)
        
        logger.info("Sistema de gestión de tamaño de posiciones inicializado")
    
    def calculate_position_size(self, balance: float, volatility: float = 1.0, 
                               risk_multiplier: float = 1.0) -> float:
        """
        Calcula el tamaño de posición recomendado según balance y volatilidad
        
        Args:
            balance: Balance actual
            volatility: Índice de volatilidad (1.0 = normal)
            risk_multiplier: Multiplicador de riesgo manual
            
        Returns:
            float: Tamaño de posición recomendado
        """
        # Tamaño base según riesgo por operación
        base_size = balance * self.config['max_risk_per_trade'] * risk_multiplier
        
        # Ajustar según volatilidad si está habilitado
        if self.config['volatility_scaling']:
            # Reducir tamaño cuando aumenta la volatilidad
            volatility_factor = 1 / max(0.2, volatility)
            base_size *= volatility_factor
        
        # Ajustar según resultados recientes si está habilitado
        if self.recent_results:
            if self.config['reduce_size_after_loss'] and self.recent_results[-1] == -1:
                # Reducir tamaño después de pérdida
                consecutive_losses = 0
                for result in reversed(self.recent_results):
                    if result == -1:
                        consecutive_losses += 1
                    else:
                        break
                
                # Reducir más en función de pérdidas consecutivas
                reduction_factor = max(0.5, 1 - (consecutive_losses * 0.1))
                base_size *= reduction_factor
            
            elif self.config['increase_size_after_win'] and self.recent_results[-1] == 1:
                # Aumentar tamaño después de ganancia
                consecutive_wins = 0
                for result in reversed(self.recent_results):
                    if result == 1:
                        consecutive_wins += 1
                    else:
                        break
                
                # Aumentar gradualmente con límite
                increase_factor = min(1.5, 1 + (consecutive_wins * 0.05))
                base_size *= increase_factor
        
        # Aplicar límite absoluto
        max_allowed = balance * self.config['absolute_position_limit']
        final_size = min(base_size, max_allowed)
        
        return final_size
    
    def update_recent_results(self, trade_profit: float) -> None:
        """
        Actualiza los resultados recientes
        
        Args:
            trade_profit: Ganancia/pérdida de la última operación
        """
        # Añadir resultado (1: ganancia, -1: pérdida)
        self.recent_results.append(1 if trade_profit > 0 else -1)
        
        # Mantener solo los últimos 10 resultados
        if len(self.recent_results) > 10:
            self.recent_results = self.recent_results[-10:]
    
    def get_open_position_limit(self) -> int:
        """
        Obtiene el límite de posiciones abiertas simultáneas
        
        Returns:
            int: Número máximo de posiciones abiertas permitidas
        """
        return self.config['max_open_positions']