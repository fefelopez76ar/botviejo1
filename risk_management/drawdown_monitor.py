"""
Sistema de monitoreo de drawdown y circuit breakers

Este módulo implementa funciones para monitorear el drawdown en tiempo real
y activar circuit breakers automáticos para detener el trading en condiciones adversas.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Tuple, Optional, Callable
from datetime import datetime, timedelta
import threading

# Configurar logging
logger = logging.getLogger(__name__)

class DrawdownMonitor:
    """Monitor de drawdown en tiempo real"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el monitor de drawdown
        
        Args:
            config: Configuración del monitor
        """
        self.config = config or {
            'warning_threshold': 0.10,  # 10% (alerta)
            'critical_threshold': 0.15,  # 15% (detener trading)
            'sampling_interval': 60,     # segundos entre muestras
            'recovery_threshold': 0.05,  # 5% (recuperación para reanudar)
            'log_file': 'drawdown_log.json'
        }
        
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.drawdown_log = []
        self.stop_trading_callbacks = []
        self.warning_callbacks = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Cargar datos históricos
        self._load_log()
        
        logger.info("Monitor de drawdown inicializado")
    
    def _load_log(self) -> None:
        """Carga el registro histórico de drawdown"""
        log_file = self.config['log_file']
        
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                
                self.peak_equity = data.get('peak_equity', 0.0)
                self.max_drawdown = data.get('max_drawdown', 0.0)
                self.drawdown_log = data.get('log', [])
                
                logger.info(f"Registro histórico de drawdown cargado desde {log_file}")
            except Exception as e:
                logger.error(f"Error al cargar registro de drawdown: {e}")
    
    def _save_log(self) -> None:
        """Guarda el registro de drawdown"""
        try:
            # Truncar log si es muy largo (mantener últimas 1000 entradas)
            if len(self.drawdown_log) > 1000:
                self.drawdown_log = self.drawdown_log[-1000:]
            
            data = {
                'peak_equity': self.peak_equity,
                'current_equity': self.current_equity,
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown,
                'log': self.drawdown_log,
                'last_update': datetime.now().isoformat()
            }
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.config['log_file']) or '.', exist_ok=True)
            
            with open(self.config['log_file'], 'w') as f:
                json.dump(data, f, indent=4)
            
            logger.debug(f"Registro de drawdown guardado en {self.config['log_file']}")
        except Exception as e:
            logger.error(f"Error al guardar registro de drawdown: {e}")
    
    def update_equity(self, current_equity: float) -> float:
        """
        Actualiza el equity actual y calcula drawdown
        
        Args:
            current_equity: Equity actual
            
        Returns:
            float: Drawdown actual (0.0 - 1.0)
        """
        self.current_equity = current_equity
        
        # Actualizar peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.current_drawdown = 0.0
        else:
            # Calcular drawdown actual
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
            
            # Actualizar máximo drawdown
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
                logger.warning(f"Nuevo máximo drawdown: {self.max_drawdown:.2%}")
        
        # Registrar drawdown
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'equity': current_equity,
            'drawdown': self.current_drawdown
        }
        self.drawdown_log.append(log_entry)
        
        # Verificar umbrales
        self._check_thresholds()
        
        # Guardar registro
        self._save_log()
        
        return self.current_drawdown
    
    def _check_thresholds(self) -> None:
        """Verifica si se han superado los umbrales de drawdown"""
        warning_threshold = self.config['warning_threshold']
        critical_threshold = self.config['critical_threshold']
        
        # Verificar umbral de alerta
        if self.current_drawdown >= warning_threshold:
            logger.warning(f"Alerta de drawdown: {self.current_drawdown:.2%} (umbral: {warning_threshold:.2%})")
            
            # Ejecutar callbacks de alerta
            for callback in self.warning_callbacks:
                try:
                    callback(self.current_drawdown)
                except Exception as e:
                    logger.error(f"Error en callback de alerta: {e}")
        
        # Verificar umbral crítico
        if self.current_drawdown >= critical_threshold:
            logger.critical(f"Drawdown crítico: {self.current_drawdown:.2%} (umbral: {critical_threshold:.2%})")
            
            # Ejecutar callbacks de detención
            for callback in self.stop_trading_callbacks:
                try:
                    callback(self.current_drawdown)
                except Exception as e:
                    logger.error(f"Error en callback de detención: {e}")
    
    def add_stop_callback(self, callback: Callable[[float], None]) -> None:
        """
        Añade un callback para ejecutar cuando el drawdown supere el umbral crítico
        
        Args:
            callback: Función a ejecutar con el drawdown actual como argumento
        """
        self.stop_trading_callbacks.append(callback)
    
    def add_warning_callback(self, callback: Callable[[float], None]) -> None:
        """
        Añade un callback para ejecutar cuando el drawdown supere el umbral de alerta
        
        Args:
            callback: Función a ejecutar con el drawdown actual como argumento
        """
        self.warning_callbacks.append(callback)
    
    def start_monitoring(self, get_equity_func: Callable[[], float]) -> None:
        """
        Inicia monitoreo continuo de drawdown en segundo plano
        
        Args:
            get_equity_func: Función para obtener equity actual
        """
        if self.monitoring_active:
            logger.warning("El monitoreo de drawdown ya está activo")
            return
        
        def monitor_loop():
            """Bucle de monitoreo en segundo plano"""
            logger.info("Iniciando monitoreo de drawdown en segundo plano")
            self.monitoring_active = True
            
            try:
                while self.monitoring_active:
                    try:
                        # Obtener equity actual
                        current_equity = get_equity_func()
                        
                        # Actualizar drawdown
                        self.update_equity(current_equity)
                        
                        # Esperar para próxima actualización
                        time.sleep(self.config['sampling_interval'])
                    except Exception as e:
                        logger.error(f"Error en monitoreo de drawdown: {e}")
                        time.sleep(10)  # Esperar antes de reintentar
            finally:
                logger.info("Monitoreo de drawdown detenido")
                self.monitoring_active = False
        
        # Iniciar hilo de monitoreo
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Detiene el monitoreo continuo de drawdown"""
        if not self.monitoring_active:
            logger.warning("El monitoreo de drawdown no está activo")
            return
        
        logger.info("Deteniendo monitoreo de drawdown")
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del drawdown
        
        Returns:
            Dict: Estadísticas de drawdown
        """
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'peak_equity': self.peak_equity,
            'current_equity': self.current_equity,
            'warning_threshold': self.config['warning_threshold'],
            'critical_threshold': self.config['critical_threshold'],
            'monitoring_active': self.monitoring_active
        }
    
    def can_resume_trading(self) -> bool:
        """
        Verifica si se puede reanudar trading después de un drawdown crítico
        
        Returns:
            bool: True si el drawdown ha bajado del umbral de recuperación
        """
        recovery_threshold = self.config.get('recovery_threshold', 0.05)
        return self.current_drawdown <= recovery_threshold

class CircuitBreaker:
    """Implementa un sistema de circuit breaker para detener trading en condiciones adversas"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el sistema de circuit breaker
        
        Args:
            config: Configuración del circuit breaker
        """
        self.config = config or {
            'max_consecutive_losses': 3,
            'max_daily_loss_pct': 0.05,  # 5% del balance inicial diario
            'max_single_loss_pct': 0.02,  # 2% del balance en una operación
            'volatility_threshold': 2.0,  # 2x la volatilidad normal
            'cooldown_period': 6,        # horas de espera para reactivar
            'max_drawdown': 0.15,        # 15% de drawdown máximo
            'log_file': 'circuit_breaker_log.json'
        }
        
        self.breakers_triggered = {}
        self.consecutive_losses = 0
        self.daily_losses = 0.0
        self.initial_daily_balance = 0.0
        self.volatility_ratio = 1.0
        self.active = True
        self.drawdown_monitor = None
        
        # Callbacks cuando se activa un circuit breaker
        self.activation_callbacks = []
        
        # Cargar estado previo
        self._load_state()
        
        logger.info("Sistema de circuit breaker inicializado")
    
    def _load_state(self) -> None:
        """Carga el estado anterior del circuit breaker"""
        log_file = self.config['log_file']
        
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                
                self.breakers_triggered = data.get('breakers_triggered', {})
                self.consecutive_losses = data.get('consecutive_losses', 0)
                self.active = data.get('active', True)
                
                # Convertir strings a datetime
                for key, value in self.breakers_triggered.items():
                    if isinstance(value, str):
                        self.breakers_triggered[key] = datetime.fromisoformat(value)
                
                logger.info(f"Estado del circuit breaker cargado desde {log_file}")
            except Exception as e:
                logger.error(f"Error al cargar estado del circuit breaker: {e}")
    
    def _save_state(self) -> None:
        """Guarda el estado actual del circuit breaker"""
        try:
            # Convertir datetime a strings
            breakers_str = {}
            for key, value in self.breakers_triggered.items():
                if isinstance(value, datetime):
                    breakers_str[key] = value.isoformat()
                else:
                    breakers_str[key] = value
            
            data = {
                'breakers_triggered': breakers_str,
                'consecutive_losses': self.consecutive_losses,
                'active': self.active,
                'last_update': datetime.now().isoformat()
            }
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.config['log_file']) or '.', exist_ok=True)
            
            with open(self.config['log_file'], 'w') as f:
                json.dump(data, f, indent=4)
            
            logger.debug(f"Estado del circuit breaker guardado en {self.config['log_file']}")
        except Exception as e:
            logger.error(f"Error al guardar estado del circuit breaker: {e}")
    
    def connect_drawdown_monitor(self, monitor: DrawdownMonitor) -> None:
        """
        Conecta con un monitor de drawdown para coordinación
        
        Args:
            monitor: Monitor de drawdown
        """
        self.drawdown_monitor = monitor
        
        # Añadir callback para activar circuit breaker cuando drawdown sea crítico
        monitor.add_stop_callback(self._drawdown_callback)
        
        logger.info("Circuit breaker conectado a monitor de drawdown")
    
    def _drawdown_callback(self, drawdown: float) -> None:
        """
        Callback para cuando el drawdown supere el umbral crítico
        
        Args:
            drawdown: Drawdown actual
        """
        self.trigger('drawdown', f"Drawdown crítico ({drawdown:.2%})")
    
    def add_activation_callback(self, callback: Callable[[str, str], None]) -> None:
        """
        Añade un callback para cuando se activa un circuit breaker
        
        Args:
            callback: Función que recibe (tipo, razón)
        """
        self.activation_callbacks.append(callback)
    
    def set_initial_daily_balance(self, balance: float) -> None:
        """
        Establece el balance inicial del día para calcular pérdidas diarias
        
        Args:
            balance: Balance inicial
        """
        self.initial_daily_balance = balance
        self.daily_losses = 0.0
    
    def update_volatility(self, current: float, average: float) -> None:
        """
        Actualiza la relación de volatilidad actual vs. normal
        
        Args:
            current: Volatilidad actual
            average: Volatilidad promedio normal
        """
        if average > 0:
            self.volatility_ratio = current / average
            
            # Verificar si la volatilidad excede el umbral
            if self.volatility_ratio >= self.config['volatility_threshold']:
                self.trigger('volatility', f"Volatilidad excesiva ({self.volatility_ratio:.2f}x)")
    
    def process_trade_result(self, profit: float, balance: float) -> bool:
        """
        Procesa el resultado de una operación para verificar circuit breakers
        
        Args:
            profit: Ganancia/pérdida de la operación
            balance: Balance actual después de la operación
            
        Returns:
            bool: True si el trading puede continuar, False si debe detenerse
        """
        if not self.active:
            return False
        
        # Actualizar pérdidas consecutivas
        if profit < 0:
            self.consecutive_losses += 1
            
            # Verificar si excede el máximo de pérdidas consecutivas
            if self.consecutive_losses >= self.config['max_consecutive_losses']:
                self.trigger('consecutive_losses', f"{self.consecutive_losses} pérdidas consecutivas")
                return False
            
            # Actualizar pérdidas diarias
            self.daily_losses += abs(profit)
            
            # Verificar si excede la pérdida diaria máxima
            if self.initial_daily_balance > 0:
                daily_loss_pct = self.daily_losses / self.initial_daily_balance
                if daily_loss_pct >= self.config['max_daily_loss_pct']:
                    self.trigger('daily_loss', f"Pérdida diaria máxima ({daily_loss_pct:.2%})")
                    return False
            
            # Verificar si excede la pérdida máxima en una operación
            if balance > 0:
                single_loss_pct = abs(profit) / balance
                if single_loss_pct >= self.config['max_single_loss_pct']:
                    self.trigger('single_loss', f"Pérdida excesiva en una operación ({single_loss_pct:.2%})")
                    return False
        else:
            # Reiniciar contador de pérdidas consecutivas
            self.consecutive_losses = 0
        
        # Si llegamos aquí, no se activó ningún circuit breaker
        return True
    
    def trigger(self, breaker_type: str, reason: str) -> None:
        """
        Activa un circuit breaker
        
        Args:
            breaker_type: Tipo de circuit breaker
            reason: Razón de la activación
        """
        logger.warning(f"Circuit breaker activado: {breaker_type} - {reason}")
        
        # Registrar activación
        self.breakers_triggered[breaker_type] = datetime.now()
        self.active = False
        
        # Notificar a callbacks
        for callback in self.activation_callbacks:
            try:
                callback(breaker_type, reason)
            except Exception as e:
                logger.error(f"Error en callback de activación: {e}")
        
        # Guardar estado
        self._save_state()
    
    def can_resume_trading(self) -> Tuple[bool, Optional[str]]:
        """
        Verifica si se puede reanudar el trading
        
        Returns:
            Tuple[bool, Optional[str]]: (puede reanudar, razón si no)
        """
        if self.active:
            return True, None
        
        # Verificar si ha pasado el tiempo de espera para algún breaker
        now = datetime.now()
        cooldown_hours = self.config['cooldown_period']
        
        for breaker_type, trigger_time in list(self.breakers_triggered.items()):
            if isinstance(trigger_time, str):
                trigger_time = datetime.fromisoformat(trigger_time)
                
            if now - trigger_time >= timedelta(hours=cooldown_hours):
                # Este breaker ya cumplió su tiempo de espera
                del self.breakers_triggered[breaker_type]
                logger.info(f"Circuit breaker {breaker_type} liberado después de período de espera")
        
        # Si no quedan breakers activos, reactivar trading
        if not self.breakers_triggered:
            self.active = True
            self.consecutive_losses = 0
            logger.info("Trading reactivado: todos los circuit breakers liberados")
            self._save_state()
            return True, None
        
        # Calcular tiempo restante para el breaker más próximo a liberarse
        if self.breakers_triggered:
            earliest_release = min(self.breakers_triggered.values())
            if isinstance(earliest_release, str):
                earliest_release = datetime.fromisoformat(earliest_release)
                
            time_passed = now - earliest_release
            time_remaining = timedelta(hours=cooldown_hours) - time_passed
            
            if time_remaining.total_seconds() > 0:
                hours_remaining = time_remaining.total_seconds() / 3600
                return False, f"Tiempo restante: {hours_remaining:.1f} horas"
        
        return False, "Circuit breakers activos"
    
    def force_resume(self) -> None:
        """Fuerza la reactivación del trading (uso manual)"""
        if self.active:
            logger.info("El trading ya está activo")
            return
        
        self.breakers_triggered = {}
        self.consecutive_losses = 0
        self.active = True
        
        logger.warning("Trading reactivado forzosamente (manual)")
        self._save_state()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del circuit breaker
        
        Returns:
            Dict: Estado actual
        """
        active_breakers = {}
        
        # Convertir datetime a strings para el reporte
        for breaker_type, trigger_time in self.breakers_triggered.items():
            if isinstance(trigger_time, datetime):
                time_passed = datetime.now() - trigger_time
                cooldown = timedelta(hours=self.config['cooldown_period'])
                time_remaining = max(timedelta(0), cooldown - time_passed)
                
                hours_remaining = time_remaining.total_seconds() / 3600
                active_breakers[breaker_type] = {
                    'triggered_at': trigger_time.isoformat(),
                    'hours_remaining': hours_remaining
                }
        
        return {
            'active': self.active,
            'consecutive_losses': self.consecutive_losses,
            'daily_losses': self.daily_losses,
            'daily_loss_pct': (self.daily_losses / self.initial_daily_balance) if self.initial_daily_balance > 0 else 0,
            'volatility_ratio': self.volatility_ratio,
            'active_breakers': active_breakers
        }