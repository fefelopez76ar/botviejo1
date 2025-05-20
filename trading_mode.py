"""
Módulo para gestionar modos de trading (paper/real)

Este módulo contiene funciones para verificar requisitos
y configurar el modo de trading (simulación o real)
"""

import os
import logging
import json
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

# Configurar logging
logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """Modos de trading disponibles"""
    PAPER = "paper"  # Trading en papel (simulación)
    REAL = "real"    # Trading real (operaciones reales)

class SecurityRequirements:
    """Requisitos de seguridad para habilitar trading real"""
    
    # Rendimiento mínimo requerido
    MIN_WIN_RATE = 0.55  # 55% de trades ganadores
    MIN_CONSECUTIVE_POSITIVE_DAYS = 7
    MAX_ALLOWED_DRAWDOWN = 0.15  # 15% drawdown máximo
    
    # Otros requisitos
    MIN_BACKTEST_DAYS = 30
    MIN_PAPER_TRADES = 100
    
    @staticmethod
    def verify_win_rate(trades: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Verifica que el win rate cumpla el requisito mínimo
        
        Args:
            trades: Lista de operaciones
            
        Returns:
            Tuple[bool, str]: (cumple requisito, mensaje)
        """
        if not trades:
            return False, "No hay operaciones para calcular win rate"
        
        winning_trades = sum(1 for t in trades if t.get('profit', 0) > 0)
        win_rate = winning_trades / len(trades)
        
        if win_rate >= SecurityRequirements.MIN_WIN_RATE:
            return True, f"Win rate: {win_rate:.2%} (mínimo: {SecurityRequirements.MIN_WIN_RATE:.2%})"
        else:
            return False, f"Win rate insuficiente: {win_rate:.2%} (mínimo: {SecurityRequirements.MIN_WIN_RATE:.2%})"
    
    @staticmethod
    def verify_consecutive_positive_days(daily_results: Dict[str, float]) -> Tuple[bool, str]:
        """
        Verifica que haya suficientes días consecutivos con rendimiento positivo
        
        Args:
            daily_results: Diccionario con fechas y rendimientos
            
        Returns:
            Tuple[bool, str]: (cumple requisito, mensaje)
        """
        if not daily_results:
            return False, "No hay datos diarios para verificar"
        
        # Ordenar fechas
        dates = sorted(daily_results.keys())
        
        # Buscar secuencia más larga de días positivos consecutivos
        max_consecutive = 0
        current_consecutive = 0
        
        for date in dates:
            if daily_results[date] > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        if max_consecutive >= SecurityRequirements.MIN_CONSECUTIVE_POSITIVE_DAYS:
            return True, f"Días positivos consecutivos: {max_consecutive} (mínimo: {SecurityRequirements.MIN_CONSECUTIVE_POSITIVE_DAYS})"
        else:
            return False, f"Días positivos consecutivos insuficientes: {max_consecutive} (mínimo: {SecurityRequirements.MIN_CONSECUTIVE_POSITIVE_DAYS})"
    
    @staticmethod
    def verify_max_drawdown(equity_curve: List[float]) -> Tuple[bool, str]:
        """
        Verifica que el drawdown máximo no exceda el límite
        
        Args:
            equity_curve: Lista de valores de equity
            
        Returns:
            Tuple[bool, str]: (cumple requisito, mensaje)
        """
        if not equity_curve:
            return False, "No hay datos de equity para calcular drawdown"
        
        # Calcular drawdown máximo
        max_dd = 0
        peak = equity_curve[0]
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            else:
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
        
        if max_dd <= SecurityRequirements.MAX_ALLOWED_DRAWDOWN:
            return True, f"Drawdown máximo: {max_dd:.2%} (máximo permitido: {SecurityRequirements.MAX_ALLOWED_DRAWDOWN:.2%})"
        else:
            return False, f"Drawdown máximo excesivo: {max_dd:.2%} (máximo permitido: {SecurityRequirements.MAX_ALLOWED_DRAWDOWN:.2%})"
    
    @staticmethod
    def verify_min_trades(trades: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Verifica que haya suficientes operaciones
        
        Args:
            trades: Lista de operaciones
            
        Returns:
            Tuple[bool, str]: (cumple requisito, mensaje)
        """
        num_trades = len(trades)
        
        if num_trades >= SecurityRequirements.MIN_PAPER_TRADES:
            return True, f"Número de operaciones: {num_trades} (mínimo: {SecurityRequirements.MIN_PAPER_TRADES})"
        else:
            return False, f"Número de operaciones insuficiente: {num_trades} (mínimo: {SecurityRequirements.MIN_PAPER_TRADES})"
    
    @staticmethod
    def verify_all_requirements(performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifica todos los requisitos de seguridad
        
        Args:
            performance_data: Datos de rendimiento del bot
            
        Returns:
            Dict: Resultados de verificación
        """
        trades = performance_data.get('trades', [])
        daily_results = performance_data.get('daily_results', {})
        equity_curve = performance_data.get('equity_curve', [])
        
        # Verificar cada requisito
        win_rate_ok, win_rate_msg = SecurityRequirements.verify_win_rate(trades)
        consecutive_days_ok, consecutive_days_msg = SecurityRequirements.verify_consecutive_positive_days(daily_results)
        drawdown_ok, drawdown_msg = SecurityRequirements.verify_max_drawdown(equity_curve)
        trades_ok, trades_msg = SecurityRequirements.verify_min_trades(trades)
        
        # Resultado general
        all_requirements_met = all([win_rate_ok, consecutive_days_ok, drawdown_ok, trades_ok])
        
        return {
            'all_requirements_met': all_requirements_met,
            'requirements': {
                'win_rate': {
                    'passed': win_rate_ok,
                    'message': win_rate_msg
                },
                'consecutive_positive_days': {
                    'passed': consecutive_days_ok,
                    'message': consecutive_days_msg
                },
                'max_drawdown': {
                    'passed': drawdown_ok,
                    'message': drawdown_msg
                },
                'min_trades': {
                    'passed': trades_ok,
                    'message': trades_msg
                }
            }
        }

def verify_api_credentials(api_key: str, api_secret: str, passphrase: Optional[str] = None) -> Tuple[bool, str]:
    """
    Verifica que las credenciales API sean válidas
    
    Args:
        api_key: Clave API
        api_secret: Secreto API
        passphrase: Frase de contraseña (opcional, requerida por algunos exchanges)
        
    Returns:
        Tuple[bool, str]: (credenciales válidas, mensaje)
    """
    try:
        import ccxt
        
        # Crear cliente de exchange
        exchange = ccxt.okx({
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase,  # OKX usa 'password' en lugar de 'passphrase'
            'enableRateLimit': True
        })
        
        # Intentar obtener balance para verificar credenciales
        exchange.fetch_balance()
        
        logger.info("Credenciales API verificadas correctamente")
        return True, "Credenciales API válidas"
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error al verificar credenciales API: {error_msg}")
        
        if "auth" in error_msg.lower() or "key" in error_msg.lower():
            return False, "Credenciales API inválidas. Verifica tu API key, secret y passphrase."
        else:
            return False, f"Error al conectar con el exchange: {error_msg}"

def set_trading_mode(mode: TradingMode, config_file: str, api_credentials: Dict[str, str], 
                    performance_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Configura el modo de trading (paper o real)
    
    Args:
        mode: Modo de trading a configurar
        config_file: Archivo de configuración
        api_credentials: Credenciales API
        performance_data: Datos de rendimiento para verificar requisitos
        
    Returns:
        Tuple[bool, str]: (cambio exitoso, mensaje)
    """
    try:
        # Si se intenta cambiar a modo real, verificar requisitos
        if mode == TradingMode.REAL:
            # 1. Verificar credenciales API
            creds_valid, creds_msg = verify_api_credentials(
                api_credentials.get('api_key', ''),
                api_credentials.get('api_secret', ''),
                api_credentials.get('passphrase', '')
            )
            
            if not creds_valid:
                return False, f"No se pudo cambiar a modo real: {creds_msg}"
            
            # 2. Verificar requisitos de rendimiento
            req_result = SecurityRequirements.verify_all_requirements(performance_data)
            
            if not req_result['all_requirements_met']:
                failed_reqs = [req for req, data in req_result['requirements'].items() if not data['passed']]
                return False, f"No se cumplen los requisitos para trading real: {', '.join(failed_reqs)}"
            
            # 3. Actualizar la configuración
            try:
                # Cargar configuración actual
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Actualizar modo
                config['trading_mode'] = mode.value
                
                # Guardar configuración actualizada
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=4)
                
                logger.info(f"Modo de trading cambiado a: {mode.value}")
                return True, f"Modo de trading cambiado a: {mode.value}"
            
            except Exception as e:
                logger.error(f"Error al actualizar configuración: {e}")
                return False, f"Error al actualizar configuración: {e}"
        
        # Cambio a modo paper (siempre permitido)
        else:
            try:
                # Cargar configuración actual
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Actualizar modo
                config['trading_mode'] = mode.value
                
                # Guardar configuración actualizada
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=4)
                
                logger.info(f"Modo de trading cambiado a: {mode.value}")
                return True, f"Modo de trading cambiado a: {mode.value}"
            
            except Exception as e:
                logger.error(f"Error al actualizar configuración: {e}")
                return False, f"Error al actualizar configuración: {e}"
    
    except Exception as e:
        logger.error(f"Error al cambiar modo de trading: {e}")
        return False, f"Error al cambiar modo de trading: {e}"

def get_current_mode(config_file: str) -> TradingMode:
    """
    Obtiene el modo de trading actual
    
    Args:
        config_file: Archivo de configuración
        
    Returns:
        TradingMode: Modo actual
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        mode_str = config.get('trading_mode', 'paper')
        
        if mode_str == 'real':
            return TradingMode.REAL
        else:
            return TradingMode.PAPER
    
    except Exception as e:
        logger.error(f"Error al obtener modo de trading: {e}")
        return TradingMode.PAPER  # En caso de error, asumimos modo paper por seguridad