"""
Módulo de configuración para el bot de trading de Solana

Proporciona funciones para cargar y guardar configuraciones, así como
parámetros por defecto para el bot.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

# Configurar logging
logger = logging.getLogger(__name__)

# Archivo de configuración por defecto
CONFIG_FILE = "bot_config.json"

# Configuración por defecto del bot
DEFAULT_CONFIG = {
    "general": {
        "symbol": "SOL-USDT",
        "timeframe": "5m",
        "paper_trading": True,
        "strategy": "adaptive_multi",
        "api_key": "",
        "api_secret": "",
        "api_password": "",
        "exchange": "okx"
    },
    "risk": {
        "max_position_size": 0.1,  # % del balance
        "stop_loss_pct": 1.0,  # %
        "take_profit_pct": 2.0,  # %
        "trailing_stop": True,
        "max_loss_per_day": 5.0,  # %
        "max_trades_per_day": 20
    },
    "interface": {
        "lang": "es",
        "theme": "dark",
        "notifications": {
            "enable": True,
            "trade_alerts": True,
            "daily_report": True
        }
    },
    "strategies": {
        "adaptive_multi": {
            "enabled_indicators": [
                "rsi",
                "macd",
                "bollinger_bands",
                "ema_cross",
                "volume_profile"
            ],
            "weight_adjustment": True
        },
        "rsi_scalping": {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "volume_factor": 1.5
        },
        "momentum_scalping": {
            "price_change_pct": 0.2,
            "volume_surge_factor": 2.0,
            "lookback_periods": 3
        }
    }
}

def load_config(config_file: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Carga la configuración desde un archivo
    
    Args:
        config_file: Ruta al archivo de configuración
        
    Returns:
        Dict: Configuración cargada o por defecto si hay error
    """
    try:
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
                logger.info(f"Configuración cargada desde {config_file}")
                return config
        else:
            logger.warning(f"Archivo de configuración {config_file} no encontrado, usando configuración por defecto")
            # Crear archivo con configuración por defecto
            save_config(DEFAULT_CONFIG, config_file)
            return DEFAULT_CONFIG
    except Exception as e:
        logger.error(f"Error cargando configuración: {e}")
        return DEFAULT_CONFIG

def save_config(config: Dict[str, Any], config_file: str = CONFIG_FILE) -> bool:
    """
    Guarda la configuración en un archivo
    
    Args:
        config: Configuración a guardar
        config_file: Ruta al archivo de configuración
        
    Returns:
        bool: True si se guardó correctamente
    """
    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Configuración guardada en {config_file}")
        return True
    except Exception as e:
        logger.error(f"Error guardando configuración: {e}")
        return False

def update_config(section: str, key: str, value: Any, config_file: str = CONFIG_FILE) -> bool:
    """
    Actualiza un valor específico en la configuración
    
    Args:
        section: Sección de la configuración
        key: Clave a actualizar
        value: Nuevo valor
        config_file: Ruta al archivo de configuración
        
    Returns:
        bool: True si se actualizó correctamente
    """
    try:
        config = load_config(config_file)
        
        if section not in config:
            config[section] = {}
        
        config[section][key] = value
        
        return save_config(config, config_file)
    except Exception as e:
        logger.error(f"Error actualizando configuración: {e}")
        return False

def get_system_status() -> Dict[str, Any]:
    """
    Obtiene el estado actual del sistema
    
    Returns:
        Dict: Estado del sistema
    """
    try:
        # Importar módulos requeridos
        from data_management.market_data import get_current_price
        
        # Obtener estado
        status = {
            "active_bots": 0,
            "last_trade_time": "N/A",
            "current_price": 0.0,
            "balance": 0.0,
            "api_connected": False
        }
        
        # Obtener precio actual
        price = get_current_price("SOL-USDT")
        if price:
            status["current_price"] = price
        
        return status
    except Exception as e:
        logger.error(f"Error obteniendo estado del sistema: {e}")
        return {
            "active_bots": 0,
            "last_trade_time": "N/A",
            "current_price": 0.0,
            "balance": 0.0,
            "api_connected": False
        }