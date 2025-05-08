"""
Módulo para gestión de configuración del bot de trading
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("ConfigManager")

# Rutas predeterminadas
CONFIG_FILE = "config.env"
CONFIG_BACKUP = "config.env.bak"

def load_config() -> Dict[str, Any]:
    """
    Carga la configuración del sistema
    
    Returns:
        Dict: Configuración cargada
    """
    config = {}
    
    # Cargar configuración desde archivo
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        config[key.strip()] = value.strip()
            
            logger.info("Configuración cargada desde archivo")
        else:
            logger.warning(f"Archivo de configuración {CONFIG_FILE} no encontrado")
            # Crear configuración predeterminada
            config = get_default_config()
            save_config(config)
    except Exception as e:
        logger.error(f"Error al cargar configuración: {e}")
        config = get_default_config()
    
    # Sobrescribir con variables de entorno
    for key in config.keys():
        env_value = os.environ.get(key)
        if env_value is not None:
            config[key] = env_value
    
    return config

def save_config(config: Dict[str, Any]) -> bool:
    """
    Guarda la configuración del sistema
    
    Args:
        config: Configuración a guardar
        
    Returns:
        bool: True si se guardó correctamente, False en caso contrario
    """
    try:
        # Crear copia de seguridad si existe el archivo
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as src, open(CONFIG_BACKUP, "w") as dst:
                    dst.write(src.read())
                logger.info(f"Copia de seguridad creada en {CONFIG_BACKUP}")
            except Exception as e:
                logger.warning(f"Error al crear copia de seguridad: {e}")
        
        # Guardar nueva configuración
        with open(CONFIG_FILE, "w") as f:
            for key, value in config.items():
                f.write(f"{key}={value}\n")
        
        logger.info("Configuración guardada correctamente")
        return True
    except Exception as e:
        logger.error(f"Error al guardar configuración: {e}")
        return False

def get_default_config() -> Dict[str, Any]:
    """
    Obtiene la configuración predeterminada
    
    Returns:
        Dict: Configuración predeterminada
    """
    return {
        # Configuración del exchange
        "EXCHANGE": "okx",
        "API_KEY": "",
        "API_SECRET": "",
        "API_PASSPHRASE": "",
        
        # Configuración general
        "DEFAULT_SYMBOL": "SOL-USDT",
        "DEFAULT_INTERVAL": "15m",
        "DEFAULT_PAPER_TRADING": "True",
        
        # Configuración de Telegram
        "TELEGRAM_TOKEN": "",
        "TELEGRAM_CHAT_ID": "",
        "TELEGRAM_NOTIFICATIONS": "False",
        
        # Configuración del sistema
        "LOG_LEVEL": "INFO",
        "DATA_RETENTION_DAYS": "30",
        "MAX_ACTIVE_BOTS": "5"
    }

def get_config_value(key: str, default: Any = None) -> Any:
    """
    Obtiene un valor específico de la configuración
    
    Args:
        key: Clave a obtener
        default: Valor predeterminado si no existe
        
    Returns:
        Any: Valor de configuración
    """
    config = load_config()
    return config.get(key, default)

def set_config_value(key: str, value: Any) -> bool:
    """
    Establece un valor específico en la configuración
    
    Args:
        key: Clave a establecer
        value: Valor a establecer
        
    Returns:
        bool: True si se estableció correctamente, False en caso contrario
    """
    config = load_config()
    config[key] = value
    return save_config(config)

def merge_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combina dos configuraciones, priorizando la segunda
    
    Args:
        config1: Primera configuración
        config2: Segunda configuración (prioritaria)
        
    Returns:
        Dict: Configuración combinada
    """
    result = config1.copy()
    result.update(config2)
    return result

def validate_config(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Valida la configuración y retorna errores
    
    Args:
        config: Configuración a validar
        
    Returns:
        Dict: Diccionario de errores (vacío si no hay errores)
    """
    errors = {}
    
    # Validar exchange
    if config.get("EXCHANGE") not in ["okx", "binance", "kucoin", "bybit"]:
        errors["EXCHANGE"] = "Exchange no válido"
    
    # Validar API key si no es paper trading
    if config.get("DEFAULT_PAPER_TRADING", "True").lower() != "true":
        if not config.get("API_KEY"):
            errors["API_KEY"] = "API Key requerida para trading real"
        if not config.get("API_SECRET"):
            errors["API_SECRET"] = "API Secret requerida para trading real"
    
    # Validar notificaciones Telegram
    if config.get("TELEGRAM_NOTIFICATIONS", "False").lower() == "true":
        if not config.get("TELEGRAM_TOKEN"):
            errors["TELEGRAM_TOKEN"] = "Token de Telegram requerido para notificaciones"
        if not config.get("TELEGRAM_CHAT_ID"):
            errors["TELEGRAM_CHAT_ID"] = "Chat ID de Telegram requerido para notificaciones"
    
    return errors