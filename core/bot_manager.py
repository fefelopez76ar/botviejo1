"""
Módulo para gestionar múltiples bots de trading

Este módulo proporciona funcionalidades para crear, iniciar,
detener y monitorear múltiples bots de trading.
"""

import os
import time
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configurar logging
logger = logging.getLogger(__name__)

class BotManager:
    """Gestor de múltiples bots de trading"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el gestor de bots
        
        Args:
            config: Configuración general
        """
        self.config = config
        self.bots = {}
        self.active_bots = set()
        self.bot_statuses = {}
        
        logger.info("Bot Manager inicializado")
    
    def create_bot(self, bot_config: Dict[str, Any]) -> str:
        """
        Crea un nuevo bot
        
        Args:
            bot_config: Configuración del bot
            
        Returns:
            str: ID del bot creado
        """
        try:
            # Generar ID único para el bot
            bot_id = f"bot_{int(time.time())}_{len(self.bots) + 1}"
            
            # Guardar configuración
            self.bots[bot_id] = bot_config
            self.bot_statuses[bot_id] = {
                "active": False,
                "last_update": datetime.now().isoformat(),
                "status": "CREATED",
                "error": None
            }
            
            logger.info(f"Bot creado: {bot_id}")
            return bot_id
            
        except Exception as e:
            logger.error(f"Error al crear bot: {e}")
            raise
    
    def start_bot(self, bot_id: str) -> bool:
        """
        Inicia un bot existente
        
        Args:
            bot_id: ID del bot a iniciar
            
        Returns:
            bool: True si se inició correctamente
        """
        try:
            if bot_id not in self.bots:
                logger.error(f"Bot no encontrado: {bot_id}")
                return False
            
            if bot_id in self.active_bots:
                logger.warning(f"Bot ya está activo: {bot_id}")
                return True
            
            # Obtener configuración del bot
            bot_config = self.bots[bot_id]
            
            # Crear instancia del bot
            from trading_bot import TradingBot
            bot = TradingBot(bot_config)
            
            # Iniciar el bot en un hilo separado
            import threading
            bot_thread = threading.Thread(target=bot.run, daemon=True)
            bot_thread.start()
            
            # Marcar como activo
            self.active_bots.add(bot_id)
            self.bot_statuses[bot_id] = {
                "active": True,
                "last_update": datetime.now().isoformat(),
                "status": "RUNNING",
                "error": None,
                "thread": bot_thread
            }
            
            logger.info(f"Bot iniciado: {bot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error al iniciar bot {bot_id}: {e}")
            self.bot_statuses[bot_id] = {
                "active": False,
                "last_update": datetime.now().isoformat(),
                "status": "ERROR",
                "error": str(e)
            }
            return False
    
    def stop_bot(self, bot_id: str) -> bool:
        """
        Detiene un bot en ejecución
        
        Args:
            bot_id: ID del bot a detener
            
        Returns:
            bool: True si se detuvo correctamente
        """
        try:
            if bot_id not in self.bots:
                logger.error(f"Bot no encontrado: {bot_id}")
                return False
            
            if bot_id not in self.active_bots:
                logger.warning(f"Bot no está activo: {bot_id}")
                return True
            
            # Detener el bot
            # Nota: En una implementación real, se enviaría una señal al bot para detenerlo
            self.active_bots.remove(bot_id)
            self.bot_statuses[bot_id] = {
                "active": False,
                "last_update": datetime.now().isoformat(),
                "status": "STOPPED",
                "error": None
            }
            
            logger.info(f"Bot detenido: {bot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error al detener bot {bot_id}: {e}")
            return False
    
    def update_bot(self, bot_id: str, new_config: Dict[str, Any]) -> bool:
        """
        Actualiza la configuración de un bot
        
        Args:
            bot_id: ID del bot a actualizar
            new_config: Nueva configuración
            
        Returns:
            bool: True si se actualizó correctamente
        """
        try:
            if bot_id not in self.bots:
                logger.error(f"Bot no encontrado: {bot_id}")
                return False
            
            # Si el bot está activo, detenerlo primero
            was_active = bot_id in self.active_bots
            if was_active:
                self.stop_bot(bot_id)
            
            # Actualizar configuración
            self.bots[bot_id] = new_config
            
            # Reiniciar si estaba activo
            if was_active:
                self.start_bot(bot_id)
            
            logger.info(f"Bot actualizado: {bot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error al actualizar bot {bot_id}: {e}")
            return False
    
    def get_bot_status(self, bot_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado actual de un bot
        
        Args:
            bot_id: ID del bot
            
        Returns:
            Dict: Estado del bot
        """
        if bot_id not in self.bots:
            return {
                "active": False,
                "status": "NOT_FOUND",
                "error": "Bot no encontrado"
            }
        
        return self.bot_statuses.get(bot_id, {
            "active": False,
            "status": "UNKNOWN",
            "error": None
        })
    
    def get_all_bots(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene información de todos los bots
        
        Returns:
            Dict: Diccionario de bots y sus configuraciones
        """
        result = {}
        for bot_id, config in self.bots.items():
            status = self.get_bot_status(bot_id)
            result[bot_id] = {
                "config": config,
                "status": status
            }
        
        return result
    
    def delete_bot(self, bot_id: str) -> bool:
        """
        Elimina un bot
        
        Args:
            bot_id: ID del bot a eliminar
            
        Returns:
            bool: True si se eliminó correctamente
        """
        try:
            if bot_id not in self.bots:
                logger.error(f"Bot no encontrado: {bot_id}")
                return False
            
            # Si el bot está activo, detenerlo primero
            if bot_id in self.active_bots:
                self.stop_bot(bot_id)
            
            # Eliminar bot
            del self.bots[bot_id]
            if bot_id in self.bot_statuses:
                del self.bot_statuses[bot_id]
            
            logger.info(f"Bot eliminado: {bot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error al eliminar bot {bot_id}: {e}")
            return False

def get_system_status() -> Dict[str, Any]:
    """
    Obtiene el estado del sistema
    
    Returns:
        Dict: Estado del sistema
    """
    try:
        # Obtener datos básicos del sistema
        status = {
            "active_bots": 0,
            "last_trade_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": 0.0
        }
        
        # Intentar obtener precio actual
        try:
            from data_management.market_data import get_current_price
            price = get_current_price("SOL-USDT")
            status["current_price"] = price
        except:
            pass
        
        return status
    
    except Exception as e:
        logger.error(f"Error obteniendo estado del sistema: {e}")
        return {
            "active_bots": 0,
            "last_trade_time": "N/A",
            "current_price": 0.0,
            "error": str(e)
        }