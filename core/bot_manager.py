"""
Módulo para el gestor de bots de trading
Administra múltiples instancias de bots de forma concurrente
"""

import os
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger("BotManager")

class BotManager:
    """Gestor de bots de trading"""
    
    def __init__(self, config: Dict):
        """
        Inicializa el gestor de bots
        
        Args:
            config: Configuración general del sistema
        """
        self.config = config
        self.bots = {}  # Diccionario de bots activos
        self.bot_threads = {}  # Hilos de los bots
        self.bot_status = {}  # Estado de los bots
        self.load_bots()  # Cargar bots guardados
    
    def load_bots(self):
        """Carga los bots guardados en el sistema"""
        try:
            if os.path.exists("data/bots.json"):
                with open("data/bots.json", "r") as f:
                    saved_bots = json.load(f)
                
                for bot_id, bot_data in saved_bots.items():
                    self.bots[bot_id] = bot_data
                    self.bot_status[bot_id] = {
                        "active": bot_data.get("active", False),
                        "last_update": datetime.now().isoformat(),
                        "status": "loaded"
                    }
                
                logger.info(f"Cargados {len(saved_bots)} bots desde el almacenamiento")
            else:
                logger.info("No se encontraron bots guardados")
        except Exception as e:
            logger.error(f"Error al cargar bots: {e}")
    
    def save_bots(self):
        """Guarda los bots en almacenamiento persistente"""
        try:
            os.makedirs("data", exist_ok=True)
            with open("data/bots.json", "w") as f:
                json.dump(self.bots, f, indent=2)
            
            logger.info("Bots guardados correctamente")
        except Exception as e:
            logger.error(f"Error al guardar bots: {e}")
    
    def create_bot(self, bot_config: Dict) -> str:
        """
        Crea un nuevo bot con la configuración proporcionada
        
        Args:
            bot_config: Configuración del bot
            
        Returns:
            str: ID del bot creado
        """
        # Generar ID único
        bot_id = f"bot_{int(time.time())}"
        
        # Agregar ID al bot
        bot_config["id"] = bot_id
        
        # Guardar bot
        self.bots[bot_id] = bot_config
        self.bot_status[bot_id] = {
            "active": False,
            "last_update": datetime.now().isoformat(),
            "status": "created"
        }
        
        # Guardar cambios
        self.save_bots()
        
        logger.info(f"Bot '{bot_config.get('name')}' creado con ID {bot_id}")
        return bot_id
    
    def update_bot(self, bot_id: str, bot_config: Dict) -> bool:
        """
        Actualiza la configuración de un bot existente
        
        Args:
            bot_id: ID del bot a actualizar
            bot_config: Nueva configuración
            
        Returns:
            bool: True si se actualizó correctamente, False en caso contrario
        """
        if bot_id not in self.bots:
            logger.error(f"Bot ID {bot_id} no encontrado")
            return False
        
        # Verificar si el bot está activo
        is_active = self.bots[bot_id].get("active", False)
        
        # Si está activo, detenerlo antes de actualizar
        if is_active:
            self.stop_bot(bot_id)
        
        # Actualizar configuración
        self.bots[bot_id] = bot_config
        self.bot_status[bot_id]["last_update"] = datetime.now().isoformat()
        self.bot_status[bot_id]["status"] = "updated"
        
        # Guardar cambios
        self.save_bots()
        
        # Si estaba activo, reiniciarlo
        if is_active:
            self.start_bot(bot_id)
        
        logger.info(f"Bot ID {bot_id} actualizado correctamente")
        return True
    
    def delete_bot(self, bot_id: str) -> bool:
        """
        Elimina un bot del sistema
        
        Args:
            bot_id: ID del bot a eliminar
            
        Returns:
            bool: True si se eliminó correctamente, False en caso contrario
        """
        if bot_id not in self.bots:
            logger.error(f"Bot ID {bot_id} no encontrado")
            return False
        
        # Si el bot está activo, detenerlo primero
        if self.bots[bot_id].get("active", False):
            self.stop_bot(bot_id)
        
        # Eliminar bot
        bot_name = self.bots[bot_id].get("name", "Bot sin nombre")
        del self.bots[bot_id]
        
        if bot_id in self.bot_status:
            del self.bot_status[bot_id]
        
        # Guardar cambios
        self.save_bots()
        
        logger.info(f"Bot '{bot_name}' (ID: {bot_id}) eliminado correctamente")
        return True
    
    def start_bot(self, bot_id: str) -> bool:
        """
        Inicia un bot
        
        Args:
            bot_id: ID del bot a iniciar
            
        Returns:
            bool: True si se inició correctamente, False en caso contrario
        """
        if bot_id not in self.bots:
            logger.error(f"Bot ID {bot_id} no encontrado")
            return False
        
        # Verificar si ya está activo
        if self.bots[bot_id].get("active", False):
            logger.warning(f"Bot ID {bot_id} ya está activo")
            return True
        
        try:
            # Marcar como activo
            self.bots[bot_id]["active"] = True
            self.bot_status[bot_id] = {
                "active": True,
                "last_update": datetime.now().isoformat(),
                "status": "starting",
                "has_position": False,
                "position_type": None,
                "position_size": 0.0,
                "entry_price": 0.0,
                "current_price": 0.0,
                "pnl": 0.0,
                "roi": 0.0
            }
            
            # Iniciar en un hilo separado
            bot_thread = threading.Thread(
                target=self._run_bot,
                args=(bot_id,),
                daemon=True
            )
            bot_thread.start()
            
            # Guardar referencia al hilo
            self.bot_threads[bot_id] = bot_thread
            
            # Guardar cambios
            self.save_bots()
            
            logger.info(f"Bot ID {bot_id} iniciado correctamente")
            return True
        
        except Exception as e:
            logger.error(f"Error al iniciar bot ID {bot_id}: {e}")
            self.bots[bot_id]["active"] = False
            self.bot_status[bot_id]["status"] = "error"
            self.bot_status[bot_id]["error"] = str(e)
            self.save_bots()
            return False
    
    def stop_bot(self, bot_id: str) -> bool:
        """
        Detiene un bot
        
        Args:
            bot_id: ID del bot a detener
            
        Returns:
            bool: True si se detuvo correctamente, False en caso contrario
        """
        if bot_id not in self.bots:
            logger.error(f"Bot ID {bot_id} no encontrado")
            return False
        
        # Verificar si está activo
        if not self.bots[bot_id].get("active", False):
            logger.warning(f"Bot ID {bot_id} no está activo")
            return True
        
        try:
            # Marcar como inactivo
            self.bots[bot_id]["active"] = False
            self.bot_status[bot_id]["active"] = False
            self.bot_status[bot_id]["status"] = "stopping"
            
            # El hilo terminará en la siguiente iteración
            # No es necesario matarlo explícitamente
            
            # Guardar cambios
            self.save_bots()
            
            logger.info(f"Bot ID {bot_id} detenido correctamente")
            return True
        
        except Exception as e:
            logger.error(f"Error al detener bot ID {bot_id}: {e}")
            return False
    
    def _run_bot(self, bot_id: str):
        """
        Ejecuta el bot en un bucle
        
        Args:
            bot_id: ID del bot a ejecutar
        """
        if bot_id not in self.bots:
            return
        
        logger.info(f"Iniciando ejecución del bot ID {bot_id}")
        
        bot_config = self.bots[bot_id]
        bot_name = bot_config.get("name", "Bot sin nombre")
        bot_symbol = bot_config.get("symbol", "SOL-USDT")
        paper_trading = bot_config.get("paper_trading", True)
        strategy_name = bot_config.get("strategy", "N/A")
        
        # Actualizar estado
        self.bot_status[bot_id]["status"] = "running"
        
        # Simulación de ejecución continua
        try:
            while self.bots[bot_id].get("active", False):
                # Actualizar precio actual
                current_price = self._get_current_price(bot_symbol)
                
                # Actualizar estado
                self.bot_status[bot_id]["current_price"] = current_price
                
                # Simular análisis del mercado
                self._analyze_market(bot_id)
                
                # Actualizar P&L si hay posición abierta
                if self.bot_status[bot_id].get("has_position", False):
                    self._update_position_pnl(bot_id)
                
                # Dormir un tiempo antes de la siguiente iteración
                time.sleep(5)  # Ajustar según necesidad
            
            logger.info(f"Bot {bot_name} (ID: {bot_id}) detenido")
            self.bot_status[bot_id]["status"] = "stopped"
        
        except Exception as e:
            logger.error(f"Error en ejecución del bot {bot_name} (ID: {bot_id}): {e}")
            self.bot_status[bot_id]["status"] = "error"
            self.bot_status[bot_id]["error"] = str(e)
            self.bots[bot_id]["active"] = False
            self.save_bots()
    
    def _get_current_price(self, symbol: str) -> float:
        """
        Obtiene el precio actual del mercado
        
        Args:
            symbol: Símbolo del mercado
            
        Returns:
            float: Precio actual
        """
        # En una implementación real, esto obtendría el precio desde el exchange
        # Aquí simulamos un precio aleatorio para demostración
        import random
        
        base_price = 150.0  # Precio base para SOL-USDT
        variation = random.uniform(-2.0, 2.0)  # Variación aleatoria
        
        return base_price + variation
    
    def _analyze_market(self, bot_id: str):
        """
        Analiza el mercado y toma decisiones de trading
        
        Args:
            bot_id: ID del bot
        """
        if bot_id not in self.bots:
            return
        
        # En una implementación real, esto analizaría el mercado según la estrategia
        # y ejecutaría operaciones según corresponda
        
        # Para demostración, simulamos aleatoriamente operaciones
        import random
        
        # Probabilidad de abrir/cerrar una operación
        if not self.bot_status[bot_id].get("has_position", False):
            # No hay posición abierta, decidir si abrir una
            if random.random() < 0.1:  # 10% de probabilidad
                self._open_position(bot_id)
        else:
            # Hay posición abierta, decidir si cerrarla
            if random.random() < 0.2:  # 20% de probabilidad
                self._close_position(bot_id)
    
    def _open_position(self, bot_id: str):
        """
        Abre una posición de trading
        
        Args:
            bot_id: ID del bot
        """
        if bot_id not in self.bots:
            return
        
        import random
        
        # Obtener configuración y precios
        bot_config = self.bots[bot_id]
        current_price = self.bot_status[bot_id].get("current_price", 0.0)
        balance = bot_config.get("balance", 1000.0)
        risk_per_trade = bot_config.get("config", {}).get("risk_per_trade", 1.0)
        
        # Calcular tamaño de posición (en USD)
        position_size_usd = balance * (risk_per_trade / 100.0)
        
        # Convertir a unidades del activo
        position_size = position_size_usd / current_price
        
        # Decidir tipo de posición (largo/corto)
        position_type = "long" if random.random() < 0.7 else "short"  # 70% largo, 30% corto
        
        # Actualizar estado
        self.bot_status[bot_id]["has_position"] = True
        self.bot_status[bot_id]["position_type"] = position_type
        self.bot_status[bot_id]["position_size"] = position_size
        self.bot_status[bot_id]["entry_price"] = current_price
        
        logger.info(f"Bot ID {bot_id} abrió posición {position_type} de {position_size:.6f} a {current_price:.2f}")
    
    def _close_position(self, bot_id: str):
        """
        Cierra una posición de trading
        
        Args:
            bot_id: ID del bot
        """
        if bot_id not in self.bots:
            return
        
        # Obtener datos de la posición
        position_type = self.bot_status[bot_id].get("position_type", None)
        position_size = self.bot_status[bot_id].get("position_size", 0.0)
        entry_price = self.bot_status[bot_id].get("entry_price", 0.0)
        current_price = self.bot_status[bot_id].get("current_price", 0.0)
        
        # Calcular P&L
        if position_type == "long":
            pnl = position_size * (current_price - entry_price)
        elif position_type == "short":
            pnl = position_size * (entry_price - current_price)
        else:
            pnl = 0.0
        
        # Actualizar balance
        self.bots[bot_id]["balance"] = self.bots[bot_id].get("balance", 1000.0) + pnl
        
        # Registrar operación en historial
        if "trade_history" not in self.bots[bot_id]:
            self.bots[bot_id]["trade_history"] = []
        
        self.bots[bot_id]["trade_history"].append({
            "timestamp": datetime.now().isoformat(),
            "type": position_type,
            "size": position_size,
            "entry_price": entry_price,
            "exit_price": current_price,
            "pnl": pnl
        })
        
        # Actualizar estado
        self.bot_status[bot_id]["has_position"] = False
        self.bot_status[bot_id]["position_type"] = None
        self.bot_status[bot_id]["position_size"] = 0.0
        self.bot_status[bot_id]["entry_price"] = 0.0
        self.bot_status[bot_id]["pnl"] = 0.0
        self.bot_status[bot_id]["roi"] = 0.0
        
        # Guardar cambios
        self.save_bots()
        
        logger.info(f"Bot ID {bot_id} cerró posición {position_type}. P&L: {pnl:.2f} USDT")
    
    def _update_position_pnl(self, bot_id: str):
        """
        Actualiza P&L de una posición abierta
        
        Args:
            bot_id: ID del bot
        """
        if bot_id not in self.bots:
            return
        
        # Obtener datos de la posición
        position_type = self.bot_status[bot_id].get("position_type", None)
        position_size = self.bot_status[bot_id].get("position_size", 0.0)
        entry_price = self.bot_status[bot_id].get("entry_price", 0.0)
        current_price = self.bot_status[bot_id].get("current_price", 0.0)
        
        # Calcular P&L
        if position_type == "long":
            pnl = position_size * (current_price - entry_price)
            roi = (current_price / entry_price - 1) * 100
        elif position_type == "short":
            pnl = position_size * (entry_price - current_price)
            roi = (entry_price / current_price - 1) * 100
        else:
            pnl = 0.0
            roi = 0.0
        
        # Actualizar estado
        self.bot_status[bot_id]["pnl"] = pnl
        self.bot_status[bot_id]["roi"] = roi
    
    def get_bot_status(self, bot_id: str) -> Dict:
        """
        Obtiene el estado actual de un bot
        
        Args:
            bot_id: ID del bot
            
        Returns:
            Dict: Estado del bot
        """
        if bot_id not in self.bot_status:
            return {"error": "Bot no encontrado"}
        
        return self.bot_status[bot_id]
    
    def get_all_bots(self) -> List[Dict]:
        """
        Obtiene todos los bots con su información
        
        Returns:
            List[Dict]: Lista de bots
        """
        result = []
        
        for bot_id, bot_data in self.bots.items():
            # Combinar datos del bot con su estado
            bot_info = bot_data.copy()
            
            if bot_id in self.bot_status:
                # Agregar campos relevantes del estado
                status = self.bot_status[bot_id]
                bot_info.update({
                    "status": status.get("status", "unknown"),
                    "has_position": status.get("has_position", False),
                    "position_type": status.get("position_type", None),
                    "position_size": status.get("position_size", 0.0),
                    "entry_price": status.get("entry_price", 0.0),
                    "current_price": status.get("current_price", 0.0),
                    "pnl": status.get("pnl", 0.0),
                    "roi": status.get("roi", 0.0)
                })
            
            result.append(bot_info)
        
        return result
    
    def shutdown(self):
        """Detiene todos los bots y libera recursos"""
        logger.info("Apagando gestor de bots...")
        
        # Detener todos los bots activos
        for bot_id, bot_data in self.bots.items():
            if bot_data.get("active", False):
                self.stop_bot(bot_id)
        
        # Guardar estado final
        self.save_bots()
        
        logger.info("Gestor de bots apagado correctamente")


# Funciones de utilidad para el CLI

def get_bots() -> List[Dict]:
    """
    Obtiene la lista de bots configurados
    
    Returns:
        List[Dict]: Lista de bots
    """
    # En una implementación real, esto obtendría los bots desde una instancia del gestor
    # Para demostración, cargamos directamente del archivo
    try:
        if os.path.exists("data/bots.json"):
            with open("data/bots.json", "r") as f:
                bots_dict = json.load(f)
            
            return list(bots_dict.values())
        else:
            return []
    except Exception as e:
        logger.error(f"Error al obtener bots: {e}")
        return []

def get_bot_history(bot_id: str, limit: int = None) -> List[Dict]:
    """
    Obtiene el historial de operaciones de un bot
    
    Args:
        bot_id: ID del bot
        limit: Límite de entradas a retornar
        
    Returns:
        List[Dict]: Historial de operaciones
    """
    # En una implementación real, esto obtendría el historial desde una base de datos
    try:
        bots = get_bots()
        
        for bot in bots:
            if bot.get("id") == bot_id:
                history = bot.get("trade_history", [])
                
                if limit and len(history) > limit:
                    return history[-limit:]
                
                return history
        
        return []
    except Exception as e:
        logger.error(f"Error al obtener historial: {e}")
        return []

def get_recent_trades(limit: int = 10) -> List[Dict]:
    """
    Obtiene las operaciones recientes de todos los bots
    
    Args:
        limit: Límite de entradas a retornar
        
    Returns:
        List[Dict]: Operaciones recientes
    """
    # En una implementación real, esto obtendría las operaciones desde una base de datos
    try:
        all_trades = []
        bots = get_bots()
        
        for bot in bots:
            bot_name = bot.get("name", "Bot sin nombre")
            bot_id = bot.get("id", "unknown")
            
            for trade in bot.get("trade_history", []):
                trade_copy = trade.copy()
                trade_copy["bot_name"] = bot_name
                trade_copy["bot_id"] = bot_id
                all_trades.append(trade_copy)
        
        # Ordenar por fecha (más recientes primero)
        all_trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Limitar cantidad
        if limit and len(all_trades) > limit:
            return all_trades[:limit]
        
        return all_trades
    except Exception as e:
        logger.error(f"Error al obtener operaciones recientes: {e}")
        return []

def get_performance_stats() -> Dict:
    """
    Obtiene estadísticas de rendimiento de los bots
    
    Returns:
        Dict: Estadísticas de rendimiento
    """
    # En una implementación real, esto calcularía estadísticas reales
    # Para demostración, retornamos datos simulados
    return {
        "strategy_stats": [
            {
                "strategy": "Cruce de Medias Móviles",
                "trade_count": 156,
                "win_rate": 58.3,
                "avg_profit": 2.45,
                "avg_loss": -1.87,
                "profit_factor": 1.31,
                "avg_duration": "2h 35m"
            },
            {
                "strategy": "RSI + Bollinger Bands",
                "trade_count": 89,
                "win_rate": 62.8,
                "avg_profit": 3.12,
                "avg_loss": -2.05,
                "profit_factor": 1.52,
                "avg_duration": "4h 10m"
            },
            {
                "strategy": "Adaptativa (Múltiples indicadores)",
                "trade_count": 127,
                "win_rate": 64.5,
                "avg_profit": 2.87,
                "avg_loss": -1.76,
                "profit_factor": 1.63,
                "avg_duration": "3h 45m"
            }
        ],
        "symbol_stats": [
            {
                "symbol": "SOL-USDT",
                "trade_count": 278,
                "win_rate": 61.5,
                "avg_profit": 2.82,
                "avg_loss": -1.90,
                "profit_factor": 1.48
            },
            {
                "symbol": "BTC-USDT",
                "trade_count": 94,
                "win_rate": 58.7,
                "avg_profit": 2.35,
                "avg_loss": -1.95,
                "profit_factor": 1.21
            }
        ],
        "monthly_stats": [
            {
                "month": "2025-04",
                "trade_count": 87,
                "win_rate": 62.1,
                "total_profit": 145.32,
                "total_loss": -78.60,
                "net_pnl": 66.72
            },
            {
                "month": "2025-03",
                "trade_count": 112,
                "win_rate": 59.8,
                "total_profit": 178.45,
                "total_loss": -103.25,
                "net_pnl": 75.20
            },
            {
                "month": "2025-02",
                "trade_count": 95,
                "win_rate": 57.9,
                "total_profit": 152.68,
                "total_loss": -98.42,
                "net_pnl": 54.26
            }
        ]
    }

def get_system_status() -> Dict:
    """
    Obtiene el estado del sistema
    
    Returns:
        Dict: Estado del sistema
    """
    try:
        bots = get_bots()
        active_bots = sum(1 for bot in bots if bot.get("active", False))
        
        recent_trades = get_recent_trades(1)
        last_trade_time = recent_trades[0].get("timestamp", "N/A") if recent_trades else "N/A"
        
        # Obtener precio actual (simulado)
        import random
        current_price = 150.0 + random.uniform(-2.0, 2.0)
        
        return {
            "active_bots": active_bots,
            "total_bots": len(bots),
            "last_trade_time": last_trade_time,
            "current_price": current_price
        }
    except Exception as e:
        logger.error(f"Error al obtener estado del sistema: {e}")
        return {
            "active_bots": 0,
            "total_bots": 0,
            "last_trade_time": "N/A",
            "current_price": 0.0
        }