#!/usr/bin/env python3
"""
Bot Battle Arena: La batalla de los bots de trading.

Este módulo implementa un sistema de competencia entre bots de trading donde:
- Múltiples bots compiten para ver quién logra el mayor porcentaje de ganancia
- Los bots con peor rendimiento "mueren" y renacen optimizados
- El sistema evoluciona continuamente para mejorar las estrategias
"""

import os
import json
import logging
import random
import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BotBattleArena')

class BotWarrior:
    """
    Representa un bot guerrero en la batalla.
    
    Cada bot tiene sus propias características, estrategias y estadísticas
    de rendimiento.
    """
    
    def __init__(self, 
               warrior_id: str,
               strategy_name: str,
               generation: int = 1,
               symbol: str = "SOL-USDT",
               timeframe: str = "5m",
               initial_balance: float = 100.0,
               leverage: int = 1,
               params: Dict[str, Any] = None,
               parent_ids: List[str] = None):
        """
        Inicializa un bot guerrero.
        
        Args:
            warrior_id: ID único del guerrero
            strategy_name: Nombre de su estrategia
            generation: Generación del guerrero (1 = primera generación)
            symbol: Par de trading
            timeframe: Marco temporal
            initial_balance: Balance inicial (todos comienzan iguales)
            leverage: Apalancamiento
            params: Parámetros específicos de estrategia
            parent_ids: IDs de los padres (para bots optimizados)
        """
        self.warrior_id = warrior_id
        self.strategy_name = strategy_name
        self.generation = generation
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.leverage = leverage
        self.params = params or {}
        self.parent_ids = parent_ids or []
        
        # Estado del guerrero
        self.active = False
        self.defeated = False
        self.start_time = None
        self.life_span = timedelta(days=0)  # Tiempo de vida del guerrero
        self.battle_stats = self._initialize_battle_stats()
        
        # Historial de rendimiento
        self.performance_history = []
        
        # Historial de mejores operaciones
        self.best_trades = []
        self.worst_trades = []
        
        # Características de personalidad (puramente decorativas, para diversión)
        self.personality = self._generate_personality()
    
    def _initialize_battle_stats(self) -> Dict[str, Any]:
        """Inicializa estadísticas de batalla."""
        return {
            "roi": 0.0,
            "roi_daily": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "profitable_trades": 0,
            "losing_trades": 0,
            "total_won": 0.0,
            "total_lost": 0.0,
            "win_streak": 0,
            "lose_streak": 0,
            "max_win_streak": 0,
            "max_lose_streak": 0,
            "battles_won": 0,
            "battles_lost": 0,
            "best_day_roi": 0.0,
            "worst_day_roi": 0.0,
            "last_update": None
        }
    
    def _generate_personality(self) -> Dict[str, Any]:
        """
        Genera una personalidad aleatoria para el guerrero.
        
        Puramente decorativo, para dar más vida a los bots.
        """
        # Nombres humanos para los bots (mezclados y diversos)
        nombres = [
            # Nombres masculinos comunes
            "Hugo", "Luis", "Pablo", "Carlos", "Juan", "Pedro", "Miguel", 
            "Antonio", "Javier", "David", "Manuel", "Roberto", "Fernando",
            "Sergio", "Daniel", "José", "Alejandro", "Diego", "Andrés",
            
            # Nombres femeninos comunes
            "Ana", "María", "Laura", "Sofía", "Lucía", "Carmen", "Isabel",
            "Elena", "Paula", "Julia", "Marta", "Patricia", "Cristina",
            "Claudia", "Sara", "Raquel", "Natalia", "Silvia", "Beatriz"
        ]
        
        personalities = [
            "agresivo", "cauteloso", "calculador", "impulsivo", 
            "paciente", "arriesgado", "conservador", "analítico",
            "intuitivo", "metódico", "adaptable", "oportunista"
        ]
        
        quotes = [
            "¡Victoria o muerte!",
            "El mercado es mi campo de batalla",
            "Análisis, paciencia, victoria",
            "Los débiles venden en pánico, los fuertes compran la caída",
            "Mi algoritmo es superior",
            "Las tendencias son mis aliadas",
            "Compro tu miedo, vendo tu codicia",
            "El que sobrevive no es el más fuerte, sino el que mejor se adapta",
            "Entro como un ninja, salgo como un samurái",
            "Las velas japonesas cuentan la historia, yo escribo el final",
            "Datos, no emociones",
            "La volatilidad es mi aliada",
            "Sigo el plan, ignoro el ruido",
            "Mi estrategia evolucionará y dominará"
        ]
        
        strengths = [
            "análisis técnico", "identificación de tendencias", "detección de patrones",
            "timing preciso", "gestión del riesgo", "scalping rápido",
            "paciencia sobrehumana", "entradas perfectas", "optimización continua",
            "adaptación rápida", "análisis multitimeframe", "trading contrarian"
        ]
        
        weaknesses = [
            "impaciencia", "exceso de confianza", "aversión al riesgo",
            "entradas tardías", "cierres prematuros", "indecisión",
            "sobreoptimización", "miedo a perderse movimientos", "sobreanálisis",
            "sesgo de confirmación", "fallos en tendencias laterales", "manejo de noticias"
        ]
        
        return {
            "name": random.choice(nombres),
            "archetype": random.choice(personalities),
            "quote": random.choice(quotes),
            "color": "#" + ''.join(random.choices('0123456789ABCDEF', k=6)),
            "strength": random.choice(strengths),
            "weakness": random.choice(weaknesses),
            "aggression": random.uniform(0.1, 1.0),
            "patience": random.uniform(0.1, 1.0),
            "adaptability": random.uniform(0.1, 1.0)
        }
    
    def update_battle_stats(self, bot_status: Dict[str, Any]):
        """
        Actualiza estadísticas de batalla basado en el estado del bot.
        
        Args:
            bot_status: Estado actual del bot de la instancia real
        """
        # Actualizar balance actual
        self.current_balance = bot_status.get("current_balance", self.current_balance)
        
        # Calcular ROI
        self.battle_stats["roi"] = (self.current_balance / self.initial_balance - 1) * 100
        
        # Actualizar estadísticas de trading
        metrics = bot_status.get("metrics", {})
        self.battle_stats["win_rate"] = metrics.get("win_rate", 0.0)
        self.battle_stats["total_trades"] = metrics.get("total_trades", 0)
        self.battle_stats["profitable_trades"] = metrics.get("profitable_trades", 0)
        self.battle_stats["losing_trades"] = metrics.get("losing_trades", 0)
        self.battle_stats["total_won"] = metrics.get("total_profit", 0.0)
        self.battle_stats["total_lost"] = metrics.get("total_loss", 0.0)
        
        # Actualizar racha actual
        if self.battle_stats["total_trades"] > 0:
            recent_trades = bot_status.get("recent_trades", [])
            current_streak = 0
            streak_type = None
            
            for trade in recent_trades:
                if trade.get("trade_type") != "EXIT":
                    continue
                
                pnl = trade.get("pnl", 0)
                
                if streak_type is None:
                    # Primera operación define tipo de racha
                    streak_type = "win" if pnl > 0 else "lose"
                    current_streak = 1
                elif (streak_type == "win" and pnl > 0) or (streak_type == "lose" and pnl <= 0):
                    # Continuación de racha
                    current_streak += 1
                else:
                    # Fin de la racha
                    break
            
            if streak_type == "win":
                self.battle_stats["win_streak"] = current_streak
                self.battle_stats["lose_streak"] = 0
                self.battle_stats["max_win_streak"] = max(self.battle_stats["max_win_streak"], current_streak)
            else:
                self.battle_stats["lose_streak"] = current_streak
                self.battle_stats["win_streak"] = 0
                self.battle_stats["max_lose_streak"] = max(self.battle_stats["max_lose_streak"], current_streak)
        
        # Actualizar mejores y peores operaciones
        if bot_status.get("recent_trades"):
            exit_trades = [t for t in bot_status["recent_trades"] if t.get("trade_type") == "EXIT"]
            
            if exit_trades:
                # Ordenar por PnL
                sorted_trades = sorted(exit_trades, key=lambda t: t.get("pnl", 0), reverse=True)
                
                # Mejores operaciones
                self.best_trades = sorted_trades[:3]
                
                # Peores operaciones
                self.worst_trades = sorted_trades[-3:] if len(sorted_trades) >= 3 else sorted_trades[-len(sorted_trades):]
        
        # Calcular ROI diario (promedio)
        daily_metrics = bot_status.get("daily_metrics", {})
        self.battle_stats["roi_daily"] = daily_metrics.get("profit_percent", 0.0)
        
        # Actualizar mejor y peor día
        if daily_metrics.get("profit_percent", 0) > self.battle_stats["best_day_roi"]:
            self.battle_stats["best_day_roi"] = daily_metrics.get("profit_percent", 0)
        
        if daily_metrics.get("profit_percent", 0) < self.battle_stats["worst_day_roi"]:
            self.battle_stats["worst_day_roi"] = daily_metrics.get("profit_percent", 0)
        
        # Actualizar timestamp
        self.battle_stats["last_update"] = datetime.now().isoformat()
        
        # Actualizar historial de rendimiento
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "balance": self.current_balance,
            "roi": self.battle_stats["roi"],
            "daily_roi": self.battle_stats["roi_daily"],
            "win_rate": self.battle_stats["win_rate"],
            "trades": self.battle_stats["total_trades"]
        })
        
        # Limitar historial a 1000 puntos para evitar crecimiento excesivo
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        # Calcular tiempo de vida
        if self.start_time:
            self.life_span = datetime.now() - self.start_time
    
    def activate(self):
        """Activa el guerrero para la batalla."""
        self.active = True
        self.defeated = False
        self.start_time = datetime.now()
    
    def deactivate(self):
        """Desactiva el guerrero de la batalla."""
        self.active = False
    
    def mark_as_defeated(self):
        """Marca al guerrero como derrotado."""
        self.active = False
        self.defeated = True
    
    def get_battle_card(self) -> Dict[str, Any]:
        """
        Obtiene una tarjeta de batalla con información del guerrero.
        
        Returns:
            Dict[str, Any]: Tarjeta de batalla
        """
        # Calcular clasificación basada en rendimiento
        if self.battle_stats["roi"] >= 50:
            rank = "S"  # Legendario
        elif self.battle_stats["roi"] >= 25:
            rank = "A"  # Élite
        elif self.battle_stats["roi"] >= 10:
            rank = "B"  # Veterano
        elif self.battle_stats["roi"] >= 0:
            rank = "C"  # Novato
        else:
            rank = "D"  # Perdedor
        
        # Calcular puntaje de poder
        power_score = (
            (100 + self.battle_stats["roi"]) *
            (1 + self.battle_stats["win_rate"] / 100) *
            (1 + min(self.battle_stats["total_trades"], 100) / 100)
        )
        
        # Calcular proyecciones de rendimiento
        daily_roi = self.battle_stats["roi_daily"]
        monthly_roi_projected = daily_roi * 30 if daily_roi else 0
        annual_roi_projected = daily_roi * 365 if daily_roi else 0
        
        # Usar nombre humano de la personalidad si existe
        human_name = self.personality.get("name", "Bot")
        
        return {
            "warrior_id": self.warrior_id,
            "name": human_name,  # Usar nombre humano
            "strategy": f"{self.strategy_name} {self.timeframe}",
            "generation": self.generation,
            "active": self.active,
            "defeated": self.defeated,
            "rank": rank,
            "power": int(power_score),
            "roi": self.battle_stats["roi"],
            "daily_roi": self.battle_stats["roi_daily"],
            "monthly_roi_projected": monthly_roi_projected,
            "annual_roi_projected": annual_roi_projected,
            "win_rate": self.battle_stats["win_rate"],
            "total_trades": self.battle_stats["total_trades"],
            "current_streak": f"{self.battle_stats['win_streak']}W" if self.battle_stats["win_streak"] > 0 else f"{self.battle_stats['lose_streak']}L",
            "life_span": str(self.life_span).split('.')[0] if self.life_span else "00:00:00",
            "days_alive": round(self.life_span.total_seconds() / 86400, 1) if self.life_span else 0,
            "color": self.personality["color"],
            "quote": self.personality["quote"],
            "archetype": self.personality["archetype"],
            "lineage": f"Gen {self.generation}" + (f" [Descendiente de {', '.join(self.parent_ids)}]" if self.parent_ids else ""),
            "strength": self.personality["strength"],
            "weakness": self.personality["weakness"]
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el guerrero a diccionario para almacenamiento.
        
        Returns:
            Dict[str, Any]: Representación en diccionario
        """
        return {
            "warrior_id": self.warrior_id,
            "strategy_name": self.strategy_name,
            "generation": self.generation,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "leverage": self.leverage,
            "params": self.params,
            "parent_ids": self.parent_ids,
            "active": self.active,
            "defeated": self.defeated,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "life_span": str(self.life_span),
            "battle_stats": self.battle_stats,
            "performance_history": self.performance_history,
            "personality": self.personality,
            "best_trades": self.best_trades,
            "worst_trades": self.worst_trades
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BotWarrior':
        """
        Crea un guerrero a partir de un diccionario.
        
        Args:
            data: Diccionario con datos del guerrero
            
        Returns:
            BotWarrior: Instancia reconstruida
        """
        warrior = cls(
            warrior_id=data.get("warrior_id", f"warrior_{int(time.time())}"),
            strategy_name=data.get("strategy_name", "unknown"),
            generation=data.get("generation", 1),
            symbol=data.get("symbol", "SOL-USDT"),
            timeframe=data.get("timeframe", "5m"),
            initial_balance=data.get("initial_balance", 100.0),
            leverage=data.get("leverage", 1),
            params=data.get("params", {}),
            parent_ids=data.get("parent_ids", [])
        )
        
        # Restaurar estado
        warrior.current_balance = data.get("current_balance", warrior.initial_balance)
        warrior.active = data.get("active", False)
        warrior.defeated = data.get("defeated", False)
        
        # Restaurar tiempo
        start_time = data.get("start_time")
        if start_time:
            warrior.start_time = datetime.fromisoformat(start_time)
        
        life_span = data.get("life_span")
        if life_span:
            days, time_str = life_span.split(' days, ') if ' days, ' in life_span else ('0', life_span)
            hours, minutes, seconds = map(int, time_str.split(':'))
            warrior.life_span = timedelta(days=int(days), hours=hours, minutes=minutes, seconds=seconds)
        
        # Restaurar estadísticas
        warrior.battle_stats = data.get("battle_stats", warrior._initialize_battle_stats())
        warrior.performance_history = data.get("performance_history", [])
        warrior.personality = data.get("personality", warrior._generate_personality())
        warrior.best_trades = data.get("best_trades", [])
        warrior.worst_trades = data.get("worst_trades", [])
        
        return warrior

class BotBattleArena:
    """
    Arena de batalla para bots de trading.
    
    Gestiona la competencia entre múltiples bots, eliminando a los
    perdedores y generando nuevos optimizados.
    """
    
    def __init__(self, 
               arena_file: str = "data/bot_battle_arena.json",
               symbol: str = "SOL-USDT",
               evaluation_period: int = 3,
               elimination_rate: float = 0.2,
               min_battle_size: int = 5,
               max_battle_size: int = 20,
               optimization_rate: float = 0.2):
        """
        Inicializa la arena de batalla.
        
        Args:
            arena_file: Archivo para almacenar la arena
            symbol: Par de trading principal
            evaluation_period: Días entre evaluaciones
            elimination_rate: Porcentaje de bots a eliminar en cada evaluación
            min_battle_size: Número mínimo de bots en la arena
            max_battle_size: Número máximo de bots en la arena
            optimization_rate: Tasa de optimización para nuevos bots
        """
        self.arena_file = arena_file
        self.symbol = symbol
        self.evaluation_period = evaluation_period
        self.elimination_rate = elimination_rate
        self.min_battle_size = min_battle_size
        self.max_battle_size = max_battle_size
        self.optimization_rate = optimization_rate
        
        # Lista de guerreros
        self.warriors: Dict[str, BotWarrior] = {}
        
        # Estadísticas de la arena
        self.arena_stats = {
            "creation_date": datetime.now().isoformat(),
            "last_evaluation": None,
            "next_evaluation": None,
            "total_battles": 0,
            "total_warriors": 0,
            "total_defeated": 0,
            "total_generations": 1,
            "best_roi_ever": 0.0,
            "worst_roi_ever": 0.0,
            "hall_of_fame": [],  # Mejores guerreros históricos
            "hall_of_shame": []  # Peores guerreros históricos
        }
        
        # Historial de evaluaciones
        self.evaluation_history = []
        
        # Registro de evolución
        self.evolution_log = []
        
        # Cargar arena si existe
        self._load_arena()
    
    def _load_arena(self):
        """Carga la arena desde archivo."""
        try:
            if os.path.exists(self.arena_file):
                with open(self.arena_file, 'r') as f:
                    data = json.load(f)
                
                # Cargar estadísticas
                self.arena_stats = data.get("arena_stats", self.arena_stats)
                
                # Cargar historiales
                self.evaluation_history = data.get("evaluation_history", [])
                self.evolution_log = data.get("evolution_log", [])
                
                # Cargar guerreros
                for warrior_data in data.get("warriors", []):
                    warrior = BotWarrior.from_dict(warrior_data)
                    self.warriors[warrior.warrior_id] = warrior
                
                logger.info(f"Arena cargada con {len(self.warriors)} guerreros")
        except Exception as e:
            logger.error(f"Error al cargar arena: {e}")
    
    def _save_arena(self):
        """Guarda la arena en archivo."""
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.arena_file), exist_ok=True)
            
            data = {
                "arena_stats": self.arena_stats,
                "evaluation_history": self.evaluation_history,
                "evolution_log": self.evolution_log,
                "warriors": [warrior.to_dict() for warrior in self.warriors.values()]
            }
            
            with open(self.arena_file, 'w') as f:
                json.dump(data, f, indent=4)
            
            logger.info(f"Arena guardada con {len(self.warriors)} guerreros")
        except Exception as e:
            logger.error(f"Error al guardar arena: {e}")
    
    def add_warrior(self, 
                  strategy_name: str,
                  timeframe: str,
                  leverage: int = 1,
                  params: Dict[str, Any] = None,
                  warrior_id: str = None) -> str:
        """
        Añade un nuevo guerrero a la arena.
        
        Args:
            strategy_name: Nombre de la estrategia
            timeframe: Marco temporal
            leverage: Apalancamiento
            params: Parámetros de la estrategia
            warrior_id: ID opcional (se genera uno si no se proporciona)
            
        Returns:
            str: ID del guerrero añadido
        """
        # Generar ID si no se proporciona
        if not warrior_id:
            warrior_id = f"{strategy_name}_{timeframe}_{int(time.time())}"
        
        # Verificar si ya existe
        if warrior_id in self.warriors:
            logger.warning(f"Ya existe un guerrero con ID {warrior_id}")
            return warrior_id
        
        # Crear nuevo guerrero
        warrior = BotWarrior(
            warrior_id=warrior_id,
            strategy_name=strategy_name,
            generation=1,  # Primera generación
            symbol=self.symbol,
            timeframe=timeframe,
            initial_balance=100.0,
            leverage=leverage,
            params=params or {}
        )
        
        # Añadir a la arena
        self.warriors[warrior_id] = warrior
        
        # Actualizar estadísticas
        self.arena_stats["total_warriors"] += 1
        
        # Guardar arena
        self._save_arena()
        
        logger.info(f"Guerrero añadido: {warrior_id}")
        return warrior_id
    
    def remove_warrior(self, warrior_id: str) -> bool:
        """
        Elimina un guerrero de la arena.
        
        Args:
            warrior_id: ID del guerrero
            
        Returns:
            bool: True si se eliminó correctamente
        """
        if warrior_id not in self.warriors:
            logger.warning(f"Guerrero no encontrado: {warrior_id}")
            return False
        
        # Eliminar de la arena
        del self.warriors[warrior_id]
        
        # Guardar arena
        self._save_arena()
        
        logger.info(f"Guerrero eliminado: {warrior_id}")
        return True
    
    def activate_warrior(self, warrior_id: str) -> bool:
        """
        Activa un guerrero para la batalla.
        
        Args:
            warrior_id: ID del guerrero
            
        Returns:
            bool: True si se activó correctamente
        """
        if warrior_id not in self.warriors:
            logger.warning(f"Guerrero no encontrado: {warrior_id}")
            return False
        
        self.warriors[warrior_id].activate()
        
        # Guardar arena
        self._save_arena()
        
        logger.info(f"Guerrero activado: {warrior_id}")
        return True
    
    def deactivate_warrior(self, warrior_id: str) -> bool:
        """
        Desactiva un guerrero de la batalla.
        
        Args:
            warrior_id: ID del guerrero
            
        Returns:
            bool: True si se desactivó correctamente
        """
        if warrior_id not in self.warriors:
            logger.warning(f"Guerrero no encontrado: {warrior_id}")
            return False
        
        self.warriors[warrior_id].deactivate()
        
        # Guardar arena
        self._save_arena()
        
        logger.info(f"Guerrero desactivado: {warrior_id}")
        return True
    
    def update_warrior_status(self, warrior_id: str, bot_status: Dict[str, Any]) -> bool:
        """
        Actualiza el estado de un guerrero con datos del bot real.
        
        Args:
            warrior_id: ID del guerrero
            bot_status: Estado actual del bot
            
        Returns:
            bool: True si se actualizó correctamente
        """
        if warrior_id not in self.warriors:
            logger.warning(f"Guerrero no encontrado: {warrior_id}")
            return False
        
        # Actualizar estadísticas del guerrero
        self.warriors[warrior_id].update_battle_stats(bot_status)
        
        # Verificar si supera récords históricos
        roi = self.warriors[warrior_id].battle_stats["roi"]
        
        if roi > self.arena_stats["best_roi_ever"]:
            self.arena_stats["best_roi_ever"] = roi
        
        if roi < self.arena_stats["worst_roi_ever"]:
            self.arena_stats["worst_roi_ever"] = roi
        
        # Guardar arena cada 5 actualizaciones
        if random.random() < 0.2:
            self._save_arena()
        
        return True
    
    def evaluate_arena(self) -> Dict[str, Any]:
        """
        Realiza una evaluación de la arena, eliminando perdedores
        y generando nuevos guerreros optimizados.
        
        Returns:
            Dict[str, Any]: Resultados de la evaluación
        """
        # Obtener guerreros activos
        active_warriors = {wid: w for wid, w in self.warriors.items() if w.active and not w.defeated}
        
        if not active_warriors:
            logger.warning("No hay guerreros activos para evaluar")
            return {
                "success": False,
                "message": "No hay guerreros activos",
                "eliminated": [],
                "new_warriors": []
            }
        
        # Ordenar por ROI (de mayor a menor)
        sorted_warriors = sorted(
            active_warriors.values(),
            key=lambda w: w.battle_stats["roi"],
            reverse=True
        )
        
        # Seleccionar ganadores y perdedores
        num_to_eliminate = max(1, int(len(sorted_warriors) * self.elimination_rate))
        
        # Asegurar que no eliminamos demasiados
        if len(sorted_warriors) - num_to_eliminate < self.min_battle_size:
            num_to_eliminate = max(0, len(sorted_warriors) - self.min_battle_size)
        
        # Los peores son eliminados
        losers = sorted_warriors[-num_to_eliminate:]
        
        # Marcar perdedores como derrotados
        eliminated_ids = []
        for loser in losers:
            loser.mark_as_defeated()
            eliminated_ids.append(loser.warrior_id)
            self.arena_stats["total_defeated"] += 1
            
            # Añadir al hall of shame si califica
            if len(self.arena_stats["hall_of_shame"]) < 10 or loser.battle_stats["roi"] < self.arena_stats["hall_of_shame"][-1]["roi"]:
                shame_entry = {
                    "warrior_id": loser.warrior_id,
                    "strategy_name": loser.strategy_name,
                    "timeframe": loser.timeframe,
                    "generation": loser.generation,
                    "roi": loser.battle_stats["roi"],
                    "win_rate": loser.battle_stats["win_rate"],
                    "total_trades": loser.battle_stats["total_trades"],
                    "eliminated_at": datetime.now().isoformat()
                }
                
                # Añadir y ordenar
                self.arena_stats["hall_of_shame"].append(shame_entry)
                self.arena_stats["hall_of_shame"] = sorted(
                    self.arena_stats["hall_of_shame"],
                    key=lambda x: x["roi"]
                )[:10]  # Mantener solo los 10 peores
        
        # Seleccionar los mejores para procrear la siguiente generación
        num_parents = min(5, max(2, len(sorted_warriors) // 2))
        parents = sorted_warriors[:num_parents]
        
        # Generar nuevos guerreros optimizados
        new_warriors = []
        next_generation = self.arena_stats["total_generations"] + 1
        
        # Crear nuevos guerreros mientras no excedamos el máximo
        while len(active_warriors) - num_to_eliminate + len(new_warriors) < self.max_battle_size:
            # Seleccionar padres (1 o 2)
            if len(parents) >= 2 and random.random() < 0.7:
                # Cruce entre dos padres
                parent1, parent2 = random.sample(parents, 2)
                parent_ids = [parent1.warrior_id, parent2.warrior_id]
                
                # Heredar estrategia y timeframe del primer padre
                strategy_name = parent1.strategy_name
                timeframe = parent1.timeframe
                leverage = max(parent1.leverage, parent2.leverage)
                
                # Mezclar y optimizar parámetros
                params = self._optimize_parameters(parent1.params, parent2.params)
            else:
                # Mutación de un solo padre
                parent = random.choice(parents)
                parent_ids = [parent.warrior_id]
                
                # Heredar estrategia y timeframe
                strategy_name = parent.strategy_name
                timeframe = parent.timeframe
                leverage = parent.leverage
                
                # Optimizar parámetros
                params = self._optimize_parameters(parent.params)
            
            # Crear nuevo guerrero
            new_id = f"{strategy_name}_{timeframe}_gen{next_generation}_{int(time.time())}"
            
            new_warrior = BotWarrior(
                warrior_id=new_id,
                strategy_name=strategy_name,
                generation=next_generation,
                symbol=self.symbol,
                timeframe=timeframe,
                leverage=leverage,
                params=params,
                parent_ids=parent_ids
            )
            
            # Activar automáticamente
            new_warrior.activate()
            
            # Añadir a la arena
            self.warriors[new_id] = new_warrior
            new_warriors.append(new_id)
            
            # Actualizar estadísticas
            self.arena_stats["total_warriors"] += 1
            
            # Registrar proceso de evolución
            evolution_entry = {
                "timestamp": datetime.now().isoformat(),
                "generation": next_generation,
                "new_warrior_id": new_id,
                "parent_ids": parent_ids,
                "strategy_name": strategy_name,
                "timeframe": timeframe,
                "params": params
            }
            
            self.evolution_log.append(evolution_entry)
        
        # Actualizar estadísticas de la arena
        self.arena_stats["total_generations"] = next_generation
        self.arena_stats["last_evaluation"] = datetime.now().isoformat()
        self.arena_stats["next_evaluation"] = (datetime.now() + timedelta(days=self.evaluation_period)).isoformat()
        self.arena_stats["total_battles"] += 1
        
        # Actualizar Hall of Fame con los mejores
        for winner in sorted_warriors[:3]:
            # Añadir al hall of fame si califica
            if len(self.arena_stats["hall_of_fame"]) < 10 or winner.battle_stats["roi"] > self.arena_stats["hall_of_fame"][-1]["roi"]:
                fame_entry = {
                    "warrior_id": winner.warrior_id,
                    "strategy_name": winner.strategy_name,
                    "timeframe": winner.timeframe,
                    "generation": winner.generation,
                    "roi": winner.battle_stats["roi"],
                    "win_rate": winner.battle_stats["win_rate"],
                    "total_trades": winner.battle_stats["total_trades"],
                    "recorded_at": datetime.now().isoformat()
                }
                
                # Añadir y ordenar
                self.arena_stats["hall_of_fame"].append(fame_entry)
                self.arena_stats["hall_of_fame"] = sorted(
                    self.arena_stats["hall_of_fame"],
                    key=lambda x: x["roi"],
                    reverse=True
                )[:10]  # Mantener solo los 10 mejores
        
        # Registrar evaluación
        evaluation_entry = {
            "timestamp": datetime.now().isoformat(),
            "total_warriors": len(active_warriors),
            "eliminated": eliminated_ids,
            "new_warriors": new_warriors,
            "best_roi": sorted_warriors[0].battle_stats["roi"] if sorted_warriors else 0,
            "worst_roi": sorted_warriors[-1].battle_stats["roi"] if sorted_warriors else 0,
            "average_roi": sum(w.battle_stats["roi"] for w in active_warriors.values()) / len(active_warriors) if active_warriors else 0
        }
        
        self.evaluation_history.append(evaluation_entry)
        
        # Guardar arena
        self._save_arena()
        
        logger.info(f"Evaluación completada: {len(eliminated_ids)} eliminados, {len(new_warriors)} nuevos guerreros")
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "active_warriors": len(active_warriors),
            "eliminated": eliminated_ids,
            "new_warriors": new_warriors,
            "next_evaluation": self.arena_stats["next_evaluation"]
        }
    
    def _optimize_parameters(self, 
                          params1: Dict[str, Any], 
                          params2: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimiza parámetros a partir de uno o dos conjuntos de parámetros.
        
        Args:
            params1: Primer conjunto de parámetros
            params2: Segundo conjunto de parámetros (opcional)
            
        Returns:
            Dict[str, Any]: Parámetros optimizados
        """
        # Si solo hay un conjunto de parámetros, mutarlo
        if params2 is None:
            new_params = params1.copy()
            
            # Aplicar mutaciones aleatorias
            for key, value in new_params.items():
                # Solo mutar valores numéricos
                if isinstance(value, (int, float)):
                    # Decidir si mutar este parámetro
                    if random.random() < self.optimization_rate:
                        if isinstance(value, int):
                            # Para enteros, cambiar en +/- 1-3 unidades
                            mutation = random.randint(-3, 3)
                            new_params[key] = max(1, value + mutation)
                        else:
                            # Para flotantes, cambiar en +/- 5-20%
                            mutation = random.uniform(-0.2, 0.2)
                            new_params[key] = max(0.01, value * (1 + mutation))
            
            return new_params
        
        # Si hay dos conjuntos, combinarlos
        new_params = {}
        
        # Unir todas las claves
        all_keys = set(params1.keys()) | set(params2.keys())
        
        for key in all_keys:
            # Si la clave está en ambos, decidir cuál usar o combinar
            if key in params1 and key in params2:
                val1 = params1[key]
                val2 = params2[key]
                
                # Si ambos son numéricos, posible combinación
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Decide entre heredar de un padre, promediar o mutar
                    r = random.random()
                    
                    if r < 0.4:
                        # Heredar de padre 1
                        new_params[key] = val1
                    elif r < 0.8:
                        # Heredar de padre 2
                        new_params[key] = val2
                    else:
                        # Promediar con posible mutación
                        avg = (val1 + val2) / 2
                        mutation = random.uniform(-0.1, 0.1)
                        
                        if isinstance(val1, int) and isinstance(val2, int):
                            new_params[key] = int(max(1, avg * (1 + mutation)))
                        else:
                            new_params[key] = max(0.01, avg * (1 + mutation))
                else:
                    # Para valores no numéricos, seleccionar aleatoriamente
                    new_params[key] = random.choice([val1, val2])
            else:
                # Si la clave está solo en uno, usarla
                new_params[key] = params1[key] if key in params1 else params2[key]
        
        return new_params
    
    def get_warrior_status(self, warrior_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene el estado de un guerrero.
        
        Args:
            warrior_id: ID del guerrero
            
        Returns:
            Optional[Dict[str, Any]]: Estado del guerrero o None si no existe
        """
        if warrior_id not in self.warriors:
            return None
        
        return self.warriors[warrior_id].get_battle_card()
    
    def get_all_warriors_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene el estado de todos los guerreros.
        
        Returns:
            Dict[str, Dict[str, Any]]: Estado de todos los guerreros
        """
        return {wid: w.get_battle_card() for wid, w in self.warriors.items()}
    
    def get_active_warriors(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene el estado de los guerreros activos.
        
        Returns:
            Dict[str, Dict[str, Any]]: Estado de los guerreros activos
        """
        return {wid: w.get_battle_card() for wid, w in self.warriors.items() if w.active and not w.defeated}
    
    def get_arena_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado de la arena.
        
        Returns:
            Dict[str, Any]: Estado de la arena
        """
        active_warriors = {wid: w for wid, w in self.warriors.items() if w.active and not w.defeated}
        
        # Calcular estadísticas
        total_balance = sum(w.current_balance for w in active_warriors.values())
        avg_roi = sum(w.battle_stats["roi"] for w in active_warriors.values()) / len(active_warriors) if active_warriors else 0
        avg_win_rate = sum(w.battle_stats["win_rate"] for w in active_warriors.values()) / len(active_warriors) if active_warriors else 0
        
        # Verificar si toca evaluación
        next_eval = None
        if self.arena_stats["next_evaluation"]:
            next_eval = datetime.fromisoformat(self.arena_stats["next_evaluation"])
            
        evaluation_due = next_eval and datetime.now() >= next_eval
        
        return {
            "active_warriors": len(active_warriors),
            "total_warriors": len(self.warriors),
            "total_defeated": self.arena_stats["total_defeated"],
            "total_generations": self.arena_stats["total_generations"],
            "total_battles": self.arena_stats["total_battles"],
            "total_balance": total_balance,
            "avg_roi": avg_roi,
            "avg_win_rate": avg_win_rate,
            "best_roi_ever": self.arena_stats["best_roi_ever"],
            "worst_roi_ever": self.arena_stats["worst_roi_ever"],
            "last_evaluation": self.arena_stats["last_evaluation"],
            "next_evaluation": self.arena_stats["next_evaluation"],
            "evaluation_due": evaluation_due,
            "hall_of_fame": self.arena_stats["hall_of_fame"][:3],  # Solo los 3 mejores
            "creation_date": self.arena_stats["creation_date"]
        }
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """
        Obtiene el tablero de clasificación.
        
        Returns:
            List[Dict[str, Any]]: Listado de guerreros ordenados por rendimiento
        """
        active_warriors = {wid: w for wid, w in self.warriors.items() if w.active and not w.defeated}
        
        # Obtener tarjetas de batalla y ordenar por ROI
        battle_cards = [w.get_battle_card() for w in active_warriors.values()]
        
        return sorted(battle_cards, key=lambda c: c["roi"], reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas detalladas de la arena.
        
        Returns:
            Dict[str, Any]: Estadísticas
        """
        active_warriors = {wid: w for wid, w in self.warriors.items() if w.active and not w.defeated}
        
        # Estrategias más exitosas
        strategy_stats = {}
        
        for w in active_warriors.values():
            strategy = w.strategy_name
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    "count": 0,
                    "total_roi": 0,
                    "total_win_rate": 0,
                    "total_trades": 0
                }
            
            strategy_stats[strategy]["count"] += 1
            strategy_stats[strategy]["total_roi"] += w.battle_stats["roi"]
            strategy_stats[strategy]["total_win_rate"] += w.battle_stats["win_rate"]
            strategy_stats[strategy]["total_trades"] += w.battle_stats["total_trades"]
        
        # Calcular promedios
        for strategy, stats in strategy_stats.items():
            if stats["count"] > 0:
                stats["avg_roi"] = stats["total_roi"] / stats["count"]
                stats["avg_win_rate"] = stats["total_win_rate"] / stats["count"]
                stats["avg_trades"] = stats["total_trades"] / stats["count"]
        
        # Timeframes más exitosos
        timeframe_stats = {}
        
        for w in active_warriors.values():
            timeframe = w.timeframe
            if timeframe not in timeframe_stats:
                timeframe_stats[timeframe] = {
                    "count": 0,
                    "total_roi": 0,
                    "total_win_rate": 0,
                    "total_trades": 0
                }
            
            timeframe_stats[timeframe]["count"] += 1
            timeframe_stats[timeframe]["total_roi"] += w.battle_stats["roi"]
            timeframe_stats[timeframe]["total_win_rate"] += w.battle_stats["win_rate"]
            timeframe_stats[timeframe]["total_trades"] += w.battle_stats["total_trades"]
        
        # Calcular promedios
        for timeframe, stats in timeframe_stats.items():
            if stats["count"] > 0:
                stats["avg_roi"] = stats["total_roi"] / stats["count"]
                stats["avg_win_rate"] = stats["total_win_rate"] / stats["count"]
                stats["avg_trades"] = stats["total_trades"] / stats["count"]
        
        # Estadísticas por generación
        generation_stats = {}
        
        for w in active_warriors.values():
            gen = w.generation
            if gen not in generation_stats:
                generation_stats[gen] = {
                    "count": 0,
                    "total_roi": 0,
                    "total_win_rate": 0
                }
            
            generation_stats[gen]["count"] += 1
            generation_stats[gen]["total_roi"] += w.battle_stats["roi"]
            generation_stats[gen]["total_win_rate"] += w.battle_stats["win_rate"]
        
        # Calcular promedios
        for gen, stats in generation_stats.items():
            if stats["count"] > 0:
                stats["avg_roi"] = stats["total_roi"] / stats["count"]
                stats["avg_win_rate"] = stats["total_win_rate"] / stats["count"]
        
        return {
            "strategy_stats": strategy_stats,
            "timeframe_stats": timeframe_stats,
            "generation_stats": generation_stats,
            "total_active": len(active_warriors),
            "total_all": len(self.warriors),
            "total_battles": self.arena_stats["total_battles"],
            "best_roi_ever": self.arena_stats["best_roi_ever"],
            "worst_roi_ever": self.arena_stats["worst_roi_ever"]
        }
    
    def generate_evolution_chart(self, output_file: str = "static/evolution_chart.png") -> bool:
        """
        Genera un gráfico de evolución de los guerreros.
        
        Args:
            output_file: Archivo donde guardar el gráfico
            
        Returns:
            bool: True si se generó correctamente
        """
        try:
            # Verificar si hay suficientes datos
            if len(self.evaluation_history) < 2:
                logger.warning("Datos insuficientes para generar gráfico de evolución")
                return False
            
            # Crear figura
            plt.figure(figsize=(12, 8))
            
            # Datos para el gráfico
            timestamps = [datetime.fromisoformat(e["timestamp"]) for e in self.evaluation_history]
            best_rois = [e["best_roi"] for e in self.evaluation_history]
            avg_rois = [e["average_roi"] for e in self.evaluation_history]
            worst_rois = [e["worst_roi"] for e in self.evaluation_history]
            
            # Graficar
            plt.plot(timestamps, best_rois, 'g-', label='Mejor ROI', linewidth=2)
            plt.plot(timestamps, avg_rois, 'b-', label='ROI Promedio', linewidth=2)
            plt.plot(timestamps, worst_rois, 'r-', label='Peor ROI', linewidth=2)
            
            # Añadir línea de base (0%)
            plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # Formato
            plt.title(f'Evolución de Rendimiento - Arena de Batalla de Bots', fontsize=16)
            plt.xlabel('Fecha de Evaluación', fontsize=12)
            plt.ylabel('Retorno sobre Inversión (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            
            # Anotar algunos puntos clave
            for i, e in enumerate(self.evaluation_history):
                if i % 5 == 0 or i == len(self.evaluation_history) - 1:  # Anotar cada 5 evaluaciones y la última
                    plt.annotate(
                        f"Gen {i+1}",
                        (timestamps[i], best_rois[i]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center'
                    )
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Guardar figura
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Gráfico de evolución generado: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error al generar gráfico de evolución: {e}")
            return False
    
    def generate_leaderboard_chart(self, output_file: str = "static/leaderboard_chart.png") -> bool:
        """
        Genera un gráfico del tablero de clasificación.
        
        Args:
            output_file: Archivo donde guardar el gráfico
            
        Returns:
            bool: True si se generó correctamente
        """
        try:
            # Obtener leaderboard
            leaderboard = self.get_leaderboard()
            
            if not leaderboard:
                logger.warning("No hay guerreros activos para generar gráfico")
                return False
            
            # Limitar a los 15 mejores
            leaderboard = leaderboard[:15]
            
            # Crear figura
            plt.figure(figsize=(12, 10))
            
            # Datos para el gráfico
            warrior_names = [f"{w['name']} (Gen {w['generation']})" for w in leaderboard]
            rois = [w["roi"] for w in leaderboard]
            colors = [w["color"] for w in leaderboard]
            
            # Crear barras horizontales
            bars = plt.barh(warrior_names, rois, color=colors, alpha=0.7)
            
            # Añadir valores
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_x_pos = width if width >= 0 else 0
                plt.text(
                    label_x_pos + 0.5, 
                    bar.get_y() + bar.get_height() / 2, 
                    f"{rois[i]:.2f}%", 
                    va='center'
                )
            
            # Formato
            plt.title(f'Tablero de Clasificación - Batalla de Bots', fontsize=16)
            plt.xlabel('Retorno sobre Inversión (%)', fontsize=12)
            plt.grid(True, alpha=0.3, axis='x')
            
            # Añadir línea de base (0%)
            plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
            
            # Invertir el eje y para que el mejor esté arriba
            plt.gca().invert_yaxis()
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Guardar figura
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Gráfico de leaderboard generado: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error al generar gráfico de leaderboard: {e}")
            return False
    
    def get_best_warrior(self) -> Optional[Dict[str, Any]]:
        """
        Obtiene al mejor guerrero actual.
        
        Returns:
            Optional[Dict[str, Any]]: Datos del mejor guerrero o None si no hay activos
        """
        leaderboard = self.get_leaderboard()
        
        if not leaderboard:
            return None
        
        return leaderboard[0]
    
    def create_standard_arena(self, strategies: List[str] = None) -> Dict[str, Any]:
        """
        Crea una arena estándar con bots predefinidos.
        
        Args:
            strategies: Lista de estrategias a incluir
            
        Returns:
            Dict[str, Any]: Resultado con los guerreros creados
        """
        # Estrategias por defecto
        if not strategies:
            strategies = [
                "breakout_scalping", 
                "momentum_scalping", 
                "mean_reversion", 
                "ml_adaptive"
            ]
        
        # Timeframes típicos
        timeframes = ["1m", "5m", "15m", "1h"]
        
        created_warriors = []
        
        # Crear bots para cada combinación
        for strategy in strategies:
            for timeframe in timeframes:
                warrior_id = f"{strategy}_{timeframe}_gen1_{int(time.time())}"
                
                # Variar tiempo para tener IDs únicos
                time.sleep(0.01)
                
                # Obtener parámetros
                params = self._get_default_params(strategy)
                
                # Crear guerrero
                self.add_warrior(
                    strategy_name=strategy,
                    timeframe=timeframe,
                    params=params,
                    warrior_id=warrior_id
                )
                
                # Activar
                self.activate_warrior(warrior_id)
                
                created_warriors.append(warrior_id)
        
        # Guardar arena
        self._save_arena()
        
        return {
            "success": True,
            "created_warriors": created_warriors,
            "total_warriors": len(created_warriors)
        }
    
    def _get_default_params(self, strategy_name: str) -> Dict[str, Any]:
        """
        Obtiene parámetros por defecto para una estrategia.
        
        Args:
            strategy_name: Nombre de la estrategia
            
        Returns:
            Dict[str, Any]: Parámetros por defecto
        """
        # Parámetros similares a los del módulo multi_bot_manager
        if strategy_name == "breakout_scalping":
            return {
                "take_profit_pct": 0.8,
                "stop_loss_pct": 0.5,
                "trailing_stop_pct": 0.3,
                "max_position_size_pct": 50.0,
                "min_volume_threshold": 1.5,
                "min_rr_ratio": 1.5,
                "max_fee_impact_pct": 0.15
            }
        elif strategy_name == "momentum_scalping":
            return {
                "take_profit_pct": 0.6,
                "stop_loss_pct": 0.4,
                "trailing_stop_pct": 0.2,
                "max_position_size_pct": 50.0,
                "rsi_period": 7,
                "ema_fast": 5,
                "ema_slow": 8,
                "min_rr_ratio": 1.5,
                "max_fee_impact_pct": 0.15
            }
        elif strategy_name == "mean_reversion":
            return {
                "take_profit_pct": 0.7,
                "stop_loss_pct": 0.5,
                "trailing_stop_pct": 0.3,
                "max_position_size_pct": 50.0,
                "bb_period": 20,
                "bb_std": 2.0,
                "rsi_period": 7,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "min_rr_ratio": 1.5,
                "max_fee_impact_pct": 0.15
            }
        elif strategy_name == "ml_adaptive":
            return {
                "confidence_threshold": 0.65,
                "take_profit_pct": 0.8,
                "stop_loss_pct": 0.6,
                "trailing_stop_pct": 0.3,
                "max_position_size_pct": 50.0,
                "min_rr_ratio": 1.5,
                "max_fee_impact_pct": 0.15
            }
        
        # Valores por defecto genéricos
        return {
            "take_profit_pct": 0.7,
            "stop_loss_pct": 0.5,
            "max_position_size_pct": 50.0
        }

def get_bot_battle_arena(arena_file: str = "data/bot_battle_arena.json") -> BotBattleArena:
    """
    Función de conveniencia para obtener una instancia de la arena.
    
    Args:
        arena_file: Archivo de configuración
        
    Returns:
        BotBattleArena: Instancia de la arena
    """
    return BotBattleArena(arena_file=arena_file)

def demo_bot_battle_arena():
    """Demostración de la Arena de Batalla de Bots."""
    print("\n⚔️ ARENA DE BATALLA DE BOTS DE TRADING ⚔️")
    print("¡Que comience la batalla por la supremacía algorítmica!")
    
    # Crear arena
    arena = BotBattleArena(arena_file="data/demo_bot_battle_arena.json")
    
    # Crear arena estándar con bots predefinidos
    result = arena.create_standard_arena()
    
    print(f"\n1. Arena creada con {result['total_warriors']} guerreros")
    
    # Simular actualizaciones de estado
    print("\n2. Simulando rendimiento de los guerreros:")
    active_warriors = arena.get_active_warriors()
    
    for i, (wid, warrior) in enumerate(list(active_warriors.items())[:5]):
        # Simular estado del bot
        bot_status = {
            "current_balance": 100 + random.uniform(-10, 30),
            "metrics": {
                "win_rate": random.uniform(40, 70),
                "total_trades": random.randint(10, 50),
                "profitable_trades": random.randint(5, 30),
                "losing_trades": random.randint(5, 20),
                "total_profit": random.uniform(20, 100),
                "total_loss": random.uniform(10, 50)
            },
            "daily_metrics": {
                "profit_percent": random.uniform(-2, 5)
            },
            "recent_trades": []
        }
        
        # Actualizar estado
        arena.update_warrior_status(wid, bot_status)
        
        print(f"   Guerrero {i+1}: {warrior['name']} - Balance: ${bot_status['current_balance']:.2f}, "
              f"Win rate: {bot_status['metrics']['win_rate']:.2f}%")
    
    # Obtener leaderboard
    leaderboard = arena.get_leaderboard()
    
    print("\n3. Tablero de clasificación actual:")
    for i, warrior in enumerate(leaderboard[:5]):
        print(f"   #{i+1}: {warrior['name']} (Gen {warrior['generation']}) - "
              f"ROI: {warrior['roi']:.2f}%, Win rate: {warrior['win_rate']:.2f}%")
    
    # Evaluar arena
    evaluation = arena.evaluate_arena()
    
    print("\n4. Evaluación completada:")
    print(f"   Guerreros eliminados: {len(evaluation['eliminated'])}")
    print(f"   Nuevos guerreros: {len(evaluation['new_warriors'])}")
    print(f"   Próxima evaluación: {evaluation['next_evaluation']}")
    
    # Estadísticas de la arena
    stats = arena.get_statistics()
    
    print("\n5. Estadísticas por estrategia:")
    for strategy, strategy_stats in stats["strategy_stats"].items():
        if "avg_roi" in strategy_stats:
            print(f"   {strategy}: ROI promedio {strategy_stats['avg_roi']:.2f}%, "
                  f"Win rate {strategy_stats['avg_win_rate']:.2f}%, "
                  f"{strategy_stats['count']} guerreros")
    
    print("\n6. Estadísticas por timeframe:")
    for timeframe, tf_stats in stats["timeframe_stats"].items():
        if "avg_roi" in tf_stats:
            print(f"   {timeframe}: ROI promedio {tf_stats['avg_roi']:.2f}%, "
                  f"Win rate {tf_stats['avg_win_rate']:.2f}%, "
                  f"{tf_stats['count']} guerreros")
    
    # Generar gráficos
    arena.generate_leaderboard_chart()
    
    print("\n✅ Demostración completada.")
    print("   Gráfico del leaderboard generado en static/leaderboard_chart.png")
    
    return arena

if __name__ == "__main__":
    try:
        arena = demo_bot_battle_arena()
    except Exception as e:
        print(f"Error en la demostración: {e}")