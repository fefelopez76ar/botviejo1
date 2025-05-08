#!/usr/bin/env python3
"""
Gestor Adaptativo de Posiciones.

Este módulo implementa un sistema avanzado para la gestión de posiciones,
incluyendo cierre escalonado de posiciones y stops dinámicos basados en ATR.
El sistema aprende y optimiza sus parámetros basándose en resultados históricos.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AdaptivePositionManager')

class AdaptivePositionManager:
    """
    Gestor adaptativo para posiciones de trading.
    
    Implementa estrategias avanzadas de gestión de posiciones, como:
    - Cierre escalonado de posiciones (25%, 50%, 25%)
    - Stops dinámicos basados en ATR
    - Optimización adaptativa de parámetros usando ML
    """
    
    def __init__(self, 
                symbol: str = "SOL-USDT", 
                data_file: str = "data/position_history.json",
                model_file: str = "data/position_model.json",
                default_trailing_percent: float = 0.5,
                default_tp1_percent: float = 0.5,
                default_tp2_percent: float = 1.0,
                default_tp3_percent: float = 1.5,
                default_atr_multiplier: float = 1.5,
                default_tp1_size: float = 0.25,
                default_tp2_size: float = 0.5,
                default_tp3_size: float = 0.25):
        """
        Inicializa el gestor de posiciones.
        
        Args:
            symbol: Par de trading (por defecto SOL-USDT)
            data_file: Archivo para almacenar historial de operaciones
            model_file: Archivo para almacenar modelos entrenados
            default_trailing_percent: % por defecto para trailing stop
            default_tp1_percent: % por defecto para primer take profit
            default_tp2_percent: % por defecto para segundo take profit
            default_tp3_percent: % por defecto para tercer take profit
            default_atr_multiplier: Multiplicador de ATR por defecto para stops
            default_tp1_size: Tamaño de la primera parte (25% por defecto)
            default_tp2_size: Tamaño de la segunda parte (50% por defecto) 
            default_tp3_size: Tamaño de la tercera parte (25% por defecto)
        """
        self.symbol = symbol
        self.data_file = data_file
        self.model_file = model_file
        
        # Parámetros por defecto
        self.params = {
            "trailing_percent": default_trailing_percent,
            "tp1_percent": default_tp1_percent,
            "tp2_percent": default_tp2_percent,
            "tp3_percent": default_tp3_percent,
            "atr_multiplier": default_atr_multiplier,
            "tp1_size": default_tp1_size,
            "tp2_size": default_tp2_size,
            "tp3_size": default_tp3_size
        }
        
        # Historial de posiciones
        self.position_history = []
        
        # Modelos de ML
        self.models = {
            "tp_model": None,
            "sl_model": None,
            "size_model": None
        }
        
        # Cargar datos históricos si existen
        self._load_data()
        
        # Cargar o entrenar modelos
        self._load_or_train_models()
    
    def _load_data(self):
        """Carga datos históricos desde archivo."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    self.position_history = json.load(f)
                logger.info(f"Cargados {len(self.position_history)} registros históricos de posiciones")
        except Exception as e:
            logger.error(f"Error al cargar datos históricos: {e}")
            self.position_history = []
    
    def _save_data(self):
        """Guarda datos históricos en archivo."""
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            
            with open(self.data_file, 'w') as f:
                json.dump(self.position_history, f, indent=4)
            logger.info(f"Guardados {len(self.position_history)} registros históricos de posiciones")
        except Exception as e:
            logger.error(f"Error al guardar datos históricos: {e}")
    
    def _load_or_train_models(self):
        """Carga modelos existentes o entrena nuevos si hay suficientes datos."""
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'r') as f:
                    model_params = json.load(f)
                
                # Reconstruir modelos a partir de parámetros
                self._rebuild_models(model_params)
                logger.info("Modelos de ML cargados correctamente")
            elif len(self.position_history) >= 50:
                # Entrenar modelos si hay suficientes datos
                self._train_models()
                logger.info("Nuevos modelos de ML entrenados")
            else:
                logger.info("Datos insuficientes para entrenar modelos, usando parámetros por defecto")
        except Exception as e:
            logger.error(f"Error al cargar/entrenar modelos: {e}")
    
    def _rebuild_models(self, model_params: Dict[str, Any]):
        """
        Reconstruye modelos a partir de parámetros guardados.
        
        Args:
            model_params: Diccionario con parámetros de modelos
        """
        # Implementación simplificada para ejemplo
        # En una implementación real, se usaría pickle para guardar/cargar modelos
        
        # Usar parámetros optimizados
        if "optimal_params" in model_params:
            self.params = model_params["optimal_params"]
        
        # Inicializar modelos vacíos
        # En una implementación real, se reconstruirían completamente
        self.models = {
            "tp_model": RandomForestRegressor(),
            "sl_model": RandomForestRegressor(),
            "size_model": RandomForestRegressor()
        }
    
    def _train_models(self):
        """Entrena modelos de ML basados en datos históricos."""
        if len(self.position_history) < 50:
            logger.warning("Datos insuficientes para entrenar modelos")
            return
        
        try:
            # Convertir historial a DataFrame
            df = pd.DataFrame(self.position_history)
            
            # Preparar características y etiquetas
            X = self._extract_features(df)
            
            # Entrenar modelo para optimizar take profits
            y_tp = df[['tp1_percent', 'tp2_percent', 'tp3_percent']].values
            X_train, X_test, y_tp_train, y_tp_test = train_test_split(X, y_tp, test_size=0.2)
            tp_model = RandomForestRegressor(n_estimators=100)
            tp_model.fit(X_train, y_tp_train)
            tp_pred = tp_model.predict(X_test)
            tp_mse = mean_squared_error(y_tp_test, tp_pred)
            
            # Entrenar modelo para optimizar stop loss (basado en ATR)
            y_sl = df[['atr_multiplier', 'trailing_percent']].values
            X_train, X_test, y_sl_train, y_sl_test = train_test_split(X, y_sl, test_size=0.2)
            sl_model = RandomForestRegressor(n_estimators=100)
            sl_model.fit(X_train, y_sl_train)
            sl_pred = sl_model.predict(X_test)
            sl_mse = mean_squared_error(y_sl_test, sl_pred)
            
            # Entrenar modelo para optimizar tamaños
            y_size = df[['tp1_size', 'tp2_size', 'tp3_size']].values
            X_train, X_test, y_size_train, y_size_test = train_test_split(X, y_size, test_size=0.2)
            size_model = RandomForestRegressor(n_estimators=100)
            size_model.fit(X_train, y_size_train)
            size_pred = size_model.predict(X_test)
            size_mse = mean_squared_error(y_size_test, size_pred)
            
            # Guardar modelos
            self.models = {
                "tp_model": tp_model,
                "sl_model": sl_model,
                "size_model": size_model
            }
            
            # Actualizar parámetros óptimos
            self._update_optimal_params()
            
            # Guardar modelos entrenados y métricas
            model_data = {
                "training_date": datetime.now().isoformat(),
                "training_samples": len(df),
                "metrics": {
                    "tp_mse": tp_mse,
                    "sl_mse": sl_mse,
                    "size_mse": size_mse
                },
                "optimal_params": self.params
            }
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            
            with open(self.model_file, 'w') as f:
                json.dump(model_data, f, indent=4)
            
            logger.info(f"Modelos entrenados y guardados: TP MSE={tp_mse:.4f}, SL MSE={sl_mse:.4f}, Size MSE={size_mse:.4f}")
        
        except Exception as e:
            logger.error(f"Error al entrenar modelos: {e}")
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extrae características para entrenar modelos.
        
        Args:
            df: DataFrame con historial de posiciones
            
        Returns:
            np.ndarray: Matriz de características
        """
        # Características básicas
        features = []
        
        # Añadir características si existen en el DataFrame
        for feature in ['volatility', 'market_trend', 'ema_trend', 'rsi_value', 'volume_ratio']:
            if feature in df.columns:
                features.append(df[feature].values.reshape(-1, 1))
        
        # Si no hay ninguna característica, usar valores simulados
        if not features:
            # Usar características simuladas para prueba
            features = [
                np.random.random(len(df)).reshape(-1, 1),
                np.random.random(len(df)).reshape(-1, 1),
                np.random.random(len(df)).reshape(-1, 1)
            ]
        
        # Combinar todas las características
        X = np.hstack(features)
        return X
    
    def _update_optimal_params(self):
        """Actualiza parámetros óptimos basados en modelos entrenados."""
        if all(model is not None for model in self.models.values()):
            try:
                # Usar características recientes para predecir
                recent_features = self._get_recent_features()
                
                if recent_features is not None:
                    # Predecir parámetros óptimos
                    tp_pred = self.models["tp_model"].predict([recent_features])[0]
                    sl_pred = self.models["sl_model"].predict([recent_features])[0]
                    size_pred = self.models["size_model"].predict([recent_features])[0]
                    
                    # Actualizar parámetros (con límites para evitar valores extremos)
                    self.params["tp1_percent"] = max(0.2, min(2.0, tp_pred[0]))
                    self.params["tp2_percent"] = max(0.5, min(3.0, tp_pred[1]))
                    self.params["tp3_percent"] = max(1.0, min(5.0, tp_pred[2]))
                    
                    self.params["atr_multiplier"] = max(0.5, min(3.0, sl_pred[0]))
                    self.params["trailing_percent"] = max(0.1, min(2.0, sl_pred[1]))
                    
                    # Asegurar que los tamaños sumen 1.0
                    size_pred = np.clip(size_pred, 0.1, 0.8)
                    total = np.sum(size_pred)
                    size_pred = size_pred / total if total > 0 else np.array([0.25, 0.5, 0.25])
                    
                    self.params["tp1_size"] = size_pred[0]
                    self.params["tp2_size"] = size_pred[1]
                    self.params["tp3_size"] = size_pred[2]
                    
                    logger.info(f"Parámetros óptimos actualizados: {self.params}")
            except Exception as e:
                logger.error(f"Error al actualizar parámetros óptimos: {e}")
    
    def _get_recent_features(self) -> Optional[np.ndarray]:
        """
        Obtiene características del mercado recientes.
        
        Returns:
            Optional[np.ndarray]: Vector de características o None si no disponible
        """
        try:
            # En una implementación real, esto obtendría datos recientes del mercado
            # Para este ejemplo, usamos valores simulados
            
            # Importar aquí para evitar dependencias circulares
            from data_management.market_data import MarketData
            
            # Crear instancia de MarketData
            md = MarketData()
            
            # Obtener datos recientes
            df = md.get_historical_data(self.symbol, "15m", limit=50)
            
            if df is not None and not df.empty:
                # Calcular indicadores
                df = md.calculate_indicators(df)
                
                # Obtener última fila
                last_row = df.iloc[-1]
                
                # Extraer características
                features = []
                
                # Volatilidad (ATR normalizado)
                if 'atr_pct' in last_row:
                    features.append(last_row['atr_pct'])
                else:
                    features.append(0.02)  # 2% por defecto
                
                # Tendencia del mercado (-1 a 1)
                if 'ema_trend' in last_row:
                    features.append(last_row['ema_trend'])
                else:
                    # Calcular tendencia basada en EMAs
                    if 'ema_9' in last_row and 'ema_21' in last_row:
                        trend = (last_row['ema_9'] / last_row['ema_21'] - 1) * 5  # Normalizar
                        features.append(max(-1, min(1, trend)))
                    else:
                        features.append(0)  # Neutral por defecto
                
                # RSI
                if 'rsi' in last_row:
                    features.append(last_row['rsi'] / 100)  # Normalizar a 0-1
                else:
                    features.append(0.5)  # 50 por defecto
                
                return np.array(features)
            
            # Si no hay datos, usar valores por defecto
            return np.array([0.02, 0, 0.5])  # volatilidad, tendencia, rsi
        
        except Exception as e:
            logger.error(f"Error al obtener características recientes: {e}")
            return None
    
    def set_position_plan(self, 
                        position: Dict[str, Any], 
                        market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Establece un plan de gestión para una posición.
        
        Args:
            position: Información de la posición
            market_data: Datos de mercado actuales
            
        Returns:
            Dict[str, Any]: Plan actualizado para la posición
        """
        # Obtener parámetros óptimos actualizados
        self._update_optimal_params()
        
        # Obtener ATR para stops dinámicos
        atr = self._get_atr_from_market_data(market_data)
        
        # Precio de entrada
        entry_price = position.get("entry_price", 0)
        
        if entry_price <= 0:
            logger.error("Precio de entrada inválido")
            return position
        
        # Dirección de la posición
        is_long = position.get("side", "long") == "long"
        
        # Calcular niveles de take profit
        if is_long:
            tp1_price = entry_price * (1 + self.params["tp1_percent"] / 100)
            tp2_price = entry_price * (1 + self.params["tp2_percent"] / 100)
            tp3_price = entry_price * (1 + self.params["tp3_percent"] / 100)
            
            # Stop loss basado en ATR
            stop_price = entry_price - (atr * self.params["atr_multiplier"])
            
            # Trailing stop
            trailing_activation = entry_price * (1 + self.params["trailing_percent"] / 100)
            trailing_distance = atr * self.params["atr_multiplier"]
        else:
            # Para posiciones cortas
            tp1_price = entry_price * (1 - self.params["tp1_percent"] / 100)
            tp2_price = entry_price * (1 - self.params["tp2_percent"] / 100)
            tp3_price = entry_price * (1 - self.params["tp3_percent"] / 100)
            
            # Stop loss basado en ATR
            stop_price = entry_price + (atr * self.params["atr_multiplier"])
            
            # Trailing stop
            trailing_activation = entry_price * (1 - self.params["trailing_percent"] / 100)
            trailing_distance = atr * self.params["atr_multiplier"]
        
        # Crear plan de posición
        position_plan = position.copy()
        
        # Añadir plan de take profits escalonados
        position_plan["tp_plan"] = {
            "tp1": {
                "price": tp1_price,
                "size": self.params["tp1_size"],
                "executed": False
            },
            "tp2": {
                "price": tp2_price,
                "size": self.params["tp2_size"],
                "executed": False
            },
            "tp3": {
                "price": tp3_price,
                "size": self.params["tp3_size"],
                "executed": False
            }
        }
        
        # Añadir plan de stops
        position_plan["sl_plan"] = {
            "initial_stop": stop_price,
            "current_stop": stop_price,
            "trailing_activation": trailing_activation,
            "trailing_distance": trailing_distance,
            "trailing_active": False
        }
        
        # ATR usado para el cálculo
        position_plan["atr_value"] = atr
        
        # Parámetros utilizados
        position_plan["params_used"] = self.params.copy()
        
        logger.info(f"Plan de posición creado para {position.get('side', 'unknown')} en {entry_price}")
        
        return position_plan
    
    def update_position(self, 
                       position_plan: Dict[str, Any], 
                       current_price: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Actualiza una posición según su plan, generando acciones si necesario.
        
        Args:
            position_plan: Plan de la posición
            current_price: Precio actual
            
        Returns:
            Tuple[Dict[str, Any], List[Dict[str, Any]]]: Plan actualizado y lista de acciones
        """
        if not position_plan or "tp_plan" not in position_plan or "sl_plan" not in position_plan:
            logger.error("Plan de posición inválido")
            return position_plan, []
        
        # Dirección de la posición
        is_long = position_plan.get("side", "long") == "long"
        
        # Copiar plan para actualizar
        updated_plan = position_plan.copy()
        
        # Lista de acciones a ejecutar
        actions = []
        
        # Comprobar stops primero
        sl_plan = updated_plan["sl_plan"]
        
        # Verificar si se ha alcanzado el stop loss
        stop_triggered = (is_long and current_price <= sl_plan["current_stop"]) or \
                          (not is_long and current_price >= sl_plan["current_stop"])
        
        if stop_triggered:
            # Generar acción de stop loss
            remaining_size = self._get_remaining_position_size(updated_plan)
            
            if remaining_size > 0:
                actions.append({
                    "action": "stop_loss",
                    "price": current_price,
                    "size": remaining_size,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Marcar todos los TP como ejecutados
                for tp_key in ["tp1", "tp2", "tp3"]:
                    updated_plan["tp_plan"][tp_key]["executed"] = True
                
                logger.info(f"Stop loss activado en {current_price}, tamaño: {remaining_size}")
                
                # Registrar operación para aprendizaje
                self._record_position_result(updated_plan, actions)
                
                return updated_plan, actions
        
        # Actualizar trailing stop si está activo
        if not sl_plan["trailing_active"]:
            # Verificar si se debe activar el trailing stop
            trail_activate = (is_long and current_price >= sl_plan["trailing_activation"]) or \
                             (not is_long and current_price <= sl_plan["trailing_activation"])
            
            if trail_activate:
                sl_plan["trailing_active"] = True
                logger.info(f"Trailing stop activado, precio: {current_price}")
        
        if sl_plan["trailing_active"]:
            # Actualizar nivel de trailing stop
            if is_long:
                new_stop = current_price - sl_plan["trailing_distance"]
                if new_stop > sl_plan["current_stop"]:
                    sl_plan["current_stop"] = new_stop
                    logger.info(f"Trailing stop actualizado a {new_stop}")
            else:
                new_stop = current_price + sl_plan["trailing_distance"]
                if new_stop < sl_plan["current_stop"]:
                    sl_plan["current_stop"] = new_stop
                    logger.info(f"Trailing stop actualizado a {new_stop}")
        
        # Comprobar take profits
        tp_plan = updated_plan["tp_plan"]
        
        # Verificar cada nivel de TP
        for tp_key in ["tp1", "tp2", "tp3"]:
            tp = tp_plan[tp_key]
            
            # Saltar si ya se ejecutó
            if tp["executed"]:
                continue
            
            # Verificar si se ha alcanzado el nivel
            tp_triggered = (is_long and current_price >= tp["price"]) or \
                           (not is_long and current_price <= tp["price"])
            
            if tp_triggered:
                # Calcular tamaño inicial
                initial_quantity = position_plan.get("quantity", 0)
                
                # Tamaño a ejecutar para este nivel
                execution_size = initial_quantity * tp["size"]
                
                # Generar acción
                actions.append({
                    "action": f"take_profit_{tp_key}",
                    "price": current_price,
                    "size": execution_size,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Marcar como ejecutado
                tp["executed"] = True
                
                logger.info(f"Take profit {tp_key} activado en {current_price}, tamaño: {execution_size}")
        
        # Si se ha cerrado toda la posición, registrar para aprendizaje
        all_executed = all(tp["executed"] for tp in [tp_plan["tp1"], tp_plan["tp2"], tp_plan["tp3"]])
        
        if all_executed and actions:
            self._record_position_result(updated_plan, actions)
        
        return updated_plan, actions
    
    def _get_atr_from_market_data(self, market_data: Dict[str, Any]) -> float:
        """
        Obtiene el valor de ATR desde los datos de mercado.
        
        Args:
            market_data: Datos de mercado
            
        Returns:
            float: Valor de ATR o estimación
        """
        # Verificar si hay datos de ATR
        df = market_data.get("dataframe")
        
        if df is not None and not df.empty and 'atr' in df.columns:
            atr = df['atr'].iloc[-1]
            
            # Si hay precio, calcular ATR como porcentaje
            if 'close' in df.columns:
                current_price = df['close'].iloc[-1]
                return atr / current_price * 100 if current_price > 0 else 1.0
            
            return atr
        
        # Si no hay ATR, estimar basado en volatilidad típica de Solana (2-5%)
        if self.symbol.startswith("SOL"):
            return market_data.get("current_price", 100) * 0.03  # 3% por defecto
        
        # Valor por defecto para otros símbolos
        return market_data.get("current_price", 100) * 0.02  # 2% por defecto
    
    def _get_remaining_position_size(self, position_plan: Dict[str, Any]) -> float:
        """
        Calcula el tamaño restante de una posición.
        
        Args:
            position_plan: Plan de la posición
            
        Returns:
            float: Tamaño restante
        """
        # Tamaño inicial
        initial_size = position_plan.get("quantity", 0)
        
        # Si no hay plan de TP, retornar todo el tamaño
        if "tp_plan" not in position_plan:
            return initial_size
        
        # Sumar tamaños ya ejecutados
        executed_size = 0
        tp_plan = position_plan["tp_plan"]
        
        for tp_key in ["tp1", "tp2", "tp3"]:
            if tp_plan[tp_key]["executed"]:
                executed_size += initial_size * tp_plan[tp_key]["size"]
        
        # Retornar tamaño restante
        return max(0, initial_size - executed_size)
    
    def _record_position_result(self, position_plan: Dict[str, Any], actions: List[Dict[str, Any]]):
        """
        Registra resultado de una posición para aprendizaje.
        
        Args:
            position_plan: Plan de la posición
            actions: Acciones ejecutadas
        """
        try:
            # Verificar si hay acciones
            if not actions:
                return
            
            # Extraer datos relevantes
            entry_price = position_plan.get("entry_price", 0)
            entry_time = position_plan.get("entry_time", datetime.now().isoformat())
            
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            
            # Calcular precio medio de salida ponderado
            total_exit_value = 0
            total_exit_size = 0
            exit_time = None
            
            for action in actions:
                size = action.get("size", 0)
                price = action.get("price", 0)
                timestamp = action.get("timestamp")
                
                if size > 0 and price > 0:
                    total_exit_value += size * price
                    total_exit_size += size
                
                # Guardar tiempo de la última acción
                if timestamp:
                    if isinstance(timestamp, str):
                        action_time = datetime.fromisoformat(timestamp)
                    else:
                        action_time = timestamp
                    
                    if exit_time is None or action_time > exit_time:
                        exit_time = action_time
            
            # Precio medio de salida
            avg_exit_price = total_exit_value / total_exit_size if total_exit_size > 0 else 0
            
            # Duración de la posición
            duration = None
            if exit_time and entry_time:
                duration = (exit_time - entry_time).total_seconds() / 3600  # en horas
            
            # P&L
            is_long = position_plan.get("side", "long") == "long"
            
            if is_long:
                pnl_percent = (avg_exit_price / entry_price - 1) * 100 if entry_price > 0 else 0
            else:
                pnl_percent = (1 - avg_exit_price / entry_price) * 100 if entry_price > 0 else 0
            
            # Obtener parámetros utilizados
            params_used = position_plan.get("params_used", {})
            
            # Construir registro
            record = {
                "symbol": position_plan.get("symbol", self.symbol),
                "side": position_plan.get("side", "long"),
                "entry_price": entry_price,
                "avg_exit_price": avg_exit_price,
                "pnl_percent": pnl_percent,
                "entry_time": entry_time.isoformat() if isinstance(entry_time, datetime) else entry_time,
                "exit_time": exit_time.isoformat() if isinstance(exit_time, datetime) else exit_time,
                "duration_hours": duration,
                "initial_quantity": position_plan.get("quantity", 0),
                "tp1_percent": params_used.get("tp1_percent", self.params["tp1_percent"]),
                "tp2_percent": params_used.get("tp2_percent", self.params["tp2_percent"]),
                "tp3_percent": params_used.get("tp3_percent", self.params["tp3_percent"]),
                "atr_multiplier": params_used.get("atr_multiplier", self.params["atr_multiplier"]),
                "trailing_percent": params_used.get("trailing_percent", self.params["trailing_percent"]),
                "tp1_size": params_used.get("tp1_size", self.params["tp1_size"]),
                "tp2_size": params_used.get("tp2_size", self.params["tp2_size"]),
                "tp3_size": params_used.get("tp3_size", self.params["tp3_size"]),
                "atr_value": position_plan.get("atr_value", 0),
                "market_type": position_plan.get("market_type", "spot"),
                "leverage": position_plan.get("leverage", 1),
                "strategy": position_plan.get("strategy", "unknown"),
                "actions": [
                    {
                        "action": a.get("action", "unknown"),
                        "price": a.get("price", 0),
                        "size": a.get("size", 0),
                        "timestamp": a.get("timestamp", "")
                    }
                    for a in actions
                ],
                # Características del mercado en momento de entrada
                "volatility": 0,  # Se añadiría en implementación real
                "market_trend": 0,  # Se añadiría en implementación real
                "rsi_value": 0  # Se añadiría en implementación real
            }
            
            # Añadir al historial
            self.position_history.append(record)
            
            # Guardar historial
            self._save_data()
            
            # Re-entrenar modelos si hay suficientes datos
            if len(self.position_history) % 10 == 0 and len(self.position_history) >= 50:
                self._train_models()
            
            logger.info(f"Resultado de posición registrado: PnL {pnl_percent:.2f}%, duración {duration:.1f}h")
        
        except Exception as e:
            logger.error(f"Error al registrar resultado de posición: {e}")
    
    def get_parameters(self, market_condition: str = None) -> Dict[str, Any]:
        """
        Obtiene parámetros óptimos, opcionalmente para una condición de mercado.
        
        Args:
            market_condition: Condición de mercado (opcional)
            
        Returns:
            Dict[str, Any]: Parámetros óptimos
        """
        # Actualizar parámetros si hay modelos
        if all(model is not None for model in self.models.values()):
            self._update_optimal_params()
        
        # En una implementación más avanzada, se retornarían parámetros
        # específicos para diferentes condiciones de mercado
        if market_condition:
            # Ejemplo: ajustar parámetros según condición
            if market_condition == "high_volatility":
                # En alta volatilidad, targets más amplios
                params = self.params.copy()
                params["tp1_percent"] *= 1.2
                params["tp2_percent"] *= 1.2
                params["tp3_percent"] *= 1.2
                params["atr_multiplier"] *= 1.3  # Stops más amplios
                return params
            elif market_condition == "trending":
                # En mercado con tendencia, más trailing stop
                params = self.params.copy()
                params["trailing_percent"] *= 0.8  # Activar trailing antes
                return params
            elif market_condition == "ranging":
                # En mercado lateral, tomar ganancias antes
                params = self.params.copy()
                params["tp1_percent"] *= 0.8
                params["tp2_percent"] *= 0.8
                params["tp3_percent"] *= 0.8
                return params
        
        return self.params
    
    def get_position_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de gestión de posiciones.
        
        Returns:
            Dict[str, Any]: Estadísticas
        """
        if not self.position_history:
            return {
                "total_positions": 0,
                "win_rate": 0,
                "avg_profit": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "avg_duration": 0,
                "best_trade": 0,
                "worst_trade": 0,
                "optimization_level": "Sin datos suficientes"
            }
        
        try:
            # Convertir a DataFrame para análisis
            df = pd.DataFrame(self.position_history)
            
            # Estadísticas básicas
            total_positions = len(df)
            profitable_trades = len(df[df['pnl_percent'] > 0])
            losing_trades = len(df[df['pnl_percent'] <= 0])
            
            win_rate = profitable_trades / total_positions if total_positions > 0 else 0
            
            # Ganancias y pérdidas
            avg_profit = df[df['pnl_percent'] > 0]['pnl_percent'].mean() if profitable_trades > 0 else 0
            avg_loss = df[df['pnl_percent'] <= 0]['pnl_percent'].mean() if losing_trades > 0 else 0
            
            total_profit = df[df['pnl_percent'] > 0]['pnl_percent'].sum()
            total_loss = abs(df[df['pnl_percent'] <= 0]['pnl_percent'].sum())
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Duración
            avg_duration = df['duration_hours'].mean() if 'duration_hours' in df.columns else 0
            
            # Mejor y peor operación
            best_trade = df['pnl_percent'].max()
            worst_trade = df['pnl_percent'].min()
            
            # Nivel de optimización
            if total_positions < 20:
                optimization_level = "Insuficiente (< 20 operaciones)"
            elif total_positions < 50:
                optimization_level = "Básico (20-50 operaciones)"
            elif total_positions < 100:
                optimization_level = "Intermedio (50-100 operaciones)"
            else:
                optimization_level = "Avanzado (> 100 operaciones)"
            
            return {
                "total_positions": total_positions,
                "win_rate": win_rate * 100,  # en porcentaje
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "avg_duration": avg_duration,
                "best_trade": best_trade,
                "worst_trade": worst_trade,
                "optimization_level": optimization_level,
                "models_trained": all(model is not None for model in self.models.values())
            }
        
        except Exception as e:
            logger.error(f"Error al calcular estadísticas: {e}")
            return {
                "total_positions": len(self.position_history),
                "error": str(e)
            }
    
    def get_optimization_suggestions(self) -> Dict[str, Any]:
        """
        Obtiene sugerencias para mejorar la gestión de posiciones.
        
        Returns:
            Dict[str, Any]: Sugerencias
        """
        if not self.position_history or len(self.position_history) < 20:
            return {
                "message": "Datos insuficientes para sugerencias de optimización",
                "min_trades_needed": 20,
                "current_trades": len(self.position_history)
            }
        
        try:
            # Convertir a DataFrame para análisis
            df = pd.DataFrame(self.position_history)
            
            suggestions = {
                "message": "Sugerencias para optimizar la gestión de posiciones",
                "parameter_suggestions": [],
                "strategy_suggestions": []
            }
            
            # Analizar rendimiento por duración
            if 'duration_hours' in df.columns:
                duration_bins = pd.cut(df['duration_hours'], bins=[0, 1, 4, 12, 24, float('inf')])
                duration_stats = df.groupby(duration_bins)['pnl_percent'].mean()
                
                best_duration = duration_stats.idxmax()
                if best_duration is not None:
                    suggestions["strategy_suggestions"].append(
                        f"Las operaciones con duración {best_duration} tienen mejor rendimiento "
                        f"({duration_stats[best_duration]:.2f}% promedio)."
                    )
            
            # Analizar rendimiento por parámetros
            for param in ['tp1_percent', 'tp2_percent', 'tp3_percent', 'atr_multiplier', 'trailing_percent']:
                if param in df.columns:
                    # Crear bins para analizar
                    param_bins = pd.qcut(df[param], q=4, duplicates='drop')
                    param_stats = df.groupby(param_bins)['pnl_percent'].mean()
                    
                    if not param_stats.empty:
                        best_param = param_stats.idxmax()
                        if best_param is not None:
                            param_name = param.replace('_percent', '%').replace('_', ' ')
                            suggestions["parameter_suggestions"].append(
                                f"El rango óptimo para {param_name} es {best_param} "
                                f"({param_stats[best_param]:.2f}% promedio)."
                            )
            
            # Analizar rendimiento por tamaño de posición
            for size_param in ['tp1_size', 'tp2_size', 'tp3_size']:
                if size_param in df.columns:
                    # Crear bins para analizar
                    try:
                        size_bins = pd.qcut(df[size_param], q=3, duplicates='drop')
                        size_stats = df.groupby(size_bins)['pnl_percent'].mean()
                        
                        if not size_stats.empty:
                            best_size = size_stats.idxmax()
                            if best_size is not None:
                                size_name = size_param.replace('_size', '').replace('_', ' ')
                                suggestions["parameter_suggestions"].append(
                                    f"El tamaño óptimo para {size_name} es {best_size} "
                                    f"({size_stats[best_size]:.2f}% promedio)."
                                )
                    except Exception:
                        # Puede fallar si hay poca variación
                        pass
            
            # Añadir sugerencias sobre estrategias si hay datos
            if 'strategy' in df.columns:
                strategy_stats = df.groupby('strategy')['pnl_percent'].agg(['mean', 'count'])
                strategy_stats = strategy_stats[strategy_stats['count'] >= 5]  # Al menos 5 operaciones
                
                if not strategy_stats.empty:
                    best_strategy = strategy_stats['mean'].idxmax()
                    suggestions["strategy_suggestions"].append(
                        f"La estrategia '{best_strategy}' tiene el mejor rendimiento "
                        f"({strategy_stats.loc[best_strategy, 'mean']:.2f}% promedio en "
                        f"{strategy_stats.loc[best_strategy, 'count']} operaciones)."
                    )
            
            # Si hay modelos entrenados, mencionar
            if all(model is not None for model in self.models.values()):
                suggestions["strategy_suggestions"].append(
                    "Los modelos de ML están entrenados y optimizando parámetros automáticamente."
                )
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error al generar sugerencias: {e}")
            return {
                "message": f"Error al generar sugerencias: {e}",
                "current_trades": len(self.position_history)
            }
    
    def get_ai_assistance(self, query: str) -> str:
        """
        Obtiene asistencia de IA basada en datos históricos.
        
        Args:
            query: Consulta o pregunta
            
        Returns:
            str: Respuesta generada
        """
        # En una implementación real, esto conectaría con un modelo de lenguaje (GPT, Claude, etc.)
        # Para este prototipo, retornamos respuestas pregeneradas según palabras clave
        
        query = query.lower()
        
        # Obtener estadísticas para enriquecer respuestas
        stats = self.get_position_stats()
        
        if "mejores parámetros" in query or "configuración óptima" in query:
            return (
                f"Basado en el análisis de {stats['total_positions']} operaciones, "
                f"los parámetros óptimos actuales son:\n"
                f"- Take Profit 1: {self.params['tp1_percent']:.2f}% con {self.params['tp1_size']*100:.0f}% del tamaño\n"
                f"- Take Profit 2: {self.params['tp2_percent']:.2f}% con {self.params['tp2_size']*100:.0f}% del tamaño\n"
                f"- Take Profit 3: {self.params['tp3_percent']:.2f}% con {self.params['tp3_size']*100:.0f}% del tamaño\n"
                f"- Multiplicador ATR: {self.params['atr_multiplier']:.2f}\n"
                f"- Activación Trailing: {self.params['trailing_percent']:.2f}%\n\n"
                f"Win rate actual: {stats['win_rate']:.2f}%, Factor de beneficio: {stats['profit_factor']:.2f}"
            )
        elif "stop loss" in query:
            return (
                f"El sistema usa stops dinámicos basados en ATR con un multiplicador de {self.params['atr_multiplier']:.2f}. "
                f"Esto adapta automáticamente el stop loss según la volatilidad actual de {self.symbol}. "
                f"También implementamos trailing stops que se activan cuando el precio alcanza {self.params['trailing_percent']:.2f}% "
                f"de ganancia, protegiendo beneficios mientras deja margen para que la operación respire."
            )
        elif "take profit" in query or "tp" in query:
            return (
                f"Usamos un sistema de take profit escalonado en 3 niveles:\n"
                f"1. Primer objetivo: {self.params['tp1_percent']:.2f}% cerrando {self.params['tp1_size']*100:.0f}% de la posición\n"
                f"2. Segundo objetivo: {self.params['tp2_percent']:.2f}% cerrando {self.params['tp2_size']*100:.0f}% de la posición\n"
                f"3. Tercer objetivo: {self.params['tp3_percent']:.2f}% cerrando el {self.params['tp3_size']*100:.0f}% restante\n\n"
                f"Este enfoque escalonado captura ganancias mientras permite que parte de la posición continúe si el movimiento se extiende."
            )
        elif "solana" in query or "sol" in query:
            return (
                f"Los parámetros actuales están optimizados para Solana basado en su perfil de volatilidad. "
                f"El ATR típico de Solana actualmente requiere un multiplicador de {self.params['atr_multiplier']:.2f} para los stops. "
                f"El sistema aprendió que para Solana, los objetivos de take profit en {self.params['tp1_percent']:.2f}%, "
                f"{self.params['tp2_percent']:.2f}% y {self.params['tp3_percent']:.2f}% ofrecen el mejor balance entre capturar "
                f"ganancias y maximizar movimientos extendidos."
            )
        elif "aprendizaje" in query or "ml" in query or "ia" in query:
            return (
                f"El sistema utiliza modelos de Random Forest para optimizar continuamente los parámetros de gestión de posiciones. "
                f"Actualmente hay {len(self.position_history)} operaciones en la base de datos de aprendizaje. "
                f"Los modelos {'' if stats.get('models_trained', False) else 'no '}están entrenados y "
                f"el nivel de optimización es '{stats.get('optimization_level', 'Desconocido')}'. "
                f"El sistema aprende de cada operación cerrada para ajustar parámetros de take profit, "
                f"stop loss y distribución de tamaños."
            )
        elif "estadísticas" in query or "rendimiento" in query:
            return (
                f"Estadísticas actuales basadas en {stats['total_positions']} operaciones:\n"
                f"- Win rate: {stats['win_rate']:.2f}%\n"
                f"- Ganancia media: {stats['avg_profit']:.2f}%\n"
                f"- Pérdida media: {stats['avg_loss']:.2f}%\n"
                f"- Factor de beneficio: {stats['profit_factor']:.2f}\n"
                f"- Duración media: {stats['avg_duration']:.2f} horas\n"
                f"- Mejor operación: {stats['best_trade']:.2f}%\n"
                f"- Peor operación: {stats['worst_trade']:.2f}%"
            )
        else:
            # Respuesta general
            return (
                f"Soy tu asistente para la gestión de posiciones de trading. Puedo ayudarte con:\n"
                f"- Información sobre parámetros óptimos\n"
                f"- Estrategias de stop loss y take profit\n"
                f"- Rendimiento y estadísticas\n"
                f"- Optimizaciones específicas para Solana\n"
                f"- Información sobre el sistema de aprendizaje automático\n\n"
                f"Actualmente gestionamos posiciones con un win rate del {stats['win_rate']:.2f}% "
                f"basado en {stats['total_positions']} operaciones históricas."
            )

def get_position_manager(symbol: str = "SOL-USDT") -> AdaptivePositionManager:
    """
    Función de conveniencia para obtener un gestor de posiciones.
    
    Args:
        symbol: Par de trading
        
    Returns:
        AdaptivePositionManager: Instancia del gestor
    """
    data_file = f"data/position_history_{symbol.replace('-', '')}.json"
    model_file = f"data/position_model_{symbol.replace('-', '')}.json"
    
    return AdaptivePositionManager(symbol=symbol, data_file=data_file, model_file=model_file)

def demo_position_manager():
    """Demostración del gestor adaptativo de posiciones."""
    print("\n🎯 GESTOR ADAPTATIVO DE POSICIONES 🎯")
    print("Este sistema implementa cierre escalonado de posiciones y stops dinámicos")
    print("con aprendizaje automático para optimizar los parámetros.")
    
    # Crear gestor para SOL-USDT
    manager = AdaptivePositionManager(symbol="SOL-USDT")
    
    # Mostrar parámetros actuales
    print("\n1. Parámetros actuales:")
    for param, value in manager.params.items():
        print(f"   {param}: {value}")
    
    # Simular creación de un plan de posición
    entry_price = 150.0
    position = {
        "side": "long",
        "entry_price": entry_price,
        "quantity": 1.0,
        "entry_time": datetime.now().isoformat(),
        "symbol": "SOL-USDT",
        "strategy": "demo_strategy",
        "market_type": "spot"
    }
    
    # Datos de mercado simulados
    market_data = {
        "dataframe": None,
        "current_price": entry_price,
        "orderbook": {},
        "timestamp": datetime.now()
    }
    
    # Crear plan de posición
    position_plan = manager.set_position_plan(position, market_data)
    
    print("\n2. Plan de posición creado:")
    print(f"   Entrada: ${position_plan['entry_price']:.2f}")
    print("   Take Profits:")
    for tp_key, tp in position_plan["tp_plan"].items():
        print(f"     {tp_key}: ${tp['price']:.2f} ({tp['size']*100:.0f}%)")
    
    print("   Stop Loss:")
    sl_plan = position_plan["sl_plan"]
    print(f"     Inicial: ${sl_plan['initial_stop']:.2f}")
    print(f"     Trailing activación: ${sl_plan['trailing_activation']:.2f}")
    print(f"     Trailing distancia: ${sl_plan['trailing_distance']:.2f}")
    
    # Simular actualización de precio
    print("\n3. Simulación de movimiento de precio:")
    
    # Primer movimiento: alcanza TP1
    tp1_price = position_plan["tp_plan"]["tp1"]["price"]
    print(f"   Precio sube a nivel TP1: ${tp1_price:.2f}")
    updated_plan, actions = manager.update_position(position_plan, tp1_price)
    
    print(f"   Acciones generadas: {len(actions)}")
    for action in actions:
        print(f"     {action['action']} a ${action['price']:.2f}, tamaño: {action['size']:.2f}")
    
    # Segundo movimiento: precio sigue subiendo
    better_price = tp1_price * 1.01
    print(f"   Precio sigue subiendo: ${better_price:.2f}")
    updated_plan, actions = manager.update_position(updated_plan, better_price)
    
    # Tercer movimiento: alcanza TP2
    tp2_price = position_plan["tp_plan"]["tp2"]["price"]
    print(f"   Precio sube a nivel TP2: ${tp2_price:.2f}")
    updated_plan, actions = manager.update_position(updated_plan, tp2_price)
    
    print(f"   Acciones generadas: {len(actions)}")
    for action in actions:
        print(f"     {action['action']} a ${action['price']:.2f}, tamaño: {action['size']:.2f}")
    
    # Obtener estadísticas
    stats = manager.get_position_stats()
    
    print("\n4. Estadísticas actuales:")
    print(f"   Total posiciones: {stats['total_positions']}")
    print(f"   Win rate: {stats['win_rate']:.2f}%")
    print(f"   Factor de beneficio: {stats['profit_factor']:.2f}")
    print(f"   Nivel de optimización: {stats['optimization_level']}")
    
    # Asistencia IA
    query = "¿Cuáles son los mejores parámetros para Solana?"
    response = manager.get_ai_assistance(query)
    
    print("\n5. Asistencia IA:")
    print(f"   Consulta: {query}")
    print(f"   Respuesta: {response}")
    
    print("\n✅ Demostración completada.")
    
    return manager

if __name__ == "__main__":
    try:
        manager = demo_position_manager()
    except Exception as e:
        print(f"Error en la demostración: {e}")