#!/usr/bin/env python3
"""
Módulo de ponderación adaptativa de indicadores para trading algorítmico de Solana.

Este módulo implementa:
- Optimización de pesos para indicadores técnicos usando aprendizaje automático
- Evaluación de eficacia de indicadores en diferentes condiciones de mercado
- Ajuste adaptativo de estrategias según desempeño histórico
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('IndicatorWeighting')

class MarketCondition(Enum):
    """Condiciones de mercado identificables"""
    STRONG_UPTREND = "strong_uptrend"
    MODERATE_UPTREND = "moderate_uptrend" 
    LATERAL_LOW_VOL = "lateral_low_vol"
    LATERAL_HIGH_VOL = "lateral_high_vol"
    MODERATE_DOWNTREND = "moderate_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    EXTREME_VOLATILITY = "extreme_volatility"
    
class TimeInterval(Enum):
    """Intervalos de tiempo para análisis"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"

class IndicatorWeighting:
    """
    Sistema de ponderación adaptativa para indicadores técnicos utilizando
    machine learning para optimizar los pesos según desempeño histórico.
    """
    
    def __init__(self, 
                training_data_file: str = None, 
                indicator_performance_file: str = "indicator_performance.json",
                use_ml: bool = True):
        """
        Inicializa el sistema de ponderación adaptativa.
        
        Args:
            training_data_file: Archivo con datos históricos para entrenamiento
            indicator_performance_file: Archivo para almacenar rendimiento de indicadores
            use_ml: Si se utiliza aprendizaje automático para optimización
        """
        self.training_data_file = training_data_file
        self.indicator_performance_file = indicator_performance_file
        self.use_ml = use_ml
        
        # Inicialización de pesos predeterminados
        self.default_weights = {
            'rsi': 0.20,
            'macd': 0.20,
            'bollinger': 0.15, 
            'sma': 0.10,
            'ema': 0.15,
            'stochastic': 0.10,
            'adx': 0.05,
            'cci': 0.05
        }
        
        # Pesos optimizados por ML (serán actualizados)
        self.optimized_weights = self.default_weights.copy()
        
        # Rendimiento de indicadores por condición de mercado
        self.indicator_performance = {}
        
        # Modelos de ML para predicción
        self.models = {}
        
        # Cargar datos de rendimiento si existen
        self._load_performance_data()
        
        # Entrenar modelos si hay datos disponibles
        if self.use_ml and self.training_data_file and os.path.exists(self.training_data_file):
            self.train_models()
    
    def _load_performance_data(self) -> None:
        """Carga datos de rendimiento de indicadores desde archivo."""
        try:
            if os.path.exists(self.indicator_performance_file):
                with open(self.indicator_performance_file, 'r') as f:
                    self.indicator_performance = json.load(f)
                logger.info(f"Datos de rendimiento cargados desde {self.indicator_performance_file}")
            else:
                logger.info("No se encontraron datos de rendimiento previos. Usando configuración por defecto.")
                # Inicialización de estructura de rendimiento para cada indicador
                for indicator in self.default_weights.keys():
                    self.indicator_performance[indicator] = {
                        'overall_accuracy': 0.0,
                        'signals_count': 0,
                        'correct_signals': 0,
                        'market_conditions': {condition.value: {
                            'accuracy': 0.0, 
                            'count': 0, 
                            'correct': 0
                        } for condition in MarketCondition},
                        'time_intervals': {interval.value: {
                            'accuracy': 0.0, 
                            'count': 0, 
                            'correct': 0
                        } for interval in TimeInterval}
                    }
        except Exception as e:
            logger.error(f"Error al cargar datos de rendimiento: {e}")
            # Inicialización por defecto si hay error
            for indicator in self.default_weights.keys():
                self.indicator_performance[indicator] = {
                    'overall_accuracy': 0.0,
                    'signals_count': 0,
                    'correct_signals': 0,
                    'market_conditions': {},
                    'time_intervals': {}
                }
    
    def _save_performance_data(self) -> None:
        """Guarda datos de rendimiento de indicadores en archivo."""
        try:
            with open(self.indicator_performance_file, 'w') as f:
                json.dump(self.indicator_performance, f, indent=4)
            logger.info(f"Datos de rendimiento guardados en {self.indicator_performance_file}")
        except Exception as e:
            logger.error(f"Error al guardar datos de rendimiento: {e}")
    
    def update_indicator_performance(self, 
                                   indicator: str, 
                                   correct: bool, 
                                   market_condition: Union[MarketCondition, str],
                                   time_interval: Union[TimeInterval, str],
                                   profit_pct: float = 0.0) -> None:
        """
        Actualiza el rendimiento de un indicador basado en el resultado de una señal.
        
        Args:
            indicator: Nombre del indicador
            correct: Si la señal fue correcta
            market_condition: Condición actual del mercado
            time_interval: Intervalo de tiempo usado
            profit_pct: Porcentaje de ganancia/pérdida
        """
        # Convertir enums a strings si es necesario
        if isinstance(market_condition, MarketCondition):
            market_condition = market_condition.value
        if isinstance(time_interval, TimeInterval):
            time_interval = time_interval.value
            
        # Asegurarse de que el indicador existe en el registro
        if indicator not in self.indicator_performance:
            self.indicator_performance[indicator] = {
                'overall_accuracy': 0.0,
                'signals_count': 0,
                'correct_signals': 0,
                'market_conditions': {},
                'time_intervals': {}
            }
            
        # Actualizar estadísticas generales
        self.indicator_performance[indicator]['signals_count'] += 1
        if correct:
            self.indicator_performance[indicator]['correct_signals'] += 1
        
        # Calcular precisión general actualizada
        self.indicator_performance[indicator]['overall_accuracy'] = (
            self.indicator_performance[indicator]['correct_signals'] / 
            self.indicator_performance[indicator]['signals_count']
        )
        
        # Actualizar rendimiento por condición de mercado
        if market_condition not in self.indicator_performance[indicator]['market_conditions']:
            self.indicator_performance[indicator]['market_conditions'][market_condition] = {
                'accuracy': 0.0,
                'count': 0,
                'correct': 0
            }
            
        self.indicator_performance[indicator]['market_conditions'][market_condition]['count'] += 1
        if correct:
            self.indicator_performance[indicator]['market_conditions'][market_condition]['correct'] += 1
            
        # Actualizar precisión para esta condición de mercado
        self.indicator_performance[indicator]['market_conditions'][market_condition]['accuracy'] = (
            self.indicator_performance[indicator]['market_conditions'][market_condition]['correct'] /
            self.indicator_performance[indicator]['market_conditions'][market_condition]['count']
        )
        
        # Actualizar rendimiento por intervalo de tiempo
        if time_interval not in self.indicator_performance[indicator]['time_intervals']:
            self.indicator_performance[indicator]['time_intervals'][time_interval] = {
                'accuracy': 0.0,
                'count': 0,
                'correct': 0
            }
            
        self.indicator_performance[indicator]['time_intervals'][time_interval]['count'] += 1
        if correct:
            self.indicator_performance[indicator]['time_intervals'][time_interval]['correct'] += 1
            
        # Actualizar precisión para este intervalo de tiempo
        self.indicator_performance[indicator]['time_intervals'][time_interval]['accuracy'] = (
            self.indicator_performance[indicator]['time_intervals'][time_interval]['correct'] /
            self.indicator_performance[indicator]['time_intervals'][time_interval]['count']
        )
        
        # Guardar datos actualizados
        self._save_performance_data()
        
        # Recalibrar pesos basados en rendimiento actualizado
        self._recalibrate_weights()
        
    def _recalibrate_weights(self) -> None:
        """Recalibra los pesos de los indicadores basándose en su rendimiento."""
        # Solo recalibrar si hay suficientes datos
        min_signals = 20
        indicators_with_data = [
            ind for ind in self.indicator_performance 
            if self.indicator_performance[ind].get('signals_count', 0) >= min_signals
        ]
        
        if not indicators_with_data:
            logger.info("No hay suficientes datos para recalibrar. Usando pesos por defecto.")
            return
            
        # Calcular pesos basados en precisión global
        total_accuracy = sum(
            self.indicator_performance[ind].get('overall_accuracy', 0) 
            for ind in indicators_with_data
        )
        
        if total_accuracy > 0:
            # Distribuir pesos proporcionalmente a la precisión
            recalibrated_weights = {}
            for ind in indicators_with_data:
                recalibrated_weights[ind] = (
                    self.indicator_performance[ind].get('overall_accuracy', 0) / total_accuracy
                )
                
            # Normalizar pesos para que sumen 1.0
            weight_sum = sum(recalibrated_weights.values())
            for ind in recalibrated_weights:
                recalibrated_weights[ind] /= weight_sum
                
            # Actualizar pesos optimizados
            self.optimized_weights = {
                **self.default_weights,
                **recalibrated_weights
            }
            
            logger.info(f"Pesos recalibrados: {self.optimized_weights}")
        else:
            logger.warning("La precisión total es cero. Usando pesos por defecto.")
    
    def get_indicator_weight(self, 
                           indicator: str, 
                           market_condition: Union[MarketCondition, str],
                           time_interval: Union[TimeInterval, str]) -> float:
        """
        Obtiene el peso adaptado de un indicador para condiciones específicas.
        
        Args:
            indicator: Nombre del indicador
            market_condition: Condición actual del mercado
            time_interval: Intervalo de tiempo actual
            
        Returns:
            float: Peso adaptado del indicador
        """
        # Convertir enums a strings si es necesario
        if isinstance(market_condition, MarketCondition):
            market_condition = market_condition.value
        if isinstance(time_interval, TimeInterval):
            time_interval = time_interval.value
        
        # Verificar si tenemos información de rendimiento para este indicador
        if indicator not in self.indicator_performance:
            return self.default_weights.get(indicator, 0.1)
            
        # Peso base del indicador según optimización
        base_weight = self.optimized_weights.get(indicator, self.default_weights.get(indicator, 0.1))
        
        # Factor de ajuste por condición de mercado
        mc_factor = 1.0
        if (market_condition in self.indicator_performance[indicator]['market_conditions'] and
            self.indicator_performance[indicator]['market_conditions'][market_condition]['count'] >= 10):
            mc_accuracy = self.indicator_performance[indicator]['market_conditions'][market_condition]['accuracy']
            overall_accuracy = self.indicator_performance[indicator]['overall_accuracy']
            
            if overall_accuracy > 0:
                # Ajustar según si es mejor o peor que el promedio general
                mc_factor = mc_accuracy / overall_accuracy
        
        # Factor de ajuste por intervalo de tiempo
        ti_factor = 1.0
        if (time_interval in self.indicator_performance[indicator]['time_intervals'] and
            self.indicator_performance[indicator]['time_intervals'][time_interval]['count'] >= 10):
            ti_accuracy = self.indicator_performance[indicator]['time_intervals'][time_interval]['accuracy']
            overall_accuracy = self.indicator_performance[indicator]['overall_accuracy']
            
            if overall_accuracy > 0:
                # Ajustar según si es mejor o peor que el promedio general
                ti_factor = ti_accuracy / overall_accuracy
        
        # Calcular peso ajustado
        adjusted_weight = base_weight * mc_factor * ti_factor
        
        return adjusted_weight
        
    def get_all_weights(self, 
                      market_condition: Union[MarketCondition, str],
                      time_interval: Union[TimeInterval, str]) -> Dict[str, float]:
        """
        Obtiene todos los pesos ajustados para las condiciones actuales.
        
        Args:
            market_condition: Condición actual del mercado
            time_interval: Intervalo de tiempo actual
            
        Returns:
            Dict[str, float]: Diccionario de pesos ajustados por indicador
        """
        adjusted_weights = {}
        
        for indicator in self.optimized_weights:
            adjusted_weights[indicator] = self.get_indicator_weight(
                indicator, market_condition, time_interval
            )
            
        # Normalizar para que sumen 1.0
        weight_sum = sum(adjusted_weights.values())
        if weight_sum > 0:
            for ind in adjusted_weights:
                adjusted_weights[ind] /= weight_sum
                
        return adjusted_weights
    
    def train_models(self) -> Dict[str, Any]:
        """
        Entrena modelos de ML para predicción de movimientos de precio.
        
        Returns:
            Dict[str, Any]: Métricas de rendimiento de los modelos
        """
        if not self.training_data_file or not os.path.exists(self.training_data_file):
            logger.error(f"Archivo de datos de entrenamiento no encontrado: {self.training_data_file}")
            return {'error': 'Datos de entrenamiento no disponibles'}
        
        try:
            # Cargar datos históricos
            df = pd.read_csv(self.training_data_file)
            logger.info(f"Datos cargados: {len(df)} registros")
            
            # Verificar que tenga las columnas necesarias
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Datos incompletos. Columnas requeridas: {required_columns}")
                return {'error': 'Formato de datos inválido'}
            
            # Preparar datos y calcular indicadores técnicos
            self._prepare_training_data(df)
            
            # Definir horizontes de predicción
            prediction_horizons = {
                'short_term': 6,    # 6 períodos adelante
                'medium_term': 24,  # 24 períodos adelante
                'long_term': 72     # 72 períodos adelante
            }
            
            results = {}
            
            # Entrenar modelo para cada horizonte de predicción
            for horizon_name, periods in prediction_horizons.items():
                model_result = self._train_price_direction_model(df, periods, horizon_name)
                results[horizon_name] = model_result
            
            logger.info(f"Modelos entrenados con éxito: {list(results.keys())}")
            return results
            
        except Exception as e:
            logger.error(f"Error al entrenar modelos: {e}")
            return {'error': str(e)}
    
    def _prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara datos para entrenamiento, calcula indicadores técnicos.
        
        Args:
            df: DataFrame con datos históricos
            
        Returns:
            pd.DataFrame: DataFrame con features agregadas
        """
        # Asegurar que el índice sea timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Ordenar por índice (tiempo)
        df.sort_index(inplace=True)
        
        # Calcular rendimientos y variables objetivo
        df['returns'] = df['close'].pct_change()
        df['next_return'] = df['returns'].shift(-1)
        df['direction'] = (df['next_return'] > 0).astype(int)
        
        # Calcular retornos para diferentes horizontes
        for period in [6, 24, 72]:
            df[f'return_{period}'] = df['close'].pct_change(period).shift(-period)
            df[f'direction_{period}'] = (df[f'return_{period}'] > 0).astype(int)
        
        # --- Calcular indicadores técnicos ---
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bandas de Bollinger
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = df['sma20'] + (df['std20'] * 2)
        df['bollinger_lower'] = df['sma20'] - (df['std20'] * 2)
        df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['sma20']
        df['bollinger_pct'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
        
        # Medias móviles adicionales
        for period in [10, 50, 100, 200]:
            df[f'sma{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Indicadores de volatilidad
        df['atr'] = self._calculate_atr(df)
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # Estocástico
        n = 14
        df['lowest_low'] = df['low'].rolling(window=n).min()
        df['highest_high'] = df['high'].rolling(window=n).max()
        df['stoch_k'] = 100 * (df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low'])
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Características de volumen
        df['volume_sma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma20']
        
        # Eliminar filas con NaN (debido a cálculos con ventanas)
        df.dropna(inplace=True)
        
        return df
        
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calcula el ATR (Average True Range).
        
        Args:
            df: DataFrame con datos OHLC
            period: Período para el cálculo
            
        Returns:
            pd.Series: Serie con valores de ATR
        """
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def _train_price_direction_model(self, 
                                   df: pd.DataFrame, 
                                   horizon: int,
                                   model_name: str) -> Dict[str, Any]:
        """
        Entrena modelo para predecir dirección de precio.
        
        Args:
            df: DataFrame con datos preparados
            horizon: Horizonte de predicción en períodos
            model_name: Nombre para identificar el modelo
            
        Returns:
            Dict[str, Any]: Resultados del entrenamiento
        """
        target = f'direction_{horizon}'
        if target not in df.columns:
            target = 'direction'
        
        # Seleccionar features
        features = [
            'rsi', 'macd', 'macd_hist', 'bollinger_pct', 'bollinger_width',
            'stoch_k', 'stoch_d', 'atr_pct', 'volume_ratio'
        ]
        
        # Añadir medias móviles
        for period in [10, 20, 50, 100, 200]:
            if f'sma{period}' in df.columns:
                features.append(f'sma{period}')
            if f'ema{period}' in df.columns:
                features.append(f'ema{period}')
        
        # Relación entre precio y medias
        df['price_vs_sma20'] = df['close'] / df['sma20']
        df['price_vs_sma50'] = df['close'] / df['sma50']
        df['price_vs_sma200'] = df['close'] / df['sma200']
        
        features.extend(['price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200'])
        
        # Eliminar características que no estén en el DataFrame
        features = [f for f in features if f in df.columns]
        
        # Separar datos
        X = df[features]
        y = df[target]
        
        # Normalizar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Entrenar modelo
        model = GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=4,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Validación cruzada para estimar generalización
        cv_score = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        
        # Guardar modelo y metadatos
        self.models[model_name] = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_accuracy_mean': cv_score.mean(),
                'cv_accuracy_std': cv_score.std()
            },
            'horizon': horizon,
            'target': target
        }
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_accuracy_mean': cv_score.mean(),
            'cv_accuracy_std': cv_score.std(),
            'feature_importance': dict(zip(features, model.feature_importances_)),
            'horizon': horizon
        }
    
    def predict_price_direction(self, 
                              data: pd.DataFrame, 
                              horizon: str = 'short_term') -> Dict[str, Any]:
        """
        Predice dirección futura del precio.
        
        Args:
            data: DataFrame con datos recientes
            horizon: Horizonte de tiempo ('short_term', 'medium_term', 'long_term')
            
        Returns:
            Dict[str, Any]: Predicción y probabilidades
        """
        if horizon not in self.models:
            return {
                'error': f'Modelo para horizonte {horizon} no encontrado',
                'available_models': list(self.models.keys())
            }
            
        model_data = self.models[horizon]
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        
        # Preparar datos con los mismos cálculos que en entrenamiento
        prepared_data = self._prepare_prediction_data(data, features)
        
        # Verificar que tenemos todas las características necesarias
        missing_features = [f for f in features if f not in prepared_data.columns]
        if missing_features:
            return {
                'error': f'Características faltantes: {missing_features}',
                'available_features': list(prepared_data.columns)
            }
            
        # Extraer y escalar características
        X = prepared_data[features].iloc[-1:].values
        X_scaled = scaler.transform(X)
        
        # Hacer predicción
        prediction = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Determinar dirección y confianza
        direction = "up" if prediction == 1 else "down"
        confidence = probabilities[prediction]
        
        # Calcular fuerza de la señal (0-100)
        if direction == "up":
            signal_strength = confidence * 100
        else:
            signal_strength = (1 - confidence) * 100
            
        # Identificar características más influyentes para esta predicción
        feature_importance = list(zip(features, model.feature_importances_))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = feature_importance[:5]
        
        return {
            'direction': direction,
            'confidence': confidence,
            'signal_strength': signal_strength,
            'probabilities': {
                'up': float(probabilities[1]),
                'down': float(probabilities[0])
            },
            'horizon': model_data['horizon'],
            'top_features': top_features,
            'timestamp': datetime.now().isoformat()
        }
    
    def _prepare_prediction_data(self, df: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
        """
        Prepara datos para predicción asegurando que tengan las mismas características.
        
        Args:
            df: DataFrame con datos recientes
            required_features: Lista de características necesarias
            
        Returns:
            pd.DataFrame: DataFrame preparado para predicción
        """
        # Crear copia para no modificar original
        data = df.copy()
        
        # Asegurar formato correcto del índice
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
        
        # Ordenar por índice (tiempo)
        data.sort_index(inplace=True)
        
        # -- Calcular todos los indicadores técnicos --
        
        # RSI
        if 'rsi' in required_features and 'rsi' not in data.columns:
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        if any(f in required_features for f in ['macd', 'macd_hist']):
            if 'macd' not in data.columns:
                ema12 = data['close'].ewm(span=12, adjust=False).mean()
                ema26 = data['close'].ewm(span=26, adjust=False).mean()
                data['macd'] = ema12 - ema26
                data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
                data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # Bandas de Bollinger
        if any(f in required_features for f in ['bollinger_pct', 'bollinger_width']):
            if 'bollinger_pct' not in data.columns:
                data['sma20'] = data['close'].rolling(window=20).mean()
                data['std20'] = data['close'].rolling(window=20).std()
                data['bollinger_upper'] = data['sma20'] + (data['std20'] * 2)
                data['bollinger_lower'] = data['sma20'] - (data['std20'] * 2)
                data['bollinger_width'] = (data['bollinger_upper'] - data['bollinger_lower']) / data['sma20']
                data['bollinger_pct'] = (data['close'] - data['bollinger_lower']) / (data['bollinger_upper'] - data['bollinger_lower'])
        
        # Medias móviles
        for period in [10, 20, 50, 100, 200]:
            if f'sma{period}' in required_features and f'sma{period}' not in data.columns:
                data[f'sma{period}'] = data['close'].rolling(window=period).mean()
            if f'ema{period}' in required_features and f'ema{period}' not in data.columns:
                data[f'ema{period}'] = data['close'].ewm(span=period, adjust=False).mean()
        
        # ATR
        if 'atr_pct' in required_features and 'atr_pct' not in data.columns:
            data['atr'] = self._calculate_atr(data)
            data['atr_pct'] = data['atr'] / data['close'] * 100
        
        # Estocástico
        if any(f in required_features for f in ['stoch_k', 'stoch_d']):
            if 'stoch_k' not in data.columns:
                n = 14
                data['lowest_low'] = data['low'].rolling(window=n).min()
                data['highest_high'] = data['high'].rolling(window=n).max()
                data['stoch_k'] = 100 * (data['close'] - data['lowest_low']) / (data['highest_high'] - data['lowest_low'])
                data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
        
        # Características de volumen
        if 'volume_ratio' in required_features and 'volume_ratio' not in data.columns:
            data['volume_sma20'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma20']
        
        # Relaciones de precio y medias móviles
        if 'price_vs_sma20' in required_features and 'price_vs_sma20' not in data.columns:
            if 'sma20' not in data.columns:
                data['sma20'] = data['close'].rolling(window=20).mean()
            data['price_vs_sma20'] = data['close'] / data['sma20']
            
        if 'price_vs_sma50' in required_features and 'price_vs_sma50' not in data.columns:
            if 'sma50' not in data.columns:
                data['sma50'] = data['close'].rolling(window=50).mean()
            data['price_vs_sma50'] = data['close'] / data['sma50']
            
        if 'price_vs_sma200' in required_features and 'price_vs_sma200' not in data.columns:
            if 'sma200' not in data.columns:
                data['sma200'] = data['close'].rolling(window=200).mean()
            data['price_vs_sma200'] = data['close'] / data['sma200']
        
        # Eliminar filas con NaN
        data.dropna(inplace=True)
        
        return data
    
    def get_market_condition(self, df: pd.DataFrame) -> MarketCondition:
        """
        Detecta la condición actual del mercado.
        
        Args:
            df: DataFrame con datos recientes
            
        Returns:
            MarketCondition: Condición de mercado detectada
        """
        if len(df) < 50:
            return MarketCondition.LATERAL_LOW_VOL
            
        # Calcular tendencia con media móvil
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        
        # Calcular volatilidad
        returns = df['close'].pct_change()
        current_vol = returns.iloc[-20:].std() * np.sqrt(252)  # Anualizada
        historical_vol = returns.iloc[:-20].std() * np.sqrt(252)
        
        # Calcular pendientes
        current_sma20 = df['sma20'].iloc[-1]
        prev_sma20 = df['sma20'].iloc[-10]
        slope_sma20 = (current_sma20 / prev_sma20 - 1) * 100
        
        current_sma50 = df['sma50'].iloc[-1]
        prev_sma50 = df['sma50'].iloc[-25]
        slope_sma50 = (current_sma50 / prev_sma50 - 1) * 100
        
        # Determinar tipo de tendencia
        if slope_sma20 > 1.5 and slope_sma50 > 0.8:
            if current_vol > historical_vol * 1.5:
                return MarketCondition.STRONG_UPTREND
            else:
                return MarketCondition.MODERATE_UPTREND
                
        elif slope_sma20 < -1.5 and slope_sma50 < -0.8:
            if current_vol > historical_vol * 1.5:
                return MarketCondition.STRONG_DOWNTREND
            else:
                return MarketCondition.MODERATE_DOWNTREND
                
        elif abs(slope_sma20) < 0.5 and abs(slope_sma50) < 0.3:
            if current_vol > historical_vol * 1.2:
                return MarketCondition.LATERAL_HIGH_VOL
            else:
                return MarketCondition.LATERAL_LOW_VOL
                
        elif current_vol > historical_vol * 2:
            return MarketCondition.EXTREME_VOLATILITY
            
        # Por defecto
        return MarketCondition.LATERAL_LOW_VOL
    
    def get_prediction_summary(self, 
                             data: pd.DataFrame, 
                             include_all_horizons: bool = True) -> Dict[str, Any]:
        """
        Obtiene un resumen de predicciones para varios horizontes temporales.
        
        Args:
            data: DataFrame con datos recientes
            include_all_horizons: Si se incluyen todos los horizontes disponibles
            
        Returns:
            Dict[str, Any]: Resumen de predicciones
        """
        # Detectar condición de mercado
        market_condition = self.get_market_condition(data)
        
        # Obtener predicciones para diferentes horizontes
        horizons = list(self.models.keys()) if include_all_horizons else ['short_term']
        
        predictions = {}
        for horizon in horizons:
            if horizon in self.models:
                predictions[horizon] = self.predict_price_direction(data, horizon)
        
        # Calcular señal combinada ponderada
        weights = {
            'short_term': 0.5,
            'medium_term': 0.3,
            'long_term': 0.2
        }
        
        # Ajustar pesos según disponibilidad
        available_horizons = [h for h in horizons if h in predictions and 'error' not in predictions[h]]
        if not available_horizons:
            return {
                'error': 'No hay predicciones disponibles',
                'market_condition': market_condition.value
            }
            
        # Recalcular pesos para que sumen 1
        available_weights = {h: weights.get(h, 0.33) for h in available_horizons}
        total_weight = sum(available_weights.values())
        normalized_weights = {h: w/total_weight for h, w in available_weights.items()}
        
        # Calcular señal combinada
        combined_signal = 0
        for horizon, weight in normalized_weights.items():
            pred = predictions[horizon]
            signal_value = pred['signal_strength']
            if pred['direction'] == 'down':
                signal_value = 100 - signal_value
            combined_signal += signal_value * weight
        
        # Interpretar señal (0-40: bajista, 40-60: neutral, 60-100: alcista)
        if combined_signal < 40:
            signal_interpretation = "bearish"
        elif combined_signal > 60:
            signal_interpretation = "bullish"
        else:
            signal_interpretation = "neutral"
            
        # Recomendación de trading
        if signal_interpretation == "bullish" and combined_signal > 75:
            recommendation = "strong_buy"
        elif signal_interpretation == "bullish":
            recommendation = "buy"
        elif signal_interpretation == "bearish" and combined_signal < 25:
            recommendation = "strong_sell"
        elif signal_interpretation == "bearish":
            recommendation = "sell"
        else:
            recommendation = "hold"
            
        return {
            'combined_signal': combined_signal,
            'interpretation': signal_interpretation,
            'recommendation': recommendation,
            'market_condition': market_condition.value,
            'confidence': max(min(abs(combined_signal - 50) / 50, 1.0), 0.0),
            'horizon_predictions': {h: predictions[h] for h in available_horizons},
            'weights_used': normalized_weights,
            'timestamp': datetime.now().isoformat()
        }

def analyze_price_prediction(symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
    """
    Función de conveniencia para analizar y predecir precios.
    
    Args:
        symbol: Par de trading (ej: SOL-USDT)
        timeframe: Intervalo temporal para análisis
        
    Returns:
        Dict[str, Any]: Análisis y predicción completa
    """
    try:
        # Importar aquí para evitar dependencias circulares
        from data_management.market_data import get_market_data
        
        # Obtener datos de mercado
        df = get_market_data(symbol, timeframe)
        if df is None or len(df) < 200:
            return {'error': f'Datos insuficientes para {symbol} en timeframe {timeframe}'}
            
        # Crear sistema de ponderación
        weighting = IndicatorWeighting(use_ml=True)
        
        # Detectar condición de mercado
        market_condition = weighting.get_market_condition(df)
        
        # Preparar datos para predicción
        model_features = [
            'rsi', 'macd', 'macd_hist', 'bollinger_pct', 'bollinger_width',
            'stoch_k', 'stoch_d', 'sma20', 'sma50', 'sma200', 'price_vs_sma20', 'atr_pct'
        ]
        
        prepared_data = weighting._prepare_prediction_data(df, model_features)
        
        # Obtener predicción
        prediction = weighting.get_prediction_summary(prepared_data)
        
        # Adaptar pesos según condición de mercado
        weights = weighting.get_all_weights(market_condition, TimeInterval.HOUR_1)
        
        # Obtener precio actual
        current_price = df['close'].iloc[-1]
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': current_price,
            'prediction': prediction,
            'market_condition': market_condition.value,
            'indicator_weights': weights,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return {'error': str(e)}

def demo_ml_prediction():
    """Demostración de predicción de precios usando ML."""
    print("\n🤖 PREDICCIÓN DE PRECIOS CON MACHINE LEARNING 🤖")
    
    # Símbolo a analizar
    symbol = "SOL-USDT"
    
    # Simular resultado de predicción
    prediction_result = {
        'symbol': symbol,
        'timeframe': '1h',
        'current_price': 150.36,
        'prediction': {
            'combined_signal': 62.5,
            'interpretation': 'bullish',
            'recommendation': 'buy',
            'market_condition': 'lateral_low_vol',
            'confidence': 0.25,
            'horizon_predictions': {
                'short_term': {
                    'direction': 'up',
                    'confidence': 0.58,
                    'signal_strength': 58.0,
                    'probabilities': {'up': 0.58, 'down': 0.42},
                    'horizon': 6,
                    'top_features': [
                        ('rsi', 0.245),
                        ('macd_hist', 0.178),
                        ('bollinger_pct', 0.156),
                        ('price_vs_sma20', 0.128),
                        ('stoch_k', 0.098)
                    ]
                },
                'medium_term': {
                    'direction': 'up',
                    'confidence': 0.64,
                    'signal_strength': 64.0,
                    'probabilities': {'up': 0.64, 'down': 0.36},
                    'horizon': 24,
                    'top_features': [
                        ('price_vs_sma50', 0.223),
                        ('sma20', 0.185),
                        ('bollinger_width', 0.147),
                        ('rsi', 0.132),
                        ('atr_pct', 0.096)
                    ]
                },
                'long_term': {
                    'direction': 'up',
                    'confidence': 0.67,
                    'signal_strength': 67.0,
                    'probabilities': {'up': 0.67, 'down': 0.33},
                    'horizon': 72,
                    'top_features': [
                        ('price_vs_sma200', 0.256),
                        ('sma50', 0.182),
                        ('volume_ratio', 0.154),
                        ('macd', 0.128),
                        ('stoch_d', 0.093)
                    ]
                }
            },
            'weights_used': {'short_term': 0.5, 'medium_term': 0.3, 'long_term': 0.2}
        },
        'market_condition': 'lateral_low_vol',
        'indicator_weights': {
            'rsi': 0.22,
            'macd': 0.19,
            'bollinger': 0.15,
            'sma': 0.12,
            'ema': 0.13,
            'stochastic': 0.10,
            'adx': 0.04,
            'cci': 0.05
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Imprimir resultados
    print(f"Análisis predictivo para {symbol} - {prediction_result['timeframe']}")
    print(f"Precio actual: ${prediction_result['current_price']:.2f}")
    print(f"Condición de mercado: {prediction_result['market_condition']}")
    
    # Señal combinada
    pred = prediction_result['prediction']
    print(f"\nSeñal combinada: {pred['combined_signal']:.1f}/100 ({pred['interpretation']})")
    print(f"Interpretación: {pred['interpretation']}")
    print(f"Recomendación: {pred['recommendation']}")
    print(f"Confianza: {pred['confidence']:.2f}")
    
    # Predicciones por horizonte
    print("\nPredicciones por horizonte temporal:")
    print(f"{'Horizonte':<12} {'Dirección':<10} {'Confianza':<10} {'Señal':<10}")
    print("-" * 50)
    
    for horizon, horizon_pred in pred['horizon_predictions'].items():
        print(f"{horizon:<12} {horizon_pred['direction']:<10} {horizon_pred['confidence']:.2f} {horizon_pred['signal_strength']:.1f}/100")
    
    # Características más importantes para predicción a corto plazo
    short_term = pred['horizon_predictions']['short_term']
    print("\nFactores más influyentes (corto plazo):")
    for feature, importance in short_term['top_features']:
        print(f"{feature}: {importance:.3f}")
    
    # Pesos de indicadores adaptados a condición actual
    print("\nPesos adaptativos para indicadores técnicos:")
    weights = prediction_result['indicator_weights']
    for indicator, weight in weights.items():
        print(f"{indicator}: {weight:.2f}")
    
    return prediction_result

if __name__ == "__main__":
    try:
        demo_ml_prediction()
    except Exception as e:
        print(f"Error en demostración: {e}")