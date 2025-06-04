"""
Módulo para estrategias basadas en Machine Learning
Implementa modelos de ML para la toma de decisiones de trading
"""

import os
import time
import numpy as np
import pandas as pd
import logging
import pickle
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Para modelos más avanzados (opcional, requiere instalación adicional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, save_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

# Para optimización
try:
    from deap import algorithms, base, creator, tools
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False

logger = logging.getLogger("ML_Strategies")

class FeatureEngineering:
    """Clase para preparar y procesar características para modelos de ML"""
    
    def __init__(self, use_ta_lib: bool = False):
        """
        Inicializa el procesador de características
        
        Args:
            use_ta_lib: Si es True, utiliza la biblioteca TA-Lib para 
                       indicadores técnicos más avanzados
        """
        self.use_ta_lib = use_ta_lib
        self.feature_config = self._get_default_feature_config()
        self.scalers = {}
    
    def _get_default_feature_config(self) -> Dict:
        """
        Obtiene la configuración predeterminada de características
        
        Returns:
            Dict: Configuración de características
        """
        return {
            # Indicadores básicos
            "rsi": {"enabled": True, "periods": [14]},
            "macd": {"enabled": True, "fast": 12, "slow": 26, "signal": 9},
            "bollinger": {"enabled": True, "period": 20, "std_dev": 2},
            "sma": {"enabled": True, "periods": [5, 10, 20, 50, 100]},
            "ema": {"enabled": True, "periods": [5, 10, 20, 50]},
            
            # Características de velas
            "candle_patterns": {"enabled": True},
            "candle_stats": {"enabled": True},
            
            # Características de volumen
            "volume_indicators": {"enabled": True},
            
            # Volatilidad
            "volatility": {"enabled": True, "periods": [14, 30]},
            
            # Características de momento
            "momentum": {"enabled": True, "periods": [14, 30]},
            
            # Características temporales
            "time_features": {"enabled": True},
            
            # Características rezagadas (pasadas)
            "lagged_features": {"enabled": True, "lags": [1, 2, 3, 5, 10]},
            
            # Señales personalizadas (pueden implementarse según tus estrategias)
            "custom_signals": {"enabled": False}
        }
    
    def update_feature_config(self, config: Dict):
        """
        Actualiza la configuración de características
        
        Args:
            config: Nueva configuración
        """
        self.feature_config.update(config)
        logger.info("Configuración de características actualizada")
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocesa los datos para entrenamiento o predicción
        
        Args:
            df: DataFrame con datos OHLCV
            target_column: Columna objetivo (si existe)
            
        Returns:
            Tuple: (X, y) donde X son las características y y es la variable objetivo
        """
        # Asegurarse de que df es una copia para no modificar el original
        df = df.copy()
        
        # Eliminar filas con NaN
        df = df.dropna()
        
        # Preparar target si existe
        target = None
        if target_column and target_column in df.columns:
            target = df[target_column]
            df = df.drop(columns=[target_column])
        
        # Obtener características
        features_df = self.create_features(df)
        
        # Eliminar NaN que puedan haber surgido al crear las características
        features_df = features_df.dropna()
        
        # Ajustar targets si se eliminaron filas
        if target is not None:
            target = target.loc[features_df.index]
        
        return features_df, target
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características para el modelo
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            pd.DataFrame: DataFrame con características
        """
        # Crear DataFrame para almacenar las características
        features = pd.DataFrame(index=df.index)
        
        # 1. RSI
        if self.feature_config["rsi"]["enabled"]:
            for period in self.feature_config["rsi"]["periods"]:
                features[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
        
        # 2. MACD
        if self.feature_config["macd"]["enabled"]:
            fast = self.feature_config["macd"]["fast"]
            slow = self.feature_config["macd"]["slow"]
            signal = self.feature_config["macd"]["signal"]
            
            features['macd'], features['macd_signal'], features['macd_hist'] = \
                self._calculate_macd(df['close'], fast, slow, signal)
        
        # 3. Bandas de Bollinger
        if self.feature_config["bollinger"]["enabled"]:
            period = self.feature_config["bollinger"]["period"]
            std_dev = self.feature_config["bollinger"]["std_dev"]
            
            features['bb_upper'], features['bb_middle'], features['bb_lower'] = \
                self._calculate_bollinger_bands(df['close'], period, std_dev)
            
            # Posición relativa en las bandas (0-1)
            features['bb_position'] = (df['close'] - features['bb_lower']) / \
                                      (features['bb_upper'] - features['bb_lower'])
        
        # 4. Medias Móviles Simple (SMA)
        if self.feature_config["sma"]["enabled"]:
            for period in self.feature_config["sma"]["periods"]:
                features[f'sma_{period}'] = df['close'].rolling(period).mean()
                
                # Distancia relativa del precio al SMA
                features[f'sma_{period}_dist'] = (df['close'] - features[f'sma_{period}']) / \
                                                features[f'sma_{period}']
        
        # 5. Medias Móviles Exponencial (EMA)
        if self.feature_config["ema"]["enabled"]:
            for period in self.feature_config["ema"]["periods"]:
                features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                
                # Distancia relativa del precio al EMA
                features[f'ema_{period}_dist'] = (df['close'] - features[f'ema_{period}']) / \
                                                features[f'ema_{period}']
        
        # 6. Estadísticas de velas
        if self.feature_config["candle_stats"]["enabled"]:
            # Tamaño de vela (body)
            features['candle_body'] = abs(df['close'] - df['open']) / df['open']
            
            # Sombras (wicks)
            features['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
            features['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
            
            # Patrones de velas básicos
            if self.feature_config["candle_patterns"]["enabled"]:
                # Doji (cuerpo pequeño)
                features['doji'] = (features['candle_body'] < 0.001).astype(int)
                
                # Martillo (lower_wick grande, upper_wick pequeño, cuerpo pequeño)
                features['hammer'] = ((features['lower_wick'] > 2 * features['candle_body']) & 
                                      (features['upper_wick'] < 0.5 * features['candle_body'])).astype(int)
                
                # Vela alcista/bajista
                features['bullish'] = (df['close'] > df['open']).astype(int)
                features['bearish'] = (df['close'] < df['open']).astype(int)
        
        # 7. Indicadores de volumen
        if self.feature_config["volume_indicators"]["enabled"]:
            # Volumen normalizado
            features['volume_norm'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # OBV (On Balance Volume)
            obv = [0]
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.append(obv[-1] + df['volume'].iloc[i])
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.append(obv[-1] - df['volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            features['obv'] = obv
            
            # Volumen creciente/decreciente
            features['volume_increasing'] = (df['volume'] > df['volume'].shift(1)).astype(int)
        
        # 8. Características de volatilidad
        if self.feature_config["volatility"]["enabled"]:
            for period in self.feature_config["volatility"]["periods"]:
                # ATR (Average True Range)
                features[f'atr_{period}'] = self._calculate_atr(df, period)
                
                # Volatilidad histórica
                features[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
        
        # 9. Características de momento
        if self.feature_config["momentum"]["enabled"]:
            for period in self.feature_config["momentum"]["periods"]:
                # ROC (Rate of Change)
                features[f'roc_{period}'] = (df['close'] / df['close'].shift(period) - 1) * 100
        
        # 10. Características temporales
        if self.feature_config["time_features"]["enabled"]:
            # Si el índice es un DatetimeIndex, extraer características temporales
            if isinstance(df.index, pd.DatetimeIndex):
                features['hour'] = df.index.hour
                features['day_of_week'] = df.index.dayofweek
                features['day_of_month'] = df.index.day
                features['month'] = df.index.month
                features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # 11. Características rezagadas (pasadas)
        if self.feature_config["lagged_features"]["enabled"]:
            for lag in self.feature_config["lagged_features"]["lags"]:
                features[f'close_lag_{lag}'] = df['close'].shift(lag)
                features[f'close_return_lag_{lag}'] = df['close'].pct_change(lag)
                features[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # 12. Señales personalizadas
        if self.feature_config["custom_signals"]["enabled"]:
            # Implementar según estrategias específicas
            pass
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula el RSI para una serie de precios"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast_period: int = 12, 
                       slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula el MACD para una serie de precios"""
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                  std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula las bandas de Bollinger para una serie de precios"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return upper, middle, lower
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula el ATR para un DataFrame OHLCV"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def scale_features(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Escala las características para el modelo
        
        Args:
            features: DataFrame con características
            fit: Si es True, ajusta el scaler a los datos
                Si es False, usa un scaler previamente ajustado
                
        Returns:
            pd.DataFrame: DataFrame con características escaladas
        """
        # Crear copia para no modificar el original
        scaled_features = features.copy()
        
        # Columnas a escalar (excluir características categóricas)
        exclude_cols = ['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
                        'doji', 'hammer', 'bullish', 'bearish', 'volume_increasing']
        
        cols_to_scale = [col for col in scaled_features.columns if col not in exclude_cols]
        
        # Escalar cada característica
        for col in cols_to_scale:
            if col not in self.scalers or fit:
                self.scalers[col] = MinMaxScaler(feature_range=(-1, 1))
                scaled_features[col] = self.scalers[col].fit_transform(
                    scaled_features[col].values.reshape(-1, 1)
                ).flatten()
            else:
                scaled_features[col] = self.scalers[col].transform(
                    scaled_features[col].values.reshape(-1, 1)
                ).flatten()
        
        return scaled_features
    
    def save_scalers(self, path: str = "models/scalers.pkl"):
        """
        Guarda los scalers para uso futuro
        
        Args:
            path: Ruta de guardado
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scalers, f)
    
    def load_scalers(self, path: str = "models/scalers.pkl"):
        """
        Carga scalers previamente guardados
        
        Args:
            path: Ruta de carga
        """
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.scalers = pickle.load(f)

class MLStrategy:
    """Clase base para estrategias basadas en ML"""
    
    def __init__(self, model_type: str = "random_forest", model_params: Dict = None,
                target_type: str = "classification", prediction_threshold: float = 0.5,
                feature_engineering: FeatureEngineering = None):
        """
        Inicializa la estrategia ML
        
        Args:
            model_type: Tipo de modelo ('random_forest', 'gradient_boosting', 'mlp', 'lstm')
            model_params: Parámetros del modelo
            target_type: Tipo de objetivo ('classification' o 'regression')
            prediction_threshold: Umbral para clasificación
            feature_engineering: Instancia de FeatureEngineering o None para usar predeterminado
        """
        self.model_type = model_type
        self.model_params = model_params or self._get_default_model_params()
        self.target_type = target_type
        self.prediction_threshold = prediction_threshold
        self.feature_engineering = feature_engineering or FeatureEngineering()
        
        self.model = None
        self.model_trained = False
        self.last_training_date = None
        self.last_predictions = None
        self.model_performance = {}
        
        # Inicializar modelo
        self._initialize_model()
    
    def _get_default_model_params(self) -> Dict:
        """
        Obtiene parámetros predeterminados según el tipo de modelo
        
        Returns:
            Dict: Parámetros del modelo
        """
        params = {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "subsample": 0.8,
                "random_state": 42
            },
            "mlp": {
                "hidden_layer_sizes": (100, 50),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.0001,
                "max_iter": 200,
                "random_state": 42
            },
            "lstm": {
                "units": 50,
                "dropout": 0.2,
                "recurrent_dropout": 0.2,
                "epochs": 50,
                "batch_size": 32,
                "patience": 10,
                "shuffle": False
            }
        }
        
        return params.get(self.model_type, params["random_forest"])
    
    def _initialize_model(self):
        """Inicializa el modelo según el tipo seleccionado"""
        if self.model_type == "random_forest":
            if self.target_type == "classification":
                self.model = RandomForestClassifier(**self.model_params)
            else:
                self.model = RandomForestRegressor(**self.model_params)
        
        elif self.model_type == "gradient_boosting":
            if self.target_type == "classification":
                self.model = GradientBoostingClassifier(**self.model_params)
            else:
                self.model = GradientBoostingRegressor(**self.model_params)
        
        elif self.model_type == "mlp":
            if self.target_type == "classification":
                self.model = MLPClassifier(**self.model_params)
            else:
                self.model = MLPRegressor(**self.model_params)
        
        elif self.model_type == "lstm" and KERAS_AVAILABLE:
            # Los modelos LSTM se inicializarán durante el entrenamiento
            # ya que requieren conocer la forma de los datos de entrada
            self.model = None
        
        else:
            logger.warning(f"Tipo de modelo '{self.model_type}' no reconocido o no disponible.")
            logger.warning("Usando RandomForestClassifier predeterminado")
            self.model = RandomForestClassifier()
            self.model_type = "random_forest"
    
    def prepare_target(self, df: pd.DataFrame, lookahead: int = 1, 
                      threshold: float = 0.0) -> pd.Series:
        """
        Prepara la variable objetivo para entrenamiento
        
        Args:
            df: DataFrame con datos OHLCV
            lookahead: Número de períodos hacia adelante para predecir
            threshold: Umbral de cambio para clasificación
            
        Returns:
            pd.Series: Variable objetivo
        """
        if self.target_type == "classification":
            # Para clasificación, predecir si el precio subirá o bajará
            future_return = df['close'].pct_change(lookahead).shift(-lookahead)
            
            if threshold == 0.0:
                # Simplemente predecir dirección (arriba/abajo)
                target = (future_return > 0).astype(int)
            else:
                # Predecir movimientos significativos
                target = pd.Series(0, index=df.index)
                target[future_return > threshold] = 1  # Subida significativa
                target[future_return < -threshold] = -1  # Bajada significativa
        
        else:  # regression
            # Para regresión, predecir el retorno futuro
            target = df['close'].pct_change(lookahead).shift(-lookahead)
        
        return target
    
    def train(self, df: pd.DataFrame, lookahead: int = 1, threshold: float = 0.0, 
             test_size: float = 0.2) -> Dict[str, Any]:
        """
        Entrena el modelo con datos históricos
        
        Args:
            df: DataFrame con datos OHLCV
            lookahead: Número de períodos hacia adelante para predecir
            threshold: Umbral de cambio para clasificación
            test_size: Proporción de datos para prueba
            
        Returns:
            Dict: Métricas de rendimiento del modelo
        """
        logger.info(f"Entrenando modelo {self.model_type} para estrategia ML")
        
        # Preparar variable objetivo
        target = self.prepare_target(df, lookahead, threshold)
        
        # Preparar características
        features, processed_target = self.feature_engineering.preprocess_data(df, target_column=None)
        processed_target = target.loc[features.index]  # Alinear con las características
        
        # Eliminar filas con NaN
        valid_indices = ~(features.isna().any(axis=1) | processed_target.isna())
        features = features[valid_indices]
        processed_target = processed_target[valid_indices]
        
        if len(features) < 100:
            logger.warning("Insuficientes datos para entrenar modelo")
            return {"error": "Insuficientes datos para entrenar modelo"}
        
        # Escalar características
        scaled_features = self.feature_engineering.scale_features(features, fit=True)
        
        # División temporal (para series temporales)
        split_idx = int(len(scaled_features) * (1 - test_size))
        X_train = scaled_features.iloc[:split_idx]
        y_train = processed_target.iloc[:split_idx]
        X_test = scaled_features.iloc[split_idx:]
        y_test = processed_target.iloc[split_idx:]
        
        # Entrenar modelo según el tipo
        if self.model_type == "lstm" and KERAS_AVAILABLE:
            # Para LSTM, necesitamos reformatear los datos y construir el modelo
            X_train_lstm = self._prepare_lstm_data(X_train)
            X_test_lstm = self._prepare_lstm_data(X_test)
            
            # Construir modelo LSTM
            self._build_lstm_model(X_train_lstm.shape[1:])
            
            # Entrenar modelo
            epochs = self.model_params.get("epochs", 50)
            batch_size = self.model_params.get("batch_size", 32)
            patience = self.model_params.get("patience", 10)
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience, restore_best_weights=True
            )
            
            self.model.fit(
                X_train_lstm, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_lstm, y_test),
                callbacks=[early_stopping],
                shuffle=self.model_params.get("shuffle", False)
            )
            
            # Evaluar modelo
            y_pred = (self.model.predict(X_test_lstm) > self.prediction_threshold).astype(int)
        else:
            # Entrenar modelo estándar de scikit-learn
            self.model.fit(X_train, y_train)
            
            # Predecir en conjunto de prueba
            if self.target_type == "classification":
                y_pred = self.model.predict(X_test)
            else:
                y_pred = self.model.predict(X_test)
        
        # Calcular métricas de rendimiento
        if self.target_type == "classification":
            performance = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1": f1_score(y_test, y_pred, average='weighted')
            }
        else:
            performance = {
                "mse": ((y_test - y_pred) ** 2).mean(),
                "mae": abs(y_test - y_pred).mean(),
                "r2": self.model.score(X_test, y_test)
            }
        
        # Guardar información de entrenamiento
        self.model_trained = True
        self.last_training_date = datetime.now()
        self.model_performance = performance
        
        # Guardar scalers
        self.feature_engineering.save_scalers()
        
        # Guardar modelo
        self.save_model()
        
        logger.info(f"Modelo entrenado. Rendimiento: {performance}")
        return performance
    
    def _prepare_lstm_data(self, features: pd.DataFrame, sequence_length: int = 10) -> np.ndarray:
        """
        Prepara datos para LSTM
        
        Args:
            features: DataFrame con características
            sequence_length: Longitud de las secuencias
            
        Returns:
            np.ndarray: Datos formateados para LSTM [samples, time_steps, features]
        """
        data = features.values
        X = []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
        
        return np.array(X)
    
    def _build_lstm_model(self, input_shape: Tuple):
        """
        Construye el modelo LSTM
        
        Args:
            input_shape: Forma de los datos de entrada (time_steps, features)
        """
        if not KERAS_AVAILABLE:
            logger.error("TensorFlow/Keras no está disponible. No se puede construir LSTM.")
            return
        
        units = self.model_params.get("units", 50)
        dropout = self.model_params.get("dropout", 0.2)
        recurrent_dropout = self.model_params.get("recurrent_dropout", 0.2)
        
        model = Sequential()
        
        # Capa LSTM
        model.add(LSTM(
            units=units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            input_shape=input_shape,
            return_sequences=False
        ))
        
        # Capa oculta
        model.add(Dense(units=20, activation='relu'))
        
        # Capa de salida
        if self.target_type == "classification":
            model.add(Dense(units=1, activation='sigmoid'))
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.add(Dense(units=1, activation='linear'))
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
        
        self.model = model
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Realiza predicciones con el modelo entrenado
        
        Args:
            df: DataFrame con datos OHLCV recientes
            
        Returns:
            Dict: Predicción y confianza
        """
        if not self.model_trained or self.model is None:
            return {"signal": 0, "confidence": 0.0, "error": "Modelo no entrenado"}
        
        # Preparar características
        features = self.feature_engineering.create_features(df)
        
        # Eliminar filas con NaN
        features = features.dropna()
        
        if len(features) == 0:
            return {"signal": 0, "confidence": 0.0, "error": "No hay suficientes datos para predecir"}
        
        # Escalar características
        self.feature_engineering.load_scalers()  # Cargar scalers guardados
        scaled_features = self.feature_engineering.scale_features(features, fit=False)
        
        # Preparar datos según el tipo de modelo
        if self.model_type == "lstm" and KERAS_AVAILABLE:
            # Preparar secuencia para LSTM
            X = self._prepare_lstm_data(scaled_features)
            
            if len(X) == 0:
                return {"signal": 0, "confidence": 0.0, "error": "Secuencia insuficiente para LSTM"}
            
            # Predecir
            raw_prediction = self.model.predict(X)[-1][0]  # Última predicción
            
            if self.target_type == "classification":
                # Convertir a señal (-1, 0, 1)
                signal = 1 if raw_prediction > self.prediction_threshold else -1
                confidence = abs(raw_prediction - 0.5) * 2  # Escalar a [0,1]
            else:
                signal = 1 if raw_prediction > 0 else -1 if raw_prediction < 0 else 0
                confidence = min(abs(raw_prediction), 0.1) / 0.1  # Limitar a [0,1]
        
        else:
            # Usar última fila para modelos scikit-learn
            X = scaled_features.iloc[-1:].values
            
            if self.target_type == "classification":
                # Para clasificadores, obtener probabilidades
                if hasattr(self.model, "predict_proba"):
                    proba = self.model.predict_proba(X)[0]
                    # Si es clasificación binaria
                    if len(proba) == 2:
                        confidence = proba[1]
                        signal = 1 if confidence > self.prediction_threshold else -1
                    # Si es multiclase
                    else:
                        # Clases: -1, 0, 1 (bajada, neutral, subida)
                        max_class_idx = np.argmax(proba)
                        signal = max_class_idx - 1  # Convertir a [-1, 0, 1]
                        confidence = proba[max_class_idx]
                else:
                    # Si no hay probabilidades, usar predicción directa
                    signal = self.model.predict(X)[0]
                    confidence = 0.6  # Confianza predeterminada
            else:
                # Para regresión, convertir el valor a señal
                raw_prediction = self.model.predict(X)[0]
                signal = 1 if raw_prediction > 0 else -1 if raw_prediction < 0 else 0
                confidence = min(abs(raw_prediction), 0.1) / 0.1  # Limitar a [0,1]
        
        # Almacenar última predicción
        self.last_predictions = {"signal": signal, "confidence": float(confidence)}
        
        return self.last_predictions
    
    def save_model(self, path: str = None):
        """
        Guarda el modelo entrenado
        
        Args:
            path: Ruta de guardado (predeterminada basada en tipo de modelo)
        """
        if not self.model_trained or self.model is None:
            logger.warning("No hay modelo entrenado para guardar")
            return
        
        if path is None:
            path = f"models/{self.model_type}_model"
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            if self.model_type == "lstm" and KERAS_AVAILABLE:
                save_model(self.model, path)
            else:
                joblib.dump(self.model, f"{path}.pkl")
            
            # Guardar metadatos
            metadata = {
                "model_type": self.model_type,
                "target_type": self.target_type,
                "prediction_threshold": self.prediction_threshold,
                "model_params": self.model_params,
                "last_training_date": self.last_training_date.isoformat() if self.last_training_date else None,
                "model_performance": self.model_performance
            }
            
            with open(f"{path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Modelo guardado en {path}")
        
        except Exception as e:
            logger.error(f"Error al guardar modelo: {e}")
    
    def load_model(self, path: str = None):
        """
        Carga un modelo guardado
        
        Args:
            path: Ruta de carga (predeterminada basada en tipo de modelo)
        """
        if path is None:
            path = f"models/{self.model_type}_model"
        
        try:
            # Cargar metadatos
            if os.path.exists(f"{path}_metadata.json"):
                with open(f"{path}_metadata.json", 'r') as f:
                    metadata = json.load(f)
                
                self.model_type = metadata.get("model_type", self.model_type)
                self.target_type = metadata.get("target_type", self.target_type)
                self.prediction_threshold = metadata.get("prediction_threshold", self.prediction_threshold)
                self.model_params = metadata.get("model_params", self.model_params)
                
                if metadata.get("last_training_date"):
                    self.last_training_date = datetime.fromisoformat(metadata["last_training_date"])
                
                self.model_performance = metadata.get("model_performance", {})
            
            # Cargar modelo
            if self.model_type == "lstm" and KERAS_AVAILABLE:
                if os.path.exists(path):
                    self.model = load_model(path)
                    self.model_trained = True
            else:
                if os.path.exists(f"{path}.pkl"):
                    self.model = joblib.load(f"{path}.pkl")
                    self.model_trained = True
            
            # Cargar scalers
            self.feature_engineering.load_scalers()
            
            logger.info(f"Modelo cargado desde {path}")
        
        except Exception as e:
            logger.error(f"Error al cargar modelo: {e}")
    
    def optimize_hyperparameters(self, df: pd.DataFrame, lookahead: int = 1, threshold: float = 0.0):
        """
        Optimiza hiperparámetros del modelo
        
        Args:
            df: DataFrame con datos OHLCV
            lookahead: Número de períodos hacia adelante para predecir
            threshold: Umbral de cambio para clasificación
            
        Returns:
            Dict: Mejores parámetros y rendimiento
        """
        logger.info(f"Optimizando hiperparámetros para modelo {self.model_type}")
        
        # Preparar variable objetivo
        target = self.prepare_target(df, lookahead, threshold)
        
        # Preparar características
        features, processed_target = self.feature_engineering.preprocess_data(df, target_column=None)
        processed_target = target.loc[features.index]  # Alinear con las características
        
        # Eliminar filas con NaN
        valid_indices = ~(features.isna().any(axis=1) | processed_target.isna())
        features = features[valid_indices]
        processed_target = processed_target[valid_indices]
        
        if len(features) < 100:
            logger.warning("Insuficientes datos para optimizar hiperparámetros")
            return {"error": "Insuficientes datos para optimizar hiperparámetros"}
        
        # Escalar características
        scaled_features = self.feature_engineering.scale_features(features, fit=True)
        
        # Definir espacio de búsqueda según tipo de modelo
        param_grid = self._get_param_grid()
        
        if not param_grid:
            logger.warning("No se pudo definir espacio de búsqueda para optimización")
            return {"error": "No se pudo definir espacio de búsqueda para optimización"}
        
        # Definir modelo base
        base_model = self._get_base_model()
        
        if not base_model:
            logger.warning("No se pudo definir modelo base para optimización")
            return {"error": "No se pudo definir modelo base para optimización"}
        
        # Configurar validación cruzada para series temporales
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Ejecutar búsqueda de cuadrícula
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='accuracy' if self.target_type == "classification" else 'neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(scaled_features, processed_target)
        
        # Actualizar modelo con mejores parámetros
        best_params = grid_search.best_params_
        self.model_params.update(best_params)
        self._initialize_model()
        
        # Entrenar modelo con mejores parámetros
        self.train(df, lookahead, threshold)
        
        return {
            "best_params": best_params,
            "best_score": grid_search.best_score_,
            "model_performance": self.model_performance
        }
    
    def _get_param_grid(self) -> Dict:
        """
        Define espacio de búsqueda para optimización de hiperparámetros
        
        Returns:
            Dict: Espacio de búsqueda
        """
        param_grids = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "gradient_boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.6, 0.8, 1.0]
            },
            "mlp": {
                "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate_init": [0.001, 0.01]
            }
        }
        
        return param_grids.get(self.model_type, {})
    
    def _get_base_model(self):
        """
        Define modelo base para optimización de hiperparámetros
        
        Returns:
            Modelo base o None si no es compatible
        """
        if self.model_type == "random_forest":
            if self.target_type == "classification":
                return RandomForestClassifier(random_state=42)
            else:
                return RandomForestRegressor(random_state=42)
        
        elif self.model_type == "gradient_boosting":
            if self.target_type == "classification":
                return GradientBoostingClassifier(random_state=42)
            else:
                return GradientBoostingRegressor(random_state=42)
        
        elif self.model_type == "mlp":
            if self.target_type == "classification":
                return MLPClassifier(random_state=42)
            else:
                return MLPRegressor(random_state=42)
        
        return None

class MLEnsembleStrategy:
    """Clase para estrategias de conjunto de modelos ML"""
    
    def __init__(self, model_configs: List[Dict] = None):
        """
        Inicializa la estrategia de conjunto
        
        Args:
            model_configs: Lista de configuraciones de modelos
        """
        self.models = []
        
        # Crear modelos según configuraciones
        if model_configs:
            for config in model_configs:
                model_type = config.get("model_type", "random_forest")
                model_params = config.get("model_params", None)
                target_type = config.get("target_type", "classification")
                prediction_threshold = config.get("prediction_threshold", 0.5)
                
                model = MLStrategy(
                    model_type=model_type,
                    model_params=model_params,
                    target_type=target_type,
                    prediction_threshold=prediction_threshold
                )
                
                self.models.append(model)
        else:
            # Crear conjunto predeterminado
            self.models = [
                MLStrategy(model_type="random_forest", target_type="classification"),
                MLStrategy(model_type="gradient_boosting", target_type="classification"),
                MLStrategy(model_type="mlp", target_type="classification")
            ]
        
        self.weights = [1.0] * len(self.models)  # Pesos iguales inicialmente
        self.performance_history = []
    
    def train_all(self, df: pd.DataFrame, lookahead: int = 1, threshold: float = 0.0):
        """
        Entrena todos los modelos del conjunto
        
        Args:
            df: DataFrame con datos OHLCV
            lookahead: Número de períodos hacia adelante para predecir
            threshold: Umbral de cambio para clasificación
            
        Returns:
            Dict: Resultados de entrenamiento
        """
        results = []
        
        for i, model in enumerate(self.models):
            logger.info(f"Entrenando modelo {i+1}/{len(self.models)}")
            result = model.train(df, lookahead, threshold)
            results.append(result)
        
        # Actualizar pesos según rendimiento
        self._update_weights()
        
        return {"models": len(self.models), "results": results, "weights": self.weights}
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Realiza predicción ponderada con todos los modelos
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            Dict: Predicción y confianza
        """
        predictions = []
        confidences = []
        
        for i, model in enumerate(self.models):
            result = model.predict(df)
            
            if "error" not in result:
                predictions.append(result["signal"])
                confidences.append(result["confidence"])
            else:
                logger.warning(f"Error en predicción del modelo {i}: {result['error']}")
        
        if not predictions:
            return {"signal": 0, "confidence": 0.0, "error": "No hay predicciones válidas"}
        
        # Calcular predicción ponderada
        weighted_sum = 0
        total_weight = 0
        
        for i, pred in enumerate(predictions):
            weight = self.weights[i] * confidences[i]
            weighted_sum += pred * weight
            total_weight += weight
        
        if total_weight == 0:
            return {"signal": 0, "confidence": 0.0, "error": "Confianza cero en todas las predicciones"}
        
        weighted_prediction = weighted_sum / total_weight
        
        # Determinar señal y confianza
        if weighted_prediction > 0.2:
            signal = 1
        elif weighted_prediction < -0.2:
            signal = -1
        else:
            signal = 0
        
        # Nivel de confianza del conjunto
        ensemble_confidence = abs(weighted_prediction)
        
        return {"signal": signal, "confidence": float(ensemble_confidence)}
    
    def _update_weights(self):
        """Actualiza los pesos de los modelos según su rendimiento"""
        performances = []
        
        for model in self.models:
            # Obtener métricas relevantes del último entrenamiento
            perf = model.model_performance
            
            if not perf:
                performances.append(0.5)  # Valor predeterminado
                continue
            
            if "accuracy" in perf:  # Clasificación
                performances.append(perf["accuracy"])
            elif "r2" in perf:  # Regresión
                performances.append(max(0, perf["r2"]))
            else:
                performances.append(0.5)
        
        # Normalizar pesos
        total_perf = sum(performances)
        if total_perf > 0:
            self.weights = [p / total_perf for p in performances]
        else:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        # Registrar historial
        self.performance_history.append({
            "date": datetime.now().isoformat(),
            "performances": performances,
            "weights": self.weights
        })
    
    def save(self, path: str = "models/ensemble"):
        """
        Guarda el conjunto de modelos
        
        Args:
            path: Ruta de guardado
        """
        os.makedirs(path, exist_ok=True)
        
        # Guardar modelos individuales
        for i, model in enumerate(self.models):
            model.save_model(f"{path}/model_{i}")
        
        # Guardar metadatos del conjunto
        metadata = {
            "weights": self.weights,
            "performance_history": self.performance_history,
            "num_models": len(self.models)
        }
        
        with open(f"{path}/ensemble_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Conjunto guardado en {path}")
    
    def load(self, path: str = "models/ensemble"):
        """
        Carga un conjunto de modelos guardado
        
        Args:
            path: Ruta de carga
        """
        if not os.path.exists(path):
            logger.error(f"La ruta {path} no existe")
            return False
        
        # Cargar metadatos
        if os.path.exists(f"{path}/ensemble_metadata.json"):
            with open(f"{path}/ensemble_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.weights = metadata.get("weights", [])
            self.performance_history = metadata.get("performance_history", [])
            num_models = metadata.get("num_models", 0)
            
            # Cargar modelos individuales
            self.models = []
            for i in range(num_models):
                model_path = f"{path}/model_{i}"
                
                # Obtener tipo de modelo de metadatos
                with open(f"{model_path}_metadata.json", 'r') as f:
                    model_metadata = json.load(f)
                
                model = MLStrategy(
                    model_type=model_metadata.get("model_type", "random_forest"),
                    model_params=model_metadata.get("model_params", None),
                    target_type=model_metadata.get("target_type", "classification"),
                    prediction_threshold=model_metadata.get("prediction_threshold", 0.5)
                )
                
                model.load_model(model_path)
                self.models.append(model)
            
            # Asegurar longitud de pesos
            if len(self.weights) != len(self.models):
                self.weights = [1.0 / len(self.models)] * len(self.models)
            
            logger.info(f"Conjunto cargado desde {path} con {len(self.models)} modelos")
            return True
        
        logger.error(f"No se encontraron metadatos del conjunto en {path}")
        return False