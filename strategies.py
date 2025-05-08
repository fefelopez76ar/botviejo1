"""
Módulo de estrategias avanzadas para el bot de trading
Incluye estrategias clásicas, estadísticas y de aprendizaje automático
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

# Configurar logging
logger = logging.getLogger("TradingStrategies")

class TechnicalIndicators:
    """Clase para cálculo de indicadores técnicos"""
    
    @staticmethod
    def sma(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calcula la Media Móvil Simple (SMA)
        
        Args:
            data: Serie de precios
            period: Período para la media móvil
            
        Returns:
            pd.Series: Media móvil calculada
        """
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calcula la Media Móvil Exponencial (EMA)
        
        Args:
            data: Serie de precios
            period: Período para la media móvil exponencial
            
        Returns:
            pd.Series: Media móvil exponencial calculada
        """
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcula el Índice de Fuerza Relativa (RSI)
        
        Args:
            data: Serie de precios
            period: Período para RSI
            
        Returns:
            pd.Series: RSI calculado
        """
        delta = data.diff()
        
        # Separar ganancias y pérdidas
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calcular promedios
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calcular RS y RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, 
             signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calcula el MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Serie de precios
            fast_period: Período para la EMA rápida
            slow_period: Período para la EMA lenta
            signal_period: Período para la línea de señal
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: MACD, señal, histograma
        """
        # Calcular EMAs
        ema_fast = TechnicalIndicators.ema(data, fast_period)
        ema_slow = TechnicalIndicators.ema(data, slow_period)
        
        # Calcular MACD y línea de señal
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        
        # Calcular histograma
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, 
                       num_std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calcula las Bandas de Bollinger
        
        Args:
            data: Serie de precios
            period: Período para la media móvil
            num_std_dev: Número de desviaciones estándar
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: Media, banda superior, banda inferior
        """
        # Calcular media móvil
        middle_band = TechnicalIndicators.sma(data, period)
        
        # Calcular desviación estándar
        std_dev = data.rolling(window=period).std()
        
        # Calcular bandas
        upper_band = middle_band + (std_dev * num_std_dev)
        lower_band = middle_band - (std_dev * num_std_dev)
        
        return middle_band, upper_band, lower_band
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
           period: int = 14) -> pd.Series:
        """
        Calcula el ATR (Average True Range)
        
        Args:
            high: Serie de precios máximos
            low: Serie de precios mínimos
            close: Serie de precios de cierre
            period: Período para el ATR
            
        Returns:
            pd.Series: ATR calculado
        """
        # Calcular el True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calcular ATR
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, 
           period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calcula el ADX (Average Directional Index)
        
        Args:
            high: Serie de precios máximos
            low: Serie de precios mínimos
            close: Serie de precios de cierre
            period: Período para el ADX
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: ADX, +DI, -DI
        """
        # Calcular DM (Directional Movement)
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        # +DM
        plus_dm = pd.Series(0, index=up_move.index)
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
        
        # -DM
        minus_dm = pd.Series(0, index=down_move.index)
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
        
        # Calcular TR (True Range)
        tr = TechnicalIndicators.atr(high, low, close, 1)
        
        # Calcular +DI y -DI
        plus_di = 100 * TechnicalIndicators.ema(plus_dm, period) / TechnicalIndicators.ema(tr, period)
        minus_di = 100 * TechnicalIndicators.ema(minus_dm, period) / TechnicalIndicators.ema(tr, period)
        
        # Calcular DX y ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = TechnicalIndicators.ema(dx, period)
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                  k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calcula el Oscilador Estocástico
        
        Args:
            high: Serie de precios máximos
            low: Serie de precios mínimos
            close: Serie de precios de cierre
            k_period: Período para %K
            d_period: Período para %D
            
        Returns:
            Tuple[pd.Series, pd.Series]: %K, %D
        """
        # Calcular %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # Calcular %D
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    @staticmethod
    def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series, 
                conversion_period: int = 9, base_period: int = 26, 
                lagging_span2_period: int = 52, displacement: int = 26
               ) -> Dict[str, pd.Series]:
        """
        Calcula el Ichimoku Cloud
        
        Args:
            high: Serie de precios máximos
            low: Serie de precios mínimos
            close: Serie de precios de cierre
            conversion_period: Período para la línea Tenkan-sen
            base_period: Período para la línea Kijun-sen
            lagging_span2_period: Período para Senkou Span B
            displacement: Desplazamiento para proyecciones
            
        Returns:
            Dict[str, pd.Series]: Componentes de Ichimoku
        """
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(window=conversion_period).max() + 
                     low.rolling(window=conversion_period).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(window=base_period).max() + 
                    low.rolling(window=base_period).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(window=lagging_span2_period).max() + 
                         low.rolling(window=lagging_span2_period).min()) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-displacement)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }


class ClassicStrategy:
    """Estrategias clásicas de trading"""
    
    @staticmethod
    def moving_average_crossover(df: pd.DataFrame, 
                                fast_period: int = 20, 
                                slow_period: int = 50
                               ) -> pd.Series:
        """
        Estrategia de cruce de medias móviles
        
        Args:
            df: DataFrame con datos OHLC
            fast_period: Período para la media rápida
            slow_period: Período para la media lenta
            
        Returns:
            pd.Series: Señales generadas (1: compra, -1: venta, 0: neutral)
        """
        # Calcular medias móviles
        df['fast_ma'] = TechnicalIndicators.sma(df['close'], fast_period)
        df['slow_ma'] = TechnicalIndicators.sma(df['close'], slow_period)
        
        # Calcular cruces
        df['signal'] = 0
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1
        
        # Detectar cambios de señal
        df['position'] = df['signal'].diff()
        
        return df['position']
    
    @staticmethod
    def rsi_strategy(df: pd.DataFrame, period: int = 14, 
                    overbought: int = 70, oversold: int = 30
                   ) -> pd.Series:
        """
        Estrategia basada en RSI
        
        Args:
            df: DataFrame con datos OHLC
            period: Período para RSI
            overbought: Nivel de sobrecompra
            oversold: Nivel de sobreventa
            
        Returns:
            pd.Series: Señales generadas (1: compra, -1: venta, 0: neutral)
        """
        # Calcular RSI
        df['rsi'] = TechnicalIndicators.rsi(df['close'], period)
        
        # Generar señales
        df['signal'] = 0
        df.loc[df['rsi'] < oversold, 'signal'] = 1  # Compra en sobreventa
        df.loc[df['rsi'] > overbought, 'signal'] = -1  # Venta en sobrecompra
        
        # Solo considerar cambios de señal
        df['position'] = df['signal'].diff().fillna(0)
        
        return df['position']
    
    @staticmethod
    def macd_strategy(df: pd.DataFrame, 
                     fast_period: int = 12, 
                     slow_period: int = 26, 
                     signal_period: int = 9
                    ) -> pd.Series:
        """
        Estrategia basada en MACD
        
        Args:
            df: DataFrame con datos OHLC
            fast_period: Período para EMA rápida
            slow_period: Período para EMA lenta
            signal_period: Período para la línea de señal
            
        Returns:
            pd.Series: Señales generadas (1: compra, -1: venta, 0: neutral)
        """
        # Calcular MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            df['close'], fast_period, slow_period, signal_period
        )
        
        df['macd'] = macd_line
        df['signal_line'] = signal_line
        df['macd_hist'] = histogram
        
        # Generar señales basadas en el cruce de MACD y línea de señal
        df['signal'] = 0
        df.loc[df['macd'] > df['signal_line'], 'signal'] = 1
        df.loc[df['macd'] < df['signal_line'], 'signal'] = -1
        
        # Detectar cambios de señal
        df['position'] = df['signal'].diff().fillna(0)
        
        return df['position']
    
    @staticmethod
    def bollinger_strategy(df: pd.DataFrame, 
                          period: int = 20, 
                          num_std_dev: float = 2.0
                         ) -> pd.Series:
        """
        Estrategia basada en Bandas de Bollinger
        
        Args:
            df: DataFrame con datos OHLC
            period: Período para la media móvil
            num_std_dev: Número de desviaciones estándar
            
        Returns:
            pd.Series: Señales generadas (1: compra, -1: venta, 0: neutral)
        """
        # Calcular Bandas de Bollinger
        middle, upper, lower = TechnicalIndicators.bollinger_bands(
            df['close'], period, num_std_dev
        )
        
        df['bb_middle'] = middle
        df['bb_upper'] = upper
        df['bb_lower'] = lower
        
        # Generar señales
        df['signal'] = 0
        df.loc[df['close'] < df['bb_lower'], 'signal'] = 1  # Compra cuando toca banda inferior
        df.loc[df['close'] > df['bb_upper'], 'signal'] = -1  # Venta cuando toca banda superior
        
        # Detectar cambios de señal
        df['position'] = df['signal'].diff().fillna(0)
        
        return df['position']
    
    @staticmethod
    def breakout_strategy(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Estrategia de ruptura de rangos
        
        Args:
            df: DataFrame con datos OHLC
            period: Período para calcular máximos y mínimos
            
        Returns:
            pd.Series: Señales generadas (1: compra, -1: venta, 0: neutral)
        """
        # Calcular máximos y mínimos recientes
        df['recent_high'] = df['high'].rolling(window=period).max().shift(1)
        df['recent_low'] = df['low'].rolling(window=period).min().shift(1)
        
        # Generar señales
        df['signal'] = 0
        df.loc[df['close'] > df['recent_high'], 'signal'] = 1  # Ruptura al alza
        df.loc[df['close'] < df['recent_low'], 'signal'] = -1  # Ruptura a la baja
        
        # Detectar cambios de señal
        df['position'] = df['signal'].diff().fillna(0)
        
        return df['position']


class StatisticalStrategy:
    """Estrategias estadísticas avanzadas para trading"""
    
    @staticmethod
    def zscore(data: pd.Series, window: int = 20) -> pd.Series:
        """
        Calcula el Z-score de una serie
        
        Args:
            data: Serie de precios o indicador
            window: Ventana para el cálculo
            
        Returns:
            pd.Series: Z-score calculado
        """
        # Calcular la media móvil y desviación estándar
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        
        # Calcular Z-score
        z_score = (data - rolling_mean) / rolling_std
        
        return z_score
    
    @staticmethod
    def mean_reversion_strategy(df: pd.DataFrame, 
                              window: int = 20, 
                              z_entry: float = 2.0, 
                              z_exit: float = 0.5
                             ) -> pd.Series:
        """
        Estrategia de reversión a la media basada en Z-score
        
        Args:
            df: DataFrame con datos OHLC
            window: Ventana para calcular Z-score
            z_entry: Nivel de Z-score para entrar
            z_exit: Nivel de Z-score para salir
            
        Returns:
            pd.Series: Señales generadas
        """
        # Calcular Z-score del precio
        df['zscore'] = StatisticalStrategy.zscore(df['close'], window)
        
        # Generar señales
        df['signal'] = 0
        
        # Señales de entrada
        df.loc[df['zscore'] < -z_entry, 'signal'] = 1  # Compra cuando precio muy bajo
        df.loc[df['zscore'] > z_entry, 'signal'] = -1  # Venta cuando precio muy alto
        
        # Señales de salida
        df.loc[(df['zscore'] > -z_exit) & (df['zscore'] < z_exit), 'signal'] = 0
        
        # Detectar cambios de señal
        df['position'] = df['signal'].diff().fillna(0)
        
        return df['position']
    
    @staticmethod
    def pairs_trading_strategy(df1: pd.DataFrame, df2: pd.DataFrame, 
                             window: int = 20, z_entry: float = 2.0, 
                             z_exit: float = 0.5) -> Tuple[pd.Series, pd.Series]:
        """
        Estrategia de trading por pares (Pairs Trading)
        
        Args:
            df1: DataFrame del primer activo
            df2: DataFrame del segundo activo
            window: Ventana para calcular la relación
            z_entry: Nivel de Z-score para entrar
            z_exit: Nivel de Z-score para salir
            
        Returns:
            Tuple[pd.Series, pd.Series]: Señales para activo 1 y activo 2
        """
        # Calcular el ratio entre precios
        ratio = df1['close'] / df2['close']
        
        # Calcular Z-score del ratio
        z_score = StatisticalStrategy.zscore(ratio, window)
        
        # Generar señales
        signals1 = pd.Series(0, index=z_score.index)
        signals2 = pd.Series(0, index=z_score.index)
        
        # Señales de entrada
        # Cuando ratio está alto: vender activo 1, comprar activo 2
        signals1[z_score > z_entry] = -1
        signals2[z_score > z_entry] = 1
        
        # Cuando ratio está bajo: comprar activo 1, vender activo 2
        signals1[z_score < -z_entry] = 1
        signals2[z_score < -z_entry] = -1
        
        # Señales de salida cuando el ratio vuelve a la normalidad
        exit_condition = (z_score > -z_exit) & (z_score < z_exit)
        signals1[exit_condition] = 0
        signals2[exit_condition] = 0
        
        # Detectar cambios de señal
        position1 = signals1.diff().fillna(0)
        position2 = signals2.diff().fillna(0)
        
        return position1, position2
    
    @staticmethod
    def test_cointegration(series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
        """
        Prueba de cointegración entre dos series
        
        Args:
            series1: Primera serie de precios
            series2: Segunda serie de precios
            
        Returns:
            Dict: Resultados de la prueba de cointegración
        """
        # Realizar prueba de cointegración
        result = coint(series1, series2)
        
        # Extraer resultados
        t_stat = result[0]
        p_value = result[1]
        critical_values = result[2]
        
        # Determinar si series están cointegradas
        is_cointegrated = p_value < 0.05
        
        # Calcular el ratio de cobertura con regresión
        model = sm.OLS(series1, sm.add_constant(series2)).fit()
        hedge_ratio = model.params[1]
        
        return {
            't_stat': t_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_cointegrated': is_cointegrated,
            'hedge_ratio': hedge_ratio
        }
    
    @staticmethod
    def kalman_filter_strategy(df1: pd.DataFrame, df2: pd.DataFrame, 
                             delta: float = 1e-4, 
                             z_entry: float = 2.0, 
                             z_exit: float = 0.5) -> Tuple[pd.Series, pd.Series]:
        """
        Estrategia de pairs trading con filtro de Kalman
        
        Args:
            df1: DataFrame del primer activo
            df2: DataFrame del segundo activo
            delta: Parámetro de ruido del proceso
            z_entry: Nivel de Z-score para entrar
            z_exit: Nivel de Z-score para salir
            
        Returns:
            Tuple[pd.Series, pd.Series]: Señales para activo 1 y activo 2
        """
        # Extraer precios
        y = df1['close'].values
        x = df2['close'].values
        
        # Inicializar filtro de Kalman
        n = len(y)
        hedge_ratio = np.zeros(n)
        e = np.zeros(n)
        Q = delta / (1 - delta) * np.ones(n)
        R = 0.001
        
        # Inicializar valores
        hedge_ratio[0] = 0
        P = np.ones(n)
        P[0] = 1.0
        
        # Ejecutar filtro de Kalman
        for t in range(1, n):
            # Predicción
            hedge_ratio[t] = hedge_ratio[t-1]
            P[t] = P[t-1] + Q[t]
            
            # Actualización
            K = P[t] * x[t] / (P[t] * x[t]**2 + R)
            hedge_ratio[t] = hedge_ratio[t] + K * (y[t] - hedge_ratio[t] * x[t])
            P[t] = P[t] * (1 - K * x[t])
            
            # Calcular error de spread
            e[t] = y[t] - hedge_ratio[t] * x[t]
        
        # Convertir a Series
        hedge_ratio_series = pd.Series(hedge_ratio, index=df1.index)
        spread = pd.Series(e, index=df1.index)
        
        # Calcular Z-score del spread
        z_score = StatisticalStrategy.zscore(spread, window=20)
        
        # Generar señales
        signals1 = pd.Series(0, index=z_score.index)
        signals2 = pd.Series(0, index=z_score.index)
        
        # Señales de entrada
        # Cuando spread está alto: vender activo 1, comprar activo 2
        signals1[z_score > z_entry] = -1
        signals2[z_score > z_entry] = hedge_ratio_series[z_score > z_entry]
        
        # Cuando spread está bajo: comprar activo 1, vender activo 2
        signals1[z_score < -z_entry] = 1
        signals2[z_score < -z_entry] = -hedge_ratio_series[z_score < -z_entry]
        
        # Señales de salida cuando el spread vuelve a la normalidad
        exit_condition = (z_score > -z_exit) & (z_score < z_exit)
        signals1[exit_condition] = 0
        signals2[exit_condition] = 0
        
        # Detectar cambios de señal
        position1 = signals1.diff().fillna(0)
        position2 = signals2.diff().fillna(0)
        
        return position1, position2


class MachineLearningStrategy:
    """Estrategias basadas en aprendizaje automático"""
    
    @staticmethod
    def create_features(df: pd.DataFrame, 
                       lookahead: int = 5, 
                       threshold: float = 0.005) -> pd.DataFrame:
        """
        Crea características (features) para modelos de aprendizaje automático
        
        Args:
            df: DataFrame con datos OHLC
            lookahead: Número de periodos para predecir
            threshold: Umbral para considerar movimiento significativo
            
        Returns:
            pd.DataFrame: DataFrame con características
        """
        # Crear copia para no modificar el original
        data = df.copy()
        
        # Crear etiquetas (target)
        future_return = data['close'].pct_change(lookahead).shift(-lookahead)
        data['target'] = 0
        data.loc[future_return > threshold, 'target'] = 1  # Subida significativa
        data.loc[future_return < -threshold, 'target'] = -1  # Bajada significativa
        
        # Características de precio
        data['return_1d'] = data['close'].pct_change(1)
        data['return_5d'] = data['close'].pct_change(5)
        data['return_10d'] = data['close'].pct_change(10)
        
        # Volatilidad
        data['volatility_5d'] = data['return_1d'].rolling(5).std()
        data['volatility_10d'] = data['return_1d'].rolling(10).std()
        
        # Características de volumen
        data['volume_change'] = data['volume'].pct_change(1)
        data['volume_ma5'] = data['volume'].rolling(5).mean()
        data['volume_ma10'] = data['volume'].rolling(10).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma5']
        
        # Características técnicas
        # RSI
        data['rsi_14'] = TechnicalIndicators.rsi(data['close'], 14)
        
        # MACD
        macd, signal, hist = TechnicalIndicators.macd(data['close'])
        data['macd'] = macd
        data['macd_signal'] = signal
        data['macd_hist'] = hist
        
        # Bandas de Bollinger
        middle, upper, lower = TechnicalIndicators.bollinger_bands(data['close'])
        data['bb_middle'] = middle
        data['bb_upper'] = upper
        data['bb_lower'] = lower
        data['bb_width'] = (upper - lower) / middle
        data['bb_position'] = (data['close'] - lower) / (upper - lower)
        
        # Distancia a medias móviles
        data['sma_10'] = TechnicalIndicators.sma(data['close'], 10)
        data['sma_20'] = TechnicalIndicators.sma(data['close'], 20)
        data['sma_50'] = TechnicalIndicators.sma(data['close'], 50)
        data['sma_200'] = TechnicalIndicators.sma(data['close'], 200)
        
        data['dist_sma_10'] = data['close'] / data['sma_10'] - 1
        data['dist_sma_20'] = data['close'] / data['sma_20'] - 1
        data['dist_sma_50'] = data['close'] / data['sma_50'] - 1
        data['dist_sma_200'] = data['close'] / data['sma_200'] - 1
        
        # Eliminar NaNs
        data = data.dropna()
        
        return data
    
    @staticmethod
    def train_classifier(df: pd.DataFrame, 
                        test_size: float = 0.2, 
                        random_state: int = 42) -> Tuple[Any, List[str], float]:
        """
        Entrena un modelo de clasificación para predecir dirección de precio
        
        Args:
            df: DataFrame con características y target
            test_size: Tamaño del conjunto de prueba
            random_state: Semilla para reproducibilidad
            
        Returns:
            Tuple: Modelo entrenado, lista de características, precisión del modelo
        """
        # Seleccionar características y target
        features = [col for col in df.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume', 'date']]
        
        X = df[features]
        y = df['target']
        
        # Normalizar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Entrenar modelo
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Importancia de características
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info(f"Top 5 features: {feature_importance['feature'][:5].tolist()}")
        
        return model, features, accuracy
    
    @staticmethod
    def train_with_time_series_cv(df: pd.DataFrame, 
                               n_splits: int = 5) -> Tuple[Any, List[str], float]:
        """
        Entrena un modelo con validación cruzada temporal
        
        Args:
            df: DataFrame con características y target
            n_splits: Número de divisiones para validación cruzada
            
        Returns:
            Tuple: Modelo entrenado, lista de características, precisión del modelo
        """
        # Seleccionar características y target
        features = [col for col in df.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume', 'date']]
        
        X = df[features]
        y = df['target']
        
        # Normalizar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Configurar validación cruzada temporal
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Inicializar modelo
        model = GradientBoostingClassifier(
            n_estimators=100, 
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        # Realizar validación cruzada
        accuracies = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracies.append(accuracy_score(y_test, y_pred))
        
        # Entrenar modelo final con todos los datos
        final_model = GradientBoostingClassifier(
            n_estimators=100, 
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        final_model.fit(X_scaled, y)
        
        # Importancia de características
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        avg_accuracy = np.mean(accuracies)
        logger.info(f"Average cross-validation accuracy: {avg_accuracy:.4f}")
        logger.info(f"Top 5 features: {feature_importance['feature'][:5].tolist()}")
        
        return final_model, features, avg_accuracy
    
    @staticmethod
    def predict_with_model(model: Any, df: pd.DataFrame, 
                          features: List[str]) -> pd.Series:
        """
        Genera predicciones con modelo entrenado
        
        Args:
            model: Modelo entrenado
            df: DataFrame con características
            features: Lista de características para el modelo
            
        Returns:
            pd.Series: Señales generadas
        """
        # Seleccionar características
        X = df[features]
        
        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Generar predicciones
        predictions = model.predict(X_scaled)
        
        # Convertir a Series
        signals = pd.Series(predictions, index=df.index)
        
        # Solo considerar cambios de señal
        position = signals.diff().fillna(0)
        
        return position


class AdaptiveStrategy:
    """Estrategia adaptativa que cambia según condiciones del mercado"""
    
    @staticmethod
    def detect_market_regime(df: pd.DataFrame, 
                            lookback: int = 20) -> Tuple[pd.Series, str]:
        """
        Detecta el régimen actual del mercado
        
        Args:
            df: DataFrame con datos OHLC
            lookback: Periodo para análisis
            
        Returns:
            Tuple[pd.Series, str]: Serie con regímenes y régimen actual
        """
        # Calcular indicadores
        adx, plus_di, minus_di = TechnicalIndicators.adx(
            df['high'], df['low'], df['close'], period=14
        )
        
        # Calcular volatilidad
        volatility = df['close'].pct_change().rolling(lookback).std()
        
        # Inicializar serie de regímenes
        regime = pd.Series('Unknown', index=df.index)
        
        # Definir regímenes
        # Tendencia alcista fuerte
        regime[(adx > 25) & (plus_di > minus_di) & (volatility < volatility.quantile(0.7))] = 'Strong_Uptrend'
        
        # Tendencia bajista fuerte
        regime[(adx > 25) & (minus_di > plus_di) & (volatility < volatility.quantile(0.7))] = 'Strong_Downtrend'
        
        # Rango con baja volatilidad
        regime[(adx < 20) & (volatility < volatility.quantile(0.5))] = 'Low_Vol_Range'
        
        # Rango con alta volatilidad
        regime[(adx < 20) & (volatility > volatility.quantile(0.7))] = 'High_Vol_Range'
        
        # Volatilidad extrema
        regime[volatility > volatility.quantile(0.9)] = 'Extreme_Volatility'
        
        # Régimen actual
        current_regime = regime.iloc[-1]
        
        return regime, current_regime
    
    @staticmethod
    def get_adaptive_strategy(df: pd.DataFrame) -> pd.Series:
        """
        Aplica estrategia adaptativa según el régimen de mercado
        
        Args:
            df: DataFrame con datos OHLC
            
        Returns:
            pd.Series: Señales generadas
        """
        # Detectar régimen
        regime, current_regime = AdaptiveStrategy.detect_market_regime(df)
        
        # Añadir régimen al DataFrame
        df['regime'] = regime
        
        # Inicializar señales
        position = pd.Series(0, index=df.index)
        
        # Aplicar estrategia según régimen
        # En tendencia alcista: usar estrategia de seguimiento de tendencia
        if current_regime == 'Strong_Uptrend':
            position = ClassicStrategy.moving_average_crossover(df, fast_period=10, slow_period=30)
        
        # En tendencia bajista: usar estrategia de seguimiento de tendencia
        elif current_regime == 'Strong_Downtrend':
            position = ClassicStrategy.moving_average_crossover(df, fast_period=10, slow_period=30)
        
        # En rango con baja volatilidad: usar estrategia de reversión a la media
        elif current_regime == 'Low_Vol_Range':
            position = StatisticalStrategy.mean_reversion_strategy(df, window=10, z_entry=1.5, z_exit=0.3)
        
        # En rango con alta volatilidad: usar estrategia de bandas de Bollinger
        elif current_regime == 'High_Vol_Range':
            position = ClassicStrategy.bollinger_strategy(df, period=10, num_std_dev=2.5)
        
        # En volatilidad extrema: no operar o reducir tamaño
        elif current_regime == 'Extreme_Volatility':
            position = pd.Series(0, index=df.index)  # No operar
        
        return position
    
    @staticmethod
    def ensemble_strategy(df: pd.DataFrame, weights: Dict[str, float] = None) -> pd.Series:
        """
        Combina múltiples estrategias con pesos
        
        Args:
            df: DataFrame con datos OHLC
            weights: Diccionario con pesos para cada estrategia
            
        Returns:
            pd.Series: Señales ponderadas
        """
        if weights is None:
            weights = {
                'ma_crossover': 0.3,
                'bollinger': 0.3,
                'rsi': 0.2,
                'mean_reversion': 0.2
            }
        
        # Calcular señales de cada estrategia
        ma_signals = ClassicStrategy.moving_average_crossover(df.copy())
        bb_signals = ClassicStrategy.bollinger_strategy(df.copy())
        rsi_signals = ClassicStrategy.rsi_strategy(df.copy())
        mr_signals = StatisticalStrategy.mean_reversion_strategy(df.copy())
        
        # Combinar señales con pesos
        combined = (
            weights['ma_crossover'] * ma_signals +
            weights['bollinger'] * bb_signals +
            weights['rsi'] * rsi_signals +
            weights['mean_reversion'] * mr_signals
        )
        
        # Normalizar y discretizar señales
        threshold = 0.3
        signals = pd.Series(0, index=combined.index)
        signals[combined > threshold] = 1
        signals[combined < -threshold] = -1
        
        # Detectar cambios de señal
        position = signals.diff().fillna(0)
        
        return position


class RiskManagement:
    """Gestión de riesgo avanzada para estrategias de trading"""
    
    @staticmethod
    def calculate_position_size(account_balance: float, 
                               risk_per_trade: float, 
                               stop_loss_pct: float, 
                               entry_price: float) -> float:
        """
        Calcula el tamaño de posición basado en riesgo
        
        Args:
            account_balance: Balance de la cuenta
            risk_per_trade: Porcentaje de riesgo por operación (0-1)
            stop_loss_pct: Porcentaje de stop loss (0-1)
            entry_price: Precio de entrada
            
        Returns:
            float: Tamaño de posición en unidades
        """
        # Cantidad a arriesgar
        risk_amount = account_balance * risk_per_trade
        
        # Pérdida por unidad
        per_unit_risk = entry_price * stop_loss_pct
        
        # Calcular tamaño de posición
        position_size = risk_amount / per_unit_risk
        
        return position_size
    
    @staticmethod
    def atr_position_size(df: pd.DataFrame, 
                         account_balance: float, 
                         risk_per_trade: float, 
                         atr_multiplier: float = 2.0) -> float:
        """
        Calcula tamaño de posición basado en ATR
        
        Args:
            df: DataFrame con datos OHLC
            account_balance: Balance de la cuenta
            risk_per_trade: Porcentaje de riesgo por operación (0-1)
            atr_multiplier: Múltiplo de ATR para stop loss
            
        Returns:
            float: Tamaño de posición en unidades
        """
        # Calcular ATR
        atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        
        # Precio actual
        current_price = df['close'].iloc[-1]
        
        # Stop loss en valor absoluto
        stop_loss_value = atr.iloc[-1] * atr_multiplier
        
        # Cantidad a arriesgar
        risk_amount = account_balance * risk_per_trade
        
        # Calcular tamaño de posición
        position_size = risk_amount / stop_loss_value
        
        return position_size
    
    @staticmethod
    def calculate_stop_loss(df: pd.DataFrame, 
                           entry_price: float, 
                           position_type: str, 
                           atr_multiplier: float = 2.0) -> float:
        """
        Calcula nivel de stop loss basado en ATR
        
        Args:
            df: DataFrame con datos OHLC
            entry_price: Precio de entrada
            position_type: Tipo de posición ('long' o 'short')
            atr_multiplier: Múltiplo de ATR para stop loss
            
        Returns:
            float: Nivel de stop loss
        """
        # Calcular ATR
        atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        
        # Calcular stop loss
        if position_type.lower() == 'long':
            stop_loss = entry_price - (atr.iloc[-1] * atr_multiplier)
        else:  # Short
            stop_loss = entry_price + (atr.iloc[-1] * atr_multiplier)
        
        return stop_loss
    
    @staticmethod
    def calculate_take_profit(entry_price: float, 
                             stop_loss: float, 
                             risk_reward_ratio: float = 2.0) -> float:
        """
        Calcula nivel de take profit basado en risk/reward
        
        Args:
            entry_price: Precio de entrada
            stop_loss: Nivel de stop loss
            risk_reward_ratio: Ratio riesgo/recompensa deseado
            
        Returns:
            float: Nivel de take profit
        """
        # Calcular distancia al stop loss
        if stop_loss < entry_price:  # Long position
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * risk_reward_ratio)
        else:  # Short position
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * risk_reward_ratio)
        
        return take_profit
    
    @staticmethod
    def calculate_trailing_stop(df: pd.DataFrame, 
                              position_type: str, 
                              entry_price: float, 
                              current_price: float, 
                              atr_multiplier: float = 2.0) -> float:
        """
        Calcula trailing stop dinámico
        
        Args:
            df: DataFrame con datos OHLC
            position_type: Tipo de posición ('long' o 'short')
            entry_price: Precio de entrada
            current_price: Precio actual
            atr_multiplier: Múltiplo de ATR para trailing stop
            
        Returns:
            float: Nivel de trailing stop
        """
        # Calcular ATR
        atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        
        # Calcular trailing stop
        if position_type.lower() == 'long':
            # Para posición larga, trailing stop debajo del precio
            trailing_stop = current_price - (atr.iloc[-1] * atr_multiplier)
            
            # No bajar el trailing stop
            initial_stop = entry_price - (atr.iloc[-1] * atr_multiplier)
            trailing_stop = max(trailing_stop, initial_stop)
            
        else:  # Short
            # Para posición corta, trailing stop encima del precio
            trailing_stop = current_price + (atr.iloc[-1] * atr_multiplier)
            
            # No subir el trailing stop
            initial_stop = entry_price + (atr.iloc[-1] * atr_multiplier)
            trailing_stop = min(trailing_stop, initial_stop)
        
        return trailing_stop
    
    @staticmethod
    def should_skip_trading(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Determina si se debe evitar operar por condiciones de mercado
        
        Args:
            df: DataFrame con datos OHLC
            
        Returns:
            Tuple[bool, str]: Bool indicando si evitar y razón
        """
        reasons = []
        
        # Verificar gaps grandes
        close_to_open_gap = abs(df['open'].iloc[-1] / df['close'].iloc[-2] - 1)
        if close_to_open_gap > 0.03:  # Gap de más del 3%
            reasons.append(f"Large gap detected: {close_to_open_gap:.2%}")
        
        # Verificar volatilidad extrema
        volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
        avg_volatility = df['close'].pct_change().rolling(20).std().rolling(100).mean().iloc[-1]
        
        if volatility > avg_volatility * 2:
            reasons.append(f"Extreme volatility: {volatility:.2%} vs avg {avg_volatility:.2%}")
        
        # Verificar noticias (simulado, en una implementación real se conectaría a una API de noticias)
        # news_impact = check_news_impact(df.index[-1])
        # if news_impact > 0.7:
        #     reasons.append(f"High impact news expected")
        
        # Verificar horario (mercado de poca liquidez)
        # hour = df.index[-1].hour
        # if 0 <= hour < 3:  # Horario de baja liquidez
        #     reasons.append(f"Low liquidity hours")
        
        should_skip = len(reasons) > 0
        reason = "; ".join(reasons) if should_skip else ""
        
        return should_skip, reason
    
    @staticmethod
    def check_max_drawdown(equity_curve: pd.Series, 
                          max_drawdown_pct: float = 0.1) -> Tuple[bool, float]:
        """
        Verifica si se ha alcanzado el drawdown máximo permitido
        
        Args:
            equity_curve: Serie con el equity curve
            max_drawdown_pct: Drawdown máximo permitido (0-1)
            
        Returns:
            Tuple[bool, float]: Si se ha alcanzado el max drawdown y drawdown actual
        """
        # Calcular máximo acumulado
        running_max = equity_curve.cummax()
        
        # Calcular drawdown
        drawdown = (equity_curve / running_max - 1)
        
        # Drawdown actual
        current_drawdown = abs(drawdown.iloc[-1])
        
        # Drawdown máximo histórico
        max_drawdown = abs(drawdown.min())
        
        # Verificar si se alcanzó el límite
        stop_trading = current_drawdown >= max_drawdown_pct
        
        return stop_trading, current_drawdown