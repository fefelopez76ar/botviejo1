"""
Módulo para análisis de patrones de mercado y toma de decisiones inteligentes

Este módulo implementa:
- Detección de patrones de reversión
- Análisis de probabilidades estadísticas
- Cálculo de amplitudes de precio por período de tiempo
- Determinación de mejores momentos para operar
- Evaluación de conveniencia de comprar/vender basada en patrones históricos
"""

import numpy as np
import pandas as pd
import logging
import math
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
from scipy import stats

logger = logging.getLogger("PatternAnalyzer")

class PatternType(Enum):
    """Tipos de patrones de mercado identificables"""
    REVERSAL_TOP = "reversal_top"               # Patrón de techo de reversión
    REVERSAL_BOTTOM = "reversal_bottom"         # Patrón de suelo de reversión
    CONTINUATION = "continuation"               # Patrón de continuación
    BREAKOUT = "breakout"                       # Ruptura de nivel
    BREAKDOWN = "breakdown"                     # Ruptura a la baja
    RANGE_BOUND = "range_bound"                 # Rango consolidado
    VOLATILITY_EXPANSION = "volatility_expansion"  # Expansión de volatilidad
    VOLATILITY_CONTRACTION = "volatility_contraction"  # Contracción de volatilidad

class SignalStrength(Enum):
    """Fuerza de una señal de trading"""
    STRONG = "strong"           # Señal fuerte
    MODERATE = "moderate"       # Señal moderada
    WEAK = "weak"               # Señal débil
    NOISE = "noise"             # Ruido de mercado, sin señal clara

class PatternAnalyzer:
    """
    Analizador de patrones de mercado para toma de decisiones
    
    Esta clase identifica patrones en datos de mercado, evalúa su fiabilidad
    estadística y determina la conveniencia de trading basándose en
    comportamientos históricos y amplitudes de precio.
    """
    
    def __init__(self, min_pattern_bars: int = 5, price_decimals: int = 2,
               volatility_window: int = 20):
        """
        Inicializa el analizador de patrones
        
        Args:
            min_pattern_bars: Mínimo número de velas para identificar un patrón
            price_decimals: Decimales para redondeo de precios
            volatility_window: Ventana para cálculo de volatilidad
        """
        self.min_pattern_bars = min_pattern_bars
        self.price_decimals = price_decimals
        self.volatility_window = volatility_window
        self.patterns_history = []
        self.success_rate_cache = {}
    
    def calculate_price_amplitudes(self, df: pd.DataFrame, 
                                 intervals: List[str] = ['1m', '5m', '15m', '1h', '4h', '1d']
                                ) -> Dict[str, float]:
        """
        Calcula las amplitudes de precio típicas para diferentes intervalos
        
        Args:
            df: DataFrame con datos OHLCV
            intervals: Lista de intervalos a analizar
            
        Returns:
            Dict[str, float]: Amplitudes por intervalo
        """
        results = {}
        
        # Asumimos que df es para el intervalo más pequeño (por ejemplo, 1m)
        # y agregamos para intervalos más grandes
        
        # Calcular amplitud para el intervalo base
        typical_amplitude = df['high'].rolling(window=100).max() - df['low'].rolling(window=100).min()
        typical_amplitude = typical_amplitude.median()
        base_interval = intervals[0]
        results[base_interval] = typical_amplitude
        
        # Calcular amplitudes para intervalos mayores por agregación
        interval_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
        base_minutes = interval_minutes.get(base_interval, 1)
        
        for interval in intervals[1:]:
            if interval in interval_minutes:
                multiplier = interval_minutes[interval] / base_minutes
                # La amplitud no crece linealmente con el tiempo, sino aproximadamente con la raíz cuadrada
                results[interval] = typical_amplitude * (multiplier ** 0.5)
        
        return results
    
    def calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula métricas de volatilidad para el instrumento
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            Dict[str, float]: Métricas de volatilidad
        """
        # Calcular retornos porcentuales
        returns = df['close'].pct_change().dropna()
        
        # Volatilidad histórica (desviación estándar de retornos)
        volatility = returns.rolling(window=self.volatility_window).std()
        current_volatility = volatility.iloc[-1]
        
        # Volatilidad anualizada
        annualized_volatility = current_volatility * np.sqrt(252)  # Asumiendo 252 días de trading por año
        
        # Volatilidad relativa (comparada con su media histórica)
        avg_volatility = volatility.mean()
        relative_volatility = current_volatility / avg_volatility if avg_volatility > 0 else 1.0
        
        # Average True Range (ATR)
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        current_atr = atr.iloc[-1]
        
        # ATR como porcentaje del precio
        atr_pct = (current_atr / df['close'].iloc[-1]) * 100
        
        return {
            'current_volatility': current_volatility,
            'annualized_volatility': annualized_volatility,
            'relative_volatility': relative_volatility,
            'atr': current_atr,
            'atr_pct': atr_pct
        }
    
    def calculate_reversal_probability(self, df: pd.DataFrame, lookback: int = 100) -> Dict[str, Any]:
        """
        Calcula la probabilidad de reversión basada en patrones históricos
        
        Args:
            df: DataFrame con datos OHLCV
            lookback: Número de velas a analizar
            
        Returns:
            Dict[str, Any]: Probabilidades de reversión
        """
        if len(df) < lookback + 10:
            lookback = max(30, len(df) - 10)
        
        # Limitar el análisis a las velas recientes
        df_recent = df.iloc[-lookback:]
        
        # Detectar picos y valles (posibles puntos de reversión)
        df_recent['prev_close'] = df_recent['close'].shift(1)
        df_recent['next_close'] = df_recent['close'].shift(-1)
        
        # Un pico local es donde el cierre es mayor que el anterior y el siguiente
        peaks = (df_recent['close'] > df_recent['prev_close']) & (df_recent['close'] > df_recent['next_close'])
        
        # Un valle local es donde el cierre es menor que el anterior y el siguiente
        valleys = (df_recent['close'] < df_recent['prev_close']) & (df_recent['close'] < df_recent['next_close'])
        
        # Contar picos y valles
        peak_count = peaks.sum()
        valley_count = valleys.sum()
        total_potential_reversals = peak_count + valley_count
        
        # Contar cuántas veces estos picos/valles resultaron en una reversión real
        # Definimos una reversión como un movimiento del precio en la dirección opuesta
        # de al menos X% en las próximas Y velas
        
        # Parámetros para definir una reversión
        reversal_threshold_pct = 0.01  # 1% de movimiento
        forward_bars = 5  # Siguiente 5 velas
        
        # Marcar picos que fueron seguidos por una caída significativa
        df_recent['future_min'] = df_recent['low'].rolling(window=forward_bars).min().shift(-forward_bars)
        valid_peaks = peaks & ((df_recent['close'] - df_recent['future_min']) / df_recent['close'] > reversal_threshold_pct)
        
        # Marcar valles que fueron seguidos por un alza significativa
        df_recent['future_max'] = df_recent['high'].rolling(window=forward_bars).max().shift(-forward_bars)
        valid_valleys = valleys & ((df_recent['future_max'] - df_recent['close']) / df_recent['close'] > reversal_threshold_pct)
        
        # Contar reversiones válidas
        valid_peak_count = valid_peaks.sum()
        valid_valley_count = valid_valleys.sum()
        total_valid_reversals = valid_peak_count + valid_valley_count
        
        # Calcular probabilidades
        peak_success_rate = valid_peak_count / peak_count if peak_count > 0 else 0
        valley_success_rate = valid_valley_count / valley_count if valley_count > 0 else 0
        overall_reversal_prob = total_valid_reversals / total_potential_reversals if total_potential_reversals > 0 else 0
        
        # Clasificar el momento actual
        is_recent_peak = peaks.iloc[-1] if len(peaks) > 0 else False
        is_recent_valley = valleys.iloc[-1] if len(valleys) > 0 else False
        
        current_pattern = "none"
        current_reversal_prob = 0.0
        
        if is_recent_peak:
            current_pattern = "potential_top"
            current_reversal_prob = peak_success_rate
        elif is_recent_valley:
            current_pattern = "potential_bottom"
            current_reversal_prob = valley_success_rate
        
        return {
            'peak_success_rate': peak_success_rate,
            'valley_success_rate': valley_success_rate,
            'overall_reversal_prob': overall_reversal_prob,
            'current_pattern': current_pattern,
            'current_reversal_prob': current_reversal_prob,
            'peak_count': peak_count,
            'valley_count': valley_count,
            'valid_peak_count': valid_peak_count,
            'valid_valley_count': valid_valley_count
        }
    
    def identify_chart_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identifica patrones gráficos comunes en los datos
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            List[Dict[str, Any]]: Patrones identificados
        """
        patterns = []
        
        # Asegurarse de que hay suficientes datos
        if len(df) < self.min_pattern_bars * 3:
            return patterns
        
        # 1. Detectar Doble Techo (Double Top)
        self._detect_double_top(df, patterns)
        
        # 2. Detectar Doble Suelo (Double Bottom)
        self._detect_double_bottom(df, patterns)
        
        # 3. Detectar Hombro-Cabeza-Hombro (Head and Shoulders)
        self._detect_head_and_shoulders(df, patterns)
        
        # 4. Detectar Triángulo Ascendente
        self._detect_ascending_triangle(df, patterns)
        
        # 5. Detectar Triángulo Descendente
        self._detect_descending_triangle(df, patterns)
        
        # 6. Detectar Formación de Bandera (Flag)
        self._detect_flag_pattern(df, patterns)
        
        return patterns
    
    def _detect_double_top(self, df: pd.DataFrame, patterns: List[Dict[str, Any]]):
        """Detecta patrón de Doble Techo"""
        # Implementación simplificada
        window = min(20, len(df) // 3)
        
        # Encontrar máximos locales
        df['rolling_max'] = df['high'].rolling(window=window, center=True).max()
        is_max = (df['high'] == df['rolling_max'])
        
        max_indices = is_max[is_max].index.tolist()
        
        # Buscar dos máximos similares con un valle entre ellos
        if len(max_indices) >= 2:
            for i in range(len(max_indices) - 1):
                idx1 = max_indices[i]
                idx2 = max_indices[i + 1]
                
                # Verificar que los máximos están separados por un mínimo
                if idx2 - idx1 > window // 2:
                    price1 = df.loc[idx1, 'high']
                    price2 = df.loc[idx2, 'high']
                    
                    # Verificar que los precios son similares (dentro del 1%)
                    if abs(price1 - price2) / price1 < 0.01:
                        # Encontrar el mínimo entre los dos máximos
                        between_min = df.loc[idx1:idx2, 'low'].min()
                        
                        # Verificar que hay un mínimo significativo entre ellos
                        if (price1 - between_min) / price1 > 0.01:
                            patterns.append({
                                'type': 'double_top',
                                'pattern_type': PatternType.REVERSAL_TOP,
                                'start_idx': idx1,
                                'end_idx': idx2,
                                'first_price': price1,
                                'second_price': price2,
                                'strength': SignalStrength.MODERATE,
                                'target': between_min,  # Objetivo es el mínimo previo
                                'entry': df.loc[idx2, 'close'],  # Entrada en el cierre del segundo pico
                                'stop_loss': max(price1, price2) * 1.01  # Stop loss por encima del máximo
                            })
    
    def _detect_double_bottom(self, df: pd.DataFrame, patterns: List[Dict[str, Any]]):
        """Detecta patrón de Doble Suelo"""
        # Implementación simplificada
        window = min(20, len(df) // 3)
        
        # Encontrar mínimos locales
        df['rolling_min'] = df['low'].rolling(window=window, center=True).min()
        is_min = (df['low'] == df['rolling_min'])
        
        min_indices = is_min[is_min].index.tolist()
        
        # Buscar dos mínimos similares con un máximo entre ellos
        if len(min_indices) >= 2:
            for i in range(len(min_indices) - 1):
                idx1 = min_indices[i]
                idx2 = min_indices[i + 1]
                
                # Verificar que los mínimos están separados por un máximo
                if idx2 - idx1 > window // 2:
                    price1 = df.loc[idx1, 'low']
                    price2 = df.loc[idx2, 'low']
                    
                    # Verificar que los precios son similares (dentro del 1%)
                    if abs(price1 - price2) / price1 < 0.01:
                        # Encontrar el máximo entre los dos mínimos
                        between_max = df.loc[idx1:idx2, 'high'].max()
                        
                        # Verificar que hay un máximo significativo entre ellos
                        if (between_max - price1) / price1 > 0.01:
                            patterns.append({
                                'type': 'double_bottom',
                                'pattern_type': PatternType.REVERSAL_BOTTOM,
                                'start_idx': idx1,
                                'end_idx': idx2,
                                'first_price': price1,
                                'second_price': price2,
                                'strength': SignalStrength.MODERATE,
                                'target': between_max,  # Objetivo es el máximo previo
                                'entry': df.loc[idx2, 'close'],  # Entrada en el cierre del segundo suelo
                                'stop_loss': min(price1, price2) * 0.99  # Stop loss por debajo del mínimo
                            })
    
    def _detect_head_and_shoulders(self, df: pd.DataFrame, patterns: List[Dict[str, Any]]):
        """Detecta patrón de Hombro-Cabeza-Hombro"""
        # Implementación simplificada - esta es una aproximación básica
        window = min(10, len(df) // 5)
        
        # Encontrar máximos locales
        df['rolling_max'] = df['high'].rolling(window=window, center=True).max()
        is_max = (df['high'] == df['rolling_max'])
        
        max_indices = is_max[is_max].index.tolist()
        
        # Se necesitan 3 máximos para este patrón
        if len(max_indices) >= 3:
            for i in range(len(max_indices) - 2):
                left_idx = max_indices[i]
                head_idx = max_indices[i + 1]
                right_idx = max_indices[i + 2]
                
                # Verificar espaciado apropiado
                if head_idx - left_idx > window and right_idx - head_idx > window:
                    left_price = df.loc[left_idx, 'high']
                    head_price = df.loc[head_idx, 'high']
                    right_price = df.loc[right_idx, 'high']
                    
                    # El máximo central debe ser mayor que los laterales
                    if head_price > left_price and head_price > right_price:
                        # Los hombros deben tener precios similares
                        if abs(left_price - right_price) / left_price < 0.05:
                            # Encontrar la "línea de cuello" (neckline)
                            min1 = df.loc[left_idx:head_idx, 'low'].min()
                            min2 = df.loc[head_idx:right_idx, 'low'].min()
                            neckline = (min1 + min2) / 2
                            
                            patterns.append({
                                'type': 'head_and_shoulders',
                                'pattern_type': PatternType.REVERSAL_TOP,
                                'left_idx': left_idx,
                                'head_idx': head_idx,
                                'right_idx': right_idx,
                                'left_price': left_price,
                                'head_price': head_price,
                                'right_price': right_price,
                                'neckline': neckline,
                                'strength': SignalStrength.STRONG,
                                'target': neckline - (head_price - neckline),  # Objetivo es la distancia de cabeza a cuello
                                'entry': df.loc[right_idx, 'close'],  # Entrada tras el hombro derecho
                                'stop_loss': head_price * 1.01  # Stop loss por encima de la cabeza
                            })
    
    def _detect_ascending_triangle(self, df: pd.DataFrame, patterns: List[Dict[str, Any]]):
        """Detecta patrón de Triángulo Ascendente"""
        # Implementación simplificada
        window = min(30, len(df) // 3)
        min_points = 3
        
        # Se necesitan al menos 'min_points' máximos similares y mínimos ascendentes
        if len(df) < window * 2:
            return
        
        # Buscar una resistencia horizontal (máximos similares)
        max_points = []
        for i in range(window, len(df) - window):
            current_high = df.iloc[i]['high']
            prev_high = df.iloc[i - window:i]['high'].max()
            next_high = df.iloc[i:i + window]['high'].max()
            
            if abs(current_high - prev_high) / prev_high < 0.01 and current_high >= next_high * 0.99:
                max_points.append((df.index[i], current_high))
        
        # Buscar mínimos ascendentes
        min_points = []
        for i in range(window, len(df) - window):
            current_low = df.iloc[i]['low']
            prev_low = df.iloc[i - window:i]['low'].min()
            next_low = df.iloc[i:i + window]['low'].min()
            
            if current_low > prev_low * 1.01 and current_low <= next_low * 1.01:
                min_points.append((df.index[i], current_low))
        
        if len(max_points) >= min_points and len(min_points) >= min_points:
            # Verificar que hay un patrón de triángulo (resistencia horizontal, soporte ascendente)
            resistance_level = np.median([p[1] for p in max_points])
            
            # Calcular línea de tendencia de soporte (simplificado)
            if len(min_points) >= 2:
                first_min = min_points[0]
                last_min = min_points[-1]
                
                # Verificar que los mínimos son ascendentes
                if last_min[1] > first_min[1]:
                    patterns.append({
                        'type': 'ascending_triangle',
                        'pattern_type': PatternType.CONTINUATION,
                        'start_idx': first_min[0],
                        'end_idx': last_min[0],
                        'resistance': resistance_level,
                        'strength': SignalStrength.MODERATE,
                        'target': resistance_level * 1.02,  # Objetivo por encima de la resistencia
                        'entry': resistance_level * 1.01,  # Entrada en ruptura confirmada
                        'stop_loss': last_min[1] * 0.99  # Stop loss debajo del último mínimo
                    })
    
    def _detect_descending_triangle(self, df: pd.DataFrame, patterns: List[Dict[str, Any]]):
        """Detecta patrón de Triángulo Descendente"""
        # Implementación simplificada (inversa del triángulo ascendente)
        window = min(30, len(df) // 3)
        min_points = 3
        
        # Se necesitan al menos 'min_points' mínimos similares y máximos descendentes
        if len(df) < window * 2:
            return
        
        # Buscar un soporte horizontal (mínimos similares)
        min_points_list = []
        for i in range(window, len(df) - window):
            current_low = df.iloc[i]['low']
            prev_low = df.iloc[i - window:i]['low'].min()
            next_low = df.iloc[i:i + window]['low'].min()
            
            if abs(current_low - prev_low) / prev_low < 0.01 and current_low <= next_low * 1.01:
                min_points_list.append((df.index[i], current_low))
        
        # Buscar máximos descendentes
        max_points = []
        for i in range(window, len(df) - window):
            current_high = df.iloc[i]['high']
            prev_high = df.iloc[i - window:i]['high'].max()
            next_high = df.iloc[i:i + window]['high'].max()
            
            if current_high < prev_high * 0.99 and current_high >= next_high * 0.99:
                max_points.append((df.index[i], current_high))
        
        if len(min_points_list) >= min_points and len(max_points) >= min_points:
            # Verificar que hay un patrón de triángulo (soporte horizontal, resistencia descendente)
            support_level = np.median([p[1] for p in min_points_list])
            
            # Calcular línea de tendencia de resistencia (simplificado)
            if len(max_points) >= 2:
                first_max = max_points[0]
                last_max = max_points[-1]
                
                # Verificar que los máximos son descendentes
                if last_max[1] < first_max[1]:
                    patterns.append({
                        'type': 'descending_triangle',
                        'pattern_type': PatternType.CONTINUATION,
                        'start_idx': first_max[0],
                        'end_idx': last_max[0],
                        'support': support_level,
                        'strength': SignalStrength.MODERATE,
                        'target': support_level * 0.98,  # Objetivo por debajo del soporte
                        'entry': support_level * 0.99,  # Entrada en ruptura confirmada
                        'stop_loss': last_max[1] * 1.01  # Stop loss arriba del último máximo
                    })
    
    def _detect_flag_pattern(self, df: pd.DataFrame, patterns: List[Dict[str, Any]]):
        """Detecta patrón de Bandera"""
        # Implementación simplificada
        # Una bandera es un patrón de consolidación tras un movimiento fuerte
        window = min(15, len(df) // 5)
        
        # Primero, detectar un movimiento fuerte (tendencia)
        for i in range(window * 2, len(df) - window):
            # Calcular retorno en la ventana previa
            start_price = df.iloc[i - window * 2]['close']
            pre_end_price = df.iloc[i - window]['close']
            
            price_change = (pre_end_price - start_price) / start_price
            
            # Considerar un movimiento fuerte si el cambio es > 3%
            is_strong_move = abs(price_change) > 0.03
            
            if is_strong_move:
                # Verificar si hay una consolidación (rango más estrecho) después
                highs = df.iloc[i - window:i]['high']
                lows = df.iloc[i - window:i]['low']
                
                pre_range = highs.max() - lows.min()
                
                # Verificar el rango en el período posterior
                post_highs = df.iloc[i:i + window]['high']
                post_lows = df.iloc[i:i + window]['low']
                
                post_range = post_highs.max() - post_lows.min()
                
                # Una bandera tiene un rango más estrecho que el movimiento anterior
                is_flag = post_range < pre_range * 0.7
                
                if is_flag:
                    flag_direction = "bullish" if price_change > 0 else "bearish"
                    
                    patterns.append({
                        'type': 'flag',
                        'pattern_type': PatternType.CONTINUATION,
                        'direction': flag_direction,
                        'start_idx': df.index[i - window * 2],
                        'pole_end_idx': df.index[i - window],
                        'flag_end_idx': df.index[i + window],
                        'strength': SignalStrength.MODERATE,
                        'target': pre_end_price + (pre_end_price - start_price) if flag_direction == "bullish" else
                                 pre_end_price - (start_price - pre_end_price),
                        'entry': df.iloc[i + window]['close'],  # Entrada al final del período de consolidación
                        'stop_loss': post_lows.min() * 0.99 if flag_direction == "bullish" else post_highs.max() * 1.01
                    })
    
    def analyze_recent_price_behavior(self, df: pd.DataFrame, lookback_bars: int = 100) -> Dict[str, Any]:
        """
        Analiza el comportamiento reciente de precios para identificar patrones
        
        Args:
            df: DataFrame con datos OHLCV
            lookback_bars: Número de velas a analizar
            
        Returns:
            Dict[str, Any]: Análisis del comportamiento reciente de precios
        """
        if len(df) < lookback_bars + 20:
            lookback_bars = max(50, len(df) - 20)
        
        recent_df = df.iloc[-lookback_bars:]
        
        # Calcular retornos
        returns = recent_df['close'].pct_change().dropna()
        
        # Calcular estadísticas de retornos
        mean_return = returns.mean()
        std_return = returns.std()
        skew = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Calcular distribución de retornos por cuartiles
        q1 = returns.quantile(0.25)
        q2 = returns.quantile(0.50)  # mediana
        q3 = returns.quantile(0.75)
        
        # Calcular proporción de velas alcistas vs bajistas
        bullish_bars = (recent_df['close'] > recent_df['open']).sum()
        bearish_bars = (recent_df['close'] < recent_df['open']).sum()
        doji_bars = lookback_bars - bullish_bars - bearish_bars
        
        bullish_ratio = bullish_bars / lookback_bars
        bearish_ratio = bearish_bars / lookback_bars
        
        # Calcular longitud promedio de secuencias (runs)
        up_down_seq = np.sign(returns)
        
        run_lengths = []
        current_run = 1
        
        for i in range(1, len(up_down_seq)):
            if up_down_seq[i] == up_down_seq[i-1]:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        
        run_lengths.append(current_run)  # Añadir la última secuencia
        
        avg_run_length = np.mean(run_lengths) if run_lengths else 0
        max_run_length = np.max(run_lengths) if run_lengths else 0
        
        # Calcular la tendencia actual (con linear regression)
        x = np.arange(len(recent_df))
        y = recent_df['close'].values
        
        slope, _, r_value, _, _ = stats.linregress(x, y)
        
        # Normalizar pendiente como % del precio medio
        avg_price = recent_df['close'].mean()
        norm_slope = (slope / avg_price) * 100
        
        # Determinar fuerza de la tendencia
        trend_strength = abs(r_value)
        
        # Calcular RSI para determinar sobrecompra/sobreventa
        delta = recent_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        # Calcular proporción de volumen
        recent_volume = recent_df['volume'].iloc[-5:].mean() if 'volume' in recent_df else 0
        avg_volume = recent_df['volume'].mean() if 'volume' in recent_df else 0
        
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Calcular días desde último máximo/mínimo
        price_max = recent_df['high'].max()
        price_min = recent_df['low'].min()
        
        days_since_max = len(recent_df) - recent_df['high'].argmax() - 1
        days_since_min = len(recent_df) - recent_df['low'].argmin() - 1
        
        # Analizar volatilidad reciente
        volatility_metrics = self.calculate_volatility_metrics(recent_df)
        
        # Calcular niveles clave
        recent_close = recent_df['close'].iloc[-1]
        support_1 = recent_df['low'].nlargest(3).mean()  # Promedio de los 3 mínimos más altos
        support_2 = recent_df['low'].nsmallest(3).mean()  # Promedio de los 3 mínimos más bajos
        resistance_1 = recent_df['high'].nsmallest(3).mean()  # Promedio de los 3 máximos más bajos
        resistance_2 = recent_df['high'].nlargest(3).mean()  # Promedio de los 3 máximos más altos
        
        return {
            'mean_return': mean_return,
            'std_return': std_return,
            'skew': skew,
            'kurtosis': kurtosis,
            'quantiles': {
                'q1': q1,
                'median': q2,
                'q3': q3
            },
            'bar_distribution': {
                'bullish_ratio': bullish_ratio,
                'bearish_ratio': bearish_ratio,
                'doji_ratio': doji_bars / lookback_bars
            },
            'run_analysis': {
                'avg_run_length': avg_run_length,
                'max_run_length': max_run_length
            },
            'trend_analysis': {
                'slope': slope,
                'norm_slope_pct': norm_slope,
                'r_squared': r_value ** 2,
                'trend_strength': trend_strength
            },
            'rsi': current_rsi,
            'volume_ratio': volume_ratio,
            'days_since_max': days_since_max,
            'days_since_min': days_since_min,
            'volatility': volatility_metrics,
            'key_levels': {
                'recent_close': recent_close,
                'support_1': support_1,
                'support_2': support_2,
                'resistance_1': resistance_1,
                'resistance_2': resistance_2
            }
        }
    
    def evaluate_trading_decision(self, df: pd.DataFrame, timeframe: str = '1h', 
                                risk_reward_min: float = 1.5) -> Dict[str, Any]:
        """
        Evalúa la conveniencia de compra/venta basada en análisis completo
        
        Args:
            df: DataFrame con datos OHLCV
            timeframe: Intervalo de tiempo ('1m', '5m', '15m', '1h', '4h', '1d')
            risk_reward_min: Ratio mínimo de riesgo/recompensa para considerar una operación
            
        Returns:
            Dict[str, Any]: Evaluación de decisión de trading
        """
        # Verificar suficientes datos
        if len(df) < 100:
            return {"error": "Insuficientes datos para análisis"}
        
        current_price = df['close'].iloc[-1]
        
        # 1. Calcular amplitudes de precio típicas
        amplitudes = self.calculate_price_amplitudes(df)
        
        # 2. Calcular probabilidades de reversión
        reversal_probs = self.calculate_reversal_probability(df)
        
        # 3. Identificar patrones gráficos
        patterns = self.identify_chart_patterns(df)
        
        # 4. Analizar comportamiento reciente de precios
        price_behavior = self.analyze_recent_price_behavior(df)
        
        # 5. Recopilar señales de diferentes indicadores técnicos
        # RSI
        rsi_signal = 1 if price_behavior['rsi'] < 30 else (-1 if price_behavior['rsi'] > 70 else 0)
        
        # Tendencia
        trend_signal = 1 if price_behavior['trend_analysis']['norm_slope_pct'] > 0.1 else \
                      (-1 if price_behavior['trend_analysis']['norm_slope_pct'] < -0.1 else 0)
        
        # Volatilidad
        is_high_volatility = price_behavior['volatility']['relative_volatility'] > 1.5
        is_low_volatility = price_behavior['volatility']['relative_volatility'] < 0.5
        
        # Señal basada en patrones gráficos
        pattern_signals = []
        for pattern in patterns:
            if pattern['pattern_type'] == PatternType.REVERSAL_TOP:
                pattern_signals.append(-1)  # Señal de venta
            elif pattern['pattern_type'] == PatternType.REVERSAL_BOTTOM:
                pattern_signals.append(1)   # Señal de compra
            elif pattern['pattern_type'] == PatternType.CONTINUATION:
                # La dirección depende del patrón específico
                if pattern.get('direction') == 'bullish':
                    pattern_signals.append(1)
                elif pattern.get('direction') == 'bearish':
                    pattern_signals.append(-1)
        
        # Promedio de señales de patrones
        pattern_signal = np.mean(pattern_signals) if pattern_signals else 0
        
        # 6. Combinar señales para decisión final
        # Calculamos un "score" combinado
        combined_score = (
            rsi_signal * 0.3 +         # RSI tiene un peso de 30%
            trend_signal * 0.4 +       # Tendencia tiene un peso de 40%
            pattern_signal * 0.3        # Patrones tienen un peso de 30%
        )
        
        # Determinar tipo de posición recomendada
        position_type = "LONG" if combined_score > 0.2 else ("SHORT" if combined_score < -0.2 else "NEUTRAL")
        
        # 7. Evaluar niveles de riesgo/recompensa
        risk = 0
        reward = 0
        
        if position_type == "LONG":
            # Para posición larga, el riesgo es la distancia a soporte más cercano
            risk = (current_price - price_behavior['key_levels']['support_1']) / current_price
            # La recompensa es la distancia a la próxima resistencia
            reward = (price_behavior['key_levels']['resistance_1'] - current_price) / current_price
        
        elif position_type == "SHORT":
            # Para posición corta, el riesgo es la distancia a resistencia más cercana
            risk = (price_behavior['key_levels']['resistance_1'] - current_price) / current_price
            # La recompensa es la distancia al próximo soporte
            reward = (current_price - price_behavior['key_levels']['support_1']) / current_price
        
        # Calcular ratio riesgo/recompensa
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Evaluar conveniencia de operar
        if position_type != "NEUTRAL" and risk_reward_ratio > risk_reward_min:
            should_trade = True
        else:
            should_trade = False
        
        # 8. Calcular horizonte temporal recomendado basado en volatilidad y amplitudes
        if is_high_volatility:
            recommended_timeframe = "SHORT"  # Horizonte corto para alta volatilidad
        elif is_low_volatility:
            recommended_timeframe = "MEDIUM"  # Horizonte medio para baja volatilidad
        else:
            recommended_timeframe = "MEDIUM"  # Horizonte medio por defecto
        
        # Recomendación basada en volatilidad y patrón
        if position_type == "NEUTRAL":
            if is_high_volatility:
                recommendation = "ESPERAR - Mercado volátil sin dirección clara"
            else:
                recommendation = "ESPERAR - Mercado sin tendencia definida"
        else:
            if should_trade:
                direction = "COMPRAR" if position_type == "LONG" else "VENDER"
                if risk_reward_ratio > 2:
                    strength = "FUERTE"
                else:
                    strength = "MODERADA"
                
                recommendation = f"{direction} - Señal {strength} (R/R: {risk_reward_ratio:.2f})"
            else:
                recommendation = "ESPERAR - Relación riesgo/recompensa insuficiente"
        
        return {
            'current_price': current_price,
            'timeframe': timeframe,
            'position_type': position_type,
            'combined_score': combined_score,
            'risk_reward_ratio': risk_reward_ratio,
            'should_trade': should_trade,
            'recommended_timeframe': recommended_timeframe,
            'recommendation': recommendation,
            'signals': {
                'rsi': rsi_signal,
                'trend': trend_signal,
                'pattern': pattern_signal
            },
            'reversal_probability': reversal_probs.get('current_reversal_prob', 0),
            'detected_patterns': [p['type'] for p in patterns],
            'price_behavior': {
                'trend_strength': price_behavior['trend_analysis']['trend_strength'],
                'rsi': price_behavior['rsi'],
                'volatility': price_behavior['volatility']['relative_volatility']
            },
            'key_levels': price_behavior['key_levels'],
            'typical_amplitude': amplitudes.get(timeframe, 0)
        }

def evaluate_trading_opportunity(symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
    """
    Función de conveniencia para evaluar oportunidades de trading
    
    Args:
        symbol: Par de trading (ej: SOL-USDT)
        timeframe: Intervalo de tiempo
        
    Returns:
        Dict[str, Any]: Evaluación completa de la oportunidad
    """
    from data_management.market_data import get_market_data, update_market_data
    
    # Obtener datos
    df = get_market_data(symbol, timeframe)
    
    if df is None or df.empty:
        df = update_market_data(symbol, timeframe)
    
    if df is None or df.empty:
        return {"error": f"No se pueden obtener datos para {symbol} en {timeframe}"}
    
    # Crear analizador y evaluar
    analyzer = PatternAnalyzer()
    evaluation = analyzer.evaluate_trading_decision(df, timeframe)
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": datetime.now().isoformat(),
        "evaluation": evaluation
    }

def analyze_multiple_timeframes(symbol: str, timeframes: List[str] = ['5m', '15m', '1h', '4h', '1d']) -> Dict[str, Any]:
    """
    Analiza un símbolo en múltiples marcos temporales
    
    Args:
        symbol: Par de trading (ej: SOL-USDT)
        timeframes: Lista de marcos temporales a analizar
        
    Returns:
        Dict[str, Any]: Análisis multi-timeframe
    """
    results = {}
    
    for tf in timeframes:
        results[tf] = evaluate_trading_opportunity(symbol, tf)
    
    # Análisis integrado de múltiples timeframes
    long_signals = 0
    short_signals = 0
    neutral_signals = 0
    
    # Contar señales de diferentes timeframes (con más peso en los más largos)
    weights = {
        '1m': 0.1,
        '5m': 0.2,
        '15m': 0.3,
        '1h': 0.6,
        '4h': 0.8,
        '1d': 1.0
    }
    
    weighted_score = 0
    total_weight = 0
    
    for tf, result in results.items():
        if 'error' not in result:
            position_type = result['evaluation']['position_type']
            weight = weights.get(tf, 0.5)
            
            if position_type == "LONG":
                long_signals += 1
                weighted_score += weight
            elif position_type == "SHORT":
                short_signals += 1
                weighted_score -= weight
            else:
                neutral_signals += 1
            
            total_weight += weight
    
    # Calcular señal multi-timeframe
    if total_weight > 0:
        multi_tf_score = weighted_score / total_weight
    else:
        multi_tf_score = 0
    
    if multi_tf_score > 0.3:
        multi_tf_signal = "STRONG_BUY"
    elif multi_tf_score > 0.1:
        multi_tf_signal = "BUY"
    elif multi_tf_score < -0.3:
        multi_tf_signal = "STRONG_SELL"
    elif multi_tf_score < -0.1:
        multi_tf_signal = "SELL"
    else:
        multi_tf_signal = "NEUTRAL"
    
    # Determinar dirección dominante de tendencia
    if long_signals > short_signals and long_signals > neutral_signals:
        dominant_trend = "UPTREND"
    elif short_signals > long_signals and short_signals > neutral_signals:
        dominant_trend = "DOWNTREND"
    else:
        dominant_trend = "SIDEWAYS"
    
    return {
        "symbol": symbol,
        "timeframes_analyzed": timeframes,
        "timestamp": datetime.now().isoformat(),
        "individual_results": results,
        "multi_timeframe_analysis": {
            "long_signals": long_signals,
            "short_signals": short_signals,
            "neutral_signals": neutral_signals,
            "weighted_score": multi_tf_score,
            "signal": multi_tf_signal,
            "dominant_trend": dominant_trend
        }
    }

def demo_pattern_analysis(symbol: str = "SOL-USDT", timeframe: str = "1h") -> None:
    """
    Función para demostrar el análisis de patrones con un ejemplo
    
    Args:
        symbol: Par de trading
        timeframe: Intervalo de tiempo
    """
    from data_management.market_data import get_market_data, update_market_data
    
    print(f"\n===== ANÁLISIS DE PATRONES PARA {symbol} ({timeframe}) =====")
    
    # Obtener datos
    df = get_market_data(symbol, timeframe)
    
    if df is None or df.empty:
        print("Obteniendo datos actualizados...")
        df = update_market_data(symbol, timeframe)
    
    if df is None or df.empty:
        print(f"No se pueden obtener datos para {symbol} en {timeframe}")
        return
    
    # Crear analizador
    analyzer = PatternAnalyzer()
    
    # 1. Calcular amplitudes
    amplitudes = analyzer.calculate_price_amplitudes(df)
    print("\n--- AMPLITUDES DE PRECIO ---")
    for interval, amplitude in amplitudes.items():
        print(f"{interval}: ${amplitude:.2f}")
    
    # 2. Calcular volatilidad
    volatility = analyzer.calculate_volatility_metrics(df)
    print("\n--- MÉTRICAS DE VOLATILIDAD ---")
    print(f"Volatilidad actual: {volatility['current_volatility']*100:.2f}%")
    print(f"Volatilidad anualizada: {volatility['annualized_volatility']*100:.2f}%")
    print(f"Volatilidad relativa: {volatility['relative_volatility']:.2f}x")
    print(f"ATR: ${volatility['atr']:.2f} ({volatility['atr_pct']:.2f}%)")
    
    # 3. Probabilidades de reversión
    reversal = analyzer.calculate_reversal_probability(df)
    print("\n--- PROBABILIDADES DE REVERSIÓN ---")
    print(f"Éxito en techos: {reversal['peak_success_rate']*100:.2f}%")
    print(f"Éxito en suelos: {reversal['valley_success_rate']*100:.2f}%")
    print(f"Patrón actual: {reversal['current_pattern']}")
    print(f"Probabilidad de reversión: {reversal['current_reversal_prob']*100:.2f}%")
    
    # 4. Patrones gráficos
    patterns = analyzer.identify_chart_patterns(df)
    print("\n--- PATRONES GRÁFICOS DETECTADOS ---")
    if patterns:
        for i, pattern in enumerate(patterns, 1):
            print(f"{i}. {pattern['type']} - Fortaleza: {pattern['strength'].value}")
    else:
        print("No se detectaron patrones gráficos claros")
    
    # 5. Comportamiento de precios
    behavior = analyzer.analyze_recent_price_behavior(df)
    print("\n--- COMPORTAMIENTO DE PRECIOS ---")
    print(f"Retorno medio: {behavior['mean_return']*100:.4f}%")
    print(f"Ratio alcista/bajista: {behavior['bar_distribution']['bullish_ratio']:.2f}/{behavior['bar_distribution']['bearish_ratio']:.2f}")
    print(f"Tendencia (pendiente): {behavior['trend_analysis']['norm_slope_pct']:.2f}%/periodo")
    print(f"Fuerza de tendencia (R²): {behavior['trend_analysis']['r_squared']:.2f}")
    print(f"RSI actual: {behavior['rsi']:.2f}")
    
    # 6. Evaluación final
    evaluation = analyzer.evaluate_trading_decision(df, timeframe)
    
    print("\n===== EVALUACIÓN DE TRADING =====")
    print(f"Precio actual: ${evaluation['current_price']:.2f}")
    print(f"Tipo de posición recomendada: {evaluation['position_type']}")
    print(f"Score combinado: {evaluation['combined_score']:.2f}")
    print(f"Ratio riesgo/recompensa: {evaluation['risk_reward_ratio']:.2f}")
    print(f"¿Se recomienda operar? {'SÍ' if evaluation['should_trade'] else 'NO'}")
    print(f"Horizonte temporal recomendado: {evaluation['recommended_timeframe']}")
    print(f"Recomendación: {evaluation['recommendation']}")
    
    print("\n--- NIVELES CLAVE ---")
    print(f"Resistencia 2: ${evaluation['key_levels']['resistance_2']:.2f}")
    print(f"Resistencia 1: ${evaluation['key_levels']['resistance_1']:.2f}")
    print(f"Precio actual: ${evaluation['current_price']:.2f}")
    print(f"Soporte 1: ${evaluation['key_levels']['support_1']:.2f}")
    print(f"Soporte 2: ${evaluation['key_levels']['support_2']:.2f}")
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "amplitudes": amplitudes,
        "volatility": volatility,
        "reversal": reversal,
        "patterns": patterns,
        "behavior": behavior,
        "evaluation": evaluation
    }

def analyze_trading_fees_for_strategy(symbol: str, style: str, is_short: bool = False) -> Dict[str, Any]:
    """
    Analiza las comisiones para diferentes estilos de trading
    
    Args:
        symbol: Par de trading (ej: SOL-USDT)
        style: Estilo de trading ("scalping", "day_trading", "swing_trading")
        is_short: Si se evalúan posiciones cortas
        
    Returns:
        Dict[str, Any]: Análisis de comisiones
    """
    from risk_management.fee_calculator import estimate_strategy_costs
    from data_management.market_data import get_market_data
    
    # Obtener precio actual
    df = get_market_data(symbol, "1h")
    current_price = df['close'].iloc[-1] if df is not None and not df.empty else 150.0
    
    # Configurar parámetros según estilo de trading
    if style == "scalping":
        params = {
            "trade_type": "futures",
            "avg_trades_per_day": 10,
            "avg_position_size": 1.0,  # 1 SOL
            "avg_price": current_price,
            "avg_hours_held": 0.5,  # 30 minutos
            "leverage": 5.0,
            "short_ratio": 0.5 if is_short else 0.0,
            "taker_ratio": 0.8,  # 80% órdenes tipo market
            "days": 30
        }
    elif style == "day_trading":
        params = {
            "trade_type": "futures",
            "avg_trades_per_day": 3,
            "avg_position_size": 2.0,  # 2 SOL
            "avg_price": current_price,
            "avg_hours_held": 4.0,  # 4 horas
            "leverage": 3.0,
            "short_ratio": 0.5 if is_short else 0.0,
            "taker_ratio": 0.5,  # 50% órdenes tipo market
            "days": 30
        }
    elif style == "swing_trading":
        params = {
            "trade_type": "margin",
            "avg_trades_per_day": 0.5,  # 1 operación cada 2 días
            "avg_position_size": 5.0,  # 5 SOL
            "avg_price": current_price,
            "avg_hours_held": 48.0,  # 2 días
            "leverage": 2.0,
            "short_ratio": 0.5 if is_short else 0.0,
            "taker_ratio": 0.3,  # 30% órdenes tipo market
            "days": 30
        }
    else:
        # Parámetros predeterminados
        params = {
            "trade_type": "futures",
            "avg_trades_per_day": 3,
            "avg_position_size": 2.0,
            "avg_price": current_price,
            "avg_hours_held": 8.0,
            "leverage": 3.0,
            "short_ratio": 0.5 if is_short else 0.0,
            "taker_ratio": 0.5,
            "days": 30
        }
    
    # Calcular costos
    costs = estimate_strategy_costs(**params)
    
    return {
        "symbol": symbol,
        "style": style,
        "is_short_enabled": is_short,
        "parameters": params,
        "costs": costs
    }