"""
Sistema de reconocimiento de patrones de velas y análisis de order flow

Este módulo implementa:
1. Detección de patrones clásicos de velas
2. Análisis de niveles clave mediante fractales
3. Análisis básico de order flow (imbalance y delta volumen)
4. Sistema de puntuación para evaluación continua de precisión
"""

import numpy as np
import pandas as pd
import logging
import json
import os
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from enum import Enum

# Configuración de logging
logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Tipos de patrones de velas soportados"""
    DOJI = "doji"                            # Indecisión
    HAMMER = "hammer"                        # Martillo (potencial reversión alcista)
    INVERTED_HAMMER = "inverted_hammer"      # Martillo invertido
    ENGULFING_BULLISH = "engulfing_bullish"  # Envolvente alcista
    ENGULFING_BEARISH = "engulfing_bearish"  # Envolvente bajista
    MORNING_STAR = "morning_star"            # Estrella de la mañana (reversión alcista)
    EVENING_STAR = "evening_star"            # Estrella de la tarde (reversión bajista)
    THREE_WHITE_SOLDIERS = "three_white_soldiers"  # Tres soldados blancos (continuidad alcista)
    THREE_BLACK_CROWS = "three_black_crows"  # Tres cuervos negros (continuidad bajista)
    SHOOTING_STAR = "shooting_star"          # Estrella fugaz (reversión bajista)
    HARAMI_BULLISH = "harami_bullish"        # Harami alcista (potencial reversión)
    HARAMI_BEARISH = "harami_bearish"        # Harami bajista
    MARUBOZU_BULLISH = "marubozu_bullish"    # Marubozu alcista (fuerte presión compradora)
    MARUBOZU_BEARISH = "marubozu_bearish"    # Marubozu bajista (fuerte presión vendedora)
    TWEEZER_TOP = "tweezer_top"              # Pinzas superior (reversión bajista)
    TWEEZER_BOTTOM = "tweezer_bottom"        # Pinzas inferior (reversión alcista)
    DARK_CLOUD_COVER = "dark_cloud_cover"    # Nube oscura (reversión bajista)
    PIERCING_LINE = "piercing_line"          # Línea penetrante (reversión alcista)
    SPINNING_TOP = "spinning_top"            # Peonza (indecisión)
    HANGING_MAN = "hanging_man"              # Hombre colgado (reversión bajista)

class PatternStrength(Enum):
    """Fuerza relativa de un patrón detectado"""
    WEAK = 1         # Patrón débil (baja confiabilidad)
    MODERATE = 2     # Patrón moderado
    STRONG = 3       # Patrón fuerte
    VERY_STRONG = 4  # Patrón muy fuerte (alta confiabilidad)

class OrderFlowSignal(Enum):
    """Señales de order flow"""
    STRONG_BID = "strong_bid"                # Fuerte presión compradora
    STRONG_ASK = "strong_ask"                # Fuerte presión vendedora
    ABSORPTION_BUY = "absorption_buy"        # Absorción de ventas (acumulación)
    ABSORPTION_SELL = "absorption_sell"      # Absorción de compras (distribución)
    DELTA_POSITIVE = "delta_positive"        # Delta de volumen positivo (bias comprador)
    DELTA_NEGATIVE = "delta_negative"        # Delta de volumen negativo (bias vendedor)
    IMBALANCE_BUY = "imbalance_buy"          # Desequilibrio comprador
    IMBALANCE_SELL = "imbalance_sell"        # Desequilibrio vendedor
    STOPS_HUNT_UP = "stops_hunt_up"          # Caza de stops hacia arriba
    STOPS_HUNT_DOWN = "stops_hunt_down"      # Caza de stops hacia abajo

class MarketCondition(Enum):
    """Condiciones de mercado para contextualizar patrones"""
    STRONG_UPTREND = "strong_uptrend"
    MODERATE_UPTREND = "moderate_uptrend"
    LATERAL_LOW_VOL = "lateral_low_vol"
    LATERAL_HIGH_VOL = "lateral_high_vol"
    MODERATE_DOWNTREND = "moderate_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    EXTREME_VOLATILITY = "extreme_volatility"

class PatternRecognition:
    """Sistema de reconocimiento de patrones de velas"""
    
    def __init__(self, config: Dict[str, Any] = None, data_file: str = "pattern_data.json"):
        """
        Inicializa el sistema de reconocimiento de patrones
        
        Args:
            config: Configuración del sistema
            data_file: Archivo para guardar/cargar datos de precisión
        """
        self.config = config or {
            # Umbral para considerar una vela como doji
            'doji_threshold': 0.05,  # Diferencia máxima entre apertura y cierre (%)
            
            # Umbral para sombras (wicks)
            'shadow_threshold': 0.5,  # Tamaño mínimo de sombra vs. cuerpo (ratio)
            
            # Umbral para envolventes
            'engulfing_threshold': 1.05,  # Factor de envolvente (105%)
            
            # Tolerancia para considerar niveles similares
            'price_tolerance': 0.001,  # 0.1% de tolerancia
            
            # Número de velas a considerar para patrones
            'pattern_lookback': 5,
            
            # Activar análisis de order flow
            'enable_order_flow': True,
            
            # Umbral para considerar volumen significativo
            'volume_threshold': 1.5,  # 1.5x el volumen promedio
            
            # Umbral para detectar imbalances
            'imbalance_threshold': 0.7,  # 70% de diferencia bid/ask
            
            # Período para fractales
            'fractal_period': 5,
            
            # Archivo de datos
            'data_file': data_file
        }
        
        # Estadísticas de precisión por patrón
        self.pattern_stats = {}
        
        # Cargar datos previos
        self._load_stats()
        
        logger.info("Sistema de reconocimiento de patrones inicializado")
    
    def _load_stats(self) -> None:
        """Carga estadísticas de precisión desde archivo"""
        if os.path.exists(self.config['data_file']):
            try:
                with open(self.config['data_file'], 'r') as f:
                    data = json.load(f)
                
                self.pattern_stats = data.get('pattern_stats', {})
                
                logger.info(f"Estadísticas de patrones cargadas desde {self.config['data_file']}")
            except Exception as e:
                logger.error(f"Error al cargar estadísticas de patrones: {e}")
                # Inicializar estadísticas por defecto
                self._initialize_pattern_stats()
        else:
            # Inicializar estadísticas por defecto
            self._initialize_pattern_stats()
    
    def _initialize_pattern_stats(self) -> None:
        """Inicializa estadísticas de precisión por defecto"""
        # Crear entrada para cada tipo de patrón
        for pattern_type in PatternType:
            self.pattern_stats[pattern_type.value] = {
                'detected': 0,           # Número total de detecciones
                'successful': 0,         # Número de detecciones exitosas
                'failed': 0,             # Número de detecciones fallidas
                'accuracy': 0.0,         # Precisión (0.0 - 1.0)
                'avg_profit': 0.0,       # Ganancia promedio
                'avg_loss': 0.0,         # Pérdida promedio
                'win_rate': 0.0,         # Tasa de acierto
                'profit_factor': 0.0,    # Factor de ganancia/pérdida
                'by_condition': {}       # Precisión por condición de mercado
            }
        
        logger.info("Estadísticas de patrones inicializadas")
    
    def _save_stats(self) -> None:
        """Guarda estadísticas de precisión en archivo"""
        try:
            data = {
                'pattern_stats': self.pattern_stats,
                'last_update': datetime.now().isoformat()
            }
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.config['data_file']) or '.', exist_ok=True)
            
            with open(self.config['data_file'], 'w') as f:
                json.dump(data, f, indent=4)
            
            logger.debug(f"Estadísticas de patrones guardadas en {self.config['data_file']}")
        except Exception as e:
            logger.error(f"Error al guardar estadísticas de patrones: {e}")
    
    def update_pattern_performance(self, pattern: PatternType, success: bool, 
                                  profit: float, condition: MarketCondition) -> None:
        """
        Actualiza estadísticas de rendimiento de un patrón
        
        Args:
            pattern: Tipo de patrón
            success: Si la predicción fue exitosa
            profit: Ganancia/pérdida obtenida
            condition: Condición del mercado
        """
        pattern_value = pattern.value
        
        # Asegurarse de que existe la entrada
        if pattern_value not in self.pattern_stats:
            self.pattern_stats[pattern_value] = {
                'detected': 0,
                'successful': 0,
                'failed': 0,
                'accuracy': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'by_condition': {}
            }
        
        # Actualizar contadores
        self.pattern_stats[pattern_value]['detected'] += 1
        
        if success:
            self.pattern_stats[pattern_value]['successful'] += 1
        else:
            self.pattern_stats[pattern_value]['failed'] += 1
        
        # Actualizar ganancia/pérdida
        if profit > 0:
            # Actualizar ganancia promedio
            prev_avg = self.pattern_stats[pattern_value]['avg_profit']
            prev_count = self.pattern_stats[pattern_value]['successful']
            new_avg = ((prev_avg * (prev_count - 1)) + profit) / prev_count if prev_count > 0 else profit
            self.pattern_stats[pattern_value]['avg_profit'] = new_avg
        else:
            # Actualizar pérdida promedio
            prev_avg = self.pattern_stats[pattern_value]['avg_loss']
            prev_count = self.pattern_stats[pattern_value]['failed']
            new_avg = ((prev_avg * (prev_count - 1)) + abs(profit)) / prev_count if prev_count > 0 else abs(profit)
            self.pattern_stats[pattern_value]['avg_loss'] = new_avg
        
        # Actualizar tasa de acierto
        total = self.pattern_stats[pattern_value]['detected']
        successful = self.pattern_stats[pattern_value]['successful']
        self.pattern_stats[pattern_value]['win_rate'] = successful / total if total > 0 else 0.0
        
        # Actualizar factor de ganancia/pérdida
        avg_profit = self.pattern_stats[pattern_value]['avg_profit']
        avg_loss = self.pattern_stats[pattern_value]['avg_loss']
        profit_factor = avg_profit / avg_loss if avg_loss > 0 else float('inf')
        self.pattern_stats[pattern_value]['profit_factor'] = profit_factor
        
        # Actualizar precisión por condición de mercado
        condition_value = condition.value
        by_condition = self.pattern_stats[pattern_value]['by_condition']
        
        if condition_value not in by_condition:
            by_condition[condition_value] = {
                'detected': 0,
                'successful': 0,
                'failed': 0,
                'win_rate': 0.0
            }
        
        by_condition[condition_value]['detected'] += 1
        
        if success:
            by_condition[condition_value]['successful'] += 1
        else:
            by_condition[condition_value]['failed'] += 1
        
        # Actualizar tasa de acierto por condición
        total = by_condition[condition_value]['detected']
        successful = by_condition[condition_value]['successful']
        by_condition[condition_value]['win_rate'] = successful / total if total > 0 else 0.0
        
        # Guardar estadísticas
        self._save_stats()
        
        logger.debug(f"Actualizadas estadísticas para patrón {pattern_value}: "
                    f"win_rate={self.pattern_stats[pattern_value]['win_rate']:.2f}, "
                    f"profit_factor={profit_factor:.2f}")
    
    def detect_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detecta patrones en un DataFrame de velas
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            List[Dict]: Lista de patrones detectados
        """
        # Verificar que el DataFrame tiene las columnas necesarias
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Columna requerida '{col}' no encontrada en DataFrame")
                return []
        
        # Lista para almacenar patrones encontrados
        patterns = []
        
        # Número mínimo de velas necesarias
        min_candles = max(5, self.config['pattern_lookback'])
        
        if len(df) < min_candles:
            logger.warning(f"Insuficientes velas para detectar patrones (mínimo {min_candles})")
            return []
        
        # Calcular algunos valores útiles
        df = df.copy()
        df['body_size'] = abs(df['close'] - df['open'])
        df['is_bullish'] = df['close'] > df['open']
        df['body_pct'] = df['body_size'] / ((df['high'] + df['low']) / 2)
        df['upper_shadow'] = df.apply(lambda x: x['high'] - max(x['open'], x['close']), axis=1)
        df['lower_shadow'] = df.apply(lambda x: min(x['open'], x['close']) - x['low'], axis=1)
        df['shadow_ratio'] = (df['upper_shadow'] + df['lower_shadow']) / df['body_size'].replace(0, 0.001)
        df['avg_price'] = (df['high'] + df['low'] + df['open'] + df['close']) / 4
        
        # Detectar patrones de una sola vela
        patterns.extend(self._detect_single_candle_patterns(df))
        
        # Detectar patrones de dos velas
        patterns.extend(self._detect_two_candle_patterns(df))
        
        # Detectar patrones de tres velas
        patterns.extend(self._detect_three_candle_patterns(df))
        
        # Añadir análisis de order flow si está habilitado
        if self.config['enable_order_flow']:
            patterns.extend(self._analyze_order_flow(df))
        
        # Detectar niveles clave mediante fractales
        key_levels = self._detect_fractals(df)
        
        # Añadir información sobre proximidad a niveles clave
        for pattern in patterns:
            close_level = self._find_closest_level(pattern['price'], key_levels)
            if close_level:
                pattern['key_level'] = close_level
        
        return patterns
    
    def _detect_single_candle_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detecta patrones de una sola vela
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            List[Dict]: Lista de patrones detectados
        """
        patterns = []
        
        # Umbral para doji
        doji_threshold = self.config['doji_threshold']
        shadow_threshold = self.config['shadow_threshold']
        
        for i in range(len(df) - 1, max(0, len(df) - 10), -1):
            row = df.iloc[i]
            
            # Calcular la media de los tamaños de cuerpo de las últimas velas
            avg_body = df['body_size'].iloc[max(0, i-5):i+1].mean()
            
            # Precio promedio de la vela
            price = row['close']
            
            # DOJI
            # Cuerpo muy pequeño, sombras pueden variar
            body_pct = row['body_size'] / (price * doji_threshold)
            if body_pct < 1.0:
                patterns.append({
                    'type': PatternType.DOJI.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.MODERATE.value,
                    'direction': 'neutral',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
            
            # MARTILLO (HAMMER)
            # Cuerpo pequeño en la parte superior, sombra inferior larga
            if (row['lower_shadow'] > 2 * row['body_size'] and
                row['upper_shadow'] < 0.3 * row['body_size'] and
                row['body_size'] < avg_body):
                
                strength = PatternStrength.STRONG.value
                
                # En una tendencia bajista es más significativo
                if i >= 3 and df['close'].iloc[i-3:i].mean() > price:
                    strength = PatternStrength.VERY_STRONG.value
                
                patterns.append({
                    'type': PatternType.HAMMER.value,
                    'position': i,
                    'price': price,
                    'strength': strength,
                    'direction': 'bullish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
            
            # MARTILLO INVERTIDO (INVERTED HAMMER)
            # Cuerpo pequeño en la parte inferior, sombra superior larga
            if (row['upper_shadow'] > 2 * row['body_size'] and
                row['lower_shadow'] < 0.3 * row['body_size'] and
                row['body_size'] < avg_body):
                
                strength = PatternStrength.MODERATE.value
                
                # En una tendencia bajista es más significativo
                if i >= 3 and df['close'].iloc[i-3:i].mean() > price:
                    strength = PatternStrength.STRONG.value
                
                patterns.append({
                    'type': PatternType.INVERTED_HAMMER.value,
                    'position': i,
                    'price': price,
                    'strength': strength,
                    'direction': 'bullish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
            
            # MARUBOZU ALCISTA
            # Cuerpo grande alcista sin sombras (o muy pequeñas)
            if (row['is_bullish'] and 
                row['body_size'] > 1.5 * avg_body and
                row['upper_shadow'] < 0.1 * row['body_size'] and
                row['lower_shadow'] < 0.1 * row['body_size']):
                
                patterns.append({
                    'type': PatternType.MARUBOZU_BULLISH.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.VERY_STRONG.value,
                    'direction': 'bullish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
            
            # MARUBOZU BAJISTA
            # Cuerpo grande bajista sin sombras (o muy pequeñas)
            if (not row['is_bullish'] and 
                row['body_size'] > 1.5 * avg_body and
                row['upper_shadow'] < 0.1 * row['body_size'] and
                row['lower_shadow'] < 0.1 * row['body_size']):
                
                patterns.append({
                    'type': PatternType.MARUBOZU_BEARISH.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.VERY_STRONG.value,
                    'direction': 'bearish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
            
            # SPINNING TOP (PEONZA)
            # Cuerpo pequeño, sombras largas arriba y abajo
            if (row['body_size'] < 0.5 * avg_body and
                row['upper_shadow'] > row['body_size'] and
                row['lower_shadow'] > row['body_size']):
                
                patterns.append({
                    'type': PatternType.SPINNING_TOP.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.WEAK.value,
                    'direction': 'neutral',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
            
            # HANGING MAN (HOMBRE COLGADO)
            # Similar al martillo pero en tendencia alcista
            if (row['lower_shadow'] > 2 * row['body_size'] and
                row['upper_shadow'] < 0.3 * row['body_size'] and
                row['body_size'] < avg_body):
                
                # Solo es hombre colgado en tendencia alcista
                if i >= 3 and df['close'].iloc[i-3:i].mean() < price:
                    patterns.append({
                        'type': PatternType.HANGING_MAN.value,
                        'position': i,
                        'price': price,
                        'strength': PatternStrength.STRONG.value,
                        'direction': 'bearish',
                        'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                    })
                    continue
            
            # SHOOTING STAR (ESTRELLA FUGAZ)
            # Similar al martillo invertido pero en tendencia alcista
            if (row['upper_shadow'] > 2 * row['body_size'] and
                row['lower_shadow'] < 0.3 * row['body_size'] and
                row['body_size'] < avg_body):
                
                # Solo es estrella fugaz en tendencia alcista
                if i >= 3 and df['close'].iloc[i-3:i].mean() < price:
                    patterns.append({
                        'type': PatternType.SHOOTING_STAR.value,
                        'position': i,
                        'price': price,
                        'strength': PatternStrength.STRONG.value,
                        'direction': 'bearish',
                        'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                    })
                    continue
        
        return patterns
    
    def _detect_two_candle_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detecta patrones de dos velas
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            List[Dict]: Lista de patrones detectados
        """
        patterns = []
        
        # Umbral para patrones de envolvente
        engulfing_threshold = self.config['engulfing_threshold']
        
        for i in range(len(df) - 1, max(1, len(df) - 10), -1):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Precio promedio de la última vela
            price = current['close']
            
            # ENGULFING ALCISTA
            # Vela actual alcista envuelve vela anterior bajista
            if (current['is_bullish'] and
                not previous['is_bullish'] and
                current['close'] > previous['open'] * engulfing_threshold and
                current['open'] < previous['close'] * engulfing_threshold):
                
                patterns.append({
                    'type': PatternType.ENGULFING_BULLISH.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.STRONG.value,
                    'direction': 'bullish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
            
            # ENGULFING BAJISTA
            # Vela actual bajista envuelve vela anterior alcista
            if (not current['is_bullish'] and
                previous['is_bullish'] and
                current['open'] > previous['close'] * engulfing_threshold and
                current['close'] < previous['open'] * engulfing_threshold):
                
                patterns.append({
                    'type': PatternType.ENGULFING_BEARISH.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.STRONG.value,
                    'direction': 'bearish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
            
            # HARAMI ALCISTA
            # Vela pequeña alcista dentro del rango de la anterior bajista
            if (current['is_bullish'] and
                not previous['is_bullish'] and
                current['high'] < previous['open'] and
                current['low'] > previous['close'] and
                current['body_size'] < previous['body_size'] * 0.6):
                
                patterns.append({
                    'type': PatternType.HARAMI_BULLISH.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.MODERATE.value,
                    'direction': 'bullish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
            
            # HARAMI BAJISTA
            # Vela pequeña bajista dentro del rango de la anterior alcista
            if (not current['is_bullish'] and
                previous['is_bullish'] and
                current['high'] < previous['close'] and
                current['low'] > previous['open'] and
                current['body_size'] < previous['body_size'] * 0.6):
                
                patterns.append({
                    'type': PatternType.HARAMI_BEARISH.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.MODERATE.value,
                    'direction': 'bearish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
            
            # TWEEZER TOP (PINZAS SUPERIOR)
            # Dos velas con máximos similares, primera alcista, segunda bajista
            if (not current['is_bullish'] and
                previous['is_bullish'] and
                abs(current['high'] - previous['high']) / previous['high'] < 0.003):
                
                patterns.append({
                    'type': PatternType.TWEEZER_TOP.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.MODERATE.value,
                    'direction': 'bearish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
            
            # TWEEZER BOTTOM (PINZAS INFERIOR)
            # Dos velas con mínimos similares, primera bajista, segunda alcista
            if (current['is_bullish'] and
                not previous['is_bullish'] and
                abs(current['low'] - previous['low']) / previous['low'] < 0.003):
                
                patterns.append({
                    'type': PatternType.TWEEZER_BOTTOM.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.MODERATE.value,
                    'direction': 'bullish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
            
            # PIERCING LINE (LÍNEA PENETRANTE)
            # Vela bajista seguida de vela alcista que cierra por encima del punto medio
            if (current['is_bullish'] and
                not previous['is_bullish'] and
                current['open'] < previous['close'] and
                current['close'] > (previous['open'] + previous['close']) / 2):
                
                patterns.append({
                    'type': PatternType.PIERCING_LINE.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.STRONG.value,
                    'direction': 'bullish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
            
            # DARK CLOUD COVER (NUBE OSCURA)
            # Vela alcista seguida de vela bajista que cierra por debajo del punto medio
            if (not current['is_bullish'] and
                previous['is_bullish'] and
                current['open'] > previous['close'] and
                current['close'] < (previous['open'] + previous['close']) / 2):
                
                patterns.append({
                    'type': PatternType.DARK_CLOUD_COVER.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.STRONG.value,
                    'direction': 'bearish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
        
        return patterns
    
    def _detect_three_candle_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detecta patrones de tres velas
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            List[Dict]: Lista de patrones detectados
        """
        patterns = []
        
        for i in range(len(df) - 1, max(2, len(df) - 10), -1):
            c1 = df.iloc[i-2]  # Primera vela (más antigua)
            c2 = df.iloc[i-1]  # Segunda vela
            c3 = df.iloc[i]    # Tercera vela (más reciente)
            
            # Precio promedio de la última vela
            price = c3['close']
            
            # MORNING STAR (ESTRELLA DE LA MAÑANA)
            # Vela bajista, vela pequeña, vela alcista
            if (not c1['is_bullish'] and
                c1['body_size'] > c2['body_size'] * 2 and
                c3['is_bullish'] and
                c3['body_size'] > c2['body_size'] * 2 and
                c2['body_size'] / ((c2['high'] + c2['low']) / 2) < 0.01 and
                c3['close'] > (c1['open'] + c1['close']) / 2):
                
                patterns.append({
                    'type': PatternType.MORNING_STAR.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.VERY_STRONG.value,
                    'direction': 'bullish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
            
            # EVENING STAR (ESTRELLA DE LA TARDE)
            # Vela alcista, vela pequeña, vela bajista
            if (c1['is_bullish'] and
                c1['body_size'] > c2['body_size'] * 2 and
                not c3['is_bullish'] and
                c3['body_size'] > c2['body_size'] * 2 and
                c2['body_size'] / ((c2['high'] + c2['low']) / 2) < 0.01 and
                c3['close'] < (c1['open'] + c1['close']) / 2):
                
                patterns.append({
                    'type': PatternType.EVENING_STAR.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.VERY_STRONG.value,
                    'direction': 'bearish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
            
            # THREE WHITE SOLDIERS (TRES SOLDADOS BLANCOS)
            # Tres velas alcistas consecutivas, cada una cerrando más alta
            if (c1['is_bullish'] and
                c2['is_bullish'] and
                c3['is_bullish'] and
                c2['close'] > c1['close'] and
                c3['close'] > c2['close'] and
                c2['open'] > c1['open'] and
                c3['open'] > c2['open']):
                
                patterns.append({
                    'type': PatternType.THREE_WHITE_SOLDIERS.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.VERY_STRONG.value,
                    'direction': 'bullish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
            
            # THREE BLACK CROWS (TRES CUERVOS NEGROS)
            # Tres velas bajistas consecutivas, cada una cerrando más baja
            if (not c1['is_bullish'] and
                not c2['is_bullish'] and
                not c3['is_bullish'] and
                c2['close'] < c1['close'] and
                c3['close'] < c2['close'] and
                c2['open'] < c1['open'] and
                c3['open'] < c2['open']):
                
                patterns.append({
                    'type': PatternType.THREE_BLACK_CROWS.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.VERY_STRONG.value,
                    'direction': 'bearish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
                continue
        
        return patterns
    
    def _analyze_order_flow(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analiza order flow básico
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            List[Dict]: Lista de señales de order flow
        """
        signals = []
        
        # Si no hay suficientes datos, retornar lista vacía
        if len(df) < 5:
            return signals
        
        # Umbral para considerar volumen significativo
        volume_threshold = self.config['volume_threshold']
        
        # Umbral para imbalances
        imbalance_threshold = self.config['imbalance_threshold']
        
        # Calcular algunos valores útiles
        df = df.copy()
        
        # Calcular delta de volumen (aproximado)
        # En datos reales, usaríamos bid/ask real, aquí usamos apertura/cierre como proxy
        df['volume_delta'] = df.apply(
            lambda x: x['volume'] * (1 if x['close'] > x['open'] else -1), 
            axis=1
        )
        
        # Calcular volumen promedio
        avg_volume = df['volume'].rolling(window=20).mean()
        
        for i in range(len(df) - 1, max(5, len(df) - 10), -1):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Precio actual
            price = row['close']
            
            # STRONG BID (FUERTE PRESIÓN COMPRADORA)
            # Volumen alto + cierre cerca del máximo + delta positivo
            close_to_high = (row['high'] - row['close']) / (row['high'] - row['low']) < 0.2
            high_volume = row['volume'] > avg_volume.iloc[i] * volume_threshold
            
            if high_volume and close_to_high and row['volume_delta'] > 0:
                signals.append({
                    'type': OrderFlowSignal.STRONG_BID.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.STRONG.value,
                    'direction': 'bullish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
            
            # STRONG ASK (FUERTE PRESIÓN VENDEDORA)
            # Volumen alto + cierre cerca del mínimo + delta negativo
            close_to_low = (row['close'] - row['low']) / (row['high'] - row['low']) < 0.2
            
            if high_volume and close_to_low and row['volume_delta'] < 0:
                signals.append({
                    'type': OrderFlowSignal.STRONG_ASK.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.STRONG.value,
                    'direction': 'bearish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
            
            # ABSORPTION BUY (ABSORCIÓN DE VENTAS)
            # Vela anterior bajista con volumen alto, vela actual alcista
            prev_high_volume = prev_row['volume'] > avg_volume.iloc[i-1] * volume_threshold
            
            if (prev_high_volume and
                not prev_row['is_bullish'] and
                row['is_bullish'] and
                row['close'] > prev_row['close']):
                
                signals.append({
                    'type': OrderFlowSignal.ABSORPTION_BUY.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.VERY_STRONG.value,
                    'direction': 'bullish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
            
            # ABSORPTION SELL (ABSORCIÓN DE COMPRAS)
            # Vela anterior alcista con volumen alto, vela actual bajista
            if (prev_high_volume and
                prev_row['is_bullish'] and
                not row['is_bullish'] and
                row['close'] < prev_row['close']):
                
                signals.append({
                    'type': OrderFlowSignal.ABSORPTION_SELL.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.VERY_STRONG.value,
                    'direction': 'bearish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
            
            # STOPS HUNT UP (CAZA DE STOPS HACIA ARRIBA)
            # Vela con mecha superior larga que sube por encima de máximos recientes y luego baja
            last_5_high = df['high'].iloc[i-5:i].max()
            
            if (row['high'] > last_5_high and
                row['close'] < last_5_high and
                row['upper_shadow'] > 2 * row['body_size']):
                
                signals.append({
                    'type': OrderFlowSignal.STOPS_HUNT_UP.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.STRONG.value,
                    'direction': 'bearish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
            
            # STOPS HUNT DOWN (CAZA DE STOPS HACIA ABAJO)
            # Vela con mecha inferior larga que baja por debajo de mínimos recientes y luego sube
            last_5_low = df['low'].iloc[i-5:i].min()
            
            if (row['low'] < last_5_low and
                row['close'] > last_5_low and
                row['lower_shadow'] > 2 * row['body_size']):
                
                signals.append({
                    'type': OrderFlowSignal.STOPS_HUNT_DOWN.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.STRONG.value,
                    'direction': 'bullish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
            
            # DELTA POSITIVO SIGNIFICATIVO
            # Acumulación sostenida (delta de volumen positivo)
            if row['volume_delta'] > 0 and row['volume'] > avg_volume.iloc[i] * 1.2:
                signals.append({
                    'type': OrderFlowSignal.DELTA_POSITIVE.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.MODERATE.value,
                    'direction': 'bullish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
            
            # DELTA NEGATIVO SIGNIFICATIVO
            # Distribución sostenida (delta de volumen negativo)
            if row['volume_delta'] < 0 and row['volume'] > avg_volume.iloc[i] * 1.2:
                signals.append({
                    'type': OrderFlowSignal.DELTA_NEGATIVE.value,
                    'position': i,
                    'price': price,
                    'strength': PatternStrength.MODERATE.value,
                    'direction': 'bearish',
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
        
        return signals
    
    def _detect_fractals(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Detecta niveles clave usando análisis fractal
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            Dict: Diccionario con niveles clave (soporte/resistencia)
        """
        supports = []
        resistances = []
        
        # Período para fractales
        period = self.config['fractal_period']
        half_period = period // 2
        
        if len(df) < period:
            return {'supports': [], 'resistances': []}
        
        # Detectar fractales
        for i in range(half_period, len(df) - half_period):
            # Fractal alcista (mínimo local)
            is_support = True
            for j in range(1, half_period + 1):
                if df['low'].iloc[i] > df['low'].iloc[i-j] or df['low'].iloc[i] > df['low'].iloc[i+j]:
                    is_support = False
                    break
            
            if is_support:
                supports.append(df['low'].iloc[i])
            
            # Fractal bajista (máximo local)
            is_resistance = True
            for j in range(1, half_period + 1):
                if df['high'].iloc[i] < df['high'].iloc[i-j] or df['high'].iloc[i] < df['high'].iloc[i+j]:
                    is_resistance = False
                    break
            
            if is_resistance:
                resistances.append(df['high'].iloc[i])
        
        # Filtrar niveles cercanos (agrupar)
        supports = self._filter_nearby_levels(supports)
        resistances = self._filter_nearby_levels(resistances)
        
        return {'supports': supports, 'resistances': resistances}
    
    def _filter_nearby_levels(self, levels: List[float]) -> List[float]:
        """
        Filtra niveles cercanos agrupándolos
        
        Args:
            levels: Lista de niveles
            
        Returns:
            List[float]: Lista filtrada
        """
        if not levels:
            return []
        
        # Ordenar niveles
        sorted_levels = sorted(levels)
        
        # Tolerancia como porcentaje
        tolerance = self.config['price_tolerance']
        
        # Filtrar niveles cercanos
        filtered = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # Si el nivel está cerca del último filtrado, ignorarlo
            if abs(level - filtered[-1]) / filtered[-1] < tolerance:
                # Actualizar el nivel existente con el promedio
                filtered[-1] = (filtered[-1] + level) / 2
            else:
                # Añadir nuevo nivel
                filtered.append(level)
        
        return filtered
    
    def _find_closest_level(self, price: float, levels: Dict[str, List[float]]) -> Optional[Dict[str, Any]]:
        """
        Encuentra el nivel clave más cercano
        
        Args:
            price: Precio actual
            levels: Diccionario con niveles clave
            
        Returns:
            Optional[Dict]: Información del nivel más cercano o None
        """
        supports = levels['supports']
        resistances = levels['resistances']
        
        closest_support = min(supports, key=lambda x: abs(x - price)) if supports else None
        closest_resistance = min(resistances, key=lambda x: abs(x - price)) if resistances else None
        
        if closest_support is None and closest_resistance is None:
            return None
        
        if closest_support is None:
            return {
                'type': 'resistance',
                'price': closest_resistance,
                'distance': abs(closest_resistance - price) / price
            }
        
        if closest_resistance is None:
            return {
                'type': 'support',
                'price': closest_support,
                'distance': abs(closest_support - price) / price
            }
        
        # Determinar cuál está más cerca
        support_distance = abs(closest_support - price) / price
        resistance_distance = abs(closest_resistance - price) / price
        
        if support_distance <= resistance_distance:
            return {
                'type': 'support',
                'price': closest_support,
                'distance': support_distance
            }
        else:
            return {
                'type': 'resistance',
                'price': closest_resistance,
                'distance': resistance_distance
            }
    
    def get_pattern_performance(self, pattern_type: PatternType) -> Dict[str, Any]:
        """
        Obtiene estadísticas de rendimiento de un patrón
        
        Args:
            pattern_type: Tipo de patrón
            
        Returns:
            Dict: Estadísticas del patrón
        """
        pattern_value = pattern_type.value
        
        if pattern_value not in self.pattern_stats:
            return {
                'detected': 0,
                'successful': 0,
                'failed': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        return self.pattern_stats[pattern_value]
    
    def get_all_pattern_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene estadísticas de rendimiento de todos los patrones
        
        Returns:
            Dict: Estadísticas de todos los patrones
        """
        return self.pattern_stats
    
    def get_best_patterns(self, min_occurrences: int = 10) -> List[Dict[str, Any]]:
        """
        Obtiene los patrones con mejor rendimiento
        
        Args:
            min_occurrences: Número mínimo de ocurrencias para considerar
            
        Returns:
            List[Dict]: Lista de patrones con mejor rendimiento
        """
        best_patterns = []
        
        for pattern, stats in self.pattern_stats.items():
            if stats['detected'] >= min_occurrences:
                best_patterns.append({
                    'pattern': pattern,
                    'win_rate': stats['win_rate'],
                    'profit_factor': stats['profit_factor'],
                    'detected': stats['detected']
                })
        
        # Ordenar por tasa de acierto
        best_patterns.sort(key=lambda x: x['win_rate'], reverse=True)
        
        return best_patterns