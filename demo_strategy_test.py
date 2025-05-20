"""
Demostración para probar el simulador y el sistema de reconocimiento de patrones
con una estrategia simple basada en indicadores básicos.
"""

import os
import pandas as pd
import numpy as np
import ccxt
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

# Importar módulos del sistema
from simulation import TradingSimulator
from pattern_recognition import PatternRecognition, PatternType, MarketCondition
from adaptive_weighting import AdaptiveWeightingSystem, MarketCondition as AWMarketCondition
from risk_management.drawdown_monitor import DrawdownMonitor
from risk_management.position_limits import PositionSizeManager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores técnicos básicos para la estrategia
    
    Args:
        df: DataFrame con datos OHLCV
        
    Returns:
        pd.DataFrame: DataFrame con indicadores añadidos
    """
    # Crear copia para no modificar el original
    df = df.copy()
    
    # Calcular medias móviles
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Calcular bandas de Bollinger (20, 2)
    df['bb_middle'] = df['sma_20']
    bollinger_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bollinger_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bollinger_std * 2)
    
    # Calcular RSI (14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Manejar los primeros 14 periodos usando SMA
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
    # Evitar división por cero
    avg_loss = avg_loss.replace(0, 0.001)
    
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Calcular MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Calcular volatilidad (ATR simplicado)
    df['atr'] = df['high'] - df['low']
    df['atr_14'] = df['atr'].rolling(window=14).mean()
    
    # Calcular volumen relativo
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Determinar la tendencia
    df['trend'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
    
    return df

def detect_market_condition(df: pd.DataFrame) -> MarketCondition:
    """
    Detecta la condición actual del mercado
    
    Args:
        df: DataFrame con indicadores
        
    Returns:
        MarketCondition: Condición del mercado
    """
    # Asegurarse de que hay suficientes datos
    if len(df) < 50:
        return MarketCondition.LATERAL_LOW_VOL
    
    # Obtener últimas filas para análisis
    recent = df.iloc[-20:]
    
    # Calcular volatilidad
    volatility = recent['atr_14'].iloc[-1] / recent['close'].iloc[-1]
    avg_volatility = recent['atr_14'].mean() / recent['close'].mean()
    high_volatility = volatility > 1.5 * avg_volatility
    
    # Calcular tendencia
    uptrend = recent['sma_20'].iloc[-1] > recent['sma_50'].iloc[-1]
    downtrend = recent['sma_20'].iloc[-1] < recent['sma_50'].iloc[-1]
    
    # Calcular fuerza de tendencia
    trend_strength = abs(recent['sma_20'].iloc[-1] - recent['sma_50'].iloc[-1]) / recent['sma_50'].iloc[-1]
    strong_trend = trend_strength > 0.03  # 3% diferencia
    
    # Detectar volatilidad extrema
    if volatility > 3 * avg_volatility:
        return MarketCondition.EXTREME_VOLATILITY
    
    # Detectar tendencias
    if uptrend:
        if strong_trend:
            return MarketCondition.STRONG_UPTREND
        else:
            return MarketCondition.MODERATE_UPTREND
    elif downtrend:
        if strong_trend:
            return MarketCondition.STRONG_DOWNTREND
        else:
            return MarketCondition.MODERATE_DOWNTREND
    
    # Mercado lateral
    if high_volatility:
        return MarketCondition.LATERAL_HIGH_VOL
    else:
        return MarketCondition.LATERAL_LOW_VOL

def pattern_based_strategy(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Estrategia basada en patrones de velas y condiciones de mercado
    
    Args:
        df: DataFrame con datos OHLCV
        
    Returns:
        Dict: Señal de trading
    """
    # Verificar que hay suficientes datos
    if len(df) < 50:
        return {}
    
    # Calcular indicadores
    df_indicators = calculate_indicators(df)
    
    # Detectar condición del mercado
    market_condition = detect_market_condition(df_indicators)
    
    # Crear instancia de reconocimiento de patrones
    pattern_recognizer = PatternRecognition()
    
    # Detectar patrones en las últimas velas
    patterns = pattern_recognizer.detect_patterns(df.iloc[-20:])
    
    # Si no hay patrones, no hay señal
    if not patterns:
        return {}
    
    # Ordenar patrones por fuerza y recencia (los más recientes primero)
    patterns.sort(key=lambda x: (x.get('position', 0), x.get('strength', 1)), reverse=True)
    
    # Tomar el patrón más reciente y fuerte
    latest_pattern = patterns[0]
    
    # Obtener rendimiento histórico del patrón
    pattern_stats = pattern_recognizer.get_pattern_performance(PatternType(latest_pattern['type']))
    
    # Considerar crear señal solo si el patrón tiene buen rendimiento en esta condición de mercado
    win_rate = pattern_stats.get('win_rate', 0)
    
    # Verificar win rate mínimo (40% para probar, en producción usar 50% o más)
    if win_rate < 0.4:
        return {}
    
    # Verificar dirección del patrón
    pattern_direction = latest_pattern.get('direction', 'neutral')
    
    # No generar señal si el patrón es neutral
    if pattern_direction == 'neutral':
        return {}
    
    # Obtener datos de RSI y MACD para confirmación
    current_rsi = df_indicators['rsi_14'].iloc[-1]
    current_macd_hist = df_indicators['macd_hist'].iloc[-1]
    
    # Señales adicionales para confirmación
    rsi_signal = 1 if current_rsi < 30 else -1 if current_rsi > 70 else 0
    macd_signal = 1 if current_macd_hist > 0 else -1 if current_macd_hist < 0 else 0
    
    # Calcular señal combinada
    signals = {
        'pattern': 1 if pattern_direction == 'bullish' else -1 if pattern_direction == 'bearish' else 0,
        'rsi': rsi_signal,
        'macd': macd_signal
    }
    
    # Ponderación simple para diferentes condiciones de mercado
    weights = {
        MarketCondition.STRONG_UPTREND: {'pattern': 0.5, 'rsi': 0.2, 'macd': 0.3},
        MarketCondition.MODERATE_UPTREND: {'pattern': 0.4, 'rsi': 0.3, 'macd': 0.3},
        MarketCondition.LATERAL_LOW_VOL: {'pattern': 0.3, 'rsi': 0.4, 'macd': 0.3},
        MarketCondition.LATERAL_HIGH_VOL: {'pattern': 0.5, 'rsi': 0.3, 'macd': 0.2},
        MarketCondition.MODERATE_DOWNTREND: {'pattern': 0.4, 'rsi': 0.3, 'macd': 0.3},
        MarketCondition.STRONG_DOWNTREND: {'pattern': 0.5, 'rsi': 0.2, 'macd': 0.3},
        MarketCondition.EXTREME_VOLATILITY: {'pattern': 0.6, 'rsi': 0.2, 'macd': 0.2}
    }
    
    # Obtener pesos para la condición actual
    current_weights = weights.get(market_condition, {'pattern': 0.33, 'rsi': 0.33, 'macd': 0.34})
    
    # Calcular señal ponderada
    weighted_signal = sum(signals[k] * current_weights[k] for k in signals)
    
    # Determinar umbral según condición de mercado
    if market_condition in [MarketCondition.LATERAL_LOW_VOL, MarketCondition.LATERAL_HIGH_VOL]:
        threshold = 0.3  # Más sensible en mercados laterales
    elif market_condition == MarketCondition.EXTREME_VOLATILITY:
        threshold = 0.6  # Menos sensible en alta volatilidad
    else:
        threshold = 0.4  # Umbral normal
    
    # Generar señal si supera umbral
    if weighted_signal > threshold:
        # Señal de compra
        
        # Calcular stop loss y take profit dinámicos
        current_price = df['close'].iloc[-1]
        atr = df_indicators['atr_14'].iloc[-1]
        
        # Usar ATR para cálculo de SL/TP
        stop_multiplier = 1.5
        tp_multiplier = 2.0
        
        stop_loss = current_price - (atr * stop_multiplier)
        take_profit = current_price + (atr * tp_multiplier)
        
        return {
            'action': 'buy',
            'reason': f"Patrón alcista: {latest_pattern['type']} - Condición: {market_condition.value}",
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'market_condition': market_condition.value,
            'risk_level': 1.0,  # Riesgo estándar
            'metadata': {
                'pattern': latest_pattern['type'],
                'pattern_strength': latest_pattern.get('strength', 1),
                'market_condition': market_condition.value,
                'signals': signals,
                'weighted_signal': weighted_signal,
                'rsi': current_rsi,
                'macd_hist': current_macd_hist
            }
        }
    elif weighted_signal < -threshold:
        # Señal de venta
        
        # Calcular stop loss y take profit dinámicos
        current_price = df['close'].iloc[-1]
        atr = df_indicators['atr_14'].iloc[-1]
        
        # Usar ATR para cálculo de SL/TP
        stop_multiplier = 1.5
        tp_multiplier = 2.0
        
        stop_loss = current_price + (atr * stop_multiplier)
        take_profit = current_price - (atr * tp_multiplier)
        
        return {
            'action': 'sell',
            'reason': f"Patrón bajista: {latest_pattern['type']} - Condición: {market_condition.value}",
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'market_condition': market_condition.value,
            'risk_level': 1.0,  # Riesgo estándar
            'metadata': {
                'pattern': latest_pattern['type'],
                'pattern_strength': latest_pattern.get('strength', 1),
                'market_condition': market_condition.value,
                'signals': signals,
                'weighted_signal': weighted_signal,
                'rsi': current_rsi,
                'macd_hist': current_macd_hist
            }
        }
    
    # Sin señal
    return {}

def get_historical_data(symbol: str = 'SOL/USDT', timeframe: str = '15m', limit: int = 1000) -> pd.DataFrame:
    """
    Obtiene datos históricos de un par de trading
    
    Args:
        symbol: Par de trading (ejemplo: 'SOL/USDT')
        timeframe: Intervalo de tiempo (ejemplo: '15m', '1h', '1d')
        limit: Número máximo de velas a obtener
        
    Returns:
        pd.DataFrame: DataFrame con datos OHLCV
    """
    try:
        # Usar ccxt para obtener datos históricos
        exchange = ccxt.okx()
        
        # Obtener velas
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Crear DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convertir timestamp a datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Datos históricos obtenidos: {symbol} {timeframe} - {len(df)} velas")
        
        return df
    
    except Exception as e:
        logger.error(f"Error al obtener datos históricos: {e}")
        
        # Si falla, generar datos simulados para prueba
        logger.warning("Generando datos simulados para pruebas")
        
        # Crear timestamps
        end_date = datetime.now()
        start_date = end_date - timedelta(days=limit * 0.01)  # Ajustar según timeframe
        
        timestamps = pd.date_range(start=start_date, end=end_date, periods=limit)
        
        # Generar precio base
        base_price = 100.0
        
        # Generar datos con tendencia y algo de aleatoriedad
        data = []
        price = base_price
        
        for i in range(limit):
            # Simular tendencia con algo de aleatoriedad
            price_change = np.random.normal(0, 0.01)  # Media 0, desviación 1%
            
            # Añadir ciclos
            price_change += 0.005 * np.sin(i / 100 * np.pi)
            
            # Aplicar cambio
            price *= (1 + price_change)
            
            # Generar high, low, open
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = low + (high - low) * np.random.random()
            
            # Generar volumen
            volume = abs(np.random.normal(1000, 200)) * (1 + abs(price_change) * 10)
            
            data.append([open_price, high, low, price, volume])
        
        # Crear DataFrame
        df = pd.DataFrame(data, index=timestamps, columns=['open', 'high', 'low', 'close', 'volume'])
        
        return df

def run_backtest():
    """Ejecuta backtest de la estrategia con datos históricos"""
    # Obtener datos históricos
    print("Obteniendo datos históricos...")
    df = get_historical_data(symbol='SOL/USDT', timeframe='15m', limit=1000)
    
    # Verificar que hay datos suficientes
    if len(df) < 200:
        print(f"Datos insuficientes: {len(df)} velas")
        return
    
    print(f"Datos cargados: {len(df)} velas de {df.index[0]} a {df.index[-1]}")
    
    # Calcular indicadores
    print("Calculando indicadores...")
    df_with_indicators = calculate_indicators(df)
    
    # Crear simulador
    simulator = TradingSimulator({
        'initial_balance': 10000.0,
        'fees': 0.001,
        'show_progress': True
    })
    
    # Ejecutar simulación
    print("Ejecutando simulación...")
    results = simulator.run_simulation(df, pattern_based_strategy)
    
    # Mostrar resultados
    print("\n===== RESULTADOS DEL BACKTEST =====")
    print(f"Balance inicial: ${results['initial_balance']:.2f}")
    print(f"Balance final: ${results['final_balance']:.2f}")
    print(f"P/L total: ${results['total_pnl']:.2f} ({results['pnl_percentage']:.2f}%)")
    print(f"Drawdown máximo: {results['max_drawdown']:.2f}%")
    print(f"Operaciones: {results['trade_count']} (ganadas: {results['win_count']}, perdidas: {results['loss_count']})")
    print(f"Win rate: {results['win_rate']:.2f}%")
    print(f"Profit factor: {results['profit_factor']:.2f}")
    print(f"Fees totales: ${results['total_fees']:.2f}")
    print(f"Slippage total: ${results['total_slippage']:.2f}")
    
    # Generar reporte
    print("Generando reporte HTML...")
    simulator.generate_report("backtest_report.html")
    print("Reporte generado: backtest_report.html")
    
    # Mostrar estadísticas de patrones
    print("\n===== RENDIMIENTO DE PATRONES =====")
    pattern_recognizer = PatternRecognition()
    best_patterns = pattern_recognizer.get_best_patterns(min_occurrences=3)
    
    if best_patterns:
        for pattern in best_patterns[:5]:  # Top 5
            print(f"{pattern['pattern']}: Win rate {pattern['win_rate']*100:.2f}% ({pattern['detected']} operaciones)")
    else:
        print("No hay estadísticas de patrones disponibles todavía")

if __name__ == "__main__":
    print("DEMO DE BACKTEST DE ESTRATEGIA CON RECONOCIMIENTO DE PATRONES")
    print("=" * 60)
    run_backtest()