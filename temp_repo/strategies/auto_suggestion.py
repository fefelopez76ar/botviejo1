"""
Módulo para sugerencia automática de estrategias basada en análisis de mercado
Este módulo combina análisis técnico, backtesting y aprendizaje para recomendar
la mejor estrategia para las condiciones actuales del mercado
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from adaptive_system.weighting import MarketCondition, TimeInterval
from backtesting.advanced_optimizer import TrendDetector, MultiStrategyBacktester
from data_management.market_data import get_market_data, update_market_data
from strategies.strategy_profiles import StrategyProfile, TradingStyle, get_default_profile_for_market_condition

logger = logging.getLogger("AutoSuggestion")

class StrategyRecommender:
    """Clase para recomendar estrategias basadas en análisis de mercado"""
    
    def __init__(self):
        """Inicializa el recomendador de estrategias"""
        self.trend_detector = TrendDetector()
        self.backtester = MultiStrategyBacktester()
        self.recent_recommendations = {}  # Caché de recomendaciones recientes
    
    def analyze_market_conditions(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analiza las condiciones actuales del mercado
        
        Args:
            symbol: Par de trading
            timeframe: Intervalo de tiempo
            
        Returns:
            Dict: Análisis de las condiciones de mercado
        """
        logger.info(f"Analizando condiciones de mercado para {symbol} en {timeframe}")
        
        # Obtener datos históricos
        data = update_market_data(symbol, timeframe)
        
        # Detectar tendencia
        trend = self.trend_detector.detect_trend(data)
        
        # Detectar condición de mercado para sistema adaptativo
        market_condition = self.trend_detector.detect_market_condition(data)
        
        # Calcular volatilidad
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5)  # Anualizada
        
        # Calcular medias móviles
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        data['sma_100'] = data['close'].rolling(window=100).mean()
        data['sma_200'] = data['close'].rolling(window=200).mean()
        
        # Calcular ADX (tendencia)
        adx = self._calculate_adx(data)
        
        # Determinar fuerza de la tendencia
        trend_strength = "neutral"
        if adx > 30:
            if data['close'].iloc[-1] > data['sma_50'].iloc[-1]:
                trend_strength = "strong_bullish" if adx > 40 else "moderate_bullish"
            else:
                trend_strength = "strong_bearish" if adx > 40 else "moderate_bearish"
        
        # Precio y medias actuales
        current_price = data['close'].iloc[-1]
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "trend": trend.value,
            "market_condition": market_condition.value,
            "volatility": volatility,
            "adx": adx,
            "trend_strength": trend_strength,
            "current_price": current_price,
            "sma_20": data['sma_20'].iloc[-1],
            "sma_50": data['sma_50'].iloc[-1],
            "sma_100": data['sma_100'].iloc[-1],
            "sma_200": data['sma_200'].iloc[-1],
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calcula el ADX (Average Directional Index)
        
        Args:
            data: DataFrame con datos OHLCV
            period: Período para el cálculo
            
        Returns:
            float: Valor del ADX
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # +DM y -DM
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        # Condición: +DM > -DM y +DM > 0
        condition1 = (plus_dm > minus_dm) & (plus_dm > 0)
        plus_dm[~condition1] = 0
        
        # Condición: -DM > +DM y -DM > 0
        condition2 = (minus_dm > plus_dm) & (minus_dm > 0)
        minus_dm[~condition2] = 0
        
        # +DI y -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # DX y ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        # Retornar último valor
        return adx.iloc[-1]
    
    def run_multi_strategy_backtest(self, symbol: str, timeframe: str, days: int = 60) -> Dict[str, Any]:
        """
        Ejecuta backtesting de todas las estrategias
        
        Args:
            symbol: Par de trading
            timeframe: Intervalo de tiempo
            days: Días para el backtesting
            
        Returns:
            Dict: Resultados del backtesting
        """
        logger.info(f"Ejecutando backtesting multi-estrategia para {symbol} en {timeframe}")
        
        # Verificar si hay resultados recientes en caché
        cache_key = f"{symbol}_{timeframe}_{days}"
        if cache_key in self.recent_recommendations:
            cached = self.recent_recommendations[cache_key]
            # Si el resultado es de menos de 1 hora, usar caché
            if (datetime.now() - cached["timestamp"]).total_seconds() < 3600:
                logger.info(f"Usando resultados en caché para {cache_key}")
                return cached["results"]
        
        # Ejecutar backtesting
        results = self.backtester.run_all_strategies(symbol, timeframe, days)
        
        # Guardar en caché
        self.recent_recommendations[cache_key] = {
            "timestamp": datetime.now(),
            "results": results
        }
        
        return results
    
    def get_recommended_strategy(self, symbol: str, timeframe: str, days: int = 60) -> Dict[str, Any]:
        """
        Obtiene la estrategia recomendada para las condiciones actuales
        
        Args:
            symbol: Par de trading
            timeframe: Intervalo de tiempo
            days: Días para el backtesting
            
        Returns:
            Dict: Información de la estrategia recomendada
        """
        logger.info(f"Obteniendo estrategia recomendada para {symbol} en {timeframe}")
        
        # Analizar condiciones de mercado
        market_analysis = self.analyze_market_conditions(symbol, timeframe)
        market_condition = market_analysis["market_condition"]
        trend = market_analysis["trend"]
        
        # Ejecutar backtesting multi-estrategia
        backtest_results = self.run_multi_strategy_backtest(symbol, timeframe, days)
        
        # Obtener las mejores estrategias
        best_strategies = self.backtester.get_best_strategies(symbol, top_n=5)
        
        # Obtener la mejor estrategia para la tendencia actual
        best_for_trend = self.backtester.get_best_strategy_for_trend(trend, symbol)
        
        # Si no hay estrategia específica para la tendencia, usar la mejor general
        if not best_for_trend and best_strategies:
            best_for_trend = best_strategies[0]["name"]
        
        # Si aún no hay estrategia, usar una predeterminada
        if not best_for_trend:
            # Usar perfiles predeterminados basados en la condición
            profile_name = get_default_profile_for_market_condition(market_condition, timeframe)
            
            # Mapear perfil a estrategia
            profile_to_strategy = {
                "Scalping_Default": "RSI Strategy",
                "DayTrading_Default": "MACD Strategy",
                "SwingTrading_Default": "Bollinger Bands"
            }
            
            best_for_trend = profile_to_strategy.get(profile_name, "MACD Strategy")
        
        # Configurar parámetros óptimos
        optimal_params = {}
        for strategy in best_strategies:
            if strategy["name"] == best_for_trend:
                optimal_params = strategy.get("params", {})
                break
        
        # Preparar resultado de la recomendación
        recommendation = {
            "symbol": symbol,
            "timeframe": timeframe,
            "market_analysis": market_analysis,
            "recommended_strategy": best_for_trend,
            "optimal_params": optimal_params,
            "best_strategies": best_strategies[:3],  # Top 3
            "timestamp": datetime.now().isoformat(),
            "days_analyzed": days,
            "profile_suggestion": get_default_profile_for_market_condition(market_condition, timeframe)
        }
        
        return recommendation
    
    def get_recommended_profiles(self, symbol: str, timeframe: str, days: int = 60) -> Dict[str, Any]:
        """
        Obtiene perfiles de estrategia recomendados
        
        Args:
            symbol: Par de trading
            timeframe: Intervalo de tiempo
            days: Días para el backtesting
            
        Returns:
            Dict: Perfiles recomendados
        """
        # Obtener recomendación de estrategia
        recommendation = self.get_recommended_strategy(symbol, timeframe, days)
        
        # Obtener perfil base según condición de mercado
        market_condition = recommendation["market_analysis"]["market_condition"]
        base_profile_name = get_default_profile_for_market_condition(market_condition, timeframe)
        
        # Crear perfiles para diferentes estilos de trading
        profiles = {}
        
        # 1. Perfil para condiciones actuales (basado en perfil predeterminado)
        profiles["current_market"] = {
            "name": base_profile_name,
            "description": f"Perfil recomendado para las condiciones actuales: {market_condition}"
        }
        
        # 2. Perfil optimizado basado en backtesting
        if recommendation["best_strategies"]:
            best_strategy = recommendation["best_strategies"][0]
            profiles["optimized"] = {
                "name": f"Optimized_{best_strategy['name']}_{symbol}",
                "strategy": best_strategy["name"],
                "params": best_strategy.get("params", {}),
                "return_pct": best_strategy.get("return_pct", 0),
                "win_rate": best_strategy.get("win_rate", 0),
                "description": f"Perfil optimizado basado en backtesting de {days} días"
            }
        
        # 3. Perfiles alternativos para diferentes estilos de trading
        trading_styles = {
            "scalping": "Scalping_Default",
            "day_trading": "DayTrading_Default",
            "swing_trading": "SwingTrading_Default"
        }
        
        for style, profile_name in trading_styles.items():
            if profile_name != base_profile_name:  # Evitar duplicar el perfil base
                profiles[style] = {
                    "name": profile_name,
                    "description": f"Perfil alternativo para {style}"
                }
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "market_condition": market_condition,
            "recommendations": profiles,
            "timestamp": datetime.now().isoformat()
        }

def suggest_best_strategy(symbol: str, timeframe: str) -> Dict[str, Any]:
    """
    Función de conveniencia para sugerir la mejor estrategia
    
    Args:
        symbol: Par de trading
        timeframe: Intervalo de tiempo
        
    Returns:
        Dict: Estrategia recomendada
    """
    recommender = StrategyRecommender()
    return recommender.get_recommended_strategy(symbol, timeframe)

def suggest_best_profiles(symbol: str, timeframe: str) -> Dict[str, Any]:
    """
    Función de conveniencia para sugerir los mejores perfiles
    
    Args:
        symbol: Par de trading
        timeframe: Intervalo de tiempo
        
    Returns:
        Dict: Perfiles recomendados
    """
    recommender = StrategyRecommender()
    return recommender.get_recommended_profiles(symbol, timeframe)