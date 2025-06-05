"""
Módulo para optimización avanzada y backtesting multiestrategia
Implementa funciones de:
- Backtesting de todas las estrategias disponibles
- Detección de tendencias y regímenes de mercado
- Optimización automática de parámetros
- Generación de estrategias adaptativas
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from enum import Enum
import concurrent.futures
import matplotlib.pyplot as plt
from tqdm import tqdm

# Importaciones del proyecto
from data_management.market_data import get_market_data, update_market_data
from adaptive_system.weighting import MarketCondition, TimeInterval
from strategies.machine_learning import MLStrategy, MLEnsembleStrategy

logger = logging.getLogger("AdvancedOptimizer")

class MarketTrend(Enum):
    """Clasificación de tendencias de mercado para análisis"""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    NEUTRAL = "neutral"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    CHOPPY = "choppy"
    VOLATILE = "volatile"

class TrendDetector:
    """Clase para detectar tendencias de mercado"""
    
    def __init__(self, window_short: int = 20, window_medium: int = 50, window_long: int = 100,
                volatility_window: int = 20):
        """
        Inicializa el detector de tendencias
        
        Args:
            window_short: Ventana para promedio móvil corto
            window_medium: Ventana para promedio móvil medio
            window_long: Ventana para promedio móvil largo
            volatility_window: Ventana para cálculo de volatilidad
        """
        self.window_short = window_short
        self.window_medium = window_medium
        self.window_long = window_long
        self.volatility_window = volatility_window
    
    def detect_trend(self, df: pd.DataFrame) -> MarketTrend:
        """
        Detecta la tendencia actual del mercado
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            MarketTrend: Tendencia detectada
        """
        # Calcular medias móviles
        df['sma_short'] = df['close'].rolling(window=self.window_short).mean()
        df['sma_medium'] = df['close'].rolling(window=self.window_medium).mean()
        df['sma_long'] = df['close'].rolling(window=self.window_long).mean()
        
        # Calcular pendientes
        df['slope_short'] = (df['sma_short'] - df['sma_short'].shift(5)) / df['sma_short'].shift(5)
        df['slope_medium'] = (df['sma_medium'] - df['sma_medium'].shift(5)) / df['sma_medium'].shift(5)
        
        # Calcular volatilidad (ATR normalizado)
        df['tr'] = np.maximum(
            np.maximum(
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift())
            ),
            abs(df['low'] - df['close'].shift())
        )
        df['atr'] = df['tr'].rolling(window=self.window_short).mean()
        df['atr_norm'] = df['atr'] / df['close']
        
        # Obtener valores actuales (último punto de datos)
        current = df.iloc[-1]
        
        # Verificar alineación de SMAs
        sma_aligned_up = (current['sma_short'] > current['sma_medium'] > current['sma_long'])
        sma_aligned_down = (current['sma_short'] < current['sma_medium'] < current['sma_long'])
        
        # Verificar pendientes
        slope_short = current['slope_short']
        slope_medium = current['slope_medium']
        
        # Volatilidad
        volatility = current['atr_norm']
        high_volatility = volatility > 0.03  # 3% volatilidad
        
        # Patrón irregular (choppy)
        crossovers = 0
        for i in range(-20, -1):
            if ((df['sma_short'].iloc[i] > df['sma_medium'].iloc[i] and 
                 df['sma_short'].iloc[i-1] < df['sma_medium'].iloc[i-1]) or
                (df['sma_short'].iloc[i] < df['sma_medium'].iloc[i] and 
                 df['sma_short'].iloc[i-1] > df['sma_medium'].iloc[i-1])):
                crossovers += 1
        
        choppy_market = crossovers > 3  # Más de 3 cruces en 20 periodos
        
        # Determinar tendencia
        if high_volatility:
            return MarketTrend.VOLATILE
        elif choppy_market:
            return MarketTrend.CHOPPY
        elif sma_aligned_up and slope_short > 0.005 and slope_medium > 0.002:
            return MarketTrend.STRONG_UPTREND
        elif sma_aligned_up and slope_short > 0:
            return MarketTrend.WEAK_UPTREND
        elif sma_aligned_down and slope_short < -0.005 and slope_medium < -0.002:
            return MarketTrend.STRONG_DOWNTREND
        elif sma_aligned_down and slope_short < 0:
            return MarketTrend.WEAK_DOWNTREND
        else:
            return MarketTrend.NEUTRAL
    
    def detect_market_condition(self, df: pd.DataFrame) -> MarketCondition:
        """
        Detecta la condición de mercado (para sistema adaptativo)
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            MarketCondition: Condición de mercado detectada
        """
        # Detectar tendencia
        trend = self.detect_trend(df)
        
        # Calcular volatilidad
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=self.volatility_window).std()
        current_volatility = volatility.iloc[-1]
        
        # Alta volatilidad
        if current_volatility > 0.04:  # 4% diario
            return MarketCondition.EXTREME_VOLATILITY
        
        # Mapear tendencia a condición de mercado
        if trend == MarketTrend.STRONG_UPTREND:
            return MarketCondition.STRONG_UPTREND
        elif trend == MarketTrend.WEAK_UPTREND:
            return MarketCondition.MODERATE_UPTREND
        elif trend == MarketTrend.STRONG_DOWNTREND:
            return MarketCondition.STRONG_DOWNTREND
        elif trend == MarketTrend.WEAK_DOWNTREND:
            return MarketCondition.MODERATE_DOWNTREND
        elif trend == MarketTrend.CHOPPY:
            if current_volatility > 0.02:  # 2% diario
                return MarketCondition.LATERAL_HIGH_VOL
            else:
                return MarketCondition.LATERAL_LOW_VOL
        else:  # NEUTRAL o VOLATILE
            if current_volatility > 0.02:
                return MarketCondition.LATERAL_HIGH_VOL
            else:
                return MarketCondition.LATERAL_LOW_VOL

class StrategyRepository:
    """Repositorio de estrategias disponibles para backtesting"""
    
    def __init__(self):
        """Inicializa el repositorio de estrategias"""
        self.strategies = self._load_strategies()
    
    def _load_strategies(self) -> Dict[str, Callable]:
        """
        Carga todas las estrategias disponibles
        
        Returns:
            Dict[str, Callable]: Diccionario de estrategias
        """
        # Importar estrategias disponibles
        try:
            from strategies.classic import (
                sma_crossover_strategy, rsi_strategy, macd_strategy, 
                bollinger_bands_strategy, mean_reversion_strategy
            )
            
            strategies = {
                "SMA Crossover": sma_crossover_strategy,
                "RSI Strategy": rsi_strategy,
                "MACD Strategy": macd_strategy,
                "Bollinger Bands": bollinger_bands_strategy,
                "Mean Reversion": mean_reversion_strategy
            }
            
            # Añadir variantes de estrategias con diferentes parámetros
            strategies["SMA Crossover (Fast)"] = lambda df: sma_crossover_strategy(df, short_period=5, long_period=20)
            strategies["SMA Crossover (Slow)"] = lambda df: sma_crossover_strategy(df, short_period=20, long_period=50)
            strategies["RSI (Aggressive)"] = lambda df: rsi_strategy(df, overbought=65, oversold=35)
            strategies["RSI (Conservative)"] = lambda df: rsi_strategy(df, overbought=75, oversold=25)
            
            # Añadir estrategias compuestas
            def rsi_macd_strategy(df):
                rsi_signal = rsi_strategy(df)
                macd_signal = macd_strategy(df)
                # Combinar señales (señal solo si ambos concuerdan)
                return pd.Series(
                    [(1 if rsi_signal.iloc[i] > 0 and macd_signal.iloc[i] > 0 else
                      -1 if rsi_signal.iloc[i] < 0 and macd_signal.iloc[i] < 0 else 0)
                     for i in range(len(rsi_signal))],
                    index=rsi_signal.index
                )
            
            strategies["RSI+MACD Combined"] = rsi_macd_strategy
            
            return strategies
        
        except ImportError as e:
            logger.error(f"Error importing strategies: {e}")
            return {}
    
    def get_strategy_names(self) -> List[str]:
        """
        Obtiene los nombres de todas las estrategias disponibles
        
        Returns:
            List[str]: Lista de nombres de estrategias
        """
        return list(self.strategies.keys())
    
    def get_strategy(self, name: str) -> Optional[Callable]:
        """
        Obtiene una estrategia por nombre
        
        Args:
            name: Nombre de la estrategia
            
        Returns:
            Optional[Callable]: Función de estrategia o None si no existe
        """
        return self.strategies.get(name)
    
    def get_all_strategies(self) -> Dict[str, Callable]:
        """
        Obtiene todas las estrategias disponibles
        
        Returns:
            Dict[str, Callable]: Diccionario de estrategias
        """
        return self.strategies

class BacktestResult:
    """Clase para almacenar resultados de backtesting"""
    
    def __init__(self, strategy_name: str, params: Dict, data: pd.DataFrame, signals: pd.Series):
        """
        Inicializa el resultado de backtesting
        
        Args:
            strategy_name: Nombre de la estrategia
            params: Parámetros utilizados
            data: DataFrame con datos OHLCV
            signals: Series con señales (-1, 0, 1)
        """
        self.strategy_name = strategy_name
        self.params = params
        self.data = data
        self.signals = signals
        self.trades = []
        self.metrics = {}
        
        # Ejecutar simulación
        self._run_simulation()
        # Calcular métricas
        self._calculate_metrics()
    
    def _run_simulation(self, commission: float = 0.001, slippage: float = 0.001,
                       initial_balance: float = 10000.0):
        """
        Ejecuta la simulación de trading
        
        Args:
            commission: Comisión por operación (0.001 = 0.1%)
            slippage: Deslizamiento por operación (0.001 = 0.1%)
            initial_balance: Balance inicial
        """
        position = 0  # 0: sin posición, 1: largo, -1: corto
        entry_price = 0.0
        entry_time = None
        balance = initial_balance
        equity = [initial_balance]
        
        for i in range(1, len(self.signals)):
            current_price = self.data['close'].iloc[i]
            current_time = self.data.index[i]
            signal = self.signals.iloc[i]
            
            # Si no hay posición y hay señal, abrir posición
            if position == 0 and signal != 0:
                position = 1 if signal > 0 else -1
                # Aplicar slippage
                if position == 1:
                    entry_price = current_price * (1 + slippage)
                else:
                    entry_price = current_price * (1 - slippage)
                
                entry_time = current_time
                
                # Registrar operación
                self.trades.append({
                    "type": "entry",
                    "position": "long" if position == 1 else "short",
                    "time": current_time,
                    "price": entry_price,
                    "balance": balance
                })
            
            # Si hay posición y hay señal contraria, cerrar posición
            elif position != 0 and ((position == 1 and signal < 0) or (position == -1 and signal > 0)):
                # Aplicar slippage
                if position == 1:
                    exit_price = current_price * (1 - slippage)
                else:
                    exit_price = current_price * (1 + slippage)
                
                # Calcular P&L
                if position == 1:
                    pnl = (exit_price / entry_price - 1) * balance
                else:
                    pnl = (entry_price / exit_price - 1) * balance
                
                # Aplicar comisión
                pnl -= balance * commission
                
                # Actualizar balance
                balance += pnl
                
                # Registrar operación
                self.trades.append({
                    "type": "exit",
                    "position": "long" if position == 1 else "short",
                    "time": current_time,
                    "price": exit_price,
                    "pnl": pnl,
                    "balance": balance
                })
                
                # Resetear posición
                position = 0
                entry_price = 0.0
                entry_time = None
            
            # Actualizar equity
            if position == 0:
                equity.append(balance)
            else:
                # Calcular equity con posición abierta
                if position == 1:
                    current_pnl = (current_price / entry_price - 1) * balance
                else:
                    current_pnl = (entry_price / current_price - 1) * balance
                
                current_pnl -= balance * commission
                equity.append(balance + current_pnl)
        
        # Cerrar posición al final si sigue abierta
        if position != 0:
            final_price = self.data['close'].iloc[-1]
            
            # Aplicar slippage
            if position == 1:
                exit_price = final_price * (1 - slippage)
            else:
                exit_price = final_price * (1 + slippage)
            
            # Calcular P&L
            if position == 1:
                pnl = (exit_price / entry_price - 1) * balance
            else:
                pnl = (entry_price / exit_price - 1) * balance
            
            # Aplicar comisión
            pnl -= balance * commission
            
            # Actualizar balance
            balance += pnl
            
            # Registrar operación
            self.trades.append({
                "type": "exit",
                "position": "long" if position == 1 else "short",
                "time": self.data.index[-1],
                "price": exit_price,
                "pnl": pnl,
                "balance": balance
            })
        
        # Guardar equity curve
        self.equity_curve = pd.Series(equity, index=self.data.index)
        self.final_balance = balance
    
    def _calculate_metrics(self):
        """Calcula métricas de rendimiento"""
        if not self.trades:
            self.metrics = {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "return_pct": 0.0,
                "annualized_return": 0.0
            }
            return
        
        # Extraer datos de trades
        exits = [t for t in self.trades if t["type"] == "exit"]
        wins = [t for t in exits if t["pnl"] > 0]
        losses = [t for t in exits if t["pnl"] <= 0]
        
        # Total de trades
        total_trades = len(exits)
        
        # Win rate
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        
        # Total profit/loss
        total_profit = sum(t["pnl"] for t in wins)
        total_loss = sum(abs(t["pnl"]) for t in losses)
        
        # Profit factor
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Máximo drawdown
        peak = self.equity_curve.iloc[0]
        max_drawdown = 0
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio
        returns = self.equity_curve.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) if len(returns) > 0 and returns.std() > 0 else 0
        
        # Retorno total
        initial_balance = self.equity_curve.iloc[0]
        final_balance = self.equity_curve.iloc[-1]
        total_return = (final_balance / initial_balance - 1) * 100
        
        # Retorno anualizado
        days = (self.data.index[-1] - self.data.index[0]).days
        annualized_return = ((final_balance / initial_balance) ** (365 / max(1, days)) - 1) * 100
        
        # Guardar métricas
        self.metrics = {
            "total_trades": total_trades,
            "win_rate": win_rate * 100,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown * 100,
            "sharpe_ratio": sharpe_ratio,
            "return_pct": total_return,
            "annualized_return": annualized_return,
            "total_profit": total_profit,
            "total_loss": total_loss
        }
    
    def to_dict(self) -> Dict:
        """
        Convierte los resultados a diccionario para serialización
        
        Returns:
            Dict: Resultados en formato diccionario
        """
        return {
            "strategy_name": self.strategy_name,
            "params": self.params,
            "final_balance": self.final_balance,
            "metrics": self.metrics,
            "trades": self.trades,
            "equity_curve": self.equity_curve.tolist()
        }
    
    def get_summary(self) -> Dict:
        """
        Obtiene un resumen de los resultados
        
        Returns:
            Dict: Resumen de resultados
        """
        return {
            "strategy": self.strategy_name,
            "total_trades": self.metrics.get("total_trades", 0),
            "win_rate": f"{self.metrics.get('win_rate', 0):.2f}%",
            "profit_factor": f"{self.metrics.get('profit_factor', 0):.2f}",
            "max_drawdown": f"{self.metrics.get('max_drawdown', 0):.2f}%",
            "sharpe_ratio": f"{self.metrics.get('sharpe_ratio', 0):.2f}",
            "return": f"{self.metrics.get('return_pct', 0):.2f}%",
            "annualized_return": f"{self.metrics.get('annualized_return', 0):.2f}%"
        }
    
    def plot(self):
        """Genera un gráfico con los resultados"""
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            
            # Graficar precios
            ax1 = plt.subplot(211)
            ax1.plot(self.data.index, self.data['close'], label='Precio')
            
            # Marcar operaciones
            buy_signals = []
            sell_signals = []
            buy_prices = []
            sell_prices = []
            
            for trade in self.trades:
                if trade["type"] == "entry" and trade["position"] == "long":
                    buy_signals.append(trade["time"])
                    buy_prices.append(trade["price"])
                elif trade["type"] == "entry" and trade["position"] == "short":
                    sell_signals.append(trade["time"])
                    sell_prices.append(trade["price"])
            
            ax1.scatter(buy_signals, buy_prices, marker='^', color='green', label='Compra')
            ax1.scatter(sell_signals, sell_prices, marker='v', color='red', label='Venta')
            
            ax1.set_title(f'Estrategia: {self.strategy_name}')
            ax1.set_ylabel('Precio')
            ax1.legend()
            
            # Graficar equity curve
            ax2 = plt.subplot(212)
            ax2.plot(self.equity_curve.index, self.equity_curve, label='Equity')
            ax2.set_title('Curva de Equity')
            ax2.set_ylabel('Equity')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            logger.error(f"Error plotting results: {e}")

class MultiStrategyBacktester:
    """Clase para backtesting de múltiples estrategias"""
    
    def __init__(self, data_manager=None):
        """
        Inicializa el backtester multiestrategia
        
        Args:
            data_manager: Gestor de datos de mercado
        """
        self.strategy_repo = StrategyRepository()
        self.trend_detector = TrendDetector()
        self.results = {}
        self.best_strategies_by_trend = {}
    
    def run_all_strategies(self, symbol: str, interval: str, days: int = 90) -> Dict[str, BacktestResult]:
        """
        Ejecuta backtesting de todas las estrategias disponibles
        
        Args:
            symbol: Par de trading
            interval: Intervalo de tiempo
            days: Días de histórico a utilizar
            
        Returns:
            Dict[str, BacktestResult]: Resultados por estrategia
        """
        logger.info(f"Running backtest for all strategies on {symbol} {interval} ({days} days)")
        
        # Obtener datos históricos
        data = update_market_data(symbol, interval)
        
        # Limitar a los días solicitados
        if days > 0:
            start_date = data.index[-1] - pd.Timedelta(days=days)
            data = data[data.index >= start_date]
        
        # Detectar tendencia
        trend = self.trend_detector.detect_trend(data)
        logger.info(f"Detected market trend: {trend.value}")
        
        # Ejecutar todas las estrategias
        results = {}
        
        strategies = self.strategy_repo.get_all_strategies()
        logger.info(f"Testing {len(strategies)} strategies")
        
        for name, strategy_fn in tqdm(strategies.items(), desc="Running strategies"):
            try:
                signals = strategy_fn(data)
                result = BacktestResult(name, {}, data, signals)
                results[name] = result
                logger.info(f"Strategy {name}: Return={result.metrics['return_pct']:.2f}%, Win Rate={result.metrics['win_rate']:.2f}%")
            except Exception as e:
                logger.error(f"Error running strategy {name}: {e}")
        
        # Guardar resultados
        self.results[symbol] = results
        
        # Clasificar estrategias por rendimiento según tendencia
        self._classify_strategies_by_trend(symbol, trend)
        
        return results
    
    def _classify_strategies_by_trend(self, symbol: str, trend: MarketTrend):
        """
        Clasifica estrategias por rendimiento según tendencia
        
        Args:
            symbol: Par de trading
            trend: Tendencia detectada
        """
        if symbol not in self.results:
            return
        
        # Ordenar por retorno
        sorted_results = sorted(
            self.results[symbol].items(),
            key=lambda x: x[1].metrics.get("return_pct", 0),
            reverse=True
        )
        
        # Guardar clasificación por tendencia
        if trend not in self.best_strategies_by_trend:
            self.best_strategies_by_trend[trend] = {}
        
        self.best_strategies_by_trend[trend][symbol] = [
            {"name": name, "metrics": result.metrics}
            for name, result in sorted_results[:5]  # Top 5
        ]
    
    def get_best_strategies(self, symbol: str, top_n: int = 5) -> List[Dict]:
        """
        Obtiene las mejores estrategias para un símbolo
        
        Args:
            symbol: Par de trading
            top_n: Número de estrategias a retornar
            
        Returns:
            List[Dict]: Mejores estrategias con sus métricas
        """
        if symbol not in self.results:
            return []
        
        sorted_results = sorted(
            self.results[symbol].items(),
            key=lambda x: x[1].metrics.get("return_pct", 0),
            reverse=True
        )
        
        return [
            {
                "name": name,
                "return_pct": result.metrics.get("return_pct", 0),
                "win_rate": result.metrics.get("win_rate", 0),
                "profit_factor": result.metrics.get("profit_factor", 0),
                "sharpe_ratio": result.metrics.get("sharpe_ratio", 0),
                "max_drawdown": result.metrics.get("max_drawdown", 0),
                "total_trades": result.metrics.get("total_trades", 0)
            }
            for name, result in sorted_results[:top_n]
        ]
    
    def get_best_strategy_for_trend(self, trend: MarketTrend, symbol: str) -> Optional[str]:
        """
        Obtiene la mejor estrategia para una tendencia específica
        
        Args:
            trend: Tendencia de mercado
            symbol: Par de trading
            
        Returns:
            Optional[str]: Nombre de la mejor estrategia o None
        """
        if trend not in self.best_strategies_by_trend:
            return None
        
        if symbol not in self.best_strategies_by_trend[trend]:
            return None
        
        strategies = self.best_strategies_by_trend[trend][symbol]
        if not strategies:
            return None
        
        return strategies[0]["name"]
    
    def generate_trend_report(self) -> Dict:
        """
        Genera un informe de las mejores estrategias por tendencia
        
        Returns:
            Dict: Informe de estrategias por tendencia
        """
        report = {}
        
        for trend, symbols_data in self.best_strategies_by_trend.items():
            report[trend.value] = {}
            
            for symbol, strategies in symbols_data.items():
                report[trend.value][symbol] = strategies
        
        return report
    
    def optimize_strategy(self, strategy_name: str, symbol: str, interval: str, 
                         param_grid: Dict, days: int = 90) -> Dict:
        """
        Optimiza los parámetros de una estrategia
        
        Args:
            strategy_name: Nombre de la estrategia
            symbol: Par de trading
            interval: Intervalo de tiempo
            param_grid: Grid de parámetros a probar
            days: Días de histórico a utilizar
            
        Returns:
            Dict: Resultados de optimización
        """
        logger.info(f"Optimizing strategy {strategy_name} for {symbol} {interval}")
        
        # Obtener función de estrategia
        strategy_fn = self.strategy_repo.get_strategy(strategy_name)
        if strategy_fn is None:
            logger.error(f"Strategy {strategy_name} not found")
            return {"error": f"Strategy {strategy_name} not found"}
        
        # Obtener datos históricos
        data = update_market_data(symbol, interval)
        
        # Limitar a los días solicitados
        if days > 0:
            start_date = data.index[-1] - pd.Timedelta(days=days)
            data = data[data.index >= start_date]
        
        # Generar todas las combinaciones de parámetros
        import itertools
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        results = []
        
        for params in tqdm(param_combinations, desc="Optimizing parameters"):
            # Construir diccionario de parámetros
            param_dict = {param_keys[i]: params[i] for i in range(len(param_keys))}
            
            try:
                # Función anónima con parámetros específicos
                strategy_with_params = lambda df: strategy_fn(df, **param_dict)
                
                # Ejecutar backtest
                signals = strategy_with_params(data)
                result = BacktestResult(strategy_name, param_dict, data, signals)
                
                # Guardar resultado
                results.append({
                    "params": param_dict,
                    "metrics": result.metrics,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Error testing parameters {param_dict}: {e}")
        
        # Ordenar por retorno
        sorted_results = sorted(
            results,
            key=lambda x: x["metrics"].get("return_pct", 0),
            reverse=True
        )
        
        best_result = sorted_results[0] if sorted_results else None
        
        return {
            "strategy": strategy_name,
            "symbol": symbol,
            "interval": interval,
            "best_params": best_result["params"] if best_result else None,
            "best_metrics": best_result["metrics"] if best_result else None,
            "all_results": [
                {
                    "params": r["params"],
                    "return_pct": r["metrics"].get("return_pct", 0),
                    "win_rate": r["metrics"].get("win_rate", 0),
                    "profit_factor": r["metrics"].get("profit_factor", 0),
                    "sharpe_ratio": r["metrics"].get("sharpe_ratio", 0)
                }
                for r in sorted_results[:10]  # Top 10
            ]
        }
    
    def save_results(self, file_path: str = "data/backtest_results.json"):
        """
        Guarda los resultados de backtesting
        
        Args:
            file_path: Ruta de archivo para guardar resultados
        """
        results_data = {}
        
        for symbol, symbol_results in self.results.items():
            results_data[symbol] = {
                name: result.to_dict()
                for name, result in symbol_results.items()
            }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w") as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Backtest results saved to {file_path}")
    
    def load_results(self, file_path: str = "data/backtest_results.json"):
        """
        Carga resultados de backtesting guardados
        
        Args:
            file_path: Ruta de archivo para cargar resultados
        """
        if not os.path.exists(file_path):
            logger.warning(f"Results file {file_path} not found")
            return
        
        try:
            with open(file_path, "r") as f:
                results_data = json.load(f)
            
            # Recrear objetos BacktestResult
            for symbol, symbol_results in results_data.items():
                if symbol not in self.results:
                    self.results[symbol] = {}
                
                for name, result_data in symbol_results.items():
                    # Recrear DataFrame
                    data = pd.DataFrame()  # Dummy DataFrame, no podemos reconstruir completamente
                    
                    # Recrear Series con equidad
                    equity_curve = pd.Series(result_data["equity_curve"])
                    
                    # Crear resultado
                    result = BacktestResult(name, result_data["params"], data, pd.Series())
                    result.metrics = result_data["metrics"]
                    result.trades = result_data["trades"]
                    result.equity_curve = equity_curve
                    result.final_balance = result_data["final_balance"]
                    
                    # Guardar resultado
                    self.results[symbol][name] = result
            
            logger.info(f"Backtest results loaded from {file_path}")
        
        except Exception as e:
            logger.error(f"Error loading backtest results: {e}")

class AutomatedLearningSystem:
    """Sistema de aprendizaje automatizado para bots de trading"""
    
    def __init__(self, db_path: str = "data/learning_system"):
        """
        Inicializa el sistema de aprendizaje
        
        Args:
            db_path: Ruta base para guardar datos de aprendizaje
        """
        self.db_path = db_path
        self.backtester = MultiStrategyBacktester()
        self.trend_detector = TrendDetector()
        self.active_bots = {}
        
        # Crear directorios
        os.makedirs(db_path, exist_ok=True)
        os.makedirs(f"{db_path}/models", exist_ok=True)
        os.makedirs(f"{db_path}/data", exist_ok=True)
        os.makedirs(f"{db_path}/bots", exist_ok=True)
    
    def analyze_market_conditions(self, symbol: str, interval: str) -> Dict:
        """
        Analiza las condiciones actuales del mercado
        
        Args:
            symbol: Par de trading
            interval: Intervalo de tiempo
            
        Returns:
            Dict: Análisis de condiciones de mercado
        """
        # Obtener datos
        data = update_market_data(symbol, interval)
        
        # Detectar tendencia
        trend = self.trend_detector.detect_trend(data)
        
        # Detectar condición de mercado
        market_condition = self.trend_detector.detect_market_condition(data)
        
        # Calcular volatilidad
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5)  # Anualizada
        
        # Calcular media móviles
        sma20 = data['close'].rolling(20).mean().iloc[-1]
        sma50 = data['close'].rolling(50).mean().iloc[-1]
        sma100 = data['close'].rolling(100).mean().iloc[-1]
        
        # Precio actual
        current_price = data['close'].iloc[-1]
        
        # Días desde que cambió la tendencia
        days_in_trend = 1
        prev_trend = None
        
        for i in range(len(data) - 2, 0, -1):
            df_subset = data.iloc[:i + 1]
            trend_i = self.trend_detector.detect_trend(df_subset)
            
            if prev_trend is None:
                prev_trend = trend_i
            elif trend_i != prev_trend:
                break
            
            days_in_trend += 1
        
        return {
            "symbol": symbol,
            "interval": interval,
            "trend": trend.value,
            "market_condition": market_condition.value,
            "volatility": volatility,
            "current_price": current_price,
            "sma20": sma20,
            "sma50": sma50,
            "sma100": sma100,
            "days_in_trend": days_in_trend,
            "timestamp": datetime.now().isoformat()
        }
    
    def find_best_strategy_for_current_conditions(self, symbol: str, interval: str) -> Dict:
        """
        Encuentra la mejor estrategia para las condiciones actuales
        
        Args:
            symbol: Par de trading
            interval: Intervalo de tiempo
            
        Returns:
            Dict: Mejor estrategia para condiciones actuales
        """
        # Analizar condiciones actuales
        analysis = self.analyze_market_conditions(symbol, interval)
        
        # Ejecutar backtesting de todas las estrategias
        self.backtester.run_all_strategies(symbol, interval, days=60)
        
        # Obtener mejores estrategias
        best_strategies = self.backtester.get_best_strategies(symbol, top_n=5)
        
        # Obtener mejor estrategia para la tendencia actual
        trend = MarketTrend(analysis["trend"])
        best_for_trend = self.backtester.get_best_strategy_for_trend(trend, symbol)
        
        # Si no hay estrategia específica para la tendencia, usar la mejor general
        if best_for_trend is None and best_strategies:
            best_for_trend = best_strategies[0]["name"]
        
        return {
            "analysis": analysis,
            "best_strategies": best_strategies,
            "best_for_current_trend": best_for_trend
        }
    
    def create_simulation_bot(self, symbol: str, interval: str, strategy_name: str = None,
                            initial_balance: float = 1000.0) -> str:
        """
        Crea un bot de simulación con la mejor estrategia para condiciones actuales
        
        Args:
            symbol: Par de trading
            interval: Intervalo de tiempo
            strategy_name: Nombre de estrategia (opcional, usa la mejor si es None)
            initial_balance: Balance inicial
            
        Returns:
            str: ID del bot creado
        """
        # Si no se proporciona estrategia, encontrar la mejor
        if strategy_name is None:
            result = self.find_best_strategy_for_current_conditions(symbol, interval)
            strategy_name = result["best_for_current_trend"]
            
            if strategy_name is None:
                logger.error("No suitable strategy found")
                return None
        
        # Verificar que la estrategia existe
        strategy_fn = self.backtester.strategy_repo.get_strategy(strategy_name)
        if strategy_fn is None:
            logger.error(f"Strategy {strategy_name} not found")
            return None
        
        # Generar ID único
        bot_id = f"sim_{int(time.time())}_{symbol.replace('-', '_')}"
        
        # Crear configuración del bot
        bot_config = {
            "id": bot_id,
            "name": f"Sim_{strategy_name}_{symbol}",
            "symbol": symbol,
            "interval": interval,
            "strategy": strategy_name,
            "initial_balance": initial_balance,
            "current_balance": initial_balance,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "trades": [],
            "performance": {},
            "last_update": datetime.now().isoformat()
        }
        
        # Guardar configuración
        self._save_bot_config(bot_id, bot_config)
        
        logger.info(f"Created simulation bot {bot_id} with strategy {strategy_name}")
        
        return bot_id
    
    def start_simulation_bot(self, bot_id: str, days_to_simulate: int = 30) -> Dict:
        """
        Inicia un bot de simulación
        
        Args:
            bot_id: ID del bot
            days_to_simulate: Días a simular
            
        Returns:
            Dict: Resultados de la simulación
        """
        # Cargar configuración
        bot_config = self._load_bot_config(bot_id)
        if bot_config is None:
            logger.error(f"Bot {bot_id} not found")
            return {"error": f"Bot {bot_id} not found"}
        
        symbol = bot_config["symbol"]
        interval = bot_config["interval"]
        strategy_name = bot_config["strategy"]
        
        logger.info(f"Starting simulation bot {bot_id} for {days_to_simulate} days")
        
        # Obtener datos
        data = update_market_data(symbol, interval)
        
        # Limitar a los días solicitados
        if days_to_simulate > 0:
            start_date = data.index[-1] - pd.Timedelta(days=days_to_simulate)
            data = data[data.index >= start_date]
        
        # Obtener estrategia
        strategy_fn = self.backtester.strategy_repo.get_strategy(strategy_name)
        if strategy_fn is None:
            logger.error(f"Strategy {strategy_name} not found")
            return {"error": f"Strategy {strategy_name} not found"}
        
        # Ejecutar backtest
        signals = strategy_fn(data)
        result = BacktestResult(strategy_name, {}, data, signals)
        
        # Actualizar configuración
        bot_config["status"] = "completed"
        bot_config["current_balance"] = result.final_balance
        bot_config["trades"] = result.trades
        bot_config["performance"] = result.metrics
        bot_config["last_update"] = datetime.now().isoformat()
        
        # Guardar configuración
        self._save_bot_config(bot_id, bot_config)
        
        # Registrar datos de aprendizaje
        self._register_learning_data(
            symbol=symbol,
            interval=interval,
            strategy_name=strategy_name,
            metrics=result.metrics,
            market_condition=self.trend_detector.detect_market_condition(data).value
        )
        
        return {
            "bot_id": bot_id,
            "symbol": symbol,
            "strategy": strategy_name,
            "initial_balance": bot_config["initial_balance"],
            "final_balance": result.final_balance,
            "return_pct": result.metrics["return_pct"],
            "trades": len(result.trades),
            "win_rate": result.metrics["win_rate"]
        }
    
    def _register_learning_data(self, symbol: str, interval: str, strategy_name: str,
                              metrics: Dict, market_condition: str):
        """
        Registra datos de aprendizaje
        
        Args:
            symbol: Par de trading
            interval: Intervalo de tiempo
            strategy_name: Nombre de estrategia
            metrics: Métricas de rendimiento
            market_condition: Condición de mercado
        """
        # Crear archivo de datos de aprendizaje si no existe
        learning_file = f"{self.db_path}/data/learning_data.json"
        
        learning_data = {}
        if os.path.exists(learning_file):
            try:
                with open(learning_file, "r") as f:
                    learning_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading learning data: {e}")
        
        # Estructura: market_condition -> strategy -> métricas promedio
        if market_condition not in learning_data:
            learning_data[market_condition] = {}
        
        if strategy_name not in learning_data[market_condition]:
            learning_data[market_condition][strategy_name] = {
                "count": 0,
                "avg_return": 0.0,
                "avg_win_rate": 0.0,
                "avg_profit_factor": 0.0,
                "symbols": []
            }
        
        # Actualizar datos
        strategy_data = learning_data[market_condition][strategy_name]
        count = strategy_data["count"]
        
        # Actualizar promedios
        if count > 0:
            strategy_data["avg_return"] = (strategy_data["avg_return"] * count + metrics["return_pct"]) / (count + 1)
            strategy_data["avg_win_rate"] = (strategy_data["avg_win_rate"] * count + metrics["win_rate"]) / (count + 1)
            strategy_data["avg_profit_factor"] = (strategy_data["avg_profit_factor"] * count + metrics["profit_factor"]) / (count + 1)
        else:
            strategy_data["avg_return"] = metrics["return_pct"]
            strategy_data["avg_win_rate"] = metrics["win_rate"]
            strategy_data["avg_profit_factor"] = metrics["profit_factor"]
        
        # Incrementar contador
        strategy_data["count"] += 1
        
        # Añadir símbolo si no existe
        if symbol not in strategy_data["symbols"]:
            strategy_data["symbols"].append(symbol)
        
        # Guardar datos
        with open(learning_file, "w") as f:
            json.dump(learning_data, f, indent=2)
        
        logger.info(f"Learning data registered for {strategy_name} in {market_condition}")
    
    def get_learning_insights(self) -> Dict:
        """
        Obtiene insights del sistema de aprendizaje
        
        Returns:
            Dict: Insights de aprendizaje
        """
        learning_file = f"{self.db_path}/data/learning_data.json"
        
        if not os.path.exists(learning_file):
            return {"error": "No learning data available"}
        
        try:
            with open(learning_file, "r") as f:
                learning_data = json.load(f)
            
            # Encontrar las mejores estrategias por condición de mercado
            best_strategies = {}
            
            for condition, strategies in learning_data.items():
                sorted_strategies = sorted(
                    strategies.items(),
                    key=lambda x: x[1]["avg_return"],
                    reverse=True
                )
                
                best_strategies[condition] = [
                    {
                        "name": name,
                        "avg_return": data["avg_return"],
                        "avg_win_rate": data["avg_win_rate"],
                        "avg_profit_factor": data["avg_profit_factor"],
                        "count": data["count"]
                    }
                    for name, data in sorted_strategies[:3]  # Top 3
                ]
            
            return {
                "best_strategies_by_condition": best_strategies,
                "total_conditions": len(learning_data),
                "total_strategies": sum(len(strategies) for strategies in learning_data.values())
            }
        
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {"error": f"Error getting learning insights: {e}"}
    
    def create_optimized_bot(self, symbol: str, interval: str, days: int = 60) -> str:
        """
        Crea un bot optimizado basado en aprendizaje
        
        Args:
            symbol: Par de trading
            interval: Intervalo de tiempo
            days: Días para backtesting
            
        Returns:
            str: ID del bot creado
        """
        # Analizar condiciones actuales
        analysis = self.analyze_market_conditions(symbol, interval)
        market_condition = analysis["market_condition"]
        
        # Obtener insights de aprendizaje
        insights = self.get_learning_insights()
        
        if "error" in insights:
            # Si no hay datos de aprendizaje, crear bot con estrategia base
            logger.warning("No learning data available, using base strategy")
            return self.create_simulation_bot(symbol, interval)
        
        # Buscar la mejor estrategia para la condición actual
        best_strategy = None
        if market_condition in insights["best_strategies_by_condition"]:
            top_strategies = insights["best_strategies_by_condition"][market_condition]
            if top_strategies:
                best_strategy = top_strategies[0]["name"]
        
        if best_strategy is None:
            # Si no hay estrategia específica, encontrar la mejor por backtesting
            self.backtester.run_all_strategies(symbol, interval, days=days)
            best_strategies = self.backtester.get_best_strategies(symbol, top_n=1)
            
            if best_strategies:
                best_strategy = best_strategies[0]["name"]
            else:
                logger.warning("No best strategy found, using default")
                best_strategy = "SMA Crossover"  # Estrategia por defecto
        
        # Verificar si se puede optimizar
        strategy_fn = self.backtester.strategy_repo.get_strategy(best_strategy)
        if strategy_fn is None:
            logger.error(f"Strategy {best_strategy} not found")
            return self.create_simulation_bot(symbol, interval)
        
        # Optimizar parámetros según la estrategia
        param_grid = self._get_param_grid_for_strategy(best_strategy)
        
        if param_grid:
            # Ejecutar optimización
            optimization_results = self.backtester.optimize_strategy(
                strategy_name=best_strategy,
                symbol=symbol,
                interval=interval,
                param_grid=param_grid,
                days=days
            )
            
            # Crear bot con parámetros optimizados
            bot_id = f"opt_{int(time.time())}_{symbol.replace('-', '_')}"
            
            # Crear configuración del bot
            bot_config = {
                "id": bot_id,
                "name": f"Opt_{best_strategy}_{symbol}",
                "symbol": symbol,
                "interval": interval,
                "strategy": best_strategy,
                "initial_balance": 1000.0,
                "current_balance": 1000.0,
                "created_at": datetime.now().isoformat(),
                "status": "created",
                "trades": [],
                "performance": {},
                "optimization_results": optimization_results,
                "params": optimization_results.get("best_params", {}),
                "last_update": datetime.now().isoformat()
            }
            
            # Guardar configuración
            self._save_bot_config(bot_id, bot_config)
            
            logger.info(f"Created optimized bot {bot_id} with strategy {best_strategy}")
            
            return bot_id
        else:
            # Si no hay grid de parámetros, crear bot normal
            return self.create_simulation_bot(symbol, interval, strategy_name=best_strategy)
    
    def _get_param_grid_for_strategy(self, strategy_name: str) -> Dict:
        """
        Obtiene grid de parámetros para optimización según estrategia
        
        Args:
            strategy_name: Nombre de estrategia
            
        Returns:
            Dict: Grid de parámetros
        """
        param_grids = {
            "SMA Crossover": {
                "short_period": [5, 10, 15, 20],
                "long_period": [30, 40, 50, 60]
            },
            "RSI Strategy": {
                "period": [7, 14, 21],
                "overbought": [65, 70, 75, 80],
                "oversold": [20, 25, 30, 35]
            },
            "MACD Strategy": {
                "fast_period": [8, 12, 16],
                "slow_period": [21, 26, 30],
                "signal_period": [7, 9, 12]
            },
            "Bollinger Bands": {
                "period": [15, 20, 25],
                "std_dev": [1.5, 2.0, 2.5]
            },
            "Mean Reversion": {
                "lookback": [20, 30, 40],
                "std_dev": [1.5, 2.0, 2.5, 3.0]
            }
        }
        
        return param_grids.get(strategy_name, {})
    
    def _save_bot_config(self, bot_id: str, config: Dict):
        """
        Guarda configuración de bot
        
        Args:
            bot_id: ID del bot
            config: Configuración
        """
        bot_file = f"{self.db_path}/bots/{bot_id}.json"
        
        with open(bot_file, "w") as f:
            json.dump(config, f, indent=2, default=str)
    
    def _load_bot_config(self, bot_id: str) -> Optional[Dict]:
        """
        Carga configuración de bot
        
        Args:
            bot_id: ID del bot
            
        Returns:
            Optional[Dict]: Configuración o None si no existe
        """
        bot_file = f"{self.db_path}/bots/{bot_id}.json"
        
        if not os.path.exists(bot_file):
            return None
        
        try:
            with open(bot_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading bot config: {e}")
            return None
    
    def get_bot_status(self, bot_id: str) -> Dict:
        """
        Obtiene estado de un bot
        
        Args:
            bot_id: ID del bot
            
        Returns:
            Dict: Estado del bot
        """
        config = self._load_bot_config(bot_id)
        if config is None:
            return {"error": f"Bot {bot_id} not found"}
        
        return {
            "id": bot_id,
            "name": config.get("name", ""),
            "symbol": config.get("symbol", ""),
            "strategy": config.get("strategy", ""),
            "status": config.get("status", ""),
            "initial_balance": config.get("initial_balance", 0.0),
            "current_balance": config.get("current_balance", 0.0),
            "return_pct": (config.get("current_balance", 0.0) / config.get("initial_balance", 1.0) - 1) * 100,
            "trades_count": len(config.get("trades", [])),
            "last_update": config.get("last_update", "")
        }
    
    def get_all_bots(self) -> List[Dict]:
        """
        Obtiene todos los bots
        
        Returns:
            List[Dict]: Lista de bots
        """
        bots = []
        
        # Buscar archivos de bots
        bot_files = os.listdir(f"{self.db_path}/bots")
        
        for file in bot_files:
            if file.endswith(".json"):
                bot_id = file.replace(".json", "")
                bot_status = self.get_bot_status(bot_id)
                
                if "error" not in bot_status:
                    bots.append(bot_status)
        
        return bots
    
    def train_ml_model(self, symbol: str, interval: str, days: int = 90, 
                     model_type: str = "random_forest") -> Dict:
        """
        Entrena un modelo de Machine Learning
        
        Args:
            symbol: Par de trading
            interval: Intervalo de tiempo
            days: Días para entrenamiento
            model_type: Tipo de modelo ('random_forest', 'gradient_boosting', 'mlp')
            
        Returns:
            Dict: Resultados del entrenamiento
        """
        # Obtener datos
        data = update_market_data(symbol, interval)
        
        # Limitar a los días solicitados
        if days > 0:
            start_date = data.index[-1] - pd.Timedelta(days=days)
            data = data[data.index >= start_date]
        
        # Crear modelo ML
        ml_strategy = MLStrategy(model_type=model_type)
        
        # Entrenar modelo
        performance = ml_strategy.train(data)
        
        # Guardar modelo
        model_path = f"{self.db_path}/models/{symbol}_{interval}_{model_type}"
        ml_strategy.save_model(model_path)
        
        return {
            "symbol": symbol,
            "interval": interval,
            "model_type": model_type,
            "performance": performance,
            "model_path": model_path
        }
    
    def create_ml_bot(self, symbol: str, interval: str, 
                    model_type: str = "random_forest") -> str:
        """
        Crea un bot basado en Machine Learning
        
        Args:
            symbol: Par de trading
            interval: Intervalo de tiempo
            model_type: Tipo de modelo
            
        Returns:
            str: ID del bot creado
        """
        # Entrenar modelo
        training_result = self.train_ml_model(symbol, interval, model_type=model_type)
        
        # Generar ID único
        bot_id = f"ml_{int(time.time())}_{symbol.replace('-', '_')}"
        
        # Crear configuración del bot
        bot_config = {
            "id": bot_id,
            "name": f"ML_{model_type}_{symbol}",
            "symbol": symbol,
            "interval": interval,
            "strategy": f"ml_{model_type}",
            "initial_balance": 1000.0,
            "current_balance": 1000.0,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "trades": [],
            "performance": {},
            "model_type": model_type,
            "model_path": training_result["model_path"],
            "training_performance": training_result["performance"],
            "last_update": datetime.now().isoformat()
        }
        
        # Guardar configuración
        self._save_bot_config(bot_id, bot_config)
        
        logger.info(f"Created ML bot {bot_id} with model type {model_type}")
        
        return bot_id
    
    def run_learning_cycle(self, symbols: List[str], interval: str = "1h") -> Dict:
        """
        Ejecuta un ciclo completo de aprendizaje
        
        Args:
            symbols: Lista de pares de trading
            interval: Intervalo de tiempo
            
        Returns:
            Dict: Resultados del ciclo de aprendizaje
        """
        results = {
            "symbols": symbols,
            "interval": interval,
            "started_at": datetime.now().isoformat(),
            "market_analysis": {},
            "strategy_results": {},
            "ml_results": {},
            "created_bots": []
        }
        
        for symbol in symbols:
            logger.info(f"Running learning cycle for {symbol}")
            
            # 1. Analizar mercado
            analysis = self.analyze_market_conditions(symbol, interval)
            results["market_analysis"][symbol] = analysis
            
            # 2. Ejecutar backtesting de estrategias
            strategy_results = self.backtester.run_all_strategies(symbol, interval, days=60)
            best_strategies = self.backtester.get_best_strategies(symbol, top_n=3)
            results["strategy_results"][symbol] = best_strategies
            
            # 3. Crear bot simulado con la mejor estrategia
            if best_strategies:
                best_strategy = best_strategies[0]["name"]
                bot_id = self.create_simulation_bot(symbol, interval, strategy_name=best_strategy)
                if bot_id:
                    # Iniciar simulación
                    sim_result = self.start_simulation_bot(bot_id, days_to_simulate=30)
                    results["created_bots"].append({
                        "bot_id": bot_id,
                        "type": "simulation",
                        "strategy": best_strategy,
                        "results": sim_result
                    })
            
            # 4. Crear bot optimizado
            opt_bot_id = self.create_optimized_bot(symbol, interval, days=60)
            if opt_bot_id:
                # Iniciar simulación
                opt_sim_result = self.start_simulation_bot(opt_bot_id, days_to_simulate=30)
                results["created_bots"].append({
                    "bot_id": opt_bot_id,
                    "type": "optimized",
                    "results": opt_sim_result
                })
            
            # 5. Entrenar modelo ML
            ml_result = self.train_ml_model(symbol, interval, days=90)
            results["ml_results"][symbol] = ml_result
            
            # 6. Crear bot ML
            ml_bot_id = self.create_ml_bot(symbol, interval)
            if ml_bot_id:
                # Iniciar simulación
                ml_sim_result = self.start_simulation_bot(ml_bot_id, days_to_simulate=30)
                results["created_bots"].append({
                    "bot_id": ml_bot_id,
                    "type": "ml",
                    "results": ml_sim_result
                })
        
        # Actualizar estado
        results["completed_at"] = datetime.now().isoformat()
        
        # Guardar resultados
        cycle_file = f"{self.db_path}/data/learning_cycle_{int(time.time())}.json"
        with open(cycle_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Learning cycle completed and saved to {cycle_file}")
        
        return results
    
    def get_learning_summary(self) -> Dict:
        """
        Obtiene resumen del sistema de aprendizaje
        
        Returns:
            Dict: Resumen de aprendizaje
        """
        # Obtener bots
        bots = self.get_all_bots()
        
        # Clasificar por tipo
        bot_types = {}
        for bot in bots:
            bot_id = bot["id"]
            
            if bot_id.startswith("sim_"):
                bot_type = "simulation"
            elif bot_id.startswith("opt_"):
                bot_type = "optimized"
            elif bot_id.startswith("ml_"):
                bot_type = "ml"
            else:
                bot_type = "unknown"
            
            if bot_type not in bot_types:
                bot_types[bot_type] = []
            
            bot_types[bot_type].append(bot)
        
        # Calcular rendimiento promedio por tipo
        performance_by_type = {}
        for bot_type, type_bots in bot_types.items():
            if not type_bots:
                continue
            
            avg_return = sum(bot.get("return_pct", 0) for bot in type_bots) / len(type_bots)
            best_bot = max(type_bots, key=lambda x: x.get("return_pct", 0))
            
            performance_by_type[bot_type] = {
                "count": len(type_bots),
                "avg_return": avg_return,
                "best_bot": {
                    "id": best_bot["id"],
                    "name": best_bot["name"],
                    "return_pct": best_bot["return_pct"]
                }
            }
        
        # Obtener insights
        insights = self.get_learning_insights()
        
        return {
            "total_bots": len(bots),
            "bots_by_type": {k: len(v) for k, v in bot_types.items()},
            "performance_by_type": performance_by_type,
            "learning_insights": insights
        }

# Funciones de utilidad para usar en el CLI

def run_multi_strategy_backtest(symbol: str, interval: str, days: int = 90) -> Dict:
    """
    Ejecuta backtesting de todas las estrategias disponibles
    
    Args:
        symbol: Par de trading
        interval: Intervalo de tiempo
        days: Días de histórico a utilizar
        
    Returns:
        Dict: Resultados de backtesting
    """
    backtester = MultiStrategyBacktester()
    backtester.run_all_strategies(symbol, interval, days)
    
    results = backtester.get_best_strategies(symbol)
    trend_report = backtester.generate_trend_report()
    
    return {
        "symbol": symbol,
        "interval": interval,
        "days": days,
        "best_strategies": results,
        "trend_report": trend_report
    }

def optimize_strategy_params(strategy_name: str, symbol: str, interval: str, days: int = 90) -> Dict:
    """
    Optimiza parámetros de una estrategia
    
    Args:
        strategy_name: Nombre de la estrategia
        symbol: Par de trading
        interval: Intervalo de tiempo
        days: Días de histórico a utilizar
        
    Returns:
        Dict: Resultados de optimización
    """
    backtester = MultiStrategyBacktester()
    
    # Definir grid de parámetros según estrategia
    param_grid = {
        "SMA Crossover": {
            "short_period": [5, 10, 15, 20],
            "long_period": [30, 40, 50, 60]
        },
        "RSI Strategy": {
            "period": [7, 14, 21],
            "overbought": [65, 70, 75, 80],
            "oversold": [20, 25, 30, 35]
        },
        "MACD Strategy": {
            "fast_period": [8, 12, 16],
            "slow_period": [21, 26, 30],
            "signal_period": [7, 9, 12]
        },
        "Bollinger Bands": {
            "period": [15, 20, 25],
            "std_dev": [1.5, 2.0, 2.5]
        },
        "Mean Reversion": {
            "lookback": [20, 30, 40],
            "std_dev": [1.5, 2.0, 2.5, 3.0]
        }
    }
    
    if strategy_name not in param_grid:
        return {
            "error": f"No parameter grid available for strategy {strategy_name}"
        }
    
    result = backtester.optimize_strategy(strategy_name, symbol, interval, param_grid[strategy_name], days)
    
    return result

def analyze_current_market(symbol: str, interval: str) -> Dict:
    """
    Analiza condiciones actuales del mercado
    
    Args:
        symbol: Par de trading
        interval: Intervalo de tiempo
        
    Returns:
        Dict: Análisis de condiciones de mercado
    """
    learning_system = AutomatedLearningSystem()
    analysis = learning_system.analyze_market_conditions(symbol, interval)
    
    return analysis

def create_auto_learning_bots(symbols: List[str], interval: str = "1h") -> Dict:
    """
    Crea bots de aprendizaje automático para múltiples símbolos
    
    Args:
        symbols: Lista de pares de trading
        interval: Intervalo de tiempo
        
    Returns:
        Dict: Resultados de creación de bots
    """
    learning_system = AutomatedLearningSystem()
    results = learning_system.run_learning_cycle(symbols, interval)
    
    return results

def get_learning_system_status() -> Dict:
    """
    Obtiene estado del sistema de aprendizaje
    
    Returns:
        Dict: Estado del sistema
    """
    learning_system = AutomatedLearningSystem()
    summary = learning_system.get_learning_summary()
    
    return summary