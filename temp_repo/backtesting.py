"""
Módulo de backtesting y optimización para el bot de trading
Permite probar estrategias en datos históricos y optimizar parámetros
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import json
import os
import ccxt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

# Importar módulos propios
from strategies import (
    TechnicalIndicators, ClassicStrategy, StatisticalStrategy, 
    MachineLearningStrategy, AdaptiveStrategy, RiskManagement
)
from indicator_weighting import IndicatorPerformanceTracker, get_weighted_decision

# Configurar logging
logger = logging.getLogger("Backtesting")

class TradingSimulator:
    """
    Simulador de trading para backtesting de estrategias
    """
    
    def __init__(self, initial_balance: float = 10000.0, 
                leverage: float = 1.0, commission: float = 0.001):
        """
        Inicializa el simulador de trading
        
        Args:
            initial_balance: Balance inicial para la simulación
            leverage: Apalancamiento (1.0 = sin apalancamiento)
            commission: Comisión por operación (0.001 = 0.1%)
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.leverage = leverage
        self.commission = commission
        
        # Posición actual
        self.position = {
            'type': None,  # 'long', 'short' o None
            'size': 0.0,
            'entry_price': 0.0,
            'entry_time': None
        }
        
        # Historial de operaciones
        self.trades = []
        
        # Métricas
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'cagr': 0.0,
            'volatility': 0.0
        }
        
        # Historial de equity
        self.equity_curve = []
    
    def reset(self):
        """Reinicia el simulador para una nueva simulación"""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = {
            'type': None,
            'size': 0.0,
            'entry_price': 0.0,
            'entry_time': None
        }
        self.trades = []
        self.equity_curve = []
    
    def open_position(self, position_type: str, price: float, size: float, 
                     timestamp: pd.Timestamp, reason: str = ""):
        """
        Abre una posición
        
        Args:
            position_type: Tipo de posición ('long' o 'short')
            price: Precio de entrada
            size: Tamaño de la posición en unidades
            timestamp: Timestamp de la entrada
            reason: Razón de la entrada
        """
        # Verificar si ya hay una posición abierta
        if self.position['type'] is not None:
            logger.warning(f"Position is already open: {self.position['type']}")
            return
        
        # Verificar que el tamaño sea válido
        if size <= 0:
            logger.warning(f"Invalid position size: {size}")
            return
        
        # Calcular valor de la posición
        position_value = price * size
        
        # Verificar si hay suficiente balance
        required_margin = position_value / self.leverage
        if required_margin > self.balance:
            logger.warning(f"Insufficient balance: {self.balance} < {required_margin}")
            return
        
        # Aplicar comisión
        commission_cost = position_value * self.commission
        self.balance -= commission_cost
        
        # Registrar posición
        self.position = {
            'type': position_type,
            'size': size,
            'entry_price': price,
            'entry_time': timestamp,
            'entry_reason': reason
        }
        
        logger.info(f"Opened {position_type} position: {size} units at {price}")
    
    def close_position(self, price: float, timestamp: pd.Timestamp, reason: str = ""):
        """
        Cierra la posición actual
        
        Args:
            price: Precio de salida
            timestamp: Timestamp de la salida
            reason: Razón de la salida
        """
        # Verificar si hay una posición abierta
        if self.position['type'] is None:
            logger.warning("No position to close")
            return
        
        # Calcular P&L
        if self.position['type'] == 'long':
            pnl = (price - self.position['entry_price']) * self.position['size']
        else:  # short
            pnl = (self.position['entry_price'] - price) * self.position['size']
        
        # Aplicar comisión
        position_value = price * self.position['size']
        commission_cost = position_value * self.commission
        net_pnl = pnl - commission_cost
        
        # Actualizar balance
        self.balance += net_pnl
        
        # Registrar operación
        trade = {
            'type': self.position['type'],
            'entry_price': self.position['entry_price'],
            'entry_time': self.position['entry_time'],
            'exit_price': price,
            'exit_time': timestamp,
            'size': self.position['size'],
            'pnl': pnl,
            'net_pnl': net_pnl,
            'commission': commission_cost,
            'duration': timestamp - self.position['entry_time'],
            'entry_reason': self.position['entry_reason'],
            'exit_reason': reason
        }
        
        self.trades.append(trade)
        
        # Actualizar equity
        self.equity = self.balance
        
        # Limpiar posición
        self.position = {
            'type': None,
            'size': 0.0,
            'entry_price': 0.0,
            'entry_time': None
        }
        
        logger.info(f"Closed position at {price}, PnL: {net_pnl:.2f}")
    
    def update_equity(self, price: float, timestamp: pd.Timestamp):
        """
        Actualiza el equity con el precio actual
        
        Args:
            price: Precio actual
            timestamp: Timestamp actual
        """
        # Calcular equity actual
        equity = self.balance
        
        # Añadir valor de posición abierta si existe
        if self.position['type'] is not None:
            if self.position['type'] == 'long':
                unrealized_pnl = (price - self.position['entry_price']) * self.position['size']
            else:  # short
                unrealized_pnl = (self.position['entry_price'] - price) * self.position['size']
            
            # Restar comisión estimada
            position_value = price * self.position['size']
            commission_cost = position_value * self.commission
            
            equity += unrealized_pnl - commission_cost
        
        self.equity = equity
        
        # Registrar punto en equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'balance': self.balance,
            'price': price
        })
    
    def calculate_metrics(self):
        """Calcula métricas de rendimiento de la estrategia"""
        if not self.trades:
            logger.warning("No trades to calculate metrics")
            return
        
        # Métricas básicas
        self.metrics['total_trades'] = len(self.trades)
        
        # Operaciones ganadoras y perdedoras
        winning_trades = [t for t in self.trades if t['net_pnl'] > 0]
        losing_trades = [t for t in self.trades if t['net_pnl'] <= 0]
        
        self.metrics['winning_trades'] = len(winning_trades)
        self.metrics['losing_trades'] = len(losing_trades)
        
        # Tasa de ganancia
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']
        
        # Ganancias y pérdidas promedio
        if winning_trades:
            self.metrics['avg_profit'] = sum(t['net_pnl'] for t in winning_trades) / len(winning_trades)
        
        if losing_trades:
            self.metrics['avg_loss'] = abs(sum(t['net_pnl'] for t in losing_trades) / len(losing_trades))
        
        # Profit factor
        if self.metrics['avg_loss'] > 0:
            self.metrics['profit_factor'] = self.metrics['avg_profit'] / self.metrics['avg_loss']
        
        # Máximo drawdown
        if self.equity_curve:
            # Convertir a DataFrame para facilitar cálculos
            equity_df = pd.DataFrame([
                {'timestamp': e['timestamp'], 'equity': e['equity']} 
                for e in self.equity_curve
            ])
            
            # Calcular drawdown
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = equity_df['equity'] - equity_df['peak']
            equity_df['drawdown_pct'] = equity_df['drawdown'] / equity_df['peak']
            
            self.metrics['max_drawdown'] = abs(equity_df['drawdown'].min())
            self.metrics['max_drawdown_pct'] = abs(equity_df['drawdown_pct'].min())
        
        # Retornos diarios para Sharpe/Sortino
        if len(self.equity_curve) > 1:
            equity_values = [e['equity'] for e in self.equity_curve]
            returns = pd.Series(equity_values).pct_change().dropna()
            
            # Volatilidad
            self.metrics['volatility'] = returns.std() * np.sqrt(252)  # Anualizada
            
            # Sharpe Ratio (asumiendo retorno libre de riesgo 0)
            avg_return = returns.mean()
            if returns.std() > 0:
                self.metrics['sharpe_ratio'] = (avg_return / returns.std()) * np.sqrt(252)
            
            # Sortino Ratio (solo considera retornos negativos)
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0 and negative_returns.std() > 0:
                self.metrics['sortino_ratio'] = (avg_return / negative_returns.std()) * np.sqrt(252)
        
        # CAGR (Compound Annual Growth Rate)
        if len(self.equity_curve) > 1:
            start_date = self.equity_curve[0]['timestamp']
            end_date = self.equity_curve[-1]['timestamp']
            years = (end_date - start_date).days / 365.25
            
            if years > 0:
                final_equity = self.equity_curve[-1]['equity']
                self.metrics['cagr'] = (final_equity / self.initial_balance) ** (1 / years) - 1
    
    def get_metrics_summary(self) -> Dict:
        """
        Obtiene un resumen de las métricas del backtest
        
        Returns:
            Dict: Resumen de métricas
        """
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'return_pct': (self.balance / self.initial_balance - 1) * 100,
            'total_trades': self.metrics['total_trades'],
            'win_rate': self.metrics['win_rate'] * 100,
            'profit_factor': self.metrics['profit_factor'],
            'max_drawdown_pct': self.metrics['max_drawdown_pct'] * 100,
            'sharpe_ratio': self.metrics['sharpe_ratio'],
            'sortino_ratio': self.metrics['sortino_ratio'],
            'cagr': self.metrics['cagr'] * 100,
            'volatility': self.metrics['volatility'] * 100
        }
    
    def plot_equity_curve(self):
        """Grafica la curva de equity"""
        if not self.equity_curve:
            logger.warning("No equity data to plot")
            return
        
        # Preparar datos
        timestamps = [e['timestamp'] for e in self.equity_curve]
        equity_values = [e['equity'] for e in self.equity_curve]
        prices = [e['price'] for e in self.equity_curve]
        
        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Graficar equity curve
        ax1.plot(timestamps, equity_values, label='Equity', color='blue')
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Equity')
        ax1.legend()
        ax1.grid(True)
        
        # Graficar precio
        ax2.plot(timestamps, prices, label='Price', color='green')
        ax2.set_title('Price')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price')
        ax2.legend()
        ax2.grid(True)
        
        # Marcar operaciones en el gráfico de precio
        for trade in self.trades:
            if trade['type'] == 'long':
                # Entrada long: flecha verde hacia arriba
                ax2.scatter(trade['entry_time'], trade['entry_price'], 
                         marker='^', color='green', s=100)
                # Salida long: flecha roja hacia abajo
                ax2.scatter(trade['exit_time'], trade['exit_price'],
                         marker='v', color='red', s=100)
            else:  # short
                # Entrada short: flecha roja hacia abajo
                ax2.scatter(trade['entry_time'], trade['entry_price'], 
                         marker='v', color='red', s=100)
                # Salida short: flecha verde hacia arriba
                ax2.scatter(trade['exit_time'], trade['exit_price'],
                         marker='^', color='green', s=100)
        
        plt.tight_layout()
        plt.show()


class BacktestEngine:
    """
    Motor de backtesting para probar estrategias
    """
    
    def __init__(self, exchange_id: str = 'okx', symbol: str = 'SOL/USDT', 
                timeframe: str = '15m', initial_balance: float = 10000.0):
        """
        Inicializa el motor de backtesting
        
        Args:
            exchange_id: ID del exchange (ccxt)
            symbol: Par de trading
            timeframe: Intervalo de tiempo
            initial_balance: Balance inicial para la simulación
        """
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        
        # Instanciar simulador
        self.simulator = TradingSimulator(initial_balance=initial_balance)
        
        # Datos históricos
        self.data = None
        
        # Tracker de rendimiento de indicadores
        self.indicator_tracker = IndicatorPerformanceTracker(symbol.replace('/', '-'))
    
    def load_data_from_exchange(self, start_date: Optional[str] = None, 
                              end_date: Optional[str] = None,
                              limit: int = 1000) -> pd.DataFrame:
        """
        Carga datos históricos desde un exchange
        
        Args:
            start_date: Fecha de inicio (formato: 'YYYY-MM-DD')
            end_date: Fecha de fin (formato: 'YYYY-MM-DD')
            limit: Límite de velas a cargar
            
        Returns:
            pd.DataFrame: DataFrame con datos históricos
        """
        try:
            # Instanciar exchange
            exchange = getattr(ccxt, self.exchange_id)()
            
            # Convertir fechas a timestamps si se proporcionan
            since = None
            if start_date:
                since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            
            # Cargar datos
            ohlcv = exchange.fetch_ohlcv(self.symbol, self.timeframe, since, limit)
            
            # Convertir a DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Filtrar por fecha de fin si se proporciona
            if end_date:
                end_timestamp = datetime.strptime(end_date, '%Y-%m-%d')
                df = df[df.index <= end_timestamp]
            
            logger.info(f"Loaded {len(df)} candles from {self.exchange_id} for {self.symbol}")
            
            self.data = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from exchange: {e}")
            return pd.DataFrame()
    
    def load_data_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Carga datos históricos desde un archivo CSV
        
        Args:
            file_path: Ruta al archivo CSV
            
        Returns:
            pd.DataFrame: DataFrame con datos históricos
        """
        try:
            df = pd.read_csv(file_path)
            
            # Verificar columnas requeridas
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    return pd.DataFrame()
            
            # Convertir timestamp a datetime
            if 'timestamp' in df.columns:
                if df['timestamp'].dtype == 'object':
                    # Intentar formato ISO
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                else:
                    # Asumir timestamp en ms
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Loaded {len(df)} candles from {file_path}")
            
            self.data = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from CSV: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, strategy_fn: Callable) -> Dict:
        """
        Ejecuta un backtest con la estrategia proporcionada
        
        Args:
            strategy_fn: Función de estrategia que recibe un DataFrame y retorna señales
            
        Returns:
            Dict: Resultados del backtest
        """
        if self.data is None or self.data.empty:
            logger.error("No data loaded for backtest")
            return {}
        
        # Reiniciar simulador
        self.simulator.reset()
        
        # Ejecutar estrategia
        try:
            signals = strategy_fn(self.data.copy())
            
            # Iterar por los datos y operar según señales
            for i in range(1, len(self.data)):
                current_row = self.data.iloc[i]
                current_timestamp = self.data.index[i]
                current_price = current_row['close']
                
                # Verificar señal
                if i < len(signals):
                    signal = signals.iloc[i]
                    
                    # Abrir posición si hay señal y no hay posición abierta
                    if signal == 1 and self.simulator.position['type'] is None:
                        # Calcular tamaño de posición (1% del balance)
                        position_size = self.simulator.balance * 0.01 / current_price
                        
                        # Abrir posición long
                        self.simulator.open_position(
                            'long', current_price, position_size, current_timestamp, 
                            reason=f"Signal: {signal}"
                        )
                    
                    elif signal == -1 and self.simulator.position['type'] is None:
                        # Calcular tamaño de posición (1% del balance)
                        position_size = self.simulator.balance * 0.01 / current_price
                        
                        # Abrir posición short
                        self.simulator.open_position(
                            'short', current_price, position_size, current_timestamp,
                            reason=f"Signal: {signal}"
                        )
                    
                    # Cerrar posición si hay señal contraria
                    elif signal == -1 and self.simulator.position['type'] == 'long':
                        self.simulator.close_position(
                            current_price, current_timestamp,
                            reason=f"Exit signal: {signal}"
                        )
                    
                    elif signal == 1 and self.simulator.position['type'] == 'short':
                        self.simulator.close_position(
                            current_price, current_timestamp,
                            reason=f"Exit signal: {signal}"
                        )
                
                # Actualizar equity
                self.simulator.update_equity(current_price, current_timestamp)
        
        except Exception as e:
            logger.error(f"Error during backtest: {e}")
            return {}
        
        # Cerrar cualquier posición abierta al final
        if self.simulator.position['type'] is not None:
            last_timestamp = self.data.index[-1]
            last_price = self.data['close'].iloc[-1]
            
            self.simulator.close_position(
                last_price, last_timestamp,
                reason="End of backtest"
            )
        
        # Calcular métricas
        self.simulator.calculate_metrics()
        
        # Retornar resultados
        return self.simulator.get_metrics_summary()
    
    def run_backtest_with_indicator_weights(self) -> Dict:
        """
        Ejecuta un backtest con ponderación adaptativa de indicadores
        
        Returns:
            Dict: Resultados del backtest
        """
        if self.data is None or self.data.empty:
            logger.error("No data loaded for backtest")
            return {}
        
        # Reiniciar simulador
        self.simulator.reset()
        
        # Preparar datos con indicadores
        df = self.data.copy()
        
        # Calcular indicadores
        df['rsi'] = TechnicalIndicators.rsi(df['close'])
        
        macd_line, signal_line, hist = TechnicalIndicators.macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = hist
        
        middle, upper, lower = TechnicalIndicators.bollinger_bands(df['close'])
        df['bb_middle'] = middle
        df['bb_upper'] = upper
        df['bb_lower'] = lower
        
        df['sma_20'] = TechnicalIndicators.sma(df['close'], 20)
        df['sma_50'] = TechnicalIndicators.sma(df['close'], 50)
        
        # Eliminar filas con NaN
        df.dropna(inplace=True)
        
        # Determinar condición de mercado por período
        market_conditions = []
        
        # Simplificado: usar ATR para determinar volatilidad
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        df['atr_pct'] = df['atr'] / df['close']  # ATR como % del precio
        
        # Calcular tendencia
        df['trend'] = 0
        df.loc[df['sma_20'] > df['sma_50'], 'trend'] = 1  # Tendencia alcista
        df.loc[df['sma_20'] < df['sma_50'], 'trend'] = -1  # Tendencia bajista
        
        # Recorrer períodos para determinar condición
        for i in range(len(df)):
            # Simplificado: usar una ventana de 20 períodos
            start_idx = max(0, i - 20)
            window = df.iloc[start_idx:i+1]
            
            # Volatilidad promedio
            avg_volatility = window['atr_pct'].mean()
            current_volatility = df['atr_pct'].iloc[i]
            
            # Tendencia
            trend = df['trend'].iloc[i]
            
            # Determinar condición de mercado
            if current_volatility > avg_volatility * 2:
                condition = 'extreme_volatility'
            elif trend > 0 and current_volatility > avg_volatility * 1.5:
                condition = 'uptrend_strong'
            elif trend > 0:
                condition = 'uptrend_weak'
            elif trend < 0 and current_volatility > avg_volatility * 1.5:
                condition = 'downtrend_strong'
            elif trend < 0:
                condition = 'downtrend_weak'
            elif current_volatility < avg_volatility * 0.5:
                condition = 'range_low_vol'
            else:
                condition = 'range_high_vol'
            
            market_conditions.append(condition)
        
        df['market_condition'] = market_conditions
        
        # Generar señales de indicadores individuales
        df['rsi_signal'] = 0
        df.loc[df['rsi'] < 30, 'rsi_signal'] = 1
        df.loc[df['rsi'] > 70, 'rsi_signal'] = -1
        
        df['macd_signal_value'] = 0
        df.loc[df['macd'] > df['macd_signal'], 'macd_signal_value'] = 1
        df.loc[df['macd'] < df['macd_signal'], 'macd_signal_value'] = -1
        
        df['bb_signal'] = 0
        df.loc[df['close'] < df['bb_lower'], 'bb_signal'] = 1
        df.loc[df['close'] > df['bb_upper'], 'bb_signal'] = -1
        
        df['ma_crossover_signal'] = 0
        df.loc[df['sma_20'] > df['sma_50'], 'ma_crossover_signal'] = 1
        df.loc[df['sma_20'] < df['sma_50'], 'ma_crossover_signal'] = -1
        
        # Ejecutar backtest
        trade_results = []
        open_position_info = None
        
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            current_timestamp = df.index[i]
            current_price = current_row['close']
            market_condition = current_row['market_condition']
            
            # Establecer contexto actual
            self.indicator_tracker.set_current_context(market_condition, self.timeframe)
            
            # Obtener pesos actuales
            weights = self.indicator_tracker.get_all_weights()
            
            # Recopilar señales actuales
            signals = {
                'rsi': current_row['rsi_signal'],
                'macd': current_row['macd_signal_value'],
                'bollinger_bands': current_row['bb_signal'],
                'sma_crossover': current_row['ma_crossover_signal']
            }
            
            # Obtener decisión ponderada
            decision, confidence, details = get_weighted_decision(signals, weights)
            
            # Trading basado en la decisión
            if decision == 1 and self.simulator.position['type'] is None and confidence > 0.3:
                # Calcular tamaño de posición (1% del balance)
                position_size = self.simulator.balance * 0.01 / current_price
                
                # Abrir posición long
                self.simulator.open_position(
                    'long', current_price, position_size, current_timestamp, 
                    reason=f"Weighted signal: {confidence:.2f}"
                )
                
                # Guardar información para evaluar después
                open_position_info = {
                    'entry_time': current_timestamp,
                    'entry_price': current_price,
                    'signals': signals.copy(),
                    'market_condition': market_condition,
                    'time_interval': self.timeframe
                }
            
            elif decision == -1 and self.simulator.position['type'] is None and confidence > 0.3:
                # Calcular tamaño de posición (1% del balance)
                position_size = self.simulator.balance * 0.01 / current_price
                
                # Abrir posición short
                self.simulator.open_position(
                    'short', current_price, position_size, current_timestamp,
                    reason=f"Weighted signal: {confidence:.2f}"
                )
                
                # Guardar información para evaluar después
                open_position_info = {
                    'entry_time': current_timestamp,
                    'entry_price': current_price,
                    'signals': signals.copy(),
                    'market_condition': market_condition,
                    'time_interval': self.timeframe
                }
            
            # Cerrar posición si hay señal contraria o baja confianza
            elif (decision == -1 and self.simulator.position['type'] == 'long' and confidence > 0.2) or \
                 (self.simulator.position['type'] == 'long' and confidence < 0.1):
                
                self.simulator.close_position(
                    current_price, current_timestamp,
                    reason=f"Exit signal: {decision} (confidence: {confidence:.2f})"
                )
                
                # Evaluar resultado
                if open_position_info:
                    profit_loss = current_price - open_position_info['entry_price']
                    is_correct = profit_loss > 0
                    
                    # Registrar resultado para cada indicador
                    for indicator, signal in open_position_info['signals'].items():
                        if signal != 0:  # Solo evaluar indicadores que dieron señal
                            # Determinar si el indicador acertó
                            indicator_correct = (signal == 1 and profit_loss > 0) or \
                                              (signal == -1 and profit_loss < 0)
                            
                            # Registrar resultado en el tracker
                            self.indicator_tracker.record_signal_result(
                                indicator, indicator_correct, profit_loss,
                                open_position_info['market_condition'], 
                                open_position_info['time_interval']
                            )
                            
                            # Guardar para análisis posterior
                            trade_results.append({
                                'indicator': indicator,
                                'signal': signal,
                                'profit_loss': profit_loss,
                                'is_correct': indicator_correct,
                                'market_condition': open_position_info['market_condition'],
                                'time_interval': open_position_info['time_interval']
                            })
                    
                    open_position_info = None
            
            elif (decision == 1 and self.simulator.position['type'] == 'short' and confidence > 0.2) or \
                 (self.simulator.position['type'] == 'short' and confidence < 0.1):
                
                self.simulator.close_position(
                    current_price, current_timestamp,
                    reason=f"Exit signal: {decision} (confidence: {confidence:.2f})"
                )
                
                # Evaluar resultado
                if open_position_info:
                    profit_loss = open_position_info['entry_price'] - current_price
                    is_correct = profit_loss > 0
                    
                    # Registrar resultado para cada indicador
                    for indicator, signal in open_position_info['signals'].items():
                        if signal != 0:  # Solo evaluar indicadores que dieron señal
                            # Determinar si el indicador acertó (en short la lógica es inversa)
                            indicator_correct = (signal == -1 and profit_loss > 0) or \
                                              (signal == 1 and profit_loss < 0)
                            
                            # Registrar resultado en el tracker
                            self.indicator_tracker.record_signal_result(
                                indicator, indicator_correct, profit_loss,
                                open_position_info['market_condition'], 
                                open_position_info['time_interval']
                            )
                            
                            # Guardar para análisis posterior
                            trade_results.append({
                                'indicator': indicator,
                                'signal': signal,
                                'profit_loss': profit_loss,
                                'is_correct': indicator_correct,
                                'market_condition': open_position_info['market_condition'],
                                'time_interval': open_position_info['time_interval']
                            })
                    
                    open_position_info = None
            
            # Actualizar equity
            self.simulator.update_equity(current_price, current_timestamp)
        
        # Cerrar cualquier posición abierta al final
        if self.simulator.position['type'] is not None:
            last_timestamp = df.index[-1]
            last_price = df['close'].iloc[-1]
            
            self.simulator.close_position(
                last_price, last_timestamp,
                reason="End of backtest"
            )
            
            # Evaluar la última operación si hay información abierta
            if open_position_info:
                if self.simulator.position['type'] == 'long':
                    profit_loss = last_price - open_position_info['entry_price']
                else:
                    profit_loss = open_position_info['entry_price'] - last_price
                
                is_correct = profit_loss > 0
                
                # Registrar resultado para cada indicador
                for indicator, signal in open_position_info['signals'].items():
                    if signal != 0:
                        # Determinar si el indicador acertó
                        if self.simulator.position['type'] == 'long':
                            indicator_correct = (signal == 1 and profit_loss > 0) or \
                                              (signal == -1 and profit_loss < 0)
                        else:
                            indicator_correct = (signal == -1 and profit_loss > 0) or \
                                              (signal == 1 and profit_loss < 0)
                        
                        # Registrar resultado
                        self.indicator_tracker.record_signal_result(
                            indicator, indicator_correct, profit_loss,
                            open_position_info['market_condition'], 
                            open_position_info['time_interval']
                        )
                        
                        trade_results.append({
                            'indicator': indicator,
                            'signal': signal,
                            'profit_loss': profit_loss,
                            'is_correct': indicator_correct,
                            'market_condition': open_position_info['market_condition'],
                            'time_interval': open_position_info['time_interval']
                        })
        
        # Calcular métricas
        self.simulator.calculate_metrics()
        
        # Guardar resultados de rendimiento de indicadores
        self.indicator_tracker.save_performance_data()
        
        # Retornar resultados
        return {
            'metrics': self.simulator.get_metrics_summary(),
            'indicator_performance': self.indicator_tracker.get_performance_summary(),
            'trade_results': trade_results
        }
    
    def optimize_strategy_parameters(self, strategy_class, param_grid: Dict) -> Dict:
        """
        Optimiza los parámetros de una estrategia mediante grid search
        
        Args:
            strategy_class: Clase de estrategia a optimizar
            param_grid: Diccionario con parámetros y valores a probar
            
        Returns:
            Dict: Mejores parámetros y resultados
        """
        if self.data is None or self.data.empty:
            logger.error("No data loaded for optimization")
            return {}
        
        best_result = {
            'params': {},
            'metrics': {
                'return_pct': -9999,
                'sharpe_ratio': -9999
            }
        }
        
        all_results = []
        
        # Generar todas las combinaciones de parámetros
        import itertools
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        combinations = list(itertools.product(*param_values))
        total_combinations = len(combinations)
        
        logger.info(f"Optimizing strategy with {total_combinations} parameter combinations")
        
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            
            logger.info(f"Testing combination {i+1}/{total_combinations}: {params}")
            
            # Crear función de estrategia con los parámetros
            def strategy_fn(df):
                return getattr(strategy_class, 'moving_average_crossover')(
                    df, **params
                )
            
            # Ejecutar backtest
            result = self.run_backtest(strategy_fn)
            
            # Guardar resultado
            result['params'] = params.copy()
            all_results.append(result)
            
            # Actualizar mejor resultado
            if result.get('return_pct', -9999) > best_result['metrics'].get('return_pct', -9999):
                best_result = {
                    'params': params.copy(),
                    'metrics': result.copy()
                }
            
            logger.info(f"Result: Return = {result.get('return_pct', 0):.2f}%, Sharpe = {result.get('sharpe_ratio', 0):.2f}")
        
        return {
            'best_result': best_result,
            'all_results': all_results
        }
    
    def plot_results(self):
        """Grafica los resultados del backtest"""
        self.simulator.plot_equity_curve()


# Ejemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Instanciar motor de backtest
    backtest = BacktestEngine(exchange_id='okx', symbol='SOL/USDT', timeframe='15m')
    
    # Cargar datos (método simulado para ejemplo)
    # backtest.load_data_from_exchange(start_date='2022-01-01', end_date='2022-12-31')
    
    # Ejecutar backtest con estrategia de cruce de medias móviles
    def sma_crossover_strategy(df):
        return ClassicStrategy.moving_average_crossover(df, fast_period=20, slow_period=50)
    
    results = backtest.run_backtest(sma_crossover_strategy)
    
    # Mostrar resultados
    print("Backtest Results:")
    for key, value in results.items():
        print(f"{key}: {value}")
    
    # Graficar resultados
    backtest.plot_results()