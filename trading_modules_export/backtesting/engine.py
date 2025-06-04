#!/usr/bin/env python3
"""
Motor de backtesting para probar y optimizar estrategias de trading
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BacktestEngine')

class TradingSimulator:
    """
    Simulador de trading para backtesting y evaluación de estrategias
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
        self.leverage = leverage
        self.commission = commission
        self.reset()
    
    def reset(self):
        """Reinicia el simulador para una nueva simulación"""
        self.balance = self.initial_balance
        self.equity_history = []
        self.trades = []
        self.current_position = None
        self.positions_history = []
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
    
    def open_position(self, position_type: str, price: float, size: float, 
                     timestamp, reason: str = ""):
        """
        Abre una posición
        
        Args:
            position_type: Tipo de posición ('long' o 'short')
            price: Precio de entrada
            size: Tamaño de la posición en unidades
            timestamp: Timestamp de la entrada
            reason: Razón de la entrada
        """
        if self.current_position is not None:
            logger.warning("Ya existe una posición abierta")
            return
        
        # Convertir timestamp a datetime si es necesario
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()
            
        # Calcular valor de la posición
        position_value = price * size
        
        # Aplicar comisión de entrada
        commission_cost = position_value * self.commission
        
        # Para posiciones cortas, se "vende" primero
        if position_type == 'short':
            entry_cost = 0  # El costo inicial es 0 para short (se vende primero)
            # Pero registramos el valor para calcular P&L después
            contract_value = position_value
        else:  # long
            entry_cost = position_value + commission_cost
            contract_value = position_value
            
            # Verificar si hay suficiente balance
            if entry_cost > self.balance:
                logger.warning(f"Balance insuficiente: {self.balance} < {entry_cost}")
                size = (self.balance / (price * (1 + self.commission)))
                position_value = price * size
                commission_cost = position_value * self.commission
                entry_cost = position_value + commission_cost
                logger.warning(f"Ajustando tamaño a: {size}")
        
        # Crear posición
        self.current_position = {
            'type': position_type,
            'entry_price': price,
            'size': size,
            'timestamp': timestamp,
            'entry_cost': entry_cost,
            'commission_paid': commission_cost,
            'contract_value': contract_value,
            'reason': reason
        }
        
        # Actualizar balance
        if position_type == 'long':
            self.balance -= entry_cost
            
        # Registrar estado
        self.update_equity(price, timestamp)
    
    def close_position(self, price: float, timestamp, reason: str = ""):
        """
        Cierra la posición actual
        
        Args:
            price: Precio de salida
            timestamp: Timestamp de la salida
            reason: Razón de la salida
        """
        if self.current_position is None:
            logger.warning("No hay posición abierta para cerrar")
            return
        
        # Convertir timestamp a datetime si es necesario
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()
            
        # Calcular valor de salida
        exit_value = price * self.current_position['size']
        
        # Aplicar comisión de salida
        commission_cost = exit_value * self.commission
        
        # Calcular P&L
        if self.current_position['type'] == 'long':
            entry_value = self.current_position['entry_price'] * self.current_position['size']
            pnl = exit_value - entry_value - commission_cost - self.current_position['commission_paid']
        else:  # short
            entry_value = self.current_position['entry_price'] * self.current_position['size']
            pnl = entry_value - exit_value - commission_cost - self.current_position['commission_paid']
        
        # Crear registro de trade
        trade = {
            **self.current_position,
            'exit_price': price,
            'exit_timestamp': timestamp,
            'exit_reason': reason,
            'pnl': pnl,
            'pnl_percent': (pnl / self.current_position['contract_value']) * 100,
            'duration': (timestamp - self.current_position['timestamp']).total_seconds() / 3600  # horas
        }
        
        # Añadir a historial
        self.trades.append(trade)
        self.positions_history.append(trade)
        
        # Actualizar balance
        self.balance += exit_value - commission_cost
        if self.current_position['type'] == 'long':
            # Ya se dedujo el costo de entrada
            pass
        else:  # short
            # Para posiciones cortas, ahora deducimos el costo de compra para cerrar
            self.balance -= exit_value
        
        # Cerrar posición
        self.current_position = None
        
        # Registrar estado
        self.update_equity(price, timestamp)
        
        return trade
    
    def update_equity(self, price: float, timestamp):
        """
        Actualiza el equity con el precio actual
        
        Args:
            price: Precio actual
            timestamp: Timestamp actual
        """
        # Convertir timestamp a datetime si es necesario
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()
            
        # Calcular equity actual
        equity = self.balance
        
        # Añadir valor de posición abierta si existe
        if self.current_position is not None:
            position_value = price * self.current_position['size']
            
            if self.current_position['type'] == 'long':
                position_pnl = position_value - self.current_position['contract_value']
            else:  # short
                position_pnl = self.current_position['contract_value'] - position_value
                
            # Descontar comisiones pendientes
            exit_commission = position_value * self.commission
            position_pnl -= exit_commission
            
            equity += position_pnl
            
            # Para longs, añadir también el valor del contrato ya que se dedujo del balance
            if self.current_position['type'] == 'long':
                equity += self.current_position['contract_value']
        
        # Registrar en historial
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': equity,
            'balance': self.balance,
            'has_position': self.current_position is not None
        })
        
        # Actualizar máximo drawdown
        if equity > self.peak_balance:
            self.peak_balance = equity
        else:
            drawdown = (self.peak_balance - equity) / self.peak_balance
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas de rendimiento de la estrategia
        
        Returns:
            Dict[str, Any]: Métricas calculadas
        """
        # Verificar que haya trades
        if not self.trades:
            return {
                'total_trades': 0,
                'net_profit': 0,
                'net_profit_percent': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_trade_duration': 0
            }
            
        # Calcular métricas básicas
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        # Evitar división por cero
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Ganancias y pérdidas
        total_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        total_loss = sum(t['pnl'] for t in losing_trades) if losing_trades else 0
        
        avg_profit = total_profit / win_count if win_count > 0 else 0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        
        # Profit factor
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Retorno total
        net_profit = total_profit + total_loss
        net_profit_percent = (net_profit / self.initial_balance) * 100
        
        # Duración promedio de trades
        avg_duration = sum(t['duration'] for t in self.trades) / total_trades if total_trades > 0 else 0
        
        # Calcular Sharpe Ratio
        if len(self.equity_history) > 1:
            equity_values = [e['equity'] for e in self.equity_history]
            returns = [equity_values[i] / equity_values[i-1] - 1 for i in range(1, len(equity_values))]
            
            annualized_factor = 365 * 24  # Asumiendo datos horarios
            sharpe_ratio = (np.mean(returns) * annualized_factor) / (np.std(returns) * np.sqrt(annualized_factor)) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': net_profit,
            'net_profit_percent': net_profit_percent,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown,
            'final_balance': self.balance,
            'final_equity': self.equity_history[-1]['equity'] if self.equity_history else self.balance,
            'sharpe_ratio': sharpe_ratio,
            'avg_trade_duration': avg_duration
        }
    
    def plot_equity_curve(self, save_path: Optional[str] = None):
        """
        Grafica la curva de equity
        
        Args:
            save_path: Ruta para guardar imagen, o None para mostrar
        """
        if not self.equity_history:
            logger.warning("No hay datos para graficar")
            return
        
        # Preparar datos
        timestamps = [entry['timestamp'] for entry in self.equity_history]
        equity_values = [entry['equity'] for entry in self.equity_history]
        
        # Crear figura
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, equity_values, label='Equity', color='blue')
        
        # Añadir línea de balance inicial
        plt.axhline(y=self.initial_balance, color='gray', linestyle='--', label='Balance inicial')
        
        # Marcar trades
        for trade in self.trades:
            if trade['pnl'] > 0:
                color = 'green'
            else:
                color = 'red'
                
            plt.axvline(x=trade['timestamp'], color=color, alpha=0.3)
            plt.axvline(x=trade['exit_timestamp'], color=color, alpha=0.3)
        
        # Configurar gráfico
        plt.title('Curva de Equity')
        plt.xlabel('Fecha')
        plt.ylabel('Equity ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Guardar o mostrar
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

class BacktestEngine:
    """
    Motor de backtesting para prueba y optimización de estrategias
    """
    
    def __init__(self, exchange: str = 'okx', 
                symbol: str = 'SOL-USDT', 
                timeframe: str = '15m', 
                initial_balance: float = 10000.0,
                leverage: float = 1.0,
                commission: float = 0.001):
        """
        Inicializa el motor de backtesting
        
        Args:
            exchange: Exchange a simular
            symbol: Par de trading
            timeframe: Intervalo de tiempo
            initial_balance: Balance inicial para simulación
            leverage: Apalancamiento
            commission: Comisión por operación
        """
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.commission = commission
        
        # Inicializar simulador
        self.simulator = TradingSimulator(
            initial_balance=initial_balance,
            leverage=leverage,
            commission=commission
        )
        
        # Datos históricos
        self.data = None
        self.start_date = None
        self.end_date = None
        
        # Resultados
        self.results = None
    
    def load_data(self, data_source: str, 
                start_date: Optional[str] = None, 
                end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Carga datos históricos para backtesting
        
        Args:
            data_source: Fuente de datos ('exchange' o ruta a CSV)
            start_date: Fecha de inicio (formato 'YYYY-MM-DD')
            end_date: Fecha de fin (formato 'YYYY-MM-DD')
            
        Returns:
            pd.DataFrame: Datos históricos
        """
        if data_source.lower() == 'exchange':
            # Importar aquí para evitar importación circular
            from data_management.market_data import get_market_data
            
            # Convertir fechas a datetime si se proporcionan
            start_dt = pd.to_datetime(start_date) if start_date else None
            end_dt = pd.to_datetime(end_date) if end_date else None
            
            # Calcular número de velas necesarias
            if start_dt and end_dt:
                # Estimar según timeframe
                if self.timeframe == '1m':
                    minutes_diff = (end_dt - start_dt).total_seconds() / 60
                    limit = int(minutes_diff) + 100  # Añadir margen
                elif self.timeframe == '5m':
                    minutes_diff = (end_dt - start_dt).total_seconds() / 60 / 5
                    limit = int(minutes_diff) + 100
                elif self.timeframe == '15m':
                    minutes_diff = (end_dt - start_dt).total_seconds() / 60 / 15
                    limit = int(minutes_diff) + 100
                elif self.timeframe == '1h':
                    hours_diff = (end_dt - start_dt).total_seconds() / 3600
                    limit = int(hours_diff) + 24
                elif self.timeframe == '4h':
                    hours_diff = (end_dt - start_dt).total_seconds() / 3600 / 4
                    limit = int(hours_diff) + 24
                elif self.timeframe == '1d':
                    days_diff = (end_dt - start_dt).days
                    limit = days_diff + 30
                else:
                    limit = 1000
            else:
                limit = 1000
            
            # Obtener datos del exchange
            data = get_market_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=min(limit, 5000),  # API puede tener límites
                exchange=self.exchange,
                with_indicators=True
            )
            
            if data is None or data.empty:
                raise ValueError(f"No se pudieron obtener datos para {self.symbol} ({self.timeframe})")
                
            # Filtrar por fechas si se proporcionan
            if start_dt:
                data = data[data.index >= start_dt]
            if end_dt:
                data = data[data.index <= end_dt]
        
        else:
            # Asumir que data_source es ruta a un CSV
            if not os.path.exists(data_source):
                raise FileNotFoundError(f"No se encontró el archivo: {data_source}")
                
            # Leer CSV
            data = pd.read_csv(data_source)
            
            # Asegurarse de que el índice sea timestamp
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
            elif 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
            
            # Filtrar por fechas si se proporcionan
            if start_date:
                start_dt = pd.to_datetime(start_date)
                data = data[data.index >= start_dt]
            if end_date:
                end_dt = pd.to_datetime(end_date)
                data = data[data.index <= end_dt]
        
        # Guardar referencia
        self.data = data
        self.start_date = data.index.min()
        self.end_date = data.index.max()
        
        return data
    
    def run_backtest(self, strategy_func: Callable, 
                   strategy_params: Dict = None) -> Dict[str, Any]:
        """
        Ejecuta un backtest con una función de estrategia
        
        Args:
            strategy_func: Función que implementa la estrategia
            strategy_params: Parámetros para la estrategia
            
        Returns:
            Dict[str, Any]: Resultados del backtest
        """
        # Verificar que haya datos
        if self.data is None or self.data.empty:
            raise ValueError("No hay datos cargados para backtesting")
            
        # Inicializar simulador
        self.simulator.reset()
        
        # Parámetros por defecto si no se proporcionan
        if strategy_params is None:
            strategy_params = {}
            
        # Variables para seguimiento de posición
        in_position = False
        position_type = None
        
        # Recorrer datos
        for i in range(1, len(self.data)):
            # Obtener datos hasta el punto actual (simulando tiempo real)
            current_data = self.data.iloc[:i+1].copy()
            current_row = current_data.iloc[-1]
            current_price = current_row['close']
            current_time = current_data.index[-1]
            
            # Ejecutar estrategia
            signal, reason = strategy_func(current_data, **strategy_params)
            
            # Procesar señal
            if signal == 1 and not in_position:  # Señal de compra
                self.simulator.open_position(
                    position_type='long',
                    price=current_price,
                    size=self.simulator.balance / current_price * 0.95,  # Usar 95% del balance
                    timestamp=current_time,
                    reason=reason or "Señal de compra"
                )
                in_position = True
                position_type = 'long'
                
            elif signal == -1 and not in_position:  # Señal de venta en corto
                self.simulator.open_position(
                    position_type='short',
                    price=current_price,
                    size=self.simulator.balance / current_price * 0.95,  # Usar 95% del balance
                    timestamp=current_time,
                    reason=reason or "Señal de venta en corto"
                )
                in_position = True
                position_type = 'short'
                
            elif signal == 0 and in_position:  # Señal de cierre
                self.simulator.close_position(
                    price=current_price,
                    timestamp=current_time,
                    reason=reason or "Señal de cierre"
                )
                in_position = False
                position_type = None
                
            elif in_position and (
                (position_type == 'long' and signal == -1) or 
                (position_type == 'short' and signal == 1)
            ):
                # Señal contraria a la posición actual
                self.simulator.close_position(
                    price=current_price,
                    timestamp=current_time,
                    reason=reason or "Señal contraria"
                )
                
                # Abrir nueva posición en dirección contraria
                new_position_type = 'short' if position_type == 'long' else 'long'
                self.simulator.open_position(
                    position_type=new_position_type,
                    price=current_price,
                    size=self.simulator.balance / current_price * 0.95,
                    timestamp=current_time,
                    reason=reason or f"Cambio a posición {new_position_type}"
                )
                
                in_position = True
                position_type = new_position_type
            
            # Actualizar equity
            self.simulator.update_equity(current_price, current_time)
        
        # Cerrar posición al final si quedó abierta
        if in_position:
            last_price = self.data.iloc[-1]['close']
            last_time = self.data.index[-1]
            
            self.simulator.close_position(
                price=last_price,
                timestamp=last_time,
                reason="Fin del período de backtest"
            )
        
        # Calcular métricas
        self.results = self.simulator.calculate_metrics()
        
        # Agregar metadatos
        self.results.update({
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_balance': self.initial_balance,
            'strategy_params': strategy_params
        })
        
        return self.results
    
    def optimize_strategy(self, strategy_func: Callable, 
                        param_grid: Dict[str, List],
                        metric: str = 'net_profit') -> Dict[str, Any]:
        """
        Optimiza los parámetros de una estrategia mediante grid search
        
        Args:
            strategy_func: Función de estrategia
            param_grid: Diccionario de parámetros a probar
            metric: Métrica a optimizar
            
        Returns:
            Dict[str, Any]: Mejores parámetros y resultados
        """
        # Verificar que haya datos
        if self.data is None or self.data.empty:
            raise ValueError("No hay datos cargados para optimización")
            
        # Preparar resultados
        all_results = []
        best_result = None
        best_params = None
        best_metric_value = float('-inf')
        
        # Generar todas las combinaciones de parámetros
        param_combinations = self._generate_param_combinations(param_grid)
        total_combinations = len(param_combinations)
        
        logger.info(f"Iniciando optimización con {total_combinations} combinaciones de parámetros")
        
        # Probar cada combinación
        for i, params in enumerate(param_combinations):
            logger.info(f"Probando combinación {i+1}/{total_combinations}: {params}")
            
            # Ejecutar backtest con estos parámetros
            result = self.run_backtest(strategy_func, params)
            
            # Guardar resultado
            all_results.append({
                'params': params,
                'metrics': result
            })
            
            # Actualizar mejor resultado
            if result[metric] > best_metric_value:
                best_metric_value = result[metric]
                best_result = result
                best_params = params
                
                logger.info(f"Nuevo mejor resultado: {metric} = {best_metric_value}")
        
        # Devolver mejores parámetros y todos los resultados
        return {
            'best_params': best_params,
            'best_result': best_result,
            'all_results': all_results
        }
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """
        Genera todas las combinaciones de parámetros para grid search
        
        Args:
            param_grid: Diccionario de parámetros con listas de valores
            
        Returns:
            List[Dict]: Lista de diccionarios con combinaciones de parámetros
        """
        from itertools import product
        
        # Extraer nombres de parámetros y listas de valores
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generar todas las combinaciones
        combinations = list(product(*param_values))
        
        # Convertir a lista de diccionarios
        result = []
        for combo in combinations:
            param_dict = {name: value for name, value in zip(param_names, combo)}
            result.append(param_dict)
            
        return result
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Grafica los resultados del backtest
        
        Args:
            save_path: Ruta para guardar imagen, o None para mostrar
        """
        if self.results is None:
            logger.warning("No hay resultados para graficar")
            return
        
        # Generar gráficos
        self.simulator.plot_equity_curve(save_path)
        
    def save_results(self, filepath: str):
        """
        Guarda los resultados del backtest en un archivo JSON
        
        Args:
            filepath: Ruta del archivo
        """
        if self.results is None:
            logger.warning("No hay resultados para guardar")
            return
        
        # Crear copia de resultados para serialización
        results_copy = dict(self.results)
        
        # Convertir fechas a strings para serialización JSON
        if 'start_date' in results_copy and hasattr(results_copy['start_date'], 'isoformat'):
            results_copy['start_date'] = results_copy['start_date'].isoformat()
        
        if 'end_date' in results_copy and hasattr(results_copy['end_date'], 'isoformat'):
            results_copy['end_date'] = results_copy['end_date'].isoformat()
        
        # Guardar archivo
        with open(filepath, 'w') as f:
            json.dump(results_copy, f, indent=2)
            
        logger.info(f"Resultados guardados en: {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Carga resultados de backtest desde un archivo JSON
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            Dict[str, Any]: Resultados cargados
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
            
        # Cargar archivo
        with open(filepath, 'r') as f:
            results = json.load(f)
            
        # Convertir fechas de strings a datetime
        if 'start_date' in results and isinstance(results['start_date'], str):
            results['start_date'] = pd.to_datetime(results['start_date'])
        
        if 'end_date' in results and isinstance(results['end_date'], str):
            results['end_date'] = pd.to_datetime(results['end_date'])
            
        # Guardar referencia
        self.results = results
        
        return results
        
    @staticmethod
    def compare_results(results_list: List[Dict[str, Any]], 
                       key_metrics: List[str] = None,
                       chart_path: Optional[str] = None) -> pd.DataFrame:
        """
        Compara resultados de múltiples backtests
        
        Args:
            results_list: Lista de resultados de backtest
            key_metrics: Lista de métricas a comparar
            chart_path: Ruta para guardar gráfico de comparación
            
        Returns:
            pd.DataFrame: DataFrame con comparativa
        """
        if not results_list:
            logger.warning("No hay resultados para comparar")
            return pd.DataFrame()
            
        # Métricas por defecto si no se especifican
        if key_metrics is None:
            key_metrics = [
                'net_profit_percent', 'win_rate', 'profit_factor', 
                'max_drawdown', 'sharpe_ratio', 'total_trades'
            ]
            
        # Extraer datos para comparación
        comparison_data = []
        
        for result in results_list:
            # Extraer identificación de la estrategia
            strategy_name = result.get('strategy_name', 'Desconocida')
            symbol = result.get('symbol', 'Desconocido')
            timeframe = result.get('timeframe', 'Desconocido')
            
            # Extraer métricas relevantes
            metrics = {metric: result.get(metric, None) for metric in key_metrics}
            
            # Combinar info
            entry = {
                'strategy': strategy_name,
                'symbol': symbol,
                'timeframe': timeframe,
                **metrics
            }
            
            comparison_data.append(entry)
            
        # Crear DataFrame
        df_comparison = pd.DataFrame(comparison_data)
        
        # Generar gráfico si se solicita
        if chart_path:
            # Seleccionar métricas numéricas
            numeric_metrics = [m for m in key_metrics 
                             if all(isinstance(r.get(m, 0), (int, float)) 
                                  for r in results_list)]
            
            if numeric_metrics:
                # Crear figura con múltiples subplots
                fig, axes = plt.subplots(
                    nrows=len(numeric_metrics), 
                    figsize=(12, 4 * len(numeric_metrics))
                )
                
                # Si solo hay una métrica, axes no es una lista
                if len(numeric_metrics) == 1:
                    axes = [axes]
                
                # Graficar cada métrica
                for i, metric in enumerate(numeric_metrics):
                    ax = axes[i]
                    df_comparison.plot(
                        x='strategy', 
                        y=metric, 
                        kind='bar',
                        ax=ax,
                        title=f'Comparación de {metric}'
                    )
                    ax.set_ylabel(metric)
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(chart_path)
                plt.close()
        
        return df_comparison