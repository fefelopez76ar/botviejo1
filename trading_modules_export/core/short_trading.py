"""
Módulo para estrategias de trading en corto (short selling)

Este módulo implementa la funcionalidad necesaria para:
- Abrir posiciones cortas
- Calcular PnL en posiciones cortas
- Gestionar riesgos específicos del trading en corto
- Simular operaciones en corto con un modelo de comisiones realista
"""

import logging
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

from risk_management.fee_calculator import FeeCalculator, calculate_trade_costs
from data_management.market_data import get_market_data, update_market_data

logger = logging.getLogger("ShortTrading")

class ShortPosition:
    """Clase para representar una posición corta"""
    
    def __init__(self, symbol: str, entry_price: float, size: float, leverage: float = 1.0,
                entry_time: Optional[datetime] = None, trade_type: str = "futures",
                exchange: str = "okx", stop_loss: Optional[float] = None,
                take_profit: Optional[float] = None):
        """
        Inicializa una posición corta
        
        Args:
            symbol: Símbolo del activo
            entry_price: Precio de entrada
            size: Tamaño de la posición (en unidades, ej: 1 SOL)
            leverage: Apalancamiento utilizado
            entry_time: Timestamp de entrada (default=ahora)
            trade_type: Tipo de mercado ("futures", "margin")
            exchange: Nombre del exchange
            stop_loss: Nivel de stop loss (precio)
            take_profit: Nivel de take profit (precio)
        """
        self.symbol = symbol
        self.entry_price = entry_price
        self.size = size
        self.leverage = leverage
        self.entry_time = entry_time if entry_time else datetime.now()
        self.trade_type = trade_type
        self.exchange = exchange
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Propiedades para tracking de la posición
        self.exit_price = None
        self.exit_time = None
        self.is_open = True
        self.liquidation_price = self._calculate_liquidation_price()
        
        # Propiedades para cálculo de comisiones
        self.entry_fee = 0.0
        self.funding_fees = 0.0
        self.exit_fee = 0.0
        
        # Calcular comisión de entrada inicial
        fee_calculator = FeeCalculator(exchange)
        self.entry_fee = fee_calculator.calculate_trade_fee(
            trade_type, "market", size, entry_price, True
        )
        
        logger.info(f"Posición corta abierta: {symbol} a {entry_price} x {size} u. (lev: {leverage}x)")
    
    def _calculate_liquidation_price(self) -> float:
        """
        Calcula el precio de liquidación para la posición corta
        
        Returns:
            float: Precio de liquidación
        """
        # Para posiciones cortas, el precio de liquidación es mayor que el precio de entrada
        # La fórmula depende del exchange y tipo de mercado, esto es una aproximación
        if self.leverage <= 1:
            # Sin apalancamiento no hay liquidación
            return float('inf')
        
        # Margen de mantenimiento (varía por exchange)
        maintenance_margin = 0.05  # 5% típico
        
        # Aproximación: entry_price * (1 + 1/(leverage * (1 - maintenance_margin)))
        liquidation_price = self.entry_price * (1 + 1/(self.leverage * (1 - maintenance_margin)))
        
        return liquidation_price
    
    def update_fees(self, current_price: float) -> float:
        """
        Actualiza el cálculo de comisiones de financiamiento
        
        Args:
            current_price: Precio actual del activo
            
        Returns:
            float: Nuevas comisiones de financiamiento acumuladas
        """
        if not self.is_open:
            return self.funding_fees
        
        # Calcular tiempo transcurrido en horas
        now = datetime.now()
        hours_passed = (now - self.entry_time).total_seconds() / 3600
        
        # Calcular comisiones de financiamiento
        fee_calculator = FeeCalculator(self.exchange)
        self.funding_fees = fee_calculator.calculate_funding_fee(
            self.trade_type, self.size, self.entry_price, hours_passed, self.leverage, True
        )
        
        return self.funding_fees
    
    def calculate_pnl(self, current_price: float, include_fees: bool = True) -> Dict[str, float]:
        """
        Calcula el P&L actual o final de la posición
        
        Args:
            current_price: Precio actual o de salida
            include_fees: Si se deben incluir las comisiones en el cálculo
            
        Returns:
            Dict[str, float]: Información de P&L
        """
        # Actualizar comisiones si la posición está abierta
        if self.is_open:
            self.update_fees(current_price)
        
        # Para posiciones cortas, el P&L es positivo cuando el precio baja
        price_diff = self.entry_price - current_price
        
        # Calcular P&L básico
        notional_value = self.size * self.entry_price
        pnl_amount = self.size * price_diff * self.leverage
        pnl_pct = (price_diff / self.entry_price) * 100 * self.leverage
        
        # Incluir comisiones si se solicita
        total_fees = 0.0
        if include_fees:
            # Si la posición está cerrada, ya tenemos todas las comisiones
            if not self.is_open:
                total_fees = self.entry_fee + self.funding_fees + self.exit_fee
            else:
                # Si está abierta, calculamos la comisión de salida estimada
                fee_calculator = FeeCalculator(self.exchange)
                exit_fee = fee_calculator.calculate_trade_fee(
                    self.trade_type, "market", self.size, current_price, True
                )
                total_fees = self.entry_fee + self.funding_fees + exit_fee
        
        # P&L neto después de comisiones
        net_pnl = pnl_amount - total_fees
        net_pnl_pct = (net_pnl / notional_value) * 100
        
        # ROI sobre el margen (solo para posiciones apalancadas)
        margin = notional_value / self.leverage if self.leverage > 1 else notional_value
        roi = (net_pnl / margin) * 100
        
        return {
            "price_diff": price_diff,
            "pnl_amount": pnl_amount,
            "pnl_pct": pnl_pct,
            "total_fees": total_fees,
            "fee_impact_pct": (total_fees / notional_value) * 100,
            "net_pnl": net_pnl,
            "net_pnl_pct": net_pnl_pct,
            "roi": roi,
            "notional_value": notional_value,
            "margin_used": margin
        }
    
    def close_position(self, exit_price: float, exit_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Cierra la posición corta
        
        Args:
            exit_price: Precio de salida
            exit_time: Timestamp de salida (default=ahora)
            
        Returns:
            Dict[str, Any]: Resultados del cierre de posición
        """
        if not self.is_open:
            logger.warning("Intento de cerrar una posición ya cerrada")
            return {"error": "Position already closed"}
        
        self.exit_price = exit_price
        self.exit_time = exit_time if exit_time else datetime.now()
        self.is_open = False
        
        # Calcular comisión de salida
        fee_calculator = FeeCalculator(self.exchange)
        self.exit_fee = fee_calculator.calculate_trade_fee(
            self.trade_type, "market", self.size, exit_price, True
        )
        
        # Actualizar comisiones de financiamiento finales
        hours_held = (self.exit_time - self.entry_time).total_seconds() / 3600
        self.funding_fees = fee_calculator.calculate_funding_fee(
            self.trade_type, self.size, self.entry_price, hours_held, self.leverage, True
        )
        
        # Calcular P&L final
        pnl_info = self.calculate_pnl(exit_price, True)
        
        logger.info(f"Posición corta cerrada: {self.symbol} a {exit_price}. P&L: {pnl_info['net_pnl']:.2f} USDT ({pnl_info['net_pnl_pct']:.2f}%)")
        
        return {
            "symbol": self.symbol,
            "position_type": "short",
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "leverage": self.leverage,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "hours_held": hours_held,
            "entry_fee": self.entry_fee,
            "funding_fees": self.funding_fees,
            "exit_fee": self.exit_fee,
            "total_fees": self.entry_fee + self.funding_fees + self.exit_fee,
            "pnl_info": pnl_info
        }
    
    def check_exit_conditions(self, current_price: float, current_time: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Verifica si se cumplen las condiciones para cerrar la posición
        
        Args:
            current_price: Precio actual
            current_time: Timestamp actual (default=ahora)
            
        Returns:
            Tuple[bool, str]: (Debe cerrar, razón)
        """
        if not self.is_open:
            return (False, "Position already closed")
        
        if current_time is None:
            current_time = datetime.now()
        
        # Comprobar stop loss (para short, el SL es un precio superior al de entrada)
        if self.stop_loss is not None and current_price >= self.stop_loss:
            return (True, "stop_loss")
        
        # Comprobar take profit (para short, el TP es un precio inferior al de entrada)
        if self.take_profit is not None and current_price <= self.take_profit:
            return (True, "take_profit")
        
        # Comprobar liquidación
        if current_price >= self.liquidation_price:
            return (True, "liquidation")
        
        return (False, "")

class ShortTradingSimulator:
    """Clase para simular operaciones en corto con cálculo de comisiones"""
    
    def __init__(self, symbol: str, trade_type: str = "futures", exchange: str = "okx"):
        """
        Inicializa el simulador
        
        Args:
            symbol: Símbolo a operar
            trade_type: Tipo de mercado ("futures", "margin")
            exchange: Nombre del exchange
        """
        self.symbol = symbol
        self.trade_type = trade_type
        self.exchange = exchange
        self.open_positions = []  # Posiciones abiertas
        self.closed_positions = []  # Historial de posiciones cerradas
        self.fee_calculator = FeeCalculator(exchange)
        
        # Inicializar balance inicial (para simulación)
        self.initial_balance = 10000.0  # 10,000 USDT
        self.current_balance = self.initial_balance
        
        logger.info(f"Simulador de trading en corto inicializado para {symbol} en {exchange} ({trade_type})")
    
    def open_short(self, price: float, size: float, leverage: float = 1.0,
                 stop_loss_pct: Optional[float] = None, take_profit_pct: Optional[float] = None) -> Dict[str, Any]:
        """
        Abre una posición corta
        
        Args:
            price: Precio de entrada
            size: Tamaño de posición (unidades)
            leverage: Apalancamiento
            stop_loss_pct: Stop loss como porcentaje por encima del precio de entrada
            take_profit_pct: Take profit como porcentaje por debajo del precio de entrada
            
        Returns:
            Dict[str, Any]: Información de la posición abierta
        """
        # Calcular stop loss y take profit si se especifican como porcentaje
        stop_loss = None
        if stop_loss_pct is not None:
            stop_loss = price * (1 + stop_loss_pct / 100)
        
        take_profit = None
        if take_profit_pct is not None:
            take_profit = price * (1 - take_profit_pct / 100)
        
        # Crear nueva posición
        position = ShortPosition(
            symbol=self.symbol,
            entry_price=price,
            size=size,
            leverage=leverage,
            trade_type=self.trade_type,
            exchange=self.exchange,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Calcular margen requerido
        position_value = size * price
        margin_required = position_value / leverage if leverage > 1 else position_value
        
        # Verificar si hay suficiente balance
        if margin_required > self.current_balance:
            logger.warning(f"Balance insuficiente para abrir posición: {margin_required} > {self.current_balance}")
            return {"error": "Insufficient balance", "margin_required": margin_required}
        
        # Restar comisión de entrada del balance
        self.current_balance -= position.entry_fee
        
        # Añadir a posiciones abiertas
        self.open_positions.append(position)
        
        return {
            "position_id": len(self.open_positions) - 1,
            "symbol": self.symbol,
            "entry_price": price,
            "size": size,
            "leverage": leverage,
            "margin_required": margin_required,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_fee": position.entry_fee,
            "liquidation_price": position.liquidation_price
        }
    
    def close_short(self, position_id: int, price: float) -> Dict[str, Any]:
        """
        Cierra una posición corta específica
        
        Args:
            position_id: ID de la posición (índice en la lista)
            price: Precio de salida
            
        Returns:
            Dict[str, Any]: Resultado del cierre
        """
        # Verificar que la posición existe y está abierta
        if position_id < 0 or position_id >= len(self.open_positions):
            return {"error": "Invalid position ID"}
        
        position = self.open_positions[position_id]
        
        if not position.is_open:
            return {"error": "Position already closed"}
        
        # Cerrar la posición
        result = position.close_position(price)
        
        # Actualizar balance
        # - Devolver el margen
        position_value = position.size * position.entry_price
        margin_used = position_value / position.leverage if position.leverage > 1 else position_value
        self.current_balance += margin_used
        
        # - Añadir/restar P&L
        self.current_balance += result["pnl_info"]["net_pnl"]
        
        # Mover a posiciones cerradas
        self.closed_positions.append(position)
        self.open_positions.pop(position_id)
        
        # Incluir balance actualizado en el resultado
        result["updated_balance"] = self.current_balance
        
        return result
    
    def check_positions(self, current_price: float) -> List[Dict[str, Any]]:
        """
        Verifica todas las posiciones abiertas y cierra las que cumplan condiciones
        
        Args:
            current_price: Precio actual
            
        Returns:
            List[Dict[str, Any]]: Resultados de las posiciones cerradas
        """
        closed_results = []
        positions_to_remove = []
        
        for i, position in enumerate(self.open_positions):
            # Comprobar condiciones de salida
            should_close, reason = position.check_exit_conditions(current_price)
            
            if should_close:
                # Cerrar la posición
                result = self.close_short(i, current_price)
                result["close_reason"] = reason
                closed_results.append(result)
                positions_to_remove.append(i)
                
                logger.info(f"Posición cerrada automáticamente: {reason}")
        
        # Actualizar lista de posiciones abiertas (eliminar las cerradas)
        self.open_positions = [p for i, p in enumerate(self.open_positions) if i not in positions_to_remove]
        
        return closed_results
    
    def update_all_fees(self, current_price: float) -> float:
        """
        Actualiza las comisiones de financiamiento para todas las posiciones abiertas
        
        Args:
            current_price: Precio actual
            
        Returns:
            float: Total de comisiones de financiamiento
        """
        total_funding_fees = 0.0
        
        for position in self.open_positions:
            funding_fee = position.update_fees(current_price)
            total_funding_fees += funding_fee
        
        return total_funding_fees
    
    def get_position_summary(self, position_id: int, current_price: float) -> Dict[str, Any]:
        """
        Obtiene un resumen de una posición específica
        
        Args:
            position_id: ID de la posición
            current_price: Precio actual
            
        Returns:
            Dict[str, Any]: Resumen de la posición
        """
        if position_id < 0 or position_id >= len(self.open_positions):
            return {"error": "Invalid position ID"}
        
        position = self.open_positions[position_id]
        
        # Actualizar comisiones
        position.update_fees(current_price)
        
        # Calcular P&L actual
        pnl_info = position.calculate_pnl(current_price)
        
        # Tiempo transcurrido
        hours_held = (datetime.now() - position.entry_time).total_seconds() / 3600
        
        return {
            "position_id": position_id,
            "symbol": position.symbol,
            "position_type": "short",
            "entry_price": position.entry_price,
            "current_price": current_price,
            "size": position.size,
            "leverage": position.leverage,
            "entry_time": position.entry_time.isoformat(),
            "hours_held": hours_held,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "liquidation_price": position.liquidation_price,
            "entry_fee": position.entry_fee,
            "current_funding_fees": position.funding_fees,
            "estimated_exit_fee": self.fee_calculator.calculate_trade_fee(
                position.trade_type, "market", position.size, current_price, True
            ),
            "pnl_info": pnl_info
        }
    
    def get_account_summary(self, current_price: float) -> Dict[str, Any]:
        """
        Obtiene un resumen de la cuenta y todas las posiciones
        
        Args:
            current_price: Precio actual
            
        Returns:
            Dict[str, Any]: Resumen de la cuenta
        """
        # Actualizar comisiones
        self.update_all_fees(current_price)
        
        # Calcular equity (balance + P&L no realizado)
        unrealized_pnl = 0.0
        position_value = 0.0
        margin_used = 0.0
        
        for position in self.open_positions:
            pnl_info = position.calculate_pnl(current_price)
            unrealized_pnl += pnl_info["net_pnl"]
            position_value += position.size * position.entry_price
            margin_used += pnl_info["margin_used"]
        
        equity = self.current_balance + unrealized_pnl
        
        # Calcular métricas de rendimiento
        profit_from_closed = sum(p.calculate_pnl(p.exit_price)["net_pnl"] for p in self.closed_positions)
        total_profit = profit_from_closed + unrealized_pnl
        roi = (total_profit / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        # Estadísticas de trading
        total_trades = len(self.closed_positions)
        winning_trades = sum(1 for p in self.closed_positions if p.calculate_pnl(p.exit_price)["net_pnl"] > 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        return {
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "unrealized_pnl": unrealized_pnl,
            "equity": equity,
            "open_positions": len(self.open_positions),
            "position_value": position_value,
            "margin_used": margin_used,
            "margin_level": (equity / margin_used) * 100 if margin_used > 0 else float('inf'),
            "free_margin": equity - margin_used,
            "closed_positions": len(self.closed_positions),
            "profit_from_closed": profit_from_closed,
            "total_profit": total_profit,
            "roi": roi,
            "win_rate": win_rate
        }
    
    def run_simulation(self, data: pd.DataFrame, strategy_fn: Callable, 
                     initial_balance: float = 10000.0, leverage: float = 1.0,
                     position_size_pct: float = 10.0, stop_loss_pct: float = 5.0,
                     take_profit_pct: float = 10.0) -> Dict[str, Any]:
        """
        Ejecuta una simulación completa basada en datos históricos
        
        Args:
            data: DataFrame con datos históricos (OHLCV)
            strategy_fn: Función de estrategia que retorna señales (-1, 0, 1)
            initial_balance: Balance inicial
            leverage: Apalancamiento
            position_size_pct: Tamaño de posición como % del balance
            stop_loss_pct: Stop loss como % por encima del precio de entrada (para shorts)
            take_profit_pct: Take profit como % por debajo del precio de entrada (para shorts)
            
        Returns:
            Dict[str, Any]: Resultados de la simulación
        """
        # Reiniciar simulador
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.open_positions = []
        self.closed_positions = []
        
        # Obtener señales de la estrategia
        signals = strategy_fn(data)
        
        # Variables para tracking
        position_open = False
        trades_log = []
        equity_curve = []
        
        # Simular trading
        for i in range(1, len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            current_signal = signals.iloc[i]
            
            # Guardar equity actual
            if self.open_positions:
                account = self.get_account_summary(current_price)
                equity_curve.append(account["equity"])
            else:
                equity_curve.append(self.current_balance)
            
            # Verificar posiciones abiertas
            if self.open_positions:
                # Comprobar condiciones de cierre
                closed = self.check_positions(current_price)
                if closed:
                    position_open = False
                    trades_log.extend(closed)
            
            # Procesar señal para abrir nueva posición corta
            if not position_open and current_signal < 0:  # Señal de venta para abrir corto
                # Calcular tamaño de posición
                position_size_usd = self.current_balance * position_size_pct / 100
                size = position_size_usd / current_price
                
                # Abrir posición corta
                result = self.open_short(
                    price=current_price,
                    size=size,
                    leverage=leverage,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct
                )
                
                if "error" not in result:
                    position_open = True
                    trades_log.append({
                        "time": current_time.isoformat(),
                        "type": "open_short",
                        "price": current_price,
                        "size": size,
                        "balance": self.current_balance
                    })
            
            # Procesar señal para cerrar posición corta
            elif position_open and current_signal > 0:  # Señal de compra para cerrar corto
                if self.open_positions:
                    result = self.close_short(0, current_price)  # Cerrar la primera posición
                    position_open = False
                    result["time"] = current_time.isoformat()
                    result["type"] = "close_short"
                    trades_log.append(result)
        
        # Cerrar posiciones abiertas al final de la simulación
        final_price = data['close'].iloc[-1]
        while self.open_positions:
            result = self.close_short(0, final_price)
            result["time"] = data.index[-1].isoformat()
            result["type"] = "close_short_final"
            trades_log.append(result)
        
        # Calcular estadísticas finales
        final_balance = self.current_balance
        total_return = ((final_balance / initial_balance) - 1) * 100
        
        # Convertir equity_curve a Series para cálculos
        equity_series = pd.Series(equity_curve, index=data.index[1:])
        
        # Calcular drawdown
        rolling_max = equity_series.cummax()
        drawdown = (equity_series / rolling_max - 1) * 100
        max_drawdown = drawdown.min()
        
        # Calcular Sharpe ratio (asumiendo 0% tasa libre de riesgo)
        daily_returns = equity_series.pct_change().dropna()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)  # Anualizado
        
        # Calcular estadísticas de trading
        total_trades = len([t for t in trades_log if t["type"] in ["close_short", "close_short_final"]])
        wins = [t for t in trades_log if t["type"] in ["close_short", "close_short_final"] and t.get("pnl_info", {}).get("net_pnl", 0) > 0]
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        
        # Calcular comisiones totales
        total_fees = sum(t.get("total_fees", 0) for t in trades_log if t["type"] in ["close_short", "close_short_final"])
        fee_impact = (total_fees / initial_balance) * 100
        
        return {
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "total_return_pct": total_return,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_fees": total_fees,
            "fee_impact_pct": fee_impact,
            "trades_log": trades_log,
            "equity_curve": equity_curve
        }

def short_trading_example():
    """Ejemplo simple de trading en corto con cálculo de comisiones"""
    # Crear simulador
    simulator = ShortTradingSimulator("SOL-USDT", "futures", "okx")
    
    # Parámetros
    entry_price = 151.0
    size = 10  # 10 SOL
    leverage = 3  # 3x
    stop_loss_pct = 5.0  # 5% por encima del precio de entrada
    take_profit_pct = 10.0  # 10% por debajo del precio de entrada
    
    # Abrir posición corta
    position = simulator.open_short(
        price=entry_price,
        size=size,
        leverage=leverage,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct
    )
    
    print("\n===== POSICIÓN CORTA ABIERTA =====")
    print(f"Símbolo: {position['symbol']}")
    print(f"Precio de entrada: ${position['entry_price']}")
    print(f"Tamaño: {position['size']} SOL")
    print(f"Apalancamiento: {position['leverage']}x")
    print(f"Margen requerido: ${position['margin_required']:.2f}")
    print(f"Stop loss: ${position['stop_loss']:.2f} (+{stop_loss_pct}%)")
    print(f"Take profit: ${position['take_profit']:.2f} (-{take_profit_pct}%)")
    print(f"Precio de liquidación: ${position['liquidation_price']:.2f}")
    print(f"Comisión de entrada: ${position['entry_fee']:.2f}")
    
    # Simular diferentes escenarios de precio
    exit_prices = [
        entry_price * 0.9,  # Escenario favorable (-10%)
        entry_price * 1.02,  # Escenario ligeramente desfavorable (+2%)
        entry_price * 0.95,  # Escenario moderadamente favorable (-5%)
    ]
    
    for i, exit_price in enumerate(exit_prices):
        print(f"\n===== ESCENARIO {i+1}: PRECIO ${exit_price:.2f} =====")
        
        # Obtener resumen de la posición con este precio
        summary = simulator.get_position_summary(0, exit_price)
        
        print(f"P&L sin comisiones: ${summary['pnl_info']['pnl_amount']:.2f} ({summary['pnl_info']['pnl_pct']:.2f}%)")
        print(f"Comisiones actuales: ${summary['entry_fee'] + summary['current_funding_fees']:.2f}")
        print(f"Comisión de salida estimada: ${summary['estimated_exit_fee']:.2f}")
        print(f"P&L neto con comisiones: ${summary['pnl_info']['net_pnl']:.2f} ({summary['pnl_info']['net_pnl_pct']:.2f}%)")
        print(f"Impacto de comisiones: {summary['pnl_info']['fee_impact_pct']:.2f}%")
        print(f"ROI sobre margen: {summary['pnl_info']['roi']:.2f}%")
    
    # Cerrar la posición con un precio favorable
    favorable_price = entry_price * 0.9  # -10%
    result = simulator.close_short(0, favorable_price)
    
    print("\n===== POSICIÓN CERRADA =====")
    print(f"Precio de salida: ${result['exit_price']}")
    print(f"Horas mantenida: {result['hours_held']:.2f}")
    print(f"Comisión de entrada: ${result['entry_fee']:.2f}")
    print(f"Comisiones de financiamiento: ${result['funding_fees']:.2f}")
    print(f"Comisión de salida: ${result['exit_fee']:.2f}")
    print(f"Comisiones totales: ${result['total_fees']:.2f}")
    print(f"P&L neto: ${result['pnl_info']['net_pnl']:.2f} ({result['pnl_info']['net_pnl_pct']:.2f}%)")
    print(f"Balance final: ${result['updated_balance']:.2f}")
    
    return {
        "position": position,
        "scenarios": [simulator.get_position_summary(0, price) for price in exit_prices],
        "closing_result": result
    }

def compare_trading_modes(symbol: str = "SOL-USDT", days: int = 30):
    """
    Compara los costos y comisiones entre diferentes modos de trading
    
    Args:
        symbol: Par de trading
        days: Días para el análisis
        
    Returns:
        Dict: Comparación de modos de trading
    """
    from risk_management.fee_calculator import estimate_strategy_costs
    
    # Parámetros comunes
    avg_price = 150.0  # Precio promedio estimado
    avg_position_size = 10.0  # 10 unidades por operación
    avg_trades_per_day = 3  # 3 operaciones por día
    taker_ratio = 0.7  # 70% de órdenes son market orders
    
    # Diferentes modos de trading
    modes = [
        {
            "name": "Spot Trading (Solo Long)",
            "trade_type": "spot",
            "leverage": 1.0,
            "avg_hours_held": 8.0,
            "short_ratio": 0.0  # 0% de operaciones en corto (solo largos)
        },
        {
            "name": "Margin Trading (Long y Short)",
            "trade_type": "margin",
            "leverage": 2.0,
            "avg_hours_held": 10.0,
            "short_ratio": 0.5  # 50% de operaciones en corto
        },
        {
            "name": "Futures Trading (Long y Short)",
            "trade_type": "futures",
            "leverage": 3.0,
            "avg_hours_held": 12.0,
            "short_ratio": 0.5  # 50% de operaciones en corto
        },
        {
            "name": "Scalping Futures (High Frequency)",
            "trade_type": "futures",
            "leverage": 5.0,
            "avg_hours_held": 2.0,
            "short_ratio": 0.5,  # 50% de operaciones en corto
            "avg_trades_per_day": 10.0  # 10 operaciones por día
        }
    ]
    
    results = {}
    
    for mode in modes:
        # Obtener parámetros específicos
        trade_type = mode["trade_type"]
        leverage = mode["leverage"]
        avg_hours_held = mode["avg_hours_held"]
        short_ratio = mode["short_ratio"]
        trades_per_day = mode.get("avg_trades_per_day", avg_trades_per_day)
        
        # Estimar costos
        cost_estimate = estimate_strategy_costs(
            trade_type=trade_type,
            avg_trades_per_day=trades_per_day,
            avg_position_size=avg_position_size,
            avg_price=avg_price,
            avg_hours_held=avg_hours_held,
            leverage=leverage,
            short_ratio=short_ratio,
            taker_ratio=taker_ratio,
            days=days
        )
        
        # Guardar resultados
        results[mode["name"]] = {
            "parameters": mode,
            "estimate": cost_estimate
        }
    
    # Generar comparación en formato tabular
    comparison = []
    
    for name, data in results.items():
        estimate = data["estimate"]
        params = data["parameters"]
        
        comparison.append({
            "mode": name,
            "leverage": params["leverage"],
            "short_enabled": params["short_ratio"] > 0,
            "avg_hours_held": params["avg_hours_held"],
            "trades_per_day": params.get("avg_trades_per_day", avg_trades_per_day),
            "total_trades": estimate["total_trades"],
            "trading_fees": estimate["total_trading_fees"],
            "funding_fees": estimate["total_funding_fees"],
            "total_fees": estimate["total_fees"],
            "daily_avg_fees": estimate["daily_avg_fees"],
            "fee_impact_per_trade_pct": estimate["fee_impact_per_trade_pct"]
        })
    
    return comparison