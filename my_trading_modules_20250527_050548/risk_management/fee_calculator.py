"""
Módulo para calcular comisiones y costos de financiamiento en operaciones

Este módulo proporciona funciones para calcular:
- Comisiones por operación (maker/taker)
- Costos de financiamiento para posiciones con apalancamiento
- Impacto de las comisiones en el rendimiento total
"""

import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
import logging

logger = logging.getLogger("FeeCalculator")

class FeeCalculator:
    """Calculadora de comisiones y costos de financiamiento"""
    
    def __init__(self, exchange: str = "okx"):
        """
        Inicializa la calculadora
        
        Args:
            exchange: Nombre del exchange (okx, binance, etc.)
        """
        self.exchange = exchange
        self.fee_structure = self._get_fee_structure(exchange)
    
    def _get_fee_structure(self, exchange: str) -> Dict[str, Any]:
        """
        Obtiene la estructura de comisiones para un exchange
        
        Args:
            exchange: Nombre del exchange
            
        Returns:
            Dict: Estructura de comisiones
        """
        # Valores estándar para exchanges populares
        # Nota: Estos valores deben actualizarse periódicamente
        fee_structures = {
            "okx": {
                "spot": {
                    "maker": 0.0008,  # 0.08%
                    "taker": 0.0010,  # 0.10%
                    "withdrawal": 0.0005  # 0.05%
                },
                "futures": {
                    "maker": 0.0002,  # 0.02%
                    "taker": 0.0005,  # 0.05%
                    "funding_interval_hours": 8,  # Cada 8 horas
                    "avg_funding_rate": 0.0001,  # 0.01% por intervalio (promedio)
                    "max_funding_rate": 0.0075,  # 0.75% máximo
                    "min_funding_rate": -0.0075  # -0.75% mínimo (pago al trader)
                },
                "margin": {
                    "maker": 0.0008,  # 0.08%
                    "taker": 0.0010,  # 0.10%
                    "daily_interest": 0.00019,  # 0.019% diario (~7% anual)
                    "hourly_interest": 0.00019 / 24  # Interés por hora
                }
            },
            "binance": {
                "spot": {
                    "maker": 0.0010,  # 0.10%
                    "taker": 0.0010,  # 0.10%
                    "withdrawal": 0.0005  # 0.05%
                },
                "futures": {
                    "maker": 0.0002,  # 0.02%
                    "taker": 0.0004,  # 0.04%
                    "funding_interval_hours": 8,  # Cada 8 horas
                    "avg_funding_rate": 0.0001,  # 0.01% por intervalo
                    "max_funding_rate": 0.0075,
                    "min_funding_rate": -0.0075
                },
                "margin": {
                    "maker": 0.0010,  # 0.10%
                    "taker": 0.0010,  # 0.10%
                    "daily_interest": 0.00027,  # 0.027% diario (~10% anual)
                    "hourly_interest": 0.00027 / 24  # Interés por hora
                }
            }
        }
        
        # Si el exchange no está en la lista, usar valores promedio
        if exchange.lower() not in fee_structures:
            logger.warning(f"Exchange {exchange} no encontrado. Usando valores promedio.")
            return {
                "spot": {
                    "maker": 0.0010,  # 0.10%
                    "taker": 0.0010,  # 0.10%
                    "withdrawal": 0.0005  # 0.05%
                },
                "futures": {
                    "maker": 0.0002,  # 0.02%
                    "taker": 0.0004,  # 0.04%
                    "funding_interval_hours": 8,  # Cada 8 horas
                    "avg_funding_rate": 0.0001,  # 0.01% por intervalo
                    "max_funding_rate": 0.0075,
                    "min_funding_rate": -0.0075
                },
                "margin": {
                    "maker": 0.0010,  # 0.10%
                    "taker": 0.0010,  # 0.10%
                    "daily_interest": 0.00022,  # 0.022% diario (~8% anual)
                    "hourly_interest": 0.00022 / 24  # Interés por hora
                }
            }
        
        return fee_structures[exchange.lower()]
    
    def calculate_trade_fee(self, trade_type: str, order_type: str, amount: float, 
                           price: float, is_short: bool = False) -> float:
        """
        Calcula la comisión por operación
        
        Args:
            trade_type: Tipo de mercado ("spot", "futures", "margin")
            order_type: Tipo de orden ("market" = taker, "limit" = maker)
            amount: Cantidad operada (en unidades base, ej: SOL)
            price: Precio de ejecución (en USDT)
            is_short: Indica si es una posición corta (solo para futures/margin)
            
        Returns:
            float: Comisión en USDT
        """
        trade_value = amount * price  # Valor de la operación en USDT
        
        # Determinar tasa de comisión
        if trade_type not in self.fee_structure:
            logger.warning(f"Tipo de mercado {trade_type} no soportado. Usando spot.")
            trade_type = "spot"
        
        fee_rates = self.fee_structure[trade_type]
        fee_rate = fee_rates["taker"] if order_type.lower() == "market" else fee_rates["maker"]
        
        # Calcular comisión
        fee = trade_value * fee_rate
        
        # En posiciones cortas, la comisión se calcula igual (sobre el valor de la posición)
        return fee
    
    def calculate_funding_fee(self, trade_type: str, position_size: float, price: float, 
                             hours_held: float, leverage: float = 1.0, 
                             is_short: bool = False) -> float:
        """
        Calcula el costo de financiamiento por mantener una posición
        
        Args:
            trade_type: Tipo de mercado ("futures", "margin")
            position_size: Tamaño de la posición (en unidades base, ej: SOL)
            price: Precio de entrada (en USDT)
            hours_held: Horas que se mantuvo la posición
            leverage: Apalancamiento utilizado
            is_short: Indica si es una posición corta
            
        Returns:
            float: Costo de financiamiento en USDT
        """
        position_value = position_size * price  # Valor de la posición en USDT
        
        if trade_type == "futures":
            # Para futuros, el financiamiento se cobra/paga en intervalos
            funding_interval = self.fee_structure["futures"]["funding_interval_hours"]
            avg_funding_rate = self.fee_structure["futures"]["avg_funding_rate"]
            
            # Número de intervalos de financiamiento
            intervals = hours_held / funding_interval
            
            # Redondear al intervalo completo siguiente
            intervals = math.ceil(intervals)
            
            # Calcular tasa total (puede ser positiva o negativa)
            # En posiciones cortas, si la tasa es positiva se paga, si es negativa se recibe
            # En posiciones largas, si la tasa es positiva se recibe, si es negativa se paga
            funding_rate = avg_funding_rate * intervals
            
            # Invertir el signo para posiciones cortas (asumiendo tasa positiva como pago)
            if is_short:
                funding_fee = position_value * funding_rate
            else:
                funding_fee = position_value * funding_rate
            
            return funding_fee
        
        elif trade_type == "margin":
            # Para margin trading, el interés se cobra por hora sobre el monto prestado
            hourly_rate = self.fee_structure["margin"]["hourly_interest"]
            
            # El interés se calcula sobre el monto prestado (valor de posición * (leverage - 1) / leverage)
            borrowed_amount = position_value * (leverage - 1) / leverage if leverage > 1 else 0
            
            # El interés se cobra por hora
            interest_fee = borrowed_amount * hourly_rate * hours_held
            
            # Tanto para long como para short se paga interés sobre lo prestado
            return interest_fee
        
        else:
            # Spot no tiene costo de financiamiento
            return 0.0
    
    def calculate_total_costs(self, trade_type: str, entry_order_type: str, exit_order_type: str,
                             position_size: float, entry_price: float, exit_price: float,
                             hours_held: float, leverage: float = 1.0, is_short: bool = False) -> Dict[str, float]:
        """
        Calcula el costo total de una operación completa
        
        Args:
            trade_type: Tipo de mercado ("spot", "futures", "margin")
            entry_order_type: Tipo de orden de entrada ("market", "limit")
            exit_order_type: Tipo de orden de salida ("market", "limit")
            position_size: Tamaño de la posición (en unidades base, ej: SOL)
            entry_price: Precio de entrada (en USDT)
            exit_price: Precio de salida (en USDT)
            hours_held: Horas que se mantuvo la posición
            leverage: Apalancamiento utilizado
            is_short: Indica si es una posición corta
            
        Returns:
            Dict[str, float]: Desglose de costos
        """
        # Calcular comisiones de entrada y salida
        entry_fee = self.calculate_trade_fee(trade_type, entry_order_type, position_size, entry_price, is_short)
        exit_fee = self.calculate_trade_fee(trade_type, exit_order_type, position_size, exit_price, is_short)
        
        # Calcular costo de financiamiento
        funding_fee = self.calculate_funding_fee(trade_type, position_size, entry_price, hours_held, leverage, is_short)
        
        # Calcular P&L antes de comisiones
        if not is_short:
            pnl_before_fees = position_size * (exit_price - entry_price)
        else:
            pnl_before_fees = position_size * (entry_price - exit_price)
        
        # Aplicar apalancamiento al P&L
        pnl_before_fees = pnl_before_fees * leverage
        
        # Calcular P&L neto después de comisiones
        total_fees = entry_fee + exit_fee + funding_fee
        pnl_after_fees = pnl_before_fees - total_fees
        
        # Calcular retorno porcentual
        position_value = position_size * entry_price
        roi_before_fees = (pnl_before_fees / position_value) * 100
        roi_after_fees = (pnl_after_fees / position_value) * 100
        
        # Calcular impacto de comisiones
        fee_impact = (total_fees / position_value) * 100
        
        return {
            "entry_fee": entry_fee,
            "exit_fee": exit_fee,
            "funding_fee": funding_fee,
            "total_fees": total_fees,
            "pnl_before_fees": pnl_before_fees,
            "pnl_after_fees": pnl_after_fees,
            "roi_before_fees": roi_before_fees,
            "roi_after_fees": roi_after_fees,
            "fee_impact_pct": fee_impact
        }
    
    def estimate_fees_for_strategy(self, trade_type: str, avg_trades_per_day: float, 
                                 avg_position_size: float, avg_price: float, 
                                 avg_hours_held: float, leverage: float = 1.0,
                                 short_ratio: float = 0.0, taker_ratio: float = 0.5,
                                 days: int = 30) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Estima los costos totales para una estrategia a lo largo del tiempo
        
        Args:
            trade_type: Tipo de mercado ("spot", "futures", "margin")
            avg_trades_per_day: Promedio de operaciones por día
            avg_position_size: Tamaño promedio de posición (unidades)
            avg_price: Precio promedio estimado
            avg_hours_held: Promedio de horas por posición
            leverage: Apalancamiento utilizado
            short_ratio: Proporción de operaciones en corto (0.0-1.0)
            taker_ratio: Proporción de órdenes tipo market/taker (0.0-1.0)
            days: Días a estimar
            
        Returns:
            Dict: Estimación de costos
        """
        total_trades = avg_trades_per_day * days
        long_trades = total_trades * (1 - short_ratio)
        short_trades = total_trades * short_ratio
        
        # Estimar comisiones por operación
        taker_trades = total_trades * taker_ratio
        maker_trades = total_trades * (1 - taker_ratio)
        
        taker_fee_rate = self.fee_structure[trade_type]["taker"]
        maker_fee_rate = self.fee_structure[trade_type]["maker"]
        
        avg_position_value = avg_position_size * avg_price
        
        # Calcular comisiones totales de trading
        taker_fees = taker_trades * avg_position_value * taker_fee_rate * 2  # Entrada y salida
        maker_fees = maker_trades * avg_position_value * maker_fee_rate * 2  # Entrada y salida
        total_trading_fees = taker_fees + maker_fees
        
        # Calcular costos de financiamiento
        if trade_type in ["futures", "margin"]:
            # Para posiciones largas
            long_funding = 0
            if long_trades > 0:
                long_funding = long_trades * self.calculate_funding_fee(
                    trade_type, avg_position_size, avg_price, avg_hours_held, leverage, False
                )
            
            # Para posiciones cortas
            short_funding = 0
            if short_trades > 0:
                short_funding = short_trades * self.calculate_funding_fee(
                    trade_type, avg_position_size, avg_price, avg_hours_held, leverage, True
                )
            
            total_funding_fees = long_funding + short_funding
        else:
            total_funding_fees = 0
        
        # Calcular costos totales
        total_fees = total_trading_fees + total_funding_fees
        daily_avg_fees = total_fees / days
        
        # Calcular impacto en el rendimiento (asumiendo una estrategia con 50% win rate y 1:1 risk:reward)
        daily_volume = avg_trades_per_day * avg_position_value * 2  # Compra y venta
        daily_fee_impact_pct = (daily_avg_fees / daily_volume) * 100 if daily_volume > 0 else 0
        
        return {
            "total_trades": total_trades,
            "long_trades": long_trades,
            "short_trades": short_trades,
            "total_trading_fees": total_trading_fees,
            "total_funding_fees": total_funding_fees,
            "total_fees": total_fees,
            "daily_avg_fees": daily_avg_fees,
            "fee_impact_per_trade_pct": (total_fees / total_trades / avg_position_value) * 100 if total_trades > 0 else 0,
            "daily_fee_impact_pct": daily_fee_impact_pct,
            "breakdown": {
                "taker_fees": taker_fees,
                "maker_fees": maker_fees,
                "long_funding": long_funding if trade_type in ["futures", "margin"] else 0,
                "short_funding": short_funding if trade_type in ["futures", "margin"] else 0
            }
        }

def calculate_trade_costs(trade_type: str, order_type: str, amount: float, 
                       price: float, hours_held: float, leverage: float = 1.0,
                       is_short: bool = False, exchange: str = "okx") -> Dict[str, float]:
    """
    Función de conveniencia para calcular costos de una operación individual
    
    Args:
        trade_type: Tipo de mercado ("spot", "futures", "margin")
        order_type: Tipo de orden ("market", "limit")
        amount: Cantidad operada
        price: Precio de ejecución
        hours_held: Horas que se mantuvo la posición
        leverage: Apalancamiento utilizado
        is_short: Indica si es una posición corta
        exchange: Nombre del exchange
        
    Returns:
        Dict: Desglose de costos
    """
    calculator = FeeCalculator(exchange)
    
    # Calcular comisión de entrada
    entry_fee = calculator.calculate_trade_fee(trade_type, order_type, amount, price, is_short)
    
    # Calcular comisión de financiamiento
    funding_fee = calculator.calculate_funding_fee(trade_type, amount, price, hours_held, leverage, is_short)
    
    # Calcular comisión de salida (asumiendo mismo tipo de orden)
    exit_fee = calculator.calculate_trade_fee(trade_type, order_type, amount, price, is_short)
    
    # Calcular total
    total_fee = entry_fee + funding_fee + exit_fee
    
    return {
        "entry_fee": entry_fee,
        "funding_fee": funding_fee,
        "exit_fee": exit_fee,
        "total_fee": total_fee,
        "total_fee_pct": (total_fee / (amount * price)) * 100
    }

def estimate_strategy_costs(trade_type: str, avg_trades_per_day: float, 
                         avg_position_size: float, avg_price: float, 
                         avg_hours_held: float, leverage: float = 1.0,
                         short_ratio: float = 0.5, taker_ratio: float = 0.5,
                         days: int = 30, exchange: str = "okx") -> Dict[str, Any]:
    """
    Función de conveniencia para estimar costos de una estrategia
    
    Args:
        trade_type: Tipo de mercado ("spot", "futures", "margin")
        avg_trades_per_day: Promedio de operaciones por día
        avg_position_size: Tamaño promedio de posición
        avg_price: Precio promedio estimado
        avg_hours_held: Promedio de horas por posición
        leverage: Apalancamiento utilizado
        short_ratio: Proporción de operaciones en corto (0.0-1.0)
        taker_ratio: Proporción de órdenes tipo market/taker (0.0-1.0)
        days: Días a estimar
        exchange: Nombre del exchange
        
    Returns:
        Dict: Estimación de costos
    """
    calculator = FeeCalculator(exchange)
    return calculator.estimate_fees_for_strategy(
        trade_type, avg_trades_per_day, avg_position_size, avg_price,
        avg_hours_held, leverage, short_ratio, taker_ratio, days
    )