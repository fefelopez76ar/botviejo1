"""
Estrategias de scalping para trading de muy corto plazo (segundos/minutos)

Este módulo contiene estrategias optimizadas para scalping, enfocadas en:
- Timeframes muy cortos (1m, 3m, 5m)
- Ejecución rápida
- Toma de beneficios en movimientos pequeños
- Uso de indicadores con respuesta rápida
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class ScalpingSignalType(Enum):
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2

class ScalpingStrategy:
    """Clase base para estrategias de scalping"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Inicializa la estrategia
        
        Args:
            params: Parámetros de la estrategia
        """
        self.params = params or {}
        self.name = "Base Scalping Strategy"
        self.description = "Estrategia base para scalping"
        self.required_timeframes = ["1m", "5m"]
        self.min_candles = 100
        
        # Estadísticas de rendimiento
        self.stats = {
            "total_signals": 0,
            "correct_signals": 0,
            "accuracy": 0.0,
            "avg_profit": 0.0,
            "avg_holding_time": 0.0  # en minutos
        }
    
    def get_signal(self, data: pd.DataFrame) -> Tuple[int, str, Dict[str, Any]]:
        """
        Genera una señal de trading
        
        Args:
            data: DataFrame con datos de mercado
            
        Returns:
            Tuple[int, str, Dict]: Señal (-2 a 2), razón y detalles
        """
        # La estrategia base no genera señales reales
        return 0, "No hay señal", {}
    
    def update_stats(self, was_correct: bool, profit: float, holding_time: float):
        """
        Actualiza estadísticas de rendimiento
        
        Args:
            was_correct: Si la señal resultó correcta
            profit: Beneficio/pérdida generado
            holding_time: Tiempo de mantenimiento en minutos
        """
        self.stats["total_signals"] += 1
        if was_correct:
            self.stats["correct_signals"] += 1
        
        self.stats["accuracy"] = self.stats["correct_signals"] / self.stats["total_signals"] if self.stats["total_signals"] > 0 else 0
        
        # Actualizar promedio de beneficio
        total_profit = self.stats["avg_profit"] * (self.stats["total_signals"] - 1)
        self.stats["avg_profit"] = (total_profit + profit) / self.stats["total_signals"]
        
        # Actualizar promedio de tiempo
        total_time = self.stats["avg_holding_time"] * (self.stats["total_signals"] - 1)
        self.stats["avg_holding_time"] = (total_time + holding_time) / self.stats["total_signals"]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de rendimiento
        
        Returns:
            Dict: Estadísticas
        """
        return self.stats
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Obtiene parámetros de la estrategia
        
        Returns:
            Dict: Parámetros
        """
        return self.params

class RSIScalping(ScalpingStrategy):
    """
    Estrategia de scalping basada en RSI y volumen
    
    Genera señales de entrada y salida basadas en:
    - RSI en zonas de sobreventa/sobrecompra
    - Confirmación con volumen
    - Multiple timeframes para confirmar tendencia
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Inicializa la estrategia
        
        Args:
            params: Parámetros de la estrategia
        """
        super().__init__(params)
        self.name = "RSI Scalping"
        self.description = "Scalping basado en RSI con confirmación de volumen"
        
        # Parámetros por defecto
        default_params = {
            "rsi_period": 9,
            "volume_factor": 1.5,
            "overbought": 70,
            "oversold": 30,
            "exit_rsi_high": 65,
            "exit_rsi_low": 35,
            "tp_percent": 0.5,  # Take profit en 0.5%
            "sl_percent": 0.3   # Stop loss en 0.3%
        }
        
        # Usar parámetros proporcionados o default
        self.params = {**default_params, **(params or {})}
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calcula el indicador RSI (Relative Strength Index)
        
        Args:
            data: DataFrame con datos de mercado
            period: Período para el RSI
            
        Returns:
            pd.Series: Valores del RSI
        """
        # Obtener diferencias de precios
        delta = data['close'].diff(1)
        
        # Separar subidas y bajadas
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calcular media exponencial de subidas y bajadas
        avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
        
        # Calcular RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calcular RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_signal(self, data: pd.DataFrame) -> Tuple[int, str, Dict[str, Any]]:
        """
        Genera una señal de trading basada en RSI
        
        Args:
            data: DataFrame con datos de mercado
            
        Returns:
            Tuple[int, str, Dict]: Señal (-2 a 2), razón y detalles
        """
        if len(data) < self.min_candles:
            return 0, "Datos insuficientes", {}
        
        # Calcular RSI
        rsi = self.calculate_rsi(data, self.params["rsi_period"])
        current_rsi = rsi.iloc[-1]
        
        # Calcular indicadores de volumen
        volume = data['volume']
        avg_volume = volume.rolling(window=20).mean()
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume.iloc[-1]
        
        # Obtener último precio
        current_price = data['close'].iloc[-1]
        
        # Generar señal
        signal = 0
        reason = "Sin señal"
        details = {
            "rsi": current_rsi,
            "volume_ratio": volume_ratio,
            "price": current_price
        }
        
        # Condiciones de compra
        if current_rsi < self.params["oversold"] and volume_ratio > self.params["volume_factor"]:
            signal = ScalpingSignalType.BUY.value
            reason = f"RSI en sobreventa ({current_rsi:.2f}) con volumen alto"
        
        # Condiciones de venta
        elif current_rsi > self.params["overbought"] and volume_ratio > self.params["volume_factor"]:
            signal = ScalpingSignalType.SELL.value
            reason = f"RSI en sobrecompra ({current_rsi:.2f}) con volumen alto"
        
        # Condiciones de salida
        elif (current_rsi > self.params["exit_rsi_high"] and volume_ratio > 1.0 and signal > 0):
            signal = ScalpingSignalType.NEUTRAL.value
            reason = f"RSI saliendo de sobreventa ({current_rsi:.2f}), momento de salir"
        elif (current_rsi < self.params["exit_rsi_low"] and volume_ratio > 1.0 and signal < 0):
            signal = ScalpingSignalType.NEUTRAL.value
            reason = f"RSI saliendo de sobrecompra ({current_rsi:.2f}), momento de salir"
        
        # Añadir información de take profit y stop loss
        if signal != 0:
            details["tp_price"] = current_price * (1 + self.params["tp_percent"]/100) if signal > 0 else current_price * (1 - self.params["tp_percent"]/100)
            details["sl_price"] = current_price * (1 - self.params["sl_percent"]/100) if signal > 0 else current_price * (1 + self.params["sl_percent"]/100)
        
        return signal, reason, details

class MomentumScalping(ScalpingStrategy):
    """
    Estrategia de scalping basada en impulso (momentum) y volumen
    
    Usa movimiento de precios rápidos con confirmación de volumen para entrar
    y salir de posiciones en plazos muy cortos (incluso por debajo de 1 minuto)
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Inicializa la estrategia
        
        Args:
            params: Parámetros de la estrategia
        """
        super().__init__(params)
        self.name = "Momentum Scalping"
        self.description = "Scalping ultra-rápido basado en impulso y volumen"
        self.required_timeframes = ["1m"]
        
        # Parámetros por defecto
        default_params = {
            "price_change_pct": 0.2,  # % de cambio necesario para detectar impulso
            "volume_surge_factor": 2.0,  # Factor de aumento de volumen
            "lookback_periods": 3,  # Períodos hacia atrás para analizar
            "tp_pct": 0.3,  # % take profit
            "sl_pct": 0.15,  # % stop loss
            "trailing_sl": True,  # Usar stop loss dinámico
            "max_holding_time": 10  # Tiempo máximo en minutos
        }
        
        # Usar parámetros proporcionados o default
        self.params = {**default_params, **(params or {})}
    
    def get_signal(self, data: pd.DataFrame) -> Tuple[int, str, Dict[str, Any]]:
        """
        Genera una señal de trading basada en impulso
        
        Args:
            data: DataFrame con datos de mercado
            
        Returns:
            Tuple[int, str, Dict]: Señal (-2 a 2), razón y detalles
        """
        if len(data) < self.min_candles:
            return 0, "Datos insuficientes", {}
        
        # Calcular cambios porcentuales
        price_pct_change = data['close'].pct_change(self.params["lookback_periods"]) * 100
        current_pct_change = price_pct_change.iloc[-1]
        
        # Analizar volumen
        volume = data['volume']
        avg_volume = volume.rolling(window=20).mean()
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume.iloc[-1]
        
        # Obtener último precio
        current_price = data['close'].iloc[-1]
        
        # Analizar velas más recientes
        recent_candles = data.iloc[-self.params["lookback_periods"]:]
        bullish_candles = sum(1 for i in range(len(recent_candles)) if recent_candles['close'].iloc[i] > recent_candles['open'].iloc[i])
        bearish_candles = sum(1 for i in range(len(recent_candles)) if recent_candles['close'].iloc[i] < recent_candles['open'].iloc[i])
        
        # Calcular fuerza relativa entre alcistas y bajistas
        candle_strength = bullish_candles - bearish_candles
        
        # Generar señal
        signal = 0
        reason = "Sin señal"
        details = {
            "price_change_pct": current_pct_change,
            "volume_ratio": volume_ratio,
            "candle_strength": candle_strength,
            "price": current_price
        }
        
        # Condiciones de compra: impulso alcista con volumen alto
        if (current_pct_change > self.params["price_change_pct"] and 
            volume_ratio > self.params["volume_surge_factor"] and
            candle_strength > 0):
            
            signal = ScalpingSignalType.STRONG_BUY.value
            reason = f"Impulso alcista fuerte ({current_pct_change:.2f}%) con volumen {volume_ratio:.1f}x"
        
        # Condiciones de venta: impulso bajista con volumen alto
        elif (current_pct_change < -self.params["price_change_pct"] and 
              volume_ratio > self.params["volume_surge_factor"] and
              candle_strength < 0):
            
            signal = ScalpingSignalType.STRONG_SELL.value
            reason = f"Impulso bajista fuerte ({current_pct_change:.2f}%) con volumen {volume_ratio:.1f}x"
        
        # Añadir información de take profit y stop loss
        if signal != 0:
            details["tp_price"] = current_price * (1 + self.params["tp_pct"]/100) if signal > 0 else current_price * (1 - self.params["tp_pct"]/100)
            details["sl_price"] = current_price * (1 - self.params["sl_pct"]/100) if signal > 0 else current_price * (1 + self.params["sl_pct"]/100)
            details["max_holding_time"] = self.params["max_holding_time"]
        
        return signal, reason, details

class GridScalping(ScalpingStrategy):
    """
    Estrategia de scalping con grid trading
    
    Establece una rejilla de niveles de compra y venta dentro
    de un rango, para aprovechar movimientos laterales de precio.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Inicializa la estrategia
        
        Args:
            params: Parámetros de la estrategia
        """
        super().__init__(params)
        self.name = "Grid Scalping"
        self.description = "Scalping con rejilla de niveles para mercados laterales"
        
        # Parámetros por defecto
        default_params = {
            "grid_levels": 5,  # Número de niveles en la rejilla
            "grid_spacing_pct": 0.1,  # Espaciado entre niveles (%)
            "profit_pct": 0.15,  # Beneficio objetivo por nivel
            "range_factor": 2.0,  # Factor para determinar si estamos en rango
            "atr_period": 14,  # Periodo para ATR (Average True Range)
            "use_dynamic_grid": True  # Usar rejilla dinámica (ajustada por ATR)
        }
        
        # Usar parámetros proporcionados o default
        self.params = {**default_params, **(params or {})}
        
        # Estado de la rejilla
        self.grid = None
        self.grid_center = None
        self.last_update_time = None
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calcula el ATR (Average True Range)
        
        Args:
            data: DataFrame con datos de mercado
            period: Período para ATR
            
        Returns:
            float: Valor del ATR
        """
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def is_ranging_market(self, data: pd.DataFrame) -> bool:
        """
        Determina si el mercado está en rango (lateral)
        
        Args:
            data: DataFrame con datos de mercado
            
        Returns:
            bool: True si el mercado está en rango
        """
        # Calcular ATR
        atr = self.calculate_atr(data, self.params["atr_period"])
        
        # Calcular rango de precios recientes
        recent_high = data['high'].tail(20).max()
        recent_low = data['low'].tail(20).min()
        recent_range = recent_high - recent_low
        
        # Calcular precio medio
        mid_price = (recent_high + recent_low) / 2
        
        # El mercado está en rango si el rango reciente no es mucho mayor que el ATR
        range_to_atr = recent_range / (atr * self.params["range_factor"])
        
        return range_to_atr < 1.0
    
    def generate_grid(self, center_price: float, atr: float) -> List[Dict[str, float]]:
        """
        Genera una rejilla de niveles alrededor del precio central
        
        Args:
            center_price: Precio central para la rejilla
            atr: ATR actual para ajustar tamaño de rejilla
            
        Returns:
            List[Dict]: Lista de niveles de la rejilla
        """
        grid = []
        
        # Número de niveles por encima y por debajo
        levels_each_side = self.params["grid_levels"] // 2
        
        # Calcular espaciado (fijo o dinámico)
        if self.params["use_dynamic_grid"]:
            # Usar ATR para espaciado dinámico
            grid_spacing = atr * 0.5
        else:
            # Usar espaciado fijo (porcentaje)
            grid_spacing = center_price * (self.params["grid_spacing_pct"] / 100)
        
        # Generar niveles
        for i in range(-levels_each_side, levels_each_side + 1):
            level_price = center_price + (i * grid_spacing)
            
            # Calcular precios de compra y venta
            if i < 0:  # Niveles inferiores (compra)
                grid.append({
                    "level": i,
                    "price": level_price,
                    "type": "buy",
                    "tp_price": level_price * (1 + self.params["profit_pct"]/100)
                })
            elif i > 0:  # Niveles superiores (venta)
                grid.append({
                    "level": i,
                    "price": level_price,
                    "type": "sell",
                    "tp_price": level_price * (1 - self.params["profit_pct"]/100)
                })
            # El nivel 0 es el central, no genera señal
        
        return grid
    
    def get_signal(self, data: pd.DataFrame) -> Tuple[int, str, Dict[str, Any]]:
        """
        Genera una señal de trading basada en la rejilla
        
        Args:
            data: DataFrame con datos de mercado
            
        Returns:
            Tuple[int, str, Dict]: Señal (-2 a 2), razón y detalles
        """
        if len(data) < self.min_candles:
            return 0, "Datos insuficientes", {}
        
        # Obtener precio actual
        current_price = data['close'].iloc[-1]
        current_time = data.index[-1]
        
        # Calcular ATR
        atr = self.calculate_atr(data, self.params["atr_period"])
        
        # Verificar si el mercado está en rango
        is_ranging = self.is_ranging_market(data)
        
        # Inicializar o regenerar la rejilla si es necesario
        should_regenerate = (
            self.grid is None or 
            self.grid_center is None or
            abs(current_price - self.grid_center) / self.grid_center > 0.02 or  # Precio se aleja más de 2% del centro
            self.last_update_time is None or
            (current_time - self.last_update_time).total_seconds() > 3600  # Actualizar cada hora
        )
        
        if should_regenerate:
            self.grid_center = current_price
            self.grid = self.generate_grid(current_price, atr)
            self.last_update_time = current_time
        
        # Generar señal
        signal = 0
        reason = "Sin señal"
        details = {
            "price": current_price,
            "is_ranging": is_ranging,
            "atr": atr
        }
        
        # En mercado no lateral, no generar señales
        if not is_ranging:
            return 0, "Mercado no está en rango, grid inactivo", details
        
        # Buscar nivel de grid más cercano
        closest_level = None
        min_distance = float('inf')
        
        for level in self.grid:
            distance = abs(current_price - level["price"])
            if distance < min_distance:
                min_distance = distance
                closest_level = level
        
        # Si estamos cerca de un nivel, generar señal
        if closest_level and min_distance / current_price < 0.001:  # Dentro de 0.1%
            if closest_level["type"] == "buy":
                signal = ScalpingSignalType.BUY.value
                reason = f"Nivel de compra en grid ({closest_level['level']})"
            elif closest_level["type"] == "sell":
                signal = ScalpingSignalType.SELL.value
                reason = f"Nivel de venta en grid ({closest_level['level']})"
            
            details["grid_level"] = closest_level["level"]
            details["tp_price"] = closest_level["tp_price"]
            details["grid_center"] = self.grid_center
        
        return signal, reason, details

def get_available_scalping_strategies() -> Dict[str, ScalpingStrategy]:
    """
    Obtiene las estrategias de scalping disponibles
    
    Returns:
        Dict[str, ScalpingStrategy]: Diccionario de estrategias
    """
    return {
        "rsi_scalping": RSIScalping(),
        "momentum_scalping": MomentumScalping(),
        "grid_scalping": GridScalping()
    }

def get_strategy_by_name(name: str) -> Optional[ScalpingStrategy]:
    """
    Obtiene una estrategia por su nombre
    
    Args:
        name: Nombre de la estrategia
        
    Returns:
        Optional[ScalpingStrategy]: Estrategia o None si no existe
    """
    strategies = get_available_scalping_strategies()
    return strategies.get(name.lower())

def demo_scalping_strategy(strategy_name: str = "rsi_scalping", 
                          symbol: str = "SOL-USDT",
                          interval: str = "1m",
                          limit: int = 200):
    """
    Función de demostración para probar estrategias de scalping
    
    Args:
        strategy_name: Nombre de la estrategia
        symbol: Par de trading
        interval: Intervalo de tiempo
        limit: Número de velas a analizar
    """
    print(f"\n===== DEMOSTRACIÓN ESTRATEGIA DE SCALPING: {strategy_name.upper()} =====")
    
    # Obtener estrategia
    strategies = get_available_scalping_strategies()
    strategy = strategies.get(strategy_name.lower())
    
    if not strategy:
        print(f"Estrategia no encontrada: {strategy_name}")
        print(f"Estrategias disponibles: {', '.join(strategies.keys())}")
        return
    
    # Obtener datos históricos
    try:
        from data_management.market_data import get_market_data
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        print(f"Obteniendo datos para {symbol} en {interval}...")
        data = get_market_data(symbol, interval, limit)
        
        if data is None or len(data) < 10:
            print("Error al obtener datos. Usando datos simulados para demo.")
            
            # Generar datos simulados para demo
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            # Crear índice de tiempo
            now = datetime.now()
            if interval == '1m':
                index = [now - timedelta(minutes=i) for i in range(limit, 0, -1)]
            elif interval == '5m':
                index = [now - timedelta(minutes=5*i) for i in range(limit, 0, -1)]
            else:
                index = [now - timedelta(minutes=15*i) for i in range(limit, 0, -1)]
            
            # Crear precio base y movimiento
            base_price = 100.0
            price_series = [base_price]
            for i in range(1, limit):
                move = np.random.normal(0, 0.1)
                new_price = price_series[-1] * (1 + move)
                price_series.append(new_price)
            
            # Crear DataFrame
            data = pd.DataFrame({
                'open': price_series,
                'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in price_series],
                'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in price_series],
                'close': [p * (1 + np.random.normal(0, 0.01)) for p in price_series],
                'volume': [abs(np.random.normal(1000, 500)) for _ in range(limit)]
            }, index=index)
    
    except Exception as e:
        print(f"Error al obtener datos: {e}")
        return
    
    # Ejecutar estrategia
    print(f"\nEjecutando estrategia {strategy.name}...")
    print(f"Descripción: {strategy.description}")
    print(f"Parámetros: {strategy.get_parameters()}")
    
    signal, reason, details = strategy.get_signal(data)
    
    # Mostrar resultados
    print("\nResultados:")
    print(f"Señal: {signal} ({ScalpingSignalType(signal).name})")
    print(f"Razón: {reason}")
    print("\nDetalles:")
    for key, value in details.items():
        print(f"  {key}: {value}")
    
    print("\n===== FIN DE DEMOSTRACIÓN =====")

if __name__ == "__main__":
    # Probar todas las estrategias
    for strategy_name in get_available_scalping_strategies().keys():
        demo_scalping_strategy(strategy_name)
        print("\n" + "-"*80 + "\n")