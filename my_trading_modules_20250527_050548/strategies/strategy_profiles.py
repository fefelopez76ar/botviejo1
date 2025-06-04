"""
Módulo para definir y cargar perfiles de estrategias de trading
Implementa un sistema para cargar configuraciones desde JSON y perfiles predefinidos
para diferentes estilos de trading (scalping, day trading, swing trading, etc.)
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import time

logger = logging.getLogger("StrategyProfiles")

class TradingStyle(Enum):
    """Estilos de trading soportados"""
    SCALPING = "scalping"
    DAY_TRADING = "day_trading"
    SWING_TRADING = "swing_trading"
    POSITION_TRADING = "position_trading"
    ADAPTIVE = "adaptive"
    ML_BASED = "ml_based"

class StrategyProfile:
    """Clase para manejar perfiles de estrategias"""
    
    def __init__(self, 
                name: str,
                trading_style: TradingStyle,
                timeframes: List[str],
                primary_timeframe: str,
                take_profit_pct: float,
                stop_loss_pct: float,
                risk_per_trade_pct: float,
                max_positions: int,
                indicators: Dict[str, Dict],
                max_spread_pct: Optional[float] = None,
                trading_hours: Optional[List[str]] = None,
                position_sizing_method: str = "fixed_risk",
                description: str = "",
                parameters: Dict[str, Any] = None):
        """
        Inicializa un perfil de estrategia
        
        Args:
            name: Nombre del perfil
            trading_style: Estilo de trading
            timeframes: Lista de timeframes a analizar
            primary_timeframe: Timeframe principal para decisiones
            take_profit_pct: Take profit en porcentaje
            stop_loss_pct: Stop loss en porcentaje
            risk_per_trade_pct: Porcentaje del balance a arriesgar por operación
            max_positions: Número máximo de posiciones simultáneas
            indicators: Diccionario de indicadores y sus parámetros
            max_spread_pct: Spread máximo como porcentaje del precio (opcional)
            trading_hours: Horas de operación (opcional, formato "HH:MM-HH:MM")
            position_sizing_method: Método de cálculo de tamaño ("fixed_risk", "fixed_size", "martingale")
            description: Descripción del perfil
            parameters: Parámetros adicionales específicos de la estrategia
        """
        self.name = name
        self.trading_style = trading_style
        self.timeframes = timeframes
        self.primary_timeframe = primary_timeframe
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_positions = max_positions
        self.indicators = indicators
        self.max_spread_pct = max_spread_pct
        self.trading_hours = trading_hours
        self.position_sizing_method = position_sizing_method
        self.description = description
        self.parameters = parameters or {}
        
        # Validación básica
        self._validate()
    
    def _validate(self):
        """Valida que los parámetros del perfil sean coherentes"""
        if self.primary_timeframe not in self.timeframes:
            raise ValueError(f"Timeframe primario '{self.primary_timeframe}' no está en la lista de timeframes")
        
        if self.take_profit_pct <= 0:
            raise ValueError(f"Take profit debe ser positivo, recibido: {self.take_profit_pct}")
        
        if self.stop_loss_pct <= 0:
            raise ValueError(f"Stop loss debe ser positivo, recibido: {self.stop_loss_pct}")
        
        if self.risk_per_trade_pct <= 0 or self.risk_per_trade_pct > 100:
            raise ValueError(f"Riesgo por operación debe estar entre 0 y 100, recibido: {self.risk_per_trade_pct}")
        
        if self.max_positions <= 0:
            raise ValueError(f"Máximo de posiciones debe ser positivo, recibido: {self.max_positions}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el perfil a diccionario para serialización
        
        Returns:
            Dict: Representación en diccionario
        """
        return {
            "name": self.name,
            "trading_style": self.trading_style.value,
            "timeframes": self.timeframes,
            "primary_timeframe": self.primary_timeframe,
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "max_positions": self.max_positions,
            "indicators": self.indicators,
            "max_spread_pct": self.max_spread_pct,
            "trading_hours": self.trading_hours,
            "position_sizing_method": self.position_sizing_method,
            "description": self.description,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyProfile':
        """
        Crea un perfil desde un diccionario
        
        Args:
            data: Diccionario con datos del perfil
            
        Returns:
            StrategyProfile: Perfil creado
        """
        # Convertir string a enum
        trading_style_str = data.pop("trading_style")
        trading_style = TradingStyle(trading_style_str)
        
        return cls(trading_style=trading_style, **data)
    
    def save_to_file(self, file_path: str) -> bool:
        """
        Guarda el perfil en un archivo JSON
        
        Args:
            file_path: Ruta de archivo
            
        Returns:
            bool: Éxito o fracaso
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            
            logger.info(f"Perfil guardado en {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error al guardar perfil: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, file_path: str) -> Optional['StrategyProfile']:
        """
        Carga un perfil desde un archivo JSON
        
        Args:
            file_path: Ruta de archivo
            
        Returns:
            Optional[StrategyProfile]: Perfil cargado o None si hay error
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            return cls.from_dict(data)
        
        except Exception as e:
            logger.error(f"Error al cargar perfil desde {file_path}: {e}")
            return None

class StrategyProfileManager:
    """Gestor de perfiles de estrategias"""
    
    def __init__(self, profiles_dir: str = "data/strategy_profiles"):
        """
        Inicializa el gestor de perfiles
        
        Args:
            profiles_dir: Directorio para guardar/cargar perfiles
        """
        self.profiles_dir = profiles_dir
        self.profiles = {}  # Dict[str, StrategyProfile]
        
        # Crear directorio si no existe
        os.makedirs(profiles_dir, exist_ok=True)
        
        # Cargar perfiles predefinidos
        self._load_predefined_profiles()
        
        # Cargar perfiles personalizados
        self._load_custom_profiles()
    
    def _load_predefined_profiles(self):
        """Carga perfiles predefinidos"""
        # Scalping
        scalping_profile = StrategyProfile(
            name="Scalping_Default",
            trading_style=TradingStyle.SCALPING,
            timeframes=["1m", "5m", "15m"],
            primary_timeframe="1m",
            take_profit_pct=0.05,  # 0.05%
            stop_loss_pct=0.07,    # 0.07%
            risk_per_trade_pct=0.5,  # 0.5% del balance
            max_positions=5,
            indicators={
                "rsi": {
                    "period": 7,
                    "overbought": 70,
                    "oversold": 30
                },
                "ema": {
                    "periods": [5, 10, 20]
                },
                "bollinger_bands": {
                    "period": 10,
                    "std_dev": 2.0
                },
                "volume": {
                    "period": 5
                }
            },
            max_spread_pct=0.02,  # 0.02% 
            trading_hours=["00:00-23:59"],  # Todo el día (crypto)
            position_sizing_method="fixed_risk",
            description="Perfil de scalping estándar con operaciones rápidas en timeframes de 1 minuto",
            parameters={
                "min_volume": 1000,
                "price_precision": 2,
                "entry_timeout_seconds": 30,
                "exit_timeout_seconds": 30,
                "max_slippage_pct": 0.01
            }
        )
        
        # Day Trading
        day_trading_profile = StrategyProfile(
            name="DayTrading_Default",
            trading_style=TradingStyle.DAY_TRADING,
            timeframes=["15m", "1h", "4h"],
            primary_timeframe="15m",
            take_profit_pct=0.3,   # 0.3%
            stop_loss_pct=0.2,     # 0.2%
            risk_per_trade_pct=1.0,  # 1% del balance
            max_positions=3,
            indicators={
                "rsi": {
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30
                },
                "macd": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9
                },
                "ema": {
                    "periods": [20, 50, 100]
                },
                "atr": {
                    "period": 14
                }
            },
            max_spread_pct=0.05,  # 0.05% 
            trading_hours=["00:00-23:59"],  # Todo el día (crypto)
            position_sizing_method="fixed_risk",
            description="Perfil de day trading estándar con operaciones intradiarias en timeframes de 15 minutos",
            parameters={
                "min_volume": 5000,
                "price_precision": 2,
                "min_atr_multiplier": 1.5,
                "exit_conditions": ["price_target", "stop_loss", "end_of_day", "signal_reversal"],
                "max_open_time_hours": 8
            }
        )
        
        # Swing Trading
        swing_trading_profile = StrategyProfile(
            name="SwingTrading_Default",
            trading_style=TradingStyle.SWING_TRADING,
            timeframes=["4h", "1d", "1w"],
            primary_timeframe="1d",
            take_profit_pct=3.0,   # 3%
            stop_loss_pct=2.0,     # 2%
            risk_per_trade_pct=1.5,  # 1.5% del balance
            max_positions=2,
            indicators={
                "rsi": {
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30
                },
                "macd": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9
                },
                "sma": {
                    "periods": [50, 100, 200]
                },
                "atr": {
                    "period": 14
                },
                "adx": {
                    "period": 14,
                    "threshold": 25
                }
            },
            max_spread_pct=0.1,  # 0.1%
            trading_hours=["00:00-23:59"],  # Todo el día (crypto)
            position_sizing_method="fixed_risk",
            description="Perfil de swing trading para operaciones de varios días basadas en tendencias más largas",
            parameters={
                "min_volume": 10000,
                "trend_confirmation_timeframes": ["4h", "1d"],
                "min_risk_reward_ratio": 1.5,
                "partial_take_profit_levels": [0.5, 0.7],  # Tomar 50% en 50% del TP, otro 30% en 70% del TP
                "max_open_time_days": 7
            }
        )
        
        # ML-Based
        ml_profile = StrategyProfile(
            name="ML_Default",
            trading_style=TradingStyle.ML_BASED,
            timeframes=["15m", "1h", "4h"],
            primary_timeframe="1h",
            take_profit_pct=0.5,   # 0.5%
            stop_loss_pct=0.3,     # 0.3%
            risk_per_trade_pct=1.0,  # 1% del balance
            max_positions=2,
            indicators={
                "ml_features": {
                    "include_technicals": True,
                    "include_price_action": True,
                    "include_volume": True,
                    "lookback_periods": [5, 10, 20]
                },
                "model_type": "random_forest",
                "prediction_threshold": 0.65
            },
            max_spread_pct=0.05,  # 0.05%
            trading_hours=["00:00-23:59"],  # Todo el día (crypto)
            position_sizing_method="fixed_risk",
            description="Perfil basado en Machine Learning que utiliza modelos para predecir movimientos de precios",
            parameters={
                "retrain_frequency_hours": 24,
                "min_prediction_confidence": 0.65,
                "feature_importance_threshold": 0.02,
                "model_params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5
                }
            }
        )
        
        # Adaptive
        adaptive_profile = StrategyProfile(
            name="Adaptive_Default",
            trading_style=TradingStyle.ADAPTIVE,
            timeframes=["5m", "15m", "1h", "4h"],
            primary_timeframe="15m",
            take_profit_pct=0.4,   # 0.4%
            stop_loss_pct=0.3,     # 0.3%
            risk_per_trade_pct=1.0,  # 1% del balance
            max_positions=3,
            indicators={
                "all_indicators": True,  # Usa todos los indicadores con pesos adaptables
                "weighting_method": "adaptive_performance",
                "market_condition_detection": True,
                "min_indicator_weight": 0.2,
                "lookback_periods": [20, 50, 100]
            },
            max_spread_pct=0.05,  # 0.05%
            trading_hours=["00:00-23:59"],  # Todo el día (crypto)
            position_sizing_method="adaptive_risk",
            description="Perfil adaptativo que ajusta pesos de indicadores y parámetros según condiciones de mercado",
            parameters={
                "learning_rate": 0.05,
                "max_weight_adjustment_pct": 20,
                "recalibration_frequency_hours": 12,
                "min_historical_trades": 50,
                "market_regimes": ["trending", "ranging", "volatile"]
            }
        )
        
        # Agregar perfiles predefinidos
        self.profiles[scalping_profile.name] = scalping_profile
        self.profiles[day_trading_profile.name] = day_trading_profile
        self.profiles[swing_trading_profile.name] = swing_trading_profile
        self.profiles[ml_profile.name] = ml_profile
        self.profiles[adaptive_profile.name] = adaptive_profile
        
        # Guardar perfiles predefinidos
        for profile in [scalping_profile, day_trading_profile, swing_trading_profile, ml_profile, adaptive_profile]:
            profile.save_to_file(os.path.join(self.profiles_dir, f"{profile.name}.json"))
    
    def _load_custom_profiles(self):
        """Carga perfiles personalizados del directorio"""
        if not os.path.exists(self.profiles_dir):
            return
        
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('.json'):
                # Evitar cargar perfiles predefinidos otra vez
                if any(name in filename for name in self.profiles.keys()):
                    continue
                
                file_path = os.path.join(self.profiles_dir, filename)
                profile = StrategyProfile.load_from_file(file_path)
                
                if profile:
                    self.profiles[profile.name] = profile
    
    def get_profile(self, name: str) -> Optional[StrategyProfile]:
        """
        Obtiene un perfil por nombre
        
        Args:
            name: Nombre del perfil
            
        Returns:
            Optional[StrategyProfile]: Perfil o None si no existe
        """
        return self.profiles.get(name)
    
    def get_profiles_by_style(self, style: TradingStyle) -> Dict[str, StrategyProfile]:
        """
        Obtiene perfiles por estilo de trading
        
        Args:
            style: Estilo de trading
            
        Returns:
            Dict[str, StrategyProfile]: Perfiles que coinciden con el estilo
        """
        return {name: profile for name, profile in self.profiles.items() 
                if profile.trading_style == style}
    
    def list_profiles(self) -> List[Dict[str, Any]]:
        """
        Lista todos los perfiles disponibles
        
        Returns:
            List[Dict]: Lista de perfiles con información básica
        """
        return [
            {
                "name": profile.name,
                "trading_style": profile.trading_style.value,
                "primary_timeframe": profile.primary_timeframe,
                "take_profit_pct": profile.take_profit_pct,
                "stop_loss_pct": profile.stop_loss_pct,
                "description": profile.description
            }
            for profile in self.profiles.values()
        ]
    
    def create_profile(self, profile: StrategyProfile) -> bool:
        """
        Crea un nuevo perfil
        
        Args:
            profile: Perfil a crear
            
        Returns:
            bool: Éxito o fracaso
        """
        # Verificar si ya existe
        if profile.name in self.profiles:
            logger.warning(f"Perfil '{profile.name}' ya existe")
            return False
        
        # Agregar perfil
        self.profiles[profile.name] = profile
        
        # Guardar en archivo
        file_path = os.path.join(self.profiles_dir, f"{profile.name}.json")
        return profile.save_to_file(file_path)
    
    def update_profile(self, profile: StrategyProfile) -> bool:
        """
        Actualiza un perfil existente
        
        Args:
            profile: Perfil con cambios
            
        Returns:
            bool: Éxito o fracaso
        """
        # Verificar si existe
        if profile.name not in self.profiles:
            logger.warning(f"Perfil '{profile.name}' no existe")
            return False
        
        # Actualizar perfil
        self.profiles[profile.name] = profile
        
        # Guardar en archivo
        file_path = os.path.join(self.profiles_dir, f"{profile.name}.json")
        return profile.save_to_file(file_path)
    
    def delete_profile(self, name: str) -> bool:
        """
        Elimina un perfil
        
        Args:
            name: Nombre del perfil
            
        Returns:
            bool: Éxito o fracaso
        """
        # Verificar si existe
        if name not in self.profiles:
            logger.warning(f"Perfil '{name}' no existe")
            return False
        
        # No permitir eliminar perfiles predefinidos
        if any(name.startswith(prefix) for prefix in ["Scalping_", "DayTrading_", "SwingTrading_", "ML_", "Adaptive_"]):
            logger.warning(f"No se puede eliminar perfil predefinido '{name}'")
            return False
        
        # Eliminar perfil
        del self.profiles[name]
        
        # Eliminar archivo
        file_path = os.path.join(self.profiles_dir, f"{name}.json")
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except Exception as e:
            logger.error(f"Error al eliminar archivo del perfil: {e}")
            return False

def get_default_profile_for_market_condition(market_condition: str, timeframe: str) -> str:
    """
    Sugiere un perfil basado en la condición de mercado actual
    
    Args:
        market_condition: Condición de mercado
        timeframe: Timeframe de análisis
        
    Returns:
        str: Nombre del perfil recomendado
    """
    # Mapeo de condiciones a perfiles
    condition_map = {
        "strong_uptrend": "DayTrading_Default",
        "moderate_uptrend": "DayTrading_Default",
        "lateral_low_vol": "SwingTrading_Default",
        "lateral_high_vol": "Scalping_Default",
        "moderate_downtrend": "DayTrading_Default",
        "strong_downtrend": "DayTrading_Default",
        "extreme_volatility": "Scalping_Default"
    }
    
    # Ajustar según timeframe
    if timeframe in ["1m", "5m"]:
        # En timeframes muy cortos, preferir scalping en general
        return "Scalping_Default"
    elif timeframe in ["15m", "30m", "1h"]:
        # En timeframes medios, usar el mapeo pero con preferencia por day trading
        return condition_map.get(market_condition, "DayTrading_Default")
    else:
        # En timeframes largos, preferir swing trading para tendencias
        if "uptrend" in market_condition or "downtrend" in market_condition:
            return "SwingTrading_Default"
        return condition_map.get(market_condition, "DayTrading_Default")

def create_profile_from_backtest_results(backtest_results: Dict, symbol: str, name: str = None) -> StrategyProfile:
    """
    Crea un perfil optimizado basado en resultados de backtesting
    
    Args:
        backtest_results: Resultados de backtesting
        symbol: Símbolo para el que se realizó el backtest
        name: Nombre para el perfil (opcional)
        
    Returns:
        StrategyProfile: Perfil optimizado
    """
    # Extraer estrategia y parámetros
    strategy = backtest_results.get("strategy", "unknown")
    best_params = backtest_results.get("best_params", {})
    interval = backtest_results.get("interval", "1h")
    
    # Crear nombre si no se proporciona
    if not name:
        timestamp = int(time.time())
        name = f"Optimized_{strategy}_{symbol}_{timestamp}"
    
    # Determinar estilo basado en timeframe
    trading_style = TradingStyle.DAY_TRADING  # Por defecto
    if interval in ["1m", "5m"]:
        trading_style = TradingStyle.SCALPING
    elif interval in ["4h", "1d"]:
        trading_style = TradingStyle.SWING_TRADING
    
    # Determinar timeframes basados en timeframe principal
    timeframes = []
    if interval == "1m":
        timeframes = ["1m", "5m", "15m"]
    elif interval == "5m":
        timeframes = ["1m", "5m", "15m", "30m"]
    elif interval == "15m":
        timeframes = ["5m", "15m", "30m", "1h"]
    elif interval == "30m":
        timeframes = ["5m", "15m", "30m", "1h"]
    elif interval == "1h":
        timeframes = ["15m", "30m", "1h", "4h"]
    elif interval == "4h":
        timeframes = ["1h", "4h", "1d"]
    else:  # 1d
        timeframes = ["4h", "1d", "1w"]
    
    # Construir configuración de indicadores basada en la estrategia
    indicators = {}
    if "sma" in strategy.lower() or "moving_average" in strategy.lower():
        indicators["sma"] = {
            "periods": [best_params.get("short_period", 10), best_params.get("long_period", 30)]
        }
    elif "rsi" in strategy.lower():
        indicators["rsi"] = {
            "period": best_params.get("period", 14),
            "overbought": best_params.get("overbought", 70),
            "oversold": best_params.get("oversold", 30)
        }
    elif "macd" in strategy.lower():
        indicators["macd"] = {
            "fast_period": best_params.get("fast_period", 12),
            "slow_period": best_params.get("slow_period", 26),
            "signal_period": best_params.get("signal_period", 9)
        }
    elif "bollinger" in strategy.lower():
        indicators["bollinger_bands"] = {
            "period": best_params.get("period", 20),
            "std_dev": best_params.get("std_dev", 2.0)
        }
    else:
        # Estrategia desconocida, usar indicadores genéricos
        indicators = {
            "rsi": {"period": 14, "overbought": 70, "oversold": 30},
            "sma": {"periods": [20, 50]},
            "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        }
    
    # Configurar take profit y stop loss basados en resultados
    metrics = backtest_results.get("best_metrics", {})
    avg_profit = metrics.get("avg_win", 0.5)  # % promedio de ganancia
    avg_loss = abs(metrics.get("avg_loss", 0.3))  # % promedio de pérdida
    
    take_profit_pct = avg_profit if 0.1 <= avg_profit <= 5.0 else 0.5
    stop_loss_pct = avg_loss if 0.1 <= avg_loss <= 3.0 else 0.3
    
    # Crear perfil optimizado
    return StrategyProfile(
        name=name,
        trading_style=trading_style,
        timeframes=timeframes,
        primary_timeframe=interval,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        risk_per_trade_pct=1.0,  # Valor conservador por defecto
        max_positions=2,  # Valor conservador por defecto
        indicators=indicators,
        max_spread_pct=0.05,  # Valor genérico
        trading_hours=["00:00-23:59"],  # Todo el día para cripto
        position_sizing_method="fixed_risk",
        description=f"Perfil optimizado basado en backtest de {strategy} para {symbol}",
        parameters=best_params
    )