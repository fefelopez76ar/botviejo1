import logging
from collections import deque
import pandas as pd
from typing import Dict, Any, Deque, Optional, Callable

# Configurar logging para este módulo
logger = logging.getLogger("SignalEngine")

# --- Concepto Técnico: DummyMachineLearningStrategy ---
# Esta clase simula lo que sería tu estrategia de Machine Learning real.
# Por ahora, simplemente genera una señal de prueba.
# En el futuro, aquí cargarías un modelo entrenado (ej. model_rf.pkl)
# y usarías indicadores técnicos para hacer predicciones reales.
class DummyMachineLearningStrategy:
    def __init__(self):
        logger.info("Cargando modelo ML dummy (simulado)...")
        # Aquí iría la lógica real para cargar un modelo, por ejemplo:
        # import joblib
        # self.model = joblib.load('models/model_rf.pkl')
        pass

    def predict(self, features: pd.DataFrame) -> str:
        """
        Simula la predicción de una señal de trading (BUY/SELL/HOLD).
        En una implementación real, 'features' serían los indicadores/datos de entrada para el modelo ML.
        """
        if features.empty:
            return "HOLD"
        
        # Lógica de predicción dummy: compra si el último precio subió, vende si bajó.
        # Esto es solo para tener algo que genere señales para la prueba de flujo.
        last_price = features['close'].iloc[-1]
        prev_price = features['close'].iloc[-2] if len(features) > 1 else last_price

        if last_price > prev_price:
            return "BUY"
        elif last_price < prev_price:
            return "SELL"
        else:
            return "HOLD"

# --- Clase Principal: SignalEngine ---
class SignalEngine:
    def __init__(self, instrument_id: str, data_buffer_size: int = 100,
                 on_signal_generated: Optional[Callable[[str, str, float, Any], None]] = None):
        self.instrument_id = instrument_id
        # --- Concepto Técnico: deque (cola de doble extremo) ---
        # data_buffer es una "memoria" que guarda los últimos 'data_buffer_size' ticks.
        # 'deque' es eficiente para añadir y quitar elementos de los extremos.
        self.data_buffer: Deque[Dict[str, Any]] = deque(maxlen=data_buffer_size)
        self.ml_strategy: Optional[DummyMachineLearningStrategy] = None
        self._load_ml_strategy()
        # --- Concepto Técnico: Callback (on_signal_generated) ---
        # on_signal_generated es una función que se llamará cuando el SignalEngine genere una señal.
        # Esto permite que el TradeExecutor "escuche" las señales del SignalEngine sin que este sepa
        # directamente del TradeExecutor, manteniendo los módulos separados y limpios.
        self.on_signal_generated = on_signal_generated 

        logger.info(f"SignalEngine inicializado para {instrument_id} con buffer de tamaño {data_buffer_size}.")

    def _load_ml_strategy(self):
        """
        Intenta cargar la estrategia de Machine Learning.
        Si el modelo no se encuentra, el motor continuará sin ML activo.
        """
        try:
            self.ml_strategy = DummyMachineLearningStrategy()
            logger.info("Estrategia de Machine Learning cargada exitosamente (dummy).")
        except FileNotFoundError:
            logger.warning("Modelo ML 'model_rf.pkl' no encontrado. SignalEngine operará sin estrategia ML.")
            self.ml_strategy = None
        except Exception as e:
            logger.error(f"Error al cargar la estrategia ML: {e}")
            self.ml_strategy = None

    def process_data(self, data: Dict[str, Any]):
        """
        Procesa nuevos datos de mercado, los añade al buffer y genera señales si es necesario.
        """
        self.data_buffer.append(data)
        logger.debug(f"Datos de {self.instrument_id} añadidos al buffer. Tamaño: {len(self.data_buffer)}")

        # Generar señal cada vez que el buffer está lleno o cada 10 ticks (para probar)
        if len(self.data_buffer) == self.data_buffer.maxlen or len(self.data_buffer) % 10 == 0:
            self._generate_signal()

    def _generate_signal(self):
        """
        Genera una señal de trading basándose en los datos del buffer y la estrategia ML.
        """
        if not self.data_buffer:
            return

        df = pd.DataFrame(list(self.data_buffer))

        # Asegurarse de que el DataFrame tenga la columna 'close' para la estrategia dummy
        if 'last_price' in df.columns:
            df['close'] = df['last_price']
        elif 'price' in df.columns:
            df['close'] = df['price']
        else:
            logger.warning("Columna 'close' o 'last_price' no encontrada en los datos para SignalEngine. No se generará señal ML.")
            return

        signal = "HOLD" # Por defecto, la señal es HOLD
        # Si hay una estrategia ML y suficientes datos, intentar predecir
        if self.ml_strategy and len(df) >= 2: # Necesitamos al menos 2 ticks para la lógica dummy
            signal = self.ml_strategy.predict(df)
        else:
            logger.debug("No hay estrategia ML cargada o datos insuficientes para generar señal.")
            
        # Si la señal no es HOLD y hay un callback, notificar
        if signal != "HOLD" and self.on_signal_generated:
            current_price = df['close'].iloc[-1]
            logger.info(f"[Signal Notifier] Nueva señal: {signal} para {self.instrument_id} a precio: {current_price:.2f}")
            # Llama a la función de callback, que será execute_order del TradeExecutor
            self.on_signal_generated(signal, self.instrument_id, current_price, df)
        elif signal == "HOLD":
            logger.debug(f"SignalEngine: No se generó señal activa (HOLD) para {self.instrument_id}.")
