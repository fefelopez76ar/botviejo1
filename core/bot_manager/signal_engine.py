import logging
from collections import deque
import pandas as pd
from typing import Dict, Any, Deque, Optional, Callable
import random # Importar random para la estrategia dummy

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
        if features.empty or len(features) < 2: # Asegurarse de tener al menos 2 puntos para comparar
            return "HOLD"
        
        # Lógica de predicción dummy: compra si el último precio subió, vende si bajó.
        # Esto es solo para tener algo que genere señales para la prueba de flujo.
        try:
            last_price = float(features['close'].iloc[-1])
            prev_price = float(features['close'].iloc[-2])
        except (KeyError, IndexError, ValueError):
            logger.warning("No se pudo obtener el precio 'close' de las features para la predicción dummy.")
            return "HOLD"

        if last_price > prev_price:
            return "BUY"
        elif last_price < prev_price:
            return "SELL"
        else:
            return "HOLD"

class SignalEngine:
    def __init__(self, instrument_id: str, data_buffer_size: int = 50,
                 ml_strategy: Optional[Any] = None,
                 on_signal_generated: Optional[Callable[[str, str, float, pd.DataFrame], Any]] = None):
        self.instrument_id = instrument_id
        self.data_buffer: Deque[Dict[str, Any]] = deque(maxlen=data_buffer_size)
        self.ml_strategy = ml_strategy if ml_strategy else DummyMachineLearningStrategy()
        self.on_signal_generated = on_signal_generated
        logger.info(f"SignalEngine inicializado para {instrument_id} con buffer de tamaño {data_buffer_size}.")

    async def process_data(self, data: Dict[str, Any]):
        # Solo procesar si son datos de ticker y para el instrumento correcto
        if data.get('arg', {}).get('channel') == 'tickers' and \
           data.get('arg', {}).get('instId') == self.instrument_id:
            
            ticker_data = data.get('data')
            if not ticker_data or not isinstance(ticker_data, list) or not ticker_data[0]:
                logger.warning("Datos de ticker no válidos o vacíos.")
                return

            # Tomar el primer elemento de la lista 'data'
            data_point = ticker_data[0]

            # Convertir todos los valores numéricos importantes a float si no lo están ya
            # Y asegurar que 'ts' (timestamp) se añada al buffer
            processed_data_point = {
                'instId': data_point.get('instId'),
                'ts': int(data_point.get('ts')), # Asegurar que es int
                'open': float(data_point.get('open', 0)) if data_point.get('open') else None,
                'high': float(data_point.get('high24h', 0)) if data_point.get('high24h') else None, # Usar high24h para simulación
                'low': float(data_point.get('low24h', 0)) if data_point.get('low24h') else None, # Usar low24h para simulación
                'volCcy24h': float(data_point.get('volCcy24h', 0)) if data_point.get('volCcy24h') else None, # Volumen de moneda
                'vol24h': float(data_point.get('vol24h', 0)) if data_point.get('vol24h') else None, # Volumen base
                'askPx': float(data_point.get('askPx', 0)) if data_point.get('askPx') else None,
                'bidPx': float(data_point.get('bidPx', 0)) if data_point.get('bidPx') else None,
                # La clave 'last' es la que usamos para el precio de cierre en la simulación
                'last': float(data_point.get('last')) if data_point.get('last') else None 
            }
            # Filtrar None values si no se esperan (depende de cómo uses los datos después)
            processed_data_point = {k: v for k, v in processed_data_point.items() if v is not None}

            self.data_buffer.append(processed_data_point)

            # Convertir buffer a DataFrame para procesamiento
            df = pd.DataFrame(list(self.data_buffer))
            df['ts'] = pd.to_datetime(df['ts'], unit='ms') # Convertir timestamp a datetime

            # Asegurarse de que el DataFrame tenga la columna 'close' para la estrategia dummy
            # Preferimos 'last' (de OKX tickers), luego 'last_price', luego 'price'
            if 'last' in df.columns:
                df['close'] = df['last']
            elif 'last_price' in df.columns:
                df['close'] = df['last_price']
            elif 'price' in df.columns:
                df['close'] = df['price']
            else:
                logger.warning("Columna 'close', 'last_price' o 'last' no encontrada en los datos para SignalEngine. No se generará señal ML.")
                return

            signal = "HOLD" # Por defecto, la señal es HOLD
            # Si hay una estrategia ML y suficientes datos, intentar predecir
            if self.ml_strategy and len(df) >= 2: # Necesitamos al menos 2 ticks para la lógica dummy
                try:
                    signal = self.ml_strategy.predict(df)
                except Exception as e:
                    logger.error(f"Error al predecir señal con estrategia ML: {e}")
                    signal = "HOLD" # Fallback a HOLD en caso de error

            # Si la señal no es HOLD y hay un callback, notificar
            if signal != "HOLD" and self.on_signal_generated:
                try:
                    # Asegurarse de que 'close' exista y sea numérica antes de acceder
                    if 'close' in df.columns and pd.api.types.is_numeric_dtype(df['close']):
                        current_price = df['close'].iloc[-1]
                        logger.info(f"[Signal Notifier] Nueva señal: {signal} para {self.instrument_id} a precio: {current_price:.2f}")
                        # Llama a la función de callback, que será execute_order del TradeExecutor
                        await self.on_signal_generated(signal, self.instrument_id, current_price, df)
                    else:
                        logger.warning("Columna 'close' no numérica o no encontrada para notificar señal.")
                except Exception as e:
                    logger.error(f"Error al notificar señal generada: {e}")
        # else:
        #     logger.debug(f"Datos recibidos no son de ticker o no son para {self.instrument_id}. Ignorando.")