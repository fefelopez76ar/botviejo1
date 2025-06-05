import logging
from typing import Dict, Any, Optional

# Configurar logging para este módulo
logger = logging.getLogger("TradeExecutor")

# --- Clase Principal: TradeExecutor ---
class TradeExecutor:
    def __init__(self, initial_balance: float = 10000.0, risk_per_trade_pct: float = 0.01):
        self.balance = initial_balance
        self.instrument_positions: Dict[str, float] = {}  # {'SOL-USDT': cantidad}
        self.risk_per_trade_pct = risk_per_trade_pct # Porcentaje de riesgo por operación
        self.trade_history: list = [] # Para registrar operaciones
        logger.info(f"TradeExecutor inicializado con balance: {self.balance:.2f} y riesgo por operación: {risk_per_trade_pct*100:.2f}%")

    def _calculate_position_size(self, current_price: float) -> float:
        """
        Calcula el tamaño de la posición basado en el balance y el riesgo por operación.
        --- Concepto Técnico: Gestión de Riesgo Básica ---
        Aquí definimos cuánto capital estamos dispuestos a arriesgar en un solo trade.
        Si risk_per_trade_pct es 0.01 (1%), y tu balance es 10000, entonces el capital
        en riesgo en esta operación sería 100.
        El tamaño de la posición se calcula para que esa cantidad represente un porcentaje
        razonable de tu balance, asumiendo que el stop-loss se definirá más adelante.
        Por ahora, es un tamaño fijo basado en un porcentaje de tu balance total.
        """
        capital_at_risk = self.balance * self.risk_per_trade_pct
        # Por simplicidad, asumimos que este capital_at_risk se usará para una posición.
        # En un sistema real, esto se ajustaría por stop-loss y volatilidad.
        # Aquí, el tamaño de la posición es el capital a arriesgar dividido por el precio actual.
        # Esto significa que el capital_at_risk define el valor monetario de la operación.
        position_size = capital_at_risk / current_price
        logger.debug(f"Calculando tamaño de posición: Capital a arriesgar={capital_at_risk:.2f}, Precio={current_price:.2f}, Tamaño={position_size:.4f}")
        return position_size

    async def execute_order(self, signal: str, instrument_id: str, current_price: float, data_context: Any):
        """
        Ejecuta una orden de compra o venta en modo simulado.
        Esta función es el "callback" que se le pasa al SignalEngine.
        """
        current_position = self.instrument_positions.get(instrument_id, 0.0)
        
        logger.info(f"TradeExecutor recibió señal: {signal} para {instrument_id} @ {current_price:.2f}. Posición actual: {current_position:.4f}")

        if signal == "BUY":
            if current_position <= 0: # Solo compramos si no tenemos una posición larga o estamos en corto
                size = self._calculate_position_size(current_price)
                cost = size * current_price
                if self.balance >= cost:
                    self.balance -= cost
                    self.instrument_positions[instrument_id] = current_position + size
                    self.trade_history.append({"type": "BUY", "instrument": instrument_id, "price": current_price, "size": size, "balance_after": self.balance})
                    logger.info(f"SIMULATED BUY: {size:.4f} de {instrument_id} @ {current_price:.2f}. Balance restante: {self.balance:.2f}")
                else:
                    logger.warning(f"SIMULATED BUY FAILED: Fondos insuficientes para comprar {size:.4f} de {instrument_id}.")
            else:
                logger.debug(f"SIMULATED BUY SKIPPED: Ya en posición larga para {instrument_id}. Posición: {current_position:.4f}")

        elif signal == "SELL":
            if current_position >= 0: # Solo vendemos si no tenemos una posición corta o estamos en largo
                size = self._calculate_position_size(current_price) # Mismo tamaño para venta simulada
                # En un caso real, SELL podría significar cerrar una posición LONG o abrir un SHORT.
                # Aquí, por simplicidad, simulamos cerrar una posición larga o abrir un corto si no tenemos posición.
                # Si tenemos una posición larga, la cerramos parcial o totalmente.
                # Si no tenemos posición, simulamos una venta en corto.
                
                # Para la dummy strategy, SELL solo se ejecutará cuando no tengamos una posición en largo activa.
                # En un sistema real, manejarías cierres de posición y apertura de cortos por separado.
                if current_position > 0: # Si tenemos una posición larga abierta, cerramos
                    size_to_close = min(size, current_position) # No vender más de lo que tenemos
                    profit = size_to_close * current_price
                    self.balance += profit
                    self.instrument_positions[instrument_id] = current_position - size_to_close
                    self.trade_history.append({"type": "SELL_CLOSE", "instrument": instrument_id, "price": current_price, "size": size_to_close, "balance_after": self.balance})
                    logger.info(f"SIMULATED SELL (CLOSE): Cerrando {size_to_close:.4f} de {instrument_id} @ {current_price:.2f}. Balance restante: {self.balance:.2f}")
                    if self.instrument_positions[instrument_id] == 0:
                        logger.info(f"Posición en {instrument_id} completamente cerrada.")
                else: # Si no tenemos posición larga, simulamos una venta en corto
                    profit = size * current_price # Simulamos que al vender en corto, recibimos el dinero
                    self.balance += profit
                    self.instrument_positions[instrument_id] = -size # Posición negativa para corto
                    self.trade_history.append({"type": "SELL_SHORT", "instrument": instrument_id, "price": current_price, "size": size, "balance_after": self.balance})
                    logger.info(f"SIMULATED SELL (SHORT): Abriendo corto por {size:.4f} de {instrument_id} @ {current_price:.2f}. Balance actual: {self.balance:.2f}")
            else:
                logger.debug(f"SIMULATED SELL SKIPPED: Ya en posición corta para {instrument_id}. Posición: {current_position:.4f}")

        elif signal == "HOLD":
            logger.debug(f"SIMULATED HOLD: No se realiza ninguna operación para {instrument_id}.")

    def get_status(self) -> Dict[str, Any]:
        """Devuelve el estado actual del TradeExecutor."""
        return {
            "balance": self.balance,
            "positions": self.instrument_positions,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "total_trades": len(self.trade_history)
        }
