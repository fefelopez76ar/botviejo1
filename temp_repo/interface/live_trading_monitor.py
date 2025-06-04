"""
Monitor de trading en tiempo real para la interfaz de línea de comandos

Muestra operaciones, estado del bot, aprendizaje y datos de mercado en tiempo real
con actualización continua en la pantalla.
"""

import os
import sys
import time
import threading
import curses
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

class LiveTradingMonitor:
    """Monitor de trading en tiempo real para CLI"""
    
    def __init__(self, bot, symbol: str, timeframe: str):
        """
        Inicializa el monitor
        
        Args:
            bot: Instancia del bot de trading
            symbol: Símbolo de trading (ej. SOL-USDT)
            timeframe: Intervalo de tiempo (ej. 1m, 5m)
        """
        self.bot = bot
        self.symbol = symbol
        self.timeframe = timeframe
        self.running = False
        self.stdscr = None
        self.trade_history = []
        self.price_history = []
        self.learning_events = []
        self.max_history = 100  # Máximo de entradas a guardar
        
        # Contador para medir ciclos de aprendizaje
        self.decision_counter = 0
        self.correct_decisions = 0
        
        # Métricas de rendimiento
        self.metrics = {
            "start_time": datetime.now(),
            "trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "profit": 0.0,
            "max_drawdown": 0.0,
            "accuracy": 0.0,
            "learning_rate": 0.0,
            "current_position": None,
            "current_price": 0.0,
            "current_balance": 1000.0,  # Balance inicial
            "peak_balance": 1000.0,
            "signals_processed": 0
        }
    
    def start(self):
        """Inicia el monitor en un hilo separado"""
        self.running = True
        curses.wrapper(self._run_monitor)
    
    def stop(self):
        """Detiene el monitor"""
        self.running = False
    
    def add_trade(self, trade_data: Dict[str, Any]):
        """
        Añade información de una operación
        
        Args:
            trade_data: Datos de la operación
        """
        self.trade_history.append({
            "timestamp": datetime.now(),
            **trade_data
        })
        
        # Limitar historial
        if len(self.trade_history) > self.max_history:
            self.trade_history.pop(0)
        
        # Actualizar métricas
        self.metrics["trades"] += 1
        if trade_data.get("profit", 0) > 0:
            self.metrics["winning_trades"] += 1
        elif trade_data.get("profit", 0) < 0:
            self.metrics["losing_trades"] += 1
        
        self.metrics["profit"] += trade_data.get("profit", 0)
        self.metrics["current_balance"] += trade_data.get("profit", 0)
        
        # Actualizar peak balance
        if self.metrics["current_balance"] > self.metrics["peak_balance"]:
            self.metrics["peak_balance"] = self.metrics["current_balance"]
        
        # Calcular drawdown
        drawdown = 1 - (self.metrics["current_balance"] / self.metrics["peak_balance"])
        if drawdown > self.metrics["max_drawdown"]:
            self.metrics["max_drawdown"] = drawdown
        
        # Actualizar precisión
        if self.metrics["trades"] > 0:
            self.metrics["accuracy"] = self.metrics["winning_trades"] / self.metrics["trades"]
    
    def add_price_update(self, price: float, signal: float = 0, signal_reason: str = ""):
        """
        Añade actualización de precio
        
        Args:
            price: Precio actual
            signal: Señal actual (-1 a 1)
            signal_reason: Razón de la señal
        """
        self.price_history.append({
            "timestamp": datetime.now(),
            "price": price,
            "signal": signal,
            "reason": signal_reason
        })
        
        # Limitar historial
        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)
        
        # Actualizar métricas
        self.metrics["current_price"] = price
        self.metrics["signals_processed"] += 1 if signal != 0 else 0
    
    def add_learning_event(self, event_type: str, description: str, success: bool = True):
        """
        Añade un evento de aprendizaje
        
        Args:
            event_type: Tipo de evento (ej. 'adjustment', 'prediction')
            description: Descripción del evento
            success: Si el evento fue exitoso
        """
        self.learning_events.append({
            "timestamp": datetime.now(),
            "type": event_type,
            "description": description,
            "success": success
        })
        
        # Limitar historial
        if len(self.learning_events) > self.max_history:
            self.learning_events.pop(0)
        
        # Actualizar contador de decisiones/aprendizaje
        self.decision_counter += 1
        if success:
            self.correct_decisions += 1
        
        # Actualizar tasa de aprendizaje
        if self.decision_counter > 0:
            self.metrics["learning_rate"] = self.correct_decisions / self.decision_counter
    
    def update_position(self, position_data: Optional[Dict[str, Any]]):
        """
        Actualiza información de posición actual
        
        Args:
            position_data: Datos de la posición o None si no hay posición
        """
        self.metrics["current_position"] = position_data
    
    def _run_monitor(self, stdscr):
        """
        Ejecuta el monitor en la pantalla de curses
        
        Args:
            stdscr: Pantalla estándar de curses
        """
        self.stdscr = stdscr
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)  # Verde para positivo
        curses.init_pair(2, curses.COLOR_RED, -1)    # Rojo para negativo
        curses.init_pair(3, curses.COLOR_YELLOW, -1) # Amarillo para avisos
        curses.init_pair(4, curses.COLOR_CYAN, -1)   # Cian para títulos
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Para barras de título
        
        # Ocultar cursor
        curses.curs_set(0)
        
        # Configurar teclas
        stdscr.nodelay(True)  # No bloquear en getch()
        stdscr.timeout(100)   # Timeout para getch() en ms
        
        # Bucle principal
        while self.running:
            # Limpiar pantalla
            stdscr.clear()
            
            # Obtener dimensiones de la pantalla
            max_y, max_x = stdscr.getmaxyx()
            
            # Dibujar marcos y contenido
            self._draw_title_bar(stdscr, max_y, max_x)
            self._draw_market_info(stdscr, max_y, max_x)
            self._draw_position_info(stdscr, max_y, max_x)
            self._draw_trade_history(stdscr, max_y, max_x)
            self._draw_learning_metrics(stdscr, max_y, max_x)
            self._draw_help(stdscr, max_y, max_x)
            
            # Refrescar pantalla
            stdscr.refresh()
            
            # Comprobar teclas
            try:
                key = stdscr.getch()
                if key == ord('q'):  # Salir con 'q'
                    self.running = False
                    break
            except:
                pass
            
            # Esperar antes de la siguiente actualización
            time.sleep(0.1)
    
    def _draw_title_bar(self, stdscr, max_y: int, max_x: int):
        """Dibuja la barra de título"""
        # Crear barra de título
        stdscr.attron(curses.color_pair(5))
        stdscr.addstr(0, 0, " " * max_x)
        
        # Texto centrado
        title = f" SOLANA TRADING BOT - {self.symbol} ({self.timeframe}) "
        x_pos = (max_x - len(title)) // 2
        stdscr.addstr(0, x_pos, title)
        
        # Añadir hora actual a la derecha
        current_time = datetime.now().strftime("%H:%M:%S")
        stdscr.addstr(0, max_x - len(current_time) - 2, current_time)
        stdscr.attroff(curses.color_pair(5))
    
    def _draw_market_info(self, stdscr, max_y: int, max_x: int):
        """Dibuja información de mercado"""
        # Iniciar en línea 2 (después de la barra de título)
        y_pos = 2
        
        # Título de sección
        stdscr.attron(curses.color_pair(4))
        stdscr.addstr(y_pos, 2, "INFORMACIÓN DE MERCADO")
        stdscr.attroff(curses.color_pair(4))
        y_pos += 1
        
        # Precio actual
        current_price = self.metrics["current_price"]
        if current_price > 0:
            # Determinar dirección de precio
            price_direction = "↑" if len(self.price_history) >= 2 and self.price_history[-1]["price"] > self.price_history[-2]["price"] else "↓"
            direction_color = curses.color_pair(1) if price_direction == "↑" else curses.color_pair(2)
            
            # Mostrar precio
            stdscr.addstr(y_pos, 2, f"Precio actual: ")
            stdscr.attron(direction_color)
            stdscr.addstr(f"${current_price:.2f} {price_direction}")
            stdscr.attroff(direction_color)
        else:
            stdscr.addstr(y_pos, 2, f"Precio actual: Esperando datos...")
        
        y_pos += 1
        
        # Última señal de trading
        if self.price_history:
            last_signal = self.price_history[-1]["signal"]
            signal_text = ""
            if last_signal > 0.5:
                signal_text = "COMPRA FUERTE"
                signal_color = curses.color_pair(1)
            elif last_signal > 0:
                signal_text = "COMPRA"
                signal_color = curses.color_pair(1)
            elif last_signal < -0.5:
                signal_text = "VENTA FUERTE"
                signal_color = curses.color_pair(2)
            elif last_signal < 0:
                signal_text = "VENTA"
                signal_color = curses.color_pair(2)
            else:
                signal_text = "NEUTRAL"
                signal_color = curses.A_NORMAL
            
            stdscr.addstr(y_pos, 2, f"Señal actual: ")
            stdscr.attron(signal_color)
            stdscr.addstr(signal_text)
            stdscr.attroff(signal_color)
            
            # Razón de la señal
            if self.price_history[-1].get("reason"):
                y_pos += 1
                stdscr.addstr(y_pos, 2, f"Razón: {self.price_history[-1]['reason'][:max_x-10]}")
        else:
            stdscr.addstr(y_pos, 2, f"Señal actual: Esperando datos...")
    
    def _draw_position_info(self, stdscr, max_y: int, max_x: int):
        """Dibuja información de posición actual y métricas"""
        # Iniciar en línea 7
        y_pos = 7
        
        # Título de sección
        stdscr.attron(curses.color_pair(4))
        stdscr.addstr(y_pos, 2, "POSICIÓN Y RENDIMIENTO")
        stdscr.attroff(curses.color_pair(4))
        y_pos += 1
        
        # Balance
        profit_color = curses.color_pair(1) if self.metrics["profit"] >= 0 else curses.color_pair(2)
        stdscr.addstr(y_pos, 2, f"Balance: ${self.metrics['current_balance']:.2f} (")
        stdscr.attron(profit_color)
        stdscr.addstr(f"{'+'if self.metrics['profit']>=0 else ''}{self.metrics['profit']:.2f}")
        stdscr.attroff(profit_color)
        stdscr.addstr(")")
        y_pos += 1
        
        # Métricas de trading
        win_rate = self.metrics["winning_trades"] / self.metrics["trades"] * 100 if self.metrics["trades"] > 0 else 0
        win_rate_color = curses.color_pair(1) if win_rate >= 50 else curses.color_pair(2)
        
        stdscr.addstr(y_pos, 2, f"Operaciones: {self.metrics['trades']} (")
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr(f"✓{self.metrics['winning_trades']}")
        stdscr.attroff(curses.color_pair(1))
        stdscr.addstr(" | ")
        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(f"✗{self.metrics['losing_trades']}")
        stdscr.attroff(curses.color_pair(2))
        stdscr.addstr(") | Win rate: ")
        stdscr.attron(win_rate_color)
        stdscr.addstr(f"{win_rate:.1f}%")
        stdscr.attroff(win_rate_color)
        y_pos += 1
        
        # Drawdown
        drawdown_pct = self.metrics["max_drawdown"] * 100
        stdscr.addstr(y_pos, 2, f"Max drawdown: ")
        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(f"{drawdown_pct:.1f}%")
        stdscr.attroff(curses.color_pair(2))
        y_pos += 1
        
        # Posición actual
        y_pos += 1
        if self.metrics["current_position"]:
            pos = self.metrics["current_position"]
            pos_type = pos.get("type", "unknown")
            
            # Color según tipo de posición
            if pos_type.lower() == "long":
                pos_color = curses.color_pair(1)
                pos_symbol = "LONG ↑"
            elif pos_type.lower() == "short":
                pos_color = curses.color_pair(2)
                pos_symbol = "SHORT ↓"
            else:
                pos_color = curses.A_NORMAL
                pos_symbol = pos_type.upper()
            
            # Mostrar información de posición
            stdscr.addstr(y_pos, 2, f"Posición actual: ")
            stdscr.attron(pos_color | curses.A_BOLD)
            stdscr.addstr(pos_symbol)
            stdscr.attroff(pos_color | curses.A_BOLD)
            
            # Detalles adicionales
            y_pos += 1
            entry_price = pos.get("entry_price", 0)
            current_price = self.metrics["current_price"]
            
            # Calcular PnL si tenemos precios
            if entry_price > 0 and current_price > 0:
                if pos_type.lower() == "long":
                    pnl_pct = (current_price / entry_price - 1) * 100
                else:  # short
                    pnl_pct = (entry_price / current_price - 1) * 100
                
                pnl_color = curses.color_pair(1) if pnl_pct >= 0 else curses.color_pair(2)
                
                stdscr.addstr(y_pos, 4, f"Entrada: ${entry_price:.2f} | Actual: ${current_price:.2f} | PnL: ")
                stdscr.attron(pnl_color)
                stdscr.addstr(f"{'+'if pnl_pct>=0 else ''}{pnl_pct:.2f}%")
                stdscr.attroff(pnl_color)
        else:
            stdscr.addstr(y_pos, 2, "Posición actual: Sin posición abierta")
    
    def _draw_trade_history(self, stdscr, max_y: int, max_x: int):
        """Dibuja historial de operaciones"""
        # Iniciar en línea 15
        y_pos = 15
        
        # Título de sección
        stdscr.attron(curses.color_pair(4))
        stdscr.addstr(y_pos, 2, "HISTORIAL DE OPERACIONES RECIENTES")
        stdscr.attroff(curses.color_pair(4))
        y_pos += 1
        
        # Verificar si hay operaciones
        if not self.trade_history:
            stdscr.addstr(y_pos, 2, "No hay operaciones registradas")
            return
        
        # Encabezado de tabla
        stdscr.attron(curses.A_BOLD)
        stdscr.addstr(y_pos, 2, "HORA    TIPO      PRECIO    TAMAÑO    PROFIT    RAZÓN")
        stdscr.attroff(curses.A_BOLD)
        y_pos += 1
        
        # Mostrar últimas operaciones (máximo 5)
        max_trades = min(5, len(self.trade_history))
        for i in range(1, max_trades + 1):
            trade = self.trade_history[-i]
            
            # Formatear datos
            time_str = trade["timestamp"].strftime("%H:%M:%S")
            trade_type = trade.get("type", "unknown").upper()
            price = trade.get("price", 0)
            size = trade.get("size", 0)
            profit = trade.get("profit", 0)
            reason = trade.get("reason", "")
            
            # Determinar color según tipo/profit
            if profit > 0:
                profit_color = curses.color_pair(1)
            elif profit < 0:
                profit_color = curses.color_pair(2)
            else:
                profit_color = curses.A_NORMAL
            
            # Mostrar fila
            stdscr.addstr(y_pos, 2, time_str)
            
            # Tipo con color
            if trade_type == "BUY" or trade_type == "LONG":
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y_pos, 11, f"{trade_type:<8}")
                stdscr.attroff(curses.color_pair(1))
            elif trade_type == "SELL" or trade_type == "SHORT":
                stdscr.attron(curses.color_pair(2))
                stdscr.addstr(y_pos, 11, f"{trade_type:<8}")
                stdscr.attroff(curses.color_pair(2))
            else:
                stdscr.addstr(y_pos, 11, f"{trade_type:<8}")
            
            # Resto de datos
            stdscr.addstr(y_pos, 21, f"${price:<8.2f}")
            stdscr.addstr(y_pos, 31, f"{size:<8.4f}")
            
            # Profit con color
            stdscr.attron(profit_color)
            stdscr.addstr(y_pos, 41, f"${profit:<8.2f}")
            stdscr.attroff(profit_color)
            
            # Razón (truncada si es necesario)
            max_reason_len = max(10, max_x - 51)
            reason_trunc = reason[:max_reason_len] + ("..." if len(reason) > max_reason_len else "")
            stdscr.addstr(y_pos, 51, reason_trunc)
            
            y_pos += 1
    
    def _draw_learning_metrics(self, stdscr, max_y: int, max_x: int):
        """Dibuja métricas de aprendizaje"""
        # Iniciar en línea 22
        y_pos = 22
        
        # Título de sección
        stdscr.attron(curses.color_pair(4))
        stdscr.addstr(y_pos, 2, "SISTEMA DE APRENDIZAJE")
        stdscr.attroff(curses.color_pair(4))
        y_pos += 1
        
        # Mostrar métricas de aprendizaje
        learning_rate = self.metrics["learning_rate"] * 100
        stdscr.addstr(y_pos, 2, f"Tasa de aprendizaje: ")
        
        # Color según tasa
        if learning_rate >= 70:
            stdscr.attron(curses.color_pair(1))
        elif learning_rate >= 50:
            stdscr.attron(curses.color_pair(3))
        else:
            stdscr.attron(curses.color_pair(2))
        
        stdscr.addstr(f"{learning_rate:.1f}%")
        stdscr.attroff(curses.color_pair(1) | curses.color_pair(2) | curses.color_pair(3))
        
        stdscr.addstr(f" ({self.correct_decisions}/{self.decision_counter} decisiones correctas)")
        y_pos += 1
        
        # Mostrar señales procesadas
        stdscr.addstr(y_pos, 2, f"Señales procesadas: {self.metrics['signals_processed']}")
        y_pos += 1
        
        # Eventos de aprendizaje recientes
        y_pos += 1
        stdscr.attron(curses.A_BOLD)
        stdscr.addstr(y_pos, 2, "EVENTOS RECIENTES DE APRENDIZAJE:")
        stdscr.attroff(curses.A_BOLD)
        y_pos += 1
        
        # Mostrar eventos (máximo 3)
        if not self.learning_events:
            stdscr.addstr(y_pos, 2, "No hay eventos de aprendizaje registrados")
            return
        
        max_events = min(3, len(self.learning_events))
        for i in range(1, max_events + 1):
            event = self.learning_events[-i]
            
            # Formatear datos
            time_str = event["timestamp"].strftime("%H:%M:%S")
            event_type = event.get("type", "").upper()
            description = event.get("description", "")
            success = event.get("success", True)
            
            # Mostrar con color según éxito
            stdscr.addstr(y_pos, 2, f"{time_str} - ")
            
            # Tipo con color
            stdscr.attron(curses.A_BOLD)
            stdscr.addstr(f"{event_type}: ")
            stdscr.attroff(curses.A_BOLD)
            
            # Descripción con color de éxito/fracaso
            color = curses.color_pair(1) if success else curses.color_pair(2)
            stdscr.attron(color)
            
            # Truncar si es necesario
            max_desc_len = max(20, max_x - 25)
            desc_trunc = description[:max_desc_len] + ("..." if len(description) > max_desc_len else "")
            stdscr.addstr(desc_trunc)
            
            stdscr.attroff(color)
            y_pos += 1
    
    def _draw_help(self, stdscr, max_y: int, max_x: int):
        """Dibuja información de ayuda"""
        # Dibujar en la última línea
        help_text = "Presiona 'q' para salir"
        stdscr.attron(curses.A_DIM)
        stdscr.addstr(max_y - 1, 2, help_text)
        stdscr.attroff(curses.A_DIM)

def run_live_monitor(bot, symbol: str, timeframe: str):
    """
    Función para ejecutar el monitor en un hilo separado
    
    Args:
        bot: Instancia del bot de trading
        symbol: Símbolo de trading
        timeframe: Intervalo de tiempo
    """
    monitor = LiveTradingMonitor(bot, symbol, timeframe)
    
    # Crear hilo para el monitor
    monitor_thread = threading.Thread(target=monitor.start)
    monitor_thread.daemon = True  # El hilo terminará cuando el hilo principal termine
    monitor_thread.start()
    
    return monitor

def simulate_trading_for_demo(monitor, duration_seconds: int = 60):
    """
    Simula actividad de trading para demostración
    
    Args:
        monitor: Instancia de LiveTradingMonitor
        duration_seconds: Duración de la simulación en segundos
    """
    import random
    import time
    from datetime import datetime, timedelta
    
    # Precio base y posición actual
    base_price = 170.0  # SOL-USDT
    position = None
    balance = 1000.0
    
    # Estrategias para selección aleatoria
    strategies = ["RSI Scalping", "Momentum Scalping", "Grid Trading"]
    
    # Nombres para razones de trading
    entry_reasons = [
        "RSI en sobreventa",
        "Impulso alcista detectado",
        "Soporte confirmado en EMA",
        "Patrón de doble suelo",
        "Señal de compra MACD",
        "Volumen incrementando",
        "Divergencia alcista detectada"
    ]
    
    exit_reasons = [
        "Toma de beneficios",
        "RSI en sobrecompra",
        "Stop loss activado",
        "Señal técnica bajista",
        "Objetivo de precio alcanzado",
        "Reversión de tendencia",
        "Gestión de riesgo proactiva",
        "Pattern de agotamiento de impulso"
    ]
    
    learning_types = [
        "AJUSTE",
        "PREDICCIÓN",
        "OPTIMIZACIÓN", 
        "RECONOCIMIENTO",
        "ADAPTACIÓN"
    ]
    
    learning_descriptions = [
        "Ajuste de parámetros RSI para condiciones actuales",
        "Optimización de toma de beneficios según volatilidad",
        "Reconocimiento de patrón de reversión",
        "Calibración temporal de indicadores",
        "Detección de falsa ruptura",
        "Mejora en timing de entrada",
        "Reducción de tiempo en posiciones perdedoras",
        "Adaptación a volatilidad del mercado",
        "Incremento de precisión en tendencia lateral",
        "Reconocimiento de cambio macroeconómico",
        "Ajuste de entrada secuencial",
        "Reducción de drawdown máximo",
        "Optimización basada en volumen",
        "Mejora de salida en breakeven"
    ]
    
    # Tiempo de inicio
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    try:
        # Bucle principal de simulación
        while time.time() < end_time and monitor.running:
            # Generar movimiento de precio
            price_change = random.normalvariate(0, 0.002)  # 0.2% de volatilidad
            current_price = base_price * (1 + price_change)
            base_price = current_price  # Actualizar para próxima iteración
            
            # Actualizar precio en el monitor
            signal = random.uniform(-0.8, 0.8)
            signal_reason = ""
            
            # Generar señal significativa ocasionalmente
            significant_signal = random.random() < 0.15  # 15% de probabilidad
            if significant_signal:
                if signal > 0:
                    signal = random.uniform(0.5, 1.0)
                    signal_reason = random.choice(entry_reasons)
                else:
                    signal = random.uniform(-1.0, -0.5)
                    signal_reason = random.choice(exit_reasons)
            
            # Actualizar precio
            monitor.add_price_update(current_price, signal, signal_reason)
            
            # Gestionar posición
            if position is None and signal > 0.5:
                # Abrir posición larga
                position = {
                    "type": "long",
                    "entry_price": current_price,
                    "size": round(random.uniform(0.1, 0.5), 4),
                    "time": datetime.now(),
                    "strategy": random.choice(strategies)
                }
                monitor.update_position(position)
                
                # Registrar operación
                monitor.add_trade({
                    "type": "BUY",
                    "price": current_price,
                    "size": position["size"],
                    "profit": 0,
                    "reason": signal_reason or "Señal de compra"
                })
                
                # Evento de aprendizaje relacionado
                learning_type = random.choice(learning_types)
                if not signal_reason:
                    description = random.choice(learning_descriptions)
                else:
                    description = f"Análisis de entrada: {signal_reason}"
                
                monitor.add_learning_event(learning_type, description, True)
                
            elif position is not None and (signal < -0.5 or random.random() < 0.1):
                # Cerrar posición
                exit_price = current_price
                position_type = position["type"]
                entry_price = position["entry_price"]
                size = position["size"]
                
                # Calcular beneficio
                if position_type == "long":
                    profit = (exit_price - entry_price) * size
                else:
                    profit = (entry_price - exit_price) * size
                
                # Aplicar comisión simulada
                profit -= (exit_price * size * 0.001)  # 0.1% comisión
                
                # Registrar cierre
                exit_reason = random.choice(exit_reasons) if not signal_reason else signal_reason
                monitor.add_trade({
                    "type": "SELL",
                    "price": exit_price,
                    "size": size,
                    "profit": profit,
                    "reason": exit_reason
                })
                
                # Evento de aprendizaje relacionado al resultado
                learning_success = profit > 0
                learning_type = random.choice(learning_types)
                
                if learning_success:
                    descriptions = [
                        f"Operación exitosa confirmada con {position['strategy']}",
                        f"Confirmación de efectividad en {exit_reason}",
                        f"Optimización de parámetros para capturar movimiento",
                        f"Validación positiva de modelo predictivo"
                    ]
                else:
                    descriptions = [
                        f"Ajuste necesario en {position['strategy']} tras pérdida",
                        f"Detección de falso positivo en señal de entrada",
                        f"Recalibración de niveles de salida",
                        f"Reducción de sensibilidad en señales falsas"
                    ]
                
                learning_desc = random.choice(descriptions)
                monitor.add_learning_event(learning_type, learning_desc, learning_success)
                
                # Resetear posición
                position = None
                monitor.update_position(None)
            
            # Eventos aleatorios de aprendizaje (independientes de operaciones)
            if random.random() < 0.1:  # 10% de probabilidad
                learning_type = random.choice(learning_types)
                description = random.choice(learning_descriptions)
                success = random.random() < 0.8  # 80% de éxito
                
                monitor.add_learning_event(learning_type, description, success)
            
            # Esperar
            time.sleep(random.uniform(0.5, 2.0))
            
    except KeyboardInterrupt:
        monitor.stop()
    
    return monitor