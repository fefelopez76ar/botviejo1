
#!/usr/bin/env python3
"""
SolanaScalper - Bot de Trading para Solana
Versión CLI optimizada para operaciones de scalping

Uso: python main.py
"""

import os
import sys
import time
import logging
import threading
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from tabulate import tabulate
import signal

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_bot.log")
    ]
)
logger = logging.getLogger("SolanaScalper")

class ScalpingBot:
    """Bot de scalping para trading de Solana en OKX"""
    
    def __init__(self):
        self.active = False
        self.exchange = None
        self.symbol = "SOL/USDT"
        self.timeframe = "1m"  # Timeframe para scalping
        self.balance = 0.0
        self.current_position = None
        self.last_trade_price = 0.0
        self.total_trades = 0
        self.profitable_trades = 0
        self.mode = "paper"  # paper o real
        self.trades_history = []
        self.price_history = []
        self.stop_event = threading.Event()
        self.status_thread = None
        self.trading_thread = None
        
    def initialize(self):
        """Inicializa el bot y conecta con el exchange"""
        try:
            # Verificar si debemos usar modo demo
            use_demo = os.environ.get('USE_DEMO_MODE', 'true').lower() == 'true'
            
            # Obtener credenciales
            api_key = os.environ.get('OKX_API_KEY')
            api_secret = os.environ.get('OKX_API_SECRET')
            password = os.environ.get('OKX_PASSPHRASE')
            
            if not all([api_key, api_secret, password]):
                logger.error("⚠️ Credenciales de API incompletas. Configura las variables de entorno.")
                return False
            
            # Configuración optimizada para el exchange
            config = {
                'apiKey': api_key,
                'secret': api_secret,
                'password': password,
                'enableRateLimit': True,
                'timeout': 10000,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'recvWindow': 5000,
                    'warnOnFetchOpenOrdersWithoutSymbol': False
                }
            }
            
            # Configurar modo demo si es necesario
            if use_demo:
                config['options']['test'] = True
                logger.info("🔄 Configurando OKX en modo DEMO")
            
            # Inicializar exchange
            self.exchange = ccxt.okx(config)
            
            # Verificar conexión obteniendo balance
            self.update_balance()
            
            logger.info(f"✅ Bot inicializado correctamente. Modo: {'DEMO' if use_demo else 'REAL'}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando el bot: {e}")
            return False
            
    def update_balance(self):
        """Actualiza el balance desde el exchange"""
        try:
            if self.exchange:
                balance = self.exchange.fetch_balance()
                self.balance = float(balance.get('total', {}).get('USDT', 0))
                logger.info(f"💰 Balance actual: ${self.balance:.2f} USDT")
            return self.balance
        except Exception as e:
            logger.error(f"❌ Error actualizando balance: {e}")
            return 0.0
    
    def get_current_price(self) -> float:
        """Obtiene el precio actual de Solana"""
        try:
            if not self.exchange:
                return 0.0
                
            ticker = self.exchange.fetch_ticker(self.symbol)
            if ticker and 'last' in ticker and ticker['last']:
                price = float(ticker['last'])
                self.price_history.append(price)
                # Mantener solo los últimos 100 precios
                if len(self.price_history) > 100:
                    self.price_history.pop(0)
                return price
            return 0.0
        except Exception as e:
            logger.error(f"❌ Error obteniendo precio: {e}")
            return 0.0
    
    def analyze_market(self):
        """Analiza el mercado y retorna señales de trading"""
        try:
            # Obtener datos de los últimos minutos
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=30)
            if not ohlcv or len(ohlcv) < 30:
                return {"signal": "neutral", "confidence": 0.0}
                
            # Convertir a DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calcular indicadores básicos
            # RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            macd_hist = macd - signal
            
            # Determinar señal
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            current_macd = macd_hist.iloc[-1] if not pd.isna(macd_hist.iloc[-1]) else 0
            
            # Lógica de señal
            signal_strength = 0.0
            signal_type = "neutral"
            
            if current_rsi < 30 and current_macd > 0:
                signal_type = "buy"
                signal_strength = 0.7
            elif current_rsi > 70 and current_macd < 0:
                signal_type = "sell"
                signal_strength = 0.7
            elif current_macd > 0 and macd_hist.iloc[-2] < 0:
                signal_type = "buy"
                signal_strength = 0.6
            elif current_macd < 0 and macd_hist.iloc[-2] > 0:
                signal_type = "sell"
                signal_strength = 0.6
            
            return {
                "signal": signal_type,
                "confidence": signal_strength,
                "rsi": current_rsi,
                "macd": current_macd
            }
        except Exception as e:
            logger.error(f"❌ Error analizando mercado: {e}")
            return {"signal": "neutral", "confidence": 0.0}
    
    def execute_trade(self, action, price, amount):
        """Ejecuta una operación de trading"""
        try:
            if self.mode == "paper":
                # Simulación de operación
                logger.info(f"📝 [PAPER] Ejecutando orden {action} para {amount} SOL a ${price}")
                if action == "buy":
                    self.current_position = {
                        "type": "long",
                        "entry_price": price,
                        "amount": amount,
                        "timestamp": datetime.now()
                    }
                    return True
                elif action == "sell" and self.current_position:
                    profit = (price - self.current_position["entry_price"]) * self.current_position["amount"]
                    self.balance += profit
                    self.total_trades += 1
                    if profit > 0:
                        self.profitable_trades += 1
                    
                    self.trades_history.append({
                        "type": self.current_position["type"],
                        "entry": self.current_position["entry_price"],
                        "exit": price,
                        "profit": profit,
                        "timestamp": datetime.now()
                    })
                    
                    logger.info(f"💰 [PAPER] Posición cerrada. Profit: ${profit:.2f}")
                    self.current_position = None
                    return True
            else:
                # Trading real
                logger.info(f"🔴 [REAL] Ejecutando orden {action} para {amount} SOL a ${price}")
                # Aquí iría el código para ejecutar ordenes reales
                return True
                
        except Exception as e:
            logger.error(f"❌ Error ejecutando operación: {e}")
            return False
    
    def trading_loop(self):
        """Loop principal de trading"""
        while not self.stop_event.is_set():
            try:
                if not self.active:
                    time.sleep(1)
                    continue
                
                # Obtener precio actual
                current_price = self.get_current_price()
                if current_price <= 0:
                    logger.warning("⚠️ No se pudo obtener el precio actual")
                    time.sleep(5)
                    continue
                
                # Analizar mercado
                analysis = self.analyze_market()
                
                # Decidir acción
                if not self.current_position:  # No tenemos posición abierta
                    if analysis["signal"] == "buy" and analysis["confidence"] > 0.5:
                        # Calcular tamaño de posición (2% del balance)
                        position_size = min(self.balance * 0.02, self.balance)
                        amount = position_size / current_price
                        
                        self.execute_trade("buy", current_price, amount)
                        
                elif self.current_position:  # Tenemos posición abierta
                    # Calcular tiempo en posición
                    time_in_position = (datetime.now() - self.current_position["timestamp"]).total_seconds() / 60
                    
                    # Calcular profit actual
                    current_profit = (current_price - self.current_position["entry_price"]) * self.current_position["amount"]
                    profit_percent = (current_profit / (self.current_position["entry_price"] * self.current_position["amount"])) * 100
                    
                    # Reglas de salida
                    should_exit = False
                    exit_reason = ""
                    
                    # Salir si tenemos beneficio de 1% o más
                    if profit_percent >= 1.0:
                        should_exit = True
                        exit_reason = "Objetivo de beneficio alcanzado"
                        
                    # Salir si llevamos más de 15 minutos en la posición
                    elif time_in_position > 15:
                        should_exit = True
                        exit_reason = "Tiempo máximo excedido"
                        
                    # Salir si tenemos pérdida de 0.5% o más
                    elif profit_percent <= -0.5:
                        should_exit = True
                        exit_reason = "Stop loss alcanzado"
                        
                    # Salir si el análisis sugiere vender con alta confianza
                    elif analysis["signal"] == "sell" and analysis["confidence"] > 0.7:
                        should_exit = True
                        exit_reason = "Señal de venta fuerte"
                    
                    if should_exit:
                        logger.info(f"🚪 Saliendo de posición. Razón: {exit_reason}")
                        self.execute_trade("sell", current_price, self.current_position["amount"])
                
                # Pausar antes de la siguiente iteración
                time.sleep(10)  # Analizar cada 10 segundos
                
            except Exception as e:
                logger.error(f"❌ Error en loop de trading: {e}")
                time.sleep(5)
    
    def status_display_loop(self):
        """Muestra estado y estadísticas del bot"""
        while not self.stop_event.is_set():
            try:
                if not self.active:
                    time.sleep(1)
                    continue
                
                # Limpiar pantalla
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Obtener precio actual
                current_price = self.get_current_price() if self.price_history else 0
                
                # Calcular estadísticas
                win_rate = (self.profitable_trades / self.total_trades * 100) if self.total_trades > 0 else 0
                
                # Mostrar encabezado
                print("\n" + "="*70)
                print(f"  SOLANA SCALPING BOT - Modo: {'DEMO' if self.mode == 'paper' else 'REAL'}")
                print("="*70)
                
                # Mostrar información general
                print(f"\n📊 ESTADO GENERAL:")
                print(f"  • Precio actual de SOL:    ${current_price:.2f}")
                print(f"  • Balance:                 ${self.balance:.2f} USDT")
                print(f"  • Estado del bot:          {'🟢 Activo' if self.active else '🔴 Inactivo'}")
                
                # Mostrar posición actual
                print(f"\n🔍 POSICIÓN ACTUAL:")
                if self.current_position:
                    current_profit = (current_price - self.current_position["entry_price"]) * self.current_position["amount"]
                    profit_percent = (current_profit / (self.current_position["entry_price"] * self.current_position["amount"])) * 100
                    time_open = (datetime.now() - self.current_position["timestamp"]).total_seconds() / 60
                    
                    print(f"  • Tipo:                    {'LONG' if self.current_position['type'] == 'long' else 'SHORT'}")
                    print(f"  • Precio de entrada:       ${self.current_position['entry_price']:.2f}")
                    print(f"  • Cantidad:                {self.current_position['amount']:.4f} SOL")
                    print(f"  • P&L actual:              ${current_profit:.2f} ({profit_percent:.2f}%)")
                    print(f"  • Tiempo abierta:          {time_open:.1f} minutos")
                else:
                    print("  • Sin posiciones abiertas")
                
                # Mostrar estadísticas
                print(f"\n📈 ESTADÍSTICAS DE TRADING:")
                print(f"  • Operaciones totales:     {self.total_trades}")
                print(f"  • Operaciones ganadoras:   {self.profitable_trades}")
                print(f"  • Ratio de éxito:          {win_rate:.1f}%")
                
                # Mostrar últimas operaciones
                print(f"\n📜 ÚLTIMAS OPERACIONES:")
                if self.trades_history:
                    headers = ["Tipo", "Entrada", "Salida", "Beneficio", "Fecha"]
                    rows = []
                    for trade in self.trades_history[-5:]:
                        rows.append([
                            trade["type"],
                            f"${trade['entry']:.2f}",
                            f"${trade['exit']:.2f}",
                            f"${trade['profit']:.2f}",
                            trade["timestamp"].strftime("%H:%M:%S")
                        ])
                    print(tabulate(rows, headers=headers, tablefmt="simple"))
                else:
                    print("  • Sin operaciones realizadas")
                
                # Mostrar opciones de control
                print("\n🎮 CONTROLES:")
                print("  [S] Detener bot   [Q] Salir   [M] Cambiar modo")
                
                # Esperar antes de actualizar de nuevo
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"Error mostrando estado: {e}")
                time.sleep(5)
    
    def start(self):
        """Inicia el bot de trading"""
        if self.active:
            logger.warning("⚠️ El bot ya está activo")
            return
            
        # Inicializar conexión con el exchange
        if not self.exchange and not self.initialize():
            logger.error("❌ No se pudo inicializar el bot")
            return
            
        self.active = True
        self.stop_event.clear()
        
        # Iniciar thread de trading
        self.trading_thread = threading.Thread(target=self.trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        # Iniciar thread de display
        self.status_thread = threading.Thread(target=self.status_display_loop)
        self.status_thread.daemon = True
        self.status_thread.start()
        
        logger.info(f"✅ Bot iniciado en modo {self.mode}")
        
    def stop(self):
        """Detiene el bot de trading"""
        self.active = False
        logger.info("🛑 Bot detenido")
        
        # Cerrar posiciones abiertas
        if self.current_position:
            current_price = self.get_current_price()
            if current_price > 0:
                logger.info("🔄 Cerrando posiciones abiertas...")
                self.execute_trade("sell", current_price, self.current_position["amount"])
    
    def shutdown(self):
        """Cierra completamente el bot"""
        self.stop()
        self.stop_event.set()
        logger.info("👋 Bot cerrado correctamente")
    
    def set_mode(self, mode):
        """Establece el modo de trading (paper o real)"""
        if mode not in ["paper", "real"]:
            logger.warning(f"⚠️ Modo no válido: {mode}. Usando paper por defecto")
            mode = "paper"
        
        # Advertir si cambia a modo real
        if mode == "real" and self.mode == "paper":
            logger.warning("🔴 CAMBIANDO A MODO REAL - OPERARÁ CON FONDOS REALES")
        
        self.mode = mode
        logger.info(f"🔄 Modo de trading cambiado a: {mode}")
        return True

def main():
    """Función principal"""
    # Verificar credenciales
    required_env_vars = ['OKX_API_KEY', 'OKX_API_SECRET', 'OKX_PASSPHRASE']
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        print("\n⚠️  FALTAN VARIABLES DE ENTORNO NECESARIAS")
        print("Por favor, configura las siguientes variables antes de ejecutar el bot:")
        for var in missing_vars:
            print(f"  • {var}")
        print("\nPuedes configurarlas en el archivo config.env o exportarlas directamente en tu terminal.\n")
        return
    
    # Crear instancia del bot
    bot = ScalpingBot()
    
    # Manejar señales de interrupción
    def signal_handler(sig, frame):
        print("\n🛑 Deteniendo bot...")
        bot.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Iniciar el bot
    bot.start()
    
    # Bucle principal para input del usuario
    while True:
        try:
            user_input = input().lower()
            
            if user_input == 'q':
                bot.shutdown()
                break
            elif user_input == 's':
                bot.stop()
            elif user_input == 'm':
                new_mode = "real" if bot.mode == "paper" else "paper"
                bot.set_mode(new_mode)
                
        except EOFError:
            # Se puede producir al detener con Ctrl+C
            break
        except Exception as e:
            logger.error(f"Error procesando input: {e}")
    
    print("Bot finalizado")

if __name__ == "__main__":
    main()
