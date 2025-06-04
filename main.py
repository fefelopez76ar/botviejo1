#!/usr/bin/env python3
"""
Bot de Trading con Aprendizaje - Modo Continuo
Ejecuta en paper trading sin riesgo real
"""
import os
import sys
import asyncio
import logging
import json
import pandas as pd
import numpy as np
import threading
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LearningBot")

# Cargar configuración
config_path = Path("config.env")
load_dotenv(dotenv_path=config_path)

# Importar módulos del bot
from api_client.modulocola import data_queue
from api_client.modulo2 import OKXWebSocketClient

# Flask app para monitoreo
app = Flask(__name__)

class PaperTradingEngine:
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.position = None
        self.trades = []
        self.market_data = []
        self.learning_data = []
        self.status = "Iniciando..."
        
    def analyze_market(self, candle_data):
        if len(self.market_data) < 20:
            return None
            
        df = pd.DataFrame(self.market_data)
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        
        current_price = float(candle_data['close'])
        sma_5 = df['sma_5'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        
        signal = None
        
        if self.position is None:
            if sma_5 > sma_20 and rsi < 70:
                signal = "BUY"
            elif sma_5 < sma_20 and rsi > 30:
                signal = "SELL"
        else:
            if self.position['type'] == 'LONG':
                if current_price >= self.position['entry_price'] * 1.01:
                    signal = "CLOSE_LONG_PROFIT"
                elif current_price <= self.position['entry_price'] * 0.99:
                    signal = "CLOSE_LONG_LOSS"
            elif self.position['type'] == 'SHORT':
                if current_price <= self.position['entry_price'] * 0.99:
                    signal = "CLOSE_SHORT_PROFIT"
                elif current_price >= self.position['entry_price'] * 1.01:
                    signal = "CLOSE_SHORT_LOSS"
        
        return {
            'signal': signal,
            'price': current_price,
            'sma_5': sma_5,
            'sma_20': sma_20,
            'rsi': rsi
        }
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def execute_trade(self, analysis):
        signal = analysis['signal']
        price = analysis['price']
        
        if signal == "BUY" and self.position is None:
            self.position = {
                'type': 'LONG',
                'entry_price': price,
                'entry_time': datetime.now(),
                'size': self.balance * 0.1
            }
            logger.info(f"COMPRA ejecutada a ${price:.4f}")
            
        elif signal == "SELL" and self.position is None:
            self.position = {
                'type': 'SHORT',
                'entry_price': price,
                'entry_time': datetime.now(),
                'size': self.balance * 0.1
            }
            logger.info(f"VENTA ejecutada a ${price:.4f}")
            
        elif signal and "CLOSE" in signal:
            if self.position:
                profit = self.calculate_profit(price)
                self.balance += profit
                
                trade_result = {
                    'entry_price': self.position['entry_price'],
                    'exit_price': price,
                    'profit': profit,
                    'type': self.position['type'],
                    'signal': signal,
                    'duration': (datetime.now() - self.position['entry_time']).seconds,
                    'analysis_data': analysis
                }
                
                self.trades.append(trade_result)
                self.learn_from_trade(trade_result)
                
                logger.info(f"Posición cerrada: {profit:+.2f} USDT (Balance: {self.balance:.2f})")
                self.position = None
    
    def calculate_profit(self, exit_price):
        if not self.position:
            return 0
            
        size = self.position['size']
        entry_price = self.position['entry_price']
        
        if self.position['type'] == 'LONG':
            return size * (exit_price - entry_price) / entry_price
        else:
            return size * (entry_price - exit_price) / entry_price
    
    def learn_from_trade(self, trade_result):
        success = trade_result['profit'] > 0
        
        learning_record = {
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'profit': trade_result['profit'],
            'indicators': {
                'sma_5': trade_result['analysis_data']['sma_5'],
                'sma_20': trade_result['analysis_data']['sma_20'],
                'rsi': trade_result['analysis_data']['rsi']
            }
        }
        
        self.learning_data.append(learning_record)
        self.save_learning_data()
        
        if len(self.trades) % 5 == 0:
            self.analyze_performance()
    
    def save_learning_data(self):
        try:
            with open('learning_data.json', 'w') as f:
                json.dump(self.learning_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando datos: {e}")
    
    def analyze_performance(self):
        if not self.trades:
            return
            
        profitable_trades = [t for t in self.trades if t['profit'] > 0]
        total_profit = sum(t['profit'] for t in self.trades)
        win_rate = len(profitable_trades) / len(self.trades) * 100
        
        logger.info(f"ESTADÍSTICAS:")
        logger.info(f"  Operaciones: {len(self.trades)}")
        logger.info(f"  Tasa éxito: {win_rate:.1f}%")
        logger.info(f"  Ganancia: {total_profit:+.2f} USDT")
        logger.info(f"  ROI: {(self.balance/self.initial_balance-1)*100:+.2f}%")
    
    def get_stats(self):
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'roi': 0,
                'balance': self.balance
            }
            
        profitable_trades = [t for t in self.trades if t['profit'] > 0]
        total_profit = sum(t['profit'] for t in self.trades)
        win_rate = len(profitable_trades) / len(self.trades) * 100
        
        return {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'roi': (self.balance/self.initial_balance-1)*100,
            'balance': self.balance,
            'position': self.position,
            'status': self.status
        }

# Instancia global del motor de trading
trading_engine = PaperTradingEngine()

class LearningTradingBot:
    def __init__(self):
        self.engine = trading_engine
        self.ws_client = None
        self.running = False
        
    async def connect_to_market(self):
        api_key = os.getenv("OKX_API_KEY")
        secret_key = os.getenv("OKX_API_SECRET")
        passphrase = os.getenv("OKX_PASSPHRASE")
        
        if not all([api_key, secret_key, passphrase]):
            logger.error("Credenciales OKX no encontradas")
            return False
            
        self.ws_client = OKXWebSocketClient(api_key, secret_key, passphrase, data_queue)
        self.ws_client.ws_url = "wss://ws.okx.com:8443/ws/v5/business"
        
        try:
            await self.ws_client.connect()
            await self.ws_client.subscribe([
                {"channel": "candle1m", "instId": "SOL-USDT"}
            ])
            logger.info("Conectado a datos de mercado")
            self.engine.status = "Conectado - Esperando datos"
            return True
        except Exception as e:
            logger.error(f"Error conectando: {e}")
            self.engine.status = f"Error: {e}"
            return False
    
    async def run_learning_mode(self):
        logger.info("Iniciando modo aprendizaje automático")
        logger.info("Paper Trading activado - Sin riesgo real")
        
        if not await self.connect_to_market():
            return
            
        self.running = True
        last_candle_time = None
        self.engine.status = "Analizando mercado"
        
        while self.running:
            try:
                if not data_queue.empty():
                    data = data_queue.get_nowait()
                    
                    if 'data' in data and data['data']:
                        candle_data = data['data'][0]
                        
                        processed_candle = {
                            'timestamp': candle_data[0],
                            'open': float(candle_data[1]),
                            'high': float(candle_data[2]),
                            'low': float(candle_data[3]),
                            'close': float(candle_data[4]),
                            'volume': float(candle_data[5])
                        }
                        
                        if processed_candle['timestamp'] != last_candle_time:
                            self.engine.market_data.append(processed_candle)
                            
                            if len(self.engine.market_data) > 100:
                                self.engine.market_data = self.engine.market_data[-100:]
                            
                            analysis = self.engine.analyze_market(processed_candle)
                            if analysis and analysis['signal']:
                                self.engine.execute_trade(analysis)
                            
                            last_candle_time = processed_candle['timestamp']
                            
                            if len(self.engine.market_data) % 5 == 0:
                                price = processed_candle['close']
                                data_count = len(self.engine.market_data)
                                balance = self.engine.balance
                                logger.info(f"SOL: ${price:.4f} | Datos: {data_count} | Balance: ${balance:.2f}")
                                self.engine.status = f"Activo - SOL: ${price:.4f}"
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error en bucle: {e}")
                await asyncio.sleep(5)

# Bot global
bot = LearningTradingBot()

# Rutas Flask para monitoreo
@app.route('/')
def dashboard():
    return f"""
    <html>
    <head>
        <title>SolanaScalper - Modo Aprendizaje</title>
        <meta http-equiv="refresh" content="10">
        <style>
            body {{ font-family: Arial; margin: 40px; background: #1a1a1a; color: #fff; }}
            .card {{ background: #2d2d2d; padding: 20px; margin: 20px 0; border-radius: 8px; }}
            .green {{ color: #4CAF50; }}
            .red {{ color: #f44336; }}
            .status {{ font-size: 24px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>🤖 SolanaScalper - Modo Aprendizaje</h1>
        
        <div class="card">
            <h2>Estado Actual</h2>
            <p class="status">{trading_engine.status}</p>
            <p>Modo: Paper Trading (Sin riesgo real)</p>
            <p>Par: SOL/USDT</p>
        </div>
        
        <div class="card">
            <h2>Estadísticas de Aprendizaje</h2>
            <p>Balance: ${trading_engine.balance:.2f} USDT</p>
            <p>Operaciones totales: {len(trading_engine.trades)}</p>
            <p>Datos de mercado: {len(trading_engine.market_data)} velas</p>
            <p>Posición actual: {'Abierta' if trading_engine.position else 'Cerrada'}</p>
        </div>
        
        <div class="card">
            <h2>Control</h2>
            <p>El bot está aprendiendo automáticamente de los movimientos del mercado</p>
            <p>Los datos se guardan en learning_data.json</p>
        </div>
        
        <p><a href="/stats">Ver estadísticas detalladas</a></p>
    </body>
    </html>
    """

@app.route('/stats')
def stats():
    return jsonify(trading_engine.get_stats())

def run_bot_async():
    """Ejecuta el bot en un hilo separado"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(bot.run_learning_mode())

# Iniciar bot en hilo separado
bot_thread = threading.Thread(target=run_bot_async, daemon=True)
bot_thread.start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)