#!/usr/bin/env python3
"""
Bot de Trading con Aprendizaje Automático
Modo paper trading para entrenamiento sin riesgo
"""
import os
import sys
import asyncio
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LearningBot")

# Cargar configuración
config_path = Path("config.env")
load_dotenv(dotenv_path=config_path)

# Importar módulos del bot
from api_client.modulocola import data_queue
from api_client.modulo2 import OKXWebSocketClient

class PaperTradingEngine:
    """Motor de paper trading con aprendizaje"""
    
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.position = None
        self.trades = []
        self.market_data = []
        self.learning_data = []
        
    def analyze_market(self, candle_data):
        """Analiza datos de mercado y toma decisiones"""
        if len(self.market_data) < 20:  # Necesitamos al menos 20 velas
            return None
            
        df = pd.DataFrame(self.market_data)
        
        # Calcular indicadores técnicos
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        
        current_price = float(candle_data['close'])
        sma_5 = df['sma_5'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        
        # Lógica de trading simple para empezar
        signal = None
        
        if self.position is None:  # No tenemos posición
            if sma_5 > sma_20 and rsi < 70:  # Señal de compra
                signal = "BUY"
            elif sma_5 < sma_20 and rsi > 30:  # Señal de venta
                signal = "SELL"
        else:  # Tenemos posición abierta
            if self.position['type'] == 'LONG':
                if current_price >= self.position['entry_price'] * 1.01:  # 1% ganancia
                    signal = "CLOSE_LONG_PROFIT"
                elif current_price <= self.position['entry_price'] * 0.99:  # 1% pérdida
                    signal = "CLOSE_LONG_LOSS"
            elif self.position['type'] == 'SHORT':
                if current_price <= self.position['entry_price'] * 0.99:  # 1% ganancia
                    signal = "CLOSE_SHORT_PROFIT"
                elif current_price >= self.position['entry_price'] * 1.01:  # 1% pérdida
                    signal = "CLOSE_SHORT_LOSS"
        
        return {
            'signal': signal,
            'price': current_price,
            'sma_5': sma_5,
            'sma_20': sma_20,
            'rsi': rsi
        }
    
    def calculate_rsi(self, prices, period=14):
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def execute_trade(self, analysis):
        """Ejecuta operación en papel"""
        signal = analysis['signal']
        price = analysis['price']
        
        if signal == "BUY" and self.position is None:
            self.position = {
                'type': 'LONG',
                'entry_price': price,
                'entry_time': datetime.now(),
                'size': self.balance * 0.1  # 10% del balance
            }
            logger.info(f"📈 COMPRA ejecutada a ${price:.4f}")
            
        elif signal == "SELL" and self.position is None:
            self.position = {
                'type': 'SHORT',
                'entry_price': price,
                'entry_time': datetime.now(),
                'size': self.balance * 0.1
            }
            logger.info(f"📉 VENTA ejecutada a ${price:.4f}")
            
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
                
                logger.info(f"🔄 Posición cerrada: {profit:+.2f} USDT (Balance: {self.balance:.2f})")
                self.position = None
    
    def calculate_profit(self, exit_price):
        """Calcula ganancia/pérdida"""
        if not self.position:
            return 0
            
        size = self.position['size']
        entry_price = self.position['entry_price']
        
        if self.position['type'] == 'LONG':
            return size * (exit_price - entry_price) / entry_price
        else:  # SHORT
            return size * (entry_price - exit_price) / entry_price
    
    def learn_from_trade(self, trade_result):
        """Aprende de los resultados de trading"""
        success = trade_result['profit'] > 0
        
        learning_record = {
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'profit': trade_result['profit'],
            'indicators': {
                'sma_5': trade_result['analysis_data']['sma_5'],
                'sma_20': trade_result['analysis_data']['sma_20'],
                'rsi': trade_result['analysis_data']['rsi']
            },
            'market_conditions': self.get_market_conditions()
        }
        
        self.learning_data.append(learning_record)
        self.save_learning_data()
        
        # Análisis de rendimiento
        if len(self.trades) % 10 == 0:  # Cada 10 operaciones
            self.analyze_performance()
    
    def get_market_conditions(self):
        """Identifica condiciones actuales del mercado"""
        if len(self.market_data) < 10:
            return "insufficient_data"
            
        recent_prices = [float(d['close']) for d in self.market_data[-10:]]
        volatility = np.std(recent_prices) / np.mean(recent_prices)
        
        if volatility > 0.02:
            return "high_volatility"
        elif volatility < 0.005:
            return "low_volatility"
        else:
            return "normal_volatility"
    
    def save_learning_data(self):
        """Guarda datos de aprendizaje"""
        try:
            with open('learning_data.json', 'w') as f:
                json.dump(self.learning_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando datos de aprendizaje: {e}")
    
    def analyze_performance(self):
        """Analiza rendimiento y muestra estadísticas"""
        if not self.trades:
            return
            
        profitable_trades = [t for t in self.trades if t['profit'] > 0]
        total_profit = sum(t['profit'] for t in self.trades)
        win_rate = len(profitable_trades) / len(self.trades) * 100
        
        logger.info(f"📊 ESTADÍSTICAS DE APRENDIZAJE:")
        logger.info(f"   Operaciones totales: {len(self.trades)}")
        logger.info(f"   Tasa de éxito: {win_rate:.1f}%")
        logger.info(f"   Ganancia total: {total_profit:+.2f} USDT")
        logger.info(f"   ROI: {(self.balance/self.initial_balance-1)*100:+.2f}%")

class LearningTradingBot:
    """Bot principal con capacidades de aprendizaje"""
    
    def __init__(self):
        self.engine = PaperTradingEngine()
        self.ws_client = None
        self.running = False
        
    async def connect_to_market(self):
        """Conecta a datos de mercado en tiempo real"""
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
            logger.info("Conectado a datos de mercado en tiempo real")
            return True
        except Exception as e:
            logger.error(f"Error conectando: {e}")
            return False
    
    async def run_learning_mode(self):
        """Ejecuta el bot en modo aprendizaje"""
        logger.info("🤖 Iniciando modo de aprendizaje automático")
        logger.info("💰 Paper Trading activado - Sin riesgo real")
        
        if not await self.connect_to_market():
            return
            
        self.running = True
        last_candle_time = None
        
        while self.running:
            try:
                if not data_queue.empty():
                    data = data_queue.get_nowait()
                    
                    if 'data' in data and data['data']:
                        candle_data = data['data'][0]
                        
                        # Convertir datos de vela
                        processed_candle = {
                            'timestamp': candle_data[0],
                            'open': float(candle_data[1]),
                            'high': float(candle_data[2]),
                            'low': float(candle_data[3]),
                            'close': float(candle_data[4]),
                            'volume': float(candle_data[5])
                        }
                        
                        # Evitar procesar la misma vela múltiples veces
                        if processed_candle['timestamp'] != last_candle_time:
                            self.engine.market_data.append(processed_candle)
                            
                            # Mantener solo las últimas 100 velas
                            if len(self.engine.market_data) > 100:
                                self.engine.market_data = self.engine.market_data[-100:]
                            
                            # Analizar mercado y ejecutar operaciones
                            analysis = self.engine.analyze_market(processed_candle)
                            if analysis and analysis['signal']:
                                self.engine.execute_trade(analysis)
                            
                            last_candle_time = processed_candle['timestamp']
                            
                            # Log de actividad cada 5 velas
                            if len(self.engine.market_data) % 5 == 0:
                                logger.info(f"💹 SOL: ${processed_candle['close']:.4f} | "
                                          f"Datos: {len(self.engine.market_data)} velas | "
                                          f"Balance: ${self.engine.balance:.2f}")
                
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Deteniendo bot por solicitud del usuario...")
                break
            except Exception as e:
                logger.error(f"Error en bucle principal: {e}")
                await asyncio.sleep(5)
        
        await self.shutdown()
    
    async def shutdown(self):
        """Cierra el bot limpiamente"""
        self.running = False
        if self.ws_client and self.ws_client.ws:
            await self.ws_client.ws.close()
        
        # Análisis final
        self.engine.analyze_performance()
        logger.info("🛑 Bot detenido - Datos de aprendizaje guardados")

async def main():
    """Función principal"""
    bot = LearningTradingBot()
    
    try:
        await bot.run_learning_mode()
    except KeyboardInterrupt:
        logger.info("Deteniendo bot...")
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())