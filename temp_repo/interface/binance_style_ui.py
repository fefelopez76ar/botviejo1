#!/usr/bin/env python3
"""
Interfaz de usuario estilo Binance para el bot de trading.

Este módulo implementa una interfaz web que emula la apariencia y funcionalidad
de Binance, permitiendo una experiencia familiar para los usuarios.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BinanceStyleUI')

class BinanceStyleUI:
    """
    Implementación de interfaz web estilo Binance para el bot de trading.
    """
    
    def __init__(self, app=None, config=None):
        """
        Inicializa la interfaz web.
        
        Args:
            app: Instancia de Flask (opcional)
            config: Configuración (opcional)
        """
        self.app = app
        self.config = config or {}
        
        # Estado del bot
        self.bot_running = False
        self.current_mode = "paper"  # 'paper' o 'live'
        self.current_market_type = "spot"  # 'spot' o 'futures'
        self.current_leverage = 1  # Solo para futuros
        
        # Historial de operaciones
        self.trading_history = []
        
        # Configuración de estrategias
        self.active_strategies = {}
        
        # Registro de rendimiento para cambio automático a modo real
        self.performance_history = {
            "daily_results": [],
            "consecutive_profitable_days": 0,
            "ready_for_live": False
        }
        
        # Inicializar rutas si se proporciona app
        if self.app:
            self._init_routes()
    
    def init_app(self, app):
        """
        Inicializa la aplicación Flask.
        
        Args:
            app: Instancia de Flask
        """
        self.app = app
        self._init_routes()
    
    def _init_routes(self):
        """Inicializa las rutas de la aplicación."""
        # Página principal - Panel de control
        self.app.route('/')(self.dashboard)
        
        # Páginas de trading
        self.app.route('/spot')(self.spot_trading)
        self.app.route('/futures')(self.futures_trading)
        
        # API para acciones del bot
        self.app.route('/api/start_bot', methods=['POST'])(self.api_start_bot)
        self.app.route('/api/stop_bot', methods=['POST'])(self.api_stop_bot)
        self.app.route('/api/change_mode', methods=['POST'])(self.api_change_mode)
        self.app.route('/api/change_market', methods=['POST'])(self.api_change_market)
        self.app.route('/api/set_leverage', methods=['POST'])(self.api_set_leverage)
        
        # API para datos en tiempo real
        self.app.route('/api/market_data')(self.api_market_data)
        self.app.route('/api/bot_status')(self.api_bot_status)
        self.app.route('/api/trading_history')(self.api_trading_history)
        self.app.route('/api/performance')(self.api_performance)
        
        # Páginas de configuración
        self.app.route('/settings')(self.settings_page)
        self.app.route('/api/save_settings', methods=['POST'])(self.api_save_settings)
        
        # Páginas de aprendizaje
        self.app.route('/learning')(self.learning_page)
        self.app.route('/api/export_brain', methods=['POST'])(self.api_export_brain)
        self.app.route('/api/import_brain', methods=['POST'])(self.api_import_brain)
        
        # Páginas de análisis
        self.app.route('/analysis')(self.analysis_page)
        self.app.route('/api/run_analysis', methods=['POST'])(self.api_run_analysis)
    
    def dashboard(self):
        """Página principal del panel de control."""
        # Obtener datos para el dashboard
        bot_status = self._get_bot_status()
        account_info = self._get_account_info()
        recent_trades = self._get_recent_trades(limit=5)
        performance_metrics = self._get_performance_metrics()
        
        # Renderizar plantilla
        return render_template(
            'dashboard.html',
            bot_status=bot_status,
            account_info=account_info,
            recent_trades=recent_trades,
            performance_metrics=performance_metrics
        )
    
    def spot_trading(self):
        """Página de trading spot."""
        # Cambiar a modo spot
        self.current_market_type = "spot"
        
        # Obtener datos para la página
        market_data = self._get_market_data()
        order_book = self._get_order_book()
        open_orders = self._get_open_orders()
        
        return render_template(
            'spot_trading.html',
            market_data=market_data,
            order_book=order_book,
            open_orders=open_orders,
            bot_status=self._get_bot_status()
        )
    
    def futures_trading(self):
        """Página de trading de futuros."""
        # Cambiar a modo futuros
        self.current_market_type = "futures"
        
        # Obtener datos para la página
        market_data = self._get_market_data()
        order_book = self._get_order_book()
        open_orders = self._get_open_orders()
        position_info = self._get_position_info()
        funding_info = self._get_funding_info()
        
        return render_template(
            'futures_trading.html',
            market_data=market_data,
            order_book=order_book,
            open_orders=open_orders,
            position_info=position_info,
            funding_info=funding_info,
            leverage=self.current_leverage,
            bot_status=self._get_bot_status()
        )
    
    def settings_page(self):
        """Página de configuración."""
        # Obtener configuraciones actuales
        strategies = self._get_available_strategies()
        active_strategies = self.active_strategies
        general_settings = self._get_general_settings()
        
        return render_template(
            'settings.html',
            strategies=strategies,
            active_strategies=active_strategies,
            general_settings=general_settings,
            bot_status=self._get_bot_status()
        )
    
    def learning_page(self):
        """Página de aprendizaje y transferencia de cerebro."""
        # Obtener datos de aprendizaje
        learning_stats = self._get_learning_stats()
        available_brains = self._get_available_brains()
        
        return render_template(
            'learning.html',
            learning_stats=learning_stats,
            available_brains=available_brains,
            bot_status=self._get_bot_status()
        )
    
    def analysis_page(self):
        """Página de análisis de mercado."""
        # Obtener datos de análisis
        market_analysis = self._get_market_analysis()
        technical_indicators = self._get_technical_indicators()
        pattern_analysis = self._get_pattern_analysis()
        
        return render_template(
            'analysis.html',
            market_analysis=market_analysis,
            technical_indicators=technical_indicators,
            pattern_analysis=pattern_analysis,
            bot_status=self._get_bot_status()
        )
    
    def api_start_bot(self):
        """API para iniciar el bot."""
        self.bot_running = True
        # TODO: Implementar la lógica de inicio del bot
        
        return jsonify({
            "success": True,
            "message": f"Bot iniciado en modo {self.current_mode} ({self.current_market_type})"
        })
    
    def api_stop_bot(self):
        """API para detener el bot."""
        self.bot_running = False
        # TODO: Implementar la lógica de detención del bot
        
        return jsonify({
            "success": True,
            "message": "Bot detenido"
        })
    
    def api_change_mode(self):
        """API para cambiar entre modo papel y real."""
        new_mode = request.json.get('mode')
        confirm = request.json.get('confirm', False)
        
        if new_mode not in ['paper', 'live']:
            return jsonify({
                "success": False,
                "message": "Modo no válido. Use 'paper' o 'live'."
            })
        
        # Si está cambiando a modo real
        if new_mode == 'live':
            # Verificar si se puede cambiar a modo real
            if not self._verify_live_readiness():
                return jsonify({
                    "success": False,
                    "message": "No se cumplen los requisitos para cambiar a modo real. Verifique sus credenciales API y el rendimiento del bot."
                })
                
            # Si no ha confirmado, pedir confirmación explícita
            if not confirm:
                return jsonify({
                    "success": False,
                    "needs_confirmation": True,
                    "message": "ADVERTENCIA: Está a punto de activar el trading en vivo con fondos reales. Esta acción puede resultar en pérdidas económicas. Por favor, confirme que entiende los riesgos."
                })
        
        # Cambiar modo y registrar en log
        previous_mode = self.current_mode
        self.current_mode = new_mode
        
        logger.warning(f"CAMBIO DE MODO: {previous_mode} -> {new_mode}")
        
        # Mensaje personalizado según el modo
        if new_mode == 'live':
            message = "¡MODO REAL ACTIVADO! El bot ahora operará con fondos reales."
        else:
            message = "Modo cambiado a simulación (paper trading)."
        
        return jsonify({
            "success": True,
            "message": message
        })
    
    def api_change_market(self):
        """API para cambiar entre spot y futuros."""
        new_market = request.json.get('market')
        
        if new_market not in ['spot', 'futures']:
            return jsonify({
                "success": False,
                "message": "Mercado no válido. Use 'spot' o 'futures'."
            })
        
        self.current_market_type = new_market
        
        return jsonify({
            "success": True,
            "message": f"Tipo de mercado cambiado a {new_market}"
        })
    
    def api_set_leverage(self):
        """API para establecer apalancamiento (solo futuros)."""
        leverage = request.json.get('leverage')
        
        try:
            leverage = int(leverage)
            if leverage < 1 or leverage > 125:  # Binance permite hasta 125x
                raise ValueError("Apalancamiento fuera de rango")
        except (ValueError, TypeError):
            return jsonify({
                "success": False,
                "message": "Apalancamiento no válido. Debe ser un número entre 1 y 125."
            })
        
        if self.current_market_type != 'futures':
            return jsonify({
                "success": False,
                "message": "El apalancamiento solo está disponible en modo futuros."
            })
        
        self.current_leverage = leverage
        
        return jsonify({
            "success": True,
            "message": f"Apalancamiento establecido en {leverage}x"
        })
    
    def api_market_data(self):
        """API para obtener datos de mercado en tiempo real."""
        symbol = request.args.get('symbol', 'SOLUSDT')
        
        market_data = self._get_market_data(symbol)
        
        return jsonify(market_data)
    
    def api_bot_status(self):
        """API para obtener el estado actual del bot."""
        return jsonify(self._get_bot_status())
    
    def api_trading_history(self):
        """API para obtener historial de trading."""
        limit = request.args.get('limit', 20)
        
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            limit = 20
        
        history = self._get_recent_trades(limit)
        
        return jsonify(history)
    
    def api_performance(self):
        """API para obtener métricas de rendimiento."""
        return jsonify(self._get_performance_metrics())
    
    def api_save_settings(self):
        """API para guardar configuraciones."""
        settings_data = request.json
        
        # TODO: Validar y guardar configuraciones
        
        return jsonify({
            "success": True,
            "message": "Configuración guardada correctamente"
        })
    
    def api_export_brain(self):
        """API para exportar el cerebro del bot."""
        from adaptive_system.brain_transfer import create_backup
        
        try:
            # Generar nombre con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"bot_brain_{timestamp}.zip"
            
            # Crear respaldo
            backup_path = create_backup(backup_name)
            
            return jsonify({
                "success": True,
                "message": f"Cerebro exportado correctamente: {backup_path}",
                "backup_path": backup_path
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Error al exportar cerebro: {e}"
            })
    
    def api_import_brain(self):
        """API para importar el cerebro del bot."""
        from adaptive_system.brain_transfer import restore_backup
        
        try:
            file_path = request.json.get('file_path')
            
            if not file_path:
                return jsonify({
                    "success": False,
                    "message": "Ruta de archivo no proporcionada"
                })
            
            # Restaurar respaldo
            result = restore_backup(file_path)
            
            if result.get('success', False):
                return jsonify({
                    "success": True,
                    "message": "Cerebro importado correctamente",
                    "details": result
                })
            else:
                return jsonify({
                    "success": False,
                    "message": f"Error al importar cerebro: {result.get('error', 'Desconocido')}",
                    "details": result
                })
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Error al importar cerebro: {e}"
            })
    
    def api_run_analysis(self):
        """API para ejecutar análisis de mercado."""
        symbol = request.json.get('symbol', 'SOLUSDT')
        timeframe = request.json.get('timeframe', '1h')
        
        # TODO: Implementar análisis de mercado
        
        return jsonify({
            "success": True,
            "message": f"Análisis ejecutado para {symbol} en timeframe {timeframe}",
            "results": self._get_market_analysis(symbol, timeframe)
        })
    
    def _get_bot_status(self):
        """Obtiene el estado actual del bot."""
        return {
            "running": self.bot_running,
            "mode": self.current_mode,
            "market_type": self.current_market_type,
            "leverage": self.current_leverage,
            "ready_for_live": self._verify_live_readiness(),
            "active_symbol": "SOLUSDT",  # TODO: Obtener del bot
            "uptime": "00:00:00",  # TODO: Calcular
            "last_operation": None,  # TODO: Obtener del bot
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_account_info(self):
        """Obtiene información de la cuenta."""
        # TODO: Implementar lógica real
        
        return {
            "balance": {
                "USDT": 10000.00,
                "SOL": 50.0,
                "BTC": 0.1
            },
            "equity": 10000.00,
            "available": 9500.00,
            "margin_used": 500.00,
            "profit_today": 125.50,
            "profit_total": 450.75,
            "win_rate": 65.0
        }
    
    def _get_recent_trades(self, limit=20):
        """Obtiene operaciones recientes."""
        # TODO: Implementar lógica real
        
        # Datos simulados para demo
        trades = []
        
        for i in range(min(limit, len(self.trading_history))):
            trades.append(self.trading_history[i])
        
        # Si no hay suficientes datos en el historial, crear algunos simulados
        if not trades:
            base_time = datetime.now()
            
            # Crear algunas operaciones simuladas
            for i in range(limit):
                trade_time = base_time - timedelta(hours=i)
                
                # Alternar entre compras y ventas
                side = "BUY" if i % 2 == 0 else "SELL"
                
                # Precio base con ligera variación
                price = 150.0 + (0.5 * (i % 5))
                
                # Cantidad aleatoria
                quantity = round(1 + (i % 10) / 10, 2)
                
                # Beneficio/pérdida
                pnl = round((i % 3 - 1) * 10.0, 2)
                
                trades.append({
                    "id": f"T{1000 + i}",
                    "symbol": "SOLUSDT",
                    "side": side,
                    "type": "MARKET",
                    "price": price,
                    "quantity": quantity,
                    "total": round(price * quantity, 2),
                    "fee": round(price * quantity * 0.001, 4),
                    "pnl": pnl,
                    "pnl_percent": round(pnl / (price * quantity) * 100, 2),
                    "timestamp": trade_time.isoformat(),
                    "status": "FILLED"
                })
        
        return trades
    
    def _get_performance_metrics(self):
        """Obtiene métricas de rendimiento."""
        # TODO: Implementar lógica real
        
        return {
            "daily": {
                "trades": 12,
                "win_rate": 75.0,
                "profit": 45.50,
                "profit_percent": 0.45,
                "best_trade": 25.30,
                "worst_trade": -10.40
            },
            "weekly": {
                "trades": 67,
                "win_rate": 67.0,
                "profit": 210.75,
                "profit_percent": 2.10,
                "best_trade": 78.20,
                "worst_trade": -25.50
            },
            "monthly": {
                "trades": 245,
                "win_rate": 62.0,
                "profit": 450.75,
                "profit_percent": 4.50,
                "best_trade": 120.40,
                "worst_trade": -45.60
            },
            "consecutive_profitable_days": 8,
            "drawdown_max": -120.30,
            "drawdown_current": 0.0
        }
    
    def _get_market_data(self, symbol="SOLUSDT"):
        """Obtiene datos de mercado para un símbolo."""
        # Importar aquí para evitar dependencias circulares
        from data_management.market_data import MarketData
        
        try:
            # Crear instancia de MarketData
            md = MarketData()
            
            # Obtener datos del ticker
            ticker_data = md.get_ticker(symbol)
            
            if not ticker_data:
                # Datos simulados si no se pueden obtener
                ticker_data = {
                    "symbol": symbol,
                    "last_price": 150.25,
                    "bid_price": 150.24,
                    "ask_price": 150.26,
                    "volume_24h": 1250000,
                    "high_24h": 152.50,
                    "low_24h": 148.75,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Obtener datos históricos para gráfico
            df = md.get_historical_data(symbol, "1h", limit=100)
            
            candles = []
            if df is not None and not df.empty:
                # Convertir DataFrame a lista de velas
                for _, row in df.iterrows():
                    candles.append([
                        int(row.name.timestamp() * 1000),  # timestamp en ms
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row["volume"])
                    ])
            
            return {
                "ticker": ticker_data,
                "candles": candles,
                "timeframe": "1h"
            }
            
        except Exception as e:
            logger.error(f"Error al obtener datos de mercado: {e}")
            
            # Retornar datos simulados en caso de error
            return {
                "ticker": {
                    "symbol": symbol,
                    "last_price": 150.25,
                    "bid_price": 150.24,
                    "ask_price": 150.26,
                    "volume_24h": 1250000,
                    "high_24h": 152.50,
                    "low_24h": 148.75,
                    "timestamp": datetime.now().isoformat()
                },
                "candles": [],
                "timeframe": "1h",
                "error": str(e)
            }
    
    def _get_order_book(self, symbol="SOLUSDT", depth=10):
        """Obtiene el libro de órdenes para un símbolo."""
        # Importar aquí para evitar dependencias circulares
        from data_management.market_data import MarketData
        
        try:
            # Crear instancia de MarketData
            md = MarketData()
            
            # Obtener libro de órdenes
            orderbook = md.get_orderbook(symbol, depth)
            
            if not orderbook:
                # Datos simulados si no se pueden obtener
                last_price = 150.25
                
                bids = []
                asks = []
                
                # Crear bids simulados (descendentes desde el precio)
                for i in range(depth):
                    price = last_price - (0.01 * (i + 1))
                    quantity = 10 + (i * 2)
                    bids.append([f"{price:.2f}", f"{quantity:.2f}"])
                
                # Crear asks simulados (ascendentes desde el precio)
                for i in range(depth):
                    price = last_price + (0.01 * (i + 1))
                    quantity = 10 + (i * 2)
                    asks.append([f"{price:.2f}", f"{quantity:.2f}"])
                
                orderbook = {
                    "symbol": symbol,
                    "bids": bids,
                    "asks": asks,
                    "timestamp": datetime.now().isoformat()
                }
            
            return orderbook
            
        except Exception as e:
            logger.error(f"Error al obtener libro de órdenes: {e}")
            
            # Retornar datos simulados en caso de error
            return {
                "symbol": symbol,
                "bids": [],
                "asks": [],
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _get_open_orders(self, symbol="SOLUSDT"):
        """Obtiene órdenes abiertas para un símbolo."""
        # TODO: Implementar lógica real
        
        # Datos simulados para demo
        open_orders = []
        
        if self.bot_running:
            # Crear algunas órdenes simuladas si el bot está corriendo
            base_time = datetime.now()
            
            # Orden de compra tipo limit
            open_orders.append({
                "id": "O1001",
                "symbol": symbol,
                "side": "BUY",
                "type": "LIMIT",
                "status": "NEW",
                "price": 149.50,
                "quantity": 2.0,
                "filled": 0.0,
                "timestamp": (base_time - timedelta(minutes=5)).isoformat()
            })
            
            # Orden de venta tipo stop loss
            open_orders.append({
                "id": "O1002",
                "symbol": symbol,
                "side": "SELL",
                "type": "STOP_LOSS_LIMIT",
                "status": "NEW",
                "price": 148.00,
                "quantity": 1.5,
                "filled": 0.0,
                "stop_price": 148.50,
                "timestamp": (base_time - timedelta(minutes=10)).isoformat()
            })
        
        return open_orders
    
    def _get_position_info(self, symbol="SOLUSDT"):
        """Obtiene información de posición para futuros."""
        # TODO: Implementar lógica real
        
        # Solo relevante en modo futuros
        if self.current_market_type != 'futures':
            return None
        
        # Datos simulados para demo
        if self.bot_running:
            return {
                "symbol": symbol,
                "leverage": self.current_leverage,
                "position_side": "LONG",
                "position_amount": 1.5,
                "entry_price": 149.75,
                "mark_price": 150.25,
                "unrealized_pnl": 0.75,
                "unrealized_pnl_percent": 0.33,
                "liquidation_price": 120.50,
                "margin_type": "CROSSED",
                "isolated_margin": 0.0,
                "isolated_wallet": 0.0,
                "timestamp": datetime.now().isoformat()
            }
        
        return None
    
    def _get_funding_info(self, symbol="SOLUSDT"):
        """Obtiene información de financiamiento para futuros."""
        # Solo relevante en modo futuros
        if self.current_market_type != 'futures':
            return None
        
        # Datos simulados para demo
        return {
            "symbol": symbol,
            "funding_rate": 0.0001,  # 0.01%
            "next_funding_time": (datetime.now() + timedelta(hours=7)).isoformat(),
            "estimated_rate": 0.00015,
            "interestRate": 0.0003,
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_available_strategies(self):
        """Obtiene estrategias disponibles."""
        # TODO: Implementar lógica real
        
        # Estrategias simuladas para demo
        return [
            {
                "id": "breakout_scalping",
                "name": "Scalping de Ruptura",
                "description": "Opera rupturas de niveles clave con confirmación de volumen",
                "timeframes": ["1m", "5m"],
                "market_types": ["spot", "futures"],
                "default_params": {
                    "take_profit_pct": 0.8,
                    "stop_loss_pct": 0.5,
                    "leverage": 3,
                    "position_size_pct": 2.0
                }
            },
            {
                "id": "momentum_scalping",
                "name": "Scalping de Momento",
                "description": "Opera en la dirección del impulso de precio con confirmación",
                "timeframes": ["1m", "5m"],
                "market_types": ["spot", "futures"],
                "default_params": {
                    "rsi_period": 7,
                    "ema_fast": 5,
                    "ema_slow": 8,
                    "take_profit_pct": 0.6,
                    "stop_loss_pct": 0.4,
                    "leverage": 3,
                    "position_size_pct": 2.0
                }
            },
            {
                "id": "mean_reversion",
                "name": "Reversión a la Media",
                "description": "Opera rebotes desde niveles sobrevendidos/sobrecomprados",
                "timeframes": ["5m", "15m"],
                "market_types": ["spot", "futures"],
                "default_params": {
                    "bb_period": 20,
                    "bb_std": 2.0,
                    "rsi_period": 7,
                    "rsi_oversold": 30,
                    "rsi_overbought": 70,
                    "take_profit_pct": 0.7,
                    "stop_loss_pct": 0.5,
                    "leverage": 2,
                    "position_size_pct": 1.5
                }
            },
            {
                "id": "ml_adaptive",
                "name": "ML Adaptativo",
                "description": "Usa machine learning para predecir movimientos de precio",
                "timeframes": ["1m", "5m", "15m", "1h"],
                "market_types": ["spot", "futures"],
                "default_params": {
                    "confidence_threshold": 0.65,
                    "take_profit_pct": 0.8,
                    "stop_loss_pct": 0.6,
                    "leverage": 2,
                    "position_size_pct": 1.5
                }
            }
        ]
    
    def _get_general_settings(self):
        """Obtiene configuraciones generales."""
        # TODO: Implementar lógica real
        
        return {
            "automatic_mode_switch": True,  # Cambiar automáticamente a modo real cuando esté listo
            "consecutive_profitable_days_required": 30,  # Días consecutivos con ganancia para modo real
            "min_win_rate_required": 60.0,  # Win rate mínimo requerido para modo real
            "max_drawdown_allowed": 5.0,  # Drawdown máximo permitido
            "restart_after_loss": True,  # Reiniciar contador después de día con pérdida
            "trading_hours": {
                "enabled": False,  # Trading 24/7 por defecto
                "start": "00:00",
                "end": "23:59"
            },
            "default_pair": "SOLUSDT",  # Par por defecto
            "notification_settings": {
                "email": False,
                "telegram": False,
                "web": True
            }
        }
    
    def _get_learning_stats(self):
        """Obtiene estadísticas de aprendizaje."""
        # TODO: Implementar lógica real
        
        return {
            "indicators_performance": {
                "rsi": 0.72,
                "macd": 0.68,
                "bollinger": 0.65,
                "ema": 0.63,
                "vwap": 0.70
            },
            "strategy_performance": {
                "breakout_scalping": 0.67,
                "momentum_scalping": 0.63,
                "mean_reversion": 0.61,
                "ml_adaptive": 0.72
            },
            "model_metrics": {
                "accuracy": 0.68,
                "precision": 0.70,
                "recall": 0.65,
                "f1_score": 0.67
            },
            "learning_progress": {
                "total_samples": 15420,
                "training_iterations": 87,
                "last_training": (datetime.now() - timedelta(hours=4)).isoformat(),
                "improvement_rate": 0.02
            }
        }
    
    def _get_available_brains(self):
        """Obtiene cerebros disponibles para importar."""
        # Importar aquí para evitar dependencias circulares
        from adaptive_system.brain_transfer import list_backups
        
        try:
            backups = list_backups()
            return backups
        except Exception as e:
            logger.error(f"Error al obtener cerebros disponibles: {e}")
            return []
    
    def _get_market_analysis(self, symbol="SOLUSDT", timeframe="1h"):
        """Obtiene análisis de mercado."""
        # TODO: Implementar lógica real
        
        # Datos simulados para demo
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "price": 150.25,
            "market_condition": "lateral_high_vol",
            "trend": {
                "short_term": "neutral",
                "medium_term": "bullish",
                "long_term": "bullish"
            },
            "support_resistance": {
                "strong_resistance": 155.80,
                "resistance": 152.30,
                "support": 149.20,
                "strong_support": 145.60
            },
            "indicators": {
                "rsi": 48.5,
                "macd": -0.12,
                "bb_width": 1.35,
                "ema9": 150.15,
                "ema21": 149.90
            },
            "signals": {
                "breakout": "neutral",
                "momentum": "neutral",
                "mean_reversion": "buy",
                "ml_prediction": "buy",
                "combined": "buy"
            },
            "recommendation": {
                "action": "buy",
                "confidence": 0.7,
                "strategy": "mean_reversion_bounce",
                "entry": 150.25,
                "target": 152.20,
                "stop": 149.20
            }
        }
    
    def _get_technical_indicators(self, symbol="SOLUSDT", timeframe="1h"):
        """Obtiene indicadores técnicos calculados."""
        # Importar aquí para evitar dependencias circulares
        from data_management.market_data import MarketData
        
        try:
            # Crear instancia de MarketData
            md = MarketData()
            
            # Obtener datos históricos
            df = md.get_historical_data(symbol, timeframe, limit=100)
            
            if df is not None and not df.empty:
                # Calcular indicadores
                df_with_indicators = md.calculate_indicators(df)
                
                # Obtener valores del último registro
                last_row = df_with_indicators.iloc[-1]
                
                indicators = {}
                
                # Extraer valores de indicadores
                if 'rsi' in df_with_indicators.columns:
                    indicators['rsi'] = float(last_row['rsi'])
                
                if 'macd_line' in df_with_indicators.columns:
                    indicators['macd'] = float(last_row['macd_line'])
                
                if 'macd_signal' in df_with_indicators.columns:
                    indicators['macd_signal'] = float(last_row['macd_signal'])
                
                if 'macd_histogram' in df_with_indicators.columns:
                    indicators['macd_histogram'] = float(last_row['macd_histogram'])
                
                # Bandas de Bollinger
                if 'bb_upper' in df_with_indicators.columns:
                    indicators['bb_upper'] = float(last_row['bb_upper'])
                
                if 'bb_middle' in df_with_indicators.columns:
                    indicators['bb_middle'] = float(last_row['bb_middle'])
                
                if 'bb_lower' in df_with_indicators.columns:
                    indicators['bb_lower'] = float(last_row['bb_lower'])
                
                if 'bb_width' in df_with_indicators.columns:
                    indicators['bb_width'] = float(last_row['bb_width'])
                
                # EMAs
                for period in [9, 21, 50, 200]:
                    if f'ema_{period}' in df_with_indicators.columns:
                        indicators[f'ema{period}'] = float(last_row[f'ema_{period}'])
                
                # ATR
                if 'atr' in df_with_indicators.columns:
                    indicators['atr'] = float(last_row['atr'])
                
                if 'atr_pct' in df_with_indicators.columns:
                    indicators['atr_pct'] = float(last_row['atr_pct'])
                
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "price": float(last_row['close']),
                    "indicators": indicators,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Retornar datos simulados si no hay datos
            return self._get_simulated_indicators(symbol, timeframe)
            
        except Exception as e:
            logger.error(f"Error al obtener indicadores técnicos: {e}")
            return self._get_simulated_indicators(symbol, timeframe)
    
    def _get_simulated_indicators(self, symbol="SOLUSDT", timeframe="1h"):
        """Obtiene indicadores técnicos simulados."""
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "price": 150.25,
            "indicators": {
                "rsi": 48.5,
                "macd": -0.12,
                "macd_signal": -0.08,
                "macd_histogram": -0.04,
                "bb_upper": 152.50,
                "bb_middle": 150.30,
                "bb_lower": 148.10,
                "bb_width": 0.029,
                "ema9": 150.15,
                "ema21": 149.90,
                "ema50": 149.50,
                "ema200": 148.80,
                "atr": 1.20,
                "atr_pct": 0.80
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_pattern_analysis(self, symbol="SOLUSDT", timeframe="1h"):
        """Obtiene análisis de patrones."""
        # TODO: Implementar lógica real
        
        # Datos simulados para demo
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "price": 150.25,
            "patterns": [
                {
                    "type": "double_bottom",
                    "reliability": 0.68,
                    "detected_at": (datetime.now() - timedelta(hours=4)).isoformat(),
                    "target_price": 154.20,
                    "stop_loss": 148.50,
                    "completed": 0.25
                },
                {
                    "type": "flag",
                    "reliability": 0.62,
                    "detected_at": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "target_price": 153.50,
                    "stop_loss": 149.00,
                    "completed": 0.10
                }
            ],
            "key_levels": {
                "resistances": [152.30, 155.80, 160.00],
                "supports": [149.20, 145.60, 140.00]
            },
            "fibonacci_levels": {
                "high": 155.80,
                "low": 145.60,
                "levels": {
                    "0.0": 145.60,
                    "0.236": 148.02,
                    "0.382": 149.50,
                    "0.5": 150.70,
                    "0.618": 151.90,
                    "0.786": 153.38,
                    "1.0": 155.80
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _verify_live_readiness(self):
        """Verifica si se cumplen los requisitos para modo real."""
        # Verificar si las credenciales de API están configuradas
        try:
            import os
            from dotenv import load_dotenv
            
            # Cargar variables de entorno si no están ya cargadas
            load_dotenv('config.env')
            
            api_key = os.environ.get('OKX_API_KEY', '')
            api_secret = os.environ.get('OKX_API_SECRET', '')
            passphrase = os.environ.get('OKX_PASSPHRASE', '')
            
            # Verificar que no son los valores por defecto de demo
            if api_key == 'demo_key' or api_secret == 'demo_secret' or passphrase == 'demo_passphrase':
                logger.warning("Las credenciales API están usando valores de demostración")
                return False
            
            # Verificar que las credenciales están presentes
            if not api_key or not api_secret or not passphrase:
                logger.warning("Faltan credenciales API para trading en vivo")
                return False
            
            # Si estamos en modo de producción, verificar métricas de rendimiento
            if not os.environ.get('DEVELOPMENT_MODE', ''):
                # Comprobaciones para permitir el cambio a modo real basado en rendimiento
                general_settings = self._get_general_settings()
                performance_metrics = self._get_performance_metrics()
                
                # Verificar si hay suficientes días consecutivos con ganancia
                consecutive_days = performance_metrics.get('consecutive_profitable_days', 0)
                required_days = general_settings.get('consecutive_profitable_days_required', 15)  # Reducido de 30 a 15
                
                # Verificar win rate mínimo
                win_rate = performance_metrics.get('monthly', {}).get('win_rate', 0)
                required_win_rate = general_settings.get('min_win_rate_required', 55.0)  # Reducido de 60 a 55
                
                # Verificar drawdown máximo permitido
                drawdown = abs(performance_metrics.get('drawdown_max', 0))
                max_drawdown = general_settings.get('max_drawdown_allowed', 7.0)  # Aumentado de 5 a 7
                
                # Solo permitir cambio a modo real si todos los criterios se cumplen
                if consecutive_days < required_days:
                    logger.warning(f"No hay suficientes días consecutivos con ganancia: {consecutive_days}/{required_days}")
                    return False
                    
                if win_rate < required_win_rate:
                    logger.warning(f"Win rate insuficiente: {win_rate}%/{required_win_rate}%")
                    return False
                    
                if drawdown > max_drawdown:
                    logger.warning(f"Drawdown demasiado alto: {drawdown}%/{max_drawdown}%")
                    return False
            
            # Si todas las verificaciones pasaron o estamos en modo desarrollo
            logger.info("Verificación para trading en vivo exitosa")
            return True
            
        except Exception as e:
            logger.error(f"Error al verificar readiness para trading en vivo: {e}")
            return False
    
    def _update_performance_history(self, daily_result):
        """Actualiza historial de rendimiento y verifica readiness."""
        self.performance_history["daily_results"].append(daily_result)
        
        # Limitar a últimos 90 días
        if len(self.performance_history["daily_results"]) > 90:
            self.performance_history["daily_results"] = self.performance_history["daily_results"][-90:]
        
        # Actualizar contador de días consecutivos con ganancia
        if daily_result.get("profit", 0) > 0:
            self.performance_history["consecutive_profitable_days"] += 1
        else:
            # Reiniciar contador si hay configuración para ello
            general_settings = self._get_general_settings()
            if general_settings.get("restart_after_loss", True):
                self.performance_history["consecutive_profitable_days"] = 0
        
        # Verificar si está listo para modo real
        self.performance_history["ready_for_live"] = self._verify_live_readiness()
    
    def add_trade_to_history(self, trade_data):
        """Añade una operación al historial de trading."""
        self.trading_history.insert(0, trade_data)
        
        # Limitar a 1000 operaciones
        if len(self.trading_history) > 1000:
            self.trading_history = self.trading_history[:1000]

def create_binance_ui(app=None, config=None):
    """
    Función de conveniencia para crear una interfaz de usuario estilo Binance.
    
    Args:
        app: Instancia de Flask (opcional)
        config: Configuración (opcional)
        
    Returns:
        BinanceStyleUI: Instancia de la interfaz
    """
    return BinanceStyleUI(app, config)

def demo_binance_ui():
    """Función de demostración para la interfaz estilo Binance."""
    print("\n🖥️ INTERFAZ ESTILO BINANCE 🖥️")
    print("Esta interfaz permite una experiencia similar a Binance")
    print("con capacidad para operar en spot y futuros.")
    
    # Crear instancia
    ui = BinanceStyleUI()
    
    # Simular actualización de estado
    ui.bot_running = True
    ui.current_mode = "paper"
    ui.current_market_type = "spot"
    
    # Mostrar estado
    status = ui._get_bot_status()
    
    print("\n1. Estado actual del bot:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Cambiar a modo futuros
    ui.current_market_type = "futures"
    ui.current_leverage = 3
    
    # Mostrar estado actualizado
    status = ui._get_bot_status()
    
    print("\n2. Estado después de cambiar a futuros:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Simular actualización de rendimiento
    for i in range(35):  # 35 días consecutivos de ganancia
        ui._update_performance_history({
            "date": (datetime.now() - timedelta(days=35-i)).strftime("%Y-%m-%d"),
            "trades": 10 + (i % 5),
            "win_rate": 65.0 + (i % 10),
            "profit": 25.0 + (i % 20),
            "profit_percent": 0.25 + (i % 10) / 100
        })
    
    # Verificar si está listo para modo real
    ready = ui._verify_live_readiness()
    
    print("\n3. Preparación para modo real:")
    print(f"  ¿Listo para operar en modo real? {'Sí' if ready else 'No'}")
    print(f"  Días consecutivos con ganancia: {ui.performance_history['consecutive_profitable_days']}")
    
    print("\n✅ La interfaz está lista para integrarse con la aplicación Flask.")
    return ui

if __name__ == "__main__":
    try:
        demo_binance_ui()
    except Exception as e:
        print(f"Error en la demostración: {e}")