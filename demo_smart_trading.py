#!/usr/bin/env python3
"""
Demostración integrada de las capacidades avanzadas del bot de trading.

Este script muestra las nuevas capacidades del bot incluyendo:
1. Sistema de transferencia de conocimiento (cerebro)
2. Estrategias de scalping optimizadas
3. Análisis predictivo con machine learning
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tabulate import tabulate

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def divider(title=""):
    """Imprime un divisor con título opcional"""
    width = 80
    if title:
        padding = (width - len(title) - 4) // 2
        print("\n" + "=" * padding + f" {title} " + "=" * padding)
    else:
        print("\n" + "=" * width)

def demo_brain_transfer():
    """Demostración del sistema de transferencia de cerebro."""
    divider("SISTEMA DE TRANSFERENCIA DE CEREBRO")
    
    try:
        from adaptive_system.brain_transfer import BrainTransfer, demo_brain_transfer
        
        # Ejecutar demostración del módulo
        demo_brain_transfer()
        
    except ImportError:
        print("⚠️ No se pudo importar el módulo de transferencia de cerebro.")
        print("   Asegúrate de que exista el archivo adaptive_system/brain_transfer.py")

def demo_scalping_strategies():
    """Demostración de estrategias de scalping."""
    divider("ESTRATEGIAS DE SCALPING")
    
    try:
        from scalping_strategies import ScalpingStrategies, demo_scalping_strategies
        
        # Ejecutar demostración del módulo
        demo_scalping_strategies()
        
    except ImportError:
        print("⚠️ No se pudo importar el módulo de estrategias de scalping.")
        print("   Asegúrate de que exista el archivo scalping_strategies.py")

def demo_ml_prediction():
    """Demostración de predicción con machine learning."""
    divider("PREDICCIÓN CON MACHINE LEARNING")
    
    try:
        from indicator_weighting import IndicatorWeighting, demo_ml_prediction
        
        # Ejecutar demostración del módulo
        demo_ml_prediction()
        
    except ImportError:
        print("⚠️ No se pudo importar el módulo de ponderación adaptativa.")
        print("   Asegúrate de que exista el archivo indicator_weighting.py")

def compare_strategies():
    """Comparación de diferentes estrategias de trading."""
    divider("COMPARACIÓN DE ESTRATEGIAS")
    
    # Definir estrategias a comparar
    strategies = [
        {
            "name": "Scalping Momentum",
            "timeframe": "1m",
            "avg_trades_per_day": 15,
            "win_rate": 0.62,
            "avg_profit_pct": 0.5,
            "avg_loss_pct": 0.3,
            "fee_impact_pct": 0.15,
            "holding_time": "5-15 minutos",
            "best_market_condition": "Alta volatilidad",
            "risk_level": "Alto"
        },
        {
            "name": "Scalping Breakout",
            "timeframe": "5m",
            "avg_trades_per_day": 8,
            "win_rate": 0.58,
            "avg_profit_pct": 0.7,
            "avg_loss_pct": 0.4,
            "fee_impact_pct": 0.12,
            "holding_time": "10-30 minutos",
            "best_market_condition": "Ruptura de rango",
            "risk_level": "Medio-alto"
        },
        {
            "name": "Swing Trading",
            "timeframe": "4h",
            "avg_trades_per_day": 0.5,
            "win_rate": 0.55,
            "avg_profit_pct": 3.5,
            "avg_loss_pct": 2.0,
            "fee_impact_pct": 0.05,
            "holding_time": "1-3 días",
            "best_market_condition": "Tendencia definida",
            "risk_level": "Medio"
        },
        {
            "name": "ML Adaptativo",
            "timeframe": "múltiple",
            "avg_trades_per_day": 5,
            "win_rate": 0.67,
            "avg_profit_pct": 1.2,
            "avg_loss_pct": 0.7,
            "fee_impact_pct": 0.08,
            "holding_time": "variable",
            "best_market_condition": "Cualquiera (adaptativo)",
            "risk_level": "Medio-bajo"
        }
    ]
    
    # Calcular métricas derivadas
    for strategy in strategies:
        trades_per_month = strategy["avg_trades_per_day"] * 30
        win_trades = trades_per_month * strategy["win_rate"]
        loss_trades = trades_per_month * (1 - strategy["win_rate"])
        
        gross_profit = win_trades * strategy["avg_profit_pct"]
        gross_loss = loss_trades * strategy["avg_loss_pct"]
        
        strategy["monthly_return_pct"] = gross_profit - gross_loss
        strategy["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        strategy["risk_adjusted_return"] = strategy["monthly_return_pct"] / float(strategy["risk_level"].count("alto") + 1)
    
    # Mostrar tabla comparativa
    print("\nComparación de rendimiento por estrategia:")
    headers = ["Estrategia", "Trades/día", "Win Rate", "Retorno Mensual", "Factor Beneficio", "Nivel Riesgo"]
    
    data = []
    for strategy in strategies:
        data.append([
            strategy["name"],
            strategy["avg_trades_per_day"],
            f"{strategy['win_rate']*100:.1f}%",
            f"{strategy['monthly_return_pct']:.1f}%",
            f"{strategy['profit_factor']:.2f}",
            strategy["risk_level"]
        ])
    
    print(tabulate(data, headers=headers, tablefmt="grid"))
    
    # Identificar estrategia óptima según condición de mercado
    print("\nEstrategia recomendada según condición de mercado:")
    market_conditions = [
        "Alta volatilidad",
        "Baja volatilidad",
        "Tendencia alcista fuerte",
        "Tendencia bajista fuerte",
        "Mercado lateral"
    ]
    
    recommendations = {
        "Alta volatilidad": "Scalping Momentum",
        "Baja volatilidad": "ML Adaptativo",
        "Tendencia alcista fuerte": "Swing Trading",
        "Tendencia bajista fuerte": "Scalping Breakout",
        "Mercado lateral": "ML Adaptativo"
    }
    
    for condition in market_conditions:
        print(f"• {condition}: {recommendations[condition]}")

def demo_automated_analysis():
    """Demostración de análisis automatizado del mercado."""
    divider("ANÁLISIS AUTOMATIZADO DE MERCADO")
    
    # Simular un análisis completo del mercado
    analysis_result = {
        "symbol": "SOL-USDT",
        "timeframe": "1h",
        "timestamp": datetime.now().isoformat(),
        "current_price": 150.37,
        "market_condition": "lateral_high_vol",
        "key_metrics": {
            "rsi": 48.5,
            "macd": -0.12,
            "bb_width": 1.35,
            "atr": 2.45,
            "volume_ratio": 1.2
        },
        "support_resistance": {
            "strong_resistance": 155.80,
            "resistance": 152.30,
            "support": 149.20,
            "strong_support": 145.60
        },
        "predictions": {
            "short_term": {
                "direction": "neutral",
                "confidence": 0.52,
                "target_price": 150.50,
                "timeframe": "2h"
            },
            "medium_term": {
                "direction": "bullish",
                "confidence": 0.67,
                "target_price": 153.20,
                "timeframe": "1d"
            },
            "long_term": {
                "direction": "bullish",
                "confidence": 0.72,
                "target_price": 158.50,
                "timeframe": "1w"
            }
        },
        "trading_signals": {
            "breakout": "neutral",
            "momentum": "neutral",
            "mean_reversion": "buy",
            "ml_prediction": "buy",
            "combined": "buy"
        },
        "order_book_analysis": {
            "buy_pressure": 1.15,  # ratio compra/venta
            "liquidity": "alta",
            "next_resistance": 151.20,
            "next_support": 149.80
        },
        "recommendation": {
            "action": "buy",
            "entry_price": 150.37,
            "take_profit": 152.20,
            "stop_loss": 149.30,
            "risk_reward_ratio": 1.8,
            "confidence": 0.7,
            "strategy": "mean_reversion_bounce",
            "position_size_pct": 2.0
        }
    }
    
    # Mostrar análisis
    print(f"Análisis para {analysis_result['symbol']} - {analysis_result['timeframe']}")
    print(f"Precio actual: ${analysis_result['current_price']:.2f}")
    print(f"Condición de mercado: {analysis_result['market_condition']}")
    
    # Mostrar métricas clave
    print("\nMétricas clave:")
    metrics = analysis_result["key_metrics"]
    for metric, value in metrics.items():
        print(f"• {metric.upper()}: {value}")
    
    # Mostrar niveles de soporte/resistencia
    sr_levels = analysis_result["support_resistance"]
    print("\nNiveles clave:")
    print(f"Resistencia fuerte: ${sr_levels['strong_resistance']:.2f}")
    print(f"Resistencia: ${sr_levels['resistance']:.2f}")
    print(f"Precio actual: ${analysis_result['current_price']:.2f}")
    print(f"Soporte: ${sr_levels['support']:.2f}")
    print(f"Soporte fuerte: ${sr_levels['strong_support']:.2f}")
    
    # Mostrar predicciones
    print("\nPredicciones:")
    for term, pred in analysis_result["predictions"].items():
        print(f"• {term}: {pred['direction']} (confianza: {pred['confidence']:.2f}) → ${pred['target_price']:.2f}")
    
    # Mostrar señales de trading
    print("\nSeñales de trading:")
    signals = analysis_result["trading_signals"]
    for strategy, signal in signals.items():
        signal_icon = "✅" if signal == "buy" else "❌" if signal == "sell" else "⚠️"
        print(f"{signal_icon} {strategy}: {signal}")
    
    # Mostrar recomendación
    rec = analysis_result["recommendation"]
    print("\nRecomendación:")
    print(f"Acción: {rec['action'].upper()}")
    print(f"Entrada: ${rec['entry_price']:.2f}")
    print(f"Take profit: ${rec['take_profit']:.2f} (+{(rec['take_profit']/rec['entry_price']-1)*100:.2f}%)")
    print(f"Stop loss: ${rec['stop_loss']:.2f} (-{(1-rec['stop_loss']/rec['entry_price'])*100:.2f}%)")
    print(f"Ratio riesgo/recompensa: {rec['risk_reward_ratio']:.2f}")
    print(f"Confianza: {rec['confidence']:.2f}")
    print(f"Estrategia: {rec['strategy']}")
    print(f"Tamaño recomendado: {rec['position_size_pct']:.1f}% del capital")

def main():
    """Función principal de demostración."""
    print("\n🧠 DEMOSTRACIÓN DE CAPACIDADES AVANZADAS DEL BOT 🧠")
    print("Este script muestra las nuevas capacidades de inteligencia")
    print("artificial, aprendizaje y portabilidad del bot de trading.")
    
    # 1. Demostración del sistema de transferencia de cerebro
    demo_brain_transfer()
    
    # 2. Demostración de estrategias de scalping
    demo_scalping_strategies()
    
    # 3. Demostración de predicción con machine learning
    demo_ml_prediction()
    
    # 4. Comparación de estrategias
    compare_strategies()
    
    # 5. Demostración de análisis automatizado
    demo_automated_analysis()
    
    print("\n✅ Demostración completa ejecutada con éxito.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemostración interrumpida por el usuario.")
    except Exception as e:
        print(f"\nError durante la demostración: {e}")