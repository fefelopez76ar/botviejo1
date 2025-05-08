#!/usr/bin/env python3
"""
Script de demostración para analizar conveniencia de trading en Solana
Incluye análisis de:
- Patrones de mercado y probabilidades de reversión
- Decisiones inteligentes de compra/venta
- Simulación de operaciones en corto
- Cálculo detallado de comisiones y costos de financiamiento
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tabulate import tabulate

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Importar módulos necesarios
from strategies.pattern_analyzer import demo_pattern_analysis, analyze_multiple_timeframes
from risk_management.fee_calculator import FeeCalculator
from core.short_trading import ShortTradingSimulator, short_trading_example

def divider(title=""):
    """Imprime un divisor con título opcional"""
    width = 80
    if title:
        padding = (width - len(title) - 4) // 2
        print("\n" + "=" * padding + f" {title} " + "=" * padding)
    else:
        print("\n" + "=" * width)

def run_pattern_analysis():
    """Ejecuta análisis de patrones para Solana"""
    divider("ANÁLISIS DE PATRONES")
    
    # Analizar Solana en timeframe de 1 hora
    demo_pattern_analysis("SOL-USDT", "1h")

def run_multi_timeframe_analysis():
    """Ejecuta análisis multi-timeframe para Solana"""
    divider("ANÁLISIS MULTI-TIMEFRAME")
    
    # Analizar Solana en múltiples timeframes
    result = analyze_multiple_timeframes("SOL-USDT", ['15m', '1h', '4h', '1d'])
    
    # Mostrar resumen
    print("\nResumen de análisis en múltiples timeframes:")
    print(f"Par: {result['symbol']}")
    print(f"Timestamp: {result['timestamp']}")
    
    mta = result['multi_timeframe_analysis']
    print(f"\nSeñales de compra: {mta['long_signals']}")
    print(f"Señales de venta: {mta['short_signals']}")
    print(f"Señales neutrales: {mta['neutral_signals']}")
    print(f"Score ponderado: {mta['weighted_score']:.2f}")
    print(f"Señal combinada: {mta['signal']}")
    print(f"Tendencia dominante: {mta['dominant_trend']}")
    
    # Mostrar resultados por timeframe
    print("\nResultados por timeframe:")
    data = []
    for tf, res in result['individual_results'].items():
        if 'error' not in res:
            eval_data = res['evaluation']
            data.append([
                tf,
                eval_data['position_type'],
                f"{eval_data['combined_score']:.2f}",
                f"{eval_data['risk_reward_ratio']:.2f}",
                "Sí" if eval_data['should_trade'] else "No"
            ])
    
    headers = ["Timeframe", "Posición", "Score", "Riesgo/Recompensa", "¿Operar?"]
    print(tabulate(data, headers=headers, tablefmt="grid"))

def run_short_trading_simulation():
    """Ejecuta simulación de trading en corto"""
    divider("SIMULACIÓN DE TRADING EN CORTO")
    
    # Ejecutar ejemplo de operación corta
    short_result = short_trading_example()
    
    # Mostrar resultados de diferentes escenarios
    print("\nComparación de diferentes escenarios de precio:")
    
    scenarios = short_result.get("scenarios", [])
    
    if scenarios:
        data = []
        for scenario in scenarios:
            data.append([
                f"${scenario['current_price']:.2f}",
                f"${scenario['pnl_info']['pnl_amount']:.2f}",
                f"{scenario['pnl_info']['pnl_pct']:.2f}%",
                f"${scenario['entry_fee'] + scenario['current_funding_fees'] + scenario['estimated_exit_fee']:.2f}",
                f"${scenario['pnl_info']['net_pnl']:.2f}",
                f"{scenario['pnl_info']['roi']:.2f}%"
            ])
        
        headers = ["Precio", "P&L Bruto", "P&L %", "Comisiones", "P&L Neto", "ROI"]
        print(tabulate(data, headers=headers, tablefmt="grid"))

def run_fee_comparison():
    """Compara comisiones entre diferentes modos de trading"""
    divider("COMPARACIÓN DE COMISIONES")
    
    # Definir manualmente diferentes modos de trading
    comparison = [
        {
            "mode": "Spot Trading (Solo Long)",
            "leverage": 1.0,
            "short_enabled": False,
            "avg_hours_held": 8.0,
            "trades_per_day": 3,
            "trading_fees": 45.00,
            "funding_fees": 0.00,
            "total_fees": 45.00,
            "daily_avg_fees": 1.50,
            "fee_impact_per_trade_pct": 0.10
        },
        {
            "mode": "Margin Trading (Long y Short)",
            "leverage": 2.0,
            "short_enabled": True,
            "avg_hours_held": 10.0,
            "trades_per_day": 3,
            "trading_fees": 36.00,
            "funding_fees": 15.00,
            "total_fees": 51.00,
            "daily_avg_fees": 1.70,
            "fee_impact_per_trade_pct": 0.11
        },
        {
            "mode": "Futures Trading (Long y Short)",
            "leverage": 3.0,
            "short_enabled": True,
            "avg_hours_held": 12.0,
            "trades_per_day": 3,
            "trading_fees": 22.50,
            "funding_fees": 18.00,
            "total_fees": 40.50,
            "daily_avg_fees": 1.35,
            "fee_impact_per_trade_pct": 0.09
        },
        {
            "mode": "Scalping Futures (High Frequency)",
            "leverage": 5.0,
            "short_enabled": True,
            "avg_hours_held": 2.0,
            "trades_per_day": 10,
            "trading_fees": 75.00,
            "funding_fees": 10.00,
            "total_fees": 85.00,
            "daily_avg_fees": 2.83,
            "fee_impact_per_trade_pct": 0.08
        }
    ]
    
    # Mostrar tabla comparativa
    data = []
    for mode in comparison:
        data.append([
            mode['mode'],
            mode['leverage'],
            "Sí" if mode['short_enabled'] else "No",
            f"{mode['avg_hours_held']:.1f}h",
            mode['trades_per_day'],
            f"${mode['trading_fees']:.2f}",
            f"${mode['funding_fees']:.2f}",
            f"${mode['total_fees']:.2f}",
            f"${mode['daily_avg_fees']:.2f}",
            f"{mode['fee_impact_per_trade_pct']:.2f}%"
        ])
    
    headers = [
        "Modo", "Apalancamiento", "Short", "Tiempo", "Ops/día", 
        "Com. Trading", "Com. Financiamiento", "Comisiones Total", 
        "Com. Diarias", "Impacto/Op"
    ]
    
    print(tabulate(data, headers=headers, tablefmt="grid"))
    
    # Destacar opciones óptimas
    min_fees_idx = min(range(len(comparison)), key=lambda i: comparison[i]['fee_impact_per_trade_pct'])
    print(f"\nModo con menor impacto de comisiones: {comparison[min_fees_idx]['mode']}")
    print(f"Impacto de comisiones por operación: {comparison[min_fees_idx]['fee_impact_per_trade_pct']:.2f}%")

def run_trade_decision_analysis():
    """Analiza decisión de compra/venta para Solana"""
    divider("ANÁLISIS DE DECISIÓN DE TRADING")
    
    from strategies.pattern_analyzer import evaluate_trading_opportunity
    from data_management.market_data import get_market_data
    
    # Obtener datos actuales
    df = get_market_data("SOL-USDT", "1h")
    current_price = df['close'].iloc[-1] if df is not None and not df.empty else 150.0
    
    print(f"Precio actual de Solana: ${current_price:.2f}")
    
    # Evaluar oportunidad en diferentes timeframes
    timeframes = ['15m', '1h', '4h', '1d']
    
    for tf in timeframes:
        print(f"\n--- Análisis en timeframe {tf} ---")
        result = evaluate_trading_opportunity("SOL-USDT", tf)
        
        if 'error' not in result:
            eval_data = result['evaluation']
            print(f"Recomendación: {eval_data['recommendation']}")
            print(f"Tipo de posición: {eval_data['position_type']}")
            print(f"Score combinado: {eval_data['combined_score']:.2f}")
            print(f"Ratio riesgo/recompensa: {eval_data['risk_reward_ratio']:.2f}")
            
            # Mostrar niveles clave
            levels = eval_data['key_levels']
            print(f"\nNiveles clave:")
            print(f"Resistencia 2: ${levels['resistance_2']:.2f}")
            print(f"Resistencia 1: ${levels['resistance_1']:.2f}")
            print(f"Precio actual: ${eval_data['current_price']:.2f}")
            print(f"Soporte 1: ${levels['support_1']:.2f}")
            print(f"Soporte 2: ${levels['support_2']:.2f}")
        else:
            print(f"Error: {result['error']}")

def main():
    """Función principal"""
    print("\n🤖 SISTEMA DE ANÁLISIS INTELIGENTE PARA SOLANA 🤖")
    print("Este análisis utiliza datos en tiempo real para evaluar")
    print("la conveniencia de comprar o vender Solana, incluyendo")
    print("análisis de patrones, costos de comisiones y simulaciones.")
    
    # 1. Análisis de patrones
    run_pattern_analysis()
    
    # 2. Análisis multi-timeframe
    run_multi_timeframe_analysis()
    
    # 3. Simulación de trading en corto
    run_short_trading_simulation()
    
    # 4. Comparación de comisiones
    run_fee_comparison()
    
    # 5. Análisis de decisión final
    run_trade_decision_analysis()
    
    print("\n✅ Análisis completo ejecutado con éxito.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAnálisis interrumpido por el usuario.")
    except Exception as e:
        print(f"\nError durante el análisis: {e}")