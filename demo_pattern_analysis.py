#!/usr/bin/env python3
"""
Demostración de análisis de patrones para evaluar conveniencia de compra/venta de Solana
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configurar logging
logging.basicConfig(level=logging.INFO)

def divider(title=""):
    """Imprime un divisor con título opcional"""
    width = 80
    if title:
        padding = (width - len(title) - 4) // 2
        print("\n" + "=" * padding + f" {title} " + "=" * padding)
    else:
        print("\n" + "=" * width)

def analyze_solana_trends():
    """
    Análisis simplificado de tendencias y patrones para Solana
    """
    divider("ANÁLISIS DE TENDENCIAS DE SOLANA")
    
    # Precio actual simulado de Solana
    current_price = 150.50
    
    # Datos simulados para diferentes timeframes
    timeframes = {
        '2m': {'trend': 'neutral', 'strength': 'weak', 'pattern': 'consolidation'},
        '15m': {'trend': 'bearish', 'strength': 'moderate', 'pattern': 'descending_triangle'},
        '1h': {'trend': 'neutral', 'strength': 'weak', 'pattern': 'flag'},
        '4h': {'trend': 'bullish', 'strength': 'weak', 'pattern': 'double_bottom'},
        '1d': {'trend': 'neutral', 'strength': 'moderate', 'pattern': 'range_bound'}
    }
    
    # Niveles clave de soporte y resistencia
    key_levels = {
        'resistance_2': 155.80,
        'resistance_1': 152.30,
        'support_1': 149.20,
        'support_2': 145.60
    }
    
    # Métricas de volatilidad
    volatility = {
        'current': 0.92,  # Volatilidad actual en porcentaje
        'historical_avg': 1.05,  # Promedio histórico
        'relative': 0.88  # Volatilidad relativa (actual/promedio)
    }
    
    # Indicadores técnicos
    indicators = {
        'rsi': 45.2,  # RSI (30 = sobrevendido, 70 = sobrecomprado)
        'macd': -0.15,  # MACD (negativo = bajista, positivo = alcista)
        'macd_signal': -0.08,  # Señal MACD
        'macd_histogram': -0.07,  # Histograma MACD
        'bb_width': 1.2,  # Ancho de bandas de Bollinger (>1 = alta volatilidad)
        'sma_20': 151.20,  # Media móvil simple 20 periodos
        'sma_50': 150.80,  # Media móvil simple 50 periodos
        'sma_200': 149.90  # Media móvil simple 200 periodos
    }
    
    # Amplitudes de precio por timeframe
    price_amplitudes = {
        '1m': 0.15,
        '5m': 0.35,
        '15m': 0.65,
        '1h': 1.20,
        '4h': 2.50,
        '1d': 4.80
    }
    
    # Probabilidades de reversión basadas en patrones históricos
    reversal_probs = {
        'from_top': 0.65,  # Probabilidad de reversión desde máximo
        'from_bottom': 0.72,  # Probabilidad de reversión desde mínimo
        'continuation': 0.58  # Probabilidad de continuación de tendencia
    }
    
    # Patrones de mercado detectados
    detected_patterns = [
        {
            'type': 'double_bottom',
            'timeframe': '4h',
            'reliability': 0.68,
            'target_price': 154.20,
            'stop_loss': 148.50
        },
        {
            'type': 'descending_triangle',
            'timeframe': '15m',
            'reliability': 0.72,
            'target_price': 148.30,
            'stop_loss': 152.10
        }
    ]
    
    # Mostrar información general
    print(f"Análisis para SOL-USDT - Precio actual: ${current_price:.2f}")
    print(f"Fecha y hora del análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Mostrar análisis por timeframe
    print("\nAnálisis por timeframe:")
    print(f"{'Timeframe':<10} {'Tendencia':<12} {'Fuerza':<12} {'Patrón':<20}")
    print("-" * 54)
    
    for tf, data in timeframes.items():
        print(f"{tf:<10} {data['trend']:<12} {data['strength']:<12} {data['pattern']:<20}")
    
    # Mostrar niveles clave
    print("\nNiveles clave:")
    print(f"Resistencia 2: ${key_levels['resistance_2']:.2f}")
    print(f"Resistencia 1: ${key_levels['resistance_1']:.2f}")
    print(f"Precio actual: ${current_price:.2f}")
    print(f"Soporte 1: ${key_levels['support_1']:.2f}")
    print(f"Soporte 2: ${key_levels['support_2']:.2f}")
    
    # Distancia a niveles clave
    r1_distance = ((key_levels['resistance_1'] / current_price) - 1) * 100
    r2_distance = ((key_levels['resistance_2'] / current_price) - 1) * 100
    s1_distance = ((current_price / key_levels['support_1']) - 1) * 100
    s2_distance = ((current_price / key_levels['support_2']) - 1) * 100
    
    print(f"\nDistancia a niveles clave:")
    print(f"A resistencia 1: {r1_distance:.2f}%")
    print(f"A resistencia 2: {r2_distance:.2f}%")
    print(f"A soporte 1: {s1_distance:.2f}%")
    print(f"A soporte 2: {s2_distance:.2f}%")
    
    # Mostrar indicadores técnicos clave
    print("\nIndicadores técnicos:")
    rsi_state = "neutral"
    if indicators['rsi'] < 30:
        rsi_state = "sobrevendido"
    elif indicators['rsi'] > 70:
        rsi_state = "sobrecomprado"
    
    macd_state = "bearish" if indicators['macd'] < 0 else "bullish"
    if abs(indicators['macd'] - indicators['macd_signal']) < 0.05:
        macd_state += " (cruce potencial)"
    
    print(f"RSI (14): {indicators['rsi']:.1f} - {rsi_state}")
    print(f"MACD: {indicators['macd']:.3f} - {macd_state}")
    print(f"Bandas Bollinger - Ancho: {indicators['bb_width']:.2f}x")
    
    # Análisis de medias móviles
    print("\nAnálisis de medias móviles:")
    print(f"SMA 20: ${indicators['sma_20']:.2f} - Precio {'por debajo' if current_price < indicators['sma_20'] else 'por encima'}")
    print(f"SMA 50: ${indicators['sma_50']:.2f} - Precio {'por debajo' if current_price < indicators['sma_50'] else 'por encima'}")
    print(f"SMA 200: ${indicators['sma_200']:.2f} - Precio {'por debajo' if current_price < indicators['sma_200'] else 'por encima'}")
    
    # Evaluación de patrones detectados
    if detected_patterns:
        divider("PATRONES DETECTADOS")
        for pattern in detected_patterns:
            print(f"Patrón: {pattern['type']} en timeframe {pattern['timeframe']}")
            print(f"Confiabilidad: {pattern['reliability'] * 100:.1f}%")
            print(f"Precio objetivo: ${pattern['target_price']:.2f} ({((pattern['target_price']/current_price)-1)*100:.2f}%)")
            print(f"Stop loss sugerido: ${pattern['stop_loss']:.2f} ({((pattern['stop_loss']/current_price)-1)*100:.2f}%)")
            
            # Calcular ratio riesgo/recompensa
            if pattern['type'] in ['double_bottom', 'ascending_triangle']:
                # Patrones alcistas
                reward = pattern['target_price'] - current_price
                risk = current_price - pattern['stop_loss']
            else:
                # Patrones bajistas
                reward = current_price - pattern['target_price']
                risk = pattern['stop_loss'] - current_price
            
            rr_ratio = reward / risk if risk > 0 else 0
            print(f"Ratio riesgo/recompensa: {rr_ratio:.2f}")
            print("")
    
    # Análisis de probabilidades de reversión
    divider("PROBABILIDADES ESTADÍSTICAS")
    
    print(f"Amplitud de precio típica:")
    for tf, amplitude in price_amplitudes.items():
        print(f"  {tf}: ${amplitude:.2f} ({amplitude/current_price*100:.2f}%)")
    
    print(f"\nProbabilidades basadas en patrones históricos:")
    print(f"  Reversión desde techo: {reversal_probs['from_top']*100:.1f}%")
    print(f"  Reversión desde suelo: {reversal_probs['from_bottom']*100:.1f}%")
    print(f"  Continuación de tendencia: {reversal_probs['continuation']*100:.1f}%")
    
    # Evaluar mejor estrategia según timeframe
    divider("RECOMENDACIÓN DE ESTRATEGIA")
    
    # Análisis combinado para decisión final
    # Ponderación de diferentes factores
    weights = {
        '2m': 0.05,
        '15m': 0.15,
        '1h': 0.25,
        '4h': 0.30,
        '1d': 0.25
    }
    
    long_score = 0
    short_score = 0
    
    # Análisis de tendencia ponderado por timeframe
    for tf, data in timeframes.items():
        weight = weights.get(tf, 0.1)
        tf_score = 0
        
        if data['trend'] == 'bullish':
            tf_score = 1
        elif data['trend'] == 'bearish':
            tf_score = -1
            
        # Ajustar por fuerza de la tendencia
        if data['strength'] == 'strong':
            tf_score *= 1.5
        elif data['strength'] == 'weak':
            tf_score *= 0.5
            
        if tf_score > 0:
            long_score += tf_score * weight
        else:
            short_score -= tf_score * weight
    
    # Ajuste por indicadores técnicos
    # RSI
    if indicators['rsi'] < 30:
        long_score += 0.2
    elif indicators['rsi'] > 70:
        short_score += 0.2
        
    # MACD
    if indicators['macd'] > 0:
        long_score += 0.15
    else:
        short_score += 0.15
    
    # Soportes y resistencias
    price_range = key_levels['resistance_1'] - key_levels['support_1']
    
    # Si está más cerca al soporte, favorece largo
    if (current_price - key_levels['support_1']) < (key_levels['resistance_1'] - current_price):
        long_score += 0.1
    else:
        short_score += 0.1
    
    # Identificar mejor estrategia basada en scores
    best_strategy = "NEUTRAL"
    if long_score > 0.5 and long_score > short_score * 1.2:
        best_strategy = "LONG"
    elif short_score > 0.5 and short_score > long_score * 1.2:
        best_strategy = "SHORT"
    
    # Analizar riesgo/recompensa
    long_risk = current_price - key_levels['support_1']
    long_reward = key_levels['resistance_1'] - current_price
    long_rr = long_reward / long_risk if long_risk > 0 else 0
    
    short_risk = key_levels['resistance_1'] - current_price
    short_reward = current_price - key_levels['support_1']
    short_rr = short_reward / short_risk if short_risk > 0 else 0
    
    # Determinar si es conveniente operar
    should_trade_long = long_rr >= 1.5 and long_score > 0.4
    should_trade_short = short_rr >= 1.5 and short_score > 0.4
    
    print(f"Score de estrategia LONG: {long_score:.2f}")
    print(f"Score de estrategia SHORT: {short_score:.2f}")
    print(f"Ratio riesgo/recompensa LONG: {long_rr:.2f}")
    print(f"Ratio riesgo/recompensa SHORT: {short_rr:.2f}")
    
    print(f"\nEstrategia recomendada: {best_strategy}")
    
    if best_strategy == "LONG":
        if should_trade_long:
            print(f"✅ COMPRAR - Configuración favorable")
            print(f"  Entrada: ${current_price:.2f}")
            print(f"  Take profit: ${key_levels['resistance_1']:.2f} (+{r1_distance:.2f}%)")
            print(f"  Stop loss: ${key_levels['support_1']:.2f} (-{s1_distance:.2f}%)")
        else:
            print(f"⚠️ ESPERAR - Sesgo alcista pero ratio riesgo/recompensa insuficiente")
            
    elif best_strategy == "SHORT":
        if should_trade_short:
            print(f"✅ VENDER - Configuración favorable")
            print(f"  Entrada: ${current_price:.2f}")
            print(f"  Take profit: ${key_levels['support_1']:.2f} (-{s1_distance:.2f}%)")
            print(f"  Stop loss: ${key_levels['resistance_1']:.2f} (+{r1_distance:.2f}%)")
        else:
            print(f"⚠️ ESPERAR - Sesgo bajista pero ratio riesgo/recompensa insuficiente")
    else:
        print(f"⚠️ NEUTRAL - Mejor esperar a una configuración más clara")
    
    # Horizontes temporales recomendados según volatilidad
    if volatility['relative'] > 1.2:
        print(f"\nVolatilidad alta: Recomendable operar en horizontes cortos (scalping/day trading)")
    elif volatility['relative'] < 0.8:
        print(f"\nVolatilidad baja: Posible oportunidad para swing trading")
    else:
        print(f"\nVolatilidad normal: Usar timeframes según tu estilo de trading habitual")
    
    return {
        "price": current_price,
        "timeframes": timeframes,
        "best_strategy": best_strategy,
        "should_trade_long": should_trade_long,
        "should_trade_short": should_trade_short,
        "long_score": long_score,
        "short_score": short_score,
        "long_rr": long_rr,
        "short_rr": short_rr
    }

def main():
    print("\n🤖 EVALUACIÓN DE CONVENIENCIA DE TRADING PARA SOLANA 🤖")
    print("Este análisis evalúa si conviene comprar o vender Solana")
    print("basándose en análisis técnico, patrones y amplitudes de precio")
    
    # Ejecutar análisis
    result = analyze_solana_trends()
    
    # Aplicar sesgo de comisiones para operaciones en corto
    fee_impact_long = 0.1  # 0.1% impacto en long
    fee_impact_short = 0.15  # 0.15% impacto en short
    
    divider("ANÁLISIS FINAL INCLUYENDO COMISIONES")
    
    print(f"Impacto de comisiones en operaciones LONG: {fee_impact_long}%")
    print(f"Impacto de comisiones en operaciones SHORT: {fee_impact_short}%")
    
    # Ajustar scores por impacto de comisiones
    long_score_adj = result["long_score"] * (1 - fee_impact_long/100)
    short_score_adj = result["short_score"] * (1 - fee_impact_short/100)
    
    # Ajustar ratios R/R por impacto de comisiones
    long_rr_adj = result["long_rr"] * (1 - fee_impact_long/100)
    short_rr_adj = result["short_rr"] * (1 - fee_impact_short/100)
    
    print(f"\nScore ajustado LONG: {long_score_adj:.2f} (antes: {result['long_score']:.2f})")
    print(f"Score ajustado SHORT: {short_score_adj:.2f} (antes: {result['short_score']:.2f})")
    print(f"R/R ajustado LONG: {long_rr_adj:.2f} (antes: {result['long_rr']:.2f})")
    print(f"R/R ajustado SHORT: {short_rr_adj:.2f} (antes: {result['short_rr']:.2f})")
    
    # Determinar conveniencia final
    if long_score_adj > short_score_adj and long_rr_adj >= 1.5:
        print(f"\n✅ CONCLUSIÓN: COMPRAR - Los beneficios superan los costos")
    elif short_score_adj > long_score_adj and short_rr_adj >= 1.5:
        print(f"\n✅ CONCLUSIÓN: VENDER - Los beneficios superan los costos")
    else:
        print(f"\n⚠️ CONCLUSIÓN: NEUTRAL - Mejor esperar")
    
    print("\n✅ Análisis completado")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError durante la demostración: {e}")