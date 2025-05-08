#!/usr/bin/env python3
"""
Demostración simplificada de trading en corto con cálculo de comisiones
"""

import os
import sys
import logging
from datetime import datetime

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

def demo_short_trade():
    """
    Simulación simplificada de una operación en corto con cálculo de comisiones
    """
    divider("SIMULACIÓN DE TRADING EN CORTO")
    
    # Datos de la operación
    symbol = "SOL-USDT"
    entry_price = 151.0   # Precio de entrada para la posición corta
    exit_price = 145.2    # Precio de salida (más bajo = ganancia)
    position_size = 10    # 10 SOL
    leverage = 3          # Apalancamiento 3x
    hours_held = 12       # Duración de la posición
    is_short = True       # Es una posición corta
    
    # Tasas y comisiones (OKX)
    maker_fee_rate = 0.0002  # 0.02% para makers
    taker_fee_rate = 0.0005  # 0.05% para takers
    funding_rate = 0.0001    # 0.01% por cada 8 horas
    
    # Cálculo de comisiones
    position_value = position_size * entry_price  # Valor de la posición
    entry_fee = position_value * taker_fee_rate   # Comisión de entrada (asumiendo orden de mercado)
    exit_fee = position_size * exit_price * taker_fee_rate  # Comisión de salida (asumiendo orden de mercado)
    
    # Cálculo del costo de financiamiento
    funding_intervals = hours_held / 8  # Cada cuántos intervalos de 8 horas
    funding_intervals = int(funding_intervals) + (1 if funding_intervals % 1 > 0 else 0)  # Redondear hacia arriba
    funding_fee = position_value * funding_rate * funding_intervals
    
    # Cálculo de P&L
    price_diff = entry_price - exit_price  # Diferencia de precio (positiva para ganancia en corto)
    gross_pnl = position_size * price_diff * leverage  # P&L bruto
    total_fees = entry_fee + exit_fee + funding_fee    # Total de comisiones
    net_pnl = gross_pnl - total_fees                   # P&L neto
    
    # Cálculo de ROI
    roi_pct = (net_pnl / (position_value / leverage)) * 100  # ROI sobre margen utilizado
    
    # Impacto de comisiones
    fee_impact_pct = (total_fees / position_value) * 100
    
    # Mostrar resultados
    print(f"Operación CORTA en {symbol}:")
    print(f"Precio entrada: ${entry_price:.2f}")
    print(f"Precio salida: ${exit_price:.2f}")
    print(f"Tamaño: {position_size} SOL")
    print(f"Apalancamiento: {leverage}x")
    print(f"Tiempo mantenida: {hours_held} horas")
    print(f"Intervalos de financiamiento: {funding_intervals}")
    
    print("\nCosto de comisiones:")
    print(f"Comisión entrada: ${entry_fee:.2f} ({taker_fee_rate*100:.3f}%)")
    print(f"Comisión salida: ${exit_fee:.2f} ({taker_fee_rate*100:.3f}%)")
    print(f"Comisión financiamiento: ${funding_fee:.2f} ({funding_rate*funding_intervals*100:.3f}%)")
    print(f"Comisiones totales: ${total_fees:.2f}")
    print(f"Impacto comisiones: {fee_impact_pct:.2f}%")
    
    print("\nResultados de la operación:")
    print(f"P&L bruto: ${gross_pnl:.2f} ({price_diff/entry_price*100*leverage:.2f}%)")
    print(f"P&L neto: ${net_pnl:.2f}")
    print(f"ROI sobre margen: {roi_pct:.2f}%")
    
    # Mostrar comparación con operación similar en largo
    divider("COMPARACIÓN CON OPERACIÓN EN LARGO")
    
    # En una posición larga, necesitamos que el precio suba para ganar
    long_entry_price = 145.2  # Compra a precio bajo
    long_exit_price = 151.0   # Vende a precio alto
    
    # Los cálculos son similares pero la diferencia de precio es inversa
    long_price_diff = long_exit_price - long_entry_price
    long_gross_pnl = position_size * long_price_diff * leverage
    long_position_value = position_size * long_entry_price
    
    long_entry_fee = long_position_value * taker_fee_rate
    long_exit_fee = position_size * long_exit_price * taker_fee_rate
    
    # El financiamiento para posiciones largas es similar
    long_funding_fee = long_position_value * funding_rate * funding_intervals
    
    long_total_fees = long_entry_fee + long_exit_fee + long_funding_fee
    long_net_pnl = long_gross_pnl - long_total_fees
    
    long_roi_pct = (long_net_pnl / (long_position_value / leverage)) * 100
    long_fee_impact_pct = (long_total_fees / long_position_value) * 100
    
    print(f"Operación LARGA comparable:")
    print(f"Precio entrada: ${long_entry_price:.2f}")
    print(f"Precio salida: ${long_exit_price:.2f}")
    
    print("\nCosto de comisiones en largo:")
    print(f"Comisiones totales: ${long_total_fees:.2f}")
    print(f"Impacto comisiones: {long_fee_impact_pct:.2f}%")
    
    print("\nResultados de la operación en largo:")
    print(f"P&L bruto: ${long_gross_pnl:.2f} ({long_price_diff/long_entry_price*100*leverage:.2f}%)")
    print(f"P&L neto: ${long_net_pnl:.2f}")
    print(f"ROI sobre margen: {long_roi_pct:.2f}%")
    
    # Comparación de diferentes niveles de apalancamiento
    divider("IMPACTO DEL APALANCAMIENTO")
    
    print(f"Comparación de rentabilidad con diferentes niveles de apalancamiento:")
    print(f"(Mismo tamaño de posición, mismo movimiento de precio)")
    
    leverage_levels = [1, 2, 3, 5, 10]
    
    print("\nPara posición CORTA:")
    print(f"{'Apalancamiento':<15} {'P&L Bruto':<12} {'Comisiones':<12} {'P&L Neto':<12} {'ROI':<10}")
    
    for lev in leverage_levels:
        lev_gross_pnl = position_size * price_diff * lev
        lev_funding_fee = position_value * funding_rate * funding_intervals * (lev if lev > 1 else 0)
        lev_total_fees = entry_fee + exit_fee + lev_funding_fee
        lev_net_pnl = lev_gross_pnl - lev_total_fees
        lev_roi_pct = (lev_net_pnl / (position_value / lev)) * 100 if lev > 0 else 0
        
        print(f"{lev}x{'':<12} ${lev_gross_pnl:<10.2f} ${lev_total_fees:<10.2f} ${lev_net_pnl:<10.2f} {lev_roi_pct:<8.2f}%")
    
    return {
        "symbol": symbol,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "position_size": position_size,
        "leverage": leverage,
        "hours_held": hours_held,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "total_fees": total_fees,
        "roi_pct": roi_pct
    }

def main():
    print("\n🤖 DEMOSTRACIÓN DE TRADING EN CORTO CON CÁLCULO DE COMISIONES 🤖")
    
    # Ejecutar demo de trading en corto
    result = demo_short_trade()
    
    print("\n✅ Demostración completada con éxito.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError durante la demostración: {e}")