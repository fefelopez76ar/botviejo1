import os
import re

file_path_m1 = 'main1.py'
with open(file_path_m1, 'r', encoding='utf-8') as f:
    content_m1 = f.read()

# Corrección 1: Línea 206 - Tickers (agregar instType)
old_tickers_sub = '        {"channel": "tickers", "instId": "SOL-USDT"}'
new_tickers_sub = '        {"channel": "tickers", "instId": "SOL-USDT", "instType": "SPOT"}'
content_m1 = content_m1.replace(old_tickers_sub, new_tickers_sub)

# Corrección 2: Línea 214 - Candles (cambiar a candle y eliminar bar)
old_candles_sub = '        {"channel": "candles", "instId": "SOL-USDT", "bar": "1m"}'
new_candles_sub = '        {"channel": "candle", "instId": "SOL-USDT"}'
content_m1 = content_m1.replace(old_candles_sub, new_candles_sub)

with open(file_path_m1, 'w', encoding='utf-8') as f:
    f.write(content_m1)

print('--> main1.py: Líneas de suscripción de Tickers y Candles corregidas directamente.')