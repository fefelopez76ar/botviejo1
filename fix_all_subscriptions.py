import os
import re

file_path_m1 = 'main1.py'
with open(file_path_m1, 'r', encoding='utf-8') as f:
    content_m1 = f.read()

# Corrección 1: candles -> candle y eliminar bar
old_candles_sub = "[{'channel': 'candles', 'instId': 'SOL-USDT', 'bar': '1m'}]"
new_candles_sub = "[{'channel': 'candle', 'instId': 'SOL-USDT'}]"
escaped_old_candles_sub = re.escape(old_candles_sub)
new_content_m1 = re.sub(escaped_old_candles_sub, new_candles_sub, content_m1, count=1)

# Corrección 2: tickers -> agregar instType SPOT
old_tickers_sub = "[{'channel': 'tickers', 'instId': 'SOL-USDT'}]"
new_tickers_sub = "[{'channel': 'tickers', 'instId': 'SOL-USDT', 'instType': 'SPOT'}]"
escaped_old_tickers_sub = re.escape(old_tickers_sub)
new_content_m1 = re.sub(escaped_old_tickers_sub, new_tickers_sub, new_content_m1, count=1)

with open(file_path_m1, 'w', encoding='utf-8') as f:
    f.write(new_content_m1)

print('--> main1.py: Suscripción a Candles corregida (candles -> candle, eliminado bar).')
print('--> main1.py: Suscripción a Tickers corregida (agregado instType: SPOT).')