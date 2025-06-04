import os
import re

file_path_m1 = 'main1.py'
with open(file_path_m1, 'r', encoding='utf-8') as f:
    content_m1 = f.read()

new_content_m1 = re.sub(r'(\{"channel": "candles", "instId": "SOL-USDT", "bar": "1m"\}\])', r'{"channel": "candles", "instId": "SOL-USDT"}]', content_m1, count=1)

with open(file_path_m1, 'w', encoding='utf-8') as f:
    f.write(new_content_m1)

print('--> main1.py: Suscripción a Candles modificada (se eliminó "bar").')