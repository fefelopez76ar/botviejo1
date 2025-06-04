#!/usr/bin/env python3
import sys
import os
import subprocess

# Cambiar al directorio de trabajo
os.chdir('/home/runner/workspace')

# Ejecutar main1.py
try:
    result = subprocess.run([sys.executable, 'main1.py'], 
                          capture_output=False, 
                          text=True, 
                          timeout=30)
except subprocess.TimeoutExpired:
    print("Bot ejecutado por 30 segundos - timeout alcanzado")
except KeyboardInterrupt:
    print("Bot detenido por usuario")
except Exception as e:
    print(f"Error al ejecutar bot: {e}")