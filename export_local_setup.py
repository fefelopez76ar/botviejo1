#!/usr/bin/env python3
"""
Script para exportar configuración del bot para uso local en PC
"""
import os
import json
from pathlib import Path

def create_local_setup():
    """Crea archivos de configuración para usar en PC local"""
    
    # Lista de archivos importantes del bot
    important_files = [
        'main.py',
        'trading_bot.py', 
        'start_bot.py',
        'log_viewer.py',
        'stats_tracker.py',
        'verify_price.py',
        'learning_data.json',
        'config.env',
        'api_client/modulo2.py',
        'api_client/modulocola.py',
        'data_management/historical_data_saver_async.py',
        'BOT_STATUS.md',
        'BOT_MANUAL.md'
    ]
    
    print("CONFIGURACIÓN PARA USO LOCAL DEL BOT")
    print("=" * 50)
    print()
    
    print("1. ARCHIVOS PRINCIPALES DEL BOT:")
    for file in important_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"   ✓ {file} ({size} bytes)")
        else:
            print(f"   ✗ {file} (NO ENCONTRADO)")
    
    print()
    print("2. DEPENDENCIAS REQUERIDAS:")
    dependencies = [
        'asyncio',
        'websockets', 
        'ccxt',
        'pandas',
        'numpy',
        'python-dotenv',
        'requests',
        'flask',
        'tabulate'
    ]
    
    for dep in dependencies:
        print(f"   - {dep}")
    
    print()
    print("3. COMANDOS PARA INSTALAR EN TU PC:")
    print("   pip install websockets ccxt pandas numpy python-dotenv requests flask tabulate")
    
    print()
    print("4. CONFIGURACIÓN DE CREDENCIALES:")
    print("   Crea archivo config.env con:")
    print("   OKX_API_KEY=tu_api_key")
    print("   OKX_API_SECRET=tu_secret_key") 
    print("   OKX_PASSPHRASE=tu_passphrase")
    
    print()
    print("5. COMANDOS PARA EJECUTAR EL BOT:")
    print("   python main.py          # Bot con interfaz web")
    print("   python trading_bot.py   # Bot modo consola")
    print("   python start_bot.py     # Bot simplificado")
    print("   python log_viewer.py    # Ver estadísticas")
    print("   python verify_price.py  # Verificar estado")
    
    # Crear archivo de instrucciones
    instructions = """# SolanaScalper Bot - Instrucciones de Uso Local

## Instalación

1. Instalar dependencias:
```bash
pip install websockets ccxt pandas numpy python-dotenv requests flask tabulate
```

2. Configurar credenciales en config.env:
```
OKX_API_KEY=tu_api_key_aqui
OKX_API_SECRET=tu_secret_key_aqui
OKX_PASSPHRASE=tu_passphrase_aqui
```

## Ejecución

### Bot Principal (Recomendado)
```bash
python main.py
```
- Interfaz web en http://localhost:5000
- Modo paper trading automático
- Dashboard de monitoreo

### Bot de Consola
```bash
python trading_bot.py
```
- Ejecución en terminal
- Logs detallados en tiempo real

### Bot Simplificado
```bash
python start_bot.py
```
- Versión básica para pruebas

## Monitoreo

### Ver Estadísticas
```bash
python log_viewer.py
```

### Monitoreo en Tiempo Real
```bash
python verify_price.py --monitor
```

## Archivos de Datos

- `learning_data.json` - Datos de aprendizaje del bot
- `trading_bot.log` - Logs de operaciones
- `market_data.db` - Base de datos de mercado (si existe)

## Modo Paper Trading

El bot opera por defecto en modo paper trading (sin dinero real):
- Balance inicial virtual: $10,000 USDT
- Todas las operaciones son simuladas
- Sin riesgo financiero real
- Aprendizaje automático activado

## Estructura del Proyecto

```
CryptoTradingBot/
├── main.py                 # Bot principal con interfaz web
├── trading_bot.py          # Bot de consola
├── start_bot.py           # Bot simplificado
├── config.env             # Configuración de credenciales
├── learning_data.json     # Datos de aprendizaje
├── api_client/            # Módulos de conexión OKX
│   ├── modulo2.py         # Cliente WebSocket
│   └── modulocola.py      # Cola de datos
├── data_management/       # Gestión de datos
└── logs/                  # Archivos de log
```

## Solución de Problemas

1. Error de conexión OKX:
   - Verificar credenciales en config.env
   - Comprobar IP whitelisted en OKX
   - Verificar permisos de API (lectura + trading)

2. Bot no ejecuta operaciones:
   - El bot necesita 5-10 minutos para acumular datos
   - Verificar que learning_data.json se está creando
   - Comprobar logs con log_viewer.py

3. Error de dependencias:
   - Reinstalar con: pip install -r requirements.txt
   - Usar entorno virtual si es necesario
"""
    
    with open('LOCAL_SETUP_INSTRUCTIONS.md', 'w') as f:
        f.write(instructions)
    
    print()
    print("6. ARCHIVO CREADO:")
    print("   ✓ LOCAL_SETUP_INSTRUCTIONS.md - Instrucciones completas")
    
    return True

if __name__ == "__main__":
    create_local_setup()