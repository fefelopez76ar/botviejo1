#!/usr/bin/env python3
"""
Script para copiar la carpeta trading_modules_export a una nueva ubicación
"""
import os
import shutil
from datetime import datetime

def copy_trading_modules():
    """Copia la carpeta trading_modules_export a una nueva ubicación"""
    
    # Crear carpeta de exportación con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_folder = f"my_trading_modules_{timestamp}"
    
    print(f"Creando exportación en: {export_folder}")
    
    # Crear la carpeta principal
    os.makedirs(export_folder, exist_ok=True)
    
    # Lista de archivos críticos a copiar
    critical_files = [
        'main.py',
        'trading_bot.py',
        'start_bot.py',
        'config.env',
        'learning_data.json',
        'log_viewer.py',
        'stats_tracker.py',
        'verify_price.py',
        'LOCAL_SETUP_INSTRUCTIONS.md',
        'PROJECT_SUMMARY.md',
        'BOT_MANUAL.md',
        'BOT_STATUS.md'
    ]
    
    # Copiar archivos individuales
    for file in critical_files:
        if os.path.exists(file):
            shutil.copy2(file, export_folder)
            print(f"✓ Copiado: {file}")
        else:
            print(f"✗ No encontrado: {file}")
    
    # Copiar carpetas completas
    folders_to_copy = [
        'api_client',
        'data_management', 
        'info'
    ]
    
    for folder in folders_to_copy:
        if os.path.exists(folder):
            dest_folder = os.path.join(export_folder, folder)
            shutil.copytree(folder, dest_folder, dirs_exist_ok=True)
            print(f"✓ Carpeta copiada: {folder}")
        else:
            print(f"✗ Carpeta no encontrada: {folder}")
    
    # Crear requirements.txt
    requirements_content = """websockets>=11.0.3
ccxt>=4.2.25
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
requests>=2.31.0
flask>=2.3.0
tabulate>=0.9.0"""
    
    with open(os.path.join(export_folder, 'requirements.txt'), 'w') as f:
        f.write(requirements_content)
    
    print(f"✓ Creado: requirements.txt")
    
    # Crear archivo README para GitHub
    readme_content = """# SolanaScalper Trading Bot

Bot de trading automatizado para Solana con aprendizaje automático.

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración

1. Renombrar `config.example.env` a `config.env`
2. Agregar credenciales OKX:
```
OKX_API_KEY=tu_api_key
OKX_API_SECRET=tu_secret
OKX_PASSPHRASE=tu_passphrase
```

## Uso

```bash
python main.py        # Interfaz web
python trading_bot.py # Modo consola
```

## Documentación

Ver archivos en carpeta `info/` para documentación completa.
"""
    
    with open(os.path.join(export_folder, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print(f"✓ Creado: README.md")
    
    # Crear config.example.env (sin credenciales reales)
    config_example = """# Configuración OKX - Reemplazar con credenciales reales
OKX_API_KEY=tu_api_key_aqui
OKX_API_SECRET=tu_secret_key_aqui
OKX_PASSPHRASE=tu_passphrase_aqui

# Parámetros de Trading
SYMBOL=SOL/USDT
TIMEFRAME=1m
POSITION_SIZE=0.1
PAPER_TRADING=True
"""
    
    with open(os.path.join(export_folder, 'config.example.env'), 'w') as f:
        f.write(config_example)
    
    print(f"✓ Creado: config.example.env")
    
    # Crear .gitignore
    gitignore_content = """# Credenciales
config.env
*.log

# Python
__pycache__/
*.pyc
*.pyo
.env

# Base de datos
*.db
*.sqlite

# IDE
.vscode/
.idea/
"""
    
    with open(os.path.join(export_folder, '.gitignore'), 'w') as f:
        f.write(gitignore_content)
    
    print(f"✓ Creado: .gitignore")
    
    print(f"\n🎉 Exportación completa en: {export_folder}")
    print(f"\nArchivos listos para:")
    print(f"1. Copiar a tu PC")
    print(f"2. Abrir en Visual Studio Code") 
    print(f"3. Subir a GitHub")
    
    return export_folder

if __name__ == "__main__":
    copy_trading_modules()