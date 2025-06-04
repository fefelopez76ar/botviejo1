#!/usr/bin/env python3
"""
Script para exportar la configuración y el estado del bot para uso en entorno local

Este script genera un archivo ZIP con todo lo necesario para ejecutar el bot
en una computadora local, incluyendo:
- Código fuente
- Estado actual del bot
- Configuración
- Datos guardados
- Instrucciones de instalación

IMPORTANTE: Este script está diseñado para facilitar la transición
entre el entorno Replit y una PC local con Python 3.11+ instalado.
"""

import os
import sys
import shutil
import tempfile
import zipfile
import json
import datetime
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ExportSetup")

# Directorio raíz del proyecto (donde se encuentra este script)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directorios y archivos a incluir/excluir
INCLUDE_DIRS = [
    "adaptive_system",
    "api_client",
    "backtesting",
    "core",
    "data",
    "data_management",
    "indicators",
    "models",
    "notifications",
    "risk_management",
    "static",
    "strategies",
    "templates",
    "utils"
]

INCLUDE_FILES = [
    "trading_bot.py",
    "main.py", 
    "app.py",
    "indicator_weighting.py",
    "README.md",
    "requirements.txt",
    "config.example.env",
    "bot_cli.py",
    "scalping_strategies.py"
]

EXCLUDE_PATTERNS = [
    "__pycache__", 
    ".git",
    ".replit",
    "replit.nix",
    ".upm",
    ".config",
    "venv",
    ".venv",
    "__init__.py",
    "*.pyc",
    "*.log",
    "*.zip"
]

def create_requirements_file() -> str:
    """
    Crea un archivo requirements.txt con las dependencias necesarias
    
    Returns:
        str: Ruta al archivo creado
    """
    requirements = [
        "flask==2.2.3",
        "flask-sqlalchemy==3.0.3",
        "pandas==2.0.3",
        "numpy==1.24.2",
        "matplotlib==3.7.1",
        "tabulate==0.9.0",
        "requests==2.28.2",
        "websocket-client==1.5.1",
        "ccxt==3.0.79",
        "python-dotenv==1.0.0",
        "scikit-learn==1.2.2",
        "statsmodels==0.13.5"
    ]
    
    # Verificar si telebot está instalado y añadirlo si es necesario
    try:
        import telebot
        requirements.append("pyTelegramBotAPI==4.12.0")
    except ImportError:
        logger.warning("telebot no instalado, no se añadirá a requirements.txt")
    
    requirements_path = os.path.join(ROOT_DIR, "requirements.txt")
    
    with open(requirements_path, 'w') as f:
        f.write('\n'.join(requirements))
    
    logger.info(f"Archivo requirements.txt creado con {len(requirements)} dependencias")
    return requirements_path

def create_readme() -> str:
    """
    Crea un archivo README con instrucciones de instalación y uso
    
    Returns:
        str: Ruta al archivo creado
    """
    readme_content = """# Bot de Trading para Solana

## Configuración para Entorno Local

Este paquete contiene todo lo necesario para ejecutar el bot de trading Solana
en tu computadora local. Sigue estas instrucciones para configurarlo correctamente.

### Requisitos Previos

- Python 3.11 o superior
- pip (gestor de paquetes de Python)
- Conexión a Internet

### Instalación

1. **Descomprime el archivo zip** en la ubicación deseada

2. **Crea un entorno virtual** (recomendado):
   ```
   python -m venv venv
   ```

3. **Activa el entorno virtual**:
   - Windows: `venv\\Scripts\\activate`
   - Linux/Mac: `source venv/bin/activate`

4. **Instala las dependencias**:
   ```
   pip install -r requirements.txt
   ```

5. **Configura las variables de entorno**:
   - Copia `config.example.env` a `config.env`
   - Edita `config.env` con tus credenciales de API (opcional)

### Ejecución

#### Interfaz Web

Para iniciar el servidor web y acceder a la interfaz de usuario:

```
python main.py
```

Luego abre tu navegador en `http://localhost:5000`

#### Interfaz de Línea de Comandos

Para usar la interfaz de línea de comandos:

```
python bot_cli.py
```

### Modo Paper Trading

**¡IMPORTANTE!** El bot está configurado para operar ÚNICAMENTE en modo paper trading (simulación),
para proteger tus fondos. La función de trading real está deshabilitada por seguridad.

### Archivos Principales

- `main.py`: Servidor web para interfaz gráfica
- `bot_cli.py`: Interfaz de línea de comandos
- `trading_bot.py`: Núcleo del bot de trading

### Respaldo de Datos

Se recomienda hacer copias de seguridad periódicas de la carpeta `data/` que
contiene el historial de operaciones y configuraciones del bot.

### Problemas Comunes

Si experimentas problemas de conexión con las APIs, verifica:
1. Tu conexión a Internet
2. Las credenciales API en config.env (si aplica)
3. Los logs en la carpeta data/logs/

### Actualizaciones

Para obtener las últimas actualizaciones y mejoras, visita el repositorio original del proyecto.

"""
    
    readme_path = os.path.join(ROOT_DIR, "README_LOCAL.md")
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info("Archivo README_LOCAL.md creado con instrucciones de instalación")
    return readme_path

def create_install_script() -> str:
    """
    Crea un script de instalación para Windows
    
    Returns:
        str: Ruta al archivo creado
    """
    windows_script = """@echo off
echo ===== Instalación del Bot de Trading Solana =====

rem Verificar si Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python no encontrado. Por favor instala Python 3.11 o superior.
    echo Puedes descargarlo desde https://www.python.org/downloads/
    pause
    exit /b
)

echo Python encontrado, continuando...

rem Crear entorno virtual
echo Creando entorno virtual...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: No se pudo crear el entorno virtual.
    pause
    exit /b
)

rem Activar entorno virtual
echo Activando entorno virtual...
call venv\\Scripts\\activate
if %errorlevel% neq 0 (
    echo ERROR: No se pudo activar el entorno virtual.
    pause
    exit /b
)

rem Instalar dependencias
echo Instalando dependencias...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: No se pudieron instalar las dependencias.
    pause
    exit /b
)

rem Configuración
echo Configurando el bot...
if not exist config.env (
    copy config.example.env config.env
)

echo ===== Instalación completada con éxito =====
echo.
echo Para iniciar el bot, usa uno de estos comandos:
echo.
echo Interfaz web: python main.py
echo Interfaz CLI: python bot_cli.py
echo.
pause
"""
    
    unix_script = """#!/bin/bash

echo "===== Instalación del Bot de Trading Solana ====="

# Verificar si Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python no encontrado. Por favor instala Python 3.11 o superior."
    echo "Puedes descargarlo desde https://www.python.org/downloads/"
    exit 1
fi

echo "Python encontrado, continuando..."

# Crear entorno virtual
echo "Creando entorno virtual..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: No se pudo crear el entorno virtual."
    exit 1
fi

# Activar entorno virtual
echo "Activando entorno virtual..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: No se pudo activar el entorno virtual."
    exit 1
fi

# Instalar dependencias
echo "Instalando dependencias..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: No se pudieron instalar las dependencias."
    exit 1
fi

# Configuración
echo "Configurando el bot..."
if [ ! -f config.env ]; then
    cp config.example.env config.env
fi

echo "===== Instalación completada con éxito ====="
echo ""
echo "Para iniciar el bot, usa uno de estos comandos:"
echo ""
echo "Interfaz web: python main.py"
echo "Interfaz CLI: python bot_cli.py"
echo ""
"""
    
    win_script_path = os.path.join(ROOT_DIR, "install_windows.bat")
    unix_script_path = os.path.join(ROOT_DIR, "install_unix.sh")
    
    with open(win_script_path, 'w') as f:
        f.write(windows_script)
    
    with open(unix_script_path, 'w') as f:
        f.write(unix_script)
    
    # Hacer el script de Unix ejecutable
    try:
        os.chmod(unix_script_path, 0o755)
    except:
        pass
    
    logger.info("Scripts de instalación creados")
    return win_script_path

def should_include_file(file_path: str) -> bool:
    """
    Determina si un archivo debe ser incluido en el paquete
    
    Args:
        file_path: Ruta del archivo
        
    Returns:
        bool: True si el archivo debe ser incluido
    """
    # Verificar patrones de exclusión
    for pattern in EXCLUDE_PATTERNS:
        if pattern in file_path or (pattern.startswith('*') and file_path.endswith(pattern[1:])):
            return False
    
    return True

def create_export_package() -> str:
    """
    Crea un paquete ZIP con todo lo necesario para la instalación local
    
    Returns:
        str: Ruta al archivo ZIP creado
    """
    # Crear carpeta temporal
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Creando paquete en directorio temporal: {temp_dir}")
        
        # Crear directorio de destino dentro de la carpeta temporal
        dest_dir = os.path.join(temp_dir, "solana_trading_bot")
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copiar directorios
        for dir_name in INCLUDE_DIRS:
            src_dir = os.path.join(ROOT_DIR, dir_name)
            if os.path.exists(src_dir):
                dest_subdir = os.path.join(dest_dir, dir_name)
                logger.info(f"Copiando directorio: {dir_name}")
                
                # Copiar directorio y su contenido
                shutil.copytree(src_dir, dest_subdir, dirs_exist_ok=True,
                               ignore=lambda _, files: [f for f in files if not should_include_file(f)])
        
        # Copiar archivos individuales
        for file_name in INCLUDE_FILES:
            src_file = os.path.join(ROOT_DIR, file_name)
            if os.path.exists(src_file):
                logger.info(f"Copiando archivo: {file_name}")
                shutil.copy2(src_file, os.path.join(dest_dir, file_name))
        
        # Crear nuevo archivo requirements.txt
        req_file = create_requirements_file()
        shutil.copy2(req_file, os.path.join(dest_dir, "requirements.txt"))
        
        # Crear README con instrucciones
        readme_file = create_readme()
        shutil.copy2(readme_file, os.path.join(dest_dir, "README.md"))
        
        # Crear scripts de instalación
        install_script = create_install_script()
        shutil.copy2(install_script, os.path.join(dest_dir, "install_windows.bat"))
        unix_script = install_script.replace("_windows.bat", "_unix.sh")
        shutil.copy2(unix_script, os.path.join(dest_dir, "install_unix.sh"))
        
        # Crear carpetas requeridas
        os.makedirs(os.path.join(dest_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "data", "logs"), exist_ok=True)
        
        # Crear archivo ZIP
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        zip_filename = f"solana_trading_bot_export_{timestamp}.zip"
        zip_path = os.path.join(ROOT_DIR, zip_filename)
        
        logger.info(f"Creando archivo ZIP: {zip_filename}")
        
        # Comprimir directorio
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(dest_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, rel_path)
        
        logger.info(f"Paquete creado con éxito: {zip_path}")
        return zip_path

def main():
    """Función principal"""
    try:
        print("=" * 60)
        print(f"EXPORTANDO BOT DE TRADING PARA USO LOCAL")
        print("=" * 60)
        print("\nEste proceso creará un paquete ZIP con todo lo necesario")
        print("para ejecutar el bot en tu computadora local.")
        print("\nEl paquete incluirá:")
        print("- Código fuente")
        print("- Configuración")
        print("- Scripts de instalación")
        print("- Instrucciones de uso")
        
        # Confirmar operación
        confirmation = input("\n¿Deseas continuar? (s/n): ").lower()
        if confirmation != 's' and confirmation != 'si' and confirmation != 'y' and confirmation != 'yes':
            print("Operación cancelada.")
            return
        
        # Crear paquete
        zip_path = create_export_package()
        
        print("\n" + "=" * 60)
        print(f"✅ EXPORTACIÓN COMPLETADA CON ÉXITO")
        print("=" * 60)
        print(f"\nPaquete creado: {os.path.basename(zip_path)}")
        print("\nPara usar en tu PC local:")
        print("1. Descarga el archivo ZIP")
        print("2. Descomprímelo en tu computadora")
        print("3. Sigue las instrucciones en el archivo README.md")
        
    except Exception as e:
        logger.error(f"Error durante la exportación: {e}")
        print(f"\n❌ ERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())