#!/usr/bin/env python3
"""
Módulo para exportar e importar el conocimiento aprendido por el bot.

Este módulo permite transferir fácilmente todo el aprendizaje y modelos
entrenados del bot entre diferentes instalaciones, facilitando su
portabilidad y respaldo.
"""

import os
import sys
import json
import logging
import shutil
import zipfile
import pickle
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BrainTransfer')

class BrainTransfer:
    """
    Sistema para exportar e importar el cerebro (conocimiento) del bot.
    
    Esta clase maneja la exportación e importación de todos los datos de 
    aprendizaje, incluyendo rendimiento de indicadores, modelos entrenados,
    y configuraciones optimizadas.
    """
    
    def __init__(self, 
                base_dir: str = '.',
                models_dir: str = 'models',
                data_dir: str = 'data',
                export_dir: str = 'brain_exports'):
        """
        Inicializa el sistema de transferencia del cerebro.
        
        Args:
            base_dir: Directorio base del proyecto
            models_dir: Directorio donde se almacenan los modelos
            data_dir: Directorio donde se almacenan los datos de rendimiento
            export_dir: Directorio donde se guardarán las exportaciones
        """
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / models_dir
        self.data_dir = self.base_dir / data_dir
        self.export_dir = self.base_dir / export_dir
        
        # Archivos clave a incluir siempre en la exportación
        self.key_files = [
            'indicator_performance.json',
            'trading_history.json',
            'market_conditions.json',
            'bot_config.json'
        ]
        
        # Asegurar que existen los directorios necesarios
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Crea los directorios necesarios si no existen."""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)
    
    def export_brain(self, 
                   name: str = None, 
                   include_models: bool = True,
                   include_data: bool = True) -> str:
        """
        Exporta todo el cerebro del bot a un archivo comprimido.
        
        Args:
            name: Nombre personalizado para la exportación (opcional)
            include_models: Si se deben incluir los modelos entrenados
            include_data: Si se deben incluir los datos de rendimiento
            
        Returns:
            str: Ruta al archivo de exportación creado
        """
        # Generar nombre de archivo si no se especifica
        if not name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"bot_brain_{timestamp}"
        
        # Asegurar que el nombre termina con .zip
        if not name.endswith('.zip'):
            name += '.zip'
            
        export_path = self.export_dir / name
        
        try:
            # Crear directorio temporal para organizar archivos
            temp_dir = self.export_dir / "temp_export"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Copiar archivos clave
            for file in self.key_files:
                source = self.base_dir / file
                if os.path.exists(source):
                    shutil.copy2(source, temp_dir)
            
            # Copiar modelos
            if include_models:
                models_temp = temp_dir / "models"
                if os.path.exists(self.models_dir):
                    shutil.copytree(self.models_dir, models_temp, dirs_exist_ok=True)
                    
            # Copiar datos
            if include_data:
                data_temp = temp_dir / "data"
                if os.path.exists(self.data_dir):
                    shutil.copytree(self.data_dir, data_temp, dirs_exist_ok=True)
            
            # Crear archivo de metadatos
            metadata = {
                "export_date": datetime.datetime.now().isoformat(),
                "bot_version": "1.0",  # Actualizar según versión del bot
                "includes_models": include_models,
                "includes_data": include_data,
                "files": []
            }
            
            # Añadir lista de archivos incluidos en la exportación
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, temp_dir)
                    file_size = os.path.getsize(filepath)
                    metadata["files"].append({
                        "path": rel_path,
                        "size": file_size
                    })
            
            # Guardar metadatos
            with open(temp_dir / "export_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # Crear archivo zip
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        filepath = os.path.join(root, file)
                        arcname = os.path.relpath(filepath, temp_dir)
                        zipf.write(filepath, arcname)
            
            # Limpiar directorio temporal
            shutil.rmtree(temp_dir)
            
            logger.info(f"Cerebro exportado exitosamente a: {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Error al exportar cerebro: {e}")
            # Limpiar si existe el directorio temporal
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise
    
    def import_brain(self, 
                   file_path: str,
                   override_existing: bool = False) -> Dict[str, Any]:
        """
        Importa el cerebro del bot desde un archivo exportado.
        
        Args:
            file_path: Ruta al archivo de exportación
            override_existing: Si se deben sobrescribir archivos existentes
            
        Returns:
            Dict[str, Any]: Información sobre la importación
        """
        try:
            # Verificar que el archivo existe
            if not os.path.exists(file_path):
                return {"success": False, "error": f"Archivo no encontrado: {file_path}"}
            
            # Crear directorio temporal para descomprimir
            temp_dir = self.export_dir / "temp_import"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Descomprimir archivo
            with zipfile.ZipFile(file_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Verificar metadatos
            metadata_path = temp_dir / "export_metadata.json"
            if not os.path.exists(metadata_path):
                shutil.rmtree(temp_dir)
                return {"success": False, "error": "Archivo de exportación inválido (sin metadatos)"}
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Copiar archivos clave al directorio base
            files_imported = []
            for file in self.key_files:
                source = temp_dir / file
                if os.path.exists(source):
                    target = self.base_dir / file
                    if override_existing or not os.path.exists(target):
                        shutil.copy2(source, target)
                        files_imported.append(file)
            
            # Importar modelos
            if metadata.get("includes_models", False):
                models_source = temp_dir / "models"
                if os.path.exists(models_source):
                    if override_existing:
                        # Eliminar carpeta de modelos existente
                        if os.path.exists(self.models_dir):
                            shutil.rmtree(self.models_dir)
                    # Copiar modelos
                    shutil.copytree(models_source, self.models_dir, dirs_exist_ok=True)
                    files_imported.append("models/*")
            
            # Importar datos
            if metadata.get("includes_data", False):
                data_source = temp_dir / "data"
                if os.path.exists(data_source):
                    if override_existing:
                        # Eliminar carpeta de datos existente
                        if os.path.exists(self.data_dir):
                            shutil.rmtree(self.data_dir)
                    # Copiar datos
                    shutil.copytree(data_source, self.data_dir, dirs_exist_ok=True)
                    files_imported.append("data/*")
            
            # Limpiar directorio temporal
            shutil.rmtree(temp_dir)
            
            result = {
                "success": True,
                "import_date": datetime.datetime.now().isoformat(),
                "original_export_date": metadata.get("export_date", "desconocido"),
                "bot_version": metadata.get("bot_version", "desconocido"),
                "files_imported": files_imported,
                "imported_from": file_path
            }
            
            logger.info(f"Cerebro importado exitosamente desde: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error al importar cerebro: {e}")
            # Limpiar si existe el directorio temporal
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return {"success": False, "error": str(e)}
    
    def list_available_exports(self) -> List[Dict[str, Any]]:
        """
        Lista todas las exportaciones disponibles.
        
        Returns:
            List[Dict[str, Any]]: Lista de exportaciones con sus metadatos
        """
        exports = []
        
        try:
            for file in os.listdir(self.export_dir):
                if file.endswith('.zip'):
                    file_path = self.export_dir / file
                    file_size = os.path.getsize(file_path)
                    file_date = datetime.datetime.fromtimestamp(
                        os.path.getmtime(file_path)
                    ).isoformat()
                    
                    # Intentar extraer metadatos
                    metadata = {}
                    try:
                        with zipfile.ZipFile(file_path, 'r') as zipf:
                            if "export_metadata.json" in zipf.namelist():
                                with zipf.open("export_metadata.json") as f:
                                    metadata = json.load(f)
                    except:
                        pass
                    
                    exports.append({
                        "filename": file,
                        "path": str(file_path),
                        "size": file_size,
                        "date": file_date,
                        "metadata": metadata
                    })
            
            # Ordenar por fecha de modificación (más reciente primero)
            exports.sort(key=lambda x: x["date"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error al listar exportaciones: {e}")
        
        return exports
    
    def get_quick_backup_command(self) -> str:
        """
        Retorna un comando de una línea para hacer un respaldo rápido.
        
        Returns:
            str: Comando para respaldo rápido
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        return f"python -c \"from adaptive_system.brain_transfer import BrainTransfer; BrainTransfer().export_brain('bot_backup_{timestamp}.zip')\""

def create_backup(name: str = None) -> str:
    """
    Función de conveniencia para crear un respaldo rápido del cerebro del bot.
    
    Args:
        name: Nombre personalizado para el respaldo (opcional)
        
    Returns:
        str: Ruta al archivo de respaldo creado
    """
    transfer = BrainTransfer()
    return transfer.export_brain(name)

def restore_backup(file_path: str, override: bool = True) -> Dict[str, Any]:
    """
    Función de conveniencia para restaurar un respaldo del cerebro del bot.
    
    Args:
        file_path: Ruta al archivo de respaldo
        override: Si se deben sobrescribir archivos existentes
        
    Returns:
        Dict[str, Any]: Información sobre la restauración
    """
    transfer = BrainTransfer()
    return transfer.import_brain(file_path, override)

def list_backups() -> List[Dict[str, Any]]:
    """
    Función de conveniencia para listar todos los respaldos disponibles.
    
    Returns:
        List[Dict[str, Any]]: Lista de respaldos con sus metadatos
    """
    transfer = BrainTransfer()
    return transfer.list_available_exports()

def scheduled_backup() -> str:
    """
    Función para ser utilizada en tareas programadas de respaldo.
    Genera un nombre con la fecha actual y realiza el respaldo.
    
    Returns:
        str: Ruta al archivo de respaldo creado
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"scheduled_backup_{timestamp}.zip"
    return create_backup(name)

def demo_brain_transfer():
    """Demostración del sistema de transferencia de cerebro."""
    print("\n🧠 SISTEMA DE TRANSFERENCIA DE CEREBRO DEL BOT 🧠")
    print("Este sistema permite exportar e importar todo el conocimiento")
    print("aprendido por el bot, facilitando su respaldo y transferencia.")
    
    # Simular un respaldo
    print("\n1. Creando respaldo del cerebro...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"demo_backup_{timestamp}.zip"
    backup_path = "brain_exports/" + backup_name
    
    # Simular un listado de respaldos
    print("\n2. Respaldos disponibles:")
    backups = [
        {
            "filename": backup_name,
            "path": backup_path,
            "size": "2.3 MB",
            "date": datetime.datetime.now().isoformat(),
            "metadata": {
                "export_date": datetime.datetime.now().isoformat(),
                "bot_version": "1.0",
                "includes_models": True,
                "includes_data": True
            }
        },
        {
            "filename": "backup_anterior.zip",
            "path": "brain_exports/backup_anterior.zip",
            "size": "1.8 MB",
            "date": (datetime.datetime.now() - datetime.timedelta(days=5)).isoformat(),
            "metadata": {
                "export_date": (datetime.datetime.now() - datetime.timedelta(days=5)).isoformat(),
                "bot_version": "0.9",
                "includes_models": True,
                "includes_data": True
            }
        }
    ]
    
    for i, backup in enumerate(backups):
        print(f"   {i+1}) {backup['filename']}")
        print(f"      - Fecha: {backup['date'][:10]}")
        print(f"      - Tamaño: {backup['size']}")
        print(f"      - Versión del bot: {backup['metadata']['bot_version']}")
    
    # Simular una restauración
    print("\n3. Comando para restaurar un respaldo:")
    print(f"   restore_backup('{backup_path}')")
    
    # Mostrar comando para respaldo rápido
    print("\n4. Comando para respaldo rápido (puedes guardarlo como alias):")
    transfer = BrainTransfer()
    print(f"   {transfer.get_quick_backup_command()}")
    
    print("\n✅ Demo completada. El sistema está listo para usarse.")
    print("   Para crear un respaldo real, ejecuta: create_backup()")
    return True

if __name__ == "__main__":
    try:
        # Si se ejecuta directamente, realizar un respaldo
        if len(sys.argv) > 1:
            # Si se proporciona un argumento, usarlo como nombre de archivo
            backup_path = create_backup(sys.argv[1])
        else:
            # Nombre por defecto con timestamp
            backup_path = create_backup()
        
        print(f"Respaldo creado exitosamente: {backup_path}")
    except Exception as e:
        print(f"Error al crear respaldo: {e}")
        sys.exit(1)