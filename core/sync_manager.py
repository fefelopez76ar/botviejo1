#!/usr/bin/env python3
"""
Gestor de Sincronización para Bot de Trading.

Este módulo permite:
- Exportar configuraciones, código y mejoras
- Sincronizar entre diferentes instalaciones
- Actualizar solo las partes modificadas
- Importar/exportar cerebros de IA y configuraciones

Es ideal para actualizar un bot local desde una versión en desarrollo
sin tener que reescribir todo el código.
"""

import os
import sys
import json
import time
import shutil
import hashlib
import logging
import zipfile
import base64
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SyncManager')

class SyncManager:
    """
    Gestor de sincronización entre instalaciones del bot.
    
    Permite exportar/importar código, configuraciones y mejoras
    entre diferentes instalaciones del bot.
    """
    
    def __init__(self, 
               base_dir: str = ".", 
               manifest_file: str = "sync_manifest.json",
               sync_dir: str = "sync_data"):
        """
        Inicializa el gestor de sincronización.
        
        Args:
            base_dir: Directorio base del bot
            manifest_file: Archivo de manifiesto para sincronización
            sync_dir: Directorio para archivos de sincronización
        """
        self.base_dir = os.path.abspath(base_dir)
        self.manifest_file = os.path.join(self.base_dir, manifest_file)
        self.sync_dir = os.path.join(self.base_dir, sync_dir)
        
        # Asegurar que exista el directorio de sincronización
        os.makedirs(self.sync_dir, exist_ok=True)
        
        # Cargar o crear manifiesto
        self.manifest = self._load_or_create_manifest()
        
        # Lista de directorios y archivos a ignorar en la sincronización
        self.ignore_dirs = [
            '.git', '__pycache__', 'venv', 'env', '.env', 
            'node_modules', '.vscode', '.idea', self.sync_dir
        ]
        
        self.ignore_files = [
            '.gitignore', '.DS_Store', 'Thumbs.db', '.env', 
            'config.local.json', 'local_settings.py', 'secrets.json',
            'api_keys.json', 'credentials.json', self.manifest_file
        ]
        
        # Archivos de configuración que deben fusionarse, no sobreescribirse
        self.merge_configs = [
            'config.json', 'settings.json', 'bot_config.json'
        ]
    
    def _load_or_create_manifest(self) -> Dict[str, Any]:
        """
        Carga el manifiesto existente o crea uno nuevo.
        
        Returns:
            Dict[str, Any]: Manifiesto de sincronización
        """
        if os.path.exists(self.manifest_file):
            try:
                with open(self.manifest_file, 'r') as f:
                    manifest = json.load(f)
                
                # Validar estructura básica
                required_keys = ["version", "last_sync", "files", "directories"]
                if not all(key in manifest for key in required_keys):
                    logger.warning("Manifiesto incompleto, creando uno nuevo")
                    return self._create_new_manifest()
                
                return manifest
            except Exception as e:
                logger.error(f"Error al cargar manifiesto: {e}")
                return self._create_new_manifest()
        else:
            return self._create_new_manifest()
    
    def _create_new_manifest(self) -> Dict[str, Any]:
        """
        Crea un nuevo manifiesto de sincronización.
        
        Returns:
            Dict[str, Any]: Nuevo manifiesto
        """
        manifest = {
            "version": "1.0.0",
            "created": datetime.now().isoformat(),
            "last_sync": None,
            "bot_name": "SolanaTradingBot",
            "files": {},
            "directories": [],
            "sync_history": []
        }
        
        # Guardar nuevo manifiesto
        with open(self.manifest_file, 'w') as f:
            json.dump(manifest, f, indent=4)
        
        return manifest
    
    def _save_manifest(self):
        """Guarda el manifiesto actual en disco."""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=4)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calcula el hash SHA256 de un archivo.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            str: Hash SHA256 hexadecimal
        """
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _should_ignore_path(self, path: str) -> bool:
        """
        Determina si una ruta debe ignorarse para sincronización.
        
        Args:
            path: Ruta a verificar
            
        Returns:
            bool: True si debe ignorarse, False en caso contrario
        """
        # Normalizar ruta
        norm_path = os.path.normpath(path)
        
        # Verificar si es un directorio ignorado
        for ignore_dir in self.ignore_dirs:
            if f"/{ignore_dir}/" in f"/{norm_path}/" or norm_path == ignore_dir:
                return True
        
        # Verificar si es un archivo ignorado
        if os.path.isfile(path):
            filename = os.path.basename(path)
            if filename in self.ignore_files:
                return True
        
        return False
    
    def scan_project(self) -> Dict[str, Any]:
        """
        Escanea el proyecto para obtener información de archivos y directorios.
        
        Returns:
            Dict[str, Any]: Información del escaneo
        """
        start_time = time.time()
        file_count = 0
        dir_count = 0
        total_size = 0
        files_info = {}
        directories = []
        
        # Recorrer directorios y archivos
        for root, dirs, files in os.walk(self.base_dir):
            # Normalizar ruta relativa al directorio base
            rel_root = os.path.relpath(root, self.base_dir)
            if rel_root == ".":
                rel_root = ""
            
            # Ignorar directorios en la lista de ignorados
            dirs[:] = [d for d in dirs if not self._should_ignore_path(os.path.join(rel_root, d))]
            
            # Procesar directorio
            if rel_root and not self._should_ignore_path(rel_root):
                directories.append(rel_root)
                dir_count += 1
            
            # Procesar archivos
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.join(rel_root, file) if rel_root else file
                
                # Ignorar archivos en la lista de ignorados
                if self._should_ignore_path(rel_path):
                    continue
                
                # Obtener información del archivo
                try:
                    file_stat = os.stat(file_path)
                    file_size = file_stat.st_size
                    file_hash = self._calculate_file_hash(file_path)
                    
                    # Guardar información del archivo
                    files_info[rel_path] = {
                        "size": file_size,
                        "hash": file_hash,
                        "last_modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    }
                    
                    file_count += 1
                    total_size += file_size
                except Exception as e:
                    logger.error(f"Error al procesar archivo {file_path}: {e}")
        
        # Actualizar manifiesto
        self.manifest["files"] = files_info
        self.manifest["directories"] = directories
        self.manifest["last_scan"] = datetime.now().isoformat()
        self.manifest["total_files"] = file_count
        self.manifest["total_dirs"] = dir_count
        self.manifest["total_size"] = total_size
        
        # Guardar manifiesto actualizado
        self._save_manifest()
        
        scan_time = time.time() - start_time
        logger.info(f"Escaneo completado: {file_count} archivos, {dir_count} directorios, {total_size/1024:.2f} KB en {scan_time:.2f}s")
        
        return {
            "file_count": file_count,
            "dir_count": dir_count,
            "total_size": total_size,
            "scan_time": scan_time
        }
    
    def create_sync_package(self, output_file: str = None) -> str:
        """
        Crea un paquete de sincronización con los archivos del proyecto.
        
        Args:
            output_file: Ruta donde guardar el paquete (opcional)
            
        Returns:
            str: Ruta al paquete de sincronización creado
        """
        # Actualizar escaneo del proyecto
        self.scan_project()
        
        # Generar nombre de archivo si no se proporciona
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.sync_dir, f"sync_package_{timestamp}.zip")
        
        # Crear paquete ZIP
        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Añadir manifiesto
            zipf.write(self.manifest_file, os.path.basename(self.manifest_file))
            
            # Añadir directorios (para preservar estructura)
            for dir_path in self.manifest["directories"]:
                full_path = os.path.join(self.base_dir, dir_path)
                if os.path.exists(full_path) and os.path.isdir(full_path):
                    # Añadir entrada de directorio vacío al ZIP
                    zipf.writestr(f"{dir_path}/", "")
            
            # Añadir archivos
            for rel_path, file_info in self.manifest["files"].items():
                full_path = os.path.join(self.base_dir, rel_path)
                if os.path.exists(full_path) and os.path.isfile(full_path):
                    zipf.write(full_path, rel_path)
        
        # Actualizar historial de sincronización
        sync_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "export",
            "package": os.path.basename(output_file),
            "file_count": self.manifest["total_files"],
            "size": os.path.getsize(output_file)
        }
        
        self.manifest["sync_history"].append(sync_entry)
        self.manifest["last_sync"] = sync_entry["timestamp"]
        self._save_manifest()
        
        logger.info(f"Paquete de sincronización creado: {output_file} ({os.path.getsize(output_file)/1024:.2f} KB)")
        return output_file
    
    def apply_sync_package(self, package_path: str) -> Dict[str, Any]:
        """
        Aplica un paquete de sincronización al proyecto actual.
        
        Args:
            package_path: Ruta al paquete de sincronización
            
        Returns:
            Dict[str, Any]: Resultado de la aplicación
        """
        if not os.path.exists(package_path):
            raise FileNotFoundError(f"Paquete de sincronización no encontrado: {package_path}")
        
        # Directorio temporal para extracción
        temp_dir = os.path.join(self.sync_dir, "temp_extract")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Extraer paquete
            with zipfile.ZipFile(package_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Cargar manifiesto del paquete
            package_manifest_path = os.path.join(temp_dir, os.path.basename(self.manifest_file))
            if not os.path.exists(package_manifest_path):
                raise ValueError("El paquete no contiene un manifiesto válido")
            
            with open(package_manifest_path, 'r') as f:
                package_manifest = json.load(f)
            
            # Validar manifiesto
            required_keys = ["version", "files", "directories"]
            if not all(key in package_manifest for key in required_keys):
                raise ValueError("El manifiesto del paquete es incompleto")
            
            # Resultados de la sincronización
            results = {
                "created_dirs": 0,
                "created_files": 0,
                "updated_files": 0,
                "unchanged_files": 0,
                "merged_configs": 0,
                "errors": 0
            }
            
            # Crear directorios primero
            for dir_path in package_manifest["directories"]:
                full_path = os.path.join(self.base_dir, dir_path)
                if not os.path.exists(full_path):
                    os.makedirs(full_path, exist_ok=True)
                    results["created_dirs"] += 1
            
            # Procesar archivos
            for rel_path, file_info in package_manifest["files"].items():
                try:
                    source_path = os.path.join(temp_dir, rel_path)
                    target_path = os.path.join(self.base_dir, rel_path)
                    
                    # Verificar si el archivo existe
                    file_exists = os.path.exists(target_path)
                    
                    # Determinar si es un archivo de configuración a fusionar
                    is_config_to_merge = any(rel_path.endswith(config) for config in self.merge_configs)
                    
                    if is_config_to_merge and file_exists:
                        # Fusionar archivos de configuración en lugar de sobrescribir
                        self._merge_config_files(source_path, target_path)
                        results["merged_configs"] += 1
                        continue
                    
                    # Verificar si el archivo ha cambiado (si existe)
                    if file_exists:
                        current_hash = self._calculate_file_hash(target_path)
                        if current_hash == file_info["hash"]:
                            # El archivo no ha cambiado
                            results["unchanged_files"] += 1
                            continue
                        
                        # El archivo ha cambiado, actualizarlo
                        shutil.copy2(source_path, target_path)
                        results["updated_files"] += 1
                    else:
                        # El archivo no existe, crearlo
                        # Asegurar que exista el directorio padre
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        shutil.copy2(source_path, target_path)
                        results["created_files"] += 1
                
                except Exception as e:
                    logger.error(f"Error al procesar {rel_path}: {e}")
                    results["errors"] += 1
            
            # Actualizar manifiesto local con la información del paquete
            self.manifest["files"].update(package_manifest["files"])
            
            # Añadir directorios que no existan ya
            for dir_path in package_manifest["directories"]:
                if dir_path not in self.manifest["directories"]:
                    self.manifest["directories"].append(dir_path)
            
            # Actualizar historial de sincronización
            sync_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "import",
                "package": os.path.basename(package_path),
                "results": results
            }
            
            self.manifest["sync_history"].append(sync_entry)
            self.manifest["last_sync"] = sync_entry["timestamp"]
            self._save_manifest()
            
            logger.info(f"Paquete de sincronización aplicado: {results}")
            return results
            
        finally:
            # Limpiar directorio temporal
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _merge_config_files(self, source_path: str, target_path: str):
        """
        Fusiona dos archivos de configuración JSON.
        
        Args:
            source_path: Ruta al archivo fuente
            target_path: Ruta al archivo destino
        """
        # Cargar configuración existente
        with open(target_path, 'r') as f:
            target_config = json.load(f)
        
        # Cargar configuración nueva
        with open(source_path, 'r') as f:
            source_config = json.load(f)
        
        # Fusionar configuraciones (nivel superior)
        for key, value in source_config.items():
            # Preservar valores locales para ciertos campos
            if key in ["api_keys", "credentials", "secrets", "custom_settings"]:
                continue
            
            # Si la clave existe y ambos son diccionarios, fusionar recursivamente
            if key in target_config and isinstance(target_config[key], dict) and isinstance(value, dict):
                self._merge_dict_recursive(target_config[key], value)
            else:
                # De lo contrario, actualizar valor
                target_config[key] = value
        
        # Guardar configuración fusionada
        with open(target_path, 'w') as f:
            json.dump(target_config, f, indent=4)
    
    def _merge_dict_recursive(self, target: Dict, source: Dict):
        """
        Fusiona dos diccionarios de forma recursiva.
        
        Args:
            target: Diccionario destino
            source: Diccionario fuente
        """
        for key, value in source.items():
            # Si la clave existe y ambos son diccionarios, fusionar recursivamente
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_dict_recursive(target[key], value)
            else:
                # De lo contrario, actualizar valor
                target[key] = value
    
    def export_brain(self, output_file: str = None) -> str:
        """
        Exporta el cerebro de IA del bot.
        
        Args:
            output_file: Ruta donde guardar el cerebro (opcional)
            
        Returns:
            str: Ruta al archivo del cerebro exportado
        """
        # Importar función de respaldo de cerebro
        from adaptive_system.brain_transfer import create_backup
        
        # Generar nombre de archivo si no se proporciona
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.sync_dir, f"brain_backup_{timestamp}.zip")
        
        # Crear respaldo
        backup_path = create_backup(output_file)
        
        # Actualizar historial de sincronización
        sync_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "brain_export",
            "file": os.path.basename(backup_path),
            "size": os.path.getsize(backup_path)
        }
        
        self.manifest["sync_history"].append(sync_entry)
        self._save_manifest()
        
        logger.info(f"Cerebro exportado: {backup_path} ({os.path.getsize(backup_path)/1024:.2f} KB)")
        return backup_path
    
    def import_brain(self, brain_path: str) -> Dict[str, Any]:
        """
        Importa un cerebro de IA.
        
        Args:
            brain_path: Ruta al archivo del cerebro
            
        Returns:
            Dict[str, Any]: Resultado de la importación
        """
        # Importar función de restauración de cerebro
        from adaptive_system.brain_transfer import restore_backup
        
        # Verificar que el archivo existe
        if not os.path.exists(brain_path):
            raise FileNotFoundError(f"Archivo de cerebro no encontrado: {brain_path}")
        
        # Restaurar cerebro
        result = restore_backup(brain_path)
        
        # Actualizar historial de sincronización
        sync_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "brain_import",
            "file": os.path.basename(brain_path),
            "result": result
        }
        
        self.manifest["sync_history"].append(sync_entry)
        self._save_manifest()
        
        logger.info(f"Cerebro importado: {brain_path}, resultado: {result}")
        return result
    
    def get_sync_history(self) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de sincronización.
        
        Returns:
            List[Dict[str, Any]]: Historial de sincronización
        """
        return self.manifest.get("sync_history", [])
    
    def get_remote_update_status(self, remote_url: str) -> Dict[str, Any]:
        """
        Verifica si hay actualizaciones disponibles en un servidor remoto.
        
        Args:
            remote_url: URL del servidor remoto
            
        Returns:
            Dict[str, Any]: Estado de actualización
        """
        try:
            # Enviar request con versión actual
            response = requests.post(
                f"{remote_url}/api/check-updates",
                json={
                    "version": self.manifest.get("version", "1.0.0"),
                    "last_sync": self.manifest.get("last_sync"),
                    "bot_name": self.manifest.get("bot_name", "SolanaTradingBot")
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error al verificar actualizaciones: {response.status_code}")
                return {
                    "status": "error",
                    "message": f"Error {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Error al verificar actualizaciones: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def download_remote_update(self, remote_url: str, update_id: str) -> Dict[str, Any]:
        """
        Descarga una actualización desde un servidor remoto.
        
        Args:
            remote_url: URL del servidor remoto
            update_id: ID de la actualización
            
        Returns:
            Dict[str, Any]: Resultado de la descarga
        """
        try:
            # Enviar request para descargar actualización
            response = requests.get(
                f"{remote_url}/api/download-update/{update_id}",
                timeout=30
            )
            
            if response.status_code == 200:
                # Guardar actualización
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                update_file = os.path.join(self.sync_dir, f"remote_update_{timestamp}.zip")
                
                with open(update_file, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Actualización descargada: {update_file} ({len(response.content)/1024:.2f} KB)")
                
                # Aplicar actualización
                result = self.apply_sync_package(update_file)
                
                return {
                    "status": "success",
                    "file": update_file,
                    "result": result
                }
            else:
                logger.error(f"Error al descargar actualización: {response.status_code}")
                return {
                    "status": "error",
                    "message": f"Error {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Error al descargar actualización: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

def get_sync_manager(base_dir: str = ".") -> SyncManager:
    """
    Función de conveniencia para obtener una instancia del gestor de sincronización.
    
    Args:
        base_dir: Directorio base del bot
        
    Returns:
        SyncManager: Instancia del gestor de sincronización
    """
    return SyncManager(base_dir=base_dir)

def demo_sync_manager():
    """Demostración del gestor de sincronización."""
    print("\n🔄 DEMOSTRACIÓN DEL GESTOR DE SINCRONIZACIÓN 🔄")
    
    # Crear gestor de sincronización
    manager = SyncManager()
    
    # Escanear proyecto
    print("\n1. Escaneando proyecto...")
    scan_result = manager.scan_project()
    print(f"   Archivos: {scan_result['file_count']}")
    print(f"   Directorios: {scan_result['dir_count']}")
    print(f"   Tamaño total: {scan_result['total_size']/1024:.2f} KB")
    print(f"   Tiempo de escaneo: {scan_result['scan_time']:.2f}s")
    
    # Crear paquete de sincronización
    print("\n2. Creando paquete de sincronización...")
    package_path = manager.create_sync_package()
    print(f"   Paquete creado: {package_path}")
    print(f"   Tamaño: {os.path.getsize(package_path)/1024:.2f} KB")
    
    # Mostrar historial de sincronización
    print("\n3. Historial de sincronización:")
    history = manager.get_sync_history()
    for i, entry in enumerate(history):
        print(f"   {i+1}. {entry['type']} - {entry['timestamp']}")
    
    print("\n✅ Demostración completada. Puedes usar este paquete para sincronizar otra instalación.")
    return manager

if __name__ == "__main__":
    try:
        demo_sync_manager()
    except Exception as e:
        print(f"Error en la demostración: {e}")