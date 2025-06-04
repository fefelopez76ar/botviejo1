#!/usr/bin/env python3
"""
Sistema de reporte de errores bidireccional para el bot de trading.

Este módulo proporciona una interfaz simplificada para reportar errores,
verificar y aplicar soluciones desde un servidor central. Permite la
comunicación bidireccional para mantener el bot actualizado y solucionar
problemas comunes automáticamente.
"""

import os
import sys
import json
import traceback
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Importar el gestor de sincronización
from core.sync_manager import SyncManager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ErrorReporter')

class ErrorReporter:
    """
    Clase para manejar el reporte de errores bidireccional.
    
    Esta clase proporciona métodos para registrar, reportar y resolver
    errores en el bot de trading, facilitando la comunicación con un
    servidor central para recibir actualizaciones y soluciones.
    """
    
    def __init__(self, 
               base_dir: str = ".",
               remote_url: str = None,
               auto_report: bool = False):
        """
        Inicializa el sistema de reporte de errores.
        
        Args:
            base_dir: Directorio base del bot
            remote_url: URL del servidor central (opcional)
            auto_report: Si se deben reportar errores automáticamente
        """
        self.base_dir = os.path.abspath(base_dir)
        self.remote_url = remote_url
        self.auto_report = auto_report
        
        # Crear instancia del gestor de sincronización
        self.sync_manager = SyncManager(base_dir=base_dir)
        
        # Cargar configuración si existe
        self.config = self._load_config()
        
        # Configurar URL remota desde configuración si no se proporciona
        if not self.remote_url and "remote_url" in self.config:
            self.remote_url = self.config["remote_url"]
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Carga la configuración del sistema de reporte de errores.
        
        Returns:
            Dict[str, Any]: Configuración del sistema
        """
        config_path = os.path.join(self.base_dir, "config", "error_reporter_config.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error al cargar configuración: {e}")
        
        # Configuración por defecto
        default_config = {
            "remote_url": None,
            "auto_report": False,
            "auto_check_fixes": True,
            "max_errors_per_session": 50,
            "report_system_info": True
        }
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Guardar configuración por defecto
        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
        except Exception as e:
            logger.error(f"Error al guardar configuración por defecto: {e}")
        
        return default_config
    
    def capture_exception(self, 
                        module_name: str, 
                        additional_info: Dict[str, Any] = None,
                        auto_report: bool = None) -> Dict[str, Any]:
        """
        Captura la excepción actual y la registra.
        
        Esta función debe llamarse dentro de un bloque except.
        
        Args:
            module_name: Nombre del módulo donde ocurrió el error
            additional_info: Información adicional sobre el error
            auto_report: Si se debe reportar automáticamente (anula la configuración global)
            
        Returns:
            Dict[str, Any]: Información del error registrado
        """
        exc_type, exc_value, exc_tb = sys.exc_info()
        
        if exc_type is None:
            logger.warning("capture_exception llamado sin excepción activa")
            return {"status": "error", "message": "No hay excepción activa"}
        
        error_type = exc_type.__name__
        error_message = str(exc_value)
        traceback_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        
        # Registrar error
        error_log = self.sync_manager.log_error(
            error_type=error_type,
            error_message=error_message,
            module_name=module_name,
            traceback_info=traceback_str,
            additional_data=additional_info
        )
        
        # Determinar si se debe reportar automáticamente
        should_auto_report = self.auto_report if auto_report is None else auto_report
        
        # Reportar automáticamente si está configurado
        if should_auto_report and self.remote_url:
            try:
                self.report_error(error_log["error_id"])
            except Exception as report_error:
                logger.error(f"Error al reportar automáticamente: {report_error}")
        
        return error_log
    
    def log_error(self, 
                error_type: str, 
                error_message: str, 
                module_name: str, 
                traceback_info: str = None,
                additional_data: Dict[str, Any] = None,
                auto_report: bool = None) -> Dict[str, Any]:
        """
        Registra un error manualmente.
        
        Args:
            error_type: Tipo de error
            error_message: Mensaje de error
            module_name: Nombre del módulo donde ocurrió el error
            traceback_info: Información de traceback (opcional)
            additional_data: Datos adicionales sobre el error
            auto_report: Si se debe reportar automáticamente
            
        Returns:
            Dict[str, Any]: Información del error registrado
        """
        # Registrar error
        error_log = self.sync_manager.log_error(
            error_type=error_type,
            error_message=error_message,
            module_name=module_name,
            traceback_info=traceback_info,
            additional_data=additional_data
        )
        
        # Determinar si se debe reportar automáticamente
        should_auto_report = self.auto_report if auto_report is None else auto_report
        
        # Reportar automáticamente si está configurado
        if should_auto_report and self.remote_url:
            try:
                self.report_error(error_log["error_id"])
            except Exception as report_error:
                logger.error(f"Error al reportar automáticamente: {report_error}")
        
        return error_log
    
    def report_error(self, 
                   error_id: str, 
                   remote_url: str = None,
                   additional_comments: str = None) -> Dict[str, Any]:
        """
        Reporta un error registrado al servidor central.
        
        Args:
            error_id: ID del error a reportar
            remote_url: URL del servidor central (anula la configuración global)
            additional_comments: Comentarios adicionales
            
        Returns:
            Dict[str, Any]: Resultado del reporte
        """
        url = remote_url or self.remote_url
        
        if not url:
            logger.error("URL remota no configurada")
            return {
                "status": "error",
                "message": "URL remota no configurada"
            }
        
        return self.sync_manager.report_error(
            error_id=error_id,
            remote_url=url,
            additional_comments=additional_comments
        )
    
    def check_for_fixes(self, remote_url: str = None) -> Dict[str, Any]:
        """
        Verifica si hay soluciones disponibles para errores reportados.
        
        Args:
            remote_url: URL del servidor central (anula la configuración global)
            
        Returns:
            Dict[str, Any]: Información sobre soluciones disponibles
        """
        url = remote_url or self.remote_url
        
        if not url:
            logger.error("URL remota no configurada")
            return {
                "status": "error",
                "message": "URL remota no configurada"
            }
        
        return self.sync_manager.check_error_fixes(url)
    
    def apply_fix(self, 
                error_id: str, 
                remote_url: str = None) -> Dict[str, Any]:
        """
        Aplica una solución para un error específico.
        
        Args:
            error_id: ID del error a solucionar
            remote_url: URL del servidor central (anula la configuración global)
            
        Returns:
            Dict[str, Any]: Resultado de la aplicación de la solución
        """
        url = remote_url or self.remote_url
        
        if not url:
            logger.error("URL remota no configurada")
            return {
                "status": "error",
                "message": "URL remota no configurada"
            }
        
        return self.sync_manager.apply_error_fix(error_id, url)
    
    def check_and_apply_all_fixes(self, remote_url: str = None) -> Dict[str, Any]:
        """
        Verifica y aplica todas las soluciones disponibles.
        
        Args:
            remote_url: URL del servidor central (anula la configuración global)
            
        Returns:
            Dict[str, Any]: Resultado de la aplicación de soluciones
        """
        url = remote_url or self.remote_url
        
        if not url:
            logger.error("URL remota no configurada")
            return {
                "status": "error",
                "message": "URL remota no configurada"
            }
        
        # Verificar soluciones disponibles
        check_result = self.sync_manager.check_error_fixes(url)
        
        if check_result.get("status") != "success":
            return check_result
        
        fixes = check_result.get("fixes", [])
        
        if not fixes:
            return {
                "status": "success",
                "message": "No hay soluciones disponibles",
                "applied_fixes": 0
            }
        
        # Aplicar cada solución
        applied_fixes = []
        failed_fixes = []
        
        for fix in fixes:
            error_id = fix.get("error_id")
            if error_id:
                result = self.sync_manager.apply_error_fix(error_id, url)
                
                if result.get("status") == "success":
                    applied_fixes.append({
                        "error_id": error_id,
                        "fix_type": fix.get("fix_type")
                    })
                else:
                    failed_fixes.append({
                        "error_id": error_id,
                        "fix_type": fix.get("fix_type"),
                        "reason": result.get("message")
                    })
        
        return {
            "status": "success",
            "total_fixes": len(fixes),
            "applied_fixes": len(applied_fixes),
            "failed_fixes": len(failed_fixes),
            "applied": applied_fixes,
            "failed": failed_fixes
        }
    
    def get_recent_errors(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Obtiene los errores más recientes.
        
        Args:
            limit: Límite de errores a retornar
            
        Returns:
            List[Dict[str, Any]]: Lista de errores recientes
        """
        return self.sync_manager.manifest.get("recent_errors", [])[:limit]
    
    def get_error_details(self, error_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene detalles de un error específico.
        
        Args:
            error_id: ID del error
            
        Returns:
            Optional[Dict[str, Any]]: Detalles del error o None si no existe
        """
        error_file = os.path.join(
            self.sync_manager.sync_dir, 
            "error_logs", 
            f"error_{error_id}.json"
        )
        
        if not os.path.exists(error_file):
            return None
        
        try:
            with open(error_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error al cargar detalles del error {error_id}: {e}")
            return None

# Instancia global para uso en cualquier parte del código
_error_reporter = None

def get_error_reporter(base_dir: str = ".", remote_url: str = None) -> ErrorReporter:
    """
    Obtiene la instancia global del reportador de errores.
    
    Args:
        base_dir: Directorio base del bot
        remote_url: URL del servidor central
        
    Returns:
        ErrorReporter: Instancia del reportador de errores
    """
    global _error_reporter
    
    if _error_reporter is None:
        _error_reporter = ErrorReporter(base_dir=base_dir, remote_url=remote_url)
    
    return _error_reporter

def report_exception(module_name: str, additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Función de conveniencia para reportar la excepción actual.
    Debe ser llamada dentro de un bloque except.
    
    Args:
        module_name: Nombre del módulo donde ocurrió el error
        additional_info: Información adicional sobre el error
        
    Returns:
        Dict[str, Any]: Información del error reportado
    """
    reporter = get_error_reporter()
    return reporter.capture_exception(module_name, additional_info)

def safe_execute(func, module_name: str, *args, **kwargs):
    """
    Ejecuta una función de forma segura, capturando y reportando cualquier excepción.
    
    Args:
        func: Función a ejecutar
        module_name: Nombre del módulo para el reporte de errores
        *args: Argumentos para la función
        **kwargs: Argumentos con nombre para la función
        
    Returns:
        El resultado de la función o None si ocurre un error
    """
    try:
        return func(*args, **kwargs)
    except Exception:
        report_exception(module_name, {
            "function": func.__name__,
            "args": str(args),
            "kwargs": str(kwargs)
        })
        return None

# Ejemplos de uso
if __name__ == "__main__":
    # Ejemplo de uso básico
    try:
        # Código que puede causar error
        result = 1 / 0
    except Exception:
        error = report_exception("ejemplos", {"operacion": "división"})
        print(f"Error capturado y reportado con ID: {error.get('error_id')}")