
#!/usr/bin/env python3
"""
Script para copiar la carpeta trading_modules_export a una nueva ubicación
"""

import os
import shutil
import sys
from datetime import datetime

def copy_trading_modules():
    """Copia la carpeta trading_modules_export a una nueva ubicación"""
    
    source_dir = "trading_modules_export"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = f"my_trading_modules_{timestamp}"
    
    print(f"🔄 Copiando módulos de trading...")
    print(f"   Origen: {source_dir}")
    print(f"   Destino: {dest_dir}")
    
    try:
        # Verificar que existe la carpeta origen
        if not os.path.exists(source_dir):
            print(f"❌ Error: No existe la carpeta {source_dir}")
            return False
        
        # Copiar toda la carpeta
        shutil.copytree(source_dir, dest_dir)
        
        print(f"✅ Copia completada exitosamente!")
        print(f"📁 Tu nueva carpeta: {dest_dir}")
        
        # Mostrar contenido
        print(f"\n📋 Contenido copiado:")
        for root, dirs, files in os.walk(dest_dir):
            level = root.replace(dest_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:3]:  # Mostrar solo los primeros 3 archivos
                print(f"{subindent}{file}")
            if len(files) > 3:
                print(f"{subindent}... y {len(files)-3} archivos más")
        
        print(f"\n🎯 Ahora puedes:")
        print(f"   1. Usar la carpeta {dest_dir}")
        print(f"   2. Modificar los archivos según necesites")
        print(f"   3. Adaptar para tu bot de acciones argentinas")
        
        return True
        
    except Exception as e:
        print(f"❌ Error al copiar: {e}")
        return False

if __name__ == "__main__":
    copy_trading_modules()
