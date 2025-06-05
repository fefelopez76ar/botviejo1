#!/usr/bin/env python3
"""
Script de inicio para el bot de trading de Solana

Este script permite iniciar el bot de trading desde la línea de comandos de
manera sencilla, sin necesidad de conocer los detalles internos del sistema.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"bot_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger("SolanaTraderBotStarter")

def main():
    """Función principal para iniciar el bot"""
    
    # Mostrar arte ASCII
    print("""
  _____       _                     _____             _            ____        _   
 / ____|     | |                   |_   _|           | |          |  _ \      | |  
| (___   ___ | | __ _ _ __   __ _    | |  _ __   __ _| |__   ___  | |_) | ___ | |_ 
 \___ \ / _ \| |/ _` | '_ \ / _` |   | | | '_ \ / _` | '_ \ / _ \ |  _ < / _ \| __|
 ____) | (_) | | (_| | | | | (_| |  _| |_| | | | (_| | | | |  __/ | |_) | (_) | |_ 
|_____/ \___/|_|\__,_|_| |_|\__,_| |_____|_| |_|\__,_|_| |_|\___| |____/ \___/ \__|
                                                                                   
    """)
    
    print("Iniciando bot de trading para Solana...")
    print("Versión: 1.0.0")
    print("Fecha: 20 de Mayo del 2025")
    print("\nConfigurado para operar principalmente en:")
    print("  - SOL-USDT en timeframes de 1m, 3m, 5m (scalping)")
    print("  - Modo paper trading habilitado por defecto para seguridad")
    print("  - Adaptación de estrategias según condiciones de mercado")
    
    try:
        # Intentar importar el módulo del menú CLI
        from bot_cli import cli_main_loop
        
        print("\nIniciando interfaz de línea de comandos...")
        time.sleep(1)
        
        # Iniciar el bucle principal del CLI
        try:
            cli_main_loop()
        except KeyboardInterrupt:
            print("\n\nBot detenido por el usuario. ¡Hasta pronto!")
        except Exception as e:
            logger.error(f"Error en bucle principal del CLI: {e}")
            print(f"\nError: {e}")
            print("El CLI ha terminado debido a un error.")
        
    except ImportError as e:
        logger.error(f"Error importando módulos necesarios: {e}")
        print(f"\n❌ Error importando módulos necesarios: {e}")
        print("\nVerifique que todos los módulos estén instalados correctamente.")
        print("Ejecute el siguiente comando para instalar dependencias:")
        print("\npip install -r requirements.txt")
        
    except Exception as e:
        logger.error(f"Error crítico: {e}")
        print(f"\n❌ Error crítico: {e}")
        print("El bot no pudo iniciarse.")
        
    print("\n¡Gracias por usar Solana Trading Bot!")

if __name__ == "__main__":
    try:
        # Iniciar el bot
        main()
    except KeyboardInterrupt:
        print("\n\nBot detenido por el usuario. ¡Hasta pronto!")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        print(f"\nError inesperado: {e}")
        print("El bot ha finalizado con errores.")