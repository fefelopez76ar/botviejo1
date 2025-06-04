import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
config_path = Path(__file__).parent / "config.env"
load_dotenv(dotenv_path=config_path)

print("=== VARIABLES DE ENTORNO CARGADAS ===")
print(f"OKX_API_KEY: {os.getenv('OKX_API_KEY')}")
print(f"OKX_API_SECRET: {os.getenv('OKX_API_SECRET')[:10]}..." if os.getenv('OKX_API_SECRET') else "OKX_API_SECRET: None")
print(f"OKX_PASSPHRASE: {os.getenv('OKX_PASSPHRASE')}")
print(f"USE_DEMO_MODE: {os.getenv('USE_DEMO_MODE')}")
print(f"DEFAULT_SYMBOL: {os.getenv('DEFAULT_SYMBOL')}")
print("=========================================")