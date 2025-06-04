# Guía de Transferencia Manual - Bot Trading Solana
## Fecha: 4 de Junio 2025 - 04:50 UTC

### MÉTODO DE TRANSFERENCIA MANUAL

Como la descarga ZIP falla, vas a copiar los archivos manualmente:

#### PASO 1: Crear Estructura en tu PC
```
CryptoTradingBot/
├── main.py
├── trading_bot.py
├── start_bot.py
├── config.env
├── learning_data.json
├── requirements.txt
├── README.md
├── .gitignore
├── api_client/
│   ├── __init__.py
│   ├── modulo2.py
│   └── modulocola.py
├── info/
│   ├── estado_final_bot_20250604.md
│   ├── cambios_implementados_20250604.md
│   └── lista_archivos_criticos_20250604.md
└── log_viewer.py
```

#### PASO 2: Copiar Archivos desde Replit

**ARCHIVOS PRINCIPALES** (ver archivos individuales en mi_trading_modules_20250604_045025/):

1. **main.py** - Copia desde archivo en Replit
2. **trading_bot.py** - Copia desde archivo en Replit  
3. **start_bot.py** - Copia desde archivo en Replit
4. **config.env** - Copia desde archivo en Replit
5. **learning_data.json** - Copia desde archivo en Replit

**CARPETA api_client/**:
6. **api_client/modulo2.py** - Archivo más importante (13KB)
7. **api_client/modulocola.py** - Archivo pequeño

**ARCHIVOS DE CONFIGURACIÓN**:
8. **requirements.txt** - Lista de dependencias
9. **README.md** - Documentación GitHub
10. **.gitignore** - Configuración Git

#### PASO 3: Instalar en PC
```bash
cd CryptoTradingBot
pip install -r requirements.txt
python main.py
```

#### PASO 4: Subir a GitHub
```bash
git init
git add .
git commit -m "Bot trading Solana completo"
git remote add origin https://github.com/fedelofedelooo/CryptoTradingBot.git
git push -u origin main
```

### ARCHIVOS LISTOS PARA COPIAR

Ver carpeta: `my_trading_modules_20250604_045025/`

Todos los archivos están preparados y organizados para transferencia manual.

### VALIDACIÓN POST-TRANSFERENCIA

Una vez copiado en tu PC, ejecutar:
```bash
python verify_price.py  # Verificar conexión
python main.py          # Iniciar bot
```

El bot debería conectar a OKX y mostrar dashboard en http://localhost:5000

### CREDENCIALES OKX

El archivo config.env ya contiene tus credenciales OKX válidas y funcionales.

**IMPORTANTE**: No subas config.env a GitHub público - usa config.example.env en su lugar.