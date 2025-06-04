# SolanaScalper Bot - Instrucciones de Uso Local

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
