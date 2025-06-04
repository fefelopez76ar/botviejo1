# Lista de Archivos Críticos para Transferencia
## Fecha: 4 de Junio 2025 - 04:17 UTC

### ARCHIVOS OBLIGATORIOS (NO OPCIONAL)

#### 1. Bot Principal
- **main.py** (13,245 bytes) - Interfaz web Flask + control del bot
- **trading_bot.py** (12,660 bytes) - Motor de trading con ML
- **start_bot.py** (2,648 bytes) - Versión simplificada para pruebas

#### 2. Configuración y Datos
- **config.env** (588 bytes) - Credenciales OKX (CRÍTICO)
- **learning_data.json** (564 bytes) - Base de datos de aprendizaje

#### 3. API Client (CARPETA COMPLETA)
- **api_client/modulo2.py** (13,770 bytes) - Cliente WebSocket OKX
- **api_client/modulocola.py** (44 bytes) - Cola de datos
- **api_client/__init__.py** (si existe) - Inicialización del módulo

#### 4. Gestión de Datos
- **data_management/historical_data_saver_async.py** (8,235 bytes)
- **data_management/__init__.py** (si existe)

### ARCHIVOS DE SOPORTE (RECOMENDADOS)

#### Herramientas de Monitoreo
- **log_viewer.py** (3,199 bytes) - Visor de estadísticas
- **stats_tracker.py** (2,558 bytes) - Monitor tiempo real
- **verify_price.py** (2,820 bytes) - Verificador de estado

#### Documentación
- **LOCAL_SETUP_INSTRUCTIONS.md** - Guía de instalación completa
- **PROJECT_SUMMARY.md** - Resumen técnico del proyecto
- **BOT_MANUAL.md** (2,277 bytes) - Manual de usuario
- **BOT_STATUS.md** (1,212 bytes) - Estado del sistema

#### Registro de Desarrollo
- **info/estado_final_bot_20250604.md** - Estado final completo
- **info/cambios_implementados_20250604.md** - Cambios realizados
- **info/lista_archivos_criticos_20250604.md** - Este archivo

### ARCHIVOS OPCIONALES (PARA DESARROLLO)

#### Scripts de Utilidad
- **export_local_setup.py** - Script de exportación
- **copy_trading_modules.py** - Copiador de módulos
- **fix_main1.py** - Script de correcciones

#### Archivos de Testing
- **test_bot.py** - Tests del bot
- **test_okx.py** - Tests de conexión OKX
- **final_test.py** - Test final
- **run_scalping_demo.py** - Demo de scalping

#### Logs y Backups
- **trading_bot.log** - Logs de operaciones
- **bot.log** - Logs del sistema
- **market_data.db** - Base de datos de mercado (si existe)

### ESTRUCTURA MÍNIMA REQUERIDA

```
CryptoTradingBot/
├── main.py                          # OBLIGATORIO
├── trading_bot.py                   # OBLIGATORIO  
├── start_bot.py                     # OBLIGATORIO
├── config.env                       # CRÍTICO - Con credenciales
├── learning_data.json               # OBLIGATORIO - Datos ML
├── LOCAL_SETUP_INSTRUCTIONS.md      # RECOMENDADO
├── PROJECT_SUMMARY.md               # RECOMENDADO
├── api_client/                      # CARPETA OBLIGATORIA
│   ├── __init__.py                  # Si existe
│   ├── modulo2.py                   # CRÍTICO - Cliente WebSocket
│   └── modulocola.py                # OBLIGATORIO
├── data_management/                 # CARPETA RECOMENDADA
│   ├── __init__.py                  # Si existe
│   └── historical_data_saver_async.py # RECOMENDADO
├── info/                            # CARPETA DE DOCUMENTACIÓN
│   ├── estado_final_bot_20250604.md
│   ├── cambios_implementados_20250604.md
│   └── lista_archivos_criticos_20250604.md
├── log_viewer.py                    # ÚTIL para monitoreo
├── stats_tracker.py                 # ÚTIL para stats
└── verify_price.py                  # ÚTIL para verificación
```

### VALIDACIÓN DE ARCHIVOS

#### Verificar Antes de Transferir
1. **config.env contiene**:
   ```
   OKX_API_KEY=abc0a2f7-...
   OKX_API_SECRET=...
   OKX_PASSPHRASE=...
   ```

2. **learning_data.json contiene**:
   ```json
   {"operations": [...], "success_rate": 0.667}
   ```

3. **api_client/modulo2.py** - Archivo más grande (~13KB)

4. **main.py** - Contiene clase Flask app

#### Comando de Verificación en PC
```bash
# Verificar archivos principales
ls -la main.py trading_bot.py config.env learning_data.json

# Verificar carpeta api_client
ls -la api_client/

# Verificar que config.env tiene credenciales
grep "OKX_API_KEY" config.env
```

### TAMAÑOS DE ARCHIVOS PARA VALIDACIÓN

| Archivo | Tamaño (bytes) | Crítico |
|---------|----------------|---------|
| main.py | 13,245 | SÍ |
| trading_bot.py | 12,660 | SÍ |
| api_client/modulo2.py | 13,770 | SÍ |
| data_management/historical_data_saver_async.py | 8,235 | NO |
| config.env | 588 | SÍ |
| learning_data.json | 564 | SÍ |
| start_bot.py | 2,648 | SÍ |

### DEPENDENCIAS REQUERIDAS

#### requirements.txt (crear en PC)
```
websockets>=11.0.3
ccxt>=4.2.25
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
requests>=2.31.0
flask>=2.3.0
tabulate>=0.9.0
```

#### Comando de Instalación
```bash
pip install websockets ccxt pandas numpy python-dotenv requests flask tabulate
```

### CHECKLIST DE TRANSFERENCIA

#### ✅ Antes de Descargar de Replit
- [ ] Verificar que el bot funciona: `python main.py`
- [ ] Comprobar conexión OKX: `python verify_price.py`  
- [ ] Validar learning_data.json tiene datos
- [ ] Confirmar config.env tiene credenciales válidas

#### ✅ Después de Extraer en PC
- [ ] Todos los archivos obligatorios presentes
- [ ] Carpeta api_client/ completa
- [ ] config.env con credenciales correctas
- [ ] Instalar dependencias con pip
- [ ] Ejecutar: `python main.py`
- [ ] Verificar dashboard en http://localhost:5000

#### ✅ Para Subir a GitHub
- [ ] `git init` en carpeta del proyecto
- [ ] `git add .` para agregar todos los archivos
- [ ] `git commit -m "Bot trading Solana completo"`
- [ ] Conectar repositorio remoto
- [ ] `git push origin main`

### CREDENCIALES A PROTEGER

#### NO SUBIR A GITHUB PÚBLICO
- **config.env** - Contiene claves API reales
- **trading_bot.log** - Puede contener información sensible

#### ALTERNATIVA SEGURA
1. Crear **config.example.env** sin credenciales reales
2. Agregar **config.env** a **.gitignore**
3. Documentar en README.md que se necesita configurar credenciales

### CONTACTO PARA SOPORTE
- Usuario GitHub: fedelofedelooo
- Email: mariolopezabraham1@gmail.com
- Repositorio: https://github.com/fedelofedelooo/CryptoTradingBot

**NOTA**: Este bot está completamente funcional y probado. Seguir esta lista garantiza transferencia exitosa.