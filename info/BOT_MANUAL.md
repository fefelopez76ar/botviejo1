# MANUAL DEL BOT DE TRADING SOLANA

## 1. MODOS DE OPERACIÓN
- **Paper Trading [1]**: Modo simulación (sin riesgo real)
- **Live Trading [2]**: Modo real (requiere validaciones)

## 2. MENÚ PRINCIPAL Y OPERACIONES
1. **Scalping en Tiempo Real [1]**
   - Configurar par (SOL-USDT default)
   - Seleccionar timeframe (1m, 3m, 5m, 15m)
   - Estrategias disponibles:
     - RSI adaptativo
     - Momentum con reversión
     - Patrones de velas
     - Estrategia adaptativa (ML)

2. **Backtesting [2]**
   - Test rápido
   - Optimización de parámetros
   - Comparar estrategias
   - Análisis adaptativo

3. **Configuración [3]**
   - Modo trading
   - API exchange
   - Gestión de riesgo
   - Notificaciones
   - Opciones avanzadas

4. **Monitor [4]**
   - Estado del sistema
   - Log de operaciones
   - Gráficos de rendimiento
   - Log de errores

## 3. SEGURIDAD Y VALIDACIONES
- Circuit breakers automáticos
- Límites de pérdida diaria
- Requisitos para modo real:
  - Min. 100 operaciones en paper
  - Win rate > 55% últimas 50 ops
  - Drawdown máx < 15%

## 4. SISTEMA ADAPTATIVO
- Ponderación dinámica
- Ajuste según mercado
- Aprendizaje de patrones
- Export/Import de "cerebro"

## 5. GESTIÓN DE RIESGOS
- Tamaño posición: 1-3%
- Stop loss dinámico
- Take profit escalonado
- Control drawdown
- Límites exposición

## 6. NOTIFICACIONES
- Alertas operaciones
- Telegram integrado
- Alertas riesgo
- Resumen diario

## 7. INTERFAZ WEB
- Dashboard tiempo real
- Gráficos rendimiento
- Estado operaciones
- Métricas trading

## 8. COMANDOS RÁPIDOS
```bash
python main.py --mode paper         # Iniciar simulación
python main.py --mode live         # Iniciar real (con validaciones)
python bot_cli.py                  # Interfaz línea comandos
```

## 9. ESTADOS DEL BOT
- RUNNING: Operando
- MONITORING: Analizando
- TRADING: En posición
- STOPPED: Detenido
- ERROR: Error detectado

## 10. ARCHIVOS PRINCIPALES
- main.py: Entrada principal
- trading_bot.py: Core del bot
- strategies/: Estrategias
- config.env: Configuración

## 11. REQUISITOS
- Python 3.11.7
- Dependencias en requirements.txt
- Conexión estable
- API keys (modo real)

## 12. MANTENIMIENTO
- Backups automáticos
- Logs detallados
- Reporte errores
- Actualizaciones auto

## Forma de Trabajo

### Documentación de Cambios
En cada cambio realizado en el proyecto, se debe:

1. **Registrar el Cambio:**
   - Anotar qué se hizo, cómo se implementó, y el estado actual del proyecto.
   - Detallar los próximos pasos necesarios para avanzar.

2. **Actualizar Archivos de Documentación:**
   - Modificar los archivos en la carpeta `info/` para reflejar los cambios realizados.
   - Crear un registro en el archivo de trabajo diario correspondiente (`work_log_YYYYMMDD.txt`).

3. **Estructura del Registro:**
   - **Qué se hizo:** Breve descripción del cambio.
   - **Cómo se implementó:** Archivos modificados, scripts creados, y ubicación en la estructura del proyecto.
   - **Estado actual:** Validación del cambio y su impacto en el proyecto.
   - **Próximos pasos:** Tareas pendientes relacionadas con el cambio.
