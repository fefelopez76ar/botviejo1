# Solana Trading Bot

Un bot de trading algorítmico para el mercado de Solana, con análisis técnico avanzado, aprendizaje adaptativo y notificaciones por Telegram.

## Características principales

- **Múltiples estrategias:** Desde estrategias clásicas (cruce de medias móviles, RSI, MACD) hasta estrategias avanzadas basadas en Machine Learning.
- **Ponderación adaptativa:** Sistema que aprende del rendimiento histórico de los indicadores y ajusta sus pesos dinámicamente.
- **Backtesting integrado:** Prueba tus estrategias con datos históricos antes de ponerlas en producción.
- **Modos de operación:** Soporte para trading simulado (paper trading) y real.
- **Gestión de múltiples bots:** Ejecuta varios bots simultáneamente con diferentes estrategias y configuraciones.
- **Gestión de riesgos:** Control avanzado de tamaño de posición, stop loss y take profit.
- **Notificaciones por Telegram:** Recibe alertas de operaciones y actualizaciones de estado.
- **Interfaz CLI amigable:** Menús fáciles de usar desde la línea de comandos.

## Estructura del proyecto

```
solana_trading_bot/
├── api_client/             # Cliente para interactuar con exchanges
├── indicators/             # Cálculo de indicadores técnicos
├── strategies/             # Estrategias de trading
├── risk_management/        # Gestión de riesgos y tamaño de posición
├── backtesting/            # Motor de backtesting y optimización
├── data_management/        # Gestión de datos de mercado
├── adaptive_system/        # Sistema adaptativo de ponderación
├── notifications/          # Sistema de notificaciones
├── interface/              # Interfaz de usuario CLI
├── db/                     # Modelos y operaciones de base de datos
├── core/                   # Núcleo del sistema
└── utils/                  # Utilidades generales
```

## Requisitos

- Python 3.8 o superior
- Paquetes Python (ver requirements.txt)
- Cuenta en OKX (opcional, solo para trading real)
- Bot de Telegram (opcional, para notificaciones)

## Instalación

1. Clona el repositorio:
   ```
   git clone https://github.com/tu-usuario/solana-trading-bot.git
   cd solana-trading-bot
   ```

2. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

3. Configura las credenciales API (para trading real):
   - Copia `config.example.env` a `config.env`
   - Edita `config.env` con tus credenciales API y preferencias

## Uso

Ejecuta el bot desde la línea de comandos:

```
python bot_cli.py
```

Sigue las instrucciones en pantalla para:
- Crear y gestionar bots
- Configurar estrategias
- Realizar backtesting
- Ver análisis de mercado
- Configurar notificaciones
- Optimizar parámetros
- Utilizar funciones de IA y aprendizaje

## Módulo de IA

El módulo de IA incorpora:

1. **Optimización de estrategias:** Analiza el rendimiento de diferentes estrategias y parámetros para encontrar las configuraciones óptimas.
2. **Ponderación adaptativa:** Aprende qué indicadores funcionan mejor en cada condición de mercado y ajusta sus pesos automáticamente.
3. **Gestión de riesgos inteligente:** Ajusta dinámicamente los parámetros de riesgo basándose en el comportamiento del mercado y la confianza en las señales.
4. **Detección de regímenes de mercado:** Identifica automáticamente condiciones de mercado (tendencias, lateralidad, volatilidad) para aplicar estrategias específicas.
5. **Machine Learning:** Modelos de clasificación y regresión para predicción de movimientos de mercado.

## Advertencia

El trading algorítmico conlleva riesgos. Este software se proporciona tal cual, sin garantía de ningún tipo. Los autores no se hacen responsables de las pérdidas que puedan resultar de su uso. Úsalo bajo tu propia responsabilidad y nunca inviertas dinero que no puedas permitirte perder.

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT. Ver el archivo LICENSE para más detalles.

---

**Nota:** Este bot está optimizado para el trading de Solana (SOL), pero puede adaptarse a otros criptoactivos ajustando la configuración y las estrategias.