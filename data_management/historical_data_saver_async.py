import aiosqlite
import logging
import os
from typing import Dict, Any

logger = logging.getLogger("SolanaScalper")

class HistoricalDataSaver:
    def __init__(self, db_path="data/market_data.db"):
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', "market_data.db")
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        self.conn = None

    async def connect(self):
        try:
            self.conn = await aiosqlite.connect(self.db_path)
            await self._create_tables()
            logger.info(f"[HistoricalDataSaver]: Conectado a DB async '{self.db_path}' y tablas verificadas.")
        except Exception as e:
            logger.error(f"[HistoricalDataSaver ERROR]: Error al conectar a la base de datos: {e}")
            self.conn = None

    async def disconnect(self):
        if self.conn:
            await self.conn.close()
            self.conn = None
            logger.info(f"[HistoricalDataSaver]: Desconectado de la base de datos '{self.db_path}'.")

    async def _create_tables(self):
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tickers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                instrument_id TEXT NOT NULL,
                last_price REAL,
                ask_1 REAL,
                bid_1 REAL,
                high_24h REAL,
                low_24h REAL,
                vol_ccy_24h REAL,
                vol_24h REAL,
                CONSTRAINT unique_ticker_timestamp_instrument UNIQUE (timestamp, instrument_id)
            )
        """)
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS order_book (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                instrument_id TEXT NOT NULL,
                best_bid REAL,
                best_bid_size REAL,
                best_ask REAL,
                best_ask_size REAL,
                checksum TEXT,
                CONSTRAINT unique_ob_timestamp_instrument UNIQUE (timestamp, instrument_id)
            )
        """)
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS candlesticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                instrument_id TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume REAL,
                volume_currency REAL,
                timestamp_received INTEGER NOT NULL,
                CONSTRAINT unique_candle_timestamp_instrument_interval UNIQUE (timestamp, instrument_id, interval)
            )
        """)
        await self.conn.commit()

    async def save_ticker_data(self, data: Dict[str, Any]):
        if not self.conn:
            logger.warning("[HistoricalDataSaver]: Conexión a DB no activa al guardar ticker. Intentando reconectar...")
            await self.connect()
            if not self.conn:
                logger.error("[HistoricalDataSaver ERROR]: No se pudo reconectar a la DB para guardar ticker.")
                return

        try:
            ticker_info = data.get('data')[0] if isinstance(data.get('data'), list) else {}
            ts = int(ticker_info.get('ts'))
            inst_id = ticker_info.get('instId')
            last = float(ticker_info.get('last', 0))
            ask_1 = float(ticker_info.get('askPx', 0))
            bid_1 = float(ticker_info.get('bidPx', 0))
            high_24h = float(ticker_info.get('high24h', 0))
            low_24h = float(ticker_info.get('low24h', 0))
            vol_ccy_24h = float(ticker_info.get('volCcy24h', 0))
            vol_24h = float(ticker_info.get('vol24h', 0))

            await self.conn.execute("""
                INSERT OR IGNORE INTO tickers (
                    timestamp, instrument_id, last_price, ask_1, bid_1,
                    high_24h, low_24h, vol_ccy_24h, vol_24h
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ts, inst_id, last, ask_1, bid_1, high_24h, low_24h, vol_ccy_24h, vol_24h))
            await self.conn.commit()
            logger.debug(f"[HistoricalDataSaver]: Ticker guardado para {inst_id} @ {last}")
        except Exception as e:
            logger.error(f"[HistoricalDataSaver ERROR]: Error al guardar ticker: {e}")

    async def save_order_book_data(self, data: Dict[str, Any]):
        if not self.conn:
            logger.warning("[HistoricalDataSaver]: Conexión a DB no activa al guardar order book. Intentando reconectar...")
            await self.connect()
            if not self.conn:
                logger.error("[HistoricalDataSaver ERROR]: No se pudo reconectar a la DB para guardar order book.")
                return

        try:
            arg = data.get('arg', {})
            inst_id = arg.get('instId')
            ob_data = data.get('data')[0] if isinstance(data.get('data'), list) else {}
            ts = int(ob_data.get('ts'))

            bids = ob_data.get('bids', [])
            asks = ob_data.get('asks', [])

            best_bid_px = float(bids[0][0]) if bids else None
            best_bid_sz = float(bids[0][1]) if bids else None
            best_ask_px = float(asks[0][0]) if asks else None
            best_ask_sz = float(asks[0][1]) if asks else None
            checksum = ob_data.get('checksum')

            await self.conn.execute("""
                INSERT OR IGNORE INTO order_book (
                    timestamp, instrument_id, best_bid, best_bid_size,
                    best_ask, best_ask_size, checksum
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ts, inst_id, best_bid_px, best_bid_sz, best_ask_px, best_ask_sz, checksum))
            await self.conn.commit()
            logger.debug(f"[HistoricalDataSaver]: Order book guardado para {inst_id}")
        except Exception as e:
            logger.error(f"[HistoricalDataSaver ERROR]: Error al guardar order book: {e}")

    async def save_candlestick_data(self, data: Dict[str, Any]):
        if not self.conn:
            logger.warning("[HistoricalDataSaver]: Conexión a DB no activa al guardar candlestick. Intentando reconectar...")
            await self.connect()
            if not self.conn:
                logger.error("[HistoricalDataSaver ERROR]: No se pudo reconectar a la DB para guardar candlestick.")
                return

        try:
            inst_id = data.get('instrument')
            interval = data.get('interval')
            timestamp_received = data.get('timestamp_received')

            candle_data_list = data.get('data', [])
            if not candle_data_list:
                logger.warning(f"[HistoricalDataSaver]: Datos de vela vacíos para {inst_id}. Saltando guardado.")
                return

            candle_data = candle_data_list[0]

            if len(candle_data) < 7:
                logger.warning(f"[HistoricalDataSaver]: Datos de vela incompletos para {inst_id} ({interval}). Saltando guardado. Datos: {candle_data}")
                return

            ts = int(candle_data[0])
            open_price = float(candle_data[1])
            high_price = float(candle_data[2])
            low_price = float(candle_data[3])
            close_price = float(candle_data[4])
            volume = float(candle_data[5])
            volume_currency = float(candle_data[6])

            await self.conn.execute(
                """
                INSERT OR IGNORE INTO candlesticks (
                    timestamp, instrument_id, interval, open_price, high_price, low_price, close_price, volume, volume_currency, timestamp_received
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, inst_id, interval, open_price, high_price, low_price, close_price, volume, volume_currency, timestamp_received)
            )
            await self.conn.commit()
        except Exception as e:
            logger.error(f"[HistoricalDataSaver ERROR]: Error al guardar datos de candlestick: {e}")
