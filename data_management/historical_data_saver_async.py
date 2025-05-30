import aiosqlite
import logging
import os
from typing import Dict, Any

logger = logging.getLogger("SolanaScalper")

class HistoricalDataSaver:
    def __init__(self, db_path="info/market_data.db"):
        self.db_path = db_path
        # Asegúrate de que el directorio de la base de datos exista
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

    async def disconnect(self):
        if self.conn:
            await self.conn.close()
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
        await self.conn.commit()

    async def save_ticker_data(self, data: Dict[str, Any]):
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

    async def save_data(self, data: Dict[str, Any]):
        if not self.conn:
            await self.connect()  # Asegurarse de que la conexión esté inicializada
        try:
            channel = data.get('arg', {}).get('channel')
            if channel == 'tickers':
                ticker_info = data.get('data')[0] if isinstance(data.get('data'), list) else {}
                await self.conn.execute(
                    """
                    INSERT OR IGNORE INTO tickers (timestamp, instrument_id, last_price, ask_1, bid_1, high_24h, low_24h, vol_ccy_24h)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(ticker_info.get('ts', 0)),
                        ticker_info.get('instId', ''),
                        float(ticker_info.get('last', 0)),
                        float(ticker_info.get('askPx', 0)),
                        float(ticker_info.get('bidPx', 0)),
                        float(ticker_info.get('high24h', 0)),
                        float(ticker_info.get('low24h', 0)),
                        float(ticker_info.get('volCcy24h', 0))
                    )
                )
            elif channel == 'books-l2-tbt':
                order_book_info = data.get('data')[0] if isinstance(data.get('data'), list) else {}
                await self.conn.execute(
                    """
                    INSERT OR IGNORE INTO order_book (timestamp, instrument_id, best_bid, best_bid_size, best_ask, best_ask_size, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(order_book_info.get('ts', 0)),
                        order_book_info.get('instId', ''),
                        float(order_book_info.get('bids', [[0]])[0][0]),
                        float(order_book_info.get('bids', [[0]])[0][1]),
                        float(order_book_info.get('asks', [[0]])[0][0]),
                        float(order_book_info.get('asks', [[0]])[0][1]),
                        order_book_info.get('checksum', '')
                    )
                )
            await self.conn.commit()
        except Exception as e:
            logger.error(f"[HistoricalDataSaver ERROR]: Error al guardar datos: {e}")
