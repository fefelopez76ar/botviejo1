import sqlite3
import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HistoricalDataSaver:
    def __init__(self, db_name='market_data.db'):
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        self._connect()

    def _connect(self):
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.cursor = self.conn.cursor()
            self._create_tables()
            logging.info(f"Conectado a la base de datos: {self.db_name}")
        except sqlite3.Error as e:
            logging.error(f"Error al conectar a la base de datos: {e}")

    def _create_tables(self):
        # Tabla para tickers
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tickers (
                instId TEXT,
                last REAL,
                askPx REAL,
                bidPx REAL,
                ts INTEGER,
                PRIMARY KEY (instId, ts)
            )
        ''')
        # Tabla para book_l2 (solo el mejor bid/ask para simplificar el almacenamiento continuo)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS order_book_l2 (
                instId TEXT,
                bidPx REAL,
                bidSz REAL,
                askPx REAL,
                askSz REAL,
                ts INTEGER,
                PRIMARY KEY (instId, ts)
            )
        ''')
        self.conn.commit()

    async def save_data(self, data_type, data):
        try:
            if data_type == 'tickers' and data:
                # 'data' es una lista de diccionarios, tomamos el primero
                ticker = data[0]
                instId = ticker.get('instId')
                last = float(ticker.get('last'))
                askPx = float(ticker.get('askPx'))
                bidPx = float(ticker.get('bidPx'))
                ts = int(ticker.get('ts')) # Timestamp en milisegundos

                self.cursor.execute('''
                    INSERT OR IGNORE INTO tickers (instId, last, askPx, bidPx, ts)
                    VALUES (?, ?, ?, ?, ?)
                ''', (instId, last, askPx, bidPx, ts))
                self.conn.commit()
                logging.debug(f"Ticker guardado para {instId} @ {last}")

            elif data_type == 'books-l2-tbt' and data and data[0].get('asks') and data[0].get('bids'):
                book = data[0]
                instId = book.get('instId')
                ts = int(book.get('ts'))
                # Solo tomamos el mejor bid/ask para almacenamiento continuo del libro de órdenes
                best_bid_px = float(book['bids'][0][0]) if book['bids'] else None
                best_bid_sz = float(book['bids'][0][1]) if book['bids'] else None
                best_ask_px = float(book['asks'][0][0]) if book['asks'] else None
                best_ask_sz = float(book['asks'][0][1]) if book['asks'] else None

                self.cursor.execute('''
                    INSERT OR IGNORE INTO order_book_l2 (instId, bidPx, bidSz, askPx, askSz, ts)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (instId, best_bid_px, best_bid_sz, best_ask_px, best_ask_sz, ts))
                self.conn.commit()
                logging.debug(f"Order book L2 guardado para {instId} @ Bid: {best_bid_px}, Ask: {best_ask_px}")

        except (sqlite3.Error, ValueError) as e:
            logging.error(f"Error al guardar datos '{data_type}': {e} - Data: {data}")
        except Exception as e:
            logging.error(f"Error inesperado en save_data: {e} - Data: {data}")

    def close(self):
        if self.conn:
            self.conn.close()
            logging.info("Conexión a la base de datos cerrada.")