import json
import logging
from data_management.database_handler import DatabaseHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HistoricalDataSaver:
    def __init__(self, db_handler: DatabaseHandler):
        self.db_handler = db_handler
        logging.info("HistoricalDataSaver inicializado.")

    async def save_data(self, data_type, instrument_id, data):
        if data_type == 'ticker':
            await self.save_ticker_data(instrument_id, data)
        elif data_type == 'order_book':
            await self.save_order_book_data(instrument_id, data)
        elif data_type == 'candle':
            await self.save_candle_data(instrument_id, data)
        else:
            logging.warning(f"Tipo de dato desconocido para guardar: {data_type}")

    async def save_ticker_data(self, instrument_id, ticker_data):
        data_to_save = {
            'instId': instrument_id,
            'idxPx': ticker_data.get('idxPx'),
            'markPx': ticker_data.get('markPx'),
            'lastId': ticker_data.get('lastId'),
            'lastPx': ticker_data.get('lastPx'),
            'sodUtc0': ticker_data.get('sodUtc0'),
            'sodUtc8': ticker_data.get('sodUtc8'),
            'open24h': ticker_data.get('open24h'),
            'high24h': ticker_data.get('high24h'),
            'low24h': ticker_data.get('low24h'),
            'volCcy24h': ticker_data.get('volCcy24h'),
            'vol24h': ticker_data.get('vol24h'),
            'ts': ticker_data.get('ts'),
            'bidPx': ticker_data.get('bidPx'),
            'bidSz': ticker_data.get('bidSz'),
            'askPx': ticker_data.get('askPx'),
            'askSz': ticker_data.get('askSz')
        }
        await self.db_handler.insert_ticker_data(data_to_save)

    async def save_order_book_data(self, instrument_id, order_book_data):
        data_to_save = {
            'instId': instrument_id,
            'channel': order_book_data.get('channel'),
            'bids': json.dumps(order_book_data.get('bids')),
            'asks': json.dumps(order_book_data.get('asks')),
            'ts': order_book_data.get('ts'),
            'seqId': order_book_data.get('seqId')
        }
        await self.db_handler.insert_order_book_l2_data(data_to_save)

    async def save_candle_data(self, instrument_id, candle_data):
        data_to_save = {
            'instId': instrument_id,
            'channel': 'candles',
            'ts': candle_data[0],
            'open': candle_data[1],
            'high': candle_data[2],
            'low': candle_data[3],
            'close': candle_data[4],
            'vol': candle_data[5],
            'volCcy': candle_data[6],
            'confirm': 1
        }
        await self.db_handler.insert_candle_data(data_to_save)

    def disconnect(self):
        """Método público para desconectar de la base de datos."""
        if self.db_handler.conn:
            self.db_handler.conn.close()
            logging.info("Conexión a la base de datos cerrada.")