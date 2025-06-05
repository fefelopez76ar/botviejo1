import asyncio
import logging
import os
from datetime import datetime
import time # Para simular timestamps reales

# Asegúrate de que las importaciones de tus clases sean correctas
# Si tus archivos están en 'data_management', esto es correcto:
from data_management.database_handler import DatabaseHandler
from data_management.historical_data_saver import HistoricalDataSaver

# Configurar logging para ver la salida
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ruta para una base de datos de prueba temporal
TEST_DB_PATH = 'data/test_market_data.db'
TEST_DB_DIR = 'data'

async def run_test():
    logger.info("Iniciando test de integración de DatabaseHandler y HistoricalDataSaver...")

    # Asegurar que el directorio de datos exista
    os.makedirs(TEST_DB_DIR, exist_ok=True)

    # Eliminar la base de datos de prueba si ya existe, para asegurar un inicio limpio
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
        logger.info(f"Base de datos de prueba existente '{TEST_DB_PATH}' eliminada para un inicio limpio.")

    # =========================================================
    # PRUEBA DE DATABASEHANDLER DIRECTAMENTE (Singleton)
    logger.info("\n--- Prueba directa de DatabaseHandler ---")
    db_h_test = DatabaseHandler(db_path=TEST_DB_PATH)
    if await db_h_test.connect():
        logger.info("Conexión directa a DatabaseHandler exitosa.")

        # Datos de ticker de ejemplo
        mock_ticker_data_single = {
            'instId': 'SOL-USDT',
            'last': 150.0,
            'askPx': 150.1,
            'bidPx': 149.9,
            'ts': int(datetime.now().timestamp() * 1000) # Milisegundos
        }
        if await db_h_test.insert_ticker_data(mock_ticker_data_single):
            logger.info("Inserción de ticker individual (DatabaseHandler) exitosa.")
        else:
            logger.error("Fallo la inserción de ticker individual (DatabaseHandler).")

        # Datos de ticker en lote de ejemplo
        mock_tickers_data_batch = [
            {'instId': 'ETH-USDT', 'last': 3000.0, 'askPx': 3000.5, 'bidPx': 2999.5, 'ts': int(datetime.now().timestamp() * 1000) - 1000},
            {'instId': 'BTC-USDT', 'last': 60000.0, 'askPx': 60001.0, 'bidPx': 59999.0, 'ts': int(datetime.now().timestamp() * 1000) - 500}
        ]
        if await db_h_test.insert_many_ticker_data(mock_tickers_data_batch):
            logger.info("Inserción de tickers en lote (DatabaseHandler) exitosa.")
        else:
            logger.error("Fallo la inserción de tickers en lote (DatabaseHandler).")

        # Datos de order book L2 de ejemplo
        mock_order_book_data_single = {
            'arg': {'channel': 'books-l2-tbt', 'instId': 'SOL-USDT'},
            'data': [{
                'instId': 'SOL-USDT',
                'asks': [['150.1', '10', '0', '1']],
                'bids': [['149.9', '5', '0', '1']],
                'ts': str(int(datetime.now().timestamp() * 1000) + 1000) # Milisegundos
            }]
        }
        if await db_h_test.insert_order_book_l2_data(mock_order_book_data_single):
            logger.info("Inserción de order book L2 individual (DatabaseHandler) exitosa.")
        else:
            logger.error("Fallo la inserción de order book L2 individual (DatabaseHandler).")

        # Datos de order book L2 en lote de ejemplo
        mock_order_book_data_batch = [
            {
                'arg': {'channel': 'books-l2-tbt', 'instId': 'ETH-USDT'},
                'data': [{
                    'instId': 'ETH-USDT',
                    'asks': [['3000.6', '12', '0', '1']],
                    'bids': [['2999.4', '7', '0', '1']],
                    'ts': str(int(datetime.now().timestamp() * 1000) + 2000) # Milisegundos
                }]
            },
            {
                'arg': {'channel': 'books-l2-tbt', 'instId': 'BTC-USDT'},
                'data': [{
                    'instId': 'BTC-USDT',
                    'asks': [['60002.0', '15', '0', '1']],
                    'bids': [['59998.0', '8', '0', '1']],
                    'ts': str(int(datetime.now().timestamp() * 1000) + 3000) # Milisegundos
                }]
            }
        ]
        if await db_h_test.insert_many_order_book_l2_data(mock_order_book_data_batch):
            logger.info("Inserción de order books L2 en lote (DatabaseHandler) exitosa.")
        else:
            logger.error("Fallo la inserción de order books L2 en lote (DatabaseHandler).")

    else:
        logger.error("Fallo la conexión directa a DatabaseHandler. Las pruebas subsiguientes no se realizarán.")
        return # Salir si la conexión inicial falla
    db_h_test.close() # Cierra la conexión de la prueba directa

    # Pequeña pausa para asegurar que la conexión se cierra antes de abrir una nueva por HDS
    await asyncio.sleep(0.5)

    # =========================================================
    # PRUEBA DE HISTORICALDATASAVER (Que usa DatabaseHandler internamente)
    logger.info("\n--- Prueba de HistoricalDataSaver (usando DatabaseHandler) ---")
    hds = HistoricalDataSaver()
    try:
        await hds.initialize() # Esto conectará la DB y creará tablas (si no existen)
        logger.info("HistoricalDataSaver inicializado y conectado exitosamente.")

        # Datos de ticker para HDS
        mock_ticker_hds = {
            'instId': 'XRP-USDT',
            'last': 0.5,
            'askPx': 0.501,
            'bidPx': 0.499,
            'ts': int(datetime.now().timestamp() * 1000) + 5000 # Milisegundos
        }
        if await hds.save_data('tickers', mock_ticker_hds):
            logger.info("Guardado de ticker (HistoricalDataSaver) exitoso.")
        else:
            logger.error("Fallo el guardado de ticker (HistoricalDataSaver).")

        # Datos de order book L2 para HDS
        mock_book_hds = {
            'arg': {'channel': 'books-l2-tbt', 'instId': 'XRP-USDT'},
            'data': [{
                'instId': 'XRP-USDT',
                'asks': [['0.502', '20', '0', '1']],
                'bids': [['0.498', '15', '0', '1']],
                'ts': str(int(datetime.now().timestamp() * 1000) + 6000) # Milisegundos
            }]
        }
        if await hds.save_data('books-l2-tbt', mock_book_hds):
            logger.info("Guardado de order book L2 (HistoricalDataSaver) exitoso.")
        else:
            logger.error("Fallo el guardado de order book L2 (HistoricalDataSaver).")

        # Datos de ticker en lote para HDS
        mock_tickers_hds_batch = [
            {'instId': 'ADA-USDT', 'last': 0.3, 'askPx': 0.301, 'bidPx': 0.299, 'ts': int(datetime.now().timestamp() * 1000) + 7000},
            {'instId': 'SOL-USDT', 'last': 151.0, 'askPx': 151.1, 'bidPx': 150.9, 'ts': int(datetime.now().timestamp() * 1000) + 7500}
        ]
        if await hds.save_many_data('tickers', mock_tickers_hds_batch):
            logger.info("Guardado de múltiples tickers (HistoricalDataSaver) exitoso.")
        else:
            logger.error("Fallo el guardado de múltiples tickers (HistoricalDataSaver).")

        # Datos de order book L2 en lote para HDS
        mock_books_hds_batch = [
            {
                'arg': {'channel': 'books-l2-tbt', 'instId': 'DOGE-USDT'},
                'data': [{
                    'instId': 'DOGE-USDT',
                    'asks': [['0.07', '100', '0', '1']],
                    'bids': [['0.069', '80', '0', '1']],
                    'ts': str(int(datetime.now().timestamp() * 1000) + 8000)
                }]
            },
            {
                'arg': {'channel': 'books-l2-tbt', 'instId': 'SHIB-USDT'},
                'data': [{
                    'instId': 'SHIB-USDT',
                    'asks': [['0.000025', '2000', '0', '1']],
                    'bids': [['0.000024', '1500', '0', '1']],
                    'ts': str(int(datetime.now().timestamp() * 1000) + 8500)
                }]
            }
        ]
        if await hds.save_many_data('books-l2-tbt', mock_books_hds_batch):
            logger.info("Guardado de múltiples order books L2 (HistoricalDataSaver) exitoso.")
        else:
            logger.error("Fallo el guardado de múltiples order books L2 (HistoricalDataSaver).")


    except Exception as e:
        logger.error(f"Error durante la prueba de HistoricalDataSaver: {e}", exc_info=True)
    finally:
        hds.close()
        logger.info("HistoricalDataSaver cerrado.")

    logger.info("\n--- Fin del test de integración ---")
    logger.info(f"Puedes verificar la base de datos creada en: {TEST_DB_PATH}")


if __name__ == "__main__":
    asyncio.run(run_test())
