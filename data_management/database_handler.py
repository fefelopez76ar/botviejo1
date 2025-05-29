import logging
import json
import time # Necesario para time.sleep
import math # Necesario para math.pow
import sqlite3
from pathlib import Path

class DatabaseHandler:
    def __init__(self, db_path='market_data.db'):
        self.db_path = db_path
        self.conn = None
        self.logger = logging.getLogger(__name__)
        # Configuración para reintentos
        self.MAX_RETRIES = 5 # Número máximo de veces que intentará una operación
        self.INITIAL_BACKOFF_SECONDS = 0.1 # Tiempo de espera inicial antes del primer reintento
        self.MAX_BACKOFF_SECONDS = 5 # Limite superior para el tiempo de espera exponencial

    def connect(self):
        """
        Intenta conectar a la base de datos con reintentos y backoff exponencial.
        """
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                self.conn = sqlite3.connect(self.db_path)
                self.conn.row_factory = sqlite3.Row # Permite acceder a las columnas por nombre
                self.logger.info(f"Conectado a la base de datos: {self.db_path}")
                return True
            except sqlite3.Error as e:
                self.logger.error(f"Error de conexión a la base de datos (Intento {retries + 1}/{self.MAX_RETRIES}): {e}")
                retries += 1
                if retries < self.MAX_RETRIES:
                    # Calcula el tiempo de espera: 0.1, 0.2, 0.4, 0.8, 1.6... segundos, limitado por MAX_BACKOFF_SECONDS
                    sleep_time = min(self.MAX_BACKOFF_SECONDS, self.INITIAL_BACKOFF_SECONDS * math.pow(2, retries - 1))
                    self.logger.warning(f"Reintentando conexión en {sleep_time:.2f} segundos...")
                    time.sleep(sleep_time)
        self.logger.critical(f"Falló la conexión a la base de datos después de {self.MAX_RETRIES} reintentos. El bot no podrá guardar datos.")
        self.conn = None
        return False

    def create_tables(self):
        """
        Crea las tablas 'tickers' y 'order_book' si no existen, con reintentos.
        """
        if not self.conn:
            self.logger.error("No hay conexión a la base de datos para crear tablas. Conéctese primero.")
            return False

        cursor = self.conn.cursor()
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tickers (
                        instId TEXT NOT NULL,
                        last REAL,
                        askPx REAL,
                        bidPx REAL,
                        ts INTEGER NOT NULL,
                        PRIMARY KEY (instId, ts)
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS order_book (
                        instId TEXT NOT NULL,
                        asks TEXT, -- Almacenará la lista de asks como JSON
                        bids TEXT, -- Almacenará la lista de bids como JSON
                        ts INTEGER NOT NULL,
                        PRIMARY KEY (instId, ts)
                    )
                ''')
                self.conn.commit()
                self.logger.info("Tablas de la base de datos creadas o ya existentes.")
                return True
            except sqlite3.OperationalError as e: # Errores como DB bloqueada, permisos, etc.
                self.logger.error(f"Error operacional al crear tablas (Intento {retries + 1}/{self.MAX_RETRIES}): {e}")
                retries += 1
                if retries < self.MAX_RETRIES:
                    sleep_time = min(self.MAX_BACKOFF_SECONDS, self.INITIAL_BACKOFF_SECONDS * math.pow(2, retries - 1))
                    self.logger.warning(f"Reintentando creación de tablas en {sleep_time:.2f} segundos...")
                    time.sleep(sleep_time)
            except sqlite3.Error as e: # Otros errores de SQLite más generales
                self.logger.error(f"Error general al crear tablas: {e}")
                return False # Otros errores no operacionales podrían ser fatales y no reintentables aquí
        self.logger.critical(f"Falló la creación de tablas después de {self.MAX_RETRIES} reintentos.")
        return False

    def insert_many_ticker_data(self, data_list):
        """
        Inserta múltiples registros de datos de tickers en la base de datos en un solo lote,
        con manejo de errores y reintentos.
        `data_list` es una lista de diccionarios, cada uno con datos de ticker.
        """
        if not self.conn:
            self.logger.error("No hay conexión a la base de datos para insertar múltiples tickers.")
            return False
        
        records = []
        for data in data_list:
            try:
                # Asegurarse de que los tipos de datos sean correctos para SQLite
                instId = str(data.get('instId'))
                last = float(data.get('last'))
                askPx = float(data.get('askPx'))
                bidPx = float(data.get('bidPx'))
                ts = int(data.get('ts'))
                records.append((instId, last, askPx, bidPx, ts))
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Datos de ticker inválidos o incompletos, se omitirán: {data} - Error: {e}")
                continue # Saltar a la siguiente iteración si los datos no son válidos

        if not records: # Si la lista de registros está vacía después de la validación
            self.logger.info("No hay registros de ticker válidos para insertar.")
            return True

        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                cursor = self.conn.cursor()
                # Usamos INSERT OR IGNORE para evitar errores si ya existe un registro con la misma PK (instId, ts)
                cursor.executemany('''
                    INSERT OR IGNORE INTO tickers (instId, last, askPx, bidPx, ts)
                    VALUES (?, ?, ?, ?, ?)
                ''', records)
                self.conn.commit()
                self.logger.info(f"Insertados {len(records)} registros de ticker en lote.")
                return True
            except sqlite3.OperationalError as e:
                self.logger.error(f"Error operacional al insertar tickers (Intento {retries + 1}/{self.MAX_RETRIES}): {e}")
                retries += 1
                if retries < self.MAX_RETRIES:
                    sleep_time = min(self.MAX_BACKOFF_SECONDS, self.INITIAL_BACKOFF_SECONDS * math.pow(2, retries - 1))
                    self.logger.warning(f"Reintentando inserción de tickers en {sleep_time:.2f} segundos...")
                    time.sleep(sleep_time)
                else:
                    self.logger.critical(f"Falló la inserción de tickers después de {self.MAX_RETRIES} reintentos. Datos potencialmente perdidos para: {len(records)} registros.")
                    # TODO: Aquí se podría implementar la lógica de fallback a disco si es una operación crítica.
            except sqlite3.Error as e:
                self.logger.error(f"Error general al insertar datos de ticker: {e}")
                return False # Otros errores de SQLite (ej. esquema incorrecto) no se reintentan

        return False # Falló después de todos los reintentos

    def close(self):
        """
        Cierra la conexión a la base de datos si está abierta.
        """
        if self.conn:
            self.conn.close()
            self.conn = None # Establece la conexión a None para indicar que está cerrada
            self.logger.info("Conexión a la base de datos cerrada.")
