import sqlite3
import os
import json # Para pretty-print bids/asks si es necesario
from pathlib import Path

# La ruta a tu base de datos (asegúrate de que sea la correcta)
DB_PATH = Path('market_data.db')

def verify_table(cursor, table_name, primary_key_cols, extra_cols=None):
    """Verifica si una tabla existe y muestra sus registros."""
    try:
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if cursor.fetchone():
            print(f"\n--- Tabla '{table_name}' ---")
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"Número de registros en '{table_name}': {count}")

            if count > 0:
                print(f"Últimos 5 registros en '{table_name}':")
                # Seleccionar todas las columnas conocidas o solo las principales
                if table_name == 'tickers':
                    cols = "instId, idxPx, markPx, lastPx, ts, bidPx, askPx"
                elif table_name == 'order_book_l2':
                    cols = "instId, ts, seqId, bids, asks" # Incluye bids y asks para mostrar como JSON
                elif table_name == 'candles':
                    cols = "instId, ts, open, high, low, close, vol, volCcy"
                else:
                    cols = ", ".join(primary_key_cols) + (f", {', '.join(extra_cols)}" if extra_cols else "")

                cursor.execute(f"SELECT {cols} FROM {table_name} ORDER BY ts DESC LIMIT 5")
                for row in cursor.fetchall():
                    # Intenta parsear bids/asks si son strings JSON
                    row_dict = dict(row)
                    if 'bids' in row_dict and isinstance(row_dict['bids'], str):
                        try:
                            row_dict['bids'] = json.loads(row_dict['bids'])
                        except json.JSONDecodeError:
                            pass # No es JSON válido, dejar como está
                    if 'asks' in row_dict and isinstance(row_dict['asks'], str):
                        try:
                            row_dict['asks'] = json.loads(row_dict['asks'])
                        except json.JSONDecodeError:
                            pass # No es JSON válido, dejar como está
                    print(row_dict)
            else:
                print(f"No hay registros en la tabla '{table_name}'.")
        else:
            print(f"¡ERROR: Tabla '{table_name}' NO encontrada en la base de datos!")
            return False
    except sqlite3.Error as e:
        print(f"Error de SQLite al verificar '{table_name}': {e}")
        return False
    except Exception as e:
        print(f"Error inesperado al verificar '{table_name}': {e}")
        return False
    return True

def main():
    if not DB_PATH.exists():
        print(f"¡ADVERTENCIA: El archivo de base de datos '{DB_PATH}' no existe! Asegúrate de que el bot lo haya creado.")
        return

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row # Permite acceder a las columnas por nombre
        cursor = conn.cursor()
        print(f"Conectado a la base de datos: {DB_PATH}")

        # Verifica la tabla 'tickers'
        verify_table(cursor, 'tickers', ['instId', 'ts'], ['idxPx', 'markPx', 'lastPx', 'bidPx', 'askPx'])

        # Verifica la tabla 'order_book_l2'
        verify_table(cursor, 'order_book_l2', ['instId', 'ts', 'seqId'], ['bids', 'asks'])

        # Verifica la tabla 'candles'
        verify_table(cursor, 'candles', ['instId', 'ts'], ['open', 'high', 'low', 'close', 'vol', 'volCcy'])

    except sqlite3.Error as e:
        print(f"Error general de SQLite: {e}")
    except Exception as e:
        print(f"Error inesperado en main: {e}")
    finally:
        if conn:
            conn.close()
            print("Conexión a la base de datos cerrada.")

if __name__ == "__main__":
    main()
