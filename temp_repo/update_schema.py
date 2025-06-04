import sqlite3

DB_PATH = "market_data.db"

def add_timestamp_column():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Agregar la columna 'timestamp' si no existe
        cursor.execute("ALTER TABLE tickers ADD COLUMN timestamp INTEGER")
        print("Columna 'timestamp' agregada exitosamente a la tabla 'tickers'.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("La columna 'timestamp' ya existe en la tabla 'tickers'.")
        else:
            print(f"Error al agregar la columna 'timestamp': {e}")
    finally:
        conn.commit()
        conn.close()

if __name__ == "__main__":
    add_timestamp_column()
