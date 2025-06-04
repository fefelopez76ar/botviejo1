import sqlite3

def initialize_database():
    db_path = "market_data.db"
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # Crear tabla 'tickers' si no existe
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Crear tabla 'order_book' si no existe
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS order_book (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            bid_price REAL NOT NULL,
            ask_price REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    connection.commit()
    connection.close()
    print("Tablas inicializadas correctamente.")

if __name__ == "__main__":
    initialize_database()
