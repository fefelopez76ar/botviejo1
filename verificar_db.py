import sqlite3

DB_PATH = "market_data.db"  # Asegúrate que es el mismo que usás en HistoricalDataSaver

def verificar_tickers():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM tickers")
    total = cursor.fetchone()[0]
    print(f"Total de registros en 'tickers': {total}")

    cursor.execute("SELECT timestamp, instrument_id, last_price FROM tickers ORDER BY timestamp DESC LIMIT 5")
    ultimos = cursor.fetchall()
    print("\nÚltimos 5 registros:")
    for row in ultimos:
        print(row)

    conn.close()

def verificar_order_book():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM order_book")
    total = cursor.fetchone()[0]
    print(f"\nTotal de registros en 'order_book': {total}")

    cursor.execute("SELECT timestamp, instrument_id, best_bid, best_ask FROM order_book ORDER BY timestamp DESC LIMIT 5")
    ultimos = cursor.fetchall()
    print("\nÚltimos 5 order books:")
    for row in ultimos:
        print(row)

    conn.close()

if __name__ == "__main__":
    verificar_tickers()
    verificar_order_book()
