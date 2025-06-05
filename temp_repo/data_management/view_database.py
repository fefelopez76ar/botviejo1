import sqlite3
from pathlib import Path

def view_database(db_path):
    """View the contents of the SQLite database."""
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"Database file {db_path} does not exist.")
        return

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    if not tables:
        print("No tables found in the database.")
        return

    print("Tables in the database:")
    for table in tables:
        print(f"- {table[0]}")

    # View contents of each table
    for table in tables:
        print(f"\nContents of table {table[0]}:")
        cursor.execute(f"SELECT * FROM {table[0]};")
        rows = cursor.fetchall()
        for row in rows:
            print(row)

    connection.close()

if __name__ == "__main__":
    db_path = "c:\\proyectos\\mibot2\\CryptoTradingBot\\market_data.db"
    view_database(db_path)
