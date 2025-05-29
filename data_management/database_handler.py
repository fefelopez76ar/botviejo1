import sqlite3
from pathlib import Path

class DatabaseHandler:
    def __init__(self, db_path):
        self.db_path = Path(db_path)

    def connect(self):
        """Establish a connection to the SQLite database."""
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()

    def close(self):
        """Close the connection to the SQLite database."""
        if hasattr(self, 'connection'):
            self.connection.close()

    def execute_query(self, query, params=None):
        """Execute a query on the database."""
        if params is None:
            params = []
        self.cursor.execute(query, params)
        self.connection.commit()

    def fetch_all(self, query, params=None):
        """Fetch all results from a query."""
        if params is None:
            params = []
        self.cursor.execute(query, params)
        return self.cursor.fetchall()

    def initialize_schema(self):
        """Initialize the database schema if it doesn't exist."""
        schema = """
        CREATE TABLE IF NOT EXISTS tickers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            instrument_id TEXT NOT NULL,
            last_price REAL,
            best_bid REAL,
            best_ask REAL
        );
        """
        self.execute_query(schema)

# Example usage
if __name__ == "__main__":
    db_handler = DatabaseHandler("market_data.db")
    db_handler.connect()
    db_handler.initialize_schema()
    print("Database initialized and ready.")
    db_handler.close()
