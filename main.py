from app import app  # noqa: F401

"""
Solana Trading Bot - Main Application
"""

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
