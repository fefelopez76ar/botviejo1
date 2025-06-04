
from typing import Dict, Any
import requests
import logging

logger = logging.getLogger(__name__)

class ExchangeClient:
    def __init__(self, api_key: str = None, secret: str = None, passphrase: str = None, is_demo: bool = True):
        self.api_key = api_key
        self.secret = secret 
        self.passphrase = passphrase
        self.is_demo = is_demo
        self.base_url = "https://www.okx.com"
        self.api_url = f"{self.base_url}/{'api/v5-demo' if is_demo else 'api/v5'}"
        
    def get_current_price(self, symbol: str) -> float:
        """Obtiene precio actual usando solo API privada"""
        try:
            endpoint = f"{self.api_url}/market/ticker/{symbol}"
            headers = self._get_auth_headers()
            response = requests.get(endpoint, headers=headers, timeout=2)
            data = response.json()
            return float(data['last'])
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            raise
            
    def _get_auth_headers(self) -> Dict[str, str]:
        """Genera headers de autenticación"""
        if not all([self.api_key, self.secret, self.passphrase]):
            raise ValueError("API credentials not configured")
            
        return {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": self._sign_request(),
            "OK-ACCESS-TIMESTAMP": str(int(time.time() * 1000)),
            "OK-ACCESS-PASSPHRASE": self.passphrase
        }
