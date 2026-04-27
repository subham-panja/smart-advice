"""
5paisa API Client Utility
=========================
Uses py5paisa SDK with OAuth authentication.

Auth Flow (from official docs):
1. Create FivePaisaClient with cred dict
2. Get OAuth session via a request token from browser login
3. OR use a saved access token with set_access_token()

Margin method: client.margin()
"""

import logging
from py5paisa import FivePaisaClient
from config import FIVEPAISA_CONFIG

logger = logging.getLogger(__name__)


class FivePaisaUtility:
    """Utility class to interact with 5paisa API."""

    def __init__(self):
        self.client = None
        self.config = FIVEPAISA_CONFIG
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the 5paisa client using OAuth."""
        try:
            if not self.config.get('api_key') or not self.config.get('encry_key'):
                logger.warning("5paisa credentials missing in .env file")
                return

            # Build the cred dict exactly as the SDK expects (uppercase keys)
            cred = {
                "APP_NAME": self.config.get('app_name'),
                "APP_SOURCE": self.config.get('app_source'),
                "USER_ID": self.config.get('user_id'),
                "PASSWORD": self.config.get('password'),
                "USER_KEY": self.config.get('api_key'),
                "ENCRYPTION_KEY": self.config.get('encry_key'),
            }

            self.client = FivePaisaClient(cred=cred)

            access_token = self.config.get('access_token')
            client_code = self.config.get('client_code')

            if access_token:
                if client_code:
                    self.client.set_access_token(access_token, client_code)
                    logger.info("5paisa client authenticated via saved access token")
                else:
                    logger.warning(
                        "5paisa: access_token found but client_code is missing. "
                        "Add FIVEPAISA_CLIENT_CODE (your 5paisa account number) to .env"
                    )
            else:
                logger.info(
                    "5paisa client initialized. Use /5paisa_login on Telegram to authenticate."
                )

        except Exception as e:
            logger.error(f"Error initializing 5paisa client: {e}")
            self.client = None

    def get_oauth_url(self):
        """Return the OAuth login URL for the user to visit in a browser."""
        user_key = self.config.get('api_key')
        redirect_url = self.config.get('redirect_url')
        if not user_key:
            return None
        return (
            f"https://dev-openapi.5paisa.com/WebVendorLogin/VLogin/Index"
            f"?VendorKey={user_key}"
            f"&ResponseURL={redirect_url}"
        )

    def login_with_request_token(self, request_token):
        """Complete OAuth login using the request token from the browser redirect."""
        if not self.client:
            return {"status": "error", "message": "Client not initialized"}

        try:
            access_token = self.client.get_oauth_session(request_token)
            if not access_token:
                return {"status": "error", "message": "Failed to get access token. The token might have expired (they expire in 30 seconds). Try generating a new one and pasting it quickly."}
                
            logger.info("5paisa OAuth login successful")
            return {
                "status": "success",
                "access_token": access_token,
                "message": "Login successful! Save this access token to your .env file as FIVEPAISA_ACCESS_TOKEN"
            }
        except Exception as e:
            logger.error(f"5paisa OAuth login failed: {e}")
            return {"status": "error", "message": str(e)}

    def get_margin(self):
        """Fetch available margin/balance using client.margin()."""
        if not self.client:
            return {"status": "error", "message": "5paisa client not initialized. Set credentials in .env"}

        # Check if we actually have an access token and client code
        if not self.config.get('access_token'):
            return {
                "status": "error",
                "message": "Not authenticated. Use /5paisa_login on Telegram first to get an access token."
            }
            
        if not self.config.get('client_code'):
            return {
                "status": "error",
                "message": "Missing CLIENT_CODE. Please add FIVEPAISA_CLIENT_CODE to your .env file."
            }

        try:
            margin_data = self.client.margin()

            if margin_data is None:
                return {"status": "error", "message": "5paisa API timed out or returned no data. Please try again in a few seconds."}

            if isinstance(margin_data, list) and len(margin_data) > 0:
                # The SDK typically returns a list of segment margins
                equity_margin = margin_data[0]
                return {
                    "status": "success",
                    "available_margin": equity_margin.get('AvailableMargin', 0),
                    "utilized_margin": equity_margin.get('UtilizedMargin', 0),
                    "net_available": equity_margin.get('NetAvailableMargin', 0),
                    "ledger_balance": equity_margin.get('ALB', equity_margin.get('LedgerBalance', 0)),
                    "raw": equity_margin,
                }
            elif isinstance(margin_data, dict):
                return {"status": "success", "raw": margin_data}
            else:
                return {"status": "error", "message": f"Unexpected data format from 5paisa: {str(margin_data)}"}

        except Exception as e:
            logger.error(f"Error fetching 5paisa margin: {e}")
            return {"status": "error", "message": str(e)}

    def get_holdings(self):
        """Fetch current holdings from 5paisa."""
        if not self.client:
            return {"status": "error", "message": "5paisa client not initialized. Set credentials in .env"}

        if not self.config.get('access_token'):
            return {"status": "error", "message": "Not authenticated. Use /5paisa_login on Telegram first."}
            
        if not self.config.get('client_code'):
            return {"status": "error", "message": "Missing CLIENT_CODE. Please add FIVEPAISA_CLIENT_CODE to your .env file."}

        try:
            # client.holdings() typically returns a list of holding dicts
            holdings_data = self.client.holdings()
            
            if holdings_data is None:
                return {"status": "error", "message": "5paisa API timed out or returned no data."}
                
            return {"status": "success", "data": holdings_data}

        except Exception as e:
            logger.error(f"Error fetching 5paisa holdings: {e}")
            return {"status": "error", "message": str(e)}


def get_5paisa_balance():
    """Convenience function for the Telegram bot."""
    utility = FivePaisaUtility()
    return utility.get_margin()

def get_5paisa_holdings():
    """Convenience function for the Telegram bot to fetch holdings."""
    utility = FivePaisaUtility()
    return utility.get_holdings()
