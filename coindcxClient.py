import hmac
import hashlib
import time
import json
import logging
import requests
import socketio

# from Utils import setup_logger # Assuming you have a setup_logger function like in BgxClient

# Placeholder for setup_logger if not available, or use standard logging
# For simplicity, using basic logging here. Integrate your setup_logger if preferred.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoinDcxError(Exception):
    """Base exception class for CoinDCX API errors"""
    pass


class CoinDcxRequestError(CoinDcxError):
    """Exception raised for network-related errors"""

    def __init__(self, message):
        super().__init__(message)


class CoinDcxAPIError(CoinDcxError):
    """Exception raised for API response errors"""

    def __init__(self, response_data, status_code):
        self.status_code = status_code
        self.message = response_data.get('message', 'Unknown API error')
        # CoinDCX errors might not always have a 'code' field in the main response body
        # It's often in the message string or implied by HTTP status.
        super().__init__(f"API Error (status {self.status_code}): {self.message}")


class CoinDcxClient:

    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key.encode('utf-8')  # Secret key needs to be bytes for HMAC
        self.base_url = "https://api.coindcx.com"

    def _generate_signature(self, body_json_str):
        """Generates HMAC SHA256 signature for the request body."""
        return hmac.new(self.secret_key, body_json_str.encode('utf-8'), hashlib.sha256).hexdigest()

    def _send_request(self, method, path, body=None):
        # Generate timestamp
        timestamp = int(time.time() * 1000)
        
        # Add timestamp to the body
        if body is None:
            body = {}
        body['timestamp'] = timestamp
        
        # Convert body to JSON string with compact separators
        json_body = json.dumps(body, separators=(',', ':'))
        
        # Generate signature - don't encode secret_key if it's already bytes
        secret_key_bytes = self.secret_key if isinstance(self.secret_key, bytes) else self.secret_key.encode('utf-8')
        
        signature = hmac.new(
            secret_key_bytes,  # Use the already encoded secret key
            json_body.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Rest of the method remains the same...
        headers = {
            'Content-Type': 'application/json',
            'X-AUTH-APIKEY': self.api_key,
            'X-AUTH-SIGNATURE': signature
        }
        
        url = f"{self.base_url}{path}"
        print(f"Request URL: {url}")
        print(f"Request Headers: {headers}")
        print(f"Request Body: {json_body}")
        
        try:
            response = requests.request(
                method,
                url,
                data=json_body,
                headers=headers,
                timeout=30
            )
            
            print(f"Response Status: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            print(f"Response Content: {response.text}")
            
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                raise CoinDcxAPIError(
                    {'message': f'Invalid JSON response: {response.text}'},
                    response.status_code
                )
                
            if not (200 <= response.status_code < 300):
                error_msg = response_data.get('message', 'Unknown error')
                raise CoinDcxAPIError({'message': error_msg}, response.status_code)
                
            return response_data
            
        except requests.exceptions.RequestException as e:
            raise CoinDcxAPIError({'message': f'Request failed: {str(e)}'}, 500)

    def get_positions(self, page=1, size=10, pairs=None, position_ids=None, margin_currency_short_name=None):
        """
        Fetch positions by pairs or position IDs or margin currency
        
        Args:
            page (int, optional): Page number. Defaults to 1.
            size (int, optional): Number of records per page. Defaults to 10.
            pairs (str, optional): Comma-separated list of instrument pairs. Example: "B-BTC_USDT,B-ETH_USDT"
            position_ids (str, optional): Comma-separated list of position IDs.
            margin_currency_short_name (list, optional): List of margin currencies. Defaults to ["USDT"].
            
        Returns:
            list: List of position objects
            
        Raises:
            CoinDcxAPIError: If the API returns an error
            CoinDcxRequestError: If the request fails
            ValueError: If neither pairs, position_ids, nor margin_currency_short_name is provided
        """
            
        body = {
            "page": str(page),
            "size": str(size)
        }
        
        # Add optional parameters if provided
        if margin_currency_short_name is not None:
            body["margin_currency_short_name"] = margin_currency_short_name
        if pairs:
            body["pairs"] = pairs
        if position_ids:
            body["position_ids"] = position_ids
            
        # At least one of these parameters must be provided
        # https://api.coindcx.com/exchange/v1/derivatives/futures/positions
        if not any([pairs, position_ids, margin_currency_short_name is not None]):
            raise ValueError("Either 'pairs', 'position_ids', or 'margin_currency_short_name' must be provided")
            
        return self._send_request('POST', '/exchange/v1/derivatives/futures/positions', body)
        
    def get_futures_instrument(self, pair, margin_currency_short_name="USDT"):
        """
        Get instrument details for a futures pair.
        
        Args:
            pair (str): Trading pair (e.g., 'B-BTC_USDT')
            margin_currency_short_name (str): Margin currency ('USDT' or 'INR')
            
        Returns:
            dict: Instrument details
        """
        params = {
            'pair': pair,
            'margin_currency_short_name': margin_currency_short_name
        }
        return self._send_request('GET', '/exchange/v1/derivatives/futures/data/instrument', params)
        
    # ===========================================
    # Futures Market Data Endpoints
    # ===========================================

    def get_futures_trades(self, pair):
        """
        Get recent trades for a futures pair.
        
        Args:
            pair (str): Trading pair (e.g., 'B-BTC_USDT')
            
        Returns:
            list: List of recent trades
        """
        params = {'pair': pair}
        return self._send_request('GET', '/exchange/v1/derivatives/futures/data/trades', params)
        
    def exit_position(self, position_id: str) -> dict:
        """
        Close an existing position by its ID.
        
        Args:
            position_id (str): The ID of the position to close
            
        Returns:
            dict: Response from the API
            
        Example:
            >>> client = CoinDcxClient(api_key, secret_key)
            >>> result = client.exit_position("a8930056-49ff-11f0-8b16-b3af8a02018e")
            >>> print(result)
        """
        body = {
            "timestamp": int(time.time() * 1000),
            "id": position_id
        }
        
        return self._send_request(
            'POST',
            '/exchange/v1/derivatives/futures/positions/exit',
            body
        )

    def get_futures_orderbook(self, instrument, depth=50):
        """
        Get order book for a futures instrument.
        
        Args:
            instrument (str): Instrument name (e.g., 'B-BTC_USDT-futures')
            depth (int): Order book depth (10, 20, or 50)
            
        Returns:
            dict: Order book data
        """
        return self._send_request('GET', f'/market_data/v3/orderbook/{instrument}/{depth}', None)
    
    # ===========================================
    # Futures Trading Endpoints
    # ===========================================
    
    def create_futures_order(self, pair, side, order_type, quantity, price=None, client_order_id=None,
                        leverage=10, reduce_only=False, time_in_force='good_till_cancel',
                        stop_price=None, take_profit=None, stop_loss=None, margin_currency_short_name="INR"):
        """
        Create a new futures order.

        Args:
            pair (str): Trading pair (e.g., 'B-ETH_USDT')
            side (str): 'buy' or 'sell'
            order_type (str): 'market_order', 'limit_order', etc.
            quantity (float): Order quantity
            price (float, optional): Required for limit orders
            client_order_id (str, optional): Custom order ID
            leverage (int, optional): Leverage (default: 10)
            reduce_only (bool, optional): If True, order can only reduce position
            time_in_force (str, optional): 'good_till_cancel', 'immediate_or_cancel', 'fill_or_kill'
            stop_price (float, optional): For stop orders
            take_profit (float, optional): Take profit price
            stop_loss (float, optional): Stop loss price
            margin_currency_short_name (str, optional): 'INR' or 'USDT' (default: 'INR')

        Returns:
            dict: Order details
        """
        # Normalize types and values
        order_type = order_type.lower()
        side = side.lower()
        margin_currency_short_name = margin_currency_short_name.upper() if margin_currency_short_name else 'USDT'
        timestamp = int(time.time() * 1000)

        # Build request body according to CoinDCX API rules - using the nested "order" structure
        order = {
            "side": side,
            "pair": pair,
            "order_type": order_type,
            "total_quantity": str(quantity),
            "leverage": int(leverage),
            "hidden": False,
            "post_only": False
        }
        
        # Add margin currency
        if margin_currency_short_name:
            order["margin_currency_short_name"] = margin_currency_short_name
            
        # Add reduce_only if true
        if reduce_only:
            order["reduce_only"] = True
            
        # Include time_in_force for all orders (API will handle validation)
        if time_in_force:
            order["time_in_force"] = time_in_force
            
        # Only include price for non-market orders
        if order_type != 'market_order' and price is not None:
            order["price"] = str(price)
            
        # Only include stop_price for stop/take_profit orders
        if order_type in ['stop_limit', 'stop_market', 'take_profit_limit', 'take_profit_market'] and stop_price is not None:
            order["stop_price"] = str(stop_price)
            
        # Optional client order ID
        if client_order_id:
            order["client_order_id"] = str(client_order_id)
            
        # Take profit and stop loss for entry orders only
        if take_profit is not None:
            order["take_profit_price"] = float(take_profit)
            
        if stop_loss is not None:
            order["stop_loss_price"] = float(stop_loss)
            
        # Create the final body with timestamp and order object
        body = {
            "timestamp": timestamp,
            "order": order
        }

        print(f"Creating order with params: {body}")
        return self._send_request('POST', '/exchange/v1/derivatives/futures/orders/create', body)

    def cancel_futures_order(self, order_id=None, client_order_id=None):
        """
        Cancel a futures order.
        
        Args:
            order_id (str, optional): Order ID
            client_order_id (str, optional): Client order ID
            
        Returns:
            dict: Cancellation status
            
        Note: Either order_id or client_order_id must be provided
        """
        if not order_id and not client_order_id:
            raise ValueError("Either order_id or client_order_id must be provided")
            
        body = {}
        if order_id:
            body['order_id'] = order_id
        if client_order_id:
            body['client_order_id'] = client_order_id
            
        return self._send_request('POST', '/exchange/v1/derivatives/futures/orders/cancel', body)
    
    def get_futures_orders(self, status='open', side=None, page=1, size=10,
                          margin_currency_short_name=None):
        """
        Get futures orders with filters.
        
        Args:
            status (str): Order status ('open', 'filled', 'cancelled', etc.)
            side (str, optional): 'buy' or 'sell'
            page (int, optional): Page number
            size (int, optional): Items per page
            margin_currency_short_name (list, optional): Margin currency filter
            
        Returns:
            list: List of orders
        """
        body = {
            'status': status,
            'page': str(page),
            'size': str(size)
        }
        
        if side:
            body['side'] = side
        if margin_currency_short_name:
            body['margin_currency_short_name'] = margin_currency_short_name
            
        return self._send_request('POST', '/exchange/v1/derivatives/futures/orders', body)
    
    def get_futures_order_status(self, order_id=None, client_order_id=None):
        """
        Get status of a specific futures order.
        
        Args:
            order_id (str, optional): Order ID
            client_order_id (str, optional): Client order ID
            
        Returns:
            dict: Order status
            
        Note: Either order_id or client_order_id must be provided
        """
        if not order_id and not client_order_id:
            raise ValueError("Either order_id or client_order_id must be provided")
            
        body = {}
        if order_id:
            body['order_id'] = order_id
        if client_order_id:
            body['client_order_id'] = client_order_id
            
        return self._send_request('POST', '/exchange/v1/derivatives/futures/orders/status', body)
    
    def cancel_all_futures_orders(self, pair=None, side=None):
        """
        Cancel all open futures orders.
        
        Args:
            pair (str, optional): Trading pair to filter by
            side (str, optional): 'buy' or 'sell' to filter by side
            
        Returns:
            dict: Cancellation status
        """
        body = {}
        if pair:
            body['pair'] = pair
        if side:
            body['side'] = side
            
        return self._send_request('POST', '/exchange/v1/derivatives/futures/orders/cancel_all', body)
    
    def get_futures_trade_history(self, pair=None, from_id=None, limit=50,
                                from_timestamp=None, to_timestamp=None):
        """
        Get futures trade history.
        
        Args:
            pair (str, optional): Trading pair to filter by
            from_id (int, optional): Fetch trades with ID > from_id
            limit (int, optional): Number of trades to return (max 1000)
            from_timestamp (int, optional): Start timestamp in milliseconds
            to_timestamp (int, optional): End timestamp in milliseconds
            
        Returns:
            list: List of trades
        """
        body = {
            'limit': limit
        }
        
        if pair:
            body['pair'] = pair
        if from_id is not None:
            body['from_id'] = from_id
        if from_timestamp is not None:
            body['from_timestamp'] = from_timestamp
        if to_timestamp is not None:
            body['to_timestamp'] = to_timestamp
            
        return self._send_request('POST', '/exchange/v1/derivatives/futures/trades', body)
    
    def get_futures_balance(self):
        """
        Get futures account balance.
        
        Args:
            margin_currency_short_name (str, optional): Filter by margin currency
            
        Returns:
            list: List of futures wallet information
        """
        # Create timestamp for the request
        timestamp = int(round(time.time() * 1000))
        
        # Create request body with timestamp
        body = {
            "timestamp": timestamp
        }
        
        # Convert body to JSON with no whitespace
        json_body = json.dumps(body, separators=(',', ':'))
        
        # Generate signature
        signature = hmac.new(
            self.secret_key if isinstance(self.secret_key, bytes) else bytes(self.secret_key, encoding='utf-8'),
            json_body.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Prepare headers
        headers = {
            'Content-Type': 'application/json',
            'X-AUTH-APIKEY': self.api_key,
            'X-AUTH-SIGNATURE': signature
        }
        
        # Use the correct endpoint URL
        url = "https://api.coindcx.com/exchange/v1/derivatives/futures/wallets"
        
        # Send GET request directly without using _send_request
        response = requests.get(url, data=json_body, headers=headers)
        
        # Check for successful response
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Error fetching futures balance: {response.status_code} - {response.text}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def transfer_funds(self, amount, currency, from_account, to_account):
        """
        Transfer funds between spot and futures accounts.
        
        Args:
            amount (float): Amount to transfer
            currency (str): Currency to transfer (e.g., 'USDT')
            from_account (str): Source account ('spot' or 'futures')
            to_account (str): Destination account ('spot' or 'futures')
            
        Returns:
            dict: Transfer status
        """
        valid_accounts = {'spot', 'futures'}
        if from_account not in valid_accounts or to_account not in valid_accounts:
            raise ValueError("Account must be either 'spot' or 'futures'")
            
        body = {
            'amount': str(amount),
            'currency': currency.upper(),
            'from_account': from_account,
            'to_account': to_account
        }
        
        return self._send_request('POST', '/exchange/v1/derivatives/futures/transfer', body)

    # --- Public Endpoints (Example: Ticker) ---
    def get_ticker(self):
        """Fetches the ticker for all markets."""
        path = "/exchange/ticker"
        return self._send_request('GET', path)

    def get_markets(self):
        """Returns an array of strings of currently active markets."""
        path = "/exchange/v1/markets"
        return self._send_request('GET', path)

    def get_market_details(self):
        """Fetches detailed information for all available markets."""
        path = "/exchange/v1/markets_details"
        return self._send_request('GET', path)

    # --- Authenticated Endpoints (Example: Get Balances) ---
    def get_balance(self, currency=None):
        """Get account balance for a specific currency or all currencies"""
        body = {}
        if currency is not None:  # Only add currency to body if it's provided
            body['currency'] = str(currency).upper()
        return self._send_request('POST', '/exchange/v1/users/balances', body)

    def get_balances(self):
        """Retrieves account's balances.
        
        Returns:
            list: A list containing all account balances.
            
        Raises:
            CoinDcxError: If there's an error while fetching the balances.
        """
        path = "/exchange/v1/users/balances"
        # Create the body with timestamp as in the working example
        body = {
            "timestamp": int(time.time() * 1000)
        }
        # Convert to JSON string without spaces to match the working example
        json_body = json.dumps(body, separators=(',', ':'))
        
        try:
            # Generate signature with the exact JSON string
            signature = hmac.new(
                self.secret_key,
                json_body.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            headers = {
                'Content-Type': 'application/json',
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature
            }
            
            response = requests.post(
                f"{self.base_url}{path}",
                data=json_body,
                headers=headers
            )
            
            if not (200 <= response.status_code < 300):
                response_data = response.json() if response.text else {}
                raise CoinDcxAPIError(
                    response_data if response_data else {'message': response.text},
                    response.status_code
                )
                
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while fetching balances: {str(e)}")
            raise CoinDcxRequestError(f"Network error while fetching balances: {str(e)}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response: {response.text}")
            raise CoinDcxError(f"Invalid response format: {response.text}") from e
        except Exception as e:
            logger.error(f"Unexpected error while fetching balances: {str(e)}")
            raise CoinDcxError(f"Failed to fetch balances: {str(e)}") from e

    def get_user_info(self):
        """Retrieves user info.
        
        Returns:
            dict: User information including account details.
            
        Raises:
            CoinDcxError: If there's an error while fetching user info.
        """
        path = "/exchange/v1/users/info"
        
        try:
            # Create the body with timestamp as required by the API
            body = {
                "timestamp": int(time.time() * 1000)  # Current timestamp in milliseconds
            }
            
            # Convert to JSON string without spaces to match the API requirement
            json_body = json.dumps(body, separators=(',', ':'))
            
            # Generate signature with the exact JSON string
            signature = hmac.new(
                self.secret_key,
                json_body.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            headers = {
                'Content-Type': 'application/json',
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature
            }
            
            # Print debug information
            logger.debug(f"Sending request to: {self.base_url}{path}")
            logger.debug(f"Headers: {headers}")
            logger.debug(f"Request body: {json_body}")
            
            response = requests.post(
                f"{self.base_url}{path}",
                data=json_body,
                headers=headers,
                timeout=10  # Add timeout to prevent hanging
            )
            
            # Print response details for debugging
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            logger.debug(f"Response body: {response.text}")
            
            if not (200 <= response.status_code < 300):
                try:
                    response_data = response.json()
                    error_msg = response_data.get('message', response.text)
                except json.JSONDecodeError:
                    error_msg = response.text
                
                logger.error(f"API Error {response.status_code}: {error_msg}")
                raise CoinDcxAPIError(
                    {'message': error_msg},
                    response.status_code
                )
                
            return response.json()
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error while fetching user info: {str(e)}"
            logger.error(error_msg)
            raise CoinDcxRequestError(error_msg) from e
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse response: {e}"
            logger.error(error_msg)
            raise CoinDcxError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error while fetching user info: {str(e)}"
            logger.error(error_msg)
            raise CoinDcxError(error_msg) from e


    # --- Order Management Endpoints ---
    def create_order(self, market, side, order_type, total_quantity, price_per_unit=None, client_order_id=None, INR=False):
        """Places a new order on the exchange.
        
        Args:
            market (str): The market symbol (e.g., 'BTCINR', 'ETHUSDT')
            side (str): 'buy' or 'sell'
            order_type (str): 'limit_order' or 'market_order'
            total_quantity (float): The quantity to buy/sell
            price_per_unit (float, optional): Required for limit orders, must be None for market orders
            client_order_id (str, optional): A unique ID for the order
            
        Returns:
            dict: The order details from the exchange
            
        Raises:
            CoinDcxError: If the order parameters are invalid
        """
        
        path = "/exchange/v1/orders/create"
        
        # Input validation
        if order_type not in ['limit_order', 'market_order']:
            raise CoinDcxError("order_type must be either 'limit_order' or 'market_order'")
            
        if side.lower() not in ['buy', 'sell']:
            raise CoinDcxError("side must be either 'buy' or 'sell'")
            
        if order_type == 'limit_order' and price_per_unit is None:
            raise CoinDcxError("price_per_unit is required for limit orders")
            
        if order_type == 'market_order' and price_per_unit is not None:
            # For market orders, we should not include the price in the request
            price_per_unit = None
            
        # Prepare the request body
        body = {
            "market": market,
            "side": side.lower(),
            "order_type": order_type,
            "total_quantity": str(total_quantity)  # Convert to string as some APIs expect string values
        }
        
        # Add price only for limit orders
        if price_per_unit is not None and order_type == "limit_order":
            body["price_per_unit"] = str(price_per_unit)  # Convert to string
            
        # Add client order ID if provided
        if client_order_id:
            body["client_order_id"] = client_order_id

        if INR:
            body["margin_currency_short_name"] = "INR"
        
        return self._send_request('POST', path, body=body)

    def create_multiple_orders(self, orders_list):
        """Places multiple orders in a single API call."""
        # Each order in orders_list should be a dict like the 'body' in create_order
        # e.g., orders_list = [{"market": "BTCINR", "side": "buy", ...}, {...}]
        path = "/exchange/v1/orders/create_multiple"
        body = {"orders": orders_list}
        return self._send_request('POST', path, body=body)

    def get_order_status(self, order_id=None, client_order_id=None):
        """Fetches status of an order by its ID or client_order_id."""
        if not order_id and not client_order_id:
            raise ValueError("Either order_id or client_order_id must be provided.")
        path = "/exchange/v1/orders/status"
        body = {}
        if order_id:
            body["id"] = order_id
        if client_order_id:
            body["client_order_id"] = client_order_id
        return self._send_request('POST', path, body=body)

    def cancel_order(self, order_id=None, client_order_id=None):
        """Cancels an active order."""
        if not order_id and not client_order_id:
            raise ValueError("Either order_id or client_order_id must be provided.")
        path = "/exchange/v1/orders/cancel"
        body = {}
        if order_id:
            body["id"] = order_id
        if client_order_id:
            body["client_order_id"] = client_order_id
        return self._send_request('POST', path, body=body)

    def cancel_multiple_orders(self, order_ids=None, client_order_ids=None):
        """Cancels multiple active orders by their IDs or client_order_ids."""
        if not order_ids and not client_order_ids:
            raise ValueError("Either order_ids or client_order_ids must be provided.")
        path = "/exchange/v1/orders/cancel_by_ids"
        body = {}
        if order_ids:
            body["ids"] = order_ids  # Array of order_ids
        if client_order_ids:
            body["client_order_ids"] = client_order_ids  # Array of client_order_ids
        return self._send_request('POST', path, body=body)

    def edit_order_price(self, new_price, order_id=None, client_order_id=None):
        """Edits the price of an active order."""
        if not order_id and not client_order_id:
            raise ValueError("Either order_id or client_order_id must be provided.")
        path = "/exchange/v1/orders/edit"
        body = {"price_per_unit": new_price}
        if order_id:
            body["id"] = order_id
        if client_order_id:
            body["client_order_id"] = client_order_id
        return self._send_request('POST', path, body=body)

    def get_trade_history(self, symbol, limit=50, from_id=None, sort="desc", from_timestamp=None, to_timestamp=None):
        """Retrieves account's trade history."""
        path = "/exchange/v1/orders/trade_history"
        body = {
            "symbol": symbol,
            "limit": limit,
            "sort": sort
        }
        if from_id is not None:
            body["from_id"] = from_id
        if from_timestamp is not None:
            body["from_timestamp"] = from_timestamp
        if to_timestamp is not None:
            body["to_timestamp"] = to_timestamp
        return self._send_request('POST', path, body=body)


class CoinDcxWebSocketClient:

    def __init__(self, api_key=None, secret_key=None,
                 log_level=logging.INFO, log_to_console=True, log_to_file=False,
                 log_file="coindcx_ws.log"):
                 
        self.api_key = api_key
        self.secret_key = secret_key.encode('utf-8') if secret_key else None
        self.sio = socketio.Client(logger=True, engineio_logger=True)  # Enable socket.io logs for debugging
        self.ws_url = "wss://stream.coindcx.com"
        self.thread = None
        self.active = False
        self.subscriptions = []  # To keep track of what we are subscribed to

        # Basic logger setup, replace with your custom setup_logger if available
        self.logger = logging.getLogger("CoinDcx_WS")
        self.logger.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if log_to_console:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        if log_to_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        self._register_handlers()

    def _generate_auth_signature(self, channel_name="coindcx"):
        if not self.secret_key:
            self.logger.warning("Secret key not provided. Cannot generate auth signature.")
            return None
        body = {"channel": channel_name}
        json_body = json.dumps(body, separators=(',', ':'))
        signature = hmac.new(self.secret_key, json_body.encode('utf-8'), hashlib.sha256).hexdigest()
        return signature

    def _register_handlers(self):

        @self.sio.event
        def connect():
            self.logger.info(f"Successfully connected to {self.ws_url}")
            # Re-subscribe to channels upon connection/reconnection
            for sub_type, channel_name, callback in self.subscriptions:
                self._subscribe_to_channel(sub_type, channel_name, callback, is_reconnect=True)

        @self.sio.event
        def connect_error(data):
            self.logger.error(f"Connection failed: {data}")

        @self.sio.event
        def disconnect():
            self.logger.warning("Disconnected from WebSocket.")
            # Reconnection is typically handled by python-socketio client by default

        # Generic handler for subscribed events
        # CoinDCX uses specific event names like 'price-change', 'candlestick', 'depth-snapshot'
        # We will map these in the subscribe methods

    def _subscribe_to_channel(self, subscription_type, channel_name, user_callback, is_reconnect=False):
        payload = {'channelName': channel_name}
        if subscription_type == "private":  # e.g., 'coindcx' for account updates
            if not self.api_key or not self.secret_key:
                self.logger.error(f"API key and secret required for private channel {channel_name}")
                return
            signature = self._generate_auth_signature(channel_name)  # or a generic one if needed
            if not signature:
                return
            payload['authSignature'] = signature
            payload['apiKey'] = self.api_key
        
        # Add to subscriptions first
        subscription = (subscription_type, channel_name, user_callback)
        if subscription not in self.subscriptions:
            self.subscriptions.append(subscription)
        
        # Only emit if connected
        if self.sio.connected:
            self.logger.info(f"Joining channel: {channel_name} with payload: {payload}")
            self.sio.emit('join', payload)
        else:
            self.logger.warning(f"WebSocket not connected. Channel {channel_name} will be joined after connection.")
            # Connection handler will resubscribe when connected

    def start(self):
        if self.active:
            self.logger.warning("WebSocket client is already running.")
            return
        self.active = True
        self.logger.info(f"Starting WebSocket connection to {self.ws_url}")
        try:
            self.sio.connect(self.ws_url, transports=['websocket'])
            # self.sio.wait() # if running in a blocking way
            # For non-blocking, you might run sio.connect in a separate thread if it's not already async
            # The python-socketio client typically handles its own loop in a background thread.
        except socketio.exceptions.ConnectionError as e:
            self.logger.error(f"Failed to connect to WebSocket: {e}")
            self.active = False

    def stop(self):
        self.logger.info("Stopping WebSocket client...")
        self.active = False
        # Unsubscribe from all channels
        for _, channel_name, _ in self.subscriptions:
            self.logger.info(f"Leaving channel: {channel_name}")
            self.sio.emit('leave', {'channelName': channel_name})
        self.subscriptions = []
        self.sio.disconnect()
        self.logger.info("WebSocket client stopped.")

    def subscribe_to_ltp(self, instrument_id, callback):
        """ Subscribes to LTP (Last Traded Price) updates for a given instrument.
            instrument_id: e.g., 'B-BTC_USDT'
            callback: function to call with new trade data
        """
        # According to docs, LTP can be from '@prices-futures' or '@trades-futures'
        # Using '@prices-futures' as it seems more direct for LTP
        channel_name = f"{instrument_id}@prices-futures"
        event_name = 'price-change'  # Event name for this channel
        
        @self.sio.on(event_name)
        def _on_price_change(data):
            # Filter by channel if multiple price-change events are handled by one handler
            # Or ensure specific handlers for specific channels if socketio allows that easily
            # For now, assuming this handler is specific enough or callback filters
            if data.get('channel', '').startswith(instrument_id):  # Basic check
                callback(data)
            elif 's' in data and data['s'] == instrument_id:  # For 'new-trade' like structure
                callback(data)
            elif 'data' in data and isinstance(data['data'], dict) and data['data'].get('s') == instrument_id:
                callback(data['data'])
            # Fallback if channel is not in data, but we are subscribed to this specific event for this instrument
            # This part might need refinement based on actual message structure for price-change
            elif not data.get('channel') and not data.get('s'):
                # If the event is 'price-change' and we subscribed to this instrument's price channel,
                # it's likely for this instrument. This assumption needs verification.
                self.logger.debug(f"Received {event_name} for {instrument_id} (no explicit symbol in root): {data}")
                callback(data)  # Pass full data if symbol not directly in root

        self._subscribe_to_channel("public", channel_name, callback)
        self.logger.info(f"Subscribed to LTP for {instrument_id} on channel {channel_name}, event {event_name}")

    def subscribe_to_candles(self, instrument_id, interval, callback):
        """ Subscribes to candlestick data.
            instrument_id: e.g., 'B-BTC_USDT'
            interval: e.g., '1m', '5m', '1h'
            callback: function to call with new candle data
        """
        channel_name = f"{instrument_id}_{interval}-futures"
        event_name = 'candlestick'  # Event name for this channel

        @self.sio.on(event_name)
        def _on_candlestick(data):
            # Filter by channel to ensure the callback is for the correct subscription
            if data.get('channel') == channel_name:
                callback(data.get('data'))  # The actual candle data is in 'data' field of the response
        
        self._subscribe_to_channel("public", channel_name, callback)
        self.logger.info(f"Subscribed to {interval} candles for {instrument_id} on channel {channel_name}, event {event_name}")

    # Add other subscription methods like subscribe_to_orderbook, subscribe_to_account_updates etc.

# Example Usage for WebSocket (add this to your __main__ or a separate test script):
# if __name__ == '__main__':
#     # ... (existing CoinDcxClient example usage) ...

#     def handle_ltp(data):
#         print(f"LTP Update: {data}")

#     def handle_candles(data):
#         print(f"Candle Update: {data}")

#     # WebSocket Client Example
#     # Public channels don't need API key/secret for subscription itself, but some might for initial connection handshake if API implies
#     ws_client = CoinDcxWebSocketClient()
#     ws_client.start()

#     # Wait for connection to establish - in a real app, handle this more gracefully
#     import time
#     time.sleep(5) 

#     if ws_client.sio.connected:
#         # Example: Subscribe to LTP for BTC-USDT (Instrument ID format might vary, check API docs for exact format)
#         # From docs: B-ID_USDT, so likely B-BTC_USDT
#         ws_client.subscribe_to_ltp("B-BTC_USDT", handle_ltp)
        
#         # Example: Subscribe to 1-minute candles for ETH-USDT
#         ws_client.subscribe_to_candles("B-ETH_USDT", "1m", handle_candles)
        
#         # Keep the main thread alive to receive messages
#         try:
#             while True:
#                 time.sleep(1)
#         except KeyboardInterrupt:
#             print("User interrupted. Stopping WebSocket client...")
#         finally:
#             ws_client.stop()
#     else:
#         print("Failed to connect to WebSocket server.")
