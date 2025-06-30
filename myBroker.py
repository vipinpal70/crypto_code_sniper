from enum import Enum
from coindcxClient import CoinDcxClient



##! helper function 
def is_active_broker(broker_name:str=None) -> bool:
    if not broker_name:
        raise ValueError(f"Broker Name is empty")
    if broker_name not in ActiveBrokers.__members__:
        raise ValueError(f"Broker {broker_name} is not active")

    return True

def is_valid_credentials(broke_name:str,credentials:dict) -> bool:
    if broke_name == "CoinDCX":
        return ActiveBrokers.CoinDcx_requirement(credentials)
    
    return False



class ActiveBrokers(Enum):
    CoinDCX = "CoinDCX"

    @staticmethod
    def CoinDcx_requirement(credentials: dict = {}) -> None:
        """
        This checks whether the credentials have api_key and secret_key
        """
        if 'api_key' not in credentials or 'secret_key' not in credentials:
            raise ValueError("Credentials must contain 'api_key' and 'secret_key'")
        
        return True


class BrokerConfiguration:
    """
    This is for finding and checking each broke requirement to connect with them

    Raises:
        ValueError: if broker setup not in your project 
    """

    def __init__(self,user_name:str,broker_name:str,credentials:dict):
        """
        Parameters:
            user_name (str): User name
            broker_name (str): Broker name
            credentials (dict): Broker credentials
        """
        self.user_name = user_name
        self.broker_name = broker_name
        self.credentials = credentials
        

    def set_client(self):
        if self.broker_name == "CoinDCX":
            self.client = CoinDcxClient(api_key=self.credentials.get('api_key'),secret_key=self.credentials.get('secret_key'))
            
        


class Broker:
    """
        This is a Interface class for all broker
        
    """
    def __init__(self,user_name:str,broker_name:str=None,credentials:dict=None) -> None:
        
        if not user_name:
            raise ValueError(f"user_name is empty")
        
        if not broker_name:
            raise ValueError(f"broker_name is empty")
        
        if not is_active_broker(broker_name):
            raise ValueError(f"Broker {broker_name} is not active")
        
        if not is_valid_credentials(broker_name,credentials):
            raise ValueError(f"Broker {broker_name} credentials are not valid")
        
        
        self.user_name = user_name
        self.broker_name = broker_name
        self.broker_config = BrokerConfiguration(user_name,broker_name,credentials)
        
        self.broker_config.set_client()
        
        self.broker =  self.broker_config.client
        
        
    def get_balance(self):
        """
        Get the balance of user from the broker
        """
        if not hasattr(self.broker, 'get_futures_balance'):
            raise NotImplementedError(f"Broker {self.broker_name} does not support get_futures_balance")
        
        return self.broker.get_futures_balance()
    
    
    def send_entry_order(self, order: dict):
        """
        Send an order to the broker (currently supports CoinDCX).
        
        Args:
            order (dict): Must contain 'symbol', 'quantity', 'price', 'side'
        
        Raises:
            ValueError: On missing/invalid fields or unsupported broker
            NotImplementedError: If broker doesn't implement required method
        """
        
        if not isinstance(order, dict) or not order:
            raise ValueError("Order must be a non-empty dictionary")

        required_keys = {'symbol', 'quantity', 'price', 'side'}
        if not required_keys.issubset(order):
            missing = required_keys - order.keys()
            raise ValueError(f"Missing order keys: {', '.join(missing)}")
        
        if order['side'] not in ['buy', 'sell']:
            raise ValueError("Order side must be 'buy' or 'sell'")

        if not isinstance(order['quantity'], (int, float)) or order['quantity'] <= 0:
            raise ValueError("Order quantity must be a positive number")
        
        if not isinstance(order['price'], (int, float)) or order['price'] <= 0:
            raise ValueError("Order price must be a positive number")

        if not hasattr(self.broker, 'create_futures_order'):
            raise NotImplementedError(f"Broker {self.broker_name} does not support create_futures_order")
        
        if self.broker_name == "CoinDCX":
            return self.broker.create_futures_order(**order)
        
        else:
            raise NotImplementedError(f"Broker {self.broker_name} does not implement create_futures_order method")
    

    def order_status(self, order_id: str):
        """
        Get the status of an entry order by its ID.
        
        Args:
            order_id (str): The ID of the order to check
        
        Raises:
            ValueError: If order_id is not provided or invalid
            NotImplementedError: If broker doesn't implement required method
        """
        
        if not order_id:
            raise ValueError("Order ID must be provided")
        
        if not isinstance(order_id, str):
            raise ValueError("Order ID must be a string")
        
        if self.broker_name == "CoinDCX":
            return self.broker.get_futures_order_status(order_id)
        
        else:  
            raise NotImplementedError(f"Broker {self.broker_name} does not implement get_futures_order_status method")
        
        
        
    def stop_loss_order(self,order: dict):
        """
        Send a stop loss order to the broker.
        
        Args:
            order (dict): Must contain 'symbol', 'quantity', 'price', 'side', 'trigger_price'
        
        Raises:
            ValueError: On missing/invalid fields or unsupported broker
            NotImplementedError: If broker doesn't implement required method
        """
        required_keys = {'symbol', 'quantity', 'price', 'side', 'trigger_price'}
        if not isinstance(order, dict) or not order:
            raise ValueError("Order must be a non-empty dictionary")

        if not required_keys.issubset(order):
            missing = required_keys - order.keys()
            raise ValueError(f"Missing order keys: {', '.join(missing)}")
        
        if order['side'] not in ['buy', 'sell']:
            raise ValueError("Order side must be 'buy' or 'sell'")
        
        if not isinstance(order['quantity'], (int, float)) or order['quantity'] <= 0:
            raise ValueError("Order quantity must be a positive number")
        
        if not isinstance(order['price'], (int, float)) or order['price'] <= 0:
            raise ValueError("Order price must be a positive number")
        
        if not isinstance(order['trigger_price'], (int, float)) or order['trigger_price'] <= 0:
            raise ValueError("Trigger price must be a positive number")
        
        if 'order_type' not in order:
            order['order_type'] = 'stop_market'
            
        if self.broker_name == "CoinDCX":
            return self.broker.create_futures_order(**order)
        
        else:
            raise NotImplementedError(f"Broker {self.broker_name} does not implement")
    
    
    def exit_trade(self, order: dict):
        """
        Send an exit trade order to the broker.
        
        Args:
            order (dict): Must contain 'symbol', 'quantity', 'price', 'side'
        
        Raises:
            ValueError: On missing/invalid fields or unsupported broker
            NotImplementedError: If broker doesn't implement required method
        """
        
        if not isinstance(order, dict) or not order:
            raise ValueError("Order must be a non-empty dictionary")

        required_keys = {'symbol', 'quantity', 'price', 'side'}
        if not required_keys.issubset(order):
            missing = required_keys - order.keys()
            raise ValueError(f"Missing order keys: {', '.join(missing)}")
        
        if order['side'] not in ['buy', 'sell']:
            raise ValueError("Order side must be 'buy' or 'sell'")
        
        if not isinstance(order['quantity'], (int, float)) or order['quantity'] <= 0:
            raise ValueError("Order quantity must be a positive number")
        
        if not isinstance(order['price'], (int, float)) or order['price'] <= 0:
            raise ValueError("Order price must be a positive number")
        
        if self.broker_name == "CoinDCX":
            return self.broker.create_futures_order(**order)
        else:
            raise NotImplementedError(f"Broker {self.broker_name} does not implement create_futures_order method for exit trades")
        
        
    