import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from typing import List, Dict, Any
from pymongo import MongoClient

from myBroker import Broker


# Database configuration 
MONGO_URL = "mongodb+srv://vipinpal7060:lRKAbH2D7W18LMZd@cluster0.fg30pmw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "CryptoSniper"
TRADES_COLLECTION = "trades"
USERS_COLLECTION = "users"
MONGO_CLIENT = MongoClient(host=MONGO_URL)
DB = MONGO_CLIENT[DB_NAME]
TRADE = DB[TRADES_COLLECTION]
USERS = DB[USERS_COLLECTION]





def get_user_by_strategy(strategy: str=None) -> list:
    """
    Get users who have a specific strategy active.
    Args:
        strategy (str): The strategy to filter users by.
    Returns:
        list: A list of user documents with the specified strategy active.
    """
    if not strategy:
        raise ValueError("Strategy cannot be empty or None.")
    if not isinstance(strategy, str):
        raise TypeError("Strategy must be a string.")
    
    # Query to find users with the specified strategy active
    # and with api_verified == True and broker_connection exists
    query = {
        "status": "Approved",
        "api_verified": True,
        "broker_connection": {"$exists": True},
        f"strategies.{strategy}.status": "active"
    }
    
    return list(USERS.find(query, {"_id": 0, "email": 1, "broker_connection": 1}))


def get_user_broker_set(users:list[dict] | None = None) -> list:
    """ Get the user broker credentials data like api key and secret
        make a broker class object using this data

    Args:
        users (list[dict] | None, optional): _description_. Defaults to None.
        
    Return:
        list of user broker class object
        
    """
    
    result_list = []
    for u in users:
        api_key = u.get('broker_connection', {}).get('api_key')
        secret_key = u.get('broker_connection', {}).get('secret_key')
        
        if not api_key or not secret_key:
            print(f"Skipping user {u.get('email')} due to missing broker credentials.")
            continue
        
        broker_name = u.get('broker_connection', {}).get('broker_name')
        
        result_list.append(Broker(user_name=u.get('email'),broker_name=broker_name,credentials={"api_key":api_key,"secret_key":secret_key}))
        
    
    print(f"Found {len(result_list)} active users with broker connections.")

    return result_list



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trade_processing.log')
    ]
)


logger = logging.getLogger('trade_processor')



async def send_order_async(broker, order_params):
    """
    Asynchronous wrapper for sending orders to brokers
    
    Args:
        broker: Broker instance
        order_params: Dictionary containing order parameters
    
    Returns:
        Tuple of (broker.user_name, result or error)
    """
    try:
        if order_params.get('type') == 'entry':
            result = broker.send_entry_order(order_params['order'])
            
        elif order_params.get('type') == 'stop_loss':
            result = broker.stop_loss_order(order_params['order'])
            
        elif order_params.get('type') == 'exit':
            result = broker.exit_trade(order_params['order'])
            
        else:
            raise ValueError(f"Unknown order type: {order_params.get('type')}")
        
        return broker.user_name, result
    
    except Exception as e:
        logger.error(f"Error sending order for user {broker.user_name}: {str(e)}")
        return broker.user_name, {'error': str(e)}


def prepare_order_params(trade: dict, broker: Broker) -> dict:
    """
    Prepare order parameters for a specific broker based on the trade
    
    Args:
        trade: Trade document from MongoDB
        broker: Broker instance
    
    Returns:
        Dictionary with order parameters
    """
    # Extract common parameters from trade
    symbol = trade.get('symbol')
    price = trade.get('price')
    side = trade.get('side')
    pair = "B-" + symbol.replace("-", "_")
    
    quantity = trade.get('quantity')
    
    order_params = {
        'type': trade.get('type'),
        'order': {
            'pair': pair,
            'quantity': quantity,
            'price': price,
            'side': side
        }
    }
    
    # Add trigger_price for stop_loss orders
    if trade.get('type') == 'stop_loss' and 'trigger_price' in trade:
        order_params['order']['trigger_price'] = trade.get('trigger_price')
        order_params['order']['order_type'] = 'stop_market'
    
    return order_params


async def process_orders_async(brokers: List[Broker], trade: dict) -> Dict[str, Any]:
    """
    Process orders asynchronously for multiple brokers
    
    Args:
        brokers: List of broker instances
        trade: Trade document from MongoDB
    
    Returns:
        Dictionary with results for each broker
    """
    start_time = time.time()
    tasks = []
    loop = asyncio.get_event_loop()
    
    # Create a thread pool for CPU-bound tasks
    with ThreadPoolExecutor(max_workers=min(32, len(brokers))) as executor:
        for broker in brokers:
            # Prepare order parameters for this specific broker
            order_params = prepare_order_params(trade, broker)
            
            # Submit the task to the thread pool
            task = loop.run_in_executor(
                executor,
                lambda b=broker, op=order_params: send_order_async(b, op)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    success_count = 0
    error_count = 0
    results_dict = {}
    
    for result in results:
        if isinstance(result, Exception):
            error_count += 1
            logger.error(f"Exception during order processing: {str(result)}")
        else:
            user_name, order_result = result
            results_dict[user_name] = order_result
            if 'error' in order_result:
                error_count += 1
            else:
                success_count += 1
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processed {len(brokers)} orders in {elapsed_time:.2f} seconds. Success: {success_count}, Errors: {error_count}")
    
    return {
        'elapsed_time': elapsed_time,
        'total_orders': len(brokers),
        'success_count': success_count,
        'error_count': error_count,
        'results': results_dict
    }

def process_trade(trade: dict):
    """
    Process a trade document from the MongoDB collection.
    Args:
        trade (dict): The trade document to process.
    """
    
    logger.info(f"Processing trade: {trade}")
    
    # Check trade type: entry, stop_loss, or exit
    trade_type = trade.get('type')
    if not trade_type:
        logger.error(f"Trade type is missing in the trade document: {trade}")
        return
    
    if trade_type == 'entry':
        logger.info(f"Entry trade detected: {trade}")
        
        # Get the strategy from the trade document
        trade_strategy = trade.get('strategy', None)
        if not trade_strategy:
            logger.error(f"Strategy is missing in the trade document: {trade}")
            return
        
        # Get users who have this strategy active
        users = get_user_by_strategy(strategy=trade_strategy)
        
        # Get broker objects for these users
        users_broker = get_user_broker_set(users=users)
        if not users_broker:
            logger.warning(f"No active users found for strategy: {trade_strategy}")
            return
        
        # Process the orders asynchronously
        logger.info(f"Processing orders for {len(users_broker)} users with strategy {trade_strategy}")
        
        # Run the async function in the event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(process_orders_async(users_broker, trade))
        
        # Log the results
        logger.info(f"Order processing complete. Processed {results['total_orders']} orders in {results['elapsed_time']:.2f} seconds.")
        logger.info(f"Success: {results['success_count']}, Errors: {results['error_count']}")
        
    elif trade_type == 'stop_loss':
        logger.info(f"Stop Loss trade detected: {trade}")
        
        # Similar implementation as entry trade
        trade_strategy = trade.get('strategy', None)
        if not trade_strategy:
            logger.error(f"Strategy is missing in the trade document: {trade}")
            return
        
        users = get_user_by_strategy(strategy=trade_strategy)
        users_broker = get_user_broker_set(users=users)
        
        if not users_broker:
            logger.warning(f"No active users found for strategy: {trade_strategy}")
            return
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(process_orders_async(users_broker, trade))
        logger.info(f"Stop loss order processing complete. Processed {results['total_orders']} orders in {results['elapsed_time']:.2f} seconds.")
        
    elif trade_type == 'exit':
        logger.info(f"Exit trade detected: {trade}")
        
        # Similar implementation as entry trade
        trade_strategy = trade.get('strategy', None)
        if not trade_strategy:
            logger.error(f"Strategy is missing in the trade document: {trade}")
            return
        
        users = get_user_by_strategy(strategy=trade_strategy)
        users_broker = get_user_broker_set(users=users)
        
        if not users_broker:
            logger.warning(f"No active users found for strategy: {trade_strategy}")
            return
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(process_orders_async(users_broker, trade))
        logger.info(f"Exit order processing complete. Processed {results['total_orders']} orders in {results['elapsed_time']:.2f} seconds.")




def watch_trades():
    """
    Watch trades in the MongoDB collection and process them.
    """
    with TRADE.watch() as stream:
        for change in stream:
            if change['operationType'] == 'insert':
                trade = change['fullDocument']
                process_trade(trade)


