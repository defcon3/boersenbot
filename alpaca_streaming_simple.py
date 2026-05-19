import json
import pymssql
from config import DB_CONFIG, ALPACA_CONFIG
import timeutil
import logging
import time
import threading
import websocket

# Setup logging
logging.basicConfig(
    filename='logs/alpaca_streaming.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
ws = None
running = True

def get_db_connection():
    return pymssql.connect(**DB_CONFIG, autocommit=True)

def insert_bar(symbol, timestamp, open_p, high, low, close, volume):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO bb_StockPrices_1min_Streaming (Symbol, Timestamp, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume) VALUES (%s, %s, %s, %s, %s, %s, %s)',
            (symbol, timestamp, float(open_p), float(high), float(low), float(close), int(volume))
        )
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        if 'Duplicate' not in str(e):
            logging.error('Insert ' + symbol + ': ' + str(e)[:60])
        return False

def on_message(ws, message):
    try:
        data = json.loads(message)
        if isinstance(data, dict):
            data = [data]
        for item in data:
            if not isinstance(item, dict):
                continue
            t = item.get('T')
            if t == 'b':  # bar message
                ts_str = item.get('t')
                if ts_str:
                    ts = timeutil.parse_alpaca_ts(ts_str)
                    insert_bar(
                        item.get('S'), ts,
                        item.get('o'), item.get('h'),
                        item.get('l'), item.get('c'),
                        item.get('v')
                    )
            elif t == 'error':
                logging.error('Stream error msg: ' + str(item)[:120])
            elif t == 'success':
                logging.info('Stream: ' + str(item.get("msg")))
    except Exception as e:
        logging.error('Message error: ' + str(e)[:80])

def on_error(ws, error):
    logging.error('WebSocket error: ' + str(error)[:100])

def on_close(ws, close_status_code, close_msg):
    logging.warning('WebSocket closed. Reconnecting...')
    time.sleep(5)
    start_stream()

def on_open(ws):
    logging.info('WebSocket connected')
    
    auth = {
        'action': 'auth',
        'key': ALPACA_CONFIG['api_key'],
        'secret': ALPACA_CONFIG['secret_key']
    }
    ws.send(json.dumps(auth))
    time.sleep(0.5)
    
    subscribe = {
        'action': 'subscribe',
        'bars': symbols
    }
    ws.send(json.dumps(subscribe))
    logging.info('Subscribed to ' + str(symbols))

def start_stream():
    global ws
    try:
        ws = websocket.WebSocketApp(
            'wss://stream.data.alpaca.markets/v2/iex',
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        ws.run_forever(ping_interval=30)
    except Exception as e:
        logging.error('Connection error: ' + str(e))
        time.sleep(5)
        start_stream()

if __name__ == '__main__':
    logging.info('=== Alpaca Streaming Started ===')
    print('Starting Alpaca WebSocket Streaming...')
    try:
        start_stream()
    except KeyboardInterrupt:
        logging.info('Stopped by user')
        print('Stopped')
