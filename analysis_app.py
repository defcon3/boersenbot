#!/usr/bin/env python3
from flask import Flask, render_template, jsonify, request
import pymssql
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

DB_CONFIG = {
    'server': '158.181.48.77',
    'database': 'dbdata',
    'user': '326773',
    'password': 'Extaler11!',
    'as_dict': False
}

def get_db():
    return pymssql.connect(**DB_CONFIG)

def get_available_symbols():
    try:
        conn = get_db()
        df = pd.read_sql("SELECT DISTINCT Symbol FROM bb_StockPrices_1min_Kaggle ORDER BY Symbol", conn)
        conn.close()
        return df['Symbol'].tolist() if not df.empty else []
    except:
        return ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA']

def get_date_range():
    try:
        conn = get_db()
        df = pd.read_sql("SELECT MIN([Timestamp]) as mn, MAX([Timestamp]) as mx FROM bb_StockPrices_1min_Kaggle", conn)
        conn.close()
        if not df.empty and df['mn'].iloc[0]:
            return df['mn'].iloc[0].strftime('%Y-%m-%d'), df['mx'].iloc[0].strftime('%Y-%m-%d')
    except:
        pass
    return '2023-05-15', '2023-05-23'

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal).mean()
    return macd, sig, macd - sig

def load_data(symbol, start_date=None, end_date=None):
    try:
        conn = get_db()
        where = f"Symbol = '{symbol}'"
        if start_date:
            where += f" AND [Timestamp] >= '{start_date}'"
        if end_date:
            where += f" AND [Timestamp] <= '{end_date} 23:59:59'"
        df = pd.read_sql(
            f"SELECT Timestamp, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume "
            f"FROM bb_StockPrices_1min_Kaggle WHERE {where} ORDER BY Timestamp",
            conn
        )
        conn.close()
        if df.empty:
            return None
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['RSI'] = compute_rsi(df['ClosePrice'])
        df['MACD'], df['Signal'], df['MACD_Hist'] = compute_macd(df['ClosePrice'])
        sma = df['ClosePrice'].rolling(20).mean()
        std = df['ClosePrice'].rolling(20).std()
        df['BB_Upper'] = sma + std * 2
        df['BB_Lower'] = sma - std * 2
        df['BB_Middle'] = sma
        return df
    except Exception as e:
        print(f"DB error: {e}")
        return None

def to_lw(ts):
    return int(ts.timestamp())

def series(df, col):
    mask = ~df[col].isna()
    return [{'time': to_lw(t), 'value': round(v, 4)}
            for t, v in zip(df.loc[mask, 'Timestamp'], df.loc[mask, col])]

def build_payload(df, show_rsi, show_macd, show_bb):
    up   = 'rgba(38,166,154,0.8)'
    down = 'rgba(239,83,80,0.8)'

    ohlcv = [
        {'time': to_lw(t), 'open': o, 'high': h, 'low': l, 'close': c}
        for t, o, h, l, c in zip(
            df['Timestamp'], df['OpenPrice'], df['HighPrice'],
            df['LowPrice'],  df['ClosePrice']
        )
    ]
    volume = [
        {'time': to_lw(t), 'value': v,
         'color': up if c >= o else down}
        for t, v, c, o in zip(
            df['Timestamp'], df['Volume'], df['ClosePrice'], df['OpenPrice']
        )
    ]

    out = {
        'ohlcv': ohlcv,
        'volume': volume,
        'stats': {
            'records': len(df),
            'from': str(df['Timestamp'].iloc[0])[:16],
            'to':   str(df['Timestamp'].iloc[-1])[:16],
            'avg_vol': int(df['Volume'].mean()),
        }
    }

    if show_bb:
        out['bb_upper']  = series(df, 'BB_Upper')
        out['bb_lower']  = series(df, 'BB_Lower')
        out['bb_middle'] = series(df, 'BB_Middle')

    if show_rsi:
        out['rsi'] = series(df, 'RSI')

    if show_macd:
        mask = ~df['MACD'].isna()
        out['macd']   = series(df, 'MACD')
        out['signal'] = series(df, 'Signal')
        out['macd_hist'] = [
            {'time': to_lw(t), 'value': round(v, 4),
             'color': 'rgba(38,166,154,0.6)' if v >= 0 else 'rgba(239,83,80,0.6)'}
            for t, v in zip(df.loc[mask, 'Timestamp'], df.loc[mask, 'MACD_Hist'])
        ]

    return out


@app.route('/')
def index():
    symbols = get_available_symbols()
    min_date, max_date = get_date_range()
    return render_template('analysis.html', symbols=symbols, min_date=min_date, max_date=max_date)

@app.route('/api/chart', methods=['POST'])
def api_chart():
    data = request.json
    df = load_data(data.get('symbol', 'AAPL'), data.get('start_date'), data.get('end_date'))
    if df is None:
        return jsonify({'error': 'Keine Daten verfügbar'}), 400
    return jsonify(build_payload(
        df,
        show_rsi=data.get('show_rsi', True),
        show_macd=data.get('show_macd', True),
        show_bb=data.get('show_bb', True)
    ))

@app.route('/api/table', methods=['POST'])
def api_table():
    data = request.json
    df = load_data(data.get('symbol', 'AAPL'), data.get('start_date'), data.get('end_date'))
    if df is None:
        return jsonify({'error': 'Keine Daten verfügbar'}), 400
    cols = ['Timestamp', 'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'Volume', 'RSI', 'MACD']
    df2 = df[cols].tail(data.get('limit', 100)).copy()
    df2['Timestamp'] = df2['Timestamp'].astype(str)
    df2 = df2.round(4)
    return jsonify({'columns': cols, 'data': df2.to_dict('records')})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)
