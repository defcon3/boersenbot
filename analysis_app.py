#!/usr/bin/env python3
"""
Interactive Kaggle Data Analysis Dashboard
Flask app with Plotly charts for 1-min trading data exploration
"""

from flask import Flask, render_template, jsonify, request
import pymssql
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json

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
    """Get list of symbols in Kaggle data"""
    try:
        conn = get_db()
        query = "SELECT DISTINCT Symbol FROM bb_StockPrices_1min_Kaggle ORDER BY Symbol"
        df = pd.read_sql(query, conn)
        conn.close()
        return df['Symbol'].tolist() if not df.empty else []
    except:
        return ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA']

def get_date_range():
    """Get min/max dates in Kaggle data"""
    try:
        conn = get_db()
        query = """
            SELECT MIN([Timestamp]) as min_date, MAX([Timestamp]) as max_date
            FROM bb_StockPrices_1min_Kaggle
        """
        df = pd.read_sql(query, conn)
        conn.close()
        if not df.empty:
            min_date = df['min_date'].iloc[0]
            max_date = df['max_date'].iloc[0]
            return min_date.strftime('%Y-%m-%d'), max_date.strftime('%Y-%m-%d')
    except:
        pass
    return '2023-05-15', '2023-05-23'

def get_kaggle_data(symbol, start_date=None, end_date=None):
    """Load Kaggle data for symbol and date range"""
    try:
        conn = get_db()

        if start_date and end_date:
            query = f"""
                SELECT Timestamp, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume
                FROM bb_StockPrices_1min_Kaggle
                WHERE Symbol = '{symbol}'
                  AND [Timestamp] >= '{start_date}'
                  AND [Timestamp] <= '{end_date}'
                ORDER BY Timestamp
            """
        else:
            query = f"""
                SELECT Timestamp, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume
                FROM bb_StockPrices_1min_Kaggle
                WHERE Symbol = '{symbol}'
                ORDER BY Timestamp
            """

        df = pd.read_sql(query, conn)
        conn.close()

        if df.empty:
            return None

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Compute indicators
        df['RSI'] = compute_rsi(df['ClosePrice'], 14)
        df['MACD'], df['Signal'], df['MACD_Hist'] = compute_macd(df['ClosePrice'])

        # Bollinger Bands
        sma = df['ClosePrice'].rolling(20).mean()
        std = df['ClosePrice'].rolling(20).std()
        df['BB_Upper'] = sma + (std * 2)
        df['BB_Lower'] = sma - (std * 2)
        df['BB_Middle'] = sma

        # Volume
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def compute_rsi(prices, period=14):
    """Compute RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(prices, fast=12, slow=26, signal=9):
    """Compute MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def create_chart(df, show_rsi=True, show_macd=True, show_bb=True):
    """Create interactive Plotly chart"""
    if df is None or df.empty:
        return None

    # Create subplots
    num_rows = 1 + (1 if show_rsi else 0) + (1 if show_macd else 0)
    row_heights = [0.6] + [0.2] * (num_rows - 1)

    fig = make_subplots(
        rows=num_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]] * (num_rows - 1)
    )

    # Row 1: Candlestick + Volume + Bollinger Bands
    fig.add_trace(
        go.Candlestick(
            x=df['Timestamp'],
            open=df['OpenPrice'],
            high=df['HighPrice'],
            low=df['LowPrice'],
            close=df['ClosePrice'],
            name='OHLC',
            showlegend=True
        ),
        row=1, col=1, secondary_y=False
    )

    # Bollinger Bands
    if show_bb:
        fig.add_trace(
            go.Scatter(
                x=df['Timestamp'], y=df['BB_Upper'],
                mode='lines', name='BB Upper',
                line=dict(color='rgba(255,0,0,0.3)', width=1),
                showlegend=True
            ),
            row=1, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=df['Timestamp'], y=df['BB_Lower'],
                mode='lines', name='BB Lower',
                line=dict(color='rgba(0,0,255,0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(128,128,255,0.1)',
                showlegend=True
            ),
            row=1, col=1, secondary_y=False
        )

    # Volume on secondary y-axis
    colors = ['red' if df['ClosePrice'].iloc[i] < df['OpenPrice'].iloc[i] else 'green'
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(
            x=df['Timestamp'], y=df['Volume'],
            name='Volume',
            marker=dict(color=colors),
            showlegend=True,
            opacity=0.5
        ),
        row=1, col=1, secondary_y=True
    )

    # Row 2: RSI
    if show_rsi:
        row_idx = 2
        fig.add_trace(
            go.Scatter(
                x=df['Timestamp'], y=df['RSI'],
                mode='lines', name='RSI (14)',
                line=dict(color='orange', width=2),
                showlegend=True
            ),
            row=row_idx, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=row_idx, col=1, annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=row_idx, col=1, annotation_text="Oversold")

    # Row 3: MACD
    if show_macd:
        row_idx = 2 + (1 if show_rsi else 0)
        fig.add_trace(
            go.Scatter(
                x=df['Timestamp'], y=df['MACD'],
                mode='lines', name='MACD',
                line=dict(color='blue', width=2),
                showlegend=True
            ),
            row=row_idx, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['Timestamp'], y=df['Signal'],
                mode='lines', name='Signal',
                line=dict(color='red', width=1),
                showlegend=True
            ),
            row=row_idx, col=1
        )
        colors_macd = ['green' if x > 0 else 'red' for x in df['MACD_Hist']]
        fig.add_trace(
            go.Bar(
                x=df['Timestamp'], y=df['MACD_Hist'],
                name='MACD Histogram',
                marker=dict(color=colors_macd),
                showlegend=True,
                opacity=0.3
            ),
            row=row_idx, col=1
        )

    # Update layout
    fig.update_layout(
        title=f"Kaggle Data Analysis - OHLCV + Indicators",
        xaxis_title="Time",
        yaxis_title="Price",
        height=800,
        template="plotly_dark",
        hovermode='x unified',
        font=dict(size=10)
    )

    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)

    return fig

# Routes

@app.route('/')
def index():
    symbols = get_available_symbols()
    min_date, max_date = get_date_range()
    return render_template('analysis.html', symbols=symbols, min_date=min_date, max_date=max_date)

@app.route('/api/chart', methods=['POST'])
def api_chart():
    """Get chart data"""
    data = request.json
    symbol = data.get('symbol', 'AAPL')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    show_rsi = data.get('show_rsi', True)
    show_macd = data.get('show_macd', True)
    show_bb = data.get('show_bb', True)

    df = get_kaggle_data(symbol, start_date, end_date)

    if df is None:
        return jsonify({'error': 'No data available'}), 400

    fig = create_chart(df, show_rsi=show_rsi, show_macd=show_macd, show_bb=show_bb)

    return jsonify({
        'chart': plotly.io.to_json(fig),
        'stats': {
            'records': len(df),
            'date_range': f"{df['Timestamp'].min()} to {df['Timestamp'].max()}",
            'avg_volume': int(df['Volume'].mean()),
            'volume_ma': int(df['Volume_MA'].iloc[-1]) if not pd.isna(df['Volume_MA'].iloc[-1]) else 0
        }
    })

@app.route('/api/table', methods=['POST'])
def api_table():
    """Get data table"""
    data = request.json
    symbol = data.get('symbol', 'AAPL')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    limit = data.get('limit', 100)

    df = get_kaggle_data(symbol, start_date, end_date)

    if df is None:
        return jsonify({'error': 'No data available'}), 400

    # Show last N rows
    df = df.tail(limit).copy()
    df['Timestamp'] = df['Timestamp'].astype(str)
    df = df.round(4)

    return jsonify({
        'columns': df.columns.tolist(),
        'data': df.to_dict('records')
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)
