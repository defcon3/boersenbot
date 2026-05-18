#!/usr/bin/env python3
"""
Börsenbot Dashboard — Flask Web Interface
Monitors data collection, streaming status, portfolio, and trading signals
"""

from flask import Flask, render_template, jsonify, request
import pymssql
import subprocess
import os
import json
from datetime import datetime, timedelta
import sys

app = Flask(__name__)

# Database config
DB_CONFIG = {
    'server': '158.181.48.77',
    'database': 'dbdata',
    'user': '326773',
    'password': 'Extaler11!',
    'as_dict': True
}

def get_db():
    """Get database connection"""
    return pymssql.connect(**DB_CONFIG, autocommit=True)

def get_data_stats():
    """Get data collection statistics from database"""
    try:
        conn = get_db()
        cursor = conn.cursor()

        stats = {}

        # Daily OHLCV data
        cursor.execute("""
            SELECT COUNT(*) as count, MAX([Date]) as latest_date
            FROM bb_StockPrices
        """)
        daily = cursor.fetchone()
        stats['daily'] = {
            'rows': daily['count'] or 0,
            'latest': daily['latest_date'].strftime('%Y-%m-%d') if daily['latest_date'] else 'N/A',
            'source': 'Yahoo Finance'
        }

        # 1-min Kaggle data
        cursor.execute("""
            SELECT COUNT(*) as count, MAX([Timestamp]) as latest_dt
            FROM bb_StockPrices_1min_Kaggle
        """)
        kaggle = cursor.fetchone()
        stats['kaggle'] = {
            'rows': kaggle['count'] or 0,
            'latest': kaggle['latest_dt'].strftime('%Y-%m-%d %H:%M') if kaggle['latest_dt'] else 'N/A',
            'source': 'Kaggle (May 2023)'
        }

        # 1-min yfinance data
        cursor.execute("""
            SELECT COUNT(*) as count, MAX([Timestamp]) as latest_dt
            FROM bb_StockPrices_1min_YFinance
        """)
        yfinance = cursor.fetchone()
        stats['yfinance'] = {
            'rows': yfinance['count'] or 0,
            'latest': yfinance['latest_dt'].strftime('%Y-%m-%d %H:%M') if yfinance['latest_dt'] else 'N/A',
            'source': 'yfinance (Polling)'
        }

        # 1-min Alpaca Streaming data
        cursor.execute("""
            SELECT COUNT(*) as count, MAX([Timestamp]) as latest_dt
            FROM bb_StockPrices_1min_Streaming
        """)
        alpaca = cursor.fetchone()
        stats['alpaca'] = {
            'rows': alpaca['count'] or 0,
            'latest': alpaca['latest_dt'].strftime('%Y-%m-%d %H:%M') if alpaca['latest_dt'] else 'N/A',
            'source': 'Alpaca WebSocket'
        }

        # Latest prices per symbol
        cursor.execute("""
            SELECT Symbol, ClosePrice as price, [Date]
            FROM bb_StockPrices
            WHERE [Date] = (SELECT MAX([Date]) FROM bb_StockPrices)
            ORDER BY Symbol
        """)
        prices = cursor.fetchall()
        stats['latest_prices'] = {row['Symbol']: row['price'] for row in prices}
        stats['price_date'] = prices[0]['Date'].strftime('%Y-%m-%d') if prices else 'N/A'

        cursor.close()
        conn.close()
        return stats
    except Exception as e:
        return {'error': str(e)}

def get_streaming_status():
    """Check if Alpaca streaming process is running"""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'alpaca_streaming'],
            capture_output=True,
            text=True,
            timeout=5
        )
        is_running = result.returncode == 0
        return {
            'running': is_running,
            'status': 'Running' if is_running else 'Stopped',
            'pid': result.stdout.strip() if is_running else None
        }
    except Exception as e:
        return {'running': False, 'status': 'Unknown', 'error': str(e)}

def control_streaming(action):
    """Control streaming: start, stop, restart"""
    try:
        # Script path
        script_path = '/home/veit/boersenbot/manage_streaming.py'

        if not os.path.exists(script_path):
            return {'success': False, 'message': 'manage_streaming.py not found'}

        result = subprocess.run(
            ['python3', script_path, action],
            capture_output=True,
            text=True,
            timeout=10,
            cwd='/home/veit/boersenbot'
        )

        return {
            'success': result.returncode == 0,
            'action': action,
            'output': result.stdout + result.stderr
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}

# Routes

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/api/stats')
def api_stats():
    """API: Get data collection statistics"""
    return jsonify(get_data_stats())

@app.route('/api/streaming')
def api_streaming():
    """API: Get streaming status"""
    return jsonify(get_streaming_status())

@app.route('/api/streaming/control', methods=['POST'])
def api_streaming_control():
    """API: Control streaming (start/stop/restart)"""
    action = request.json.get('action', 'status')
    if action not in ['start', 'stop', 'restart', 'status']:
        return jsonify({'success': False, 'message': 'Invalid action'}), 400

    result = control_streaming(action)
    return jsonify(result)

@app.route('/api/health')
def api_health():
    """API: Health check"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    })

if __name__ == '__main__':
    # Run on localhost:5000 (Nginx will proxy to this)
    app.run(host='127.0.0.1', port=5000, debug=False)
