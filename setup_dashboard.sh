#!/bin/bash
# Börsenbot Dashboard Setup Script
# Run on VPS: bash setup_dashboard.sh

set -e
cd /home/veit/boersenbot

echo "=== Creating app.py ==="
cat > app.py << 'EOFAPP'
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
            SELECT COUNT(*) as count, MAX([DateTime]) as latest_dt
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
            SELECT COUNT(*) as count, MAX([DateTime]) as latest_dt
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
            SELECT COUNT(*) as count, MAX([DateTime]) as latest_dt
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
            SELECT Symbol, [Close] as price, [Date]
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
EOFAPP

echo "=== Creating templates directory ==="
mkdir -p templates

echo "=== Creating templates/index.html ==="
cat > templates/index.html << 'EOFHTML'
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Börsenbot Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1419;
            color: #e0e6ed;
            line-height: 1.6;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        header h1 {
            font-size: 28px;
            margin-bottom: 5px;
        }
        header p {
            font-size: 14px;
            opacity: 0.9;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: #1a1e27;
            border: 1px solid #2a2e37;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .card h2 {
            font-size: 16px;
            margin-bottom: 15px;
            color: #667eea;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stat {
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid #2a2e37;
        }
        .stat:last-child {
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }
        .stat-label {
            font-size: 12px;
            color: #999;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .stat-value {
            font-size: 18px;
            font-weight: 600;
            color: #fff;
        }
        .stat-source {
            font-size: 11px;
            color: #666;
            margin-top: 2px;
        }
        .status {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-top: 8px;
        }
        .status.running {
            background: #10b981;
            color: #fff;
        }
        .status.stopped {
            background: #ef4444;
            color: #fff;
        }
        .control-buttons {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        button {
            flex: 1;
            min-width: 80px;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            text-transform: uppercase;
        }
        .btn-primary {
            background: #667eea;
            color: #fff;
        }
        .btn-primary:hover {
            background: #5568d3;
        }
        .btn-danger {
            background: #ef4444;
            color: #fff;
        }
        .btn-danger:hover {
            background: #dc2626;
        }
        .btn-warning {
            background: #f59e0b;
            color: #fff;
        }
        .btn-warning:hover {
            background: #d97706;
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .full-width {
            grid-column: 1 / -1;
        }
        .price-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }
        .price-card {
            background: #0f1419;
            border: 1px solid #2a2e37;
            border-radius: 4px;
            padding: 12px;
            text-align: center;
        }
        .price-symbol {
            font-size: 14px;
            font-weight: 600;
            color: #667eea;
            margin-bottom: 4px;
        }
        .price-value {
            font-size: 20px;
            font-weight: 700;
            color: #fff;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #999;
        }
        .error {
            background: #7f1d1d;
            border: 1px solid #dc2626;
            color: #fecaca;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 12px;
            border-top: 1px solid #2a2e37;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <header>
        <h1>📊 Börsenbot Dashboard</h1>
        <p>Real-time data collection monitoring & trading control</p>
    </header>

    <div class="container">
        <div id="error-container"></div>

        <div class="grid">
            <div class="card">
                <h2>📈 Daily Data (OHLCV)</h2>
                <div id="daily-stats" class="loading">Loading...</div>
            </div>

            <div class="card">
                <h2>📊 Kaggle Data (1min)</h2>
                <div id="kaggle-stats" class="loading">Loading...</div>
            </div>

            <div class="card">
                <h2>📡 YFinance Data (1min)</h2>
                <div id="yfinance-stats" class="loading">Loading...</div>
            </div>

            <div class="card">
                <h2>⚡ Alpaca Streaming (1min)</h2>
                <div id="alpaca-stats" class="loading">Loading...</div>
            </div>

            <div class="card full-width">
                <h2>🎛️ Streaming Control</h2>
                <div id="streaming-status" class="loading">Loading...</div>
                <div class="control-buttons">
                    <button class="btn-primary" onclick="controlStreaming('start')">Start</button>
                    <button class="btn-warning" onclick="controlStreaming('restart')">Restart</button>
                    <button class="btn-danger" onclick="controlStreaming('stop')">Stop</button>
                </div>
            </div>

            <div class="card full-width">
                <h2>💰 Latest Prices</h2>
                <div id="prices" class="price-grid loading">Loading...</div>
            </div>
        </div>

        <div class="footer">
            <p>Auto-refresh every 10 seconds | Börsenbot v1.0</p>
        </div>
    </div>

    <script>
        function showError(message) {
            const container = document.getElementById('error-container');
            container.innerHTML = `<div class="error">⚠️ ${message}</div>`;
            setTimeout(() => { container.innerHTML = ''; }, 5000);
        }

        function formatDate(dateStr) {
            if (!dateStr || dateStr === 'N/A') return 'N/A';
            const date = new Date(dateStr);
            return date.toLocaleString('de-DE');
        }

        function renderStats(element, data) {
            if (!element) return;
            const html = `
                <div class="stat">
                    <div class="stat-label">Rows</div>
                    <div class="stat-value">${data.rows.toLocaleString()}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Latest Update</div>
                    <div class="stat-value">${data.latest}</div>
                    <div class="stat-source">${data.source}</div>
                </div>
            `;
            element.innerHTML = html;
        }

        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();

                if (data.error) {
                    showError(`Database error: ${data.error}`);
                    return;
                }

                renderStats(document.getElementById('daily-stats'), data.daily);
                renderStats(document.getElementById('kaggle-stats'), data.kaggle);
                renderStats(document.getElementById('yfinance-stats'), data.yfinance);
                renderStats(document.getElementById('alpaca-stats'), data.alpaca);

                const pricesElement = document.getElementById('prices');
                if (data.latest_prices && Object.keys(data.latest_prices).length > 0) {
                    const pricesHtml = Object.entries(data.latest_prices)
                        .map(([symbol, price]) => `
                            <div class="price-card">
                                <div class="price-symbol">${symbol}</div>
                                <div class="price-value">$${parseFloat(price).toFixed(2)}</div>
                            </div>
                        `)
                        .join('');
                    pricesElement.innerHTML = pricesHtml;
                    pricesElement.classList.remove('loading');
                } else {
                    pricesElement.innerHTML = '<p>No data</p>';
                }
            } catch (error) {
                showError(`Failed to load stats: ${error.message}`);
            }
        }

        async function loadStreamingStatus() {
            try {
                const response = await fetch('/api/streaming');
                const data = await response.json();

                const element = document.getElementById('streaming-status');
                const statusClass = data.running ? 'running' : 'stopped';
                const statusText = data.running ? '🟢 Running' : '🔴 Stopped';

                element.innerHTML = `
                    <div class="stat">
                        <div class="stat-label">Status</div>
                        <div class="stat-value">
                            <span class="status ${statusClass}">${statusText}</span>
                        </div>
                        ${data.pid ? `<div class="stat-source">PID: ${data.pid}</div>` : ''}
                    </div>
                `;
                element.classList.remove('loading');
            } catch (error) {
                showError(`Failed to load streaming status: ${error.message}`);
            }
        }

        async function controlStreaming(action) {
            const buttons = document.querySelectorAll('.control-buttons button');
            buttons.forEach(btn => btn.disabled = true);

            try {
                const response = await fetch('/api/streaming/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action: action })
                });
                const data = await response.json();

                if (data.success) {
                    setTimeout(loadStreamingStatus, 1000);
                } else {
                    showError(`Action failed: ${data.message || data.output}`);
                }
            } catch (error) {
                showError(`Control error: ${error.message}`);
            } finally {
                buttons.forEach(btn => btn.disabled = false);
            }
        }

        loadStats();
        loadStreamingStatus();
        setInterval(() => {
            loadStats();
            loadStreamingStatus();
        }, 10000);
    </script>
</body>
</html>
EOFHTML

echo "=== Creating requirements_dash.txt ==="
cat > requirements_dash.txt << 'EOFREQ'
Flask==2.3.3
pymssql==2.2.8
gunicorn==21.2.0
EOFREQ

echo "=== Installing Flask dependencies ==="
source venv/bin/activate
pip install -r requirements_dash.txt

echo "=== Creating systemd service ==="
sudo tee /etc/systemd/system/boersenbot_dashboard.service > /dev/null << 'EOFSVC'
[Unit]
Description=Börsenbot Flask Dashboard
After=network.target

[Service]
Type=notify
User=veit
WorkingDirectory=/home/veit/boersenbot
ExecStart=/home/veit/boersenbot/venv/bin/gunicorn --workers 2 --bind 127.0.0.1:5000 --timeout 120 app:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOFSVC

echo "=== Enabling and starting service ==="
sudo systemctl daemon-reload
sudo systemctl enable boersenbot_dashboard
sudo systemctl start boersenbot_dashboard

echo "=== Checking service status ==="
sudo systemctl status boersenbot_dashboard

echo ""
echo "✅ Dashboard setup complete!"
echo "Access: http://veitluther.de (once DNS propagates)"
echo "Local: http://localhost:5000 (on VPS)"
