#!/usr/bin/env python3
"""
Börsenbot Dashboard — Flask Web Interface
Monitors data collection, streaming status, portfolio, and trading signals
"""

from flask import Flask, render_template, jsonify, request
import pymssql
import sqlite3
import subprocess
import os
import json
from collections import defaultdict
from datetime import datetime, timedelta
import sys
import timeutil

app = Flask(__name__)

# Dividend-Tracker (eigene SQLite-DB, reine Beobachtung — KEINE Empfehlung,
# #11/#12/#13 falsifiziert, siehe veitluther.de/done)
DIVIDEND_DB = '/home/veit/boersenbot/dividend_tracker.db'

def get_dividend_db():
    """SQLite-Verbindung zur Dividend-Tracker-DB."""
    conn = sqlite3.connect(DIVIDEND_DB)
    conn.row_factory = sqlite3.Row
    return conn

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

def _freshness(latest_dt, max_age_days, static=False, tz_utc=False):
    """Bewertet, wie aktuell ein Datenstand ist.

    Gibt age_days, fresh-Flag und einen menschenlesbaren Hinweis zurück.
    static=True => bewusst historischer Datensatz (kein Veraltet-Alarm).
    """
    if latest_dt is None:
        return {'age_days': None, 'fresh': False, 'static': static,
                'note': 'keine Daten'}
    # bb_StockPrices.[Date] kommt als date, die 1min-Tabellen als datetime
    if not isinstance(latest_dt, datetime):
        latest_dt = datetime(latest_dt.year, latest_dt.month, latest_dt.day)
    if tz_utc:
        # Streaming-Timestamps sind naive UTC -> zentraler Zeit-Vertrag
        age_sec = timeutil.age_seconds(latest_dt)
    else:
        age_sec = max(0.0, (datetime.now() - latest_dt).total_seconds())
    age_days = age_sec / 86400.0
    if static:
        return {'age_days': round(age_days, 1), 'fresh': True, 'static': True,
                'note': 'historischer Datensatz (fix)'}
    fresh = age_days <= max_age_days
    if age_sec < 3600:
        human = f'vor {int(age_sec // 60)} Min'
    elif age_days < 1:
        human = f'vor {int(age_sec // 3600)} Std'
    else:
        human = f'vor {int(round(age_days))} Tag(en)'
    return {'age_days': round(age_days, 1), 'fresh': fresh, 'static': False,
            'note': ('aktuell, ' if fresh else 'VERALTET, ') + human}

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
            'source': 'Yahoo Finance',
            'freshness': _freshness(daily['latest_date'], max_age_days=4)
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
            'source': 'Kaggle (May 2023)',
            'freshness': _freshness(kaggle['latest_dt'], max_age_days=0, static=True)
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
            'source': 'yfinance (Polling)',
            'freshness': _freshness(yfinance['latest_dt'], max_age_days=4)
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
            'source': 'Alpaca WebSocket',
            'freshness': _freshness(alpaca['latest_dt'], max_age_days=1, tz_utc=True)
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
    """Echter Streaming-Status: Prozess UND tatsächlicher Datenfluss.

    Wichtig: Ein laufender Prozess heißt NICHT, dass Daten ankommen
    (Alpaca lieferte wegen 401 Unauthorized nie Daten, der Prozess
    lebte aber). Der Status berücksichtigt daher die letzte
    tatsächlich geschriebene Zeile in der Streaming-Tabelle.
    """
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'alpaca_streaming'],
            capture_output=True, text=True, timeout=5
        )
        proc_up = result.returncode == 0
        pid = result.stdout.strip() if proc_up else None

        last_data = None
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("SELECT MAX([Timestamp]) AS m FROM bb_StockPrices_1min_Streaming")
            row = cur.fetchone()
            last_data = row['m'] if row else None
            cur.close()
            conn.close()
        except Exception:
            pass

        # Datenfluss = Zeile in den letzten 15 Minuten
        data_flowing = False
        if last_data is not None:
            data_flowing = timeutil.age_seconds(last_data) < 900

        if proc_up and data_flowing:
            cls, text = 'running', '🟢 Aktiv (Daten fließen)'
        elif proc_up and not data_flowing:
            cls, text = 'idle', '🟠 Prozess läuft, aber KEINE Daten'
        else:
            cls, text = 'stopped', '🔴 Gestoppt'

        return {
            'running': proc_up,
            'data_flowing': data_flowing,
            'status': text,
            'status_class': cls,
            'last_data': timeutil.to_local_str(last_data),
            'pid': pid
        }
    except Exception as e:
        return {'running': False, 'data_flowing': False,
                'status': '⚪ Unbekannt', 'status_class': 'stopped',
                'last_data': 'nie', 'error': str(e)}

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

@app.route('/fazit')
def fazit_seite():
    """Boersenbot Fazit-Seite (Ergebnisse + Architektur)"""
    return render_template('fazit.html')

@app.route('/done')
def done_seite():
    return render_template('done.html')

@app.route('/overnight-intraday')
def overnight_intraday_seite():
    """Show-Data: Overnight-vs-Intraday-Zerlegung (YELLOW, nicht handelbar)."""
    return render_template('overnight_intraday.html')

@app.route('/dividend-watch')
def dividend_watch():
    """Dividend-Tracker-Dashboard: Ex-Tag-Events + Kursverlauf Aktie vs. Optionsschein.

    Reine Daten-Sammlung NACH Falsifikation von #11/#12/#13 — kein Trading-Signal.
    """
    conn = get_dividend_db()
    events = conn.execute("""
        SELECT e.event_id, e.ticker, e.name, e.ex_date, e.gross_div, e.currency,
               COUNT(DISTINCT w.warrant_id) AS n_warrants
        FROM events e
        LEFT JOIN warrants w ON w.event_id = e.event_id
        GROUP BY e.event_id, e.ticker, e.name, e.ex_date, e.gross_div, e.currency
        ORDER BY e.ex_date DESC, e.ticker
    """).fetchall()

    stock_rows = conn.execute("""
        SELECT event_id, snapshot_date, days_to_ex, stock_close
        FROM stock_snapshots
        WHERE stock_close IS NOT NULL
        ORDER BY snapshot_date
    """).fetchall()

    warr_rows = conn.execute("""
        SELECT w.event_id, w.wkn, w.issuer, w.opt_type, w.strike,
               ws.snapshot_date, ws.mid, ws.hebel
        FROM warrant_snapshots ws
        JOIN warrants w ON w.warrant_id = ws.warrant_id
        WHERE ws.mid IS NOT NULL
        ORDER BY ws.snapshot_date
    """).fetchall()
    conn.close()

    stock_by_ev = defaultdict(list)
    for r in stock_rows:
        stock_by_ev[r['event_id']].append(
            {'date': r['snapshot_date'], 'close': r['stock_close'], 'dte': r['days_to_ex']})

    warr_by_ev = defaultdict(lambda: defaultdict(list))
    warr_meta = {}
    for r in warr_rows:
        warr_by_ev[r['event_id']][r['wkn']].append(
            {'date': r['snapshot_date'], 'mid': r['mid'], 'hebel': r['hebel']})
        warr_meta[(r['event_id'], r['wkn'])] = {
            'issuer': r['issuer'], 'opt_type': r['opt_type'], 'strike': r['strike']}

    today = datetime.now().date()
    events_out, charts = [], []
    for e in events:
        try:
            dte = (datetime.strptime(e['ex_date'], '%Y-%m-%d').date() - today).days
        except (ValueError, TypeError):
            dte = None
        sdata = sorted(stock_by_ev.get(e['event_id'], []), key=lambda x: x['date'])
        uniq_days = sorted({s['date'] for s in sdata})
        events_out.append({
            'event_id': e['event_id'], 'ticker': e['ticker'], 'name': e['name'],
            'ex_date': e['ex_date'], 'gross_div': e['gross_div'],
            'currency': e['currency'] or 'EUR', 'n_warrants': e['n_warrants'],
            'days_to_ex': dte, 'n_stock_days': len(uniq_days),
            'has_chart': len(uniq_days) >= 2,
        })
        if len(uniq_days) >= 2:
            warrants = []
            for wkn, pts in warr_by_ev.get(e['event_id'], {}).items():
                m = warr_meta.get((e['event_id'], wkn), {})
                warrants.append({
                    'wkn': wkn, 'issuer': m.get('issuer'),
                    'opt_type': m.get('opt_type'), 'strike': m.get('strike'),
                    'points': sorted(pts, key=lambda x: x['date']),
                })
            charts.append({
                'event_id': e['event_id'], 'ticker': e['ticker'], 'name': e['name'],
                'ex_date': e['ex_date'], 'gross_div': e['gross_div'],
                'currency': e['currency'] or 'EUR',
                'stock': sdata, 'warrants': warrants,
            })

    return render_template('dividend_watch.html',
                           events=events_out, charts_json=json.dumps(charts))

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
