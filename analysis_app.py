#!/usr/bin/env python3
from flask import Flask, render_template, jsonify, request
import pymssql
import pandas as pd
import numpy as np

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

# ── Indicator calculations ──────────────────────────────────────────────────

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss))

def compute_macd(prices, fast=12, slow=26, sig=9):
    ema_f = prices.ewm(span=fast).mean()
    ema_s = prices.ewm(span=slow).mean()
    macd  = ema_f - ema_s
    signal = macd.ewm(span=sig).mean()
    return macd, signal, macd - signal

def compute_stochastic(high, low, close, k=14, d=3):
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    pct_k = 100 * (close - ll) / (hh - ll)
    return pct_k, pct_k.rolling(d).mean()

def compute_atr(high, low, close, period=14):
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def compute_cci(high, low, close, period=20):
    tp  = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad)

def compute_williams_r(high, low, close, period=14):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll)

def compute_obv(close, volume):
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

def compute_adx(high, low, close, period=14):
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    up   = high.diff()
    down = -low.diff()
    dm_p = up.where((up > down) & (up > 0), 0)
    dm_m = down.where((down > up) & (down > 0), 0)
    atr_e = tr.ewm(span=period, adjust=False).mean()
    di_p  = 100 * dm_p.ewm(span=period, adjust=False).mean() / atr_e
    di_m  = 100 * dm_m.ewm(span=period, adjust=False).mean() / atr_e
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m)
    return dx.ewm(span=period, adjust=False).mean(), di_p, di_m

def compute_vwap(high, low, close, volume):
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()

def compute_psar(high, low, af0=0.02, af_max=0.2):
    n   = len(high)
    sar = np.zeros(n)
    ep  = np.zeros(n)
    af  = np.zeros(n)
    trend = np.ones(n)   # 1 = uptrend
    hi, lo = high.values, low.values

    sar[0], ep[0], af[0] = lo[0], hi[0], af0

    for i in range(1, n):
        if trend[i-1] == 1:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = min(sar[i], lo[max(0,i-1)], lo[max(0,i-2)])
            if lo[i] < sar[i]:
                trend[i] = -1; sar[i] = ep[i-1]; ep[i] = lo[i]; af[i] = af0
            else:
                trend[i] = 1
                ep[i]  = max(ep[i-1], hi[i])
                af[i]  = min(af[i-1] + af0, af_max) if hi[i] > ep[i-1] else af[i-1]
        else:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = max(sar[i], hi[max(0,i-1)], hi[max(0,i-2)])
            if hi[i] > sar[i]:
                trend[i] = 1; sar[i] = ep[i-1]; ep[i] = hi[i]; af[i] = af0
            else:
                trend[i] = -1
                ep[i]  = min(ep[i-1], lo[i])
                af[i]  = min(af[i-1] + af0, af_max) if lo[i] < ep[i-1] else af[i-1]

    return pd.Series(sar, index=high.index), pd.Series(trend, index=high.index)

# ── Data loading ────────────────────────────────────────────────────────────

def load_data(symbol, start_date=None, end_date=None):
    try:
        conn = get_db()
        where = f"Symbol = '{symbol}'"
        if start_date: where += f" AND [Timestamp] >= '{start_date}'"
        if end_date:   where += f" AND [Timestamp] <= '{end_date} 23:59:59'"
        df = pd.read_sql(
            f"SELECT Timestamp, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume "
            f"FROM bb_StockPrices_1min_Kaggle WHERE {where} ORDER BY Timestamp", conn)
        conn.close()
    except Exception as e:
        print(f"DB error: {e}")
        return None

    if df.empty:
        return None

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    c, h, l, v = df['ClosePrice'], df['HighPrice'], df['LowPrice'], df['Volume']

    df['RSI']        = compute_rsi(c)
    df['MACD'], df['MACD_Sig'], df['MACD_Hist'] = compute_macd(c)
    sma20            = c.rolling(20).mean()
    std20            = c.rolling(20).std()
    df['BB_U']       = sma20 + std20 * 2
    df['BB_L']       = sma20 - std20 * 2
    df['BB_M']       = sma20
    df['STOCH_K'], df['STOCH_D'] = compute_stochastic(h, l, c)
    df['ATR']        = compute_atr(h, l, c)
    df['CCI']        = compute_cci(h, l, c)
    df['WILLR']      = compute_williams_r(h, l, c)
    df['OBV']        = compute_obv(c, v)
    df['ADX'], df['DI_P'], df['DI_M'] = compute_adx(h, l, c)
    df['VWAP']       = compute_vwap(h, l, c, v)
    df['PSAR'], df['PSAR_TREND'] = compute_psar(h, l)
    return df

# ── Serialization ───────────────────────────────────────────────────────────

def to_lw(ts):
    return int(ts.timestamp())

def series(df, col):
    mask = ~df[col].isna() & ~np.isinf(df[col])
    return [{'time': to_lw(t), 'value': round(float(v), 4)}
            for t, v in zip(df.loc[mask,'Timestamp'], df.loc[mask, col])]

def build_payload(df):
    up   = 'rgba(38,166,154,0.8)'
    down = 'rgba(239,83,80,0.8)'
    ohlcv  = [{'time': to_lw(t), 'open': o, 'high': h, 'low': l, 'close': c}
              for t,o,h,l,c in zip(df['Timestamp'],df['OpenPrice'],df['HighPrice'],df['LowPrice'],df['ClosePrice'])]
    volume = [{'time': to_lw(t), 'value': float(v),
               'color': up if c >= o else down}
              for t,v,c,o in zip(df['Timestamp'],df['Volume'],df['ClosePrice'],df['OpenPrice'])]

    # MACD histogram with colors
    mask = ~df['MACD_Hist'].isna()
    macd_hist = [{'time': to_lw(t), 'value': round(float(v), 4),
                  'color': 'rgba(38,166,154,0.6)' if v >= 0 else 'rgba(239,83,80,0.6)'}
                 for t, v in zip(df.loc[mask,'Timestamp'], df.loc[mask,'MACD_Hist'])]

    # PSAR split by trend
    psar_up   = [{'time': to_lw(t), 'value': round(float(v), 4)}
                 for t,v,tr in zip(df['Timestamp'], df['PSAR'], df['PSAR_TREND'])
                 if not np.isnan(v) and tr == 1]
    psar_down = [{'time': to_lw(t), 'value': round(float(v), 4)}
                 for t,v,tr in zip(df['Timestamp'], df['PSAR'], df['PSAR_TREND'])
                 if not np.isnan(v) and tr == -1]

    return {
        'ohlcv':  ohlcv,
        'volume': volume,
        'stats': {
            'records': len(df),
            'from': str(df['Timestamp'].iloc[0])[:16],
            'to':   str(df['Timestamp'].iloc[-1])[:16],
            'avg_vol': int(df['Volume'].mean()),
        },
        'indicators': {
            'rsi':        series(df, 'RSI'),
            'macd':       series(df, 'MACD'),
            'macd_sig':   series(df, 'MACD_Sig'),
            'macd_hist':  macd_hist,
            'bb_upper':   series(df, 'BB_U'),
            'bb_lower':   series(df, 'BB_L'),
            'bb_middle':  series(df, 'BB_M'),
            'stoch_k':    series(df, 'STOCH_K'),
            'stoch_d':    series(df, 'STOCH_D'),
            'atr':        series(df, 'ATR'),
            'cci':        series(df, 'CCI'),
            'willr':      series(df, 'WILLR'),
            'obv':        series(df, 'OBV'),
            'adx':        series(df, 'ADX'),
            'di_plus':    series(df, 'DI_P'),
            'di_minus':   series(df, 'DI_M'),
            'vwap':       series(df, 'VWAP'),
            'psar_up':    psar_up,
            'psar_down':  psar_down,
        }
    }

# ── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    symbols = get_available_symbols()
    min_date, max_date = get_date_range()
    return render_template('analysis.html', symbols=symbols, min_date=min_date, max_date=max_date)

@app.route('/api/chart', methods=['POST'])
def api_chart():
    data = request.json
    df = load_data(data.get('symbol','AAPL'), data.get('start_date'), data.get('end_date'))
    if df is None:
        return jsonify({'error': 'Keine Daten verfügbar'}), 400
    return jsonify(build_payload(df))

@app.route('/api/table', methods=['POST'])
def api_table():
    data = request.json
    df = load_data(data.get('symbol','AAPL'), data.get('start_date'), data.get('end_date'))
    if df is None:
        return jsonify({'error': 'Keine Daten verfügbar'}), 400
    cols = ['Timestamp','OpenPrice','HighPrice','LowPrice','ClosePrice','Volume','RSI','MACD','ATR']
    df2 = df[cols].tail(data.get('limit',200)).copy()
    df2['Timestamp'] = df2['Timestamp'].astype(str)
    return jsonify({'columns': cols, 'data': df2.round(4).to_dict('records')})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)
