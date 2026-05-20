#!/usr/bin/env python3
from flask import Flask, render_template, jsonify, request
import pymssql
import pandas as pd
import numpy as np
from scipy.optimize import linprog

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

def signal_rule(df, rsi_buy=70, rsi_sell=30, min_gap=0, zero_line=False,
                macd_fast=12, macd_slow=26, macd_sig=9, max_sig=0):
    """Parametrierte Buy/Sell-Regel — EINE Quelle der Wahrheit.

    BUY  : MACD_Hist wechselt <=0 -> >0  und  RSI < rsi_buy
    SELL : MACD_Hist wechselt >=0 -> <0  und  RSI > rsi_sell
    zero_line : zusätzlich BUY nur wenn MACD>0, SELL nur wenn MACD<0
                (echte Trendwechsel statt Mini-Wackler)
    min_gap   : Entprellung — Mindestabstand in Bars zwischen Signalen
    macd_*    : MACD-Perioden (live justierbar, MACD wird hier neu gerechnet)
    max_sig   : 0 = alle; sonst nur die N STÄRKSTEN Signale (größtes
                |MACD_Hist| am Kreuzungs-Bar), Zeitreihenfolge bleibt
    """
    c = df['ClosePrice']
    macd, _sig, hist = compute_macd(c, macd_fast, macd_slow, macd_sig)
    rsi  = df['RSI']
    prev = hist.shift(1)
    buy  = (prev <= 0) & (hist > 0) & (rsi < rsi_buy)
    sell = (prev >= 0) & (hist < 0) & (rsi > rsi_sell)
    if zero_line:
        buy  = buy  & (macd > 0)
        sell = sell & (macd < 0)
    ts = df['Timestamp']
    hv = hist.values
    out, last_i = [], -10**9
    for i, (b, s) in enumerate(zip(buy.values, sell.values)):
        if not (b or s):
            continue
        if i - last_i < min_gap:
            continue
        out.append({'time': to_lw(ts.iloc[i]), 'side': 'buy' if b else 'sell',
                    '_str': abs(float(hv[i]))})
        last_i = i
    if max_sig and len(out) > max_sig:
        keep = sorted(sorted(range(len(out)), key=lambda k: -out[k]['_str'])[:max_sig])
        out = [out[k] for k in keep]
    for o in out:
        o.pop('_str', None)
    return out

def compute_signals(df):
    """Rückwärtskompatibler Default (für Ad-hoc-Backtest-Skripte)."""
    return signal_rule(df)

def _price_index(df):
    pbt = {to_lw(t): float(c) for t, c in zip(df['Timestamp'], df['ClosePrice'])}
    return pbt, float(df['ClosePrice'].iloc[-1]), float(df['ClosePrice'].iloc[0])

def backtest_flip(signals, pbt, last_close, start=1000.0, lev=3.0, cost_rt=0.0):
    """Flip-Strategie: BUY->Call 3x, SELL->Put 3x, erste Position beim 1. BUY,
    offene Position am letzten Kurs glattgestellt. cost_rt = Kosten je Wechsel.
    """
    eq, pos, entry, trades = start, None, None, []
    for sg in signals:
        p = pbt.get(sg['time'])
        if p is None:
            continue
        side = sg['side']
        if pos == 'call' and side == 'sell':
            r = (p - entry) / entry
            eq *= (1 + lev * r) * (1 - cost_rt); trades.append(lev * r); pos = entry = None
        elif pos == 'put' and side == 'buy':
            r = (p - entry) / entry
            eq *= (1 + lev * (-r)) * (1 - cost_rt); trades.append(lev * (-r)); pos = entry = None
        if pos is None:
            if side == 'buy':
                pos, entry = 'call', p
            elif side == 'sell' and trades:
                pos, entry = 'put', p
    if pos == 'call':
        r = (last_close - entry) / entry
        eq *= (1 + lev * r) * (1 - cost_rt); trades.append(lev * r)
    elif pos == 'put':
        r = (last_close - entry) / entry
        eq *= (1 + lev * (-r)) * (1 - cost_rt); trades.append(lev * (-r))
    w = sum(1 for x in trades if x > 0)
    return {'eur': round(eq, 2), 'n': len(trades),
            'hit': round(100 * w / len(trades), 1) if trades else 0.0}

def _params(d):
    """Signal-Parameter aus Request lesen + sinnvoll klemmen."""
    def clamp(v, lo, hi, dv):
        try:
            return max(lo, min(hi, type(dv)(v)))
        except (TypeError, ValueError):
            return dv
    return dict(
        rsi_buy=clamp(d.get('rsi_buy'), 50, 95, 70),
        rsi_sell=clamp(d.get('rsi_sell'), 5, 50, 30),
        min_gap=clamp(d.get('min_gap'), 0, 600, 0),
        zero_line=bool(d.get('zero_line', False)),
        macd_fast=clamp(d.get('macd_fast'), 2, 30, 12),
        macd_slow=clamp(d.get('macd_slow'), 5, 60, 26),
        macd_sig=clamp(d.get('macd_sig'), 2, 20, 9),
        max_sig=clamp(d.get('max_sig'), 0, 50, 0),
    )

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

@app.route('/api/signals', methods=['POST'])
def api_signals():
    """Live-Signale + Flip-3x-Backtest für die aktuellen Slider-Parameter."""
    d = request.json or {}
    df = load_data(d.get('symbol', 'AAPL'), d.get('start_date'), d.get('end_date'))
    if df is None:
        return jsonify({'error': 'Keine Daten verfügbar'}), 400
    p = _params(d)
    sig = signal_rule(df, **p)
    pbt, lc, fc = _price_index(df)
    return jsonify({
        'signals': sig,
        'params': p,
        'stats': {
            'n':      len(sig),
            'hit':    backtest_flip(sig, pbt, lc)['hit'],
            'eur_0':  backtest_flip(sig, pbt, lc, cost_rt=0.0)['eur'],
            'eur_05': backtest_flip(sig, pbt, lc, cost_rt=0.005)['eur'],
            'eur_1':  backtest_flip(sig, pbt, lc, cost_rt=0.01)['eur'],
            'bh':     round((lc - fc) / fc * 100, 2),
        }
    })

@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    """Grid-Search über RSI-Schwellen mit In-Sample/Out-of-Sample-Split
    (gegen Curve-Fitting) + Simplex-LP-Nostalgie (Kapitalaufteilung)."""
    d = request.json or {}
    sym = d.get('symbol', 'AAPL')
    df = load_data(sym, d.get('start_date'), d.get('end_date'))
    if df is None:
        return jsonify({'error': 'Keine Daten verfügbar'}), 400
    base = _params(d)
    cut = int(len(df) * 0.65)
    is_df, oos_df = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    is_pbt,  is_lc,  _  = _price_index(is_df)
    oos_pbt, oos_lc, _  = _price_index(oos_df)

    def score(sub_df, pbt, lc, rb, rs):
        pr = dict(base); pr['rsi_buy'] = rb; pr['rsi_sell'] = rs
        s = signal_rule(sub_df, **pr)
        bt = backtest_flip(s, pbt, lc, cost_rt=0.005)   # realistischer Spread
        return bt['eur'], bt['hit'], bt['n']

    buys  = list(range(55, 86, 5))     # 7
    sells = list(range(15, 46, 5))     # 7  -> 49 Kombis
    grid, best = [], None
    for rs in sells:
        for rb in buys:
            ie, ih, ino = score(is_df,  is_pbt,  is_lc,  rb, rs)
            oe, oh, ono = score(oos_df, oos_pbt, oos_lc, rb, rs)
            cell = {'rsi_buy': rb, 'rsi_sell': rs,
                    'is_eur': ie, 'is_hit': ih,
                    'oos_eur': oe, 'oos_hit': oh}
            grid.append(cell)
            if best is None or ie > best['is_eur']:
                best = cell

    # ── Simplex-LP-Nostalgie: Kapitalaufteilung über die Symbole ──────────
    # Ehrlich: optimiert NUR die Aufteilung gegebener Strategien, NICHT die
    # Signalregel (die ist nicht-konvex -> Grid oben). Klassisches LP:
    #   max  Σ netᵢ·xᵢ   s.t.  Σ xᵢ = 1000,  0 ≤ xᵢ ≤ 400
    syms = get_available_symbols()
    nets, rows = [], []
    for s in syms:
        sdf = load_data(s, d.get('start_date'), d.get('end_date'))
        if sdf is None or sdf.empty:
            continue
        spbt, slc, _ = _price_index(sdf)
        bt = backtest_flip(signal_rule(sdf, **base), spbt, slc, cost_rt=0.005)
        net = bt['eur'] / 1000.0 - 1.0
        nets.append(net); rows.append({'symbol': s, 'net_pct': round(net * 100, 2)})
    lp = None
    if nets:
        n = len(nets)
        res = linprog(c=[-x for x in nets],                 # max -> min(-)
                      A_eq=[[1.0] * n], b_eq=[1000.0],
                      bounds=[(0.0, 400.0)] * n,
                      method='highs-ds')                     # HiGHS Dual-Simplex
        if res.success:
            for r, x in zip(rows, res.x):
                r['alloc_eur'] = round(float(x), 2)
            lp = {
                'method': 'HiGHS Dual-Simplex (scipy.optimize.linprog)',
                'objective': 'max Σ netᵢ·xᵢ  |  Σxᵢ=1000 €,  0≤xᵢ≤400 €',
                'rows': rows,
                'profit_eur': round(float(-res.fun), 2),
                'note': ('Nostalgie/Demo: LP optimiert die KAPITALAUFTEILUNG '
                         'gegebener Strategien — NICHT die Signalregel. '
                         'Ein fixer 6-Tage-Datensatz hat keine Aussagekraft.')
            }

    return jsonify({
        'grid': grid, 'buys': buys, 'sells': sells, 'best_is': best,
        'split': {'is_bars': cut, 'oos_bars': len(df) - cut},
        'lp': lp,
        'caveat': ('In-Sample optimiert, Out-of-Sample gemessen. Großer '
                   'IS→OOS-Abfall = Curve-Fitting. 3x-Modell ohne '
                   'Theta/Vega/Knock-out, ein Symbol/Zeitraum.')
    })

def _pick_som_examples(p):
    """Picked vier Lesebeispiele aus dem SOM-Cache:
    Schoenwetter (Index-Knoten max med_ret), Sturm (min med_ret),
    Heutiges Regime (letzter Trajektorien-Punkt), Bestes Edge-Beispiel.
    Jede Pick = dict mit title, subtitle, kv-Liste, takeaway."""
    tiles = p['index_faces']['tiles']
    eligible = [t for t in tiles if t['n'] >= 20]
    if not eligible:
        return None

    def pct_signed(x, dec=2):
        return f"{x*100:+.{dec}f} %"

    def pct_abs(x, dec=2):
        return f"{x*100:.{dec}f} %"

    def fmt_tile(tile, title, takeaway):
        r = tile['raw']
        return {
            'title': title,
            'subtitle': f"Knoten {tile['node']} (r{tile['row']},c{tile['col']}) "
                        f"&middot; {tile['n']} Tage im Cluster",
            'kv': [
                ('Median-Tagesrendite', pct_signed(r.get('med_ret', 0))),
                ('Markt-Vol (20T)', pct_abs(r.get('mkt_vol20', 0))),
                ('Breadth (&Uuml;ber SMA50)', f"{r.get('breadth',0)*100:.0f} %"),
                ('Dispersion', pct_abs(r.get('disp', 0))),
            ],
            'takeaway': takeaway,
        }

    # Mediane fuer "ruhig vs stuermisch"-Filter
    med_vol = sorted(t['raw'].get('mkt_vol20', 0) for t in eligible)[len(eligible)//2]
    med_brd = sorted(t['raw'].get('breadth', 0)  for t in eligible)[len(eligible)//2]

    # Schoenwetter = Aufwaerts-Median + breite Beteiligung + Ruhe.
    # Pick max med_ret unter den Knoten, die ALLE drei Bedingungen erfuellen.
    # Fallback: Knoten mit hoechster Breadth und med_ret>0.
    sun_cands = [t for t in eligible
                 if t['raw'].get('med_ret', 0) > 0
                 and t['raw'].get('breadth', 0) >= med_brd
                 and t['raw'].get('mkt_vol20', 0) <= med_vol]
    if sun_cands:
        sun = max(sun_cands, key=lambda t: t['raw'].get('med_ret', 0))
    else:
        pos = [t for t in eligible if t['raw'].get('med_ret', 0) > 0]
        sun = max(pos or eligible, key=lambda t: t['raw'].get('breadth', 0))

    # Sturm = Abwaerts-Median + schmale Beteiligung + hohe Vol.
    storm_cands = [t for t in eligible
                   if t['raw'].get('med_ret', 0) < 0
                   and t['raw'].get('breadth', 0) <= med_brd
                   and t['raw'].get('mkt_vol20', 0) >= med_vol]
    if storm_cands:
        storm = min(storm_cands, key=lambda t: t['raw'].get('med_ret', 0))
    else:
        storm = min(eligible, key=lambda t: t['raw'].get('med_ret', 0))

    picks = []
    picks.append(fmt_tile(
        sun, "Sch&ouml;nwetter-Regime",
        "Tage in diesem Cluster sind im Median klar positiv, Vola "
        "niedrig, viele Aktien &uuml;ber ihrer 50-Tage-Linie &mdash; "
        "klassische ruhige Aufw&auml;rtsphase."))
    picks.append(fmt_tile(
        storm, "Sturm-Regime",
        "Im Median klar negativer Tag, hohe 20-Tage-Vola, wenige "
        "Aktien &uuml;ber ihrer 50-Tage-Linie &mdash; typisches Sell-off-"
        "/Stress-Cluster."))

    # Heutiges Regime = letzter Punkt der Trajektorie + dessen Tile
    traj = p.get('trajectory') or []
    if traj:
        last = traj[-1]
        tile_today = next((t for t in tiles if t['node'] == last['node']), None)
        if tile_today is not None:
            picks.append({
                'title': 'Heutiges Regime',
                'subtitle': f"Stand {last['date']} &middot; "
                            f"Knoten {last['node']} (r{last['row']},c{last['col']}) "
                            f"&middot; Cluster mit {tile_today['n']} Tagen historisch",
                'kv': [
                    ('Tag-Median im Cluster', pct_signed(tile_today['raw'].get('med_ret', 0))),
                    ('Markt-Vol (20T) heute', pct_abs(last.get('mkt_vol20', 0))),
                    ('Breadth heute', f"{last.get('breadth',0)*100:.0f} %"),
                    ('Train/OOS-Bucket', last.get('split', '?').upper()),
                ],
                'takeaway': "Der animierte Pfad endet hier &mdash; das ist "
                            "der Markt-Zustand, in dem der j&uuml;ngste "
                            "Handelstag gelandet ist. Beschreibend, kein Signal.",
            })

    # Edge: Knoten mit niedrigstem p, sofern <0.05 — sonst groesster |edge_vs_base| mit n>=30
    edge_nodes = (p.get('edge') or {}).get('nodes', [])
    base_rate = (p.get('edge') or {}).get('base_rate')
    if edge_nodes:
        sig = [n for n in edge_nodes if n.get('p_value') is not None
               and float(n['p_value']) < 0.05]
        if sig:
            best = min(sig, key=lambda n: float(n['p_value']))
            head = "Auff&auml;lligster Edge-Knoten (p&lt;0,05)"
        else:
            cands = [n for n in edge_nodes if n.get('n', 0) >= 30]
            if cands:
                best = max(cands, key=lambda n: abs(float(n.get('edge_vs_base', 0))))
                head = "St&auml;rkster Edge-Knoten (deskriptiv, p&ge;0,05)"
            else:
                best = None
        if best is not None:
            ev = float(best.get('edge_vs_base', 0))
            picks.append({
                'title': head,
                'subtitle': f"Knoten {best['node']} (r{best['row']},c{best['col']}) "
                            f"&middot; {best['n']} OOS-Signaltage",
                'kv': [
                    ('Trefferquote im Knoten', f"{float(best['hit_rate'])*100:.1f} %"),
                    ('OOS-Basisrate (alle Signale)',
                     f"{float(base_rate)*100:.1f} %" if base_rate else "&mdash;"),
                    ('&Delta; gegen Basis', f"{ev*100:+.2f}&nbsp;%-Punkte"),
                    ('p-Wert (unkorrigiert)', f"{float(best['p_value']):.3f}"),
                ],
                'takeaway': ("Bei vielen belegten Knoten taucht zuf&auml;llig "
                             "~5&nbsp;% mit p&lt;0,05 auf &mdash; ein "
                             "einzelner Treffer ist <b>Hypothese</b>, "
                             "kein best&auml;tigter Edge. Validierung "
                             "br&auml;uchte Mehrfachtest-Korrektur und "
                             "ein neues, unber&uuml;hrtes Test-Fenster."),
            })

    return picks


@app.route('/som')
def som_view():
    """SOM-Marktregime: Index- + Symbol-Tag-Karte als Chernoff-Faces,
    plus OOS-Edge-Test je Regime-Knoten. EXPLORATIV, kein Prädiktor."""
    import som_regime, pickle, os
    if not os.path.exists(som_regime.CACHE_PATH):
        return render_template('som.html', ready=False)
    with open(som_regime.CACHE_PATH, 'rb') as f:
        p = pickle.load(f)
    examples = _pick_som_examples(p)
    return render_template(
        'som.html', ready=True,
        meta=p['meta'], split_date=p['split_date'],
        index_faces=p['index_faces'], symday_faces=p['symday_faces'],
        edge=p['edge'], trajectory=p['trajectory'],
        examples=examples,
        index_qe=round(p['index_map']['qe'], 3),
        symday_qe=round(p['symday_map']['qe'], 3))


@app.route('/som/rebuild', methods=['POST'])
def som_rebuild():
    """Cache neu bauen (~35 s, lädt bb_StockPrices komplett)."""
    import som_regime
    try:
        som_regime.build_all(force=True)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)
