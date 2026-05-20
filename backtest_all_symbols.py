#!/usr/bin/env python3
"""Cross-Symbol-Backtest mit User-Sliderwerten aus dem Kaggle Data Explorer.

Parameter aus dem AAPL-Screenshot 2026-05-20:
  RSI Buy < 58, RSI Sell > 45, Entprellung 0 Bars
  MACD fast 12, slow 26, signal 12, max Signale 13
  zero_line = False
  Zeitraum 2023-05-15 bis 2023-05-23
  Leverage 3x (Flip-Strategie), Start 1000 EUR

Druckt Tabelle: Symbol | Signals | Hit% | EUR_0 | EUR_0.5% | EUR_1% | B&H%
"""
import pymssql
import pandas as pd
import numpy as np

DB_CONFIG = {
    'server': '158.181.48.77',
    'database': 'dbdata',
    'user': '326773',
    'password': 'Extaler11!',
    'as_dict': False,
}

# --- User-Parameter aus dem Screenshot ---------------------------------------
PARAMS = dict(
    rsi_buy=58, rsi_sell=45, min_gap=0, zero_line=False,
    macd_fast=12, macd_slow=26, macd_sig=12, max_sig=13,
)
START_DATE = '2023-05-15'
END_DATE   = '2023-05-23'
LEV        = 3.0
START_EUR  = 1000.0


# --- Indikatoren (1:1 aus analysis_app.py) -----------------------------------
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


def to_lw(ts):
    return int(ts.timestamp())


def signal_rule(df, rsi_buy=70, rsi_sell=30, min_gap=0, zero_line=False,
                macd_fast=12, macd_slow=26, macd_sig=9, max_sig=0):
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
        out.append({'time': to_lw(ts.iloc[i]),
                    'side': 'buy' if b else 'sell',
                    '_str': abs(float(hv[i]))})
        last_i = i
    if max_sig and len(out) > max_sig:
        keep = sorted(sorted(range(len(out)),
                             key=lambda k: -out[k]['_str'])[:max_sig])
        out = [out[k] for k in keep]
    for o in out:
        o.pop('_str', None)
    return out


def backtest_flip(signals, pbt, last_close, start=1000.0, lev=3.0, cost_rt=0.0):
    eq, pos, entry, trades = start, None, None, []
    for sg in signals:
        p = pbt.get(sg['time'])
        if p is None:
            continue
        side = sg['side']
        if pos == 'call' and side == 'sell':
            r = (p - entry) / entry
            eq *= (1 + lev * r) * (1 - cost_rt); trades.append(lev * r)
            pos = entry = None
        elif pos == 'put' and side == 'buy':
            r = (p - entry) / entry
            eq *= (1 + lev * (-r)) * (1 - cost_rt); trades.append(lev * (-r))
            pos = entry = None
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


# --- Daten laden -------------------------------------------------------------
def get_symbols():
    with pymssql.connect(**DB_CONFIG) as conn:
        df = pd.read_sql(
            "SELECT DISTINCT Symbol FROM bb_StockPrices_1min_Kaggle "
            "ORDER BY Symbol", conn)
    return df['Symbol'].tolist()


def load_data(symbol, start_date, end_date):
    with pymssql.connect(**DB_CONFIG) as conn:
        df = pd.read_sql(
            "SELECT Timestamp, OpenPrice, HighPrice, LowPrice, ClosePrice, "
            f"Volume FROM bb_StockPrices_1min_Kaggle "
            f"WHERE Symbol = '{symbol}' "
            f"AND [Timestamp] >= '{start_date}' "
            f"AND [Timestamp] <= '{end_date} 23:59:59' "
            f"ORDER BY Timestamp", conn)
    if df.empty:
        return None
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['RSI'] = compute_rsi(df['ClosePrice'])
    return df


# --- Main --------------------------------------------------------------------
def main():
    print("="*90)
    print(f"CROSS-SYMBOL-BACKTEST  {START_DATE} - {END_DATE}")
    print(f"Parameter: RSI Buy<{PARAMS['rsi_buy']} / Sell>{PARAMS['rsi_sell']}, "
          f"MACD {PARAMS['macd_fast']}/{PARAMS['macd_slow']}/{PARAMS['macd_sig']}, "
          f"max {PARAMS['max_sig']}, Entprellung {PARAMS['min_gap']}, "
          f"zero_line={PARAMS['zero_line']}")
    print(f"Flip-Strategie {LEV}x Leverage, Start {START_EUR:.0f} EUR")
    print("="*90)

    syms = get_symbols()
    print(f"Symbole im Kaggle-Set: {len(syms)} -> {syms}")
    print()

    rows = []
    for s in syms:
        df = load_data(s, START_DATE, END_DATE)
        if df is None or len(df) < 50:
            rows.append((s, 'kein/wenig Datenpunkt', None, None, None, None, None))
            continue
        sig = signal_rule(df, **PARAMS)
        pbt = {to_lw(t): float(c)
               for t, c in zip(df['Timestamp'], df['ClosePrice'])}
        first_close = float(df['ClosePrice'].iloc[0])
        last_close  = float(df['ClosePrice'].iloc[-1])
        bh_pct      = (last_close - first_close) / first_close * 100
        bt0   = backtest_flip(sig, pbt, last_close, START_EUR, LEV, 0.0)
        bt05  = backtest_flip(sig, pbt, last_close, START_EUR, LEV, 0.005)
        bt1   = backtest_flip(sig, pbt, last_close, START_EUR, LEV, 0.01)
        rows.append((s, bt0['n'], bt0['hit'], bt0['eur'], bt05['eur'],
                     bt1['eur'], bh_pct))

    # --- Tabellenausgabe ----------------------------------------------------
    print(f"{'Symbol':<8} {'Sig':>4} {'Hit%':>6} "
          f"{'EUR_0':>10} {'EUR_0.5%':>10} {'EUR_1%':>10} {'B&H%':>8}")
    print("-"*64)
    for r in rows:
        s, n, hit, e0, e05, e1, bh = r
        if isinstance(n, str):
            print(f"{s:<8} {n}")
            continue
        print(f"{s:<8} {n:>4} {hit:>6.1f} "
              f"{e0:>10.2f} {e05:>10.2f} {e1:>10.2f} {bh:>+8.2f}")
    print()

    # --- Aggregat -----------------------------------------------------------
    valid = [r for r in rows if not isinstance(r[1], str)]
    if valid:
        ns   = [r[1] for r in valid]
        hits = [r[2] for r in valid]
        e0s  = [r[3] for r in valid]
        e05s = [r[4] for r in valid]
        e1s  = [r[5] for r in valid]
        bhs  = [r[6] for r in valid]
        print(f"Aggregat ueber {len(valid)} Symbole:")
        print(f"  Ø Signale     {np.mean(ns):.1f}  (min {min(ns)}, max {max(ns)})")
        print(f"  Ø Hit%        {np.mean(hits):.1f}%")
        print(f"  Ø EUR_0       {np.mean(e0s):.2f}  -> "
              f"{(np.mean(e0s)/START_EUR - 1)*100:+.2f}%")
        print(f"  Ø EUR_0.5%    {np.mean(e05s):.2f}  -> "
              f"{(np.mean(e05s)/START_EUR - 1)*100:+.2f}%")
        print(f"  Ø EUR_1%      {np.mean(e1s):.2f}  -> "
              f"{(np.mean(e1s)/START_EUR - 1)*100:+.2f}%")
        print(f"  Ø B&H%        {np.mean(bhs):+.2f}%")
        win0   = sum(1 for e in e0s  if e > START_EUR)
        win05  = sum(1 for e in e05s if e > START_EUR)
        win1   = sum(1 for e in e1s  if e > START_EUR)
        print(f"  Symbole>1000  {win0}/{len(valid)} (roh)  "
              f"{win05}/{len(valid)} (0.5%)  "
              f"{win1}/{len(valid)} (1%)")


if __name__ == "__main__":
    main()
