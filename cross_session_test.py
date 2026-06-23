"""
CROSS-SESSION-TEST — 2026-06-23
Frage (Nutzer): Vergleiche Nachmittag[t] mit Vormittag[t+1] — und so weiter.
=> Sagt eine Tageshaelfte die naechste (ueber Nacht hinweg) vorher?

Zerlegung je Handelstag (ET):
  Vormittag  AM[t] = mid(13:00) / open(09:30) - 1
  Nachmittag PM[t] = close(16:00) / mid(13:00) - 1
  Overnight  GAP[t] = open(09:30)[t] / close(16:00)[t-1] - 1   (dazwischen)

Kette der Zeit: ... PM[t-1]  GAP[t]  AM[t]  PM[t]  GAP[t+1]  AM[t+1] ...
Lag-1-Korrelationen + Vorzeichen-Trefferquoten je Uebergang, IMMER innerhalb
desselben Symbols und aufeinanderfolgender Handelstage.

Datenquellen:
  - bb_StockPrices_1min_YFinance (5 Aktien, ~30 Tage)  [Minuten, UTC/EDT]
  - spy_30min_sip_2016_2026.pkl (SPY, 2016-2026)        [30-Min, definitiv]
"""
import pickle
import numpy as np
import pandas as pd
import pymssql

DB = dict(server='158.181.48.77', user='326773', password='Extaler11!', database='dbdata')
MID_ET = '13:00'  # Vormittag/Nachmittag-Grenze


def segs_from_minute():
    conn = pymssql.connect(**DB)
    df = pd.read_sql("SELECT Symbol,[Timestamp],OpenPrice,ClosePrice "
                     "FROM bb_StockPrices_1min_YFinance ORDER BY Symbol,[Timestamp]", conn)
    conn.close()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['date'] = df['Timestamp'].dt.date
    df['umin'] = df['Timestamp'].dt.hour * 60 + df['Timestamp'].dt.minute
    mid_min = 17 * 60  # 13:00 ET = 17:00 UTC (EDT)
    rows = []
    for (s, d), g in df.groupby(['Symbol', 'date']):
        g = g.sort_values('Timestamp')
        before = g[g['umin'] < mid_min]
        if len(before) < 30 or len(g) - len(before) < 30:
            continue
        rows.append(dict(Symbol=s, date=d,
                         openp=float(g.iloc[0]['OpenPrice']),
                         midp=float(before.iloc[-1]['ClosePrice']),
                         closep=float(g.iloc[-1]['ClosePrice'])))
    return pd.DataFrame(rows)


def segs_from_sip():
    with open('spy_30min_sip_2016_2026.pkl', 'rb') as f:
        bars = pickle.load(f)
    df = pd.DataFrame(bars)
    df['et'] = pd.to_datetime(df['t'], utc=True).dt.tz_convert('America/New_York')
    df['date'] = df['et'].dt.date
    df['tod'] = df['et'].dt.strftime('%H:%M')
    rows = []
    for d, g in df.groupby('date'):
        gi = g.set_index('tod')
        if not {'09:30', '12:30', '15:30'}.issubset(gi.index):
            continue
        rows.append(dict(Symbol='SPY', date=d,
                         openp=float(gi.loc['09:30', 'o']),
                         midp=float(gi.loc['12:30', 'c']),
                         closep=float(gi.loc['15:30', 'c'])))
    return pd.DataFrame(rows)


def analyze(seg, name):
    seg = seg.sort_values(['Symbol', 'date']).copy()
    seg['AM'] = seg['midp'] / seg['openp'] - 1
    seg['PM'] = seg['closep'] / seg['midp'] - 1
    grp = seg.groupby('Symbol')
    seg['prev_close'] = grp['closep'].shift(1)
    seg['GAP'] = seg['openp'] / seg['prev_close'] - 1
    # "naechste" Segmente (innerhalb Symbol)
    seg['AM_next'] = grp['AM'].shift(-1)
    seg['PM_next'] = grp['PM'].shift(-1)
    seg['GAP_next'] = grp['GAP'].shift(-1)

    def stat(a, b):
        m = seg[[a, b]].dropna()
        if len(m) < 10:
            return None
        c = np.corrcoef(m[a], m[b])[0, 1]
        hit = (np.sign(m[a]) == np.sign(m[b])).mean() * 100
        return c, hit, len(m)

    print('\n' + '=' * 78)
    print(f'{name}')
    print('=' * 78)
    print(f"{'Uebergang':<34}{'corr':>8}{'Hit%':>8}{'n':>7}")
    transitions = [
        ('AM[t]  -> PM[t]   (gleicher Tag)', 'AM', 'PM'),
        ('PM[t]  -> GAP[t+1] (ueber Nacht)', 'PM', 'GAP_next'),
        ('PM[t]  -> AM[t+1]  (Nachm.->Vorm.)', 'PM', 'AM_next'),
        ('GAP[t] -> AM[t]    (Gap->Vormittag)', 'GAP', 'AM'),
        ('PM[t]  -> PM[t+1]', 'PM', 'PM_next'),
        ('AM[t]  -> AM[t+1]', 'AM', 'AM_next'),
    ]
    res = {}
    for lbl, a, b in transitions:
        r = stat(a, b)
        if r:
            c, hit, n = r
            flag = '  <==' if a == 'PM' and b == 'AM_next' else ''
            print(f'{lbl:<34}{c:>+8.3f}{hit:>7.1f}%{n:>7}{flag}')
            res[(a, b)] = r
    return res


def main():
    print('=' * 78)
    print('CROSS-SESSION: sagt eine Tageshaelfte die naechste vorher?')
    print('=' * 78)
    rs = analyze(segs_from_sip(), 'SPY 30-Min SIP 2016-2026 (DEFINITIV, ~2630 Naechte)')
    rm = analyze(segs_from_minute(), 'YFinance-Minuten 5 Aktien (~30 Tage)')

    print('\n' + '=' * 78)
    print('FAZIT — Nachmittag[t] -> Vormittag[t+1]')
    print('=' * 78)
    for tag, r in [('SPY 2016-2026', rs), ('5 Aktien 30 Tage', rm)]:
        c, hit, n = r[('PM', 'AM_next')]
        print(f'  {tag:<18} corr={c:+.3f}  Hit={hit:.1f}%  (n={n})')
    print('  corr~0 & Hit~50% => kein Vorhersagewert ueber Nacht (Random Walk).')


if __name__ == '__main__':
    main()
