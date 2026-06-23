"""
INTRADAY-VOLUMEN-PROFIL — 2026-06-23
Frage (Nutzer): Gibt es Handelszeiten mit viel/wenig Volumen? 2h-Fenster.
Gibt es eine Mittagspause?

Datenbasis:
  - PRIMAER: bb_StockPrices_1min_YFinance (56k Zeilen, ~30 Tage Mai-Jun 2026,
    AAPL/MSFT/GOOGL/AMZN/TSLA). Timestamps in UTC.
  - CROSS-CHECK: bb_StockPrices_1min_Kaggle (7 Tage Mai 2023, gleiche 5 Symbole).
    Timestamps in ET (native).

Zeitzone: Beide Perioden liegen komplett in EDT (Sommerzeit, UTC-4), keine
DST-Bruchstelle. -> YFinance: ET = UTC - 4h. Kaggle: bereits ET.

Methodik (entscheidend gegen Verzerrung):
  - Normalisierung pro (Symbol, Handelstag): jede Minute = Anteil am
    Tages-Gesamtvolumen dieses Symbols. So zaehlt jeder Symbol-Tag gleich viel,
    egal ob TSLA-Hochvol-Tag oder ruhiger MSFT-Tag.
  - Kennzahl je Zeitfenster:
      * Anteil am Tagesvolumen (%)
      * REL. INTENSITAET = Anteil / (Minuten-Anteil des Fensters am Tag).
        1.0 = Durchschnitt, >1 = ueberdurchschnittlich (Open/Close),
        <1 = unterdurchschnittlich (Mittagsdelle).
"""
import pymssql
import numpy as np
import pandas as pd

DB = dict(server='158.181.48.77', user='326773', password='Extaler11!', database='dbdata')

RTH_OPEN_MIN = 9 * 60 + 30   # 570  (09:30 ET)
RTH_CLOSE_MIN = 16 * 60      # 960  (16:00 ET)
TRADING_MINUTES = RTH_CLOSE_MIN - RTH_OPEN_MIN  # 390

# 2-Stunden-Fenster (Rest 15:30-16:00 = 30 Min)
WINDOWS = [
    ('09:30-11:30', 570, 690),
    ('11:30-13:30', 690, 810),
    ('13:30-15:30', 810, 930),
    ('15:30-16:00', 930, 960),
]


def load(table, utc_offset_hours):
    conn = pymssql.connect(**DB)
    df = pd.read_sql(f'SELECT Symbol, [Timestamp], Volume FROM {table}', conn)
    conn.close()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # in ET-Marktzeit bringen
    df['et'] = df['Timestamp'] - pd.Timedelta(hours=utc_offset_hours)
    df['date'] = df['et'].dt.date
    df['min_et'] = df['et'].dt.hour * 60 + df['et'].dt.minute
    # nur regulaere Handelszeit (RTH), Volumen > 0
    df = df[(df['min_et'] >= RTH_OPEN_MIN) & (df['min_et'] < RTH_CLOSE_MIN)]
    df = df[df['Volume'] > 0].copy()
    return df


def normalize(df):
    """Jede Minute -> Anteil am Tages-Gesamtvolumen des Symbols."""
    daily = df.groupby(['Symbol', 'date'])['Volume'].transform('sum')
    df = df[daily > 0].copy()
    df['share'] = df['Volume'] / daily
    return df


def profile_by_buckets(df, edges_labels):
    """edges_labels: Liste (label, start_min, end_min).
    Rueckgabe: DataFrame je Bucket mit mean-Anteil, std, rel. Intensitaet."""
    # Bucket je Zeile zuweisen
    def assign(m):
        for lbl, a, b in edges_labels:
            if a <= m < b:
                return lbl
        return None
    df = df.copy()
    df['bucket'] = df['min_et'].apply(assign)
    df = df[df['bucket'].notna()]
    # je (Symbol, Tag, Bucket): Summe der Anteile = Tagesanteil dieses Buckets
    per = df.groupby(['Symbol', 'date', 'bucket'])['share'].sum().reset_index()
    rows = []
    for lbl, a, b in edges_labels:
        sub = per[per['bucket'] == lbl]['share']
        n_min = b - a
        mean_share = sub.mean()
        rel = mean_share / (n_min / TRADING_MINUTES)  # 1.0 = flach
        rows.append({
            'fenster': lbl, 'minuten': n_min,
            'anteil_%': mean_share * 100,
            'std_%': sub.std() * 100,
            'rel_intensitaet': rel,
            'n_symtage': len(sub),
        })
    return pd.DataFrame(rows)


def bars(val, scale, width=40, ch='#'):
    n = int(round(val / scale * width))
    return ch * max(0, n)


def thirty_min_buckets():
    out = []
    m = RTH_OPEN_MIN
    while m < RTH_CLOSE_MIN:
        end = min(m + 30, RTH_CLOSE_MIN)
        lbl = f'{m//60:02d}:{m%60:02d}-{end//60:02d}:{end%60:02d}'
        out.append((lbl, m, end))
        m += 30
    return out


def run(table, utc_off, name):
    print('\n' + '=' * 78)
    print(f'{name}   ({table}, ET-Marktzeit)')
    print('=' * 78)
    df = load(table, utc_off)
    df = normalize(df)
    ndays = df['date'].nunique()
    nsym = df['Symbol'].nunique()
    print(f'{len(df):,} Minuten-Bars | {nsym} Symbole | {ndays} Handelstage '
          f'| {df["date"].min()} .. {df["date"].max()}')

    print('\n--- 2-STUNDEN-FENSTER ---')
    w = profile_by_buckets(df, WINDOWS)
    print(f"{'Fenster':<13}{'Min':>4}{'Anteil%':>9}{'±std':>7}{'rel.Int':>9}  Intensitaet (1.0=flach)")
    for _, r in w.iterrows():
        marker = '  <-- ' + ('SPITZE' if r['rel_intensitaet'] >= 1.15
                             else ('DELLE' if r['rel_intensitaet'] <= 0.85 else ''))
        print(f"{r['fenster']:<13}{int(r['minuten']):>4}{r['anteil_%']:>9.2f}"
              f"{r['std_%']:>7.2f}{r['rel_intensitaet']:>9.2f}  "
              f"{bars(r['rel_intensitaet'],1.0,28,'#')}{marker.rstrip()}")

    print('\n--- 30-MINUTEN-PROFIL (rel. Intensitaet, 1.0 = Tagesdurchschnitt) ---')
    h = profile_by_buckets(df, thirty_min_buckets())
    peak = h.loc[h['rel_intensitaet'].idxmax()]
    trough = h.loc[h['rel_intensitaet'].idxmin()]
    for _, r in h.iterrows():
        tag = ''
        if r['fenster'] == peak['fenster']:
            tag = ' <== MAX'
        elif r['fenster'] == trough['fenster']:
            tag = ' <== MIN'
        print(f"{r['fenster']:<13}{r['rel_intensitaet']:>5.2f} |{bars(r['rel_intensitaet'],2.2,46,'#')}{tag}")
    return df, w, h


def main():
    print('=' * 78)
    print('INTRADAY-VOLUMEN-PROFIL — viel/wenig Handel & Mittagspause?')
    print('=' * 78)
    _, wY, hY = run('bb_StockPrices_1min_YFinance', 4, 'PRIMAER: YFinance Mai-Jun 2026')
    _, wK, hK = run('bb_StockPrices_1min_Kaggle', 0, 'CROSS-CHECK: Kaggle Mai 2023')

    print('\n' + '=' * 78)
    print('MITTAGS-CHECK (12:00-13:30 ET vs Open & Close)')
    print('=' * 78)
    for name, h in [('YFinance 2026', hY), ('Kaggle 2023', hK)]:
        midday = h[h['fenster'].isin(['12:00-12:30', '12:30-13:00', '13:00-13:30'])]['rel_intensitaet'].mean()
        openw = h[h['fenster'] == '09:30-10:00']['rel_intensitaet'].values[0]
        closew = h[h['fenster'] == '15:30-16:00']['rel_intensitaet'].values[0]
        print(f'{name:<16} Open(9:30-10:00)={openw:.2f}  Mittag(12:00-13:30)={midday:.2f}  '
              f'Close(15:30-16:00)={closew:.2f}  -> Mittag ist {(1-midday)*100:.0f}% unter Schnitt')


if __name__ == '__main__':
    main()
