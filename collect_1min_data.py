import yfinance as yf
import pymssql
from config import DB_CONFIG
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    filename='logs/1min_collection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

logging.info('Starting 1-minute data collection...')

conn = pymssql.connect(**DB_CONFIG, autocommit=True)
cursor = conn.cursor()

end_date = datetime.now()
start_date = end_date - timedelta(days=7)

total_imported = 0
total_dupes = 0
total_errors = 0

for symbol in symbols:
    try:
        df = yf.download(symbol, start=start_date.date(), end=end_date.date(), interval='1m', progress=False)

        if df is None or len(df) == 0:
            logging.warning(symbol + ': No data')
            continue

        df = df.reset_index()

        # yfinance liefert seit v0.2.x MultiIndex-Spalten -> flatten
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)

        # Datetime-Spalte: Timezone entfernen
        dt_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
        df[dt_col] = df[dt_col].dt.tz_localize(None)

        rows_added = 0
        rows_dupe = 0

        for idx, row in df.iterrows():
            try:
                cursor.execute(
                    'INSERT INTO bb_StockPrices_1min_YFinance (Symbol, Timestamp, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume) VALUES (%s, %s, %s, %s, %s, %s, %s)',
                    (symbol, row[dt_col], float(row['Open']), float(row['High']), float(row['Low']), float(row['Close']), int(row['Volume']))
                )
                rows_added += 1
            except pymssql.IntegrityError:
                # PK/UNIQUE-Verletzung (Fehler 2627) = Zeile schon vorhanden -> erwartete Dublette
                rows_dupe += 1
            except Exception as e:
                # echter, unerwarteter Fehler -> NICHT verschlucken, ins Log schreiben
                logging.error(symbol + ' insert: ' + str(e)[:200])
                total_errors += 1

        logging.info(symbol + ': Added=' + str(rows_added) + ', Duplicates=' + str(rows_dupe))
        total_imported += rows_added
        total_dupes += rows_dupe

    except Exception as e:
        logging.error(symbol + ': ' + str(e)[:200])
        total_errors += 1

cursor.close()
conn.close()

logging.info('Collection complete: ' + str(total_imported) + ' rows, ' + str(total_dupes) + ' duplicates, ' + str(total_errors) + ' errors')
print('OK: ' + str(total_imported) + ' rows imported, ' + str(total_dupes) + ' duplicates')
