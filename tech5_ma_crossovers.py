#!/usr/bin/env python3
"""
MA20/MA50 Crossover-Report fuer die Top-5 Tech-Aktien.
Schreibt HTML auf den Desktop.
"""
import os
from datetime import datetime, timedelta

import yfinance as yf

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "NVDA": "Nvidia",
}
YEARS_BACK = 5
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
OUT = os.path.join(DESKTOP, "tech5_ma_crossovers.html")


def find_crossovers(df):
    """Liefert Liste von Crossover-Events: (date, close, direction).
    direction: 'golden' (MA20 schneidet MA50 von unten) oder 'death' (von oben).
    """
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df = df.dropna(subset=["MA20", "MA50"])

    events = []
    prev_diff = None
    for idx, row in df.iterrows():
        diff = float(row["MA20"]) - float(row["MA50"])
        if prev_diff is not None:
            if prev_diff <= 0 and diff > 0:
                events.append((idx, float(row["Close"]), "golden"))
            elif prev_diff >= 0 and diff < 0:
                events.append((idx, float(row["Close"]), "death"))
        prev_diff = diff
    return events, df


def price_range_between(df, start, end):
    """High-Low-Spannweite zwischen zwei Zeitpunkten (inkl. Endpunkten)."""
    mask = (df.index >= start) & (df.index <= end)
    sub = df.loc[mask]
    if sub.empty:
        return None
    hi = float(sub["High"].max())
    lo = float(sub["Low"].min())
    return hi, lo


def build_html(per_ticker):
    css = """
    <style>
      body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
             color: #1d1d1f; max-width: 980px; margin: 0 auto; padding: 18px;
             background: #fafafa; }
      h1 { font-size: 24px; border-bottom: 3px solid #1d1d1f; padding-bottom: 8px; }
      h2 { font-size: 20px; margin-top: 30px; color: #1d1d1f;
           padding: 6px 10px; background: #eef1f7; border-left: 5px solid #2a4a8a;
           border-radius: 4px; }
      table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 14px;
              background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
      th, td { border: 1px solid #e3e3e3; padding: 8px 10px; text-align: right; }
      th { background: #f4f4f4; text-align: center; font-weight: 600; }
      td.left, th.left { text-align: left; }
      .gold { color: #0a7a1f; font-weight: 700; background: #e6f7ea; padding: 2px 8px;
              border-radius: 4px; }
      .death { color: #c41313; font-weight: 700; background: #fde9e9; padding: 2px 8px;
               border-radius: 4px; }
      .muted { color: #888; font-size: 12px; }
      .stat { font-weight: 600; color: #2a4a8a; }
      .arrow-up { color: #0a7a1f; }
      .arrow-dn { color: #c41313; }
      .summary { margin: 10px 0 6px 0; font-size: 13px; color: #555; }
    </style>
    """
    now = datetime.now()
    html = ["<html><head><meta charset='utf-8'>", css,
            f"<title>Top-5 Tech MA20/MA50 Crossover</title></head><body>"]
    html.append(f"<h1>Top-5 Tech &mdash; MA20 / MA50 Crossover-Historie</h1>")
    html.append(f"<p class='muted'>Stand: {now.strftime('%Y-%m-%d %H:%M')} &middot; "
                f"Zeitraum: letzte {YEARS_BACK} Jahre &middot; "
                f"Datenquelle: Yahoo Finance (yfinance)</p>")
    html.append("<p class='summary'>"
                "<span class='gold'>&#9650; Golden Cross</span> = MA20 schneidet MA50 von unten (bullish). "
                "<span class='death'>&#9660; Death Cross</span> = MA20 schneidet MA50 von oben (bearish). "
                "Spannweite = High/Low-Range bis zum naechsten Crossover.</p>")

    for tk in TICKERS:
        events, df = per_ticker[tk]
        html.append(f"<h2>{NAMES[tk]} &mdash; {tk}</h2>")

        if not events:
            html.append("<p class='muted'>Keine Crossover im Zeitraum.</p>")
            continue

        last_close = float(df["Close"].iloc[-1])
        first_date = df.index[0].strftime("%Y-%m-%d")
        last_date = df.index[-1].strftime("%Y-%m-%d")
        html.append(f"<p class='summary'>Datenfenster: {first_date} bis {last_date} &middot; "
                    f"aktueller Schlusskurs: <span class='stat'>{last_close:,.2f} USD</span> &middot; "
                    f"Crossover-Ereignisse: <span class='stat'>{len(events)}</span></p>")

        html.append("<table>")
        html.append("<tr><th>#</th><th class='left'>Datum</th><th class='left'>Typ</th>"
                    "<th>Kurs am Crossover</th><th>Naechstes Ereignis</th>"
                    "<th>Tage bis Naechstes</th><th>Spannweite (Low - High)</th>"
                    "<th>Spannweite %</th></tr>")

        for i, (dt, close, direction) in enumerate(events, 1):
            if i < len(events):
                next_dt = events[i][0]
                next_label = next_dt.strftime("%Y-%m-%d")
            else:
                next_dt = df.index[-1]
                next_label = f"heute ({next_dt.strftime('%Y-%m-%d')})"

            days = (next_dt - dt).days

            rng = price_range_between(df, dt, next_dt)
            if rng:
                hi, lo = rng
                span_str = f"{lo:,.2f} &mdash; {hi:,.2f}"
                if close > 0:
                    span_pct = (hi - lo) / close * 100
                    span_pct_str = f"{span_pct:.1f}%"
                else:
                    span_pct_str = "&mdash;"
            else:
                span_str = "&mdash;"
                span_pct_str = "&mdash;"

            if direction == "golden":
                badge = "<span class='gold'>&#9650; Golden Cross</span>"
            else:
                badge = "<span class='death'>&#9660; Death Cross</span>"

            html.append(
                f"<tr>"
                f"<td>{i}</td>"
                f"<td class='left'>{dt.strftime('%Y-%m-%d')}</td>"
                f"<td class='left'>{badge}</td>"
                f"<td>{close:,.2f} USD</td>"
                f"<td class='left'>{next_label}</td>"
                f"<td>{days}</td>"
                f"<td>{span_str} USD</td>"
                f"<td>{span_pct_str}</td>"
                f"</tr>"
            )
        html.append("</table>")

    html.append(f"<p class='muted' style='margin-top:40px'>"
                f"Generiert mit yfinance &middot; MA-Berechnung auf Tagesschlusskursen "
                f"&middot; Spannweite = max(High) und min(Low) im Intervall</p>")
    html.append("</body></html>")
    return "".join(html)


def main():
    end = datetime.now()
    start = end - timedelta(days=365 * YEARS_BACK + 80)  # +80 Tage Buffer fuer MA50

    per_ticker = {}
    for tk in TICKERS:
        print(f"Lade {tk} ...", flush=True)
        df = yf.download(tk, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"),
                         progress=False, auto_adjust=False)
        if df.empty:
            print(f"  WARN: keine Daten fuer {tk}")
            per_ticker[tk] = ([], df)
            continue
        if isinstance(df.columns, type(df.columns)) and hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        events, df = find_crossovers(df)
        # Erst Events nach Beginn des Anzeigefensters behalten
        cutoff = end - timedelta(days=365 * YEARS_BACK)
        events = [(d, c, dr) for (d, c, dr) in events if d >= cutoff]
        print(f"  {len(events)} Crossover-Events seit {cutoff.strftime('%Y-%m-%d')}")
        per_ticker[tk] = (events, df)

    html = build_html(per_ticker)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nGeschrieben: {OUT}")


if __name__ == "__main__":
    main()
