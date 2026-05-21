#!/usr/bin/env python3
"""
FRED API Helper - Wrapper für FRED Data Access mit API-Key.

Key-Management:
  1. Aus Umgebungsvariable FRED_API_KEY
  2. Aus C:\\Users\\defco\\boersenbot\\.fred_key (Plain-Text)
  3. Notfalls als Argument

Beispiel:
  from fred_helper import get_series
  data = get_series("BAMLH0A0HYM2", start="2010-01-01")
"""
import os
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime

KEY_FILE = Path(r"C:\Users\defco\boersenbot\.fred_key")
CACHE_DIR = Path(r"C:\Users\defco\boersenbot\.fred_cache")
CACHE_DIR.mkdir(exist_ok=True)


def load_key():
    """Lade FRED API-Key aus env oder Datei."""
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key.strip()

    if KEY_FILE.exists():
        return KEY_FILE.read_text().strip()

    raise RuntimeError(
        "Kein FRED API-Key gefunden!\n"
        "Speichere den Key in einer dieser Stellen:\n"
        f"  1. Datei: {KEY_FILE}\n"
        "  2. Env-Variable: setx FRED_API_KEY \"dein_key\"\n"
    )


def get_series(series_id, start="2000-01-01", end=None, use_cache=True, force_refresh=False):
    """
    Lade FRED-Series via API.

    Args:
        series_id: z.B. "BAMLH0A0HYM2"
        start: ISO-Date (Default 2000-01-01)
        end: ISO-Date (Default heute)
        use_cache: Lokales Cache nutzen
        force_refresh: Cache umgehen

    Returns:
        pd.Series mit Datums-Index
    """
    # Cache-Pfad
    cache_file = CACHE_DIR / f"{series_id}_{start}_{end or 'now'}.csv"

    # Aus Cache?
    if use_cache and not force_refresh and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df['value']

    # API-Call
    api_key = load_key()
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start
    }
    if end:
        params["observation_end"] = end

    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise RuntimeError(f"FRED API-Fehler fuer {series_id}: {e}")

    if "observations" not in data:
        raise RuntimeError(f"FRED unerwartete Antwort: {data}")

    # In DataFrame
    obs = data["observations"]
    df = pd.DataFrame(obs)
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value']).set_index('date')[['value']]

    # Cache schreiben
    if use_cache:
        df.to_csv(cache_file)

    return df['value']


def get_series_meta(series_id):
    """Hole Metadaten (Name, Frequency, Units)."""
    api_key = load_key()
    url = "https://api.stlouisfed.org/fred/series"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()["seriess"][0]


def list_popular_series():
    """Auflistung wichtiger Series für Macro-Trading."""
    return {
        # Zinsen / Yields
        "DGS2": "2-Year Treasury Yield",
        "DGS10": "10-Year Treasury Yield",
        "DGS30": "30-Year Treasury Yield",
        "T10Y2Y": "10Y-2Y Yield Spread (Recession-Indicator)",
        "T10Y3M": "10Y-3M Yield Spread",
        "DFF": "Effective Fed Funds Rate",

        # Credit-Spreads
        "BAMLH0A0HYM2": "High-Yield OAS Spread",
        "BAMLC0A0CM": "Corporate-Investment-Grade Spread",
        "TEDRATE": "TED Spread (deprecated)",
        "BAMLH0A0HYM2EY": "HY Effective Yield",

        # Konjunktur
        "GDP": "Gross Domestic Product",
        "GDPC1": "Real GDP",
        "INDPRO": "Industrial Production Index",
        "UNRATE": "Unemployment Rate",
        "PAYEMS": "Non-Farm Payrolls",
        "ICSA": "Initial Jobless Claims",
        "CCSA": "Continuing Jobless Claims",
        "UMCSENT": "Michigan Consumer Sentiment",
        "DEXUSEU": "USD/EUR Exchange Rate",

        # Inflation
        "CPIAUCSL": "Consumer Price Index (CPI)",
        "CPILFESL": "Core CPI",
        "PCE": "Personal Consumption Expenditures",
        "PCEPILFE": "Core PCE Price Index",

        # Markt-Stress
        "NFCI": "Chicago Fed National Financial Conditions Index",
        "STLFSI4": "St. Louis Fed Financial Stress Index",
        "VIXCLS": "VIX Closing",

        # Geldmenge
        "M2SL": "M2 Money Supply",
        "WALCL": "Fed Balance Sheet",
        "RRPONTSYD": "Fed Reverse Repo (Liquidity)",

        # Wohnungsmarkt
        "HOUST": "Housing Starts",
        "MORTGAGE30US": "30Y Mortgage Rate",
        "CSUSHPINSA": "Case-Shiller Home Price Index",
    }


if __name__ == "__main__":
    # Test
    print("FRED Helper - Test")
    print("="*60)

    try:
        key = load_key()
        print(f"[OK] API-Key gefunden ({len(key)} Zeichen)")
    except RuntimeError as e:
        print(f"[FAIL] {e}")
        exit(1)

    # Test-Series laden
    print("\nLade T10Y2Y (Yield-Spread) als Test...")
    try:
        data = get_series("T10Y2Y", start="2000-01-01")
        print(f"[OK] {len(data)} Werte von {data.index.min().date()} bis {data.index.max().date()}")
        print(f"     Min: {data.min():.2f}, Max: {data.max():.2f}, Aktuell: {data.iloc[-1]:.2f}")
    except Exception as e:
        print(f"[FAIL] {e}")

    print("\nVerfuegbare Series ({} insgesamt):".format(len(list_popular_series())))
    for sid, name in list(list_popular_series().items())[:10]:
        print(f"  {sid:20s} - {name}")
    print("  ...")
