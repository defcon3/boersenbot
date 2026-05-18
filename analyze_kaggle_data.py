#!/usr/bin/env python3
"""
Kaggle 1-min data analysis for feature engineering
Loads Kaggle data from SQL Server, computes technical indicators,
and analyzes signal strength for logistic regression
"""

import pymssql
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Database config
DB_CONFIG = {
    'server': '158.181.48.77',
    'database': 'dbdata',
    'user': '326773',
    'password': 'Extaler11!',
    'as_dict': False
}

def get_kaggle_data():
    """Load Kaggle 1-min data from database"""
    try:
        conn = pymssql.connect(**DB_CONFIG)
        query = """
            SELECT Timestamp, Symbol, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume
            FROM bb_StockPrices_1min_Kaggle
            ORDER BY Symbol, Timestamp
        """
        df = pd.read_sql(query, conn)
        conn.close()

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def compute_indicators(df):
    """Compute technical indicators"""
    # RSI (14-period)
    delta = df['ClosePrice'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    exp1 = df['ClosePrice'].ewm(span=12).mean()
    exp2 = df['ClosePrice'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands (20, 2)
    df['BB_Middle'] = df['ClosePrice'].rolling(20).mean()
    df['BB_Std'] = df['ClosePrice'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    df['BB_Position'] = (df['ClosePrice'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    # Momentum
    df['Returns'] = df['ClosePrice'].pct_change() * 100  # in %
    df['Returns_1min'] = df['ClosePrice'].diff()
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

    # High-Low range
    df['HL_Range'] = ((df['HighPrice'] - df['LowPrice']) / df['ClosePrice']) * 100

    # Label: next 1-min return direction (UP=1, DOWN=0)
    df['Next_Return'] = df['Returns_1min'].shift(-1)
    df['Label'] = (df['Next_Return'] > 0).astype(int)

    return df

def analyze_symbol(symbol_data, symbol):
    """Analyze single symbol"""
    symbol_data = symbol_data.dropna(subset=['RSI', 'MACD', 'Label'])

    print(f"\n{'='*60}")
    print(f"SYMBOL: {symbol}")
    print(f"{'='*60}")
    print(f"Records: {len(symbol_data):,}")
    print(f"Date range: {symbol_data['Timestamp'].min()} to {symbol_data['Timestamp'].max()}")

    # Label distribution
    up_pct = symbol_data['Label'].mean() * 100
    print(f"\nLabel distribution:")
    print(f"  UP (next min > 0): {up_pct:.2f}%")
    print(f"  DOWN (next min <= 0): {100-up_pct:.2f}%")

    # Return statistics
    print(f"\nReturn statistics (per minute):")
    print(f"  Mean: {symbol_data['Returns'].mean():.4f}%")
    print(f"  Std: {symbol_data['Returns'].std():.4f}%")
    print(f"  Max: {symbol_data['Returns'].max():.4f}%")
    print(f"  Min: {symbol_data['Returns'].min():.4f}%")

    # Indicator ranges
    print(f"\nIndicator ranges:")
    print(f"  RSI: {symbol_data['RSI'].min():.1f} - {symbol_data['RSI'].max():.1f}")
    print(f"  MACD: {symbol_data['MACD'].min():.6f} - {symbol_data['MACD'].max():.6f}")
    print(f"  BB_Position: {symbol_data['BB_Position'].min():.2f} - {symbol_data['BB_Position'].max():.2f}")

    # Correlation with next return
    print(f"\nCorrelation with next 1-min return:")
    features = ['RSI', 'MACD', 'MACD_Hist', 'BB_Position', 'Returns', 'Volume_Ratio', 'HL_Range']
    correlations = {}
    for feat in features:
        corr = symbol_data[feat].corr(symbol_data['Next_Return'])
        correlations[feat] = corr
        print(f"  {feat:15s}: {corr:7.4f}")

    # Signal strength: RSI oversold/overbought
    rsi_oversold = (symbol_data['RSI'] < 30).mean() * 100
    rsi_overbought = (symbol_data['RSI'] > 70).mean() * 100
    print(f"\nRSI extremes:")
    print(f"  Oversold (RSI < 30): {rsi_oversold:.2f}%")
    print(f"  Overbought (RSI > 70): {rsi_overbought:.2f}%")

    # MACD crossover frequency
    macd_positive = (symbol_data['MACD'] > 0).mean() * 100
    print(f"\nMACD signal:")
    print(f"  Positive (MACD > 0): {macd_positive:.2f}%")

    return correlations, symbol_data

def main():
    print("Loading Kaggle data...")
    df = get_kaggle_data()

    if df is None or df.empty:
        print("No data loaded")
        return

    print(f"Total records: {len(df):,}")
    print(f"Symbols: {df['Symbol'].unique()}")

    # Compute indicators
    print("\nComputing technical indicators...")
    df = compute_indicators(df)

    # Analyze each symbol
    all_correlations = {}
    for symbol in sorted(df['Symbol'].unique()):
        symbol_data = df[df['Symbol'] == symbol].copy()
        corr, _ = analyze_symbol(symbol_data, symbol)
        all_correlations[symbol] = corr

    # Overall correlation summary
    print(f"\n{'='*60}")
    print("OVERALL CORRELATION SUMMARY (across all symbols)")
    print(f"{'='*60}")

    features = list(all_correlations[list(all_correlations.keys())[0]].keys())
    for feat in features:
        avg_corr = np.mean([all_correlations[sym][feat] for sym in all_correlations.keys()])
        print(f"{feat:15s}: {avg_corr:7.4f} (avg)")

    # Visualization
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Correlation heatmap
    ax = axes[0, 0]
    first_symbol = list(all_correlations.keys())[0]
    symbol_data = df[df['Symbol'] == first_symbol].copy().dropna(subset=['RSI', 'MACD', 'Label'])
    features_to_plot = ['RSI', 'MACD', 'MACD_Hist', 'BB_Position', 'Returns', 'Volume_Ratio', 'Next_Return']
    corr_matrix = symbol_data[features_to_plot].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title(f'Feature Correlations ({first_symbol})')

    # 2. RSI distribution
    ax = axes[0, 1]
    for symbol in sorted(df['Symbol'].unique())[:3]:  # Top 3 symbols
        symbol_data = df[df['Symbol'] == symbol].dropna(subset=['RSI'])
        ax.hist(symbol_data['RSI'], bins=50, alpha=0.5, label=symbol)
    ax.axvline(30, color='red', linestyle='--', linewidth=1, label='Oversold (30)')
    ax.axvline(70, color='red', linestyle='--', linewidth=1, label='Overbought (70)')
    ax.set_xlabel('RSI')
    ax.set_ylabel('Frequency')
    ax.set_title('RSI Distribution (Sample Symbols)')
    ax.legend()

    # 3. Returns distribution
    ax = axes[1, 0]
    returns = df['Returns'].dropna()
    ax.hist(returns, bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.4f}%')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('1-Minute Return Distribution (All Data)')
    ax.legend()

    # 4. Label distribution by RSI
    ax = axes[1, 1]
    first_symbol = list(all_correlations.keys())[0]
    symbol_data = df[df['Symbol'] == first_symbol].copy().dropna(subset=['RSI', 'Label'])
    rsi_bins = pd.cut(symbol_data['RSI'], bins=[0, 30, 50, 70, 100])
    label_by_rsi = symbol_data.groupby(rsi_bins)['Label'].mean() * 100
    label_by_rsi.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_xlabel('RSI Range')
    ax.set_ylabel('% UP (next minute)')
    ax.set_title(f'Next 1-Min UP Probability by RSI ({first_symbol})')
    ax.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig('kaggle_analysis.png', dpi=100, bbox_inches='tight')
    print("✅ Saved: kaggle_analysis.png")

    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print("\nKey findings:")
    print("- Check which indicators have highest correlation with next return")
    print("- Look for RSI oversold/overbought patterns")
    print("- Check if mean reversion or momentum works better")
    print("- Consider symbol-specific vs universal features")

if __name__ == '__main__':
    main()
