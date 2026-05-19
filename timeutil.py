"""Zentraler Zeit-Vertrag fuer das Boersenbot-Streaming.

Konvention (Single Source of Truth):
    Alle bb_StockPrices_1min_Streaming-Timestamps sind *naive UTC*
    (datetime ohne tzinfo).

    - Der Producer (alpaca_streaming_simple.py) erzeugt sie via
      parse_alpaca_ts().
    - Consumer (app.py: _freshness, get_streaming_status) messen Alter
      ausschliesslich via age_seconds() und zeigen Zeiten via
      to_local_str().

Die Umrechnung in Lokalzeit (Wanduhr) passiert NUR hier am Rand.
Damit ist der wiederkehrende "naive UTC vs. lokale now"-Versatz
strukturell ausgeschlossen: keine Call-Site rechnet selbst mit Zeit.
"""
from datetime import datetime, timezone


def now_utc_naive():
    """Jetzt als naive UTC -- direkt vergleichbar mit gespeicherten Werten."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def parse_alpaca_ts(ts_str):
    """Alpaca-Bar-Timestamp ('2026-05-19T16:49:00Z') -> naive UTC datetime.

    Bewusst nur die ersten 19 Zeichen (YYYY-MM-DDTHH:MM:SS), robust
    gegen 'Z' und optionale Sekundenbruchteile.
    """
    return datetime.strptime(ts_str[:19], '%Y-%m-%dT%H:%M:%S')


def age_seconds(ts):
    """Alter eines naive-UTC-Timestamps in Sekunden.

    Gibt None bei ts is None und klemmt gegen negative Werte
    (geringer Uhren-Versatz Server vs. Datenquelle).
    """
    if ts is None:
        return None
    return max(0.0, (now_utc_naive() - ts).total_seconds())


def to_local_str(ts, fmt='%Y-%m-%d %H:%M'):
    """Naive-UTC-Timestamp -> lokale Zeit (Server-Wanduhr) als String."""
    if ts is None:
        return 'nie'
    return ts.replace(tzinfo=timezone.utc).astimezone().strftime(fmt)
