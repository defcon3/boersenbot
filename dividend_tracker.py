"""
DIVIDEND TRACKER — Live-Beobachtung von DE-Aktien rund um Ex-Dividenden-Tage.

Zweck: empirische Datenbasis fuer die auf veitluther.de/done falsifizierten
Dividend-Capture-Hypothesen (#11 / #12 / #13) durch *Live-Tracking* von
Aktien- UND Optionsschein-Preisen ueber Ex-Tag-Fenster. Historisch sind
Optionsschein-Daten in DE nicht frei verfuegbar — also sammeln wir sie ab jetzt.

KEIN Trading-System. KEINE Empfehlung. Nur Beobachtung.

Modi:
  --scan      Taeglich morgens: Universum nach Ex-Tagen in den naechsten
              3 Trading-Tagen scannen, Treffer in DB + Mail mit Put-Auswahl
  --snapshot  Taeglich nach Boersenschluss: fuer aktive Events (T-5 bis T+25)
              Stock-Close + Warrant-Quotes in DB schreiben
"""
import argparse
import datetime as dt
import logging
import os
import smtplib
import sqlite3
import sys
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import yfinance as yf

import warrant_search as ws

# ---------- KONFIGURATION ----------
DB_PATH = Path(__file__).parent / "dividend_tracker.db"
LOG_PATH = Path(__file__).parent / "dividend_tracker.log"

MAIL_HOST, MAIL_PORT = "mail.gmx.net", 587
MAIL_USER = os.environ.get("DT_MAIL_USER", "veit.luther@gmx.de")
MAIL_PASS = os.environ.get("DT_MAIL_PASS", "Extaler00!")
MAIL_TO = os.environ.get("DT_MAIL_TO", "veit.luther@gmx.de")

# Universum: DAX-40 + MDAX-50 + SDAX-70 (insgesamt ~160 Werte, dedupliziert)
DAX40 = [
    'ADS.DE', 'AIR.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BEI.DE', 'BMW.DE',
    'BNR.DE', 'CBK.DE', 'CON.DE', '1COV.DE', 'DBK.DE', 'DB1.DE', 'DHL.DE',
    'DTE.DE', 'DTG.DE', 'ENR.DE', 'EOAN.DE', 'FRE.DE', 'HEI.DE', 'HEN3.DE',
    'HNR1.DE', 'IFX.DE', 'MBG.DE', 'MRK.DE', 'MTX.DE', 'MUV2.DE', 'P911.DE',
    'PAH3.DE', 'QIA.DE', 'RHM.DE', 'RWE.DE', 'SAP.DE', 'SHL.DE', 'SIE.DE',
    'SRT3.DE', 'SY1.DE', 'VOW3.DE', 'VNA.DE', 'ZAL.DE',
]
MDAX50 = [
    'AIXA.DE', 'BC8.DE', 'BOSS.DE', 'COK.DE', 'DEZ.DE', 'DUE.DE', 'EVD.DE',
    'EVK.DE', 'EVT.DE', 'FIE.DE', 'FNTN.DE', 'FPE3.DE', 'FRA.DE', 'G1A.DE',
    'GBF.DE', 'GFT.DE', 'GLJ.DE', 'GXI.DE', 'HAB.DE', 'HEN.DE', 'HFG.DE',
    'HLE.DE', 'HOT.DE', 'JUN3.DE', 'KGX.DE', 'KRN.DE', 'LEG.DE', 'LHA.DE',
    'LXS.DE', 'NDA.DE', 'NEM.DE', 'PNE3.DE', 'PSM.DE', 'RAA.DE', 'RHK.DE',
    'SAX.DE', 'SDF.DE', 'SOW.DE', 'STO3.DE', 'SZG.DE', 'SZU.DE', 'TKA.DE',
    'TLX.DE', 'TUI1.DE', 'WAF.DE', 'WCH.DE',
]
# SDAX-70 (Stand 2026-05-26): Tickers via yfinance verifiziert, Quelle de.wikipedia.org/wiki/SDAX
SDAX70 = [
    'UN0.DE', 'ADN1.DE', 'ADV.DE', 'ACT.DE', 'AOF.DE', 'BFSA.DE', 'BVB.DE',
    'AFX.DE', 'CWC.DE', 'DMP.DE', 'DBAN.DE', 'DEQ.DE', 'PBB.DE', 'DOU.DE',
    'DRW3.DE', 'EUZ.DE', 'EIN.DE', 'ELG.DE', 'EKT.DE', 'VH2.DE', 'GYC.DE',
    'HABA.DE', 'HDD.DE', 'HBH.DE', 'HYQ.DE', 'INH.DE', 'IXX.DE', 'JST.DE',
    'KCO.DE', 'KTN.DE', 'KSB3.DE', 'KWS.DE', 'MBB.DE', 'ILM1.DE', 'MLP.DE',
    'MUX.DE', 'NA9.DE', 'NOEJ.DE', 'OBK.F', 'PAT.DE', 'TPE.DE', 'SFQ.DE',
    '1SXP.DE', 'YSN.DE', 'F3C.DE', 'SLYG.DE', 'SIX2.DE', 'S92.DE', 'SPG.DE',
    'STM.DE', 'SBS.DE', 'SMHN.DE', 'TMV.DE', 'TNIE.DE', 'VBK.DE', 'VRV.DE',
    'VOS.DE', 'WAC.DE',
]
# Dedup: SDAX enthaelt einige Werte, die auch im MDAX-Code-Stand stehen
UNIVERSE = list(dict.fromkeys(DAX40 + MDAX50 + SDAX70))

SCAN_LOOKAHEAD_TRADING_DAYS = 3
SNAPSHOT_WINDOW = (-5, 25)  # T-5 bis T+25 Trading-Tage
PREFERRED_ISSUERS = {
    'Soci', 'Societe', 'Société',  # Société Générale
    'Vontobel',
    'UBS',
    'Commerzbank',
    'HSBC',
    'Goldman',
    'Morgan Stanley',
    'BNP',
    'Citi',
    'Deutsche Bank',
    'DZ',
}

# DE-Feiertage (vereinfacht — ueberregional anerkannt)
DE_HOLIDAYS_FIXED = {
    (1, 1),     # Neujahr
    (5, 1),     # Tag der Arbeit
    (10, 3),    # Tag der Deutschen Einheit
    (12, 25),   # 1. Weihnachten
    (12, 26),   # 2. Weihnachten
}
# Beweglich (Ostern-basiert): vereinfachen, nicht reinrechnen.
# Karfreitag/Ostermontag/Christi Himmelfahrt/Pfingstmontag werden ggf. uebersehen,
# tolerieren wir — Risiko: 1-2 Tage Off-by-One pro Jahr.


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("dividend_tracker")


# ---------- DB ----------
def db_connect():
    con = sqlite3.connect(str(DB_PATH))
    con.execute("PRAGMA foreign_keys = ON")
    return con


def db_init():
    con = db_connect()
    con.executescript("""
    CREATE TABLE IF NOT EXISTS events (
        event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker        TEXT NOT NULL,
        name          TEXT,
        isin          TEXT,
        ex_date       DATE NOT NULL,
        gross_div     REAL,
        currency      TEXT,
        detected_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, ex_date)
    );

    CREATE TABLE IF NOT EXISTS stock_snapshots (
        snapshot_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id       INTEGER NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
        snapshot_date  DATE NOT NULL,
        days_to_ex     INTEGER,
        stock_close    REAL,
        UNIQUE(event_id, snapshot_date)
    );

    CREATE TABLE IF NOT EXISTS warrants (
        warrant_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id       INTEGER NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
        wkn            TEXT,
        isin           TEXT,
        issuer         TEXT,
        opt_type       TEXT,
        strike         REAL,
        expiry         DATE,
        first_found_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(event_id, wkn)
    );

    CREATE TABLE IF NOT EXISTS warrant_snapshots (
        snapshot_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        warrant_id     INTEGER NOT NULL REFERENCES warrants(warrant_id) ON DELETE CASCADE,
        snapshot_date  DATE NOT NULL,
        bid            REAL,
        ask            REAL,
        mid            REAL,
        implied_vol    REAL,
        hebel          REAL,
        spread_pct     REAL,
        UNIQUE(warrant_id, snapshot_date)
    );

    CREATE INDEX IF NOT EXISTS idx_events_ex_date ON events(ex_date);
    CREATE INDEX IF NOT EXISTS idx_stock_snap_event ON stock_snapshots(event_id);
    CREATE INDEX IF NOT EXISTS idx_warrant_snap_event ON warrant_snapshots(warrant_id);
    """)
    con.commit()
    con.close()


# ---------- TRADING-DAYS ----------
def is_trading_day(d: dt.date) -> bool:
    if d.weekday() >= 5:
        return False
    if (d.month, d.day) in DE_HOLIDAYS_FIXED:
        return False
    return True


def trading_days_between(start: dt.date, end: dt.date) -> int:
    """Anzahl Trading-Tage zwischen start (exklusiv) und end (inklusiv). Negativ wenn end vor start."""
    if end == start:
        return 0
    step = 1 if end > start else -1
    d = start
    count = 0
    while d != end:
        d += dt.timedelta(days=step)
        if is_trading_day(d):
            count += step
    return count


# ---------- HELPERS ----------
def _parse_iv(s):
    """'42,49 %' -> 42.49"""
    if not s:
        return None
    try:
        return float(s.replace('%', '').replace(',', '.').strip())
    except (ValueError, AttributeError):
        return None


def _parse_spread(s):
    """'1,92 %' -> 1.92"""
    return _parse_iv(s)


def _parse_expiry(s):
    """'19.06.26' -> date(2026, 6, 19); 'Open End' -> None"""
    if not s or 'open' in s.lower() or 'endlos' in s.lower():
        return None
    try:
        parts = s.split('.')
        if len(parts) == 3:
            day = int(parts[0])
            month = int(parts[1])
            year_short = int(parts[2])
            year = 2000 + year_short if year_short < 70 else 1900 + year_short
            return dt.date(year, month, day)
    except (ValueError, IndexError):
        pass
    return None


def _company_short_name(yticker, fallback_ticker):
    """Versuche aus yfinance einen sauberen Comdirect-Such-Namen abzuleiten."""
    try:
        info = yticker.info
        name = info.get('longName') or info.get('shortName') or fallback_ticker
        return name
    except Exception:
        return fallback_ticker


# Manuelle Mappings fuer Ticker, wo Auto-Resolve schiefgeht
# (Auto-Resolve probiert sonst diverse Varianten — Fallback hier)
_MANUAL_COMDIRECT_NAME = {
    'LEG.DE': 'LEG Immobilien',
    'RHK.DE': 'RHOEN',
    'HEN3.DE': 'Henkel',
    'HEN.DE': 'Henkel',
    'P911.DE': 'Porsche',
    'PAH3.DE': 'Porsche',
    'HNR1.DE': 'Hannover Rueck',
    'MUV2.DE': 'Muenchener Rueck',
    '1COV.DE': 'Covestro',
    'SRT3.DE': 'Sartorius',
    'VOW3.DE': 'Volkswagen',
    'DB1.DE': 'Deutsche Boerse',
    'DTE.DE': 'Deutsche Telekom',
    'BNR.DE': 'Brenntag',
    'CON.DE': 'Continental',
    'BAYN.DE': 'Bayer',
    'ALV.DE': 'Allianz',
    'TUI1.DE': 'TUI',
}

_SUFFIX_STRIP = ['Aktiengesellschaft', 'AG & Co. KGaA', 'SE & Co. KGaA',
                 '& Co. KGaA', 'Co. KGaA', 'KGaA', 'N.V.', 'plc', 'S.A.']
import re as _re
_TRAILING_COMDIRECT_FLAG = _re.compile(r'\s+[INO]\s*$')


def _search_variants_for(ticker, short, long_):
    """Liefert eine Liste von Suchnamen, geordnet nach erwarteter Trefferquote."""
    out = []
    if ticker in _MANUAL_COMDIRECT_NAME:
        out.append(_MANUAL_COMDIRECT_NAME[ticker])
    for src in (short, long_):
        if not src:
            continue
        src = _TRAILING_COMDIRECT_FLAG.sub('', src).strip()
        out.append(src)
        s = src
        for suf in _SUFFIX_STRIP:
            if s.endswith(suf):
                s = s[:-len(suf)].strip(' ,.')
        if s and s != src:
            out.append(s)
    # Dedup case-insensitive
    seen, dedup = set(), []
    for v in out:
        k = v.lower().strip()
        if k and k not in seen:
            seen.add(k); dedup.append(v)
    return dedup


def _is_relevant_match(comdirect_name, ticker, short, long_):
    """Prueft ob der Comdirect-Trefferkandidat plausibel zur Ziel-Aktie passt.
    Heuristik: mindestens ein 4+-Buchstaben-Token muss in beiden Namen vorkommen."""
    if not comdirect_name:
        return False
    cd_tokens = set(t.lower() for t in _re.findall(r'[A-Za-zÄÖÜäöüß]+', comdirect_name) if len(t) >= 4)
    src_text = ' '.join(filter(None, [short, long_]))
    src_tokens = set(t.lower() for t in _re.findall(r'[A-Za-zÄÖÜäöüß]+', src_text) if len(t) >= 4)
    # Manuell-Mapping: Token-Match gegen den Mapping-Namen
    if ticker in _MANUAL_COMDIRECT_NAME:
        m = _MANUAL_COMDIRECT_NAME[ticker]
        src_tokens |= set(t.lower() for t in _re.findall(r'[A-Za-zÄÖÜäöüß]+', m) if len(t) >= 4)
    return bool(cd_tokens & src_tokens)


def resolve_underlying_for_event(ticker, name, log):
    """Gibt {name, id_notation} oder None. name = longName aus yfinance."""
    # Auch shortName probieren — kommt aus build-time
    try:
        yt = yf.Ticker(ticker)
        short = (yt.info or {}).get('shortName')
    except Exception:
        short = None
    variants = _search_variants_for(ticker, short, name)
    for v in variants:
        try:
            cands = ws.resolve_underlying(v)
        except Exception as e:
            log.warning(f"  resolve '{v}' err: {e}")
            continue
        for c in cands:
            if _is_relevant_match(c.get('name'), ticker, short, name):
                log.info(f"  resolve {ticker} via '{v}' -> {c['name']}")
                return c
    log.warning(f"  resolve {ticker}: keine plausible Variante gefunden (probiert: {variants})")
    return None


# ---------- SCAN ----------
def scan_universe(log):
    today = dt.date.today()
    cutoff = today
    days_added = 0
    while days_added < SCAN_LOOKAHEAD_TRADING_DAYS:
        cutoff += dt.timedelta(days=1)
        if is_trading_day(cutoff):
            days_added += 1
    log.info(f"Scan: today={today}, cutoff={cutoff} ({SCAN_LOOKAHEAD_TRADING_DAYS} Trading-Tage)")

    hits = []
    for ticker in UNIVERSE:
        try:
            tk = yf.Ticker(ticker)
            cal = tk.calendar or {}
            ex_date = cal.get('Ex-Dividend Date')
            if not ex_date:
                continue
            if isinstance(ex_date, list):
                ex_date = ex_date[0]
            if isinstance(ex_date, dt.datetime):
                ex_date = ex_date.date()
            if today <= ex_date <= cutoff:
                # Aktueller Kurs + Dividende
                hist = tk.history(period='5d')
                spot = float(hist['Close'].iloc[-1]) if len(hist) else None
                # Letzte bekannte Dividende als Schaetzer fuer kommenden Betrag
                divs = tk.dividends
                gross_div = float(divs.iloc[-1]) if len(divs) else None
                name = _company_short_name(tk, ticker)
                isin = (tk.info or {}).get('isin')
                hits.append({
                    'ticker': ticker, 'name': name, 'isin': isin,
                    'ex_date': ex_date, 'gross_div': gross_div, 'spot': spot,
                })
                log.info(f"  HIT {ticker} ({name}): Ex={ex_date}, div~{gross_div}, spot={spot}")
        except Exception as e:
            log.warning(f"  {ticker}: scan error {e}")
        time.sleep(0.05)
    return hits


def find_puts_for_event(event, log, top_n=3):
    """Sucht passende Put-Optionsscheine ueber comdirect."""
    spot = event['spot']
    if not spot:
        return []
    underlying = resolve_underlying_for_event(event['ticker'], event['name'], log)
    if not underlying:
        return []
    strike_low = round(spot * 0.90, 2)
    strike_high = round(spot * 1.10, 2)
    try:
        result = ws.search_warrants(
            underlying,
            opt_type='PUT',
            strike_from=strike_low,
            strike_to=strike_high,
            maturity_from='Range_NOW',
            maturity_to='Range_ENDLESS',
            limit=30,
        )
    except Exception as e:
        log.warning(f"  put-search err {event['ticker']}: {e}")
        return []
    warrants = result.get('warrants', [])
    if not warrants:
        # Fallback ohne Strike-Range
        try:
            result = ws.search_warrants(underlying, opt_type='PUT', limit=30)
            warrants = result.get('warrants', [])
        except Exception as e:
            log.warning(f"  put-search fallback err {event['ticker']}: {e}")
            return []

    if not warrants:
        return []

    def _score(w):
        issuer = (w.get('issuer') or '')
        pref = any(p.lower() in issuer.lower() for p in PREFERRED_ISSUERS)
        omega = w.get('omega') or 0
        return (-int(pref), -omega)
    warrants.sort(key=_score)
    log.info(f"  {event['ticker']}: {len(warrants)} Puts gefunden, top {min(top_n, len(warrants))}")
    return warrants[:top_n]


def insert_event_and_warrants(con, event, warrants, log):
    """Speichert Event + Warrants in DB (idempotent via UNIQUE-Constraints)."""
    cur = con.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO events (ticker, name, isin, ex_date, gross_div, currency)
        VALUES (?, ?, ?, ?, ?, 'EUR')
    """, (event['ticker'], event['name'], event['isin'],
          event['ex_date'].isoformat(), event['gross_div']))
    cur.execute("SELECT event_id FROM events WHERE ticker=? AND ex_date=?",
                (event['ticker'], event['ex_date'].isoformat()))
    event_id = cur.fetchone()[0]

    # Initial-Stock-Snapshot vom Scan-Tag
    today_iso = dt.date.today().isoformat()
    days_to_ex = trading_days_between(dt.date.today(), event['ex_date'])
    cur.execute("""
        INSERT OR IGNORE INTO stock_snapshots
            (event_id, snapshot_date, days_to_ex, stock_close)
        VALUES (?, ?, ?, ?)
    """, (event_id, today_iso, days_to_ex, event['spot']))

    # Warrants speichern
    for w in warrants:
        cur.execute("""
            INSERT OR IGNORE INTO warrants
                (event_id, wkn, isin, issuer, opt_type, strike, expiry)
            VALUES (?, ?, ?, ?, 'PUT', ?, ?)
        """, (event_id, w.get('wkn'), w.get('isin'), w.get('issuer'),
              w.get('strike'), _parse_expiry(w.get('expiry_str')).isoformat()
              if _parse_expiry(w.get('expiry_str')) else None))
        cur.execute("SELECT warrant_id FROM warrants WHERE event_id=? AND wkn=?",
                    (event_id, w.get('wkn')))
        wrow = cur.fetchone()
        if wrow:
            warrant_id = wrow[0]
            mid = ((w.get('bid') or 0) + (w.get('ask') or 0)) / 2 \
                  if w.get('bid') and w.get('ask') else None
            cur.execute("""
                INSERT OR IGNORE INTO warrant_snapshots
                    (warrant_id, snapshot_date, bid, ask, mid, implied_vol, hebel, spread_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (warrant_id, today_iso, w.get('bid'), w.get('ask'), mid,
                  _parse_iv(w.get('implied_vol_str')), w.get('omega'),
                  _parse_spread(w.get('spread_str'))))
    con.commit()
    return event_id


# ---------- MAIL ----------
def build_html_mail(events_with_warrants):
    today = dt.date.today().strftime('%d.%m.%Y')
    n_events = len(events_with_warrants)
    rows = []
    for ev, warrants in events_with_warrants:
        ex_str = ev['ex_date'].strftime('%d.%m.%Y')
        days_to_ex = trading_days_between(dt.date.today(), ev['ex_date'])
        gross = f"{ev['gross_div']:.2f} EUR" if ev['gross_div'] else "?"
        spot = f"{ev['spot']:.2f} EUR" if ev['spot'] else "?"
        yield_pct = (ev['gross_div'] / ev['spot'] * 100) if ev['gross_div'] and ev['spot'] else 0
        # Warrant-Mini-Tabelle
        if warrants:
            w_rows = "".join([
                f"""<tr>
                    <td style="font-family:monospace;font-size:0.85em;">{w.get('wkn','')}</td>
                    <td>{w.get('issuer','')[:14]}</td>
                    <td style="text-align:right;">{w.get('strike','')}</td>
                    <td style="font-size:0.85em;">{w.get('expiry_str','')}</td>
                    <td style="text-align:right;">{(w.get('ask') or 0):.2f}</td>
                    <td style="text-align:right;">{(w.get('omega') or 0):.1f}x</td>
                    <td style="font-size:0.85em;">{w.get('implied_vol_str','')}</td>
                    </tr>"""
                for w in warrants
            ])
            w_table = f"""<table style="width:100%;border-collapse:collapse;font-size:0.88em;margin-top:6px;">
                <thead><tr style="background:#34495e;color:white;">
                    <th style="padding:4px;text-align:left;">WKN</th>
                    <th style="padding:4px;text-align:left;">Emittent</th>
                    <th style="padding:4px;text-align:right;">Strike</th>
                    <th style="padding:4px;text-align:left;">Fälligkeit</th>
                    <th style="padding:4px;text-align:right;">Brief</th>
                    <th style="padding:4px;text-align:right;">Hebel</th>
                    <th style="padding:4px;text-align:left;">IV</th>
                </tr></thead><tbody>{w_rows}</tbody></table>"""
        else:
            w_table = "<em>keine passenden Puts gefunden</em>"
        rows.append(f"""
        <div style="background:#fff;border-left:5px solid #34495e;padding:14px 18px;margin:14px 0;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
            <h3 style="margin:0 0 8px 0;color:#2c3e50;">{ev['name']} <span style="font-family:monospace;color:#7f8c8d;font-size:0.85em;">{ev['ticker']}</span></h3>
            <div style="font-size:0.92em;color:#34495e;">
                <strong>Ex-Tag:</strong> {ex_str} (in {days_to_ex} Trading-Tagen) ·
                <strong>Brutto-Div:</strong> {gross} ({yield_pct:.2f} %) ·
                <strong>Spot:</strong> {spot}
            </div>
            {w_table}
        </div>
        """)

    body = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,Segoe UI,sans-serif;background:#f4f4f8;padding:20px;color:#2c3e50;">
<div style="max-width:780px;margin:0 auto;">

    <div style="background:linear-gradient(135deg,#34495e,#2c3e50);color:#fff;padding:24px;border-radius:12px;margin-bottom:18px;">
        <h2 style="margin:0;">📊 Dividend Tracker</h2>
        <p style="margin:6px 0 0 0;opacity:0.85;font-size:0.95em;">{today} · {n_events} Event{'s' if n_events != 1 else ''} in den nächsten {SCAN_LOOKAHEAD_TRADING_DAYS} Trading-Tagen</p>
    </div>

    <div style="background:#fdecea;border-left:5px solid #e74c3c;padding:14px 18px;margin-bottom:18px;border-radius:8px;">
        <strong style="color:#922b21;">⚠️ BEOBACHTUNG — KEINE EMPFEHLUNG.</strong><br>
        Die Strategie „Put-Optionsschein vor Ex-Tag" ist auf
        <a href="https://veitluther.de/done#h12" style="color:#922b21;">veitluther.de/done#h12</a>
        als Hypothese #12 falsifiziert (Put-Call-Parity preist die Dividende ein).
        Erwarteter P&amp;L: <strong>≈ −2 %/Trade nach Spread + Theta + Vol-Crush.</strong>
        Dieser Tracker dient ausschließlich der Live-Daten-Sammlung für ein
        späteres Re-Validierung der Theorie mit echten Optionsschein-Bewegungen.
    </div>

    {''.join(rows)}

    <div style="margin-top:24px;padding:12px;background:#ecf0f1;border-radius:8px;font-size:0.82em;color:#7f8c8d;">
        Universum: DAX-40 + MDAX-50.<br>
        Daten: yfinance .calendar (Ex-Dates), comdirect Optionsschein-Finder (Warrants).<br>
        Tracking-Fenster: T−5 bis T+25 Trading-Tage je Event (DB: dividend_tracker.db).
    </div>
</div>
</body>
</html>
"""
    return body


def send_mail(subject, html_body, log):
    msg = MIMEMultipart('alternative')
    msg['From'] = MAIL_USER
    msg['To'] = MAIL_TO
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html', 'utf-8'))
    with smtplib.SMTP(MAIL_HOST, MAIL_PORT, timeout=30) as s:
        s.starttls()
        s.login(MAIL_USER, MAIL_PASS)
        s.sendmail(MAIL_USER, [MAIL_TO], msg.as_string())
    log.info(f"Mail gesendet an {MAIL_TO}: {subject}")


# ---------- SNAPSHOT ----------
def snapshot_active_events(log):
    con = db_connect()
    cur = con.cursor()
    today = dt.date.today()
    # Aktive Events: Ex-Tag innerhalb +/- Snapshot-Fenster
    win_start = today - dt.timedelta(days=abs(SNAPSHOT_WINDOW[1]) * 2)
    win_end = today + dt.timedelta(days=abs(SNAPSHOT_WINDOW[0]) * 2)
    cur.execute("""
        SELECT event_id, ticker, name, ex_date, gross_div FROM events
        WHERE ex_date >= ? AND ex_date <= ?
    """, (win_start.isoformat(), win_end.isoformat()))
    events = cur.fetchall()
    log.info(f"Snapshot: {len(events)} Events im Fenster {win_start}..{win_end}")

    for event_id, ticker, name, ex_date_str, gross_div in events:
        ex_date = dt.date.fromisoformat(ex_date_str)
        days_to_ex = trading_days_between(today, ex_date)
        if not (SNAPSHOT_WINDOW[0] <= days_to_ex <= SNAPSHOT_WINDOW[1]):
            continue
        # Stock-Close
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period='5d')
            if len(hist):
                close = float(hist['Close'].iloc[-1])
                cur.execute("""
                    INSERT OR IGNORE INTO stock_snapshots
                        (event_id, snapshot_date, days_to_ex, stock_close)
                    VALUES (?, ?, ?, ?)
                """, (event_id, today.isoformat(), days_to_ex, close))
                log.info(f"  stock {ticker} ({name}): close={close} dte={days_to_ex}")
        except Exception as e:
            log.warning(f"  stock {ticker}: err {e}")
        # Warrants des Events: aktuelle Quotes
        cur.execute("SELECT warrant_id, wkn FROM warrants WHERE event_id=?", (event_id,))
        warrants_in_db = cur.fetchall()
        if warrants_in_db:
            try:
                underlying = resolve_underlying_for_event(ticker, name, log)
                if not underlying:
                    log.warning(f"  {ticker}: konnte underlying nicht resolven, skip warrants")
                    continue
                result = ws.search_warrants(
                    underlying, opt_type='PUT',
                    strike_from=None, strike_to=None,
                    maturity_from='Range_NOW', maturity_to='Range_ENDLESS',
                    limit=80,
                )
                # Map WKN -> warrant-dict
                wkn_map = {w.get('wkn'): w for w in result.get('warrants', [])}
                for warrant_id, wkn in warrants_in_db:
                    if wkn in wkn_map:
                        w = wkn_map[wkn]
                        mid = ((w.get('bid') or 0) + (w.get('ask') or 0)) / 2 \
                              if w.get('bid') and w.get('ask') else None
                        cur.execute("""
                            INSERT OR IGNORE INTO warrant_snapshots
                                (warrant_id, snapshot_date, bid, ask, mid, implied_vol, hebel, spread_pct)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (warrant_id, today.isoformat(),
                              w.get('bid'), w.get('ask'), mid,
                              _parse_iv(w.get('implied_vol_str')),
                              w.get('omega'),
                              _parse_spread(w.get('spread_str'))))
                        log.info(f"  warrant {wkn}: bid={w.get('bid')} ask={w.get('ask')} hebel={w.get('omega')}")
                    else:
                        log.info(f"  warrant {wkn}: NICHT mehr in Comdirect-Trefferliste")
            except Exception as e:
                log.warning(f"  warrants {ticker}: err {e}")
        con.commit()
        time.sleep(0.4)
    con.close()


# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan', action='store_true', help='Universum scannen + Mail')
    parser.add_argument('--snapshot', action='store_true', help='Tagliche Quote-Snapshots')
    parser.add_argument('--no-mail', action='store_true', help='Kein Mailversand (Dry-Run)')
    parser.add_argument('--init', action='store_true', help='DB initialisieren')
    args = parser.parse_args()

    log = setup_logging()
    db_init()

    if args.init:
        log.info(f"DB initialisiert: {DB_PATH}")
        return

    if args.scan:
        log.info("=" * 60)
        log.info("SCAN START")
        hits = scan_universe(log)
        if not hits:
            log.info("Keine Ex-Tage im Lookahead-Fenster.")
            return
        con = db_connect()
        events_with_warrants = []
        for ev in hits:
            warrants = find_puts_for_event(ev, log)
            insert_event_and_warrants(con, ev, warrants, log)
            events_with_warrants.append((ev, warrants))
            time.sleep(0.4)
        con.close()

        if not args.no_mail:
            html = build_html_mail(events_with_warrants)
            subject_names = ", ".join(ev['name'][:18] for ev, _ in events_with_warrants[:3])
            if len(events_with_warrants) > 3:
                subject_names += f", +{len(events_with_warrants)-3} weitere"
            subject = f"Dividend Tracker: {len(events_with_warrants)} Ex-Tag(e) bevorstehend — {subject_names}"
            send_mail(subject, html, log)
        log.info("SCAN DONE")
        return

    if args.snapshot:
        log.info("=" * 60)
        log.info("SNAPSHOT START")
        snapshot_active_events(log)
        log.info("SNAPSHOT DONE")
        return

    parser.print_help()


if __name__ == '__main__':
    main()
