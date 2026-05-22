"""
comdirect Optionsschein-Finder als browserlose requests-Pipeline.

1) resolve_underlying(name) -> [(Name, ID_NOTATION)] via underlyings.ajax (XML/CDATA)
2) search_warrants(...)     -> Trefferliste (trefferliste.html) parsen
   Filter serverseitig: Basiswert (ID_NOTATION), Typ (CALL/PUT), Strike-Range, Restlaufzeit.
   Filter clientseitig: EUR-Marktpreis (Brief) max/min.

Kein Browser, kein Cookie-Consent-Layer nötig — direkter GET liefert die Ergebnis-Tabelle.
"""
import re
import time
import requests
from html import unescape

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

BASE = "https://www.comdirect.de"
RESOLVE_URL = BASE + "/inf/snippet$lsg.warrant.selector.underlyings.ajax"
TREFFER_URL = BASE + "/inf/optionsscheine/selector/trefferliste.html"

_COMPARATORS = ["IMPLIED_VOLATILITY", "DELTA", "LEVERAGE", "PREMIUM_PER_ANNUM",
                "GEARING", "PRESENT_VALUE", "SPREAD_ASK_PCT", "THETA_DAY",
                "THEORETICAL_VALUE", "INTRINSIC_VALUE", "BREAK_EVEN",
                "MONEYNESS", "VEGA", "GAMMA"]

_session = None
_session_ts = 0.0
_SESSION_TTL = 600  # s


def _get_session() -> requests.Session:
    """Session mit qSession-Cookie (einmal finder.html laden), gecacht."""
    global _session, _session_ts
    if _session is None or (time.time() - _session_ts) > _SESSION_TTL:
        s = requests.Session()
        s.headers.update({"User-Agent": UA, "Accept-Language": "de-DE,de;q=0.9"})
        try:
            s.get(BASE + "/inf/optionsscheine/finder.html",
                  params={"CIF_Check": "true"}, timeout=15)
        except requests.RequestException:
            pass
        _session = s
        _session_ts = time.time()
    return _session


def _to_num(s):
    if s is None:
        return None
    m = re.search(r"-?\d{1,3}(?:\.\d{3})*(?:,\d+)?", s)
    if not m:
        return None
    return float(m.group(0).replace(".", "").replace(",", "."))


def resolve_underlying(name: str, max_results: int = 8):
    """Basiswert-Name -> Liste von {name, id_notation}. Auch ISIN/WKN/Index erlaubt."""
    name = (name or "").strip()
    if len(name) < 3:
        return []
    s = _get_session()
    r = s.get(RESOLVE_URL, params={"q": name},
              headers={"X-Requested-With": "XMLHttpRequest"}, timeout=15)
    m = re.search(r"<!\[CDATA\[(.*?)\]\]>", r.text, re.S)
    if not m:
        return []
    out = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        nm, _, idn = line.rpartition("|")
        nm, idn = nm.strip(), idn.strip()
        if nm and idn.isdigit():
            out.append({"name": nm, "id_notation": idn})
        if len(out) >= max_results:
            break
    return out


def _quote_to_float(s):
    """Kurs-String robust zu float — erkennt US- (1,234.56) UND deutsche (1.234,56) Notation."""
    s = (s or "").strip()
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):           # deutsch: 1.234,56
            s = s.replace(".", "").replace(",", ".")
        else:                                      # US: 1,234.56
            s = s.replace(",", "")
    elif "," in s:                                 # nur Komma
        s = s.replace(",", ".") if re.match(r"^\d+,\d{1,2}$", s) else s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_underlying_quote(html: str):
    """Underlying-Kurs aus dem Trefferlisten-Kopf: 'Aktuell: 215.45 USD | 22.05.26 21:15'."""
    i = html.find("Aktuell:")
    if i < 0:
        return None
    window = unescape(re.sub(r"<[^>]+>", " ", html[i:i + 140]))
    m = re.search(
        r"Aktuell:\s*([\d.,]+)\s*([A-Za-z€$]{1,4})(?:\s*\|\s*([\d.]{6,12}\s+[\d:]+))?",
        window)
    if not m:
        return None
    return {"price_str": m.group(1).strip(),
            "price": _quote_to_float(m.group(1)),
            "currency": m.group(2).strip(),
            "as_of": (m.group(3) or "").strip()}


# ---- Trefferlisten-Parser -------------------------------------------------

def _cell(row_html: str, label: str):
    """Erste nicht-leere data-label-Zelle eines Labels (Text, ohne Tags)."""
    for m in re.finditer(
            r'data-label="' + re.escape(label) + r'"[^>]*>(.*?)</td>',
            row_html, re.S):
        txt = re.sub(r"<[^>]+>", " ", m.group(1))
        txt = unescape(txt).replace("\xa0", " ")
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt:
            return txt
    return None


def _parse_trefferliste(html: str):
    rows = []
    # Datenzeilen tragen eine SELECTED_VALUE-Checkbox mit value="notUL|notWarrant"
    for rm in re.finditer(
            r'name="SELECTED_VALUE"\s+value="([^"]*)"(.*?)(?=name="SELECTED_VALUE"|</table>)',
            html, re.S):
        sel_value = rm.group(1)
        block = rm.group(2)
        isin_m = re.search(r'/inf/optionsscheine/([A-Z0-9]{12})', block)
        wkn = _cell(block, "WKN")
        if not isin_m and not wkn:
            continue
        bid = _to_num(_cell(block, "Geld"))
        ask = _to_num(_cell(block, "Brief"))
        rows.append({
            "isin": isin_m.group(1) if isin_m else None,
            "wkn": wkn,
            "strike": _to_num(_cell(block, "Basispreis")),
            "ratio_str": _cell(block, "Bez.-Verh.") or _cell(block, "Bez.Verh."),
            "expiry_str": _cell(block, "Fälligkeit"),
            "last_trade_str": _cell(block, "letzter H.-Tag") or _cell(block, "letzter H.Tag"),
            "bid": bid,
            "ask": ask,
            "issuer": _cell(block, "Emittent"),
            "implied_vol_str": _cell(block, "Impl. Vola"),
            "omega": _to_num(_cell(block, "Omega")),
            "spread_str": _cell(block, "Spread"),
            "exchange": _cell(block, "Börsenplatz"),
            "_sel": sel_value,
        })
    return rows


def search_warrants(underlying, opt_type="CALL", strike_from=None, strike_to=None,
                    maturity_from="Range_NOW", maturity_to="Range_ENDLESS",
                    price_max=None, price_min=None, limit=50, max_pages=8):
    """
    underlying: dict {name,id_notation} ODER Name-String (wird aufgelöst, 1. Treffer).
    opt_type:  'CALL' | 'PUT' | '' (alle)
    strike_from/to: absolute Basispreise (Zahl)
    price_max/min:  EUR-Marktpreis (Brief) Filter, clientseitig
    Rückgabe: {"underlying": {...}, "count": n, "warrants": [...]}
    """
    if isinstance(underlying, str):
        cands = resolve_underlying(underlying)
        if not cands:
            return {"underlying": None, "count": 0, "warrants": [],
                    "error": f"Basiswert '{underlying}' nicht gefunden."}
        underlying = cands[0]

    params = {
        "PRG_T": "/inf/optionsscheine/selector/trefferliste.html",
        "FORM_NAME": "DerivativesSelectorOptionsscheineForm",
        "PRESELECTION": opt_type or "",
        "UNDERLYING_TYPE": "FREI",
        "UNDERLYING_NAME_SEARCH": underlying["name"],
        "ID_NOTATION_UNDERLYING": underlying["id_notation"],
        "DATE_TIME_MATURITY_FROM": maturity_from,
        "DATE_TIME_MATURITY_TO": maturity_to,
    }
    for c in _COMPARATORS:
        params[c + "_COMPARATOR"] = "gt"
    if strike_from is not None:
        params["STRIKE_ABS_FROM"] = str(strike_from).replace(".", ",")
    if strike_to is not None:
        params["STRIKE_ABS_TO"] = str(strike_to).replace(".", ",")

    def _passes(w):
        if price_max is not None and (w["ask"] is None or w["ask"] > price_max):
            return False
        if price_min is not None and (w["ask"] is None or w["ask"] < price_min):
            return False
        return True

    # Über mehrere Trefferlisten-Seiten paginieren. comdirect liefert ~23 Zeilen/Seite;
    # Folgeseiten via &OFFSET=<seite-1> (OFFSET=1 -> Seite 2). Sammeln bis genug Zeilen
    # NACH Preisfilter zusammen sind oder die letzte Seite erreicht ist.
    s = _get_session()
    collected, seen = [], set()
    underlying_quote = None
    for page in range(max(1, max_pages)):
        page_params = dict(params)
        if page > 0:
            page_params["OFFSET"] = str(page)
        try:
            r = s.get(TREFFER_URL, params=page_params, timeout=25)
        except requests.RequestException:
            break
        if page == 0:
            underlying_quote = _parse_underlying_quote(r.text)
        page_rows = _parse_trefferliste(r.text)
        if not page_rows:
            break
        new = 0
        for w in page_rows:
            key = w["isin"] or w["wkn"]
            if not key or key in seen:
                continue
            seen.add(key)
            new += 1
            if _passes(w):
                collected.append(w)
        if new == 0 or len(collected) >= limit or len(page_rows) < 20:
            break
        time.sleep(0.3)  # comdirect nicht hämmern

    collected = collected[:limit]
    return {"underlying": underlying, "underlying_quote": underlying_quote,
            "count": len(collected), "warrants": collected}


if __name__ == "__main__":
    import json, sys
    name = sys.argv[1] if len(sys.argv) > 1 else "Nvidia"
    print("Auflösung:", json.dumps(resolve_underlying(name), ensure_ascii=False))
    res = search_warrants(name, opt_type="CALL", limit=8)
    print("Basiswert:", res["underlying"], "| Treffer:", res["count"])
    for w in res["warrants"]:
        print(f"  {w['wkn']:>8} {w['isin']}  Strike={w['strike']}  "
              f"Brief={w['ask']}  Verh={w['ratio_str']}  Fäll={w['expiry_str']}  {w['issuer']}")
