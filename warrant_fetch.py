"""
Holt einen Optionsschein von comdirect.de und parsed die Kennzahlen.
Cached die Antworten auf Platte (TTL 15 min).
"""
import json
import re
import time
from datetime import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright

CACHE_DIR = Path(__file__).parent / "data" / "warrants"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SEC = 15 * 60


def _to_isin(wkn_or_isin: str) -> str:
    s = wkn_or_isin.strip().upper()
    if len(s) == 12 and s.startswith("DE"):
        return s
    if len(s) == 6:
        return f"DE000{s}0"
    return s


def _to_num(s):
    if s is None:
        return None
    m = re.search(r"-?\d{1,3}(?:\.\d{3})*(?:,\d+)?", s)
    if not m:
        return None
    return float(m.group(0).replace(".", "").replace(",", "."))


def _parse_ratio(s: str):
    if not s:
        return None
    # "1 : 1"  oder "0,01 : 1"  oder "10 : 1"
    m = re.match(r"(-?[\d.,]+)\s*:\s*(-?[\d.,]+)", s)
    if not m:
        return None
    num = _to_num(m.group(1))
    den = _to_num(m.group(2))
    if num is None or den in (None, 0):
        return None
    return num / den


def _parse_expiry(s: str):
    # "19.06.26"  -> 2026-06-19
    m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})", s.strip())
    if not m:
        return None
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 100:
        y += 2000
    try:
        return datetime(y, mo, d).date().isoformat()
    except ValueError:
        return None


def parse_dump(text: str) -> dict:
    lines = [l.rstrip() for l in text.splitlines()]
    out = {}

    NUM_LABELS = {
        "Delta": "delta",
        "Gamma": "gamma",
        "Hebel": "hebel",
        "Omega": "omega",
        "Implizite Volatilität": "implied_vol_pct",
        "Theta (EUR/Tag)": "theta_per_day",
        "Aufgeld": "aufgeld_pct",
        "Aufgeld p. a.": "aufgeld_pa_pct",
        "Zeitwert": "time_value",
        "Innerer Wert": "intrinsic_value",
        "Break Even": "break_even",
        "Moneyness": "moneyness",
        "Theoretischer Wert": "theoretical_value",
        "Basispreis": "strike",
        "Kurs Basiswert": "underlying_price",
    }
    STR_LABELS = {
        "Fälligkeit": "expiry_str",
        "letzter Handelstag": "last_trade_str",
        "Typ": "type_str",
        "Basiswert": "underlying_name",
        "Emittent": "issuer",
        "Bezugsverhältnis": "ratio_str",
        "WKN": "wkn",
        "ISIN": "isin",
        "Währung": "currency",
    }

    for line in lines:
        if "\t" in line:
            label, _, value = line.partition("\t")
            label = label.strip()
            value = value.strip()
            if label in NUM_LABELS:
                num = _to_num(value)
                if num is not None:
                    out[NUM_LABELS[label]] = num
            if label in STR_LABELS:
                out[STR_LABELS[label]] = value

    # Vega: Label auf eigener Zeile, dann "(EUR/Vol.-Pkt.)\t0,01"
    for i, line in enumerate(lines):
        if line.strip() == "Vega" and i + 1 < len(lines):
            num = _to_num(lines[i + 1])
            if num is not None:
                out["vega_per_volpkt"] = num
                break

    # Geld/Brief: Label\t (leer), Wert auf naechster Zeile
    for i, line in enumerate(lines):
        s = line.strip()
        if s in ("Geld", "Brief"):
            key = "bid" if s == "Geld" else "ask"
            for j in range(i + 1, min(i + 4, len(lines))):
                v = lines[j].strip()
                if not v:
                    continue
                num = _to_num(v)
                if num is not None and 0 < num < 100000:
                    out[key] = num
                    break

    # Type Call/Put
    if "type_str" in out:
        t = out["type_str"].lower()
        out["is_call"] = "call" in t
        out["is_american"] = "amer" in t

    # Bezugsverhaeltnis numerisch
    if "ratio_str" in out:
        r = _parse_ratio(out["ratio_str"])
        if r is not None:
            out["ratio"] = r

    # Faelligkeit als ISO + Tage bis Faelligkeit
    if "expiry_str" in out:
        iso = _parse_expiry(out["expiry_str"])
        if iso:
            out["expiry"] = iso
            try:
                days = (datetime.fromisoformat(iso).date() - datetime.utcnow().date()).days
                out["days_to_expiry"] = days
            except Exception:
                pass

    return out


def fetch_warrant(wkn_or_isin: str, force_refresh: bool = False) -> dict:
    isin = _to_isin(wkn_or_isin)
    cache_file = CACHE_DIR / f"{isin}.json"
    if not force_refresh and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < CACHE_TTL_SEC:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            data["_cache_age_sec"] = int(age)
            return data

    url = f"https://www.comdirect.de/inf/optionsscheine/{isin}"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1400, "height": 2200},
            locale="de-DE",
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"),
        )
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        for sel in [
            'button:has-text("Alle akzeptieren")',
            'button:has-text("Akzeptieren")',
            '#privacy-init-wall-accept-all-button',
            '[data-testid="uc-accept-all-button"]',
        ]:
            try:
                page.locator(sel).first.click(timeout=2500)
                break
            except Exception:
                continue
        page.wait_for_timeout(3000)
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass
        body_text = page.locator("body").inner_text(timeout=5000)
        browser.close()

    if "wurde nicht gefunden" in body_text.lower() or "seite nicht gefunden" in body_text.lower():
        raise ValueError(f"WKN/ISIN {isin} bei comdirect nicht gefunden (404).")

    data = parse_dump(body_text)
    if data.get("strike") is None and data.get("delta") is None:
        raise ValueError(f"Keine Optionsschein-Kennzahlen extrahiert fuer {isin} – evtl. KO-Zertifikat oder anderes Produkt.")
    data["_fetched_at"] = datetime.utcnow().isoformat() + "Z"
    data["_source_url"] = url
    cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    data["_cache_age_sec"] = 0
    return data


if __name__ == "__main__":
    import sys
    wkn = sys.argv[1] if len(sys.argv) > 1 else "MG721A"
    d = fetch_warrant(wkn, force_refresh="--refresh" in sys.argv)
    print(json.dumps(d, ensure_ascii=False, indent=2))
