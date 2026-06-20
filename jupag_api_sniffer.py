#!/usr/bin/env python3
"""
Jupiter API Sniffer — schneidet die Netzwerk-Requests von jup.ag/prediction mit,
um die interne API (Preise/Orderbook) zu finden.

Nutzt Chrome Performance-Log (CDP Network events).
"""

import json
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Konkreter Markt aus Screenshot 1 (germany - ivory); fällt auf Übersicht zurück.
TARGET_URL = sys.argv[1] if len(sys.argv) > 1 else "https://jup.ag/prediction/POLY-351748"

# Begriffe, die auf eine Daten-API hindeuten (nicht Fonts/Bilder/Tracking).
INTERESTING = ["api", "prediction", "market", "order", "book", "price",
               "quote", "poly", "jup.ag/", "datapi", "gamma", "clob"]
IGNORE = ["font", ".woff", ".css", ".png", ".jpg", ".svg", ".webp", ".ico",
          "google", "sentry", "analytics", "bing", "facebook", "adform",
          "sportradar", "growthbuddy", "cloudflare", "reown", "tiplink",
          "_next/static", ".js", "turnstile"]


def main():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0 Safari/537.36")
    opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    print(f"Lade {TARGET_URL} ...")
    driver = webdriver.Chrome(options=opts)
    try:
        driver.get(TARGET_URL)
        time.sleep(18)  # JS-Requests abwarten

        logs = driver.get_log("performance")
        seen = {}  # url -> requestId

        for entry in logs:
            try:
                msg = json.loads(entry["message"])["message"]
            except Exception:
                continue
            if msg.get("method") != "Network.responseReceived":
                continue
            resp = msg["params"]["response"]
            url = resp.get("url", "")
            req_id = msg["params"].get("requestId")
            low = url.lower()

            if any(ig in low for ig in IGNORE):
                continue
            if not any(k in low for k in INTERESTING):
                continue
            if url not in seen:
                seen[url] = req_id

        print(f"\n=== {len(seen)} interessante API-URLs ===\n")
        for url in sorted(seen):
            print(url)

        # Versuche Response-Bodies der vielversprechendsten URLs zu holen.
        print("\n=== Response-Vorschau (erste 600 Zeichen) ===\n")
        for url, req_id in seen.items():
            if not any(k in url.lower() for k in ["api", "market", "price", "order", "quote", "poly", "clob", "datapi"]):
                continue
            try:
                body = driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": req_id})
                content = body.get("body", "")
                print(f"--- {url}")
                print(content[:600])
                print()
            except Exception as e:
                print(f"--- {url}  (Body nicht abrufbar: {e})")

        # Alles in Datei dumpen
        out = Path("jupag_api_urls.txt")
        out.write_text("\n".join(sorted(seen)), encoding="utf-8")
        print(f"\nURLs gespeichert in {out}")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
