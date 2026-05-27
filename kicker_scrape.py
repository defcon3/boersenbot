"""
KICKER.DE-SCRAPER fuer Spieltag-Daten der englischen Ligen + NL.

Liefert das, was football-data.co.uk nicht hat: die offizielle Spieltag-
Nummer pro Spiel. Wird gebraucht, weil bei League One/Two/Conference die
fixturedownload.com-Quelle (siehe FixtureDownloadClient.vb) keine Daten hat
und die Gap-Fill-Heuristik bei verlegten Spielen falsch zaehlt.

Bot-Protection: Datadome blockt curl/headless. Loesung: Playwright
persistent context mit headed Browser (Profile-Dir wird wiederverwendet,
sodass Datadome nach dem ersten Lauf den Cookie kennt).
"""
import argparse
import asyncio
import csv
import os
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

PROFILE_DIR = Path(os.path.expandvars(r"%LOCALAPPDATA%\PlaywrightProfiles\kicker"))
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

LEAGUE_PATH = {
    "epl":   "premier-league",
    "ech":   "championship",
    "el1":   "league-one",
    "el2":   "league-two",
    "econf": "national-league",
    "nl1":   "eredivisie",
}

# League One / Two / Conference: 24 Teams -> 46 Spieltage
# EPL: 20 Teams -> 38; Championship: 24 -> 46; Eredivisie: 18 -> 34
MATCHDAYS = {
    "epl": 38, "ech": 46, "el1": 46, "el2": 46, "econf": 46, "nl1": 34,
}


def parse_matchday_page(html: str) -> list[dict]:
    """Liefert eine Liste {date, home, away, fth, fta, hth, hta, match_id, away_url}."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    rows = soup.find_all("div", class_="kick__v100-gameList__gameRow")
    for row in rows:
        # Datums-Header: vorheriger gameList__header
        date_el = row.find_previous("div", class_="kick__v100-gameList__header")
        date_txt = date_el.get_text(" ", strip=True) if date_el else ""
        # "Samstag, 06.08.2016" -> "06.08.2016"
        m = re.search(r"(\d{2}\.\d{2}\.\d{4})", date_txt)
        date_iso = ""
        if m:
            dd, mm, yyyy = m.group(1).split(".")
            date_iso = f"{yyyy}-{mm}-{dd}"

        teams = row.find_all("div", class_="kick__v100-gameCell__team__name")
        if len(teams) < 2:
            continue
        home = teams[0].get_text(" ", strip=True).split("\n")[0].strip()
        away = teams[1].get_text(" ", strip=True).split("\n")[0].strip()

        sb = row.find("a", class_="kick__v100-scoreBoard")
        match_id = ""
        if sb and sb.get("href"):
            # /bolton-gegen-sheffield-u-2016-league-one-3315222/schema
            m_id = re.search(r"-(\d{5,})/", sb["href"])
            if m_id:
                match_id = m_id.group(1)

        # ScoreHolder: 1. = FT, 2. mit --subscore = HT
        holders = sb.find_all("div", class_="kick__v100-scoreBoard__scoreHolder") if sb else []
        fth = fta = hth = hta = ""
        if len(holders) >= 1:
            sc = holders[0].find_all("div", class_="kick__v100-scoreBoard__scoreHolder__score")
            if len(sc) == 2:
                fth, fta = sc[0].get_text(strip=True), sc[1].get_text(strip=True)
        if len(holders) >= 2:
            sc = holders[1].find_all("div", class_="kick__v100-scoreBoard__scoreHolder__score")
            if len(sc) == 2:
                hth, hta = sc[0].get_text(strip=True), sc[1].get_text(strip=True)

        out.append({
            "date":     date_iso,
            "home":     home,
            "away":     away,
            "fth":      fth,
            "fta":      fta,
            "hth":      hth,
            "hta":      hta,
            "match_id": match_id,
        })
    return out


async def scrape_season(league: str, season_start: int,
                        max_matchdays: int | None = None,
                        matchday_from: int = 1) -> list[dict]:
    if league not in LEAGUE_PATH:
        raise SystemExit(f"unbekannte Liga: {league}; erlaubt: {list(LEAGUE_PATH)}")
    n_matchdays = max_matchdays or MATCHDAYS[league]
    path = LEAGUE_PATH[league]
    season_url = f"{season_start}-{str(season_start + 1)[-2:]}"  # 2016 -> "2016-17"

    rows_total = []
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            str(PROFILE_DIR),
            headless=False,
            viewport={"width": 1280, "height": 900},
            locale="de-DE",
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()

        # Warmup: Liga-Hauptseite, damit Datadome-Cookie aktiv ist
        warmup = f"https://www.kicker.de/{path}/spieltag/{season_url}"
        print(f"[warmup] {warmup}")
        resp = await page.goto(warmup, wait_until="domcontentloaded", timeout=45000)
        if resp.status == 403:
            print("[warmup] 403 - bitte loese die Captcha im Fenster und druecke ENTER",
                  file=sys.stderr)
            input()
        await page.wait_for_timeout(1500)

        for spt in range(matchday_from, n_matchdays + 1):
            url = f"https://www.kicker.de/{path}/spieltag/{season_url}/{spt}"
            # Goto mit eigener Timeout-Behandlung (Network-Lag soll Skript nicht killen)
            try:
                r = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception as ex:
                print(f"[spt {spt:>2}] Goto-Fehler ({type(ex).__name__}): {ex} - retry nach 5s",
                      file=sys.stderr)
                await page.wait_for_timeout(5000)
                try:
                    r = await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                except Exception as ex2:
                    print(f"[spt {spt:>2}] erneut fehlgeschlagen - SKIP",
                          file=sys.stderr)
                    continue
            # Retry-Schleife: bei 403 erst Sleep+Reload, dann Captcha-Pause
            if r.status == 403:
                print(f"[spt {spt:>2}] 403 - retry nach 6s", file=sys.stderr)
                await page.wait_for_timeout(6000)
                r = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            if r.status == 403:
                print(f"[spt {spt:>2}] erneut 403 - Captcha im Fenster loesen, "
                      "dann ENTER druecken", file=sys.stderr)
                input()
                r = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            if r.status != 200:
                print(f"[spt {spt:>2}] STATUS={r.status} - skip")
                await page.wait_for_timeout(2000)
                continue
            await page.wait_for_timeout(900)
            html = await page.content()
            matches = parse_matchday_page(html)
            for m in matches:
                m["matchday"] = spt
                m["season"]   = season_url
                m["league"]   = league
            print(f"[spt {spt:>2}] {len(matches)} Spiele")
            rows_total.extend(matches)

        await ctx.close()
    return rows_total


def write_csv(rows: list[dict], path: Path):
    if not rows:
        print("Keine Zeilen - nichts geschrieben.")
        return
    cols = ["league", "season", "matchday", "date", "home", "away",
            "fth", "fta", "hth", "hta", "match_id"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"OK -> {path} ({len(rows)} Zeilen)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True, choices=list(LEAGUE_PATH))
    ap.add_argument("--season", required=True, type=int,
                    help="Saison-Startjahr, z.B. 2016 fuer 2016-17")
    ap.add_argument("--out",    default=None,
                    help="Output-CSV (default: kicker_<league>_<season>.csv)")
    ap.add_argument("--max-matchdays", type=int, default=None,
                    help="Override Spieltag-Anzahl (debug)")
    ap.add_argument("--from", dest="matchday_from", type=int, default=1,
                    help="Erster Spieltag (default 1)")
    args = ap.parse_args()

    rows = asyncio.run(scrape_season(args.league, args.season,
                                     args.max_matchdays, args.matchday_from))
    out = Path(args.out or f"kicker_{args.league}_{args.season}.csv")
    write_csv(rows, out)


if __name__ == "__main__":
    main()
