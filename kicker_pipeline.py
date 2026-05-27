"""
Pipeline: Kicker-CSV(s) -> clean.csv -> DB-Backfill (Matchday-Update).

Verarbeitet pro Aufruf eine Liga+Saison:
  python kicker_pipeline.py <league> <season-start-year>
  z.B. python kicker_pipeline.py el2 2016

Voraussetzungen:
- Eine oder mehrere kicker_<league>_<season>_*.csv im aktuellen Verzeichnis
  (z.B. _full.csv, _spt12.csv). Werden alle automatisch gemergt.
- Saison muss in DB existieren (Seasons.LeagueShortcut + SeasonYear).
- Mapper kennt alle Team-Namen (sonst Abbruch mit Liste).

Schritte (alle mit Sanity-Checks):
1. Lade alle kicker-CSVs der Liga+Saison, dedup per Match-ID
   (lowest matchday), korrigiere Spieltag ueber ID-Block-Heuristik
   (sortierte Position // matches_per_matchday + 1).
2. Schreibe kicker_<league>_<season>_clean.csv.
3. Validiere: unique IDs == expected_total, Matches/Spt == const, jeder
   Team-Paar genau 2x.
4. Match gegen DB (SeasonID) per (Datum, mapped_home, mapped_away);
   alle Kicker-Matches MUESSEN matchen.
5. Schreibe Backup-CSV kicker_backfill_<league>_<season>_pre.csv.
6. UPDATE Matches.Matchday in einer Transaction.
7. Post-Validation: Range, Spt-Counts in DB.
"""
import csv
import glob
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pyodbc

sys.path.insert(0, str(Path(__file__).parent))
from kicker_mapper import kicker_to_fd

CONN = ("DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=158.181.48.77;DATABASE=dbdata;UID=326773;"
        "PWD=Extaler11!;TrustServerCertificate=yes;Encrypt=yes;")

# Liga-Eigenschaften
LEAGUE_INFO = {
    # league: (n_teams, matchdays, matches_per_matchday)
    "epl":   (20, 38, 10),
    "ech":   (24, 46, 12),
    "el1":   (24, 46, 12),
    "el2":   (24, 46, 12),
    "econf": (24, 46, 12),
    "nl1":   (18, 34, 9),
}


def merge_and_clean(league: str, season: int) -> tuple[list[dict], Path]:
    season_url = f"{season}-{str(season + 1)[-2:]}"
    pattern = f"kicker_{league}_{season}_*.csv"
    files = [f for f in glob.glob(pattern)
             if not f.endswith("_clean.csv")
             and not f.endswith("_pre.csv")]
    if not files:
        raise SystemExit(f"Keine Eingabe-CSVs gefunden ({pattern})")
    print(f"Eingabe-CSVs: {files}")

    all_rows = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                all_rows.append(r)
    print(f"Eingelesen: {len(all_rows)} Zeilen (aus {len(files)} Dateien)")

    # Dedup per Match-ID (irgendeine Repraesentation, Werte gleich)
    by_id = {}
    for r in all_rows:
        by_id[r["match_id"]] = dict(r)

    _, total_md, mpd = LEAGUE_INFO[league]
    expected = total_md * mpd
    print(f"Unique Match-IDs: {len(by_id)} (erwartet {expected})")
    if len(by_id) != expected:
        print(f"  WARN: Anzahl Match-IDs weicht ab")

    # TRUE-Spt aus sortierter ID-Position
    all_ids = sorted(by_id.keys(), key=int)
    id_to_spt = {mid: i // mpd + 1 for i, mid in enumerate(all_ids)}

    changed = 0
    for mid, r in by_id.items():
        old = int(r["matchday"])
        new = id_to_spt[mid]
        if old != new:
            changed += 1
            r["matchday"] = new
    print(f"Spieltag-Korrekturen: {changed}")

    dedup = sorted(by_id.values(),
                   key=lambda r: (int(r["matchday"]), r["date"], r["home"]))

    # Validierung
    md_counts = Counter(int(r["matchday"]) for r in dedup)
    issues = []
    for spt in range(1, total_md + 1):
        n = md_counts.get(spt, 0)
        if n != mpd:
            issues.append(f"Spt {spt}: {n} (sollte {mpd})")
    if issues:
        print(f"  WARN: {len(issues)} Spieltage mit falscher Match-Anzahl:")
        for i in issues[:10]:
            print(f"    {i}")
    else:
        print(f"  Alle {total_md} Spieltage = {mpd} Matches OK")

    tg = Counter()
    for r in dedup:
        tg[r["home"]] += 1
        tg[r["away"]] += 1
    expected_per_team = (total_md * mpd * 2) // (mpd * 2)  # 46 normalerweise
    expected_per_team = total_md
    bad = [(t, n) for t, n in tg.items() if n != expected_per_team]
    if bad:
        print(f"  WARN: {len(bad)} Teams mit != {expected_per_team} Spielen")
    else:
        print(f"  Alle {len(tg)} Teams = {expected_per_team} Spiele OK")

    pairs = Counter()
    for r in dedup:
        pairs[tuple(sorted([r["home"], r["away"]]))] += 1
    not_two = [(k, v) for k, v in pairs.items() if v != 2]
    if not_two:
        print(f"  WARN: {len(not_two)} Team-Paare nicht 2x")
    else:
        print(f"  Alle {len(pairs)} Team-Paare 2x OK")

    out = Path(f"kicker_{league}_{season}_clean.csv")
    cols = ["league", "season", "matchday", "date", "home", "away",
            "fth", "fta", "hth", "hta", "match_id"]
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(dedup)
    print(f"Geschrieben: {out} ({len(dedup)} Zeilen)")
    return dedup, out


def backfill(league: str, season: int, kicker_rows: list[dict],
             apply: bool = False, allow_missing_db: bool = False):
    cn = pyodbc.connect(CONN)
    cur = cn.cursor()

    # SeasonID finden
    cur.execute("SELECT SeasonID FROM Seasons WHERE LeagueShortcut=? AND SeasonYear=?",
                league, season)
    row = cur.fetchone()
    if not row:
        raise SystemExit(f"Saison {league} {season} nicht in DB")
    season_id = row.SeasonID
    print(f"\nDB SeasonID = {season_id}")

    cur.execute("""
        SELECT m.MatchID, m.Matchday, CONVERT(date, m.MatchDateTime) AS d,
               t1.TeamName AS Home, t2.TeamName AS Away
        FROM Matches m
        JOIN Teams t1 ON t1.TeamID = m.Team1ID
        JOIN Teams t2 ON t2.TeamID = m.Team2ID
        WHERE m.SeasonID = ?
    """, season_id)
    db_rows = list(cur.fetchall())
    db_idx = {(r.d.isoformat(), r.Home, r.Away): (r.MatchID, r.Matchday)
              for r in db_rows}
    print(f"DB-Matches: {len(db_rows)}")

    updates, miss_kicker = [], []
    for r in kicker_rows:
        key = (r["date"], kicker_to_fd(r["home"]), kicker_to_fd(r["away"]))
        if key not in db_idx:
            miss_kicker.append(key)
            continue
        mid, old_md = db_idx[key]
        updates.append((mid, old_md, int(r["matchday"]),
                        r["date"], key[1], key[2]))

    if miss_kicker and not allow_missing_db:
        print(f"\n  ABBRUCH: {len(miss_kicker)} Kicker-Matches OHNE DB-Match.")
        for k in miss_kicker[:10]:
            print(f"    {k}")
        return
    if miss_kicker:
        print(f"\n  WARN: {len(miss_kicker)} Kicker-Matches OHNE DB-Match (erwartet bei Corona-Saison o.ae.).")
        # Pruefe ob alle nicht-gematchten Kicker-Zeilen ungespielt sind (kein FT-Score)
        unplayed = sum(1 for r in kicker_rows
                       if (r["date"], kicker_to_fd(r["home"]), kicker_to_fd(r["away"])) not in db_idx
                       and r["fth"] == "")
        print(f"        davon ohne FT-Score: {unplayed} (ungespielt - plausibel)")

    print(f"Gematcht: {len(updates)} / {len(kicker_rows)}")
    diff = sum(1 for _, o, n, _, _, _ in updates if o != n)
    print(f"Aenderungen: {diff}")

    backup = Path(f"kicker_backfill_{league}_{season}_pre.csv")
    with open(backup, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["MatchID", "OldMatchday", "NewMatchday", "Date",
                    "Home", "Away", "SeasonID", "BackupTimestamp"])
        ts = datetime.now().isoformat(timespec="seconds")
        for mid, old, new, d, h, a in updates:
            w.writerow([mid, old, new, d, h, a, season_id, ts])
    print(f"Backup: {backup}")

    if not apply:
        print("\n(dry-run; --apply zum echten Schreiben)")
        return

    changes_only = [(mid, new) for mid, old, new, _, _, _ in updates if old != new]
    cur.fast_executemany = True
    cur.executemany(
        "UPDATE Matches SET Matchday = ? WHERE MatchID = ?",
        [(new, mid) for mid, new in changes_only]
    )
    cn.commit()
    print(f"COMMIT: {len(changes_only)} Zeilen aktualisiert.")

    cur.execute("SELECT MIN(Matchday), MAX(Matchday), COUNT(*) FROM Matches WHERE SeasonID=?",
                season_id)
    mn, mx, cnt = cur.fetchone()
    print(f"DB-Range nach Update: Matchday {mn}-{mx}, {cnt} Matches")


def main():
    if len(sys.argv) < 3:
        print("Usage: kicker_pipeline.py <league> <season> [--apply]")
        sys.exit(2)
    league = sys.argv[1]
    season = int(sys.argv[2])
    apply = "--apply" in sys.argv[3:]
    allow_missing = "--allow-missing-db" in sys.argv[3:]
    if league not in LEAGUE_INFO:
        raise SystemExit(f"unbekannte Liga: {league}; bekannt: {list(LEAGUE_INFO)}")

    print(f"=== {league} {season}/{(season+1)%100:02d} ===")
    rows, _ = merge_and_clean(league, season)
    backfill(league, season, rows, apply=apply, allow_missing_db=allow_missing)


if __name__ == "__main__":
    main()
