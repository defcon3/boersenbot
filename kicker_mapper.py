"""
Mappt Kicker-Teamnamen auf football-data.co.uk-Teamnamen.

Strategie: Heuristik (FC/AFC-Prefix entfernen) + explizite Overrides fuer
inkonsistente football-data-Abkuerzungen ("Peterboro", "Bristol Rvs").

Validiert wird gegen die fd-Namen, die schon in der DB-Tabelle
TeamNameMappings (Source='football-data.co.uk') stehen.
"""
import csv
import sys
from pathlib import Path

# Teams, die football-data exakt so abkuerzt, wie Kicker den Klubnamen
# liefert (Schreibweise identisch). Werden NICHT angefasst.
_KEEP_AS_IS = {
    "Sheffield United",       # nicht zu "Sheffield" (es gibt Sh. Wednesday)
    "Fleetwood Town",         # nicht zu "Fleetwood"
    "Milton Keynes Dons",     # ganzer Markenname
    "AFC Wimbledon",          # Praefix gehoert zum eingebuergerten Namen
    # League Two 2016/17 Edge-Cases: fd LAESST "Town"/"Country" hier drin
    "Crawley Town",
    "Leyton Orient",
    "Newport County",
    "Notts County",
    # National League 2016/17 Edge-Cases: fd LAESST den Suffix drin
    "Boreham Wood",
    "Braintree Town",
    "Dover Athletic",
    # National League 2023/24
    "Oxford City",            # nicht zu "Oxford" (es gibt Oxford United in el1)
    # National League 2015/16
    "Welling United",         # DB hat "Welling United" (nicht zu "Welling")
    # League One 2014/15
    "Bristol City",           # DB hat "Bristol City" (nicht zu "Bristol")
    # National League 2014/15
    "Alfreton Town",          # DB hat "Alfreton Town"
    "Nuneaton Town",          # DB hat "Nuneaton Town"
    "Telford United",         # DB hat "Telford United"
    # National League 2011/12
    "Bath City",              # DB hat "Bath City"
    "Kettering Town",         # DB hat "Kettering Town"
    # Eredivisie (nl1): fd behaelt hier das "FC" (nicht zu "Emmen" strippen)
    "FC Emmen",
    "Go Ahead Eagles",        # DB hat "Go Ahead Eagles"
}

# Sonderschreibweisen, die fd kuerzer / anders abkuerzt als kicker.
# (kicker-Name -> fd-Name). Nach Bedarf je Saison erweitern.
_OVERRIDES = {
    # League One 2016/17 (verifiziert)
    "Peterborough United": "Peterboro",
    "Bristol Rovers":      "Bristol Rvs",
    # League Two 2016/17 (verifiziert)
    "Accrington Stanley":  "Accrington",
    "Plymouth Argyle":     "Plymouth",
    # National League 2016/17 (verifiziert)
    "Dagenham & Redbridge":     "Dag and Red",
    "North Ferriby United AFC": "North Ferriby",
    "Solihull Moors":           "Solihull",
    # League One 2018/19 (verifiziert)
    "AFC Sunderland":           "Sunderland",
    "Burton Albion":            "Burton",
    # League One 2021/22 (verifiziert)
    "Sheffield Wednesday":      "Sheffield Weds",
    # National League 2021/22 (verifiziert)
    "Stockport County":         "Stockport",
    "King's Lynn":              "King�s Lynn",
    # League One 2022/23 (verifiziert)
    "Derby County":             "Derby",
    # National League 2022/23 (verifiziert)
    "Dorking Wanderers FC":     "Dorking",
    # National League 2023/24 (verifiziert)
    "Kidderminster Harriers":   "Kidderminster",
    # National League 2024/25 (verifiziert)
    "Boston United":            "Boston Utd",
    # League One 2014/15
    "Preston North End":        "Preston",
    # National League 2013/14
    "FC Hyde":                  "Hyde United",
    # National League 2010/11
    "Rushden & Diamonds":       "Rushden & D",
    # ── Eredivisie (nl1) ── kicker -> football-data (DB-Schreibweise)
    # Verifiziert gegen DB 2017 (SeasonID 106). Zusaetze fuer 2010/2013/2020
    # (Cambuur/Sittard/Graafschap/Nijmegen/Waalwijk) je Saison per Self-Test
    # bestaetigen.
    "ADO Den Haag":         "Den Haag",
    "Ajax Amsterdam":       "Ajax",
    "Excelsior Rotterdam":  "Excelsior",
    "Feyenoord Rotterdam":  "Feyenoord",
    "Heracles Almelo":      "Heracles",
    "PEC Zwolle":           "Zwolle",
    "Roda Kerkrade":        "Roda",
    "SC Heerenveen":        "Heerenveen",
    "Twente Enschede":      "Twente",
    "VVV-Venlo":            "VVV Venlo",
    "Vitesse Arnheim":      "Vitesse",
    "Willem II Tilburg":    "Willem II",
    "SC Cambuur":               "Cambuur",
    "Cambuur Leeuwarden":       "Cambuur",      # Kicker-Variante mit Stadt (2013)
    "Fortuna Sittard":          "For Sittard",
    "De Graafschap":            "Graafschap",
    "De Graafschap Doetinchem": "Graafschap",   # Kicker-Variante mit Stadt (2010)
    "Go Ahead Eagles Deventer": "Go Ahead Eagles",  # Kicker-Variante mit Stadt (2013)
    "NEC Nijmegen":             "Nijmegen",
    "RKC Waalwijk":             "Waalwijk",
    # Verlaeufige Sammlung weiterer EN-Edge-Cases (vorsorglich -
    # werden bei Bedarf gegen DB-Daten verifiziert):
    "Wycombe Wanderers":   "Wycombe",
    "Crewe Alexandra":     "Crewe",
    "Crystal Palace":      "Crystal Palace",
    "Nottingham Forest":   "Nott'm Forest",
    "Brighton & Hove Albion": "Brighton",
    "Birmingham City":     "Birmingham",
    "Manchester United":   "Man United",
    "Manchester City":     "Man City",
    "Tottenham Hotspur":   "Tottenham",
    "Newcastle United":    "Newcastle",
    "West Ham United":     "West Ham",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
    "Leicester City":      "Leicester",
    "Leeds United":        "Leeds",
    "Hull City":           "Hull",
    "Cardiff City":        "Cardiff",
    "Stoke City":          "Stoke",
    "Norwich City":        "Norwich",
    "Aston Villa":         "Aston Villa",
    "Queens Park Rangers": "QPR",
    "Blackburn Rovers":    "Blackburn",
    "Sunderland AFC":      "Sunderland",
    "Swansea City":        "Swansea",
    "Burnley FC":          "Burnley",
    "Watford FC":          "Watford",
    "Liverpool FC":        "Liverpool",
    "Chelsea FC":          "Chelsea",
    "Everton FC":          "Everton",
    "Arsenal FC":          "Arsenal",
    "Reading FC":          "Reading",
}

# Komplexe Suffixe, die generell entfernt werden (nach FC/AFC-Strip)
# AUSSER der ganze Name ist in _KEEP_AS_IS.
_SUFFIX_STRIP = (" Wanderers", " Athletic", " United", " Town", " City", " Rovers")


def kicker_to_fd(name: str) -> str:
    """Liefert den football-data.co.uk-Schreibweise zum Kicker-Namen."""
    name = name.strip()
    if name in _KEEP_AS_IS:
        return name
    if name in _OVERRIDES:
        return _OVERRIDES[name]

    # Praefix-Strip: "FC X" -> "X", "AFC X" -> "X" (AUSSER AFC Wimbledon, s.o.)
    for prefix in ("FC ", "AFC "):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Falls jetzt _KEEP_AS_IS oder _OVERRIDES greift (z.B. "FC Burnley" -> "Burnley FC"-Override greift nicht mehr):
    if name in _KEEP_AS_IS:
        return name
    if name in _OVERRIDES:
        return _OVERRIDES[name]

    # Generic Suffix-Strip
    for suf in _SUFFIX_STRIP:
        if name.endswith(suf):
            stripped = name[: -len(suf)]
            # Nur strippen, wenn das Restwort nicht zu kurz wird
            if len(stripped) >= 4:
                return stripped

    return name


# --------------------------------------------------------
# Self-test: gegen DB-Mappings einer Saison validieren.
# --------------------------------------------------------
if __name__ == "__main__":
    import pyodbc

    if len(sys.argv) < 3:
        print("Usage: kicker_mapper.py <kicker-csv> <season-id>")
        print("  Liest die Kicker-CSV, holt die fd-Teamnamen fuer SeasonID")
        print("  aus DB und vergleicht.")
        sys.exit(2)

    csv_path = Path(sys.argv[1])
    season_id = int(sys.argv[2])

    CONN = ("DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=158.181.48.77;DATABASE=dbdata;UID=326773;"
            "PWD=Extaler11!;TrustServerCertificate=yes;Encrypt=yes;")

    # 1) Unique Kicker-Teamnamen aus CSV
    with open(csv_path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        names = set()
        for r in rd:
            names.add(r["home"])
            names.add(r["away"])

    # 2) fd-Mappings aus DB
    cn = pyodbc.connect(CONN)
    cur = cn.cursor()
    cur.execute("""
        SELECT DISTINCT tm.ExternalName
        FROM TeamNameMappings tm
        WHERE tm.Source = 'football-data.co.uk'
          AND tm.TeamID IN (
              SELECT Team1ID FROM Matches WHERE SeasonID = ?
              UNION
              SELECT Team2ID FROM Matches WHERE SeasonID = ?
          )
    """, season_id, season_id)
    fd_names = {r.ExternalName for r in cur.fetchall()}

    print(f"Kicker-Teams: {len(names)}; fd-Teams: {len(fd_names)}")
    print()
    print(f"{'Kicker':<28}  {'->':<3}  {'fd (gemappt)':<22}  {'in DB?'}")
    print("-" * 75)
    ok = miss = 0
    for n in sorted(names):
        mapped = kicker_to_fd(n)
        hit = mapped in fd_names
        flag = "OK" if hit else "FEHLT"
        if hit:
            ok += 1
        else:
            miss += 1
        print(f"{n:<28}  ->   {mapped:<22}  {flag}")

    print(f"\nSumme: {ok} OK, {miss} FEHLT")
    unmatched_fd = fd_names - {kicker_to_fd(n) for n in names}
    if unmatched_fd:
        print(f"\nfd-Teams, die KEIN Kicker-Team angelaufen hat:")
        for n in sorted(unmatched_fd):
            print(f"  {n}")
