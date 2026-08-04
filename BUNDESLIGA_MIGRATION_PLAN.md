# Migrationsplan: Centron ablösen — ASPX → Flask, MSSQL → PostgreSQL

**Angelegt:** 2026-08-04 · **Status:** Plan, nichts umgesetzt
**Ziel:** 24,50 €/Monat → 11,50 €/Monat, ein Host statt drei Verträge.

---

## Was heute wo läuft (gemessen 04.08.2026, nicht angenommen)

| Vertrag | Preis | Was wirklich drin ist |
|---|---|---|
| **Contabo** 144.91.98.234 | 5 € | `veitluther.de`, 9 Börsenbot-Dienste. Ubuntu 24.04, 4 vCPU, **8 GB RAM (6,6 frei)**, **126 GB frei**, Load 0,08. **Docker am 04.08. vollständig entfernt** (Hop + Superset verworfen, +2,4 GB RAM, +7 GB Platte) |
| **Centron / internet1** | 13 € | Windows-Hosting: IIS 10 + ASP.NET 4.0 unter `frau-von-allerliebst.de/bundesliga/` (158.181.48.160) **plus** MSSQL `dbcentral12.internet1.de` (158.181.48.77) |
| **Infomaniak** | 6,5 € | 6 TB Webspeicher, Auslagerungsziel der Backup-Kette. **Bleibt** — sichert gegen Contabo-Ausfall und ist der einzige Vertrag, den kein anderer ersetzen kann |

**Wichtig:** Die 13 € sind kein Datenbank-Abo. Die DB fährt im Windows-Paket mit.
Nur die DB umzuziehen spart deshalb **nichts** — Centron fällt erst weg, wenn auch
die ASPX-Seite weg ist.

### Wer die Datenbank benutzt

`dbdata`, **400 MB**, 31 Tabellen (16 `bb_` Börsenbot, 15 Fußball).

| Konsument | Ort | Dateien mit hartkodierter IP |
|---|---|---|
| Börsenbot-Stack | Contabo | 39 |
| Repo (Entwicklung) | dieser Rechner | 37 |
| `bundesliga`-Import | **NAS** (Cron 06:00/18:00) | 8 |
| ASPX-Seite | Centron | 1 (`web.config`) |
| Superset | Contabo-Container | 1 (Verbindung in der Superset-DB) |

Das NAS (`armv7l`, **502 MB RAM**) scheidet als DB-Host aus — SQL Server gibt es
für ARM nicht, und Postgres wäre auf dem DS215j die Bremse.

---

## Die zentrale Erkenntnis: die Seite ist rein lesend

```
18 .aspx        Language="C#", inline <script runat="server">
0 Code-Behind   keine .aspx.cs/.vb
0 eigene DLLs   bin/ leer
10.073 Zeilen   Markup + Code zusammen
17/18 Seiten    lesen per SqlConnection
0 Seiten        schreiben (kein INSERT/UPDATE/DELETE/ExecuteNonQuery)
0 Seiten        rufen externe APIs
```

Daraus folgt der ganze Zuschnitt: **die neue Seite kann vollständig gegen eine
Postgres-Kopie gebaut und getestet werden, während Centron unverändert Master
bleibt.** Kein Dual-Write, kein Konsistenzproblem, jederzeit Seite-an-Seite
vergleichbar. Erst ganz am Ende wechseln die Schreiber.

---

## Warum PostgreSQL und nicht MariaDB

Entscheidend ist, was die 18 Seiten an SQL benutzen — gemessen:

| Konstrukt | Vorkommen | Postgres | MariaDB |
|---|---|---|---|
| `ROW_NUMBER/RANK/OVER(...)` | **26** | 1:1 | ab 10.2 ok |
| CTEs (`WITH ranked AS …`) | durchgehend | 1:1 | ok |
| `OUTER APPLY` | 1 | `LEFT JOIN LATERAL` | **kein LATERAL** |
| `STRING_AGG` | 1 | gleicher Name | `GROUP_CONCAT` |
| `SELECT TOP n` | 3 | `LIMIT n` | `LIMIT n` |
| `GETDATE()` | 4 | `NOW()` | `NOW()` |

`OUTER APPLY` gibt MariaDB nicht her — das allein entscheidet. Dazu kommt, dass
die Abfragen stark auf Fensterfunktionen und CTEs bauen, wo Postgres der
robustere Partner ist. **Insgesamt sind es 9 dialektbehaftete Fundstellen in
10.000 Zeilen** — die SQL-Portierung ist der kleinste Teil der Arbeit.

Im Börsenbot-Repo dasselbe Bild: 57 Dateien nutzen `pymssql`, aber nur 12
enthalten T-SQL-Eigenheiten.

---

## Phasen

### P1 — Postgres auf Contabo, Erstladung, Abgleichschleife

**Nativ per `apt` (PGDG), kein Container.** Entscheidung vom 04.08.: Docker ist
von der Maschine herunter, damit wäre ein einzelner DB-Container der einzige
Fremdkörper in einem Stack, der sonst durchgehend systemd + venv ist.
Standardpfade, `psql`/`pg_dump` im PATH, Sicherheitsupdates über die
Paketverwaltung.

- Schema + Daten per `pgloader` aus MSSQL (400 MB, einmalig Minuten).
- Bis zum Umschalttag ein nächtlicher Wiederholabgleich.
- Port 5432 **nicht** öffentlich; das NAS bekommt WireGuard oder eine
  IP-Freigabe (s. P4).
- **Centron bleibt in dieser Phase unverändert Master.**

**Sicherung — der Punkt, an dem die Intuition trügt.** Ein laufendes
PGDATA-Verzeichnis darf man nicht wegkopieren: der NAS-Pull läuft **31,5 Minuten**,
in der Zeit schreibt die DB weiter, und man bekäme eine zerrissene Kopie, die im
Ernstfall nicht startet. (Das gilt für Container genauso — ein Image sichert die
Software, nicht den Zustand.) Richtig ist ein **logischer Dump**:

```
03:20  pg_dump | gzip → /home/veit/db_dumps/dbdata_JJJJ-MM-TT.sql.gz  (~100–150 MB)
03:40  collect_system_config.sh        (läuft schon)
04:00  NAS zieht /home/veit            (läuft schon)
       → CloudSync → Infomaniak
```

Eine konsistente Datei, fertig vor dem Backup-Fenster, danach trägt die
bestehende Kette sie ohne eine Zeile Änderung nach Infomaniak. Rücksicherung:
`gunzip | psql`. Weil der rsync **kein `--delete`** hat, dürfen die Dumps auf dem
VPS nach sieben Tagen weg — auf NAS und Cloud bleiben trotzdem alle liegen.
Ergibt Punkt-für-Punkt-Wiederherstellung über Monate für ~55 GB/Jahr bei 6 TB.

Damit liegt die Datenbank **erstmals überhaupt in einer eigenen Sicherung** —
heute deckt die Kette nur den VPS ab, nicht Centron.

### P2 — Die 18 Seiten als Flask-App neu bauen
Zielform wie der übrige Stack: Blueprint in `bundesliga_app.py`, Jinja-Templates,
gunicorn hinter nginx, eigene systemd-Unit (Muster: `boersenbot_optionen`).

**Die eine echte Übersetzungsleistung:** 16 der 18 Seiten arbeiten mit
`AutoPostBack` + `IsPostBack` (Ligaauswahl, Saison, Spieltag). WebForms hält den
Zustand in ViewState; in Flask werden daraus **GET-Parameter**
(`/bundesliga/tabelle?liga=bl1&saison=2025&spieltag=33`). Das ist mehr Denkarbeit
als Tipparbeit — und liefert nebenbei etwas, das die alte Seite nie konnte:
teilbare, lesezeichenfähige URLs.

Reihenfolge nach Abhängigkeit, nicht nach Wichtigkeit: erst `Tabelle`, `Heute`,
`Tore` (einfache Abfragen, etabliert das Muster), dann die Heatmaps
(`Heatmap` 913, `FormHeatmap` 940, `HalbzeitHeatmap` 653 Zeilen), zuletzt
`LayBacktest` (1.062 Zeilen, 21 Controls — die dickste Seite).

**Abnahme je Seite:** alte und neue Seite nebeneinander mit identischen
Parametern, Ergebnistabellen müssen zeilengleich sein. Weil beide nur lesen,
ist dieser Vergleich jederzeit gefahrlos möglich.

### P3 — Börsenbot-Code auf Postgres umstellen
- `pymssql` → `psycopg`; die 12 Dateien mit T-SQL-Eigenheiten zuerst.
- Gemeinsame Verbindungsfunktion einführen, statt die IP ein 40. Mal zu kopieren
  — hartkodierte Zugangsdaten bleiben Projektstandard, aber an **einer** Stelle.
- Superset-Container auf den neuen Treiber umhängen.

### P4 — Umschalttag (die einzige Stunde mit Risiko)
Verbotene Zeiten: **15:35/15:40 Mo–Fr** (`signal_to_orders.py`,
`random5_crossover_bot.py` mit `BOERSENBOT_LIVE=1`) und **06:00/18:00**
(Bundesliga-Import). Empfohlen: **Sonntagvormittag.**

1. Schreiber stoppen (NAS-Cron, Contabo-Timer)
2. Letzter Abgleich MSSQL → Postgres
3. NAS-`bundesliga` (8 Dateien) + Contabo (39) auf Postgres umstellen
4. Schreiber starten, einen Importlauf abwarten und prüfen
5. DNS `frau-von-allerliebst.de` → 144.91.98.234, nginx-vhost + certbot
   (Certbot läuft auf Contabo bereits)

### P5 — Nachlauf, dann kündigen
Centron **zwei Wochen weiterlaufen lassen** (Rückfallebene, DNS-TTL, stille
Nutzer). Erst danach kündigen. Ab dann: **11,50 €/Monat.**

---

## Risiken, ehrlich benannt

1. **Ein Ausfall trifft künftig alles.** Heute überlebt die DB einen
   Contabo-Ausfall. Danach nicht mehr. Gegenwert: 156 €/Jahr. Gemildert dadurch,
   dass die DB erstmals in der eigenen Backup-Kette liegt.
2. **Betriebsverantwortung wandert zu dir** — Updates, Sicherung, Verfügbarkeit
   waren Teil der 13 €.
3. **Port 5432 muss das NAS erreichen.** Heute ist MSSQL öffentlich erreichbar,
   sicherheitstechnisch wird es also nicht schlechter; sauber wäre WireGuard.
4. **Der NAS-Börsenbot-Zweig ist nicht in diesem Git-Repo.** Er taucht in der
   IP-Zählung nicht auf, holt seine Verbindung also anders — **vor P3 prüfen.**
5. **`web.config` enthält einen zweiten Schlüssel** (`OddsApiKey`), den keine der
   18 Seiten benutzt. Vor dem Abschalten klären, wer ihn braucht.
6. **RAM ist kein Thema mehr.** Nach dem Docker-Abbau sind 6,6 GB frei; Postgres
   neben 9 Diensten hat reichlich Luft.

## Aufwandsgefühl

P2 ist 80 % der Arbeit — 18 Seiten, ~10.000 Zeilen, davon drei große
Heatmap-Seiten und ein 1.000-Zeilen-Backtest. P1/P3/P4 sind Handwerk ohne
Unbekannte. Kein Abendprojekt, aber auch keine Forschung: es gibt keine offene
technische Frage mehr, nur Umsetzung.
