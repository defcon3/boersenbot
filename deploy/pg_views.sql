-- Die drei Views der Centron-`dbdata`, nach PostgreSQL portiert (P1).
-- Von Hand uebersetzt statt generiert: nur eine der drei ist dialektbehaftet,
-- und genau die braucht eine bewusste Entscheidung (Zeitzone).
--
--   psql -h 127.0.0.1 -U boersenbot -d dbdata -f pg_views.sql

-- 1) vw_StockStreaks — reines Standard-SQL, 1:1 uebernommen.
--    LAG/LEAD/ROW_NUMBER gibt es in PostgreSQL unveraendert.
DROP VIEW IF EXISTS vw_stockstreaks;
CREATE VIEW vw_stockstreaks AS
WITH base AS (
  SELECT symbol, date, closeprice,
         LAG(closeprice) OVER (PARTITION BY symbol ORDER BY date) AS prevclose
  FROM bb_stockprices
),
ret AS (
  SELECT symbol, date, closeprice,
         CASE WHEN prevclose IS NULL THEN NULL
              ELSE (closeprice - prevclose) / prevclose END AS ret
  FROM base
),
dir AS (
  SELECT symbol, date, closeprice, ret,
         CASE WHEN ret IS NULL THEN NULL
              WHEN ret < 0 THEN -1
              WHEN ret > 0 THEN 1 ELSE 0 END AS dir
  FROM ret
),
grp AS (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date)
    - ROW_NUMBER() OVER (PARTITION BY symbol, dir ORDER BY date) AS island
  FROM dir
  WHERE dir IS NOT NULL
)
SELECT symbol, date, closeprice, ret, dir,
       ROW_NUMBER() OVER (PARTITION BY symbol, dir, island ORDER BY date) AS streaklen,
       LEAD(ret) OVER (PARTITION BY symbol ORDER BY date) AS nextret
FROM grp;

-- 2) bb_StockPrices_1min_Combined — die einzige dialektbehaftete View.
--    T-SQL: CAST(ts AS datetime2) AT TIME ZONE 'Eastern Standard Time' AT TIME ZONE 'UTC'
--    PG:    (ts AT TIME ZONE 'America/New_York') AT TIME ZONE 'UTC'
--    Beide Fassungen lesen den naiven Kaggle-Wert als Ortszeit New York
--    (inkl. Sommerzeit) und geben einen naiven UTC-Wert zurueck.
--    Die Windows-Zonen-ID 'Eastern Standard Time' traegt die DST-Regeln mit,
--    ihre IANA-Entsprechung ist deshalb 'America/New_York', nicht 'EST'.
--    "timestamp" bleibt gequotet — unquoted liest der PG-Parser dort einen Typ.
--
--    GEMESSEN 08.08.2026, beide Engines gegeneinander:
--      * alle 13.624 Kaggle-Zeitstempel im Bestand: identisch
--      * Winter (Jan), Sommer (Jul), Vorstell-Luecke (12.03. 01:30/03:30): identisch
--      * Rueckstelltag (05.11. 01:30): **eine Stunde Unterschied.**
--        Diese Ortszeit existiert zweimal; MSSQL loest sie als Sommerzeit auf
--        (05:30 UTC), PostgreSQL als Standardzeit (06:30 UTC). PG bietet keinen
--        Schalter fuer die Ambiguitaets-Aufloesung.
--    Bewusst NICHT nachgebaut: Handelszeit ist 09:30–16:00 ET, die doppelte
--    Stunde 01:00–02:00 kommt in Kursdaten nicht vor. Sollte der 1-Min-Bestand
--    je auf 24h erweitert werden, ist genau dieser Tag neu zu pruefen.
DROP VIEW IF EXISTS bb_stockprices_1min_combined;
CREATE VIEW bb_stockprices_1min_combined AS
SELECT
    k.symbol,
    (k."timestamp" AT TIME ZONE 'America/New_York') AT TIME ZONE 'UTC' AS "timestamp",
    k.openprice, k.highprice, k.lowprice, k.closeprice, k.volume,
    'Kaggle'::varchar(8) AS source
FROM bb_stockprices_1min_kaggle k
UNION ALL
SELECT
    y.symbol, y."timestamp",
    y.openprice, y.highprice, y.lowprice, y.closeprice, y.volume,
    'Live'::varchar(8) AS source
FROM bb_stockprices_1min_yfinance y;

-- 3) vMatchesWithOdds — Standard-SQL, 1:1 uebernommen.
DROP VIEW IF EXISTS vmatcheswithodds;
CREATE VIEW vmatcheswithodds AS
SELECT
    m.matchid,
    s.leagueshortcut,
    s.leaguename,
    s.seasonyear,
    m.matchday,
    m.matchdatetime,
    t1.teamname              AS hometeam,
    t2.teamname              AS awayteam,
    m.fulltimegoalsteam1     AS homegoals,
    m.fulltimegoalsteam2     AS awaygoals,
    m.isfinished,
    CASE
        WHEN m.fulltimegoalsteam1 > m.fulltimegoalsteam2 THEN 'H'
        WHEN m.fulltimegoalsteam1 < m.fulltimegoalsteam2 THEN 'A'
        WHEN m.fulltimegoalsteam1 = m.fulltimegoalsteam2 THEN 'D'
        ELSE NULL
    END                      AS result,
    bm.bookmakercode,
    bm.bookmakername,
    o.homeodds,
    o.drawodds,
    o.awayodds,
    o.over25odds,
    o.under25odds
FROM matches m
INNER JOIN seasons    s  ON s.seasonid     = m.seasonid
INNER JOIN teams      t1 ON t1.teamid      = m.team1id
INNER JOIN teams      t2 ON t2.teamid      = m.team2id
LEFT  JOIN odds       o  ON o.matchid      = m.matchid
LEFT  JOIN bookmakers bm ON bm.bookmakerid = o.bookmakerid;
