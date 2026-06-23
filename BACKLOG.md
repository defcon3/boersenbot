# BACKLOG — aufgeschobene Ideen / nächste Forschungs-Kandidaten

Zentrale Liste bewusst zurückgestellter Aufgaben (nicht „jetzt", aber nicht vergessen).
Neueste oben. Erledigtes nach unten in „## Erledigt / verworfen" oder löschen.

---

## Offen

### Overnight-Edge auf SIP-30-Min-Datensatz neu testen
**Hinzugefügt:** 2026-06-23 · **Status:** BACKLOG, nicht begonnen · **Priorität:** mittel

**Idee:** Die Zerlegung der SPY-Tagesrendite in **Overnight** (Close→Open) vs.
**Intraday** (Open→Close) erneut auf sauberen Vollmarkt-Daten prüfen. Bekannter
Literatur-Befund: der Großteil der Aktienprämie fällt overnight an, intraday ist
historisch ~flach/negativ.

**Warum jetzt möglich (was sich geändert hat):** In der Session 2026-06-23 hat sich
herausgestellt, dass **Alpacas SIP-Feed gratis für Historie >15 Min** ist und bis
**2016** zurückreicht (Vollmarkt, sauber). Tool dafür existiert bereits:
`fetch_spy_30min.py` (läuft auf VPS, Keys in `config.py`). Damit entfällt die
yfinance-30-Tage-Grenze, die den bisherigen Intraday-Strang limitierte.

**Vorheriger Stand (nicht bei Null anfangen):** Der Overnight/Intraday-Strang lief
schon, endete aber **YELLOW / G0-limitiert**:
- `overnight_intraday_g3_pit.py`, `overnight_intraday_rolling_bootstrap.py`,
  `overnight_es_futures_g0check.py`, Cache `spy_overnight_ohlc.pkl`.
- Commits: `97b3b371` (Rolling-Stabilität + Bootstrap-KI + Strukturbruch),
  `0c6b0487` (Pre-Reg ES-Futures), `c21ce7ce` (ES-Futures **G0-FAIL** + YELLOW-Schluss).
- Knackpunkt war Datenqualität/Granularität — genau das adressiert der SIP-Datensatz.

**Vorbefund (2026-06-23, `cross_session_test.py`):** Erste Hinweise schon da.
Auf SPY 30-Min SIP (2016–2026, ~2630 Nächte): Nachmittag[t]→Vormittag[t+1] ist
**null** (corr −0,015, Hit 50,5 %). ABER **PM[t]→Overnight-Gap[t+1] = −0,156**
(t≈8) und PM[t]→PM[t+1] = −0,108 → **milde Overnight-/Tages-Reversion** der
Spätbewegung (R² nur ~2,4 %). Statistisch real, aber vermutlich NICHT handelbar:
zum Ausnutzen müsste man über Nacht short gehen (gegen positiven Overnight-Drift
+ ~72 bps Gap-Varianz + Spread/Finanzierung). Genau hier ansetzen beim echten Test.

**Konkrete nächste Schritte:**
1. `fetch_spy_30min.py` verallgemeinern → ggf. mehr Symbole / nur Open- & Close-Bars
   ziehen (Overnight braucht Open + vorigen Close, nicht zwingend volle 30-Min-Bars).
2. Pre-Reg G1–G5 schreiben (Overnight-Rendite vs. Intraday, netto nach Kosten;
   Achtung: Overnight ist NICHT handelbar ohne Übernacht-Halten → Strategie sauber
   definieren, z. B. „nur overnight halten" via MOC-Kauf / MOO-Verkauf).
3. OOS-Split + COVID-Kontrolle wie gehabt. Kosten realistisch (Übernacht-Spread,
   Finanzierung). Ehrlicher FAIL einkalkuliert (Projekt-Gesamtlage: Edges sterben OOS).

**Verweise:** `intraday_volume_profile.py` (U-Form-Motivation),
`preregs/intraday_momentum_spy_2026_06_23.md` (Methodik-Vorlage, HAC-Implementierung
in `intraday_momentum_test.py` wiederverwendbar).
