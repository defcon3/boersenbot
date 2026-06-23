# BACKLOG — aufgeschobene Ideen / nächste Forschungs-Kandidaten

Zentrale Liste bewusst zurückgestellter Aufgaben (nicht „jetzt", aber nicht vergessen).
Neueste oben. Erledigtes nach unten in „## Erledigt / verworfen" oder löschen.

---

## Offen

_(derzeit keine offenen Einträge)_

---

## Erledigt / verworfen

### VRP-Sleeve mit Hybrid-Risk-System kombinieren
**Status:** ✅ ERLEDIGT 2026-06-23 — **GREEN (qualifiziert)**. `vrp_hybrid_combo.py`:
VRP-gemanagt + Hybrid, monatlich, gemeinsames Fenster 2006–2026, Gewichte auf IS
gewählt / OOS evaluiert, lookahead-freier Hybrid. corr(Hybrid, VRP) OOS −0,10.
**OOS-Gewichts-Sweep:** mit steigendem VRP-Anteil sinkt MaxDD monoton
(−18,5 % → −12,0 % @50 %), Sharpe steigt (0,75 → 0,86), aber Rendite fällt
(9,8 % → 6,2 %). Diversifikation real & robust (nicht knife-edge).
**Caveats:** (a) De-Risking, kein Rendite-Boost; (b) schlägt OOS NICHT die nackte
SPY-Sharpe (0,91 — nur deren MaxDD −24 %); (c) IS-optimales Gewicht 60 % ist
overfit-anfällig (50/50 war OOS besser) → **modestes VRP-Gewicht ~25–40 % empfohlen**;
(d) Skew verschlechtert sich (Short-Vol-Charakter bleibt). Fazit: VRP **validiert als
diversifizierender Drawdown-Dämpfer** für den Hybrid, nicht als Rendite-Maschine.
Vor Live: Straddle-Approximation gegen echte Options-Daten prüfen. Test `vrp_hybrid_combo.py`.

### Overnight-Edge auf SIP-30-Min-Datensatz neu testen
**Hinzugefügt:** 2026-06-23 · **Status:** ✅ ERLEDIGT 2026-06-23 — **RED** (per Pre-Reg,
SPY primär): G2 OOS + G3 handelbar gescheitert. Phänomen deskriptiv real (QQQ/IWM),
aber Overnight-only schlägt Buy&Hold nie netto → keine Gelddruckmaschine. Details:
`preregs/overnight_intraday_2026_06_23.md`, Test `overnight_edge_test.py`.

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
