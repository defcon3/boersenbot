# Seoul-Seebrise-Deckel — Befund & Pre-Reg (2026-07-22)

**Kontext:** Intraday-Nowcast-Scalp (Prio 1). Ausgangsidee (Nutzer): nähere
Wetterstationen + Windrichtung von Wunderground einbeziehen — bei Wind aus einer
Richtung die dortige Station als Frühindikator heranziehen (Upwind-Advektion).

## Meteorologischer Befund (deskriptiv, IEM ASOS, N=780 Tage, Jun–Aug 2018–2026)

Settlement-Station **Seoul = RKSI (Incheon)** ist eine **Küstenstation** und an
49/52 Tagen die kälteste im Großraum. Nachbarn: RKSS (Gimpo, NO 32 km),
RKSM (Seoul-AB, O 59 km), RKSO (Osan, SO 66 km) — alle im Ost-/Inlandsektor.

Offset **RKSS − RKSI** ist monoton in RKSIs **eigener Nachmittags-Windrichtung**
(lokal 12–16 h, Median):

| Wind aus | n | Offset RKSS−RKSI |
|---|---|---|
| E | 28 | +0,25 |
| NE | 37 | +0,54 |
| SE | 45 | +0,64 |
| S | 105 | +1,11 |
| SW | 181 | +1,68 |
| NW | 130 | +2,09 |
| W | 249 | +2,43 |

- **Ostsektor (NE/E/SE), n=110:** RKSS ≈ RKSI (Offset +0,51±0,94; Bucket exakt
  38 %, |diff|≤1 **85 %**) → RKSS als Echtzeit-Proxy nutzbar, **ohne** Korrektur.
- **Westsektor (W/NW/SW), n=560 = 72 % der Sommertage:** RKSI gedeckelt
  (Offset +2,1±1,2). Nachbar als Proxy **untradbar** (Bucket exakt 7 % roh,
  36 % selbst mit −2 °C-Korrektur). Starker Wind >6 kt → Offset 2,16 vs 1,65
  (schwach) → Seebrise-Mechanik bestätigt.

**Zwei Nebenbefunde, die die ursprüngliche Idee einschränken:**
1. Zeitliches Lead-Lag RKSS→RKSI ≈ **0 min** (r=0,92 synchron) → Nachbarn
   liefern **keinen** Frühwarn-Vorsprung. RKSM/RKSO hinken sogar hinterher.
2. Nur **RKSS** ist präzise genug (std 0,94); RKSM/RKSO std 1,4–2,1 → verworfen.

## Tradbare Hypothese (gegen Marktpreise zu testen)

Nicht die Nachbarn sind der Edge, sondern **RKSIs Windrichtung selbst**:

> **H:** Dreht der Nachmittagswind an einem forecast-heiß gepreisten Seoul-Tag
> auf **West/Seebrise**, wird RKSIs Settlement-Max ~2–2,5 °C unter das
> Inland-Potenzial gedeckelt. Der dünne Jupiter-Markt preist das nicht sofort
> → **Downside-Edge** (untere Buckets unterbewertet).

**Gates (an Jupiter-Preisen in `bb_WeatherLatency` zu prüfen):**
- **G1** An Westwind-Tagen liegt der realisierte RKSI-Bucket signifikant unter
  dem Markt-Favoriten zum Zeitpunkt des Winddrehers (mittlere Bucket-Differenz
  Markt−Ist > 0, t > 2).
- **G2** Der Effekt hält OOS (z. B. Tage nach 15.07., als der Logger lief) und
  ist nicht durch 1–2 Extremtage getrieben.
- **G3** Netto nach Fee (~5 % von min(p,1−p)) bleibt ein positiver EV auf dem
  Lay der Markt-Favoriten-/oberen Buckets.
- **G4** Genügend West-Signal-Tage/Monat für Handelbarkeit.

**Status:** Meteorologie bestätigt. Markt-Test = nächster Schritt
(`weather_seoul_seabreeze_market.py`, WIP). Kein Echtgeld bis Gates grün.
