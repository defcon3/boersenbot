#!/usr/bin/env python3
"""
weather_stations.py — Stationsdaten fuer die Wetter-Bretter, EINE Quelle.

Warum es dieses Modul gibt: Die meisten Aufloesungsstationen sind Flughaefen und
stehen in airportsdata. Manche nicht — und deren Sonderdaten wurden bisher in
jeder Datei erneut gebraucht. Die STATIONS-Maps liegen ohnehin schon in vier
Kopien (weather_source_compare, weather_outlier_screen, weather_outlier_screen_low,
weather_ladder_logger); dass Moskau am 25.07. in nur drei davon landete und die
vierte bis zum 26.07. zurueckblieb, ist genau der Schaden, den die Kopier-Lehre
aus dem Beijing-33-Verlust beschreibt: ein Eintrag in nur einer Kopie ist kein
Eintrag. Neuer gemeinsamer Code kommt deshalb hierher.

BEWUSST LEICHTGEWICHTIG: nur stdlib + airportsdata. weather_source_compare zieht
numpy/scipy mit; der Ladder-Logger laeuft auf dem VPS im Timer und soll dafuer
nicht den Kalibrierer importieren muessen.
"""

import airportsdata

_AP_CACHE = None

# Aufloesungsstationen, die KEINE Flughaefen sind und deshalb nicht in
# airportsdata stehen. Felder wie bei airportsdata (lat/lon/country/tz), damit
# die Verbraucher denselben Zugriff benutzen koennen.
SPECIAL_STATIONS = {
    # Hong Kong Observatory, Tsim Sha Tsui. Die HK-Bretter settlen auf die
    # HKO-Klimareihe ("Absolute Daily Max (deg. C)", weather.gov.hk), auf EINE
    # Dezimalstelle genau — nicht auf Wunderground und nicht auf METAR.
    # VHHH (Chek Lap Kok) ist KEIN Ersatz: ueber 700 Tage liegt es an 68 % der
    # Tage in einem anderen ganzen Grad (MAE 0,92 C), gemessen 26.07.
    # Actuals holt man mit weather_source_compare.fetch_actual_daily_extreme_hko;
    # METAR und Wunderground haben fuer diese Station NICHTS.
    "HKO": {"name": "Hong Kong Observatory", "country": "HK",
            "lat": 22.3019, "lon": 114.1740, "tz": "Asia/Hong_Kong"},
}

# Stationen ohne METAR/Wunderground-Ist. Settle-Pfade muessen sie ueberspringen,
# statt an einem leeren Abruf haengenzubleiben.
NO_METAR = {"HKO"}
NO_WUNDERGROUND = {"HKO"}

# Staedte, deren BUCKET-SEMANTIK noch nicht geklaert ist: Kalibrierung liegt vor,
# aber es steht nicht fest, welches Intervall ein Bucket-Titel meint. Fuer diese
# Staedte wird KEIN mu_ens geschrieben — ohne mu gibt es kein offset_fav, und
# ohne offset_fav sieht der -1-Autobuy sie nicht. Die Preiszeilen werden trotzdem
# geloggt, die sind auch ohne Modell wertvoll.
#
# Hong Kong (26.07.): Die HKO misst auf EINE Dezimalstelle (32.3), und die
# Marktregel spricht vom "temperature range that CONTAINS" den Wert. Bei allen
# anderen Staedten liefert die Quelle ganze Grad, und "28C" meint half_up-
# gerundet [27,5 .. 28,5).
#
# GEMESSEN am 26.07. an 18 gesettelten HK-Brettern (Juli 2026, max und min
# gemischt; Markt-result gegen den HKO-Ist-Wert aus dem tagesaktuellen Daily
# Extract, hko.gov.hk/cis/dailyExtract/dailyExtract_YYYYMM.xml — liefert JSON):
#
#     floor 11 | half_up 0 | nicht unterscheidbar 7 | Widerspruch 0
#
# Also gilt hier FLOOR: "28C" meint [28,0 .. 28,9]. Belege u. a. 12.07. max 33,8
# -> Markt 33 (nicht 34), 18.07. max 30,5 -> Markt 30, 15.07. min 25,7 -> Markt 25.
#
# Die Sperre bleibt trotzdem, bis der Code das umsetzt — betroffen sind ZWEI
# Stellen, nicht nur eine: (a) der Favorit k0 = half_up(mu) im Ladder-Logger und
# (b) die Bucket-Wahrscheinlichkeit ncdf((k+0,5-mu)/s) - ncdf((k-0,5-mu)/s) in den
# Screens und in weather_error_quantiles.offsets_for(), deren Grenzen fuer
# floor-Staedte auf [k, k+1] wandern muessen. Ohne (b) laege jede Bucket-Chance
# ein halbes Sigma daneben. Geplant als Stadt-Eigenschaft BUCKET_FLOOR hier im
# Modul; danach Hong Kong hier austragen.
MU_PENDING = {"Hong Kong"}


def mu_erlaubt(city):
    """Darf fuer diese Stadt ein mu_ens/offset_fav berechnet werden?"""
    return city not in MU_PENDING


def airports():
    """airportsdata einmal laden (der Aufruf kostet spuerbar)."""
    global _AP_CACHE
    if _AP_CACHE is None:
        _AP_CACHE = airportsdata.load("ICAO")
    return _AP_CACHE


def station_info(icao):
    """Stationsdaten zu einem ICAO/Pseudo-ICAO oder None.

    Deckt airportsdata UND die Sonderstationen ab — ueberall dort benutzen, wo
    frueher direkt AP[icao] bzw. AP.get(icao) stand, sonst faellt eine
    Sonderstation stumm durch das 'keine Station -> skip' der Aufrufer."""
    if not icao:
        return None
    return airports().get(icao) or SPECIAL_STATIONS.get(icao)


def has_metar(icao):
    """Liefert die Station eine METAR-Reihe (IEM)? Fuer Settle-Pfade."""
    return icao not in NO_METAR


def has_wunderground(icao):
    """Speist Wunderground eine Seite fuer diese Station? Fuer Settle-Pfade."""
    return icao not in NO_WUNDERGROUND
