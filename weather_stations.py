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
# gerundet [27,5 .. 28,5). Hier waere "28C" eher [28,0 .. 29,0) — ein halbes Grad
# Versatz. Konkret gerechnet fuer den 27.07.: mu = 29,61 ergibt half_up 30, aber
# floor 29. Zwei verschiedene Favoriten, also offset_fav um 1 verschoben und der
# Autobuy laegt den falschen Bucket. Klaeren an einem gesettelten HK-Brett
# (Bucket-Titel gegen den HKO-Ist-Wert), danach hier austragen.
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
