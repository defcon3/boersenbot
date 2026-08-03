# -*- coding: utf-8 -*-
"""weather_outlier_screen.py — Lay-Kandidaten fuer Jupiter-Wetter-Buckets
("Highest temperature in X on <Datum>?").

Entstanden 2026-07-08 (Generalisierung des Session-Screenings vom 07./08.07.,
das zweimal im fluechtigen Scratchpad neu gebaut werden musste — deshalb jetzt
im Repo). Methodik:

  1. Jupiter /events (category=weather) -> alle Grad-Buckets + Preise des Zieltags
  2. Open-Meteo-Forecast (GFS/ICON/UKMO/JMA/ECMWF) -> Tageshoch je Modell, lokale Zeit
  3. Bias/Sigma-Kalibrierung (700d, Lead 24h) aus preregs/weather_source_calib_*.csv
     -> P(Bucket) je Modell + Ensemble (Normal-Annahme, Bucket = [k-0.5, k+0.5))
  4. Kandidat (Lay/NO) nur wenn: Abstand Bucket<->korr. Ensemble-Forecast >= MIN_DIST,
     KEIN Modell gibt dem Bucket > MAX_PMODEL, YES-Preis >= MIN_YES (sonst Rendite tot).

Lektion 08.07. ("Cape Town zu knapp"): 1 Grad neben dem Favoriten ist keine Marge;
die Gewinner-Struktur ist grosser Grad-Abstand + Modell-Einigkeit, nicht nur Preis.

Lektion 11.07. ("Shanghai 28C Regentag"): Die Kalibrierung ist ein Allwetter-
Durchschnitt. Sehen die Modelle fuer den Zieltag nennenswert Regen/Sturm, VOR dem
Setzen weather_wet_conditional.py --city X laufen lassen und P gegen den Nass-
Split halten (Shanghai: kalte Flanke an >=9mm-Tagen 4-6x fetter, Kandidat trotz
bestandener Doppel-Kalibrierung verworfen).

Lektion 14.07. ("Beijing 33C" — der erste echte Verlierer, siehe
preregs/weather_lay_postmortem_2026_07_14_beijing.md): Der Lay wurde von einem
JMA-Lauf getragen, der 3,6°C ueber den anderen vier Modellen lag — und am Ende
4,1°C danebenlag. Das arithmetische Ensemble-Mittel hat gegen so etwas keine
Abwehr; die daraus abgeleiteten 5,6 % waren ein Artefakt. Ohne den Ausreisser und
mit der Sommer- statt der Ganzjahres-Kalibrierung: P 20,3 % vs BE 21,0 % = kein
Trade. Daraus vier Aenderungen, alle unten als Konstanten:
  1. Ausreisser-robustes Ensemble (OUTLIER_DEG) — gewertet wird die
     PESSIMISTISCHERE von voller und bereinigter Sicht.
  2. Harter Spannen-Veto (MAX_SPREAD) — ein Ensemble mit >3°C Streuung ist als
     Handelsgrundlage wertlos, egal wie die gemittelte Zahl aussieht.
  3. Doppel-Kalibrierung im Code statt im Kopf: 700d (Ganzjahr) UND 40d (Sommer)
     muessen bestehen. Fuer Beijing dreht der ENS-Bias das Vorzeichen
     (700d -0,88 / 40d +0,02) — die 700d-Sicht schiebt Buckets UNTER dem
     Forecast kuenstlich weit weg.
  4. EV-Marge statt "P < BE" (MIN_EV) — ein Pass mit +0,7pp ist kein Trade.

Nachtrag 14.07. (Pre-Reg preregs/weather_spread_sigma_2026_07_14.md, ausgewertet):
Das feste Sigma je Stadt ist ein Mittel ueber ALLE Spannen — und damit auf RUHIGEN
Tagen viel zu breit (Sommer-OOS: sagt 3,3 %, liefert 1,4 %; Faktor 0,43). Der Screen
rechnete dort also zu hohe P und liess sichere Lays liegen. Das Ensemble-Sigma ist
jetzt spannen-konditioniert: sigma(s) = a_city + b*s (siehe ens_sigma()). a und b
stehen in der Kalibrier-CSV, NICHT als Konstante im Code — die Steigung ist
saisonabhaengig (Sommer 0,107 / Winter 0,177 / Ganzjahr 0,140), das 40d-Sommer-
Fenster muss also seine eigene benutzen duerfen.

Was dabei ausdruecklich NICHT passiert ist: den Spannen-Veto durch sigma(s) zu
ersetzen. G4 der Pre-Reg zeigte, dass das ALLE vetoierten Buckets wieder geoeffnet
haette — inklusive Beijing 32° mit der Signatur des Verlierers. Die Modellspanne ist
eben nicht nur ein Breiten-Signal, sondern vor allem eines fuer ein korrumpiertes mu
(ein Ausreisser zieht das Mittel). Kein Sigma repariert einen verzogenen Mittelwert.
sigma(s) wirkt daher nur INNERHALB der nicht-vetoierten, ruhigen Zone.

Nachtrag 16.07. (Wetterfrosch-Doktrin, nach dem BA-Brett vom 15.07. in mehreren
Runden vom Betreiber geschaerft): Der Screen ist ein WETTERFROSCH, kein Markt-
Beobachter. Er bildet aus den Quellen eine eigene, KONSISTENTE Aussage zum
Endergebnis (korrigiertes Ensemble; ein stabiler eigener Fehler — "immer 1
drueber oder 1,3" — waere ok, denn er ist kalibrierbar; Konsistenz zaehlt).
Ausgewaehlt wird allein relativ zu DIESER Aussage: Sicherheit = Abstand zur
eigenen Prognose, Ranking = P_pess aufsteigend ("lieber weniger Profit als
Kipp-Risiko", Spatz in der Hand). Der Markt liefert nur noch den PREIS
(Profitschwelle MIN_YES) — er ist KEIN Auswahlkriterium: wer den Favoriten
anschaut, hat bereits den Markt und die anderen Teilnehmer im Blick. Deshalb
entfiel am 16.07. das Beijing-Gate Nr. 4 (MIN_EV — Markt-relativ); die
markt-blinden Qualitaets-Gates (Spanne, dist, P_max, Doppel-Kalibrierung)
bleiben, denn sie sichern die Konsistenz der eigenen Aussage. "Setze 3 ist
setze 3": die 3 sichersten ueber der Profitschwelle — geben die Quellen weniger
als 3 konsistente Kandidaten her, wird das explizit gemeldet (Betreiber
entscheidet), nicht still gelockert und nicht still reduziert. Beleg BA 15.07.:
Der 23er lag 1,6° von der EIGENEN Prognose (21,4°) — das war der Fehler, nicht
die Fav-Naehe an sich; die P der drei Lays stieg ueber Nacht 14->30 / 2,7->14 /
0,2->3,2 %, exakt skalierend mit der Naehe zur Prognose. Das EV-Ranking hatte
genau diesen Bucket nach oben gestellt.

Nachtrag 17.07. (Backlog Prio 0+1, preregs/weather_calib_prio01_2026_07_17.md):
  1. DEBIAS-VOR-MITTELUNG (build_views): jedes Modell wird erst um seinen eigenen
     Kalibrier-Bias korrigiert, dann gemittelt — statt rohes Mittel minus
     ENS-Bias. Behebt die gemessene Inkonsistenz bei Modell-Ausfall (bis +0,98°,
     Jeddah ohne JMA; UKMO fehlt an ~150/700 Kalibriertagen) und macht die
     robuste Sicht ehrlich (sie bekam vorher den vollen ENS-Bias).
  2. sigma(s) auch fuer die EINZELMODELL-Vetos (model_sigma) — die Kalibrier-CSVs
     tragen a/b seit 17.07. je Quelle, nicht nur fuer ensemble_mean.
  3. LEAD-AUTOWAHL: bei Zieltag >24h laedt der Screen die Lead-N-Kalibrierfamilie
     (weather_source_calib_leadN_*) oder bricht mit Anleitung ab (--force-lead1
     als markierter Notbehelf). Vorher nur eine Warnung — die 17.07.-Lays wurden
     am 15.07. faktisch mit Lead-24h-P gescreent.
  4. ZURUECKGENOMMEN am 02.08.2026. Von 17.07. bis dahin wurde Shenzhen gegen
     die "WU-Settlement-Reihe" kalibriert (--actuals wu, eigene ueberschreibende
     CSV ..._shenzhen_wu.csv), weil WU dort nur an 4/15 Tagen mit dem METAR
     uebereinstimmte. Der Grund fuer die Abweichung war ein anderer als gedacht:
     der Locator ZGSZ:9:CN liefert gar nicht Shenzhen Bao'an, sondern die
     Station "Lau Fau Shan" (obs_id 45035) rund 25 km entfernt in Hong Kong. Die
     Sonder-CSVs kalibrierten also gegen eine andere Stadt und blaehten sigma um
     rund 37 % auf (ensemble_mean 1,49 statt 1,08). Sie sind entfernt; Shenzhen
     laeuft wie alle anderen Staedte auf METAR. Beleg: der Markt loeste an allen
     fuenf uneinigen Stadt-Tagen nach METAR auf, nie nach WU.

Aufruf:
  python weather_outlier_screen.py                  # Zieltag = morgen (UTC)
  python weather_outlier_screen.py --date 2026-07-10
Setzen dann manuell via jupiter_buy.py --market POLY-XXXX --no --usd 5 --limit <Ask+0.02> --send
"""
import argparse
import csv
import glob
import math
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import airportsdata

from weather_stations import (bucket_grenzen, canonical_city, favorit_k,
                              station_info)
import requests

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

API = "https://api.jup.ag/prediction/v1"
OM = "https://api.open-meteo.com/v1/forecast"
# Zwei Kalibrier-Familien, beide Lead-24h. Die 40d-Sicht ist die Sommer-Sicht und
# weicht teils drastisch ab (Beijing-ENS-Bias 700d -0,884 vs 40d +0,017; Seoul -3,08).
# Erzeugt mit: weather_source_compare.py --days N --calib-csv <pfad>
CALIB_GLOB = r"preregs/weather_source_calib_*.csv"      # 700d, Ganzjahr
CALIB40_GLOB = r"preregs/weather_source_calib40d_*.csv"  # 40d, Sommer

MODELS = ["gfs_seamless", "icon_seamless", "ukmo_seamless", "jma_seamless", "ecmwf_ifs025"]
SHORT = {"gfs_seamless": "GFS", "icon_seamless": "ICON", "ukmo_seamless": "UKMO",
         "jma_seamless": "JMA", "ecmwf_ifs025": "ECMWF", "ensemble_mean": "ENS"}

# Aufloesungs-Stationen (Polymarket-Beschreibung = Wunderground-Station!).
# Basis: weather_source_compare.STATIONS; +6 vom 08.07. nachaufgeloest.
# ACHTUNG Panama City = MPMG (Albrook, Stadtflughafen), NICHT Tocumen MPTO.
STATIONS = {
    "Wellington": "NZWN", "Tokyo": "RJTT", "Seoul": "RKSI", "Shanghai": "ZSPD",
    "Beijing": "ZBAA", "Kuala Lumpur": "WMKK", "Shenzhen": "ZGSZ", "Chengdu": "ZUUU",
    "Karachi": "OPKC", "Jeddah": "OEJN", "Ankara": "LTAC", "Helsinki": "EFHK",
    "London": "EGLC", "Paris": "LFPB", "Madrid": "LEMD", "Milan": "LIMC",
    "Munich": "EDDM", "Amsterdam": "EHAM", "Warsaw": "EPWA", "Cape Town": "FACT",
    "Mexico City": "MMMX", "Buenos Aires": "SAEZ",
    "Sao Paulo": "SBGR", "Taipei": "RCSS", "Tel Aviv": "LLBG",
    "Toronto": "CYYZ", "Wuhan": "ZHHH", "Panama City": "MPMG",
    # Moskau (25.07.): UUWW Vnukovo, laut Polymarket-Beschreibung. ACHTUNG —
    # dieser Markt settelt ueber NOAA (weather.gov/wrh/timeseries?site=UUWW),
    # nicht ueber Wunderground wie die uebrigen Staedte.
    "Moscow": "UUWW",
    # Hong Kong nachgeruestet 26.07.: Pseudo-Station "HKO" (weather_stations.
    # SPECIAL_STATIONS). Settelt auf die HKO-Klimareihe, NICHT auf METAR/
    # Wunderground — Settle-Pfade muessen die Station ueberspringen.
    "Hong Kong": "HKO",
    # NYC nachgeruestet 26.07.: Settlement-Station laut Marktregel ist
    # LaGuardia (KLGA), NICHT Central Park — "the lowest temperature recorded at
    # the LaGuardia Airport Station", Quelle wunderground.com/history/daily/us/
    # ny/new-york-city/KLGA. ACHTUNG: Die NYC-Bretter laufen in FAHRENHEIT mit
    # 2-Grad-Buckets ("70-71F"); der Ladder-Logger verwirft sie deshalb weiter
    # (celsius-Pruefung), die Kalibrierung hier ist einheitenunabhaengig.
    "NYC": "KLGA",
}

MIN_DIST = 2.0     # Grad Abstand Bucket <-> korrigierter Ensemble-Forecast
MAX_PMODEL = 0.10  # kein Modell darf dem Bucket mehr geben (ueber BEIDE Kalibrierungen)
MIN_YES = 0.025    # Bucket muss am Markt noch nennenswert gepreist sein (Rendite)
# --- Konsequenzen aus dem Beijing-33-Verlust (14.07.), siehe Modulkopf ---
MAX_SPREAD = 3.0   # Modellspanne (roh, max-min) in Grad -> darueber kein Kandidat
OUTLIER_DEG = 2.0  # Modell > X Grad vom Median der UEBRIGEN = Ausreisser, fliegt aus dem Mittel
# MIN_EV (Beijing-Gate Nr. 4, 0.05) entfiel am 16.07.: EV ist Markt-relativ und
# damit kein Auswahlkriterium mehr (Wetterfrosch-Doktrin, s. Nachtrag oben);
# EV wird weiter angezeigt, entschieden wird markt-blind + Profitschwelle MIN_YES.
SIGMA_FLOOR = 0.3  # untere Schranke fuer sigma(s); a und b kommen aus der Kalibrier-CSV
# --- Einigkeits-Gate (28.07.) ---
# Die Doppel-Kalibrierungs-Regel war bis hierhin nur zur Haelfte im Code: beide
# Sichten MUESSEN bestehen (seit 14.07.), aber wie weit ihre mu auseinanderliegen,
# war kein Kriterium. Genau daran wurden Seoul 23 (12.07.) und Hong Kong min
# (26.07.) von Hand aussortiert — Kopfsache, also irgendwann vergessen.
# Die Wetterfrosch-Doktrin begruendet es: eine Stadt ohne konsistente eigene
# Aussage ist nicht prognostizierbar, also nicht handelbar.
#
# GEMESSEN (weather_calib_divergence_eval.py, 28.07., 27 Staedte x ~10 gesettelte
# Zieltage): Staedte mit D >= 0,7 treffen den Ist-Bucket in 22 % der Faelle,
# Staedte darunter in 39 % (|e| 1,04 vs 0,76). Der Zusammenhang ueber alle
# Staedte ist da, aber nicht stark: r = +0,36, t = 1,94.
# EHRLICH DAZU: vier Schwellen (0,3/0,5/0,7/1,0) angesehen, alle zeigen dieselbe
# Richtung; 0,7 hat den groessten Kontrast. Die Zahl ist damit nicht
# out-of-sample belegt, sondern eine Doktrin-Umsetzung mit stuetzender Messung.
# Kosten: 24 % der bisherigen -1-Kandidatentage (Beijing, Jeddah, Munich, Seoul,
# Taipei, Tel Aviv).
MAX_DIVERGENZ = 0.7  # |bias_700d - bias_40d| des Ensembles in K -> darueber kein Kandidat

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0"


def ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bucket_prob(kind, k, mu, sigma, city=""):
    """P(Tageshoch faellt in Bucket). Grenzen aus weather_stations.bucket_grenzen:
    normal [k-0.5, k+0.5), fuer BUCKET_FLOOR-Staedte (Hong Kong) [k, k+1)."""
    lo, hi = bucket_grenzen(k, city)
    if kind == "le":
        return ncdf((hi - mu) / sigma)
    if kind == "ge":
        return 1.0 - ncdf((lo - mu) / sigma)
    return ncdf((hi - mu) / sigma) - ncdf((lo - mu) / sigma)


def dist_deg(kind, k, mu, city=""):
    """Sicherheitsabstand in Grad zwischen Verlust-Grenze des Buckets und mu."""
    lo, hi = bucket_grenzen(k, city)
    if kind == "le":
        return mu - hi
    if kind == "ge":
        return lo - mu
    # eq: Abstand zur naeheren Bucketgrenze waere zu streng — gemeint ist der
    # Abstand zur Bucket-MITTE, die bei floor-Staedten auf k+0,5 liegt.
    return abs((lo + hi) / 2.0 - mu)


def median(vs):
    s = sorted(vs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def robust_mean(raw):
    """(bereinigtes Mittel, verworfene Modelle) — ein Modell gilt als Ausreisser,
    wenn es > OUTLIER_DEG vom Median ALLER Modelle abweicht.

    Median ueber alle (nicht Leave-one-out): bei 5 Modellen hat der Leave-one-out-
    Median 4 Werte, faellt also in die Luecke zwischen den mittleren beiden — bei
    breit gestreuten Staedten (Chengdu 14.07.: 32,9 / 34,5 / 37,2 / 38,4 / 38,6)
    flaggt das 4 von 5 Modellen als Ausreisser und das "bereinigte" Mittel ist ein
    einzelnes Modell. Der Median ueber alle kann nicht so degenerieren.

    Lehre 14.07.: Der Beijing-33-Lay wurde von einem JMA-Lauf getragen, der 3,6°C
    ueber den anderen vier lag und am Ende 4,1°C danebenlag. Das arithmetische
    Mittel hat keine Abwehr dagegen — es hat den Ausreisser sogar unsichtbar
    gemacht, sodass der Spannen-Filter, der am selben Tag Wuhan und Milan
    aussortierte, bei Beijing gar nicht erst ansprang.

    ACHTUNG: Die ENS-Kalibrierung (bias/sigma) ist am 5-Modell-Mittel gemessen.
    Auf das bereinigte Mittel angewandt ist sie eine Naeherung — deshalb wird sie
    NIE allein benutzt, sondern immer nur als zusaetzliche, pessimistischere
    Sicht neben dem vollen Mittel (siehe views() in main)."""
    med = median(list(raw.values()))
    drop = {m: v for m, v in raw.items() if abs(v - med) > OUTLIER_DEG}
    keep = {m: v for m, v in raw.items() if m not in drop}
    if not keep:  # kann bei Median-ueber-alle nicht passieren, aber sicher ist sicher
        return sum(raw.values()) / len(raw), {}
    return sum(keep.values()) / len(keep), drop


def load_calib(pattern, exclude=()):
    """city,model -> (bias, sigma, a, b) aus allen passenden CSVs (spaetere ueberschreiben).
    a/b beschreiben sigma(s) = a + b*Spanne und stehen nur in den ensemble_mean-Zeilen;
    fehlen sie (aeltere CSVs), faellt ens_sigma() auf das feste sigma zurueck.
    exclude: Pfad-Teilstrings, die uebersprungen werden — '_min_' sind die Tagestief-
    Kalibrierungen (gehoeren zu weather_outlier_screen_low.py), 'calib40' ist die
    Sommer-Familie, die die 700d-Basis nicht ueberschreiben darf."""
    calib = {}
    for path in sorted(glob.glob(pattern)):
        if any(x in path for x in exclude):
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                av, bv = (row.get("a") or "").strip(), (row.get("b") or "").strip()
                calib[(row["city"], row["model"])] = (
                    float(row["bias"]), float(row["sigma"]),
                    float(av) if av else None, float(bv) if bv else None)
    return calib


def model_sigma(cal, city, m, spread):
    """Sigma einer Quelle (Einzelmodell ODER ensemble_mean) — spannen-konditioniert
    (sigma(s) = a + b*Spanne), sobald die Kalibrier-CSV a/b fuer diese Zeile fuehrt.
    Seit 17.07. schreibt weather_source_compare.py a/b fuer ALLE Quellen (Backlog
    Prio 1): das feste Modell-Sigma war auf ruhigen Tagen zu breit — dieselbe
    Physik wie beim ENS (Pre-Reg weather_spread_sigma_2026_07_14.md). Aeltere
    CSVs ohne Modell-a/b fallen automatisch aufs feste Sigma zurueck."""
    _, sig_fix, a, b = cal[(city, m)]
    if a is None or b is None:
        return sig_fix
    return max(a + b * spread, SIGMA_FLOOR)


def build_views(city, raw, calib, calib40, spread, dropped):
    """Bis zu vier korrigierte Ensemble-Sichten: {voll, robust} x {700d, 40d}.

    DEBIAS-VOR-MITTELUNG (17.07., Backlog Prio 1): jedes Modell wird erst um
    seinen EIGENEN Kalibrier-Bias korrigiert, DANN wird gemittelt. Vorher wurde
    das rohe Mittel um den ENS-Bias korrigiert — das bricht, sobald die live
    verfuegbare Modellmenge nicht die Kalibrier-Menge ist:
      (a) Modell-Ausfall: UKMO fehlt an ~150/700 Kalibriertagen; der Screen
          rechnete ab 3 Modellen trotzdem mit der 5er-ENS-Korrektur — gemessene
          Verschiebung des Mittels bis +0,98° (Jeddah ohne JMA).
      (b) Ausreisser-Drop: die robuste Sicht bekam den vollen ENS-Bias, obwohl
          z. B. ein JMA mit Riesen-Bias gerade rausgeflogen war (der Code nannte
          es selbst 'Naeherung').
    Bei voller Modellmenge verschiebt der Umbau nur um eine kleine Konstante
    (der ENS-Bias ist auf der Alle-5-Schnittmenge gemessen, die Modell-Biases
    je Modell-Tagesmenge) — das Verhalten aendert sich dort kaum.

    Bewusst NICHT geaendert: die Drop-MENGE kommt weiterhin aus robust_mean auf
    ROHWERTEN (Ausreisser/Spanne signalisieren korrumpiertes mu — Schutz-Gate-
    Semantik unveraendert, Beijing-Lehre); Modelle OHNE Kalibrier-Zeile fliegen
    aus dem debiasten Mittel (statt unkorrigiert mitzumitteln); sigma bleibt das
    ENS-sigma(s) der Familie (auf der robusten Sicht eine Naeherung, wie zuvor).

    Rueckgabe: [(label, mu, sigma), ...] — leer, wenn keine Familie >=3
    kalibrierte Modelle hat."""
    views = []
    for cname, cal in (("700d", calib), ("40d", calib40)):
        if (city, "ensemble_mean") not in cal:
            continue
        deb = {m: raw[m] - cal[(city, m)][0] for m in raw if (city, m) in cal}
        if len(deb) < 3:
            continue
        s = ens_sigma(cal, city, spread)
        views.append((cname, sum(deb.values()) / len(deb), s))
        keep = [m for m in deb if m not in dropped]
        if dropped and keep:
            views.append((f"{cname}/rob", sum(deb[m] for m in keep) / len(keep), s))
    return views


def divergenz(calib, calib40, city):
    """|mu_700d - mu_40d| dieser Stadt in K — die Uneinigkeit der beiden Sichten.

    mu = roh_ens - bias, und roh_ens ist fuer beide Sichten dasselbe. Die
    Differenz der mu ist damit exakt die Differenz der ENS-Biases; die Forecasts
    muessen dafuer nicht angefasst werden.

    None, wenn eine der beiden Kalibrierungen fehlt — dann greift ohnehin schon
    das aeltere has40-Gate."""
    a = calib.get((city, "ensemble_mean"))
    b = calib40.get((city, "ensemble_mean"))
    if not a or not b:
        return None
    return abs(a[0] - b[0])


def ens_sigma(cal, city, spread):
    """Sigma des Ensembles — spannen-konditioniert, wenn die Kalibrierung a und b
    mitbringt, sonst das alte feste Sigma.

    WOFUER (Pre-Reg weather_spread_sigma_2026_07_14.md): Das feste Sigma ist ein
    Mittel ueber ALLE Spannen und deshalb auf RUHIGEN Tagen viel zu breit — im
    Sommer-OOS sagt es 3,3 % und liefert 1,4 % (Faktor 0,43). Der Screen rechnet
    dort also zu hohe P und laesst sichere Lays liegen. sigma(s) repariert genau das.

    WOFUER NICHT: um zerklueftete Tage handelbar zu machen. Die bleiben durch
    MAX_SPREAD gesperrt. G4 der Pre-Reg zeigte, dass sigma(s) ALLE vetoierten
    Buckets wieder geoeffnet haette — inklusive des Beijing-Verlierers. Die Spanne
    ist eben nicht nur ein Breiten-, sondern vor allem ein mu-Korruptions-Signal,
    und kein Sigma repariert einen verzogenen Mittelwert.

    b kommt aus der CSV, nicht aus einer Konstante: Die Steigung ist saisonabhaengig
    (Sommer 0,107 / Winter 0,177 / Ganzjahr 0,140), also muss die 40d-Sommer-Sicht
    ihre eigene benutzen duerfen."""
    return model_sigma(cal, city, "ensemble_mean", spread)


_AP_CACHE = None


def airports():
    """airportsdata einmal laden (der Aufruf kostet spuerbar)."""
    global _AP_CACHE
    if _AP_CACHE is None:
        _AP_CACHE = airportsdata.load("ICAO")
    return _AP_CACHE


def fetch_raw_models(city, target_day, var="max", session=None):
    """Rohe Tagesextrema der 5 Modelle fuer eine Stadt -> (raw|None, grund|None).

    Aus main() herausgeloest, damit ANDERE Verbraucher (weather_minus1_autobuy)
    dieselbe Abfrage benutzen koennen, statt sie zu kopieren. Die Kopier-Lehre
    steht im Modulkopf des Low-Screens: der Beijing-33-Verlust entstand an einer
    Mittelung, die in zwei Kopien lebte, und ein Fix in nur einer ist kein Fix.

    target_day: 'YYYY-MM-DD' in LOKALER Zeit der Station (timezone=auto).
    """
    # canonical_city auch hier: der Autobuy reicht den Stadtnamen aus der DB
    # herein, und Altzeilen aus der Zeit vor einer Umbenennung tragen noch den
    # Marktnamen. Ohne die Aufloesung faellt der Kandidat mit "keine Station" aus
    # der Spannen-Pruefung — stillschweigend, weil das auch der normale Weg fuer
    # stationslose Staedte ist.
    icao = STATIONS.get(canonical_city(city))
    st = station_info(icao)          # deckt auch Sonderstationen wie HKO ab
    if not st:
        return None, "keine Station"
    lat, lon = st["lat"], st["lon"]
    try:
        r = (session or S).get(OM, params={
            "latitude": lat, "longitude": lon, "hourly": "temperature_2m",
            "models": ",".join(MODELS), "timezone": "auto", "forecast_days": 5,
        }, timeout=30)
        r.raise_for_status()
        j = r.json()
    except Exception as ex:
        return None, f"Open-Meteo-Fehler {ex}"
    hh = j.get("hourly", {})
    times = hh.get("time", [])
    agg = max if var == "max" else min
    raw = {}
    for mdl in MODELS:
        vals = hh.get(f"temperature_2m_{mdl}", [])
        raw[mdl] = agg((v for t, v in zip(times, vals)
                        if v is not None and t.startswith(target_day)), default=None)
    have = [m for m in MODELS if raw[m] is not None]
    if len(have) < 3:
        return None, f"nur {len(have)} Modelle"
    return {m: raw[m] for m in have}, None


def model_spread(raw):
    """Rohe Modellspanne (max-min) in Grad — Basis des Spannen-Vetos."""
    return max(raw.values()) - min(raw.values())


def reject_reasons(r):
    """Warum ist dieser Bucket kein Kandidat? (leer = er ist einer)"""
    why = []
    if not r["has40"]:
        why.append("keine 40d-Kalibrierung -> Doppel-Check unmoeglich")
    # Einigkeits-Gate: nicht ob beide Sichten existieren, sondern ob sie dasselbe
    # sagen. .get() statt [] — aeltere Aufrufer ohne das Feld sollen nicht brechen.
    if (r.get("divergenz") or 0.0) > MAX_DIVERGENZ:
        why.append(f"KALIBRIERUNGEN UNEINIG {r['divergenz']:.2f}K>{MAX_DIVERGENZ} "
                   f"(700d vs 40d) -> Stadt nicht prognostizierbar")
    if r["spread"] > MAX_SPREAD:
        why.append(f"SPANNE {r['spread']:.1f}°>{MAX_SPREAD}")
    if r["dist"] < MIN_DIST:
        why.append(f"dist {r['dist']:.1f}°<{MIN_DIST}")
    if r["p_max"] > MAX_PMODEL:
        why.append(f"P_max {r['p_max']:.0%} ({r['p_max_src']})")
    if r["buyYes"] < MIN_YES:
        why.append(f"YES {r['buyYes']:.3f} unter Profitschwelle")
    return why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="Zieltag YYYY-MM-DD (default: morgen UTC)")
    ap.add_argument("--force-lead1", action="store_true",
                    help="bei Zieltag >24h trotzdem mit der Lead-24h-Kalibrierung rechnen "
                         "(nur zur Orientierung — nie darauf setzen, Madrid-Lehre 13.07.)")
    args = ap.parse_args()
    if args.date:
        target = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target = (datetime.now(timezone.utc) + timedelta(days=1)).date()
    target_day = target.isoformat()
    title_day = f"{MONTHS[target.month - 1]} {target.day}"
    title_re = re.compile(rf"^Highest temperature in (.+) on {re.escape(title_day)}\?")

    # Lead-Autowahl (17.07., Backlog Prio 1): bei Zieltag >24h laedt der Screen die
    # passende Lead-N-Kalibrierfamilie (weather_source_calib_leadN_*) statt nur zu
    # warnen — die 17.07.-Lays wurden am 15.07. noch mit Lead-24h-P gescreent.
    # Madrid-Lehre 13.07.: 3 von 4 formalen Screen-Passes waren nach Lead-2 -EV.
    lead_days = max(1, (target - datetime.now(timezone.utc).date()).days)
    use_lead = 1 if args.force_lead1 else lead_days
    if use_lead > 1:
        cg = CALIB_GLOB.replace("weather_source_calib_", f"weather_source_calib_lead{use_lead}_")
        cg40 = CALIB40_GLOB.replace("weather_source_calib40d_", f"weather_source_calib40d_lead{use_lead}_")
        # Lead-N-Dateien tragen "_lead" im Namen und sind hier explizit gewollt.
        calib = load_calib(cg, exclude=("_min_", "calib40"))
        calib40 = load_calib(cg40, exclude=("_min_",))
        if not calib or not calib40:
            sys.exit(f"Zieltag ist {lead_days} Tage weg, aber keine Lead-{use_lead}-Kalibrierung "
                     f"gefunden ({cg} / {cg40}). Erzeugen mit:\n"
                     f"  python weather_source_compare.py --days 700 --lead {use_lead} "
                     f"--calib-csv preregs/weather_source_calib_lead{use_lead}_YYYY_MM_DD.csv\n"
                     f"  python weather_source_compare.py --days 40 --lead {use_lead} "
                     f"--calib-csv preregs/weather_source_calib40d_lead{use_lead}_YYYY_MM_DD.csv\n"
                     f"(Notbehelf --force-lead1: rechnet mit Lead-24h — nur Orientierung, "
                     f"nie darauf setzen. Madrid-Lehre 13.07.)")
        print(f"Lead-{use_lead}-Kalibrierung geladen ({len({c for c, _ in calib})} Staedte 700d, "
              f"{len({c for c, _ in calib40})} Staedte 40d).")
    else:
        # "_lead"-Exclude: der Basis-Glob matcht auch weather_source_calib_leadN_* —
        # die Lead-N-Familien duerfen die Lead-24h-Basis nicht ueberschreiben.
        calib = load_calib(CALIB_GLOB, exclude=("_min_", "calib40", "_lead"))
        calib40 = load_calib(CALIB40_GLOB, exclude=("_min_", "_lead"))
        if not calib:
            sys.exit(f"Keine Kalibrierung unter {CALIB_GLOB} gefunden (im Repo-Root ausfuehren).")
        if not calib40:
            sys.exit(f"Keine 40d-Sommer-Kalibrierung unter {CALIB40_GLOB} gefunden. Erzeugen mit:\n"
                     f"  python weather_source_compare.py --days 40 "
                     f"--calib-csv preregs/weather_source_calib40d_YYYY_MM_DD.csv\n"
                     f"(Seit dem Beijing-33-Verlust vom 14.07. ist die Doppel-Kalibrierung Pflicht, "
                     f"keine Kopfsache mehr — siehe Modulkopf.)")
        if lead_days > 1:
            print(f"\n!! ACHTUNG (--force-lead1): Zieltag ist {lead_days} Tage weg, gerechnet wird "
                  f"mit der Lead-24h-Kalibrierung.")
            print("   Nur zur Orientierung — vor dem Setzen die Lead-Familie erzeugen "
                  "(Madrid-Lehre 13.07.: 3 von 4 formalen Screen-Passes waren nach Lead-2 -EV).\n")

    # ---------------- 1) Jupiter: Events des Zieltags ----------------
    print(f"Ziel: 'Highest temperature in ... on {title_day}?' ({target_day})")
    print("Lade Jupiter-Wetter-Events ...", flush=True)
    # Jupiter verteilt die Temperatur-Bretter auf ZWEI Kategorien (Fund 19.07.:
    # London/NYC/Hong Kong liefen unter "climate & science" und fehlten dem
    # Screen komplett) -> beide laden, Dedupe per eventId.
    events, seen_ev = [], set()
    for cat in ("weather", "climate & science"):
        for s in range(0, 120, 10):
            page = None
            for attempt in range(4):
                r = S.get(f"{API}/events", params={"category": cat, "start": s, "end": s + 10}, timeout=30)
                if r.status_code == 429 or r.status_code >= 500:
                    wait = 6 * (attempt + 1)
                    print(f"  {r.status_code} bei page {s} ({cat}), warte {wait}s ...", flush=True)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                page = r.json().get("data", [])
                break
            if page is None:
                print(f"  page {s} ({cat}): aufgegeben (429/5xx)")
                break
            for e in page:
                eid = e.get("eventId")
                if eid not in seen_ev:
                    seen_ev.add(eid)
                    events.append(e)
            if len(page) < 10:
                break
            time.sleep(1.5)
    print(f"  {len(events)} Events geladen (weather + climate & science).")

    targets = {}
    for e in events:
        t = e.get("metadata", {}).get("title", "")
        m = title_re.match(t)
        if not m:
            continue
        city = canonical_city(m.group(1))     # umbenannte Bretter, siehe CITY_ALIASES
        mks, celsius = [], True
        for mk in e.get("markets", []):
            ti = mk.get("title", "")
            if "°C" not in ti:
                celsius = False
                break
            num = re.search(r"(-?\d+)", ti)
            if not num:
                continue
            kind = "le" if "below" in ti else ("ge" if "higher" in ti else "eq")
            pr = mk.get("pricing") or {}
            mks.append({
                "kind": kind, "k": int(num.group(1)), "title": ti,
                "marketId": mk.get("marketId"), "status": mk.get("status"),
                "buyYes": (pr.get("buyYesPriceUsd") or 0) / 1e6,
                "buyNo": (pr.get("buyNoPriceUsd") or 0) / 1e6,
            })
        if celsius and mks:
            targets[city] = sorted(mks, key=lambda x: x["k"])
    print(f"  Staedte in °C: {sorted(targets)}")

    # ---------------- 2) Forecasts (5 Modelle) ----------------
    AP = airports()
    rows = []
    city_info = {}
    for city, mks in sorted(targets.items()):
        icao = STATIONS.get(city)
        # station_info statt "icao in AP": sonst faellt jede Sonderstation durch,
        # die nicht in airportsdata steht (HKO/Hong Kong) — genau der Grund, aus
        # dem Hong Kong hier bis zum 26.07. uebersprungen wurde, OBWOHL Eintrag
        # und Kalibrierung laengst vorlagen.
        if not station_info(icao):
            print(f"  {city}: keine Station -> skip (ggf. via Polymarket-Beschreibung aufloesen + kalibrieren)")
            continue
        if (city, "ensemble_mean") not in calib:
            print(f"  {city}: keine Kalibrierung -> skip (weather_source_compare.py --calib-csv nachziehen)")
            continue
        raw, grund = fetch_raw_models(city, target_day, "max")
        if raw is None:
            print(f"  {city}: {grund} -> skip")
            continue
        _, dropped = robust_mean(raw)   # Drop-Menge (auf Rohwerten, s. build_views)
        spread = model_spread(raw)
        has40 = (city, "ensemble_mean") in calib40

        # Bis zu vier Sichten: {volles Mittel, ausreisser-bereinigt} x {700d, 40d},
        # jedes Modell einzeln debiast VOR der Mittelung (17.07., s. build_views).
        # Gewertet wird IMMER die pessimistischste (hoechstes P, kleinster Abstand) —
        # so ist es egal, ob der Ausreisser warm oder kalt war.
        views = build_views(city, raw, calib, calib40, spread, dropped)
        if not views:
            print(f"  {city}: <3 kalibrierte Modelle in jeder Familie -> skip")
            continue

        open_mks = [x for x in mks if x["status"] == "open"]
        fav = max(open_mks, key=lambda x: x["buyYes"]) if open_mks else None
        city_info[city] = {"raw": raw, "views": views, "dropped": dropped, "spread": spread,
                           "fav": fav, "mks": mks, "icao": icao}

        for x in open_mks:
            # P je Einzelmodell ueber BEIDE Kalibrierungen -> die schlechteste zaehlt.
            # (Beijing 32° am 16.07.: ECMWF 9,2 % unter 700d, aber 15,8 % unter 40d.)
            # Seit 17.07. mit sigma(s) auch fuer Einzelmodelle (model_sigma) —
            # das feste Sigma war auf ruhigen Tagen zu breit (Backlog Prio 1).
            probs = {}
            for cal in (calib, calib40):
                for m in raw:
                    if (city, m) not in cal:
                        continue
                    b = cal[(city, m)][0]
                    s = model_sigma(cal, city, m, spread)
                    probs[m] = max(probs.get(m, 0.0),
                                   bucket_prob(x["kind"], x["k"], raw[m] - b, s, city))
            pmax_m = max(probs, key=probs.get) if probs else None

            pv = [(bucket_prob(x["kind"], x["k"], mu, s, city), lbl, s) for lbl, mu, s in views]
            p_use, p_src, sig_use = max(pv)
            d = min(dist_deg(x["kind"], x["k"], mu, city) for _, mu, _ in views)
            be = 1.0 - x["buyNo"]

            rows.append({
                "city": city, **x,
                "p_ens": p_use, "p_src": p_src,
                "p_max": probs[pmax_m] if pmax_m else 1.0,
                "p_max_src": SHORT[pmax_m] if pmax_m else "?",
                "dist": d, "dist_sig": d / sig_use if sig_use else 0.0,
                "spread": spread, "has40": has40, "be": be, "ev": be - p_use,
                "divergenz": divergenz(calib, calib40, city),
            })
        time.sleep(0.5)

    # ---------------- 3) Ranking ----------------
    print("\n" + "=" * 112)
    print(f"KANDIDATEN-FILTER (markt-blind bis auf die Profitschwelle): dist>={MIN_DIST}°C | "
          f"Modellspanne<={MAX_SPREAD}°C | jedes Modell P<={MAX_PMODEL:.0%} (700d UND 40d) | buyYes>={MIN_YES:.0%} | "
          f"700d/40d uneinig<={MAX_DIVERGENZ}K")
    print(f"P_pess = hoechstes P ueber alle Sichten (voll/bereinigt x 700d/40d); EV = BE - P_pess (nur Anzeige); "
          f"Ranking: P_pess aufsteigend (Wetterfrosch-Doktrin 16.07. — Abstand zur eigenen Prognose, Fav ignoriert)")
    print("=" * 112)
    cand = [r for r in rows if not reject_reasons(r) and 0 < r["buyNo"] < 1]
    # Wetterfrosch-Doktrin 16.07.: sicherste zuerst (P_pess aufsteigend), Tie-Break
    # groesster Abstand zur eigenen Prognose — beides markt-blind. Die alte
    # EV-Sortierung stellte genau die Zentrums-Naehe nach oben (BA-23er-Lehre).
    cand.sort(key=lambda r: (r["p_ens"], -r["dist_sig"]))

    hdr = (f"{'Stadt':13} {'Bucket':>15} {'YES':>5} {'NO':>6} {'Rend%':>6} {'BE':>5} "
           f"{'P_pess':>15} {'EV':>8} {'P_max':>11} {'Span':>6} {'dist':>6}  Markt-ID")
    print(hdr)
    print("-" * len(hdr))
    for r in cand[:18]:
        rend = (1 - r["buyNo"]) / r["buyNo"] * 100
        print(f"{r['city']:13} {r['title']:>15} {r['buyYes']:5.2f} {r['buyNo']:6.3f} {rend:6.1f} "
              f"{r['be']*100:4.1f}% {r['p_ens']*100:6.1f}%/{r['p_src']:<7} {r['ev']*100:+6.1f}pp "
              f"{r['p_max']*100:5.1f}%/{r['p_max_src']:5} {r['spread']:5.1f}° {r['dist']:5.1f}°  {r['marketId']}")
    if not cand:
        print("(keiner — das ist das haeufige Ergebnis, kein Fehler)")

    cand_ids = {r["marketId"] for r in cand}
    near = [r for r in rows if r["marketId"] not in cand_ids and r["buyYes"] >= 0.06
            and 0 < r["buyNo"] < 1 and r["dist"] >= 1.0 and r["ev"] > -0.15]
    near.sort(key=lambda r: r["ev"], reverse=True)
    if near:
        print("\nVERWORFEN (Info — genau hier stand am 13.07. der Beijing-33-Verlierer):")
        for r in near[:10]:
            rend = (1 - r["buyNo"]) / r["buyNo"] * 100
            # BE/P (Spatz-Vorlage 16.07.): wie viel Vielfache des Modell-Risikos der
            # Markt zahlt — reine Vorlage-Info fuer den Einzelfall-Entscheid des
            # Betreibers (15.07.: 25er 10x = Spatz, 24er 1,9x = grenzwertig).
            ratio = r["be"] / r["p_ens"] if r["p_ens"] > 1e-9 else float("inf")
            ratio_s = f"{ratio:4.1f}x" if ratio < 100 else ">99x"
            print(f"  {r['city']:13} {r['title']:>15} NO {r['buyNo']:.3f} ({rend:5.1f} %) "
                  f"EV {r['ev']*100:+6.1f}pp BE/P {ratio_s} -> {', '.join(reject_reasons(r))}")

    # ---------------- 4) Kompakt-Leitern ----------------
    for city in sorted(city_info):
        ci = city_info[city]
        raw_s = "  ".join(f"{SHORT[m]} {v:.1f}{'*' if m in ci['dropped'] else ''}"
                          for m, v in ci["raw"].items())
        flag = "  (* = Ausreisser, aus dem bereinigten Mittel entfernt)" if ci["dropped"] else ""
        veto = "  << SPANNEN-VETO" if ci["spread"] > MAX_SPREAD else ""
        print(f"\n--- {city} ({ci['icao']})  roh: {raw_s}  | Spanne {ci['spread']:.1f}°{veto}{flag} ---")
        print("    korr. Sichten: " + "   ".join(f"{lbl} {mu:.1f}±{s:.1f}" for lbl, mu, s in ci["views"]))
        for x in ci["mks"]:
            if x["buyYes"] < 0.02 and not (ci["fav"] and x["marketId"] == ci["fav"]["marketId"]):
                continue
            p_e = max(bucket_prob(x["kind"], x["k"], mu, s, city) for _, mu, s in ci["views"])
            mark = " <FAV" if ci["fav"] and x["marketId"] == ci["fav"]["marketId"] else ""
            inc = " *" if x["marketId"] in cand_ids else ""
            print(f"   {x['title']:>15}  YES {x['buyYes']:.2f}  NO {x['buyNo']:.3f}  "
                  f"P_pess {p_e*100:5.1f}%  {x['status']}{mark}{inc}")

    print(f"\n(Stand {datetime.now(timezone.utc).strftime('%d.%m. %H:%M UTC')}; P aus Normal-Annahme, "
          f"Doppel-Kalibrierung 700d + 40d, Lead-{use_lead * 24}h, Debias-vor-Mittelung, "
          f"sigma(s) fuer ENS + Einzelmodelle.)")


if __name__ == "__main__":
    main()
