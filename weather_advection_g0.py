#!/usr/bin/env python3
"""
weather_advection_g0.py — G0-Gate der Advektions-Prognose (Quelle B).

Pre-Reg: preregs/weather_advection_quelleB_2026_07_23.md

Frage (nur Physik, noch KEIN Markt): Fuehrt die entrendete Temperatur-Fluktuation
einer LUVseitigen Nachbar-Station die der Settlement-Station? Und zwar staerker als
eine LEEseitige Kontrolle? Kein Lead -> Quelle B physikalisch tot, STOPP.

Methode:
  - IEM ASOS 30-Min-METAR (tmpc,drct,sknt) fuer Settlement + Nachbarn <=90 km,
    ueber die geloggten Markttage (aus bb_WeatherLatency). Roh-CSV gecacht.
  - Pro Markttag: mittlere FROM-Windrichtung der Settlement-Station im Heizfenster
    (sknt-gewichtet). Nachbar = LUV, wenn Peilung(settlement->nachbar) ~ FROM-Ri.
    (±60°); LEE, wenn ~ FROM+180°.
  - Entrenden: 30-Min-Gitter, Residuum = temp - zentrierter 3h-Rollmittel.
  - Gepoolt ueber Tage je Stadt: Kreuzkorr r_settle(t) vs r_neighbor(t-τ),
    τ in {-90..+180} min. Positiver Lag = Nachbar fuehrt.
  - Jackknife ueber Tage -> t der Groesse (Luv-Lead-Korr − Lee-Lead-Korr).
"""
import io
import math
import os
import sys
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

try:
    import pymssql
except ImportError:
    pymssql = None

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

DB = dict(server="158.181.48.77", database="dbdata", user="326773", password="Extaler11!")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "AppData", "Local", "Temp", "claude")
CACHE = os.environ.get("ADVECT_CACHE",
                       r"C:\Users\defco\AppData\Local\Temp\claude\C--projekte-boersenbot\ee674f64-3c58-4f13-a113-de085ab35899\scratchpad\iem_cache")
os.makedirs(CACHE, exist_ok=True)

S = requests.Session()
S.headers["User-Agent"] = "boersenbot-advect/1.0"

# Settlement-Stationen (ICAO -> lat,lon,country) fuer das EU4-Startset.
CITIES = {
    "Munich": ("EDDM", 48.353, 11.786, "DE"),
    "London": ("EGLC", 51.505, 0.055, "GB"),
    "Paris":  ("LFPB", 48.969, 2.441, "FR"),
    "Madrid": ("LEMD", 40.472, -3.561, "ES"),
}
NEIGH_RADIUS_KM = 90
HEAT_UTC = {  # grobes Heizfenster in UTC-Stunden je Stadt (lokaler Nachmittag)
    "Munich": (8, 16), "London": (9, 17), "Paris": (8, 16), "Madrid": (9, 17),
}


def haversine(a, b, c, d):
    R = 6371.0
    p1, p2 = math.radians(a), math.radians(c)
    dphi = math.radians(c - a); dl = math.radians(d - b)
    x = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(x))


def bearing(lat1, lon1, lat2, lon2):
    """Peilung von (1)->(2) in Grad (0=N,90=E)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def angdiff(a, b):
    d = abs((a - b + 180) % 360 - 180)
    return d


def neighbors(cc, icao, lat, lon):
    g = S.get(f"https://mesonet.agron.iastate.edu/geojson/network/{cc}__ASOS.geojson",
              timeout=60).json()
    out = []
    for f in g["features"]:
        c = f["geometry"]["coordinates"]; sid = f["properties"]["sid"]
        if sid == icao:
            continue
        dkm = haversine(lat, lon, c[1], c[0])
        if dkm <= NEIGH_RADIUS_KM:
            out.append(dict(sid=sid, lat=c[1], lon=c[0], dist=dkm,
                            brg=bearing(lat, lon, c[1], c[0])))
    out.sort(key=lambda x: x["dist"])
    return out


def fetch_asos(station, d0, d1):
    """IEM ASOS 30-Min tmpc/drct/sknt, [d0,d1] inklusiv. Gecacht."""
    fn = os.path.join(CACHE, f"{station}_{d0}_{d1}.csv")
    if os.path.exists(fn) and os.path.getsize(fn) > 40:
        return pd.read_csv(fn)
    y0, m0, dd0 = d0.split("-"); y1, m1, dd1 = d1.split("-")
    p = dict(station=station, data="tmpc,drct,sknt", tz="UTC", format="onlycomma",
             latlon="no", missing="M", trace="T",
             year1=y0, month1=m0, day1=dd0, year2=y1, month2=m1, day2=dd1)
    r = S.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
              params=p, timeout=120)
    r.raise_for_status()
    with open(fn, "w") as fh:
        fh.write(r.text)
    return pd.read_csv(io.StringIO(r.text))


def load_series(station, d0, d1):
    df = fetch_asos(station, d0, d1)
    if df.empty or "valid" not in df.columns:
        return None
    df["valid"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    for c in ("tmpc", "drct", "sknt"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["valid"]).set_index("valid").sort_index()


def resample30(s):
    """auf 30-Min-Gitter (nearest innerhalb 20 min)."""
    if s is None or s["tmpc"].dropna().empty:
        return None
    idx = pd.date_range(s.index.min().floor("h"), s.index.max().ceil("h"), freq="30min", tz="UTC")
    return s["tmpc"].reindex(idx, method="nearest", tolerance=pd.Timedelta("20min"))


def detrend(t):
    """Residuum = temp − zentriertes 3h-Rollmittel (7 Punkte auf 30-min)."""
    if t is None:
        return None
    roll = t.rolling(7, center=True, min_periods=4).mean()
    return t - roll


def prevailing_from(s, h0, h1):
    """sknt-gewichtete mittlere FROM-Richtung im UTC-Fenster [h0,h1)."""
    w = s[(s.index.hour >= h0) & (s.index.hour < h1)]
    w = w.dropna(subset=["drct", "sknt"])
    w = w[w["sknt"] > 1]
    if len(w) < 3:
        return None, 0.0
    rad = np.radians(w["drct"].values)
    u = np.sum(w["sknt"].values * np.sin(rad)); v = np.sum(w["sknt"].values * np.cos(rad))
    frm = (math.degrees(math.atan2(u, v)) + 360) % 360
    spd = math.hypot(u, v) / max(w["sknt"].sum(), 1e-9)  # Richtungs-Konstanz 0..1
    return frm, spd


def xcorr_at_lags(rs, rn, lags_steps):
    """Korr r_settle(t) vs r_neighbor(t-τ). τ>0 = Nachbar fuehrt. Ein Lag-Step=30min."""
    out = {}
    for k in lags_steps:
        a = rs
        b = rn.shift(k)  # neighbor(t-k): positive k verschiebt Nachbar nach spaeter -> Nachbar fuehrt
        m = pd.concat([a, b], axis=1).dropna()
        if len(m) >= 5 and m.iloc[:, 0].std() > 1e-6 and m.iloc[:, 1].std() > 1e-6:
            out[k] = m.iloc[:, 0].corr(m.iloc[:, 1])
        else:
            out[k] = np.nan
    return out


def run_city(name):
    icao, lat, lon, cc = CITIES[name]
    h0, h1 = HEAT_UTC[name]
    conn = pymssql.connect(**DB); cur = conn.cursor()
    cur.execute("SELECT DISTINCT CONVERT(date,ts_utc) FROM bb_WeatherLatency WHERE city=%s ORDER BY 1",
                (name,))
    dates = [str(r[0]) for r in cur.fetchall()]
    conn.close()
    if not dates:
        print(f"[{name}] keine Markttage"); return None
    d0, d1 = min(dates), max(dates)
    nb = neighbors(cc, icao, lat, lon)
    print(f"[{name}] {icao}: {len(dates)} Tage {d0}..{d1}, {len(nb)} Nachbarn <=90km")

    sett = load_series(icao, d0, d1)
    nb_series = {}
    for n in nb:
        s = load_series(n["sid"], d0, d1)
        if s is not None and not s["tmpc"].dropna().empty:
            nb_series[n["sid"]] = (n, s)

    lead_lags = [1, 2, 3, 4]     # +30..+120 min = "Nachbar fuehrt"
    lag_lags = [-1, -2, -3]      # -30..-90 min = "Nachbar hinkt" (Kontrolle)
    # pro Tag: Mittel ueber Luv- bzw. Lee-Nachbarn, getrennt fuer Lead- und Lag-Fenster
    per_day = []
    for d in dates:
        day0 = pd.Timestamp(d + " 00:00", tz="UTC"); day1 = day0 + timedelta(days=1)
        ss = sett[(sett.index >= day0) & (sett.index < day1)] if sett is not None else None
        if ss is None or ss["tmpc"].dropna().shape[0] < 8:
            continue
        frm, konst = prevailing_from(ss, h0, h1)
        if frm is None or konst < 0.4:
            continue
        rs = detrend(resample30(ss.loc[(ss.index.hour >= h0-1) & (ss.index.hour < h1+1)]))
        if rs is None:
            continue
        luv_lead, luv_lag, lee_lead, lee_lag = [], [], [], []
        for sid, (meta, s_all) in nb_series.items():
            sd = s_all[(s_all.index >= day0) & (s_all.index < day1)]
            sd = sd[(sd.index.hour >= h0-1) & (sd.index.hour < h1+1)]
            rn = detrend(resample30(sd))
            if rn is None or rn.dropna().shape[0] < 6:
                continue
            xc = xcorr_at_lags(rs, rn, lead_lags + lag_lags)
            lead = np.nanmean([xc[k] for k in lead_lags])
            lag = np.nanmean([xc[k] for k in lag_lags])
            if np.isnan(lead) or np.isnan(lag):
                continue
            if angdiff(meta["brg"], frm) <= 60:            # Nachbar in FROM-Ri = luv
                luv_lead.append(lead); luv_lag.append(lag)
            elif angdiff(meta["brg"], (frm + 180) % 360) <= 60:
                lee_lead.append(lead); lee_lag.append(lag)
        per_day.append(dict(date=d, frm=frm,
                            luv_lead=np.nanmean(luv_lead) if luv_lead else np.nan,
                            luv_lag=np.nanmean(luv_lag) if luv_lag else np.nan,
                            lee_lead=np.nanmean(lee_lead) if lee_lead else np.nan))

    pdd = pd.DataFrame(per_day)
    v = pdd.dropna(subset=["luv_lead"])
    print(f"[{name}] auswertbare Tage (Luv vorhanden): {len(v)} / {len(pdd)}")
    if len(v) < 4:
        print(f"[{name}] zu wenig Luv-Tage -> unklar"); return None

    def tt(x):
        x = pd.Series(x).dropna()
        if len(x) < 3 or x.std(ddof=1) == 0:
            return np.nan, np.nan, len(x)
        return x.mean(), x.mean() / (x.std(ddof=1) / math.sqrt(len(x))), len(x)

    # Test 1 (richtungssauber, kein Lee noetig): fuehrt Luv? Lead > Lag, gepaart je Tag
    d1_m, d1_t, d1_n = tt(v["luv_lead"] - v["luv_lag"])
    # Test 2 (Luv vs Lee, ungepaart ueber Tagesmittel): Welch
    a = v["luv_lead"].dropna(); b = pdd["lee_lead"].dropna()
    if len(a) >= 3 and len(b) >= 3:
        sa, sb = a.std(ddof=1), b.std(ddof=1)
        se = math.sqrt(sa*sa/len(a) + sb*sb/len(b))
        d2_m = a.mean() - b.mean(); d2_t = d2_m / se if se > 0 else np.nan
    else:
        d2_m = d2_t = np.nan
    print(f"[{name}] Luv-Lead Ø={v['luv_lead'].mean():+.3f}  Luv-Lag Ø={v['luv_lag'].mean():+.3f}  "
          f"Lee-Lead Ø={pdd['lee_lead'].mean():+.3f}")
    print(f"[{name}] T1 Luv(Lead−Lag) Ø={d1_m:+.3f} t={d1_t:+.2f} (n={d1_n}) | "
          f"T2 (Luv−Lee)Lead Ø={d2_m:+.3f} t={d2_t:+.2f}")
    return dict(city=name, luv_lead=float(v["luv_lead"].mean()),
                lee_lead=float(pdd["lee_lead"].mean()),
                t1_m=float(d1_m), t1_t=float(d1_t), t2_m=float(d2_m), t2_t=float(d2_t), n=len(v))


def main():
    if pymssql is None:
        print("pymssql fehlt"); return 1
    only = sys.argv[1:] or list(CITIES)
    res = []
    for c in only:
        r = run_city(c)
        if isinstance(r, dict):
            res.append(r)
    if res:
        print("\n=== G0-Zusammenfassung (Luv muss fuehren: T1>0 & T2>0, beide t>2) ===")
        for r in res:
            ok = (r["t1_m"] > 0 and r["t1_t"] > 2) and (r["t2_m"] > 0 and r["t2_t"] > 2)
            print(f"  {r['city']:8} T1={r['t1_m']:+.3f}(t{r['t1_t']:+.2f}) "
                  f"T2={r['t2_m']:+.3f}(t{r['t2_t']:+.2f}) n={r['n']} -> "
                  f"{'LEAD' if ok else 'kein Lead'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
