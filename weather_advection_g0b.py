#!/usr/bin/env python3
"""
weather_advection_g0b.py — G0'-Gate Quelle B, DICHTER PWS-Ring (Madrid).

Pre-Reg: preregs/weather_advection_quelleB_2026_07_23.md (Nachtrag PWS-Gradient).

Idee (2D-Version des ΔT-Insights): statt einer Luv-Station den echten raeumlichen
Gradienten des ANOMALIEfeldes rechnen. Jede PWS wird zeitlich entrendet (trailing
Rollmittel) -> Residuum r_i(t); das killt statische Waermeinsel + Sensor-Bias.
Ebene r(x,y)=a+b·x+c·y ueber den Ring fitten -> ∇r=(b,c) [°C/km]. Mit dem
LEMD-Wind (IEM-METAR, verlaesslich) V [km/h] die Advektions-Tendenz:
    A'(t) = −V·∇r   [°C/h]   (Bewegungsvektor = FROM+180)
Linearer Fit -> ∇r ortsunabhaengig, LEMD-Koordinate egal (Robustheit).

G0'-Test (voll kausal): sagt A'(t) die KUENFTIGE LEMD-Temperaturaenderung der
naechsten 60 min voraus, ZUSAETZLICH zum juengsten Eigen-Trend?
    y = T_L(t+60) − T_L(t)  ~  1 + [T_L(t)−T_L(t−60)] + A'(t)
β auf A' > 0 signifikant (Tag-Jackknife-Cluster-t) => Lead. Placebo: A' vs
VERGANGENE Aenderung muss schwaecher sein. Kein Lead => Quelle B endgueltig tot.
"""
import io
import json
import math
import os
import sys
import time
from datetime import timedelta

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
CACHE = os.environ.get("ADVECT_CACHE",
    r"C:\Users\defco\AppData\Local\Temp\claude\C--projekte-boersenbot\ee674f64-3c58-4f13-a113-de085ab35899\scratchpad\iem_cache")
os.makedirs(CACHE, exist_ok=True)

S = requests.Session()
S.headers["User-Agent"] = "boersenbot-advect/1.0"
WU_KEY = "e1f10a1e78da46f5b10a1e78da96f525"

CITY = "Madrid"
LEMD_ICAO = "LEMD"
LEMD_LAT, LEMD_LON = 40.4936, -3.5668
PWS = ["IGETAF33", "IMAJAD13", "IMADRI651", "IALCOB23", "IDAGAN15", "IALGET16", "IRIVAS74"]
HEAT_UTC = (7, 18)     # grobes Heizfenster (Madrid lokal UTC+2)
GRID = "30min"
LAG_MIN = 60           # Prognosehorizont
STEP = 2               # 60 min / 30-min-Gitter = 2 Schritte


# ---------- LEMD-Wind + Temp aus IEM ASOS (gecacht) ----------
def fetch_iem(station, d0, d1):
    fn = os.path.join(CACHE, f"{station}_{d0}_{d1}.csv")
    if os.path.exists(fn) and os.path.getsize(fn) > 40:
        return pd.read_csv(fn)
    y0, m0, dd0 = d0.split("-"); y1, m1, dd1 = d1.split("-")
    p = dict(station=station, data="tmpc,drct,sknt", tz="UTC", format="onlycomma",
             latlon="no", missing="M", trace="T",
             year1=y0, month1=m0, day1=dd0, year2=y1, month2=m1, day2=dd1)
    r = S.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params=p, timeout=120)
    r.raise_for_status()
    open(fn, "w").write(r.text)
    return pd.read_csv(io.StringIO(r.text))


def lemd_series(d0, d1):
    df = fetch_iem(LEMD_ICAO, d0, d1)
    df["valid"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    for c in ("tmpc", "drct", "sknt"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["valid"]).set_index("valid").sort_index()


# ---------- WU-PWS-History (gecacht) ----------
def fetch_pws(station, date_iso):
    ymd = date_iso.replace("-", "")
    fn = os.path.join(CACHE, f"pws_{station}_{ymd}.json")
    if os.path.exists(fn) and os.path.getsize(fn) > 40:
        try:
            return json.load(open(fn))
        except Exception:
            pass
    p = dict(stationId=station, format="json", units="m", date=ymd,
             apiKey=WU_KEY, numericPrecision="decimal")
    r = S.get("https://api.weather.com/v2/pws/history/all", params=p, timeout=30)
    if r.status_code != 200:
        json.dump({"observations": []}, open(fn, "w")); return {"observations": []}
    d = r.json(); json.dump(d, open(fn, "w")); time.sleep(0.25)
    return d


def pws_day_temp(station, date_iso):
    """UTC-indizierte tempAvg-Reihe + (lat,lon) fuer einen lokalen Tag."""
    d = fetch_pws(station, date_iso)
    obs = d.get("observations") or []
    if not obs:
        return None, None, None
    ts, tp = [], []
    lat = lon = None
    for o in obs:
        t = o.get("obsTimeUtc"); m = (o.get("metric") or {})
        tv = m.get("tempAvg")
        if t is None or tv is None:
            continue
        ts.append(t); tp.append(float(tv))
        if lat is None:
            lat, lon = o.get("lat"), o.get("lon")
    if not ts:
        return None, None, None
    s = pd.Series(tp, index=pd.to_datetime(ts, utc=True)).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s, lat, lon


def enu(lat, lon):
    x = (lon - LEMD_LON) * math.cos(math.radians(LEMD_LAT)) * 111.320
    y = (lat - LEMD_LAT) * 110.574
    return x, y


def to_grid(s, day0, day1):
    """auf 30-Min-UTC-Gitter im [day0,day1), nearest <=20min."""
    idx = pd.date_range(day0, day1, freq=GRID, tz="UTC")
    return s.reindex(idx, method="nearest", tolerance=pd.Timedelta("20min"))


def trailing_detrend(t, win=5):
    """Residuum = temp − TRAILING Rollmittel (kausal), win Punkte."""
    roll = t.rolling(win, min_periods=3).mean()
    return t - roll


def plane_grad(xs, ys, rs):
    """LS-Fit r=a+b·x+c·y -> (b,c). >=4 Punkte noetig."""
    m = np.isfinite(rs)
    if m.sum() < 4:
        return None
    A = np.column_stack([np.ones(m.sum()), np.array(xs)[m], np.array(ys)[m]])
    try:
        coef, *_ = np.linalg.lstsq(A, np.array(rs)[m], rcond=None)
    except Exception:
        return None
    return coef[1], coef[2]


def main():
    conn = pymssql.connect(**DB); cur = conn.cursor()
    cur.execute("SELECT DISTINCT CONVERT(date,ts_utc) FROM bb_WeatherLatency WHERE city=%s ORDER BY 1", (CITY,))
    dates = [str(r[0]) for r in cur.fetchall()]
    conn.close()
    d0, d1 = min(dates), max(dates)
    print(f"[{CITY}] {len(dates)} Markttage {d0}..{d1}; {len(PWS)} PWS")

    lemd = lemd_series(d0, d1)

    # PWS-Koordinaten einmal bestimmen (erste verfuegbare Obs)
    coords = {}
    for st in PWS:
        for d in dates:
            s, la, lo = pws_day_temp(st, d)
            if la is not None:
                coords[st] = enu(la, lo)
                print(f"  {st}: lat/lon {la:.3f}/{lo:.3f} -> ENU ({coords[st][0]:+.1f},{coords[st][1]:+.1f}) km "
                      f"| Peilung {(math.degrees(math.atan2(*enu(la,lo)))+360)%360:.0f}°")
                break
    xs = [coords[st][0] for st in PWS if st in coords]
    ys = [coords[st][1] for st in PWS if st in coords]
    ok_st = [st for st in PWS if st in coords]

    # Pro Tag Gitter-Reihen einmal bauen (Daten gecacht)
    days_grid = {}
    for d in dates:
        day0 = pd.Timestamp(d + " 00:00", tz="UTC"); day1 = day0 + timedelta(days=1)
        ld = lemd[(lemd.index >= day0 - pd.Timedelta("3h")) & (lemd.index < day1 + pd.Timedelta("2h"))]
        if ld.empty:
            continue
        TL = to_grid(ld["tmpc"], day0, day1)
        DR = to_grid(ld["drct"], day0, day1)
        SK = to_grid(ld["sknt"], day0, day1)
        res = {}
        for st in ok_st:
            s, _, _ = pws_day_temp(st, d)
            if s is None:
                continue
            res[st] = trailing_detrend(to_grid(s, day0, day1))
        if len(res) < 4:
            continue
        # Advektions-Tendenz A'(t) je Gitterpunkt (horizontunabhaengig)
        grid = TL.index
        Ap = pd.Series(index=grid, dtype=float)
        for i, t in enumerate(grid):
            drct, sknt = DR.iloc[i], SK.iloc[i]
            if pd.isna(drct) or pd.isna(sknt):
                continue
            rvals = [res[st].iloc[i] if st in res else np.nan for st in ok_st]
            g = plane_grad(xs, ys, rvals)
            if g is None:
                continue
            b, c = g
            spd = sknt * 1.852
            mv = math.radians((drct + 180) % 360)
            Ap.iloc[i] = -(spd * math.sin(mv) * b + spd * math.cos(mv) * c)
        days_grid[d] = (TL, Ap, grid)

    def run_horizon(step):
        rows = []
        for d, (TL, Ap, grid) in days_grid.items():
            for i, t in enumerate(grid):
                if not (HEAT_UTC[0] <= t.hour < HEAT_UTC[1]):
                    continue
                if i - step < 0 or i + step >= len(grid):
                    continue
                tl_now, tl_prev, tl_fut, ap = TL.iloc[i], TL.iloc[i - step], TL.iloc[i + step], Ap.iloc[i]
                if any(pd.isna(v) for v in (tl_now, tl_prev, tl_fut, ap)):
                    continue
                rows.append((d, tl_fut - tl_now, tl_now - tl_prev, ap))
        df = pd.DataFrame(rows, columns=["day", "y_fut", "x_recent", "Aprime"])
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < 30:
            return None
        X = np.column_stack([np.ones(len(df)), df["x_recent"].values, df["Aprime"].values])
        y = df["y_fut"].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        jb = []
        for dd in df["day"].unique():
            m = df["day"].values != dd
            if m.sum() < 10:
                continue
            bi, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
            jb.append(bi[2])
        jb = np.array(jb); nd = len(jb)
        se = math.sqrt((nd - 1) / nd * np.sum((jb - jb.mean()) ** 2)) if nd > 2 else float("nan")
        t = beta[2] / se if se == se and se else float("nan")
        return dict(step=step, n=len(df), nd=df["day"].nunique(), beta=beta[2], t=t,
                    r_fut=df["Aprime"].corr(df["y_fut"]),
                    r_past=df["Aprime"].corr(df["x_recent"]),
                    q05=df["Aprime"].quantile(.05), q95=df["Aprime"].quantile(.95),
                    mabs=df["Aprime"].abs().median())

    print(f"\n[{CITY}] Horizont-Sweep (30/60/90 min); vorregistriert = 60 min")
    print(f"  {'τ[min]':>6} {'n':>4} {'Tage':>4} {'β(A′)':>8} {'Jack-t':>7} {'r_fut':>7} {'r_past':>7}")
    any_lead = False
    for step in (1, 2, 3):
        r = run_horizon(step)
        if r is None:
            print(f"  {step*30:>6}  zu wenig"); continue
        lead = (r["beta"] > 0 and r["t"] > 2 and r["r_fut"] > abs(r["r_past"]))
        any_lead = any_lead or (step == 2 and lead)
        print(f"  {r['step']*30:>6} {r['n']:>4} {r['nd']:>4} {r['beta']:>+8.3f} {r['t']:>+7.2f} "
              f"{r['r_fut']:>+7.3f} {r['r_past']:>+7.3f}  {'LEAD' if lead else 'kein Lead'}")
        if step == 2:
            print(f"         A'-Spannweite [°C/h] {r['q05']:+.2f}..{r['q95']:+.2f}, Median |A'| {r['mabs']:.2f}")
    print(f"\n=== G0' Verdikt (60 min, vorregistriert): "
          f"{'LEAD -> weiter zu G1' if any_lead else 'KEIN LEAD -> Quelle B endgültig tot'} ===")


if __name__ == "__main__":
    sys.exit(main())
