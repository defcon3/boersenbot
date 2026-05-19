#!/usr/bin/env python3
"""
SOM-Marktregime — Kohonen-Karten auf der Daily-S&P-Historie (bb_StockPrices).

EHRLICHE LEITPLANKE
-------------------
Eine Self-Organizing Map ist explorativ/beschreibend, KEIN Prädiktor und KEIN
Edge. Sie clustert ähnliche Marktzustände räumlich. Wert entsteht erst durch
den nachgelagerten, OOS-geprüften Edge-Test (streak_edge_by_regime): läuft ein
bereits validierter Edge (Streak-Mean-Reversion, S&P-weit p<0.01) in
bestimmten Regime-Knoten systematisch anders? Nur das wäre ein Ergebnis — die
Karte selbst ist Explorationshilfe + Deko.

Zwei Karten (User-Vorgabe "beides nebeneinander"):
  * INDEX-Regime : ein Feature-Vektor je Handelstag, marktbreit aggregiert.
  * SYMBOL-TAG   : ein Feature-Vektor je (Symbol, Tag).

Train/OOS strikt chronologisch getrennt: Scaler + SOM nur auf Train gefittet.
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import pymssql
from minisom import MiniSom
from scipy.stats import binomtest

warnings.filterwarnings("ignore")

DB_CONFIG = {
    "server": "158.181.48.77",
    "database": "dbdata",
    "user": "326773",
    "password": "Extaler11!",
    "as_dict": False,
}

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "som_cache.pkl")

# Chronologischer Train-Anteil; Rest = strikt Out-of-Sample.
TRAIN_FRAC = 0.70
INDEX_GRID = 8          # 8x8 Index-Regime-Knoten
SYMDAY_GRID = 12        # 12x12 Symbol-Tag-Knoten
SYMDAY_TRAIN_SAMPLE = 60000   # minisom ist reines Python -> Train-Subsample
SOM_ITERS = 8000
STREAK_LEN = 6          # Streak-Mean-Reversion: 6 Abwärtstage -> nächster Tag?

INDEX_FEATURES = ["med_ret", "disp", "mkt_vol20", "breadth",
                  "med_rsi", "med_adx", "updown_vol"]
SYMDAY_FEATURES = ["ret", "vol20", "rsi14", "adx14", "dist_sma50", "vol_z"]


# ── Indikatoren (Wilder), je Symbol angewandt ───────────────────────────────

def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    rs = (up.ewm(alpha=1 / n, adjust=False).mean() /
          dn.ewm(alpha=1 / n, adjust=False).mean())
    return 100.0 - 100.0 / (1.0 + rs)


def _adx(high, low, close, n=14):
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def _per_symbol(g):
    g = g.sort_values("Date").copy()
    c, h, l, v = g["ClosePrice"], g["HighPrice"], g["LowPrice"], g["Volume"]
    g["ret"] = c.pct_change()
    g["vol20"] = g["ret"].rolling(20).std()
    g["rsi14"] = _rsi(c)
    g["adx14"] = _adx(h, l, c)
    sma50 = c.rolling(50).mean()
    g["above_sma50"] = (c > sma50).astype(float)
    g["dist_sma50"] = c / sma50 - 1.0
    vmu = v.rolling(60).mean()
    vsd = v.rolling(60).std()
    g["vol_z"] = (v - vmu) / vsd
    g["down_streak"] = 0
    neg = (g["ret"] < 0).astype(int).values
    streak = np.zeros(len(neg), dtype=int)
    run = 0
    for i, x in enumerate(neg):
        run = run + 1 if x else 0
        streak[i] = run
    g["down_streak"] = streak
    g["ret_fwd"] = c.pct_change().shift(-1)   # Outcome für Edge-Test
    return g


# ── Datenpipeline ───────────────────────────────────────────────────────────

def load_panel():
    conn = pymssql.connect(**DB_CONFIG)
    df = pd.read_sql(
        "SELECT Symbol, Date, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume "
        "FROM bb_StockPrices ORDER BY Symbol, Date", conn)
    conn.close()
    df["Date"] = pd.to_datetime(df["Date"])
    parts = [_per_symbol(g) for _, g in df.groupby("Symbol")]
    return pd.concat(parts, ignore_index=True)


def build_index_features(panel):
    """Ein marktbreit aggregierter Vektor je Handelstag."""
    g = panel.groupby("Date")
    idx = pd.DataFrame({
        "med_ret": g["ret"].median(),
        "disp": g["ret"].std(),
        "breadth": g["above_sma50"].mean(),
        "med_rsi": g["rsi14"].median(),
        "med_adx": g["adx14"].median(),
    })
    up_vol = panel[panel["ret"] > 0].groupby("Date")["Volume"].sum()
    dn_vol = panel[panel["ret"] < 0].groupby("Date")["Volume"].sum()
    idx["updown_vol"] = np.log((up_vol + 1) / (dn_vol + 1))
    idx["mkt_vol20"] = idx["med_ret"].rolling(20).std()
    idx = idx[INDEX_FEATURES].dropna()
    return idx


def build_symday_features(panel):
    s = panel.dropna(subset=SYMDAY_FEATURES).copy()
    return s


# ── SOM-Fit (Train/OOS strikt getrennt) ─────────────────────────────────────

def _zfit(X_train):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def _zapply(X, mu, sd):
    return (X - mu) / sd


def fit_som(X, grid, iters=SOM_ITERS, seed=42):
    som = MiniSom(grid, grid, X.shape[1], sigma=1.5,
                  learning_rate=0.5, neighborhood_function="gaussian",
                  random_seed=seed)
    som.pca_weights_init(X)
    som.train_random(X, iters)
    return som


def _bmu(som, X):
    return np.array([som.winner(x) for x in X])  # (n,2) -> (row,col)


def _node_id(coords, grid):
    return coords[:, 0] * grid + coords[:, 1]


def build_index_map(idx):
    n = len(idx)
    split = int(n * TRAIN_FRAC)
    train, oos = idx.iloc[:split], idx.iloc[split:]
    mu, sd = _zfit(train.values)
    Xtr = _zapply(train.values, mu, sd)
    Xall = _zapply(idx.values, mu, sd)
    som = fit_som(Xtr, INDEX_GRID)
    coords = _bmu(som, Xall)
    res = idx.copy()
    res["node"] = _node_id(coords, INDEX_GRID)
    res["row"] = coords[:, 0]
    res["col"] = coords[:, 1]
    res["split"] = np.where(np.arange(n) < split, "train", "oos")
    qe = float(som.quantization_error(Xtr))
    return {"som": som, "df": res, "mu": mu, "sd": sd,
            "split_date": str(idx.index[split].date()),
            "grid": INDEX_GRID, "qe": qe,
            "trajectory": res[["row", "col"]].tail(60).values.tolist()}


def build_symday_map(sym):
    sym = sym.sort_values("Date")
    dates = sym["Date"].sort_values().unique()
    split_date = dates[int(len(dates) * TRAIN_FRAC)]
    train = sym[sym["Date"] < split_date]
    mu, sd = _zfit(train[SYMDAY_FEATURES].values)
    rng = np.random.default_rng(42)
    tr_vals = train[SYMDAY_FEATURES].values
    if len(tr_vals) > SYMDAY_TRAIN_SAMPLE:
        sel = rng.choice(len(tr_vals), SYMDAY_TRAIN_SAMPLE, replace=False)
        tr_vals = tr_vals[sel]
    Xtr = _zapply(tr_vals, mu, sd)
    som = fit_som(Xtr, SYMDAY_GRID)
    Xall = _zapply(sym[SYMDAY_FEATURES].values, mu, sd)
    coords = _bmu(som, Xall)
    res = sym[["Symbol", "Date"] + SYMDAY_FEATURES +
              ["down_streak", "ret_fwd"]].copy()
    res["node"] = _node_id(coords, SYMDAY_GRID)
    res["row"] = coords[:, 0]
    res["col"] = coords[:, 1]
    res["split"] = np.where(res["Date"].values < split_date, "train", "oos")
    qe = float(som.quantization_error(Xtr))
    return {"som": som, "df": res, "mu": mu, "sd": sd,
            "split_date": str(pd.Timestamp(split_date).date()),
            "grid": SYMDAY_GRID, "qe": qe}


# ── Edge-Test: Streak-Mean-Reversion bedingt auf Regime-Knoten (nur OOS) ─────

def streak_edge_by_regime(symday_map):
    """Bekannter Edge: nach STREAK_LEN Abwärtstagen — ist der Folgetag positiv?
    Bedingt auf den SOM-Knoten des Signaltags. NUR Out-of-Sample.
    """
    df = symday_map["df"]
    oos = df[(df["split"] == "oos") &
             (df["down_streak"] >= STREAK_LEN) &
             df["ret_fwd"].notna()].copy()
    if oos.empty:
        return {"base_rate": None, "base_n": 0, "nodes": []}
    oos["win"] = (oos["ret_fwd"] > 0).astype(int)
    base_rate = float(oos["win"].mean())
    base_n = int(len(oos))
    rows = []
    for node, grp in oos.groupby("node"):
        n = len(grp)
        if n < 20:
            continue
        wins = int(grp["win"].sum())
        hr = wins / n
        p = binomtest(wins, n, base_rate, alternative="two-sided").pvalue
        rows.append({"node": int(node),
                     "row": int(grp["row"].iloc[0]),
                     "col": int(grp["col"].iloc[0]),
                     "n": n, "hit_rate": round(hr, 4),
                     "edge_vs_base": round(hr - base_rate, 4),
                     "p_value": round(float(p), 5)})
    rows.sort(key=lambda r: r["p_value"])
    return {"base_rate": round(base_rate, 4), "base_n": base_n,
            "streak_len": STREAK_LEN, "nodes": rows}


# ── Knoten-Profile (für Chernoff-Faces) ─────────────────────────────────────

def node_profiles(som_map, features):
    df = som_map["df"]
    grid = som_map["grid"]
    prof = df.groupby(["row", "col"])[features].mean()
    counts = df.groupby(["row", "col"]).size()
    # Normiere je Feature auf [0,1] über belegte Knoten (für Face-Mapping)
    norm = (prof - prof.min()) / (prof.max() - prof.min()).replace(0, 1)
    out = []
    for (r, c), row in prof.iterrows():
        out.append({
            "row": int(r), "col": int(c),
            "node": int(r) * grid + int(c),
            "n": int(counts.loc[(r, c)]),
            "raw": {k: round(float(row[k]), 5) for k in features},
            "norm": {k: round(float(norm.loc[(r, c), k]), 4) for k in features},
        })
    return {"grid": grid, "features": features, "profiles": out}


# ── Chernoff-Faces (SVG, browsertauglich) ───────────────────────────────────
#
# Je SOM-Knoten ein Gesicht. Die normierten Knoten-Mittel der Features werden
# positionsweise auf Gesichtszüge gemappt — so wird "Stress" vs. "ruhiger
# Trend" sofort als Fratze erkennbar. Reine Explorationshilfe, kein Edge.
#
# Trait-Reihenfolge (Feature-Index -> Gesichtszug):
#   0 Mundkrümmung (0=mürrisch ... 1=Lächeln)
#   1 Gesichtsbreite (Volatilität: schmal ... aufgebläht)
#   2 Brauenneigung (Dispersion/Stress: entspannt ... zornig)
#   3 Gesichtsfarbe (rot=schwach ... grün=stark, z.B. Breadth)
#   4 Augengröße (RSI: klein ... groß/überkauft)
#   5 Nasenlänge (Trendstärke ADX: kurz ... lang)
#   6 Mundbreite (Volumen-Asymmetrie)

def chernoff_svg(norm_vals, size=120):
    """norm_vals: Liste normierter Werte [0,1] in INDEX_/SYMDAY_FEATURES-
    Reihenfolge. Liefert ein <svg>-Fragment (String)."""
    v = list(norm_vals) + [0.5] * 7
    cx, cy = size / 2, size / 2
    fw = 0.32 * size + 0.16 * size * v[1]          # Gesichtsbreite
    fh = 0.40 * size
    g = int(60 + 150 * v[3])                       # Farbkanal: rot->grün
    r = int(210 - 150 * v[3])
    fill = f"rgb({r},{g},90)"
    eye_r = 4 + 7 * v[4]                           # Augengröße
    eye_dx = fw * 0.42
    eye_y = cy - fh * 0.18
    brow = (v[2] - 0.5) * 16                       # Brauenneigung
    nose_len = 6 + 18 * v[5]                       # Nasenlänge
    mw = fw * (0.45 + 0.45 * v[6])                 # Mundbreite
    mouth_cy = cy + fh * 0.42
    curve = (v[0] - 0.5) * 2 * fh * 0.45           # Mundkrümmung
    p = []
    p.append(f'<svg viewBox="0 0 {size} {size}" width="{size}" '
             f'height="{size}" xmlns="http://www.w3.org/2000/svg">')
    p.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{fw}" ry="{fh}" '
             f'fill="{fill}" stroke="#222" stroke-width="1.5"/>')
    for s in (-1, 1):
        ex = cx + s * eye_dx
        p.append(f'<ellipse cx="{ex}" cy="{eye_y}" rx="{eye_r}" '
                 f'ry="{eye_r}" fill="#fff" stroke="#222"/>')
        p.append(f'<circle cx="{ex}" cy="{eye_y}" r="{max(1.5, eye_r*0.4)}" '
                 f'fill="#111"/>')
        by = eye_y - eye_r - 4
        p.append(f'<line x1="{ex - eye_r-2}" y1="{by + s*0 + brow}" '
                 f'x2="{ex + eye_r+2}" y2="{by - brow}" '
                 f'stroke="#111" stroke-width="2.5"/>')
    p.append(f'<line x1="{cx}" y1="{cy - 2}" x2="{cx}" '
             f'y2="{cy + nose_len}" stroke="#222" stroke-width="2"/>')
    p.append(f'<path d="M {cx-mw/2} {mouth_cy} Q {cx} '
             f'{mouth_cy + curve} {cx+mw/2} {mouth_cy}" '
             f'fill="none" stroke="#111" stroke-width="3"/>')
    p.append('</svg>')
    return "".join(p)


def faces_grid(profiles, edge=None):
    """Baut für ein node_profiles()-Resultat eine Liste von Face-Kacheln
    (für das Template). edge: optional streak_edge_by_regime()-Knotenliste."""
    feats = profiles["features"]
    edge_map = {}
    if edge:
        for r in edge.get("nodes", []):
            edge_map[r["node"]] = r
    tiles = []
    for pr in profiles["profiles"]:
        nv = [pr["norm"][k] for k in feats]
        tiles.append({
            "row": pr["row"], "col": pr["col"], "node": pr["node"],
            "n": pr["n"], "svg": chernoff_svg(nv),
            "raw": pr["raw"],
            "edge": edge_map.get(pr["node"]),
        })
    return {"grid": profiles["grid"], "features": feats, "tiles": tiles}


# ── Build + Cache ───────────────────────────────────────────────────────────

def build_all(force=True):
    if not force and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    panel = load_panel()
    idx = build_index_features(panel)
    sym = build_symday_features(panel)
    imap = build_index_map(idx)
    smap = build_symday_map(sym)
    iprof = node_profiles(imap, INDEX_FEATURES)
    sprof = node_profiles(smap, SYMDAY_FEATURES)
    edge = streak_edge_by_regime(smap)
    # Schlanker Cache: NUR Anzeige-Daten. Die SOM-Objekte und die 767k-
    # Zeilen-DataFrames bleiben im Build-Prozess und werden NICHT gepickelt
    # (sonst 87 MB pro /som-Request zu entpickeln).
    payload = {
        "index_map": {"grid": imap["grid"], "qe": imap["qe"],
                      "split_date": imap["split_date"]},
        "symday_map": {"grid": smap["grid"], "qe": smap["qe"],
                       "split_date": smap["split_date"]},
        "index_profiles": iprof,
        "symday_profiles": sprof,
        "edge": edge,
        "index_faces": faces_grid(iprof),
        "symday_faces": faces_grid(sprof, edge),
        "trajectory": imap["trajectory"],
        "split_date": imap["split_date"],
        "meta": {
            "symbols": int(panel["Symbol"].nunique()),
            "rows": int(len(panel)),
            "date_min": str(panel["Date"].min().date()),
            "date_max": str(panel["Date"].max().date()),
            "index_days": int(len(idx)),
            "symday_rows": int(len(sym)),
        },
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(payload, f)
    return payload


if __name__ == "__main__":
    import time
    t0 = time.time()
    p = build_all(force=True)
    m = p["meta"]
    print(f"\n=== SOM-Marktregime — Build in {time.time()-t0:.1f}s ===")
    print(f"Panel: {m['rows']} Zeilen, {m['symbols']} Symbole, "
          f"{m['date_min']}..{m['date_max']}")
    print(f"Index-Regime: {m['index_days']} Tage, Grid "
          f"{p['index_map']['grid']}x{p['index_map']['grid']}, "
          f"Split @ {p['index_map']['split_date']}, "
          f"QE={p['index_map']['qe']:.3f}")
    print(f"Symbol-Tag : {m['symday_rows']} Zeilen, Grid "
          f"{p['symday_map']['grid']}x{p['symday_map']['grid']}, "
          f"Split @ {p['symday_map']['split_date']}, "
          f"QE={p['symday_map']['qe']:.3f}")
    e = p["edge"]
    print(f"\n--- Edge-Test (OOS): {STREAK_LEN} Abwärtstage -> Folgetag positiv? ---")
    print(f"Basisrate OOS = {e['base_rate']}  (n={e['base_n']} Signale)")
    print(f"{'Knoten':>6} {'r,c':>6} {'n':>5} {'Trefferq.':>9} "
          f"{'Edge':>7} {'p':>8}")
    for r in e["nodes"][:12]:
        print(f"{r['node']:>6} {r['row']},{r['col']:>3} {r['n']:>5} "
              f"{r['hit_rate']:>9} {r['edge_vs_base']:>+7} {r['p_value']:>8}")
    sig = [r for r in e["nodes"] if r["p_value"] < 0.05]
    print(f"\nKnoten mit p<0.05 (OOS, unkorrigiert): {len(sig)} von "
          f"{len(e['nodes'])} belegten. ACHTUNG Mehrfachtests — "
          f"deskriptiv, kein bestätigter Edge.")
