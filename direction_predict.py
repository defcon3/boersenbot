#!/usr/bin/env python3
"""
RICHTUNGS-VORHERSAGE (Backlog-Idee 4) — 6 Klassifikatoren, 3 Horizonte.

Voll nachvollziehbar:
  1. Datenquelle: Centron-DB dbdata, Tabelle bb_StockPrices (taeglich von Yahoo
     befuellte Tages-OHLCV, kompletter S&P 500). Symbol via Dropdown.
  2. EIN T-SQL-SELECT (geschichtete CTEs, Window-Funktionen) erzeugt die komplette
     Feature-+-Label-Grundtabelle. Dieses SELECT wird auf der Web-Seite woertlich
     angezeigt -> der User kann es 1:1 selbst gegen Centron ausfuehren.
  3. Walk-Forward (expanding window, Refit alle 21 Handelstage, Scaler NUR auf
     Train) liefert ehrliche OOS-Metriken. Kein Lookahead.
  4. Live-Vorhersage: Modell auf voller (gelabelter) Historie -> P fuer den
     juengsten Tag. Konfidenz = |P-0.5|*2. Ensemble = Mehrheitsvotum.

Modelle: Logit, KNN, Decision Tree, Random Forest, SVM (RBF), LDA.
Features (17, alle kausal, nur Vergangenheit bis Tag t): siehe GRUND_SELECT.
Labels: y_h = 1 wenn Close(t+h) > Close(t), h in {5,10,15}.
"""
import re
import numpy as np
import pandas as pd
import pymssql

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

# Centron-DB (dieselbe Quelle wie die uebrigen /analysis-Seiten). Tabelle
# bb_StockPrices: taeglich von Yahoo befuellte Tages-OHLCV, 504 S&P-500-Symbole.
DB_CONFIG = {
    "server": "158.181.48.77",
    "database": "dbdata",
    "user": "326773",
    "password": "Extaler11!",
}
TABLE = "bb_StockPrices"
HORIZONS = [5, 10, 15]
REFIT_EVERY = 21
MIN_TRAIN = 252          # >=1 Jahr bevor die erste Vorhersage faellt

# ── Feature-Liste (Name -> Klartext fuer die natuerlichsprachliche Erklaerung)
FEATURES = [
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5",
    "roc5", "roc10", "roc20",
    "ma5_dist", "ma10_dist", "ma20_dist", "ma50_dist", "sma5_20",
    "rsi14", "stoch14", "vol10", "vol20", "range_rel",
]

# ── DAS Grund-SELECT (T-SQL gegen Centron bb_StockPrices) ────────────────────
# Erzeugt aus der Tages-OHLCV-Tabelle (ein Symbol) die Feature-+-Label-
# Grundtabelle. Geschichtete CTEs, jeder Rechenschritt sichtbar:
#   base : Tagesrendite + Roh-Lags (LAG)
#   win  : Fenster-Aggregate (AVG/MIN/MAX + STDEV nativ fuer die Volatilitaet)
#   feat : finale Features (Verhaeltnisse, RSI, Stochastik, Vola)
#   final: Features + Zukunfts-Labels (LEAD) -> Richtung in 5/10/15 Tagen
# {sym} wird vor Ausfuehrung durch das (validierte) Symbol ersetzt -> die auf
# der Seite angezeigte Query ist exakt die ausgefuehrte, 1:1 selbst pruefbar.
GRUND_SELECT = """WITH base AS (
  SELECT Date AS d, ClosePrice AS cl, HighPrice AS hi, LowPrice AS lo,
         ClosePrice / LAG(ClosePrice,1) OVER (ORDER BY Date) - 1.0 AS ret1,
         LAG(ClosePrice,1) OVER (ORDER BY Date) AS c1,
         LAG(ClosePrice,2) OVER (ORDER BY Date) AS c2,
         LAG(ClosePrice,3) OVER (ORDER BY Date) AS c3,
         LAG(ClosePrice,4) OVER (ORDER BY Date) AS c4,
         LAG(ClosePrice,5) OVER (ORDER BY Date) AS c5,
         LAG(ClosePrice,6) OVER (ORDER BY Date) AS c6,
         LAG(ClosePrice,10) OVER (ORDER BY Date) AS c10,
         LAG(ClosePrice,20) OVER (ORDER BY Date) AS c20
  FROM {table}
  WHERE Symbol = '{sym}'
),
win AS (
  SELECT *,
    AVG(cl) OVER (ORDER BY d ROWS BETWEEN 4  PRECEDING AND CURRENT ROW) AS sma5,
    AVG(cl) OVER (ORDER BY d ROWS BETWEEN 9  PRECEDING AND CURRENT ROW) AS sma10,
    AVG(cl) OVER (ORDER BY d ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sma20,
    AVG(cl) OVER (ORDER BY d ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS sma50,
    MIN(lo) OVER (ORDER BY d ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS ll14,
    MAX(hi) OVER (ORDER BY d ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS hh14,
    -- RSI(14), SMA-Variante: durchschnittl. Gewinn/Verlust ueber 14 Tage
    AVG(CASE WHEN ret1 > 0 THEN ret1 ELSE 0 END) OVER (ORDER BY d ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_gain,
    AVG(CASE WHEN ret1 < 0 THEN -ret1 ELSE 0 END) OVER (ORDER BY d ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_loss,
    -- Volatilitaet = Standardabweichung der Tagesrendite (STDEV nativ)
    STDEV(ret1) OVER (ORDER BY d ROWS BETWEEN 9  PRECEDING AND CURRENT ROW) AS vol10,
    STDEV(ret1) OVER (ORDER BY d ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS vol20,
    COUNT(*)    OVER (ORDER BY d ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS n_hist
  FROM base
),
feat AS (
  SELECT d, cl, hi, lo,
    c1/c2 - 1.0 AS ret_lag1,
    c2/c3 - 1.0 AS ret_lag2,
    c3/c4 - 1.0 AS ret_lag3,
    c5/c6 - 1.0 AS ret_lag5,
    cl/c5  - 1.0 AS roc5,
    cl/c10 - 1.0 AS roc10,
    cl/c20 - 1.0 AS roc20,
    cl/sma5  - 1.0 AS ma5_dist,
    cl/sma10 - 1.0 AS ma10_dist,
    cl/sma20 - 1.0 AS ma20_dist,
    cl/sma50 - 1.0 AS ma50_dist,
    sma5/sma20 - 1.0 AS sma5_20,
    100.0 - 100.0/(1.0 + avg_gain/NULLIF(avg_loss,0)) AS rsi14,
    (cl - ll14)/NULLIF(hh14 - ll14, 0)                AS stoch14,
    vol10, vol20,
    (hi - lo)/cl AS range_rel,
    n_hist
  FROM win
)
SELECT
  d, cl AS [close],
  ret_lag1, ret_lag2, ret_lag3, ret_lag5,
  roc5, roc10, roc20,
  ma5_dist, ma10_dist, ma20_dist, ma50_dist, sma5_20,
  rsi14, stoch14, vol10, vol20, range_rel,
  CASE WHEN LEAD(cl,5)  OVER (ORDER BY d) > cl THEN 1 ELSE 0 END AS y5,
  CASE WHEN LEAD(cl,10) OVER (ORDER BY d) > cl THEN 1 ELSE 0 END AS y10,
  CASE WHEN LEAD(cl,15) OVER (ORDER BY d) > cl THEN 1 ELSE 0 END AS y15
FROM feat
WHERE n_hist >= 50          -- erst wenn SMA50 & ROC20 voll definiert sind
ORDER BY d"""


def models():
    """Frische Modell-Instanzen (eine pro Walk-Forward-Fit)."""
    return {
        "Logistische Regression": LogisticRegression(max_iter=1000),
        "KNN":                    KNeighborsClassifier(n_neighbors=15),
        "Decision Tree":          DecisionTreeClassifier(max_depth=4, random_state=0),
        "Random Forest":          RandomForestClassifier(n_estimators=120, max_depth=6,
                                                         random_state=0, n_jobs=-1),
        "SVM (RBF)":              SVC(kernel="rbf", random_state=0),
        "LDA":                    LinearDiscriminantAnalysis(),
    }


# ── Daten laden (Centron) ────────────────────────────────────────────────────
def get_conn():
    return pymssql.connect(**DB_CONFIG)


def get_symbols():
    """Liste der waehlbaren S&P-500-Symbole (Symbol, CompanyName) fuer Dropdown."""
    try:
        conn = get_conn()
        df = pd.read_sql("SELECT Symbol, CompanyName FROM bb_Stocks ORDER BY Symbol", conn)
        conn.close()
        df["CompanyName"] = df["CompanyName"].fillna("")
        return list(df.itertuples(index=False, name=None))
    except Exception:
        return [("AAPL", "Apple"), ("MSFT", "Microsoft")]


def _safe_symbol(symbol):
    """Nur erlaubte Ticker-Zeichen — verhindert SQL-Injection beim Inlinen."""
    s = (symbol or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9.\-]{1,12}", s):
        raise ValueError(f"Ungueltiges Symbol: {symbol!r}")
    return s


def query_for(symbol):
    """Das ausfuehrbare/anzeigbare Grund-SELECT mit eingesetztem Symbol."""
    return GRUND_SELECT.format(table=TABLE, sym=_safe_symbol(symbol))


def build_dataset(symbol, conn=None):
    """Fuehrt das Grund-SELECT gegen Centron aus -> DataFrame (Features+Labels)."""
    own = conn is None
    if own:
        conn = get_conn()
    df = pd.read_sql(query_for(symbol), conn)
    if own:
        conn.close()
    if df.empty:
        raise ValueError(f"Keine Daten fuer Symbol '{symbol}' in {TABLE}")
    return df


# ── Walk-Forward ─────────────────────────────────────────────────────────────
def pos_score(m, X):
    """Score fuer Klasse 1: echte Wahrscheinlichkeit, oder sigmoid(decision_function)
    fuer Modelle ohne predict_proba (SVM). Liegt immer in [0,1], 0.5 = Grenze."""
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    return 1.0 / (1.0 + np.exp(-m.decision_function(X)))


def walk_forward(df, make_model, label, refit=REFIT_EVERY, min_train=MIN_TRAIN):
    """Expanding window, Refit alle `refit` Tage, Scaler nur auf Train.
    Gibt (y_true, y_pred, y_score) ueber alle OOS-Tage zurueck."""
    sub = df.dropna(subset=FEATURES + [label]).reset_index(drop=True)
    X = sub[FEATURES].values.astype(float)
    y = sub[label].values.astype(int)
    n = len(sub)
    yt, yp, ypr = [], [], []
    i = min_train
    while i < n:
        j = min(i + refit, n)
        if len(np.unique(y[:i])) < 2:        # Train braucht beide Klassen
            i = j; continue
        sc = StandardScaler().fit(X[:i])
        m = make_model().fit(sc.transform(X[:i]), y[:i])
        Xt = sc.transform(X[i:j])
        yt.extend(y[i:j])
        yp.extend(m.predict(Xt).astype(int))      # Klassen-Vorhersage (korrekt je Modell)
        ypr.extend(pos_score(m, Xt))              # Score fuer ROC-AUC
        i = j
    return np.array(yt), np.array(yp), np.array(ypr)


def metrics(yt, yp, ypr):
    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return dict(accuracy=np.nan, precision=np.nan, recall=np.nan,
                    f1=np.nan, roc_auc=np.nan, n=len(yt))
    return dict(
        accuracy=accuracy_score(yt, yp),
        precision=precision_score(yt, yp, zero_division=0),
        recall=recall_score(yt, yp, zero_division=0),
        f1=f1_score(yt, yp, zero_division=0),
        roc_auc=roc_auc_score(yt, ypr),
        n=int(len(yt)),
    )


def live_predict(df, make_model, label):
    """Fit auf voller gelabelter Historie -> P(steigt) fuer den juengsten Tag
    (letzte Zeile mit gueltigen Features, Label noch unbekannt)."""
    train = df.dropna(subset=FEATURES + [label])
    feat_rows = df.dropna(subset=FEATURES)
    if train.empty or feat_rows.empty or len(np.unique(train[label])) < 2:
        return None
    sc = StandardScaler().fit(train[FEATURES].values)
    m = make_model().fit(sc.transform(train[FEATURES].values),
                         train[label].values.astype(int))
    last = feat_rows.iloc[[-1]]
    x = sc.transform(last[FEATURES].values)
    p = float(pos_score(m, x)[0])
    return dict(date=str(last["d"].iloc[0]), proba=p,
                direction=int(p >= 0.5), confidence=abs(p - 0.5) * 2)


# ── Orchestrierung (eine komplette Analyse je Symbol) ────────────────────────
def analyze(symbol):
    symbol = _safe_symbol(symbol)
    conn = get_conn()
    df = build_dataset(symbol, conn)
    conn.close()
    out = {"symbol": symbol, "n_rows": len(df),
           "date_from": str(df["d"].iloc[0]), "date_to": str(df["d"].iloc[-1]),
           "last_close": float(df["close"].iloc[-1]), "models": {}}
    for name in models():
        mk = lambda nm=name: models()[nm]
        per_h = {}
        for h in HORIZONS:
            label = f"y{h}"
            yt, yp, ypr = walk_forward(df, mk, label)
            mt = metrics(yt, yp, ypr)
            lv = live_predict(df, mk, label)
            per_h[h] = {"metrics": mt, "live": lv,
                        "explain": explain_prediction(name, h, mt, lv)}
        out["models"][name] = per_h
    out["ensemble"] = ensemble_vote(out)
    out["ensemble_explain"] = {h: explain_ensemble(h, out["ensemble"][h])
                               for h in HORIZONS}
    out["select_sql"] = query_for(symbol)
    out["features"] = FEATURES
    out["horizons"] = HORIZONS
    return out


def explain_prediction(model_name, h, mt, lv):
    """Ein deutscher Satz je Modell/Horizont — ehrlich, mit Guete-Einordnung."""
    if lv is None:
        return f"{model_name}: keine Vorhersage moeglich (zu wenig Daten)."
    richtung = "STEIGEN" if lv["direction"] else "FALLEN"
    p = lv["proba"] * 100
    konf = lv["confidence"]
    konf_txt = ("sehr unsicher (fast Muenzwurf)" if konf < 0.15 else
                "eher unsicher" if konf < 0.35 else
                "mittlere Sicherheit" if konf < 0.6 else "hohe Sicherheit")
    auc = mt["roc_auc"]
    if np.isnan(auc):
        guete = "die historische Trennschaerfe ist nicht bestimmbar"
    elif auc < 0.53:
        guete = (f"im Walk-Forward-Test war die Trennschaerfe mit ROC-AUC "
                 f"{auc:.2f} aber kaum ueber Zufall (0,50) — also mit grosser "
                 f"Vorsicht zu lesen")
    elif auc < 0.58:
        guete = (f"die historische Trennschaerfe (ROC-AUC {auc:.2f}) ist "
                 f"schwach, aber leicht ueber Zufall")
    else:
        guete = (f"die historische Trennschaerfe (ROC-AUC {auc:.2f}) ist "
                 f"vergleichsweise ordentlich")
    return (f"{model_name} erwartet fuer die naechsten {h} Handelstage "
            f"{richtung} (Wahrscheinlichkeit {p:.0f} %, {konf_txt}). "
            f"Historisch traf das Modell in {mt['accuracy']*100:.0f} % der "
            f"OOS-Tage richtig; {guete}.")


def explain_ensemble(h, e):
    if e is None:
        return f"{h} Tage: kein Ensemble-Votum moeglich."
    richtung = "STEIGEN" if e["direction"] else "FALLEN"
    return (f"{h} Handelstage: {e['up_votes']} von {e['total']} Modellen "
            f"stimmen fuer STEIGEN, {e['down_votes']} fuer FALLEN — Mehrheit "
            f"sagt {richtung} (Konsens {e['consensus']*100:.0f} %, mittlere "
            f"Wahrscheinlichkeit {e['avg_proba']*100:.0f} %). Bei durchweg "
            f"schwacher Trennschaerfe der Einzelmodelle ist auch der Konsens "
            f"kein verlaesslicher Trade — er buendelt nur, was die Modelle sehen.")


def ensemble_vote(out):
    """Mehrheitsvotum + Durchschnitts-Wahrscheinlichkeit je Horizont."""
    ens = {}
    for h in HORIZONS:
        ups, probas = 0, []
        for name, per_h in out["models"].items():
            lv = per_h[h]["live"]
            if lv is None:
                continue
            ups += lv["direction"]
            probas.append(lv["proba"])
        total = len(probas)
        if total == 0:
            ens[h] = None; continue
        avg_p = float(np.mean(probas))
        ens[h] = dict(up_votes=ups, down_votes=total - ups, total=total,
                      avg_proba=avg_p, direction=int(ups > total / 2),
                      consensus=max(ups, total - ups) / total,
                      confidence=abs(avg_p - 0.5) * 2)
    return ens


if __name__ == "__main__":
    import sys, time
    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    t0 = time.time()
    res = analyze(sym)
    dt = time.time() - t0
    print(f"\n{sym}: {res['n_rows']} Zeilen {res['date_from']}..{res['date_to']} "
          f"| letzter Close {res['last_close']:.2f} | {dt:.1f}s")
    for name, per_h in res["models"].items():
        print(f"\n{name}")
        for h in HORIZONS:
            mt = per_h[h]["metrics"]; lv = per_h[h]["live"]
            d = "STEIGT" if lv and lv["direction"] else "FAELLT"
            print(f"  {h:2}T: Acc {mt['accuracy']:.3f} F1 {mt['f1']:.3f} "
                  f"AUC {mt['roc_auc']:.3f} (n={mt['n']}) -> "
                  f"{d} P={lv['proba']:.3f} Konf={lv['confidence']:.2f}")
    print("\nENSEMBLE:")
    for h in HORIZONS:
        e = res["ensemble"][h]
        d = "STEIGT" if e["direction"] else "FAELLT"
        print(f"  {h:2}T: {d} ({e['up_votes']}:{e['down_votes']}) "
              f"Konsens {e['consensus']*100:.0f}% avgP={e['avg_proba']:.3f}")
