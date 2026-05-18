#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_models.py
=================

Vergleichs-Framework: LDA vs. SVM vs. XGBoost fuer Aktien-Richtungsprognose.

Baseline-Skript fuer den Boersenbot (Intraday Momentum, IBKR via ib_insync,
Contabo VPS Ubuntu 24.04). Drei Klassifikatoren werden auf *identischen*
Features und *identischem* zeitbasiertem Split verglichen, damit sichtbar
wird, welches Modell fuer die Daten tatsaechlich Mehrwert bringt.

Bewusst simpel und lesbar gehalten -- das ist eine Baseline, kein
Production-Code. Eine Datei, klare Sektionen.

Requirements:
    pip install yfinance pandas numpy scikit-learn xgboost ta

Beispiele:
    python compare_models.py
    python compare_models.py --ticker AAPL --period 10y --interval 1d
    python compare_models.py --ticker SPY --horizon 1 --save-predictions

Spaeter selbst dranzubauen (Struktur ist darauf vorbereitet, hier NICHT
implementiert):
    * Datenquelle: load_data() gegen IBKR Historical Data austauschen
    * Echte Slippage-/Gebuehren-Modellierung im Backtest
    * Walk-Forward-Optimization mit Re-Training (Schleife um evaluate_model)
    * Position Sizing via Probability-Output (proba steht bereits bereit)
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Warnings von yfinance / sklearn bewusst unterdruecken -- Baseline-Lautstaerke.
warnings.filterwarnings("ignore")

import yfinance as yf  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

import ta  # noqa: E402

RANDOM_STATE = 42
N_SPLITS = 5  # aeussere TimeSeriesSplit-Folds

# Annualisierungsfaktor je Intervall (Perioden pro Jahr) fuer die Sharpe Ratio.
PERIODS_PER_YEAR: Dict[str, float] = {
    "1d": 252.0,
    "1wk": 52.0,
    "1mo": 12.0,
    "1h": 252.0 * 6.5,   # ~6.5 Handelsstunden/Tag
    "60m": 252.0 * 6.5,
    "30m": 252.0 * 13.0,
    "15m": 252.0 * 26.0,
    "5m": 252.0 * 78.0,
}


# ===========================================================================
# 1. Datenquelle
# ===========================================================================
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Laedt OHLCV-Daten via yfinance (auto_adjust=True).

    Diese Funktion ist die einzige Stelle, die spaeter auf IBKR Historical
    Data umgestellt werden muss -- der Rest des Skripts arbeitet nur noch
    mit dem zurueckgegebenen OHLCV-DataFrame.
    """
    print(f"[Daten]  Lade {ticker}  period={period}  interval={interval} ...")
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise SystemExit(f"Keine Daten fuer {ticker} erhalten -- Argumente pruefen.")

    # yfinance liefert je nach Version MultiIndex-Spalten -> flach ziehen.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    print(f"[Daten]  {len(df)} Zeilen  ({df.index[0].date()} - {df.index[-1].date()})")
    return df


# ===========================================================================
# 2. Feature Engineering
# ===========================================================================
def build_features(df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Baut technische Features aus OHLCV.

    Alle Features verwenden ausschliesslich Vergangenheits-/Gegenwartsinfo.
    Das Target schaut um ``horizon`` Perioden in die Zukunft -- der einzige
    erlaubte Blick nach vorne.

    Returns
    -------
    X : Feature-Matrix (nur Vergangenheit)
    y : Binaeres Target (1 = Close steigt nach ``horizon`` Perioden)
    fwd_ret : Forward-Return ueber ``horizon`` Perioden (fuer den Backtest)
    """
    out = pd.DataFrame(index=df.index)
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    # --- Returns ueber mehrere Lags -------------------------------------
    for lag in (1, 2, 3, 5, 10):
        out[f"ret_{lag}"] = close.pct_change(lag)
    out["log_ret_1"] = np.log(close).diff()

    # --- Momentum -------------------------------------------------------
    out["mom_10"] = close / close.shift(10) - 1.0
    out["mom_20"] = close / close.shift(20) - 1.0

    # --- RSI ------------------------------------------------------------
    out["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # --- MACD -----------------------------------------------------------
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    # --- ATR (absolut + relativ zum Kurs) -------------------------------
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    out["atr_14"] = atr
    out["atr_pct"] = atr / close

    # --- Gleitende Durchschnitte + Verhaeltnisse ------------------------
    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    out["close_sma10"] = close / sma10
    out["close_sma20"] = close / sma20
    out["close_sma50"] = close / sma50
    out["sma10_sma50"] = sma10 / sma50
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["ema12_ema26"] = ema12 / ema26

    # --- Volumen-Features ----------------------------------------------
    out["vol_chg_1"] = vol.pct_change(1)
    out["vol_ratio_20"] = vol / vol.rolling(20).mean()

    # --- Volatilitaets-Features ----------------------------------------
    out["volat_5"] = out["ret_1"].rolling(5).std()
    out["volat_10"] = out["ret_1"].rolling(10).std()
    out["volat_20"] = out["ret_1"].rolling(20).std()
    out["range_pct"] = (high - low) / close

    # --- Target + Forward-Return ---------------------------------------
    # Forward-Return ueber den Prognosehorizont; Backtest realisiert exakt
    # diesen Return, wenn die Prediction = 1 ist (sonst flat).
    fwd_ret = close.pct_change(horizon).shift(-horizon)
    y = (fwd_ret > 0).astype(int)
    y.name = "target"

    # NaNs sauber droppen (Indikator-Warmup + letzte horizon Zeilen).
    full = out.copy()
    full["__y"] = y
    full["__fwd"] = fwd_ret
    full.dropna(inplace=True)

    X = full.drop(columns=["__y", "__fwd"])
    y = full["__y"].astype(int)
    fwd = full["__fwd"]
    print(f"[Feat.]  {X.shape[1]} Features, {len(X)} verwendbare Zeilen "
          f"(Klassenverteilung: {y.mean():.3f} up)")
    return X, y, fwd


# ===========================================================================
# 3. Modelle
# ===========================================================================
def build_models() -> Dict[str, object]:
    """Erzeugt die drei zu vergleichenden Schaetzer.

    LDA und SVM laufen in einer Pipeline mit RobustScaler -- so wird der
    Scaler garantiert nur auf den Trainingsdaten gefittet (kein Leakage).
    XGBoost braucht keine Skalierung.
    """
    inner_cv = TimeSeriesSplit(n_splits=3)  # innere CV fuer das SVM-Grid

    lda = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", LinearDiscriminantAnalysis()),
    ])

    svm_grid = GridSearchCV(
        estimator=Pipeline([
            ("scaler", RobustScaler()),
            ("clf", SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        param_grid={
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale", 0.01, 0.1],
        },
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
    )

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return {"LDA": lda, "SVM": svm_grid, "XGBoost": xgb}


# ===========================================================================
# 4. Evaluation
# ===========================================================================
@dataclass
class FoldResult:
    """Kennzahlen eines einzelnen Test-Folds."""
    fold: int
    accuracy: float
    f1: float
    roc_auc: float
    hit_rate: float
    strat_return: float
    buy_hold: float
    sharpe: float


@dataclass
class ModelResult:
    """Gesammelte Ergebnisse eines Modells ueber alle Folds."""
    name: str
    folds: List[FoldResult] = field(default_factory=list)
    predictions: List[pd.DataFrame] = field(default_factory=list)

    def mean_row(self) -> Dict[str, float]:
        """Mittelwerte aller Fold-Kennzahlen als flaches Dict."""
        df = pd.DataFrame([f.__dict__ for f in self.folds]).drop(columns="fold")
        return {"Modell": self.name, **df.mean().to_dict()}


def _sharpe(returns: np.ndarray, ann_factor: float) -> float:
    """Annualisierte Sharpe Ratio ohne Risk-Free-Rate."""
    r = returns[~np.isnan(returns)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ann_factor))


def backtest(y_pred: np.ndarray, fwd_ret: np.ndarray, ann_factor: float) -> Tuple[float, float, float, float]:
    """Simple Backtest-Kennzahlen auf einem Test-Fold.

    Strategie: long (voll investiert) wenn Prediction == 1, sonst flat.
    Keine Slippage/Gebuehren -- das wandert spaeter in den echten Backtester.
    """
    strat = np.where(y_pred == 1, fwd_ret, 0.0)
    hit_rate = float(np.mean((y_pred == 1) & (fwd_ret > 0)) / max(np.mean(y_pred == 1), 1e-9))
    strat_cum = float(np.prod(1.0 + strat) - 1.0)
    bh_cum = float(np.prod(1.0 + fwd_ret) - 1.0)
    sharpe = _sharpe(strat, ann_factor)
    return hit_rate, strat_cum, bh_cum, sharpe


def evaluate_model(
    name: str,
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    fwd: pd.Series,
    ann_factor: float,
) -> ModelResult:
    """Walk-forward-Evaluation eines Modells ueber TimeSeriesSplit-Folds."""
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    result = ModelResult(name=name)

    print(f"\n[{name}] Starte {N_SPLITS}-Fold TimeSeriesSplit ...")
    for i, (tr_idx, te_idx) in enumerate(splitter.split(X), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        fwd_te = fwd.iloc[te_idx].to_numpy()

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        # Wahrscheinlichkeit fuer Klasse 1 -- spaeter Basis fuer Position Sizing.
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_te)[:, 1]
        else:  # pragma: no cover - alle drei Modelle koennen proba
            y_proba = y_pred.astype(float)

        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_te, y_proba)
        except ValueError:
            auc = float("nan")  # nur eine Klasse im Test-Fold
        hit, strat_c, bh_c, shp = backtest(y_pred, fwd_te, ann_factor)

        result.folds.append(FoldResult(i, acc, f1, auc, hit, strat_c, bh_c, shp))
        result.predictions.append(pd.DataFrame({
            "model": name,
            "fold": i,
            "y_true": y_te.values,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "fwd_ret": fwd_te,
        }, index=X_te.index))

        print(f"  Fold {i}: acc={acc:.3f}  f1={f1:.3f}  auc={auc:.3f}  "
              f"strat={strat_c:+.2%}  b&h={bh_c:+.2%}  sharpe={shp:.2f}")

    return result


# ===========================================================================
# 5. Reporting
# ===========================================================================
def print_comparison(results: List[ModelResult]) -> pd.DataFrame:
    """Baut die finale Vergleichstabelle (Mittel ueber alle Folds)."""
    table = pd.DataFrame([r.mean_row() for r in results])
    table = table.rename(columns={
        "accuracy": "Accuracy",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
        "hit_rate": "HitRate",
        "strat_return": "StratRet",
        "buy_hold": "BuyHold",
        "sharpe": "Sharpe",
    }).set_index("Modell")

    fmt = table.copy()
    for col in ("Accuracy", "F1", "ROC-AUC", "HitRate"):
        fmt[col] = fmt[col].map("{:.3f}".format)
    for col in ("StratRet", "BuyHold"):
        fmt[col] = fmt[col].map("{:+.2%}".format)
    fmt["Sharpe"] = fmt["Sharpe"].map("{:.2f}".format)

    print("\n" + "=" * 72)
    print("VERGLEICHSTABELLE  (Mittelwerte ueber alle Folds)")
    print("=" * 72)
    print(fmt.to_string())
    print("=" * 72)
    return table


def print_xgb_importance(model: XGBClassifier, feature_names: List[str], top: int = 15) -> None:
    """Top-N Feature Importances des (zuletzt gefitteten) XGBoost-Modells."""
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=False).head(top)
    print(f"\n[XGBoost] Top-{top} Feature Importances:")
    for rank, (feat, val) in enumerate(imp.items(), start=1):
        print(f"  {rank:2d}. {feat:<14s} {val:.4f}")


# ===========================================================================
# 6. CLI / Main
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vergleich LDA vs. SVM vs. XGBoost fuer Aktien-Richtungsprognose.",
    )
    p.add_argument("--ticker", default="SPY", help="Yahoo-Finance-Ticker (Default: SPY)")
    p.add_argument("--period", default="5y", help="Zeitraum, z.B. 5y, 10y, max (Default: 5y)")
    p.add_argument("--interval", default="1d", help="Intervall, z.B. 1d, 1h, 30m (Default: 1d)")
    p.add_argument("--horizon", type=int, default=1,
                   help="Prognosehorizont in Perioden: steigt Close[t+h] > Close[t]? (Default: 1)")
    p.add_argument("--save-predictions", action="store_true",
                   help="Predictions aller Folds/Modelle als CSV exportieren")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ann_factor = PERIODS_PER_YEAR.get(args.interval, 252.0) / max(args.horizon, 1)

    df = load_data(args.ticker, args.period, args.interval)
    X, y, fwd = build_features(df, horizon=args.horizon)

    models = build_models()
    results: List[ModelResult] = []
    for name, model in models.items():
        results.append(evaluate_model(name, model, X, y, fwd, ann_factor))

    print_comparison(results)

    # XGBoost ist nach evaluate_model auf dem letzten Fold gefittet -> Importance.
    print_xgb_importance(models["XGBoost"], list(X.columns), top=15)

    if args.save_predictions:
        all_preds = pd.concat(
            [pd.concat(r.predictions) for r in results]
        ).sort_index()
        fname = f"predictions_{args.ticker}_{args.interval}_h{args.horizon}.csv"
        all_preds.to_csv(fname)
        print(f"\n[Output] Predictions gespeichert -> {fname}  ({len(all_preds)} Zeilen)")

    print("\nFertig.")


if __name__ == "__main__":
    main()
