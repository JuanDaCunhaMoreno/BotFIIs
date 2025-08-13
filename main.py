import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, recall_score
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# ====== Fee configuration (CM Capital example) ======
BROKER_FEE = 0.0025   # 0.25%
B3_FEE = 0.0003       # 0.03%
TOTAL_FEE = BROKER_FEE + B3_FEE


def generate_indicators(df):
    """Generate technical indicators for the given DataFrame."""
    df = df.copy()
    df['rsi14'] = RSIIndicator(df['Close'].squeeze(), window=14).rsi()
    df['ma10'] = SMAIndicator(df['Close'].squeeze(), window=10).sma_indicator()
    df['ma30'] = SMAIndicator(df['Close'].squeeze(), window=30).sma_indicator()
    df['roc5'] = df['Close'].pct_change(periods=5)
    df.dropna(inplace=True)
    return df


def max_drawdown(series):
    """Calculate the maximum drawdown percentage."""
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min() * 100


def backtest_with_fees(df, threshold):
    """Run a backtest with gross and net equity calculation considering fees."""
    df = df.copy()
    df['signal'] = (df['returns'] > threshold).astype(int)

    # Gross equity (no fees)
    df['equity_gross'] = (1 + df['returns'] * df['signal']).cumprod()

    # Net returns: apply fees only when a trade occurs
    df['net_returns'] = np.where(
        df['signal'] == 1,
        df['returns'] - TOTAL_FEE,
        0
    )
    df['equity_net'] = (1 + df['net_returns']).cumprod()

    trades = df['signal'].sum()
    gross_return = (df['equity_gross'].iloc[-1] - 1) * 100
    net_return = (df['equity_net'].iloc[-1] - 1) * 100
    dd_max_net = max_drawdown(df['equity_net'])

    return gross_return, net_return, trades, dd_max_net


def evaluate_fii(fii):
    print(f"\n{'='*50}")
    print(f"Evaluating {fii}")
    print(f"{'='*50}")

    # Download 10 years of historical data
    df = yf.download(fii, period="10y", interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        print("❌ No data found.")
        return None

    df = generate_indicators(df)
    df['returns'] = df['Close'].pct_change().shift(-1)
    df.dropna(inplace=True)

    features = ['rsi14', 'ma10', 'ma30', 'roc5']
    results = []

    thresholds = [0.005, 0.01]
    for threshold in thresholds:
        df['target'] = (df['returns'] > threshold).astype(int)
        X = df[features]
        y = df['target']

        if len(X) < 200:
            print("⚠️ Not enough data for training.")
            continue

        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        recall_scorer = make_scorer(recall_score, pos_label=1)

        param_dist = {
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 150],
            'scale_pos_weight': [1, sum(y == 0) / sum(y == 1)]
        }

        tscv = TimeSeriesSplit(n_splits=3)
        rand_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=10,
            scoring=recall_scorer,
            cv=tscv,
            n_jobs=1,
            verbose=0,
            random_state=42
        )

        rand_search.fit(X, y)
        best_model = rand_search.best_estimator_

        # Backtest with fees
        gross_return, net_return, trades, dd_max_net = backtest_with_fees(df, threshold)

        # Store results
        results.append({
            "FII": fii,
            "Threshold": threshold,
            "Accuracy": best_model.score(X, y),
            "Recall": recall_score(y, best_model.predict(X)),
            "Gross Return (%)": gross_return,
            "Net Return (%)": net_return,
            "Trades": trades,
            "Max Drawdown Net (%)": dd_max_net,
            "Best Parameters": rand_search.best_params_
        })

        # Detailed print
        print(f"✅ Average Accuracy:  {best_model.score(X, y):.3f}")
        print(f"✅ Recall Class 1:    {recall_score(y, best_model.predict(X)):.3f}")
        print(f"💰 Gross Return:      {gross_return:.2f}%")
        print(f"💰 Net Return:        {net_return:.2f}%")
        print(f"📈 Trades Count:      {trades}")
        print(f"📉 Max Drawdown Net:  {dd_max_net:.2f}%")
        print(f"⚙️ Best Parameters:   {rand_search.best_params_}")
        print("-" * 60)

    return results


if __name__ == "__main__":
    fiis = [
        "HGLG11.SA", "KNRI11.SA", "VISC11.SA", "MXRF11.SA",
        "XPML11.SA", "GGRC11.SA", "JSRE11.SA", "ALZR11.SA", "IRDM11.SA"
    ]

    all_results = []
    for fii in fiis:
        res = evaluate_fii(fii)
        if res:
            all_results.extend(res)

    # Final consolidated report
    df_results = pd.DataFrame(all_results)
    df_results.sort_values(by="Net Return (%)", ascending=False, inplace=True)
    print("\n📊 FINAL REPORT 📊")
    print(df_results.to_string(index=False))
