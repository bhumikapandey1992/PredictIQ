import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.datasets import load_breast_cancer


def load_stocks(ticker: str = "AAPL", period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    df = df.reset_index()
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    return df


def load_sales() -> pd.DataFrame:
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    day_of_year = np.arange(n)
    promo = np.random.randint(0, 2, n)
    sales = (
        50
        + 20 * np.random.randn(n)
        + 15 * promo
        + np.sin(day_of_year * 2 * np.pi / 365) * 10
    ).clip(1).astype(int)

    return pd.DataFrame({
        "date": dates,
        "store_id": np.random.randint(1, 6, n),
        "product_id": np.random.randint(1, 11, n),
        "price": np.round(np.random.uniform(5, 100, n), 2),
        "promo": promo,
        "temperature": np.round(np.random.uniform(10, 35, n), 1),
        "sales": sales,
    })


def load_healthcare() -> pd.DataFrame:
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df
