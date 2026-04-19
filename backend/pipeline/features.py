import pandas as pd


def engineer_stocks(df: pd.DataFrame):
    df = df.copy()
    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_ratio"] = df["ma_10"] / df["ma_20"]
    df["vol_change"] = df["volume"].pct_change()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    features = ["return_1d", "return_5d", "ma_ratio", "vol_change", "rsi"]
    df = df.dropna()
    return df[features], df["target"], features


def engineer_sales(df: pd.DataFrame):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["rolling_7"] = df["sales"].shift(1).rolling(7).mean()

    features = [
        "store_id", "product_id", "price", "promo", "temperature",
        "day_of_week", "month", "day_of_year", "lag_1", "lag_7", "rolling_7",
    ]
    df = df.dropna()
    return df[features], df["sales"], features


def engineer_healthcare(df: pd.DataFrame):
    features = [c for c in df.columns if c != "target"]
    return df[features], df["target"], features
