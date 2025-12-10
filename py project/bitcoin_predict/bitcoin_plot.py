# bitcoin_advanced_fixed_v2.py
"""
Fixed & improved version of advanced Bitcoin LSTM predictor.
Usage:
    pip install yfinance pandas numpy matplotlib scikit-learn tensorflow
    python bitcoin_advanced_fixed_v2.py
"""

import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------- Parameters ----------------
TICKER = "BTC-USD"
START_DATE = "2018-01-01"
END_DATE = None  # None -> up to today
LOOKBACK = 90
TEST_RATIO = 0.2
EPOCHS = 60
BATCH_SIZE = 32
MODEL_FILE = "bitcoin_lstm_advanced.h5"
PREDICTIONS_CSV = "predictions.csv"
PLOT_FILE = "pred_vs_real.png"
RANDOM_SEED = 42
# --------------------------------------------

# reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def download_data(ticker=TICKER, start=START_DATE, end=END_DATE):
    print("Downloading data...")
    if end is None:
        df = yf.download(ticker, start=start, progress=False)
    else:
        df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        raise RuntimeError("No data downloaded — check network or ticker.")
    print(f"Data downloaded: {len(df)} rows")
    return df


def add_technical_indicators(df):
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    # RSI (14) -- simple version
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df = df.dropna()
    return df


def create_features_and_labels_with_dates(df, lookback=LOOKBACK):
    """
    Returns:
      X: array shape (samples, lookback, n_features)
      y: array shape (samples,)   -> scaled close values
      y_dates: corresponding dates for each y
      scaler: fitted MinMaxScaler (on full features)
      feature_names: list of features used
    """
    df = df.copy()
    feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD']
    # keep only features (will raise if columns missing)
    df = df[feature_names]
    data = df.values.astype(np.float32)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y, y_dates = [], [], []
    dates = df.index.to_numpy()
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i - lookback:i, :])          # (lookback, n_features)
        y.append(data_scaled[i, feature_names.index('Close')])  # scaled close
        y_dates.append(dates[i])
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    y_dates = pd.to_datetime(np.array(y_dates))
    return X, y, y_dates, scaler, feature_names


def train_test_split_timewise_with_dates(X, y, y_dates, test_ratio=TEST_RATIO):
    total = len(X)
    test_size = int(total * test_ratio)
    if test_size == 0:
        raise ValueError("Test size is zero — reduce LOOKBACK or TEST_RATIO.")
    train_X = X[:-test_size]
    train_y = y[:-test_size]
    train_dates = y_dates[:-test_size]
    test_X = X[-test_size:]
    test_y = y[-test_size:]
    test_dates = y_dates[-test_size:]
    return train_X, train_y, train_dates, test_X, test_y, test_dates


def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.25),
        LSTM(64, return_sequences=True),
        Dropout(0.2),LSTM(64, return_sequences=False),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def inverse_transform_predictions(preds, scaler, feature_index=3):
    """
    preds: (n_samples, 1) scaled predictions (0..1) for the Close column
    scaler: fitted MinMaxScaler on full feature set
    feature_index: index of Close in original features
    returns: (n_samples,) inverse-transformed Close prices (USD)
    """
    preds = np.array(preds).reshape(-1, 1)
    n_features = scaler.n_features_in_
    # build placeholder array where column feature_index is preds, others are zeros (scaled=0 -> min)
    full = np.zeros((len(preds), n_features), dtype=np.float32)
    full[:, feature_index] = preds[:, 0]
    inv = scaler.inverse_transform(full)
    return inv[:, feature_index]


def main():
    # 1) Download and prepare
    df = download_data()
    df = add_technical_indicators(df)
    X, y, y_dates, scaler, feature_names = create_features_and_labels_with_dates(df, LOOKBACK)
    train_X, train_y, train_dates, test_X, test_y, test_dates = train_test_split_timewise_with_dates(X, y, y_dates, TEST_RATIO)

    print(f"Shapes: X_train={train_X.shape}, X_test={test_X.shape}")

    # 2) Build model
    model = build_model((train_X.shape[1], train_X.shape[2]))
    model.summary()

    # 3) Callbacks
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    ckpt = ModelCheckpoint(MODEL_FILE, monitor='val_loss', save_best_only=True, verbose=1)

    # 4) Train
    history = model.fit(
        train_X, train_y,
        validation_data=(test_X, test_y),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es, ckpt],
        verbose=1
    )

    # 5) Predict on test (scaled)
    preds_scaled = model.predict(test_X)
    # convert scaled -> real USD
    preds = inverse_transform_predictions(preds_scaled, scaler, feature_index=feature_names.index('Close'))
    real = inverse_transform_predictions(test_y, scaler, feature_index=feature_names.index('Close'))

    # 6) Metrics
    mae = mean_absolute_error(real, preds)
    rmse = math.sqrt(mean_squared_error(real, preds))
    r2 = r2_score(real, preds)
    print(f"\nEvaluation on test set:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.4f}")

    # 7) Save predictions with dates
    out = pd.DataFrame({
        'Date': pd.to_datetime(test_dates),
        'Real_Close': real,
        'Predicted_Close': preds
    })
    out.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Predictions saved to {PREDICTIONS_CSV}")

    # 8) Plot and save
    plt.style.use('dark_background')          # يجعل الخلفية سوداء
    plt.figure(figsize=(12, 6), facecolor='black')  # يخلي القFigure نفسها سوداء
    plt.plot(test_dates, real, label='Real Close')
    plt.plot(test_dates, preds, label='Predicted Close')
    plt.title('Bitcoin Price: Real vs Predicted (test set)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.show()
    print(f"Plot saved to {PLOT_FILE}")

    # 9) Ensure model saved (ModelCheckpoint already saved best)
    if not os.path.exists(MODEL_FILE):
        model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    # 10) Predict next day using latest available window
    last_window = X[-1:]  # last scaled window
    next_scaled = model.predict(last_window)
    next_price = inverse_transform_predictions(next_scaled, scaler, feature_index=feature_names.index('Close'))[0]
    print(f"\nPredicted BTC price for next step: ${next_price:,.2f}")


if __name__ == "__main__":
    main()