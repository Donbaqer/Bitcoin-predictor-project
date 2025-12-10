# ================= Bitcoin Price Predictor GUI (Final Enhanced Version) =================
import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from tkinter import *
from tkinter import ttk, messagebox
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# ---------------- Config ----------------
TICKER = "BTC-USD"
DATA_PERIOD_YEARS = 2
LOOKBACK = 60
MODEL_FILE = "bitcoin_predictor_advanced.h5"

# ---------------- Helper functions ----------------
def download_data(ticker, years):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=365 * years)
    data = yf.download(ticker, start=start, end=end, interval="1d")
    data.dropna(inplace=True)
    return data


def prepare_sequences(series, lookback):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler, scaled_data


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# ---------------- Main GUI Class ----------------
class BitcoinApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bitcoin Price Prediction (LSTM)")
        self.root.geometry("720x520")
        self.root.config(bg="#0e1111")

        self.df = None
        self.model = None
        self.scaler = None

        # Title Label
        title = Label(root, text="Bitcoin Price Predictor", font=("Helvetica", 18, "bold"), fg="gold", bg="#0e1111")
        title.pack(pady=10)

        # Info frame
        info_frame = Frame(root, bg="#0e1111")
        info_frame.pack(pady=10)

        Label(info_frame, text="Current BTC price (USD):", font=("Helvetica", 12), fg="white", bg="#0e1111").grid(row=0, column=0, padx=5)
        self.current_price = Label(info_frame, text="—", font=("Helvetica", 12, "bold"), fg="lime", bg="#0e1111")
        self.current_price.grid(row=0, column=1, padx=5)

        Label(info_frame, text="Predicted next-day price:", font=("Helvetica", 12), fg="white", bg="#0e1111").grid(row=1, column=0, padx=5)
        self.predicted_price = Label(info_frame, text="—", font=("Helvetica", 12, "bold"), fg="cyan", bg="#0e1111")
        self.predicted_price.grid(row=1, column=1, padx=5)

        Label(info_frame, text="Current Time:", font=("Helvetica", 12), fg="white", bg="#0e1111").grid(row=2, column=0, padx=5)
        self.current_time = Label(info_frame, text="—", font=("Helvetica", 12, "bold"), fg="orange", bg="#0e1111")
        self.current_time.grid(row=2, column=1, padx=5)

        # Update time dynamically
        self.update_time()

        # Buttons
        btn_frame = Frame(root, bg="#0e1111")
        btn_frame.pack(pady=15)

        ttk.Button(btn_frame, text="📥 Load Live Price", command=self.refresh_data).grid(row=0, column=0, padx=8)
        ttk.Button(btn_frame, text="📈 Train Model", command=self.train_model).grid(row=0, column=1, padx=8)
        ttk.Button(btn_frame, text="🔮 Predict Next Day", command=self.predict_next_day).grid(row=0, column=2, padx=8)
        ttk.Button(btn_frame, text="📊 Plot (Actual vs Predicted)", command=self.plot_predictions).grid(row=0, column=3, padx=8)

        # Status bar
        self.status = Label(root, text="Ready", font=("Helvetica", 10), fg="white", bg="#1a1a1a", anchor="w")
        self.status.pack(fill="x", side="bottom")

    def update_time(self):
        """Update current date and time on GUI."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.current_time.config(text=now)
        self.root.after(1000, self.update_time)

    def set_status(self, text):
        self.status.config(text=text)
        self.root.update_idletasks()

    def refresh_data(self):
        try:
            self.set_status("Downloading latest Bitcoin data...")
            self.df = download_data(TICKER, DATA_PERIOD_YEARS)
            last_close = float(self.df["Close"].iloc[-1])
            self.current_price.config(text=f"${last_close:,.2f}")
            self.set_status("Live data updated successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh live data.\n{e}")

    def train_model(self):
        try:
            if self.df is None:
                self.df = download_data(TICKER, DATA_PERIOD_YEARS)
            self.set_status("Preparing and training model...")
            X_train, y_train, scaler, _ = prepare_sequences(self.df["Close"], LOOKBACK)

            model = build_model((X_train.shape[1], 1))
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
            model.save(MODEL_FILE)
            self.model = model
            self.scaler = scaler
            self.set_status("Model trained and saved successfully.")
            messagebox.showinfo("Success", "Model trained successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed.\n{e}")

    def predict_next_day(self):
        try:
            if self.df is None:
                self.df = download_data(TICKER, DATA_PERIOD_YEARS)
            if self.model is None:
                self.model = load_model(MODEL_FILE)

            series = self.df["Close"]
            _, _, scaler, scaled = prepare_sequences(series, LOOKBACK)
            last_window = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
            pred_scaled = self.model.predict(last_window)
            pred = scaler.inverse_transform(pred_scaled)
            next_price = pred[0]

            # 🔧 Fix for unsupported format (convert Series/array to float)
            if isinstance(next_price, (np.ndarray, list, pd.Series)):
                next_price = float(next_price[0])

            self.predicted_price.config(text=f"${next_price:,.2f}")
            self.set_status("Prediction complete.")
            messagebox.showinfo("Prediction", f"Predicted next-day BTC price: ${next_price:,.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{e}")
            self.set_status("Prediction failed.")

    def plot_predictions(self):
        try:
            if self.df is None:
                self.df = download_data(TICKER, DATA_PERIOD_YEARS)
            if self.model is None:
                self.model = load_model(MODEL_FILE)

            X, y, scaler, scaled = prepare_sequences(self.df["Close"], LOOKBACK)
            pred_scaled = self.model.predict(X)
            pred = scaler.inverse_transform(pred_scaled)

            plt.figure(figsize=(10, 5))
            plt.plot(self.df.index[-len(y):], y * (max(self.df["Close"]) - min(self.df["Close"])) + min(self.df["Close"]),
                     color="blue", label="Actual Price")
            plt.plot(self.df.index[-len(pred):], pred, color="orange", label="Predicted Price")
            plt.title("Bitcoin Price Prediction (Actual vs Predicted)")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Plotting failed:\n{e}")
            self.set_status("Plot failed.")


# ---------------- Run App ----------------
if __name__ == "__main__":
    root = Tk()
    app = BitcoinApp(root)
    root.mainloop()
