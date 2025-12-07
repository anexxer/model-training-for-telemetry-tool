# train.py
import os, joblib
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import struct

# Import shared constants from generator if possible, or redefine. 
# For safety/standalone, we can redefine or import. Importing is cleaner.
from generator import read_bin, FIELD_NAMES

# paths
DATA_BIN = "data/telemetry.bin"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Reading telemetry from", DATA_BIN)
if not os.path.exists(DATA_BIN):
    raise SystemExit(f"No telemetry found at {DATA_BIN}; run 'python generate_data.py' first.")

df = read_bin(DATA_BIN)
if df.empty:
    raise SystemExit("Telemetry file is empty.")

# features for IsolationForest
# We use: Battery, Solar, Temp, CPU, and the 8 Extra noise channels
features = ["battery_v","solar_i","temp","cpu"] + [f"extra{i}" for i in range(8)]
X = df[features].fillna(0.0).values

# isolate normal subset (assume majority normal) - here take first 70% as "train"
n = int(0.7 * len(X))
X_train = X[:n]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)

print("Training IsolationForest...")
iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
iso.fit(X_train_s)

# Save scaler and iso model
joblib.dump(scaler, os.path.join(MODEL_DIR,"scaler.joblib"))
joblib.dump(iso, os.path.join(MODEL_DIR,"isoforest.joblib"))
print("Saved IsolationForest and scaler to", MODEL_DIR)

# Train Linear Regression for battery prediction (window method)
win = 5
batt = df["battery_v"].fillna(method="ffill").values
Xw, yw = [], []
for i in range(len(batt)-win-1):
    Xw.append(batt[i:i+win])
    yw.append(batt[i+win])
Xw = np.array(Xw); yw = np.array(yw)
try:
    split = int(0.7 * len(Xw))
    Xb_train, yb_train = Xw[:split], yw[:split]

    print("Training LinearRegression for battery...")
    lr = LinearRegression()
    lr.fit(Xb_train, yb_train)
    joblib.dump(lr, os.path.join(MODEL_DIR,"lr_battery.joblib"))
    print("Saved LR model to", MODEL_DIR)
except Exception as e:
    print(f"Skipping LR training due to error: {e}")

print("All models trained and saved to 'model/' directory.")
