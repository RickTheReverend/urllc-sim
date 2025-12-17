import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

CSV_PATH = "/content/URLLC Larger Dataset.csv"

WINDOW = 64
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

EPOCHS = 80
BATCH = 256
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

UNDER_WEIGHT = 6.0
OVER_WEIGHT = 1.0

ROLL_N = 50
EWMA_SPAN = 50

USE_WINSORIZE = True
WINSOR_PCTS = (1, 99)

HUBER_DELTA = 1.0

def safety_metrics(y_true, y_pred, under_weight=6.0, over_weight=1.0):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    under = y_pred < y_true
    under_rate = float(np.mean(under))

    under_err = (y_true - y_pred)[under]
    over_err  = (y_pred - y_true)[~under]

    under_mae = float(np.mean(under_err)) if under_err.size else 0.0
    over_mae  = float(np.mean(over_err))  if over_err.size  else 0.0
    under_p95 = float(np.percentile(under_err, 95)) if under_err.size else 0.0

    harm = np.mean(under_weight * np.maximum(0.0, y_true - y_pred) +
                   over_weight  * np.maximum(0.0, y_pred - y_true))
    w = np.where(under, under_weight, over_weight)
    wmae = float(np.mean(w * np.abs(y_true - y_pred)))

    coverage = float(np.mean(y_true <= y_pred))

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "UnderpredictionRate": under_rate,
        "Under_MAE": under_mae,
        "Under_P95": under_p95,
        "Over_MAE": over_mae,
        "HarmScore": float(harm),
        "WeightedMAE": float(wmae),
        "Coverage (y <= yhat)": coverage,
    }

def build_sequences(df, feature_cols, target_col, window):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)
    Xs, ys = [], []
    for t in range(window, len(df)):
        Xs.append(X[t-window:t])
        ys.append(y[t])
    return np.stack(Xs), np.array(ys, dtype=np.float32)

df = pd.read_csv(CSV_PATH)
df = df.sort_values("Receiver Timestamp").reset_index(drop=True)

for c in ["inter_arrival_time (microSec)", "latency(microSec)", "jitter_(microSec)"]:
    if c in df.columns:
        df = df[df[c].notna()]
        df = df[df[c] >= 0]
df = df.reset_index(drop=True)

def add_roll_and_ewma(base_col, prefix):
    if base_col not in df.columns:
        return
    df[f"{prefix}_ma_{ROLL_N}"] = df[base_col].rolling(ROLL_N, min_periods=1).mean()
    df[f"{prefix}_std_{ROLL_N}"] = df[base_col].rolling(ROLL_N, min_periods=1).std().fillna(0.0)
    df[f"{prefix}_ewma_{EWMA_SPAN}"] = df[base_col].ewm(span=EWMA_SPAN, adjust=False).mean()

add_roll_and_ewma("Sender Payload(bytes)", "len")
add_roll_and_ewma("inter_arrival_time (microSec)", "iat")
add_roll_and_ewma("throughput_mbps_50ms", "thr")

df = df.fillna(0.0)

target_col = "Sender Payload(bytes)"

base_features = [
    "Sender Payload(bytes)",
    "Receiver payload(bytes)",
    "gap",
    "inter_arrival_time (microSec)",
    "latency(microSec)",
    "jitter_(microSec)",
    "throughput_mbps_50ms",
]

extra_features = [
    f"len_ma_{ROLL_N}", f"len_std_{ROLL_N}", f"len_ewma_{EWMA_SPAN}",
    f"iat_ma_{ROLL_N}", f"iat_std_{ROLL_N}", f"iat_ewma_{EWMA_SPAN}",
    f"thr_ma_{ROLL_N}", f"thr_std_{ROLL_N}", f"thr_ewma_{EWMA_SPAN}",
]

feature_cols = [c for c in (base_features + extra_features) if c in df.columns]

X, y_abs = build_sequences(df, feature_cols, target_col, WINDOW)

last_payload = X[:, -1, feature_cols.index("Sender Payload(bytes)")]
y_delta = y_abs - last_payload

N = len(y_delta)
n_train = int(N * TRAIN_FRAC)
n_val = int(N * (TRAIN_FRAC + VAL_FRAC))

X_train, y_train = X[:n_train], y_delta[:n_train]
X_val,   y_val   = X[n_train:n_val], y_delta[n_train:n_val]
X_test,  y_test  = X[n_val:], y_delta[n_val:]

last_test = last_payload[n_val:]

F = X.shape[-1]
x_scaler = StandardScaler()
x_scaler.fit(X_train.reshape(-1, F))

X_train = x_scaler.transform(X_train.reshape(-1, F)).reshape(X_train.shape)
X_val   = x_scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape)
X_test  = x_scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape)

y_scaler = StandardScaler()
y_train_s = y_scaler.fit_transform(y_train.reshape(-1,1)).reshape(-1)
y_val_s   = y_scaler.transform(y_val.reshape(-1,1)).reshape(-1)
y_test_s  = y_scaler.transform(y_test.reshape(-1,1)).reshape(-1)

if USE_WINSORIZE:
    lo, hi = np.percentile(y_train_s, WINSOR_PCTS)
    y_train_s = np.clip(y_train_s, lo, hi)

inp = layers.Input(shape=(WINDOW, F))

x = layers.Conv1D(96, 3, padding="causal", activation="relu")(inp)
x = layers.LayerNormalization()(x)
x = layers.Dropout(0.15)(x)

x = layers.Conv1D(96, 3, padding="causal", activation="relu")(x)
x = layers.LayerNormalization()(x)
x = layers.Dropout(0.15)(x)

x = layers.MaxPooling1D(pool_size=2)(x)

x = layers.LSTM(96)(x)
x = layers.Dropout(0.15)(x)

x = layers.Dense(96, activation="relu")(x)
out = layers.Dense(1)(x)

model = models.Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),
    loss=tf.keras.losses.Huber(delta=HUBER_DELTA)
)

cb = [
    callbacks.EarlyStopping(monitor="val_loss", patience=14, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", patience=7, factor=0.5, min_lr=1e-5),
]

t0 = time.time()
model.fit(
    X_train, y_train_s,
    validation_data=(X_val, y_val_s),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=cb,
    verbose=1
)
t1 = time.time()

pred_test_s = model.predict(X_test, batch_size=BATCH).reshape(-1)
pred_delta = y_scaler.inverse_transform(pred_test_s.reshape(-1,1)).reshape(-1)

pred_abs = np.clip(last_test + pred_delta, 0.0, None)
true_abs = last_test + y_test

m = safety_metrics(true_abs, pred_abs, UNDER_WEIGHT, OVER_WEIGHT)

print("\n=== Test Metrics (ABS payload) ===")
for k, v in m.items():
    print(f"{k:>22s}: {v:.6f}")

print(f"\nTrain sequences: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
print(f"Total training time (seconds): {t1 - t0:.3f}")
print("Features used:", feature_cols)

model.save("cnn_lstm_rmse_reduced.keras")
print("Saved model -> cnn_lstm_rmse_reduced.keras")
