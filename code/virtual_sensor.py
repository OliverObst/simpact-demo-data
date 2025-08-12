#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from sklearn.neural_network import MLPRegressor

# ------------------ config ------------------
# Data directory
DATADIR = Path("../data")

# Ensure ../figures exists 
FIGDIR = Path("../figures")
FIGDIR.mkdir(exist_ok=True)

# Input
PIVOT_CSV = DATADIR / Path("sopivot+idx.csv")     # day,hour,SENS0008,...
# Outputs
OUT_PNG   = FIGDIR / Path("backup_virtual_sensor.png")
OUT_PDF   = FIGDIR / Path("backup_virtual_sensor.pdf")

RNG       = 42
TARGET    = "SENS0021"            # the “faulty” sensor to virtualise
NEIGHBOUR = "SENS0012"            # informational neighbour
PRED_FRAC = 0.35                  # last 35% of timeline is the “failure”/prediction span
# --------------------------------------------

# Load & build a proper DateTime index from day + hour
df = pd.read_csv(PIVOT_CSV)
dt = pd.to_datetime(df["day"]) + pd.to_timedelta(df["hour"], unit="h")
df.index = dt
df = df.drop(columns=["day","hour"]).sort_index()

# basic insanity
for col in [TARGET, NEIGHBOUR]:
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in {PIVOT_CSV}")

# Split point: use last PRED_FRAC part as failure window
n = len(df)
split_idx = int(np.floor(n * (1.0 - PRED_FRAC)))
t_pred_start = df.index[split_idx]

# Train tiny MLP on the portion BEFORE failure
X_train = df[[NEIGHBOUR]].iloc[:split_idx].values
y_train = df[TARGET].iloc[:split_idx].values

mlp = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation="relu",
    solver="adam",
    random_state=RNG,
    max_iter=4000,
    early_stopping=True,
    n_iter_no_change=20,
    tol=1e-5,
)
mlp.fit(X_train, y_train)

# Predict across the whole series, but mask predictions before the failure time
X_all = df[[NEIGHBOUR]].values
yhat_all = mlp.predict(X_all)
yhat_masked = yhat_all.copy()
yhat_masked[:split_idx] = np.nan  # hide virtual sensor before hand-over

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(8.8, 3.4))
# Style + masks for pre/post handover
COL_BLUE  = "#1f77b4"   # actual TARGET
COL_GREEN = "#2ca02c"   # virtual sensor
COL_ORANGE= "#ff8c00"
COL_GREY  = "0.55"      
pre_mask  = df.index < t_pred_start
post_mask = ~pre_mask

# Actual TARGET: solid before handover, dashed after
ax.plot(df.index[pre_mask],  df[TARGET].values[pre_mask],  lw=1.8, color=COL_BLUE,  label=f"{TARGET} (actual)")
ax.plot(df.index[post_mask], df[TARGET].values[post_mask], lw=1.8, color=COL_GREY,  linestyle="--")

# Neighbour for context
ax.plot(df.index, df[NEIGHBOUR].values, lw=1.2, color=COL_ORANGE, alpha=0.85, label=f"{NEIGHBOUR} (actual)")

# Virtual (from handover)
ax.plot(df.index, yhat_masked, lw=2.0, color=COL_GREEN, label=f"Virtual {TARGET} (MLP)")

# Vertical hand-over marker
ax.axvline(t_pred_start, color="0.2", lw=0.8, linestyle=":", label="handover")

ax.set_ylabel("Soil moisture (VWC %)")
ax.set_xlabel("")  # keep clean

# Clean date ticks
locator = AutoDateLocator(minticks=4, maxticks=8)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))

ax.grid(True, which="major", axis="y", alpha=0.25)
ax.margins(x=0.01)
ax.legend(ncol=2, frameon=False, loc="upper left")

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300)
fig.savefig(OUT_PDF)
plt.close(fig)

print(f"Saved: {OUT_PNG}\n       {OUT_PDF}\nHandover at: {t_pred_start}")
