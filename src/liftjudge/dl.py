"""Deep learning: a sequence model that reads a whole rep and predicts its faults.

The rules look at a few hand-picked numbers per rep (depth at the bottom, the
largest drop on the way up...). The network instead reads the full time series
of the rep, frame by frame, and learns the patterns itself:

    rep  ->  (N_STEPS frames x N_FEATURES signals)  ->  Conv1D -> GRU -> 5 fault probabilities

It is multi-label: one rep can have several faults at once (e.g. high AND a
bounce). "Good lift" = no fault above the threshold.

There is one model per lift, because the faults and the key signal differ:
  squat     depth, lockout, bounce, downward, feet      (key signal: hip depth)
  deadlift  lockout, downward, hitching, feet          (key signal: bar height)

Training plan (sim-to-real):
  1. pre-train on synthetic reps (liftjudge.synthetic), where labels are free;
  2. fine-tune on real labelled reps (data/labels.csv), freezing the early layers.

Keras 3 is imported lazily, so the rest of the package works without it.
It runs on TensorFlow, PyTorch or JAX - whichever backend you have installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .reps import Rep, segment_reps

FAULTS = ("depth", "lockout", "bounce", "downward", "feet")          # squat (kept for compatibility)
LIFT_FAULTS = {
    "squat": FAULTS,
    "deadlift": ("lockout", "downward", "hitching", "feet"),
}
FAULT_TEXT = {
    "depth": "not deep enough",
    "lockout": "knees not locked at start/finish",
    "bounce": "double bounce in the hole",
    "downward": "downward movement on the way up",
    "feet": "feet moved",
    "hitching": "hitching (bar supported on the thighs)",
}
DEADLIFT_TEXT = {"lockout": "not locked out (knees or hips)", "downward": "bar went down before lockout"}
CHANNELS = ("depth", "hip_height", "hip_vel", "knee_angle", "hip_angle",
            "ankle_dx", "ankle_dy", "shoulder_dx", "leg_conf", "duration")
# deadlift: the bar (wrists) leads; hip depth means nothing in a pull
LIFT_CHANNELS = {
    "squat": CHANNELS,
    "deadlift": ("bar_height", "hip_height", "bar_vel", "knee_angle", "hip_angle",
                 "ankle_dx", "ankle_dy", "bar_dx", "leg_conf", "duration"),
}
DEFAULT_MODEL_PATHS = {"squat": "models/rep_fault_net.keras", "deadlift": "models/rep_fault_net_deadlift.keras"}


def fault_text(lift: str, fault: str) -> str:
    if lift == "deadlift" and fault in DEADLIFT_TEXT:
        return DEADLIFT_TEXT[fault]
    return FAULT_TEXT[fault]
N_STEPS = 96
N_FEATURES = len(CHANNELS)


# --------------------------------------------------------------------------
# rep -> fixed-length sequence
# --------------------------------------------------------------------------
def rep_sequence(sig: pd.DataFrame, rep: Rep, n_steps: int = N_STEPS, pad_s: float = 0.4,
                 lift: str = "squat") -> np.ndarray:
    """One rep as an (n_steps, N_FEATURES) array.

    The window runs from a little before the descent to a little after the
    lift is finished (so lockout is visible) and is resampled to n_steps,
    which makes slow and fast reps comparable. The real duration is kept as a
    channel. All signals are already in thigh lengths / degrees, scaled to ~[-1, 1].
    """
    fps = sig.attrs.get("fps", 30.0)
    pad = int(pad_s * fps)
    a, b = max(0, rep.start - pad), min(len(sig) - 1, rep.end + pad)
    w = sig.iloc[a: b + 1]
    thigh = float(sig["thigh_px"].iloc[0])
    s0 = sig.iloc[rep.start]
    cols = {
        "depth": w["depth"],
        "hip_height": w["hip_height"] / 2.0,
        "hip_vel": np.clip(w["hip_vel"] / 5.0, -2, 2),
        "knee_angle": w["knee_angle"] / 180.0,
        "hip_angle": w["hip_angle"] / 180.0,
        "ankle_dx": (w["ankle_x"] - s0["ankle_x"]) / thigh,
        "ankle_dy": (w["ankle_y"] - s0["ankle_y"]) / thigh,
        "shoulder_dx": (w["shoulder_x"] - s0["shoulder_x"]) / thigh,
        "leg_conf": w["leg_conf"],
        "duration": np.full(len(w), np.log1p((rep.end - rep.start) / fps)),
    }
    if lift == "deadlift":
        bar = (w["ankle_y"] - w["wrist_y"]) / thigh
        cols.update(
            bar_height=bar / 2.0,
            bar_vel=np.clip(np.gradient(bar.to_numpy()) * fps / 5.0, -2, 2) if len(w) > 1 else np.zeros(len(w)),
            bar_dx=(w["wrist_x"] - s0["wrist_x"]) / thigh,
        )
    src = np.linspace(0, 1, len(w))
    dst = np.linspace(0, 1, n_steps)
    out = np.stack([np.interp(dst, src, pd.Series(np.asarray(cols[c], float)).ffill().bfill().fillna(0).to_numpy())
                    for c in LIFT_CHANNELS[lift]], axis=1)
    return out.astype("float32")


def analysis_sequences(analysis) -> np.ndarray:
    """All reps of an `Analysis` as an array (n_reps, N_STEPS, N_FEATURES)."""
    reps = [Rep(v.rep, v.start, v.bottom, v.end, True) for v in analysis.verdicts]
    if not reps:
        return np.zeros((0, N_STEPS, N_FEATURES), "float32")
    return np.stack([rep_sequence(analysis.signals, r, lift=analysis.lift) for r in reps])


# --------------------------------------------------------------------------
# synthetic training data
# --------------------------------------------------------------------------
def synthetic_training_set(n_reps: int = 1500, n_lifters: int = 60, true_line_deg: float = 3.0,
                           fault_rate: float = 0.18, seed: int = 0, lift: str = "squat"):
    """Labelled synthetic reps. Returns X (n, N_STEPS, F), Y (n, faults), meta DataFrame.

    lift="deadlift" simulates deadlifts with the faults in LIFT_FAULTS["deadlift"].

    Each simulated lifter has their own tempo, keypoint noise, filmed side and
    frame rate, so a lifter-grouped split tests generalisation to new people.
    `true_line_deg` is the hidden "referee" depth line the model has to learn.
    """
    from . import analyze_pose
    from .synthetic import make_squat

    if lift == "deadlift":
        return _synthetic_deadlifts(n_reps, n_lifters, fault_rate, seed)
    rng = np.random.default_rng(seed)
    lifters = [dict(tempo=rng.uniform(0.6, 1.8), noise=rng.uniform(0.8, 4.0),
                    side=rng.choice(["left", "right"]), fps=float(rng.choice([30, 30, 60])))
               for _ in range(n_lifters)]
    X, Y, meta = [], [], []
    i = 0
    while len(X) < n_reps:
        i += 1
        lid = int(rng.integers(n_lifters))
        L = lifters[lid]
        depth_deg = float(rng.uniform(-12, 18))
        f = {k: bool(rng.random() < fault_rate) for k in ("lockout", "bounce", "downward", "feet")}
        kw = dict(depth_deg=depth_deg, fps=L["fps"], side=L["side"], seed=i,
                  tempo=L["tempo"] * rng.uniform(0.85, 1.15), noise_px=L["noise"],
                  drop_frac=float(rng.choice([0, 0, 0.03, 0.08])),
                  bounce=f["bounce"], ascent_dip=float(rng.uniform(0.08, 0.18)) if f["downward"] else 0.0,
                  finish_s=float(rng.uniform(0.22, 0.4)) if f["lockout"] else 0.0,
                  step_px=float(rng.uniform(40, 70)) if f["feet"] else 0.0)
        res = analyze_pose(make_squat(**kw), "squat")
        if len(res.verdicts) != 1:      # rep splitter disagreed: skip, like a bad recording
            continue
        X.append(analysis_sequences(res)[0])
        y = {"depth": depth_deg <= true_line_deg, **f}
        Y.append([float(y[k]) for k in FAULTS])
        v = res.verdicts[0]
        meta.append(dict(lifter=f"lifter_{lid:02d}", depth_deg=depth_deg, rule_status=v.status.value,
                         **{f"rule_{k}": float(k in _rule_faults(v)) for k in FAULTS}))
    return np.stack(X), np.array(Y, "float32"), pd.DataFrame(meta)


def _synthetic_deadlifts(n_reps: int, n_lifters: int, fault_rate: float, seed: int):
    """Deadlifts from simulated lifters; labels from what the simulator did."""
    from . import analyze_pose
    from .synthetic import make_deadlift

    rng = np.random.default_rng(seed)
    lifters = [dict(tempo=rng.uniform(0.6, 1.8), noise=rng.uniform(0.8, 4.0),
                    side=rng.choice(["left", "right"]), fps=float(rng.choice([30, 30, 60])))
               for _ in range(n_lifters)]
    faults = LIFT_FAULTS["deadlift"]
    X, Y, meta = [], [], []
    i = 0
    while len(X) < n_reps:
        i += 1
        lid = int(rng.integers(n_lifters))
        L = lifters[lid]
        f = {k: bool(rng.random() < fault_rate) for k in faults}
        if f["hitching"]:
            f["downward"] = False           # the simulator can't do both in one pull
        lean = knee = 0.0
        if f["lockout"]:                    # soft hips, soft knees, or both
            kind = rng.choice(["hips", "knees", "both"])
            lean = float(rng.uniform(16, 30)) if kind != "knees" else 0.0
            knee = float(rng.uniform(18, 28)) if kind != "hips" else 0.0
        pose = make_deadlift(fps=L["fps"], side=L["side"], seed=i, tempo=L["tempo"] * rng.uniform(0.85, 1.15),
                             noise_px=L["noise"], lean_top=lean, knee_top=knee,
                             dip=float(rng.uniform(0.08, 0.14)) if f["downward"] else 0.0,
                             hitch=f["hitching"], step_px=float(rng.uniform(40, 70)) if f["feet"] else 0.0)
        drop = float(rng.choice([0, 0, 0.03, 0.08]))
        if drop:
            miss = rng.random(len(pose)) < drop
            pose.xy[miss], pose.conf[miss] = np.nan, 0.0
        res = analyze_pose(pose, "deadlift")
        if len(res.verdicts) != 1:
            continue
        X.append(analysis_sequences(res)[0])
        Y.append([float(f[k]) for k in faults])
        v = res.verdicts[0]
        meta.append(dict(lifter=f"lifter_{lid:02d}", rule_status=v.status.value,
                         **{f"rule_{k}": float(k in _rule_faults(v)) for k in faults}))
    return np.stack(X), np.array(Y, "float32"), pd.DataFrame(meta)


def _rule_faults(v) -> set[str]:
    """Map the rule checks of a verdict onto the model's fault names (for comparison)."""
    m = {"squat_depth": "depth", "squat_lockout_start": "lockout", "squat_lockout_finish": "lockout",
         "squat_bounce": "bounce", "squat_downward_movement": "downward", "squat_feet": "feet",
         "deadlift_knees_locked": "lockout", "deadlift_erect": "lockout",
         "deadlift_downward_movement": "downward", "deadlift_hitching": "hitching", "deadlift_feet": "feet"}
    return {m[c.code] for c in v.checks if c.status.value != "pass" and c.code in m}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def build_model(train_X: np.ndarray | None = None, n_steps: int = N_STEPS,
                n_features: int = N_FEATURES, n_faults: int = len(FAULTS), lift: str | None = None):
    """Conv1D + bidirectional GRU, one sigmoid output per fault.

    Conv1D layers learn short shapes in the signals (a bounce, a dip, a step);
    the GRU reads them in order (what happened before / after the bottom).
    A Normalization layer is fitted on the training data and saved inside the model.
    """
    import keras
    from keras import layers

    if lift is not None:
        n_faults = len(LIFT_FAULTS[lift])
    norm = layers.Normalization(axis=-1, name="normalize")
    if train_X is not None:
        norm.adapt(train_X)
    inp = keras.Input((n_steps, n_features), name="rep_signals")
    x = norm(inp)
    x = layers.Conv1D(32, 5, padding="same", activation="relu", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Conv1D(48, 5, padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling1D(2, name="pool")(x)
    x = layers.Bidirectional(layers.GRU(32), name="gru")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    x = layers.Dense(32, activation="relu", name="dense")(x)
    out = layers.Dense(n_faults, activation="sigmoid", name="faults")(x)
    model = keras.Model(inp, out, name=f"rep_fault_net_{lift or 'squat'}")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(multi_label=True, num_labels=n_faults, name="auc"),
                           keras.metrics.BinaryAccuracy(name="acc")])
    return model


def fine_tune(model, X: np.ndarray, Y: np.ndarray, epochs: int = 40, lr: float = 3e-4,
              freeze_up_to: str = "gru", validation_split: float = 0.2):
    """Adapt a synthetic-pretrained model to real labelled reps.

    Layers up to and including `freeze_up_to` keep their weights (they already
    know what squat signals look like); only the head is retrained, with a small
    learning rate. Unfreeze more layers once you have a few hundred real reps.
    """
    import keras

    frozen = True
    for layer in model.layers:
        layer.trainable = not frozen
        if layer.name == freeze_up_to:
            frozen = False
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(multi_label=True, num_labels=Y.shape[1], name="auc")])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    return model.fit(X, Y, epochs=epochs, batch_size=16, validation_split=validation_split,
                     callbacks=[stop], verbose=0)


def load_model(path: str | Path):
    import keras
    return keras.saving.load_model(path)


def predict_faults(model, analysis, threshold: float = 0.5) -> pd.DataFrame:
    """Fault probabilities for every rep of an `Analysis` (the model must match its lift)."""
    faults = list(LIFT_FAULTS.get(analysis.lift, FAULTS))
    X = analysis_sequences(analysis)
    if len(X) == 0:
        return pd.DataFrame(columns=["rep", *faults, "dl_label"])
    P = np.asarray(model.predict(X, verbose=0))
    if P.shape[1] != len(faults):
        raise ValueError(f"This model predicts {P.shape[1]} faults, but a {analysis.lift} has {len(faults)}: "
                         f"use the {analysis.lift} model ({DEFAULT_MODEL_PATHS.get(analysis.lift)}).")
    df = pd.DataFrame(P, columns=faults)
    df.insert(0, "rep", np.arange(1, len(df) + 1))
    df["dl_label"] = np.where((P >= threshold).any(1), "NO LIFT", "GOOD LIFT")
    return df
