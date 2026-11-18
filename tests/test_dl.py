import numpy as np
import pytest

from liftjudge import analyze_pose
from liftjudge.dl import FAULTS, N_FEATURES, N_STEPS, analysis_sequences, synthetic_training_set
from liftjudge.synthetic import make_squat


def test_rep_sequence_shape_and_no_nans():
    res = analyze_pose(make_squat(reps=2, drop_frac=0.1, seed=3), "squat")
    X = analysis_sequences(res)
    assert X.shape == (2, N_STEPS, N_FEATURES)
    assert np.isfinite(X).all()


def test_same_rep_at_different_speeds_looks_alike():
    slow = analysis_sequences(analyze_pose(make_squat(tempo=0.7, noise_px=0), "squat"))[0]
    fast = analysis_sequences(analyze_pose(make_squat(tempo=1.6, noise_px=0), "squat"))[0]
    # each rep is resampled to the same length, so a slow and a fast squat have the
    # same depth curve shape and range (only the fixed lockout padding differs)
    assert np.corrcoef(slow[:, 0], fast[:, 0])[0, 1] > 0.9
    assert abs(slow[:, 0].max() - fast[:, 0].max()) < 0.03
    assert slow[0, -1] > fast[0, -1]  # the duration channel tells them apart (slow = longer)


def test_synthetic_labels_match_the_faults():
    X, Y, meta = synthetic_training_set(n_reps=60, n_lifters=10, seed=1)
    assert X.shape[0] == Y.shape[0] == len(meta) == 60
    assert Y.shape[1] == len(FAULTS)
    np.testing.assert_array_equal(Y[:, 0], (meta.depth_deg <= 3.0).astype("float32"))


def test_model_trains_and_predicts(tmp_path):
    keras = pytest.importorskip("keras")
    from liftjudge.dl import build_model, load_model, predict_faults

    keras.utils.set_random_seed(0)
    X, Y, _ = synthetic_training_set(n_reps=80, n_lifters=10, seed=2)
    model = build_model(X)
    hist = model.fit(X, Y, epochs=3, batch_size=16, verbose=0)
    assert hist.history["loss"][-1] < hist.history["loss"][0]

    model.save(tmp_path / "m.keras")
    probs = predict_faults(load_model(tmp_path / "m.keras"), analyze_pose(make_squat(reps=2), "squat"))
    assert list(probs.columns) == ["rep", *FAULTS, "dl_label"]
    p = probs[list(FAULTS)].to_numpy()
    assert len(probs) == 2 and ((p >= 0) & (p <= 1)).all()
