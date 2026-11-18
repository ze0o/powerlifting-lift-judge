"""Command line.

  liftjudge analyze VIDEO --lift squat --out results/ [--dl-model models/rep_fault_net.keras]
  liftjudge train-dl --out models/rep_fault_net.keras
  liftjudge label data/raw/squat --out data/labels_draft.csv --snapshots data/snapshots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import analyze_pose
from .keypoints import DEFAULT_MODEL, PoseSequence, extract_pose
from .rules import SquatConfig


def _analyze(args) -> int:
    src = Path(args.video)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() == ".csv":
        pose = PoseSequence.load_csv(src)
    else:
        pose = extract_pose(src, model=args.model)
        pose.save_csv(out / f"{src.stem}_keypoints.csv")

    cfg = SquatConfig(crease_offset=args.crease_offset)
    res = analyze_pose(pose, args.lift, args.side, cfg)
    print(res.summary())

    res.signals.to_csv(out / f"{src.stem}_signals.csv", index=False)
    with open(out / f"{src.stem}_verdicts.json", "w") as f:
        json.dump([v.to_dict() for v in res.verdicts], f, indent=2)

    if args.dl_model:
        from .dl import FAULT_TEXT, FAULTS, load_model, predict_faults
        probs = predict_faults(load_model(args.dl_model), res)
        probs.to_csv(out / f"{src.stem}_dl_faults.csv", index=False)
        print("\nNeural network (second opinion):")
        for _, r in probs.iterrows():
            found = [f"{FAULT_TEXT[f]} ({r[f]:.0%})" for f in FAULTS if r[f] >= 0.5]
            print(f"Rep {int(r.rep)}: {r.dl_label}" + (f" - {'; '.join(found)}" if found else ""))

    if not args.no_video and src.suffix.lower() != ".csv":
        from .render import render_video
        path = render_video(src, out / f"{src.stem}_judged.mp4", pose, res.signals,
                            res.verdicts, cfg.crease_offset)
        print(f"Annotated video: {path}")
    return 0


def _train_dl(args) -> int:
    import keras
    from sklearn.model_selection import GroupShuffleSplit

    from .dl import build_model, synthetic_training_set

    keras.utils.set_random_seed(args.seed)
    print(f"Generating {args.reps} synthetic reps...")
    X, Y, meta = synthetic_training_set(args.reps, seed=args.seed)
    tr, va = next(GroupShuffleSplit(test_size=0.2, random_state=args.seed).split(X, Y, meta.lifter))
    model = build_model(X[tr])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    hist = model.fit(X[tr], Y[tr], validation_data=(X[va], Y[va]), epochs=args.epochs,
                     batch_size=32, callbacks=[stop], verbose=2 if args.verbose else 0)
    best = min(range(len(hist.history["val_loss"])), key=hist.history["val_loss"].__getitem__)
    print(f"Best epoch {best + 1}: val_loss {hist.history['val_loss'][best]:.3f}, "
          f"val_auc {hist.history['val_auc'][best]:.3f} (unseen simulated lifters)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"Saved {out}")
    return 0


def _label(args) -> int:
    from .labeling import draft_labels

    df = draft_labels(args.folder, args.out, args.lift, args.snapshots, args.model)
    unsure = (df["good_lift"] == "").sum() if len(df) else 0
    print(f"\n{len(df)} reps drafted -> {args.out}  ({unsure} need a human decision: empty good_lift)")
    if args.snapshots:
        print(f"Bottom-position snapshots -> {args.snapshots}")
    print("Check every row, fix what's wrong, set judge to your name, save as data/labels.csv")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="liftjudge", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="judge every rep in a video")
    a.add_argument("video", help="video file, or a keypoints .csv saved by a previous run")
    a.add_argument("--lift", default="squat", choices=["squat", "bench", "deadlift"])
    a.add_argument("--model", default=DEFAULT_MODEL, help="YOLO pose weights")
    a.add_argument("--side", choices=["left", "right"], help="body side facing the camera")
    a.add_argument("--out", default="results", help="output folder")
    a.add_argument("--no-video", action="store_true", help="skip the annotated video")
    a.add_argument("--crease-offset", type=float, default=SquatConfig.crease_offset)
    a.add_argument("--dl-model", help="trained rep fault network (.keras) for a second opinion")
    a.set_defaults(func=_analyze)

    t = sub.add_parser("train-dl", help="pre-train the rep fault network on synthetic squats")
    t.add_argument("--out", default="models/rep_fault_net.keras")
    t.add_argument("--reps", type=int, default=2000)
    t.add_argument("--epochs", type=int, default=120)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--verbose", action="store_true")
    t.set_defaults(func=_train_dl)

    lb = sub.add_parser("label", help="draft labels for a folder of videos (you then check them)")
    lb.add_argument("folder", help="folder with videos, e.g. data/raw/squat")
    lb.add_argument("--out", default="data/labels_draft.csv")
    lb.add_argument("--snapshots", default="data/snapshots", help="folder for bottom-position images ('' to skip)")
    lb.add_argument("--lift", default="squat", choices=["squat"])
    lb.add_argument("--model", default=DEFAULT_MODEL, help="YOLO pose weights")
    lb.set_defaults(func=_label)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
