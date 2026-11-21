# 🏋️ Powerlifting Lift Judge

**Computer vision that judges powerlifting technique (squat and deadlift) from a phone video, using the IPF rules.**

Upload a side-view video of a squat or deadlift set and get, for every rep:
**GOOD LIFT / NO LIFT / BORDERLINE**, the referee card colour (🔴 🔵 🟡) and the
reason, such as *"High: hip crease did not go below the top of the knee"* or
*"Downward movement during the ascent"*.

Built for **lifters** checking their training sets, **coaches** reviewing technique
remotely, and **federations** that want an objective second opinion on close calls.

> Status: squat and deadlift rule engines, camera-angle detection, deep learning model (squat), web demo and Flutter app. Bench press is on the [roadmap](docs/roadmap.md).

---

## How it works

```
video ─► YOLO26-pose ─► keypoints per frame ─► clean signals ─► split into reps ─► IPF rules ─► verdict + annotated video
         (17 joints)     (lifter only)          (depth, angles,     (peaks in       (red / blue /
                                                  velocities)        hip depth)      yellow checks)
```

1. **Pose estimation**: Ultralytics [YOLO26-pose](https://docs.ultralytics.com/models/yolo26/) finds 17 body keypoints.
   The lifter is the tracked person with the largest total box area, so spotters and referees are ignored.
2. **Signals**: the side facing the camera is picked automatically. Gaps are interpolated and the signals are
   smoothed (Savitzky–Golay). Everything is measured in **thigh lengths**, so camera distance doesn't matter.
3. **Rep splitting**: peaks in hip depth with a minimum prominence, so a wobble in the hole stays inside one rep.
4. **Rules**: each IPF failure becomes a measurable check ([docs/rules.md](docs/rules.md)).
5. **Output**: a printed verdict, a JSON file, a per-frame signals CSV and an annotated video.

### The interesting problem: joints vs. surfaces

The IPF rule compares the **top surface of the thigh at the hip crease** with the **top of the knee**.
A pose model only gives **joint centres**. So `hip.y > knee.y` is not the rule. The judge adds a calibrated
**crease offset** and a **borderline band** where it refuses to call the lift, instead of pretending
keypoints are exact. Calibrating that offset on labelled videos is phase 2.

## Squat checks

| Card | Failure | Check |
|---|---|---|
| 🔴 | Not deep enough | hip–knee depth at the bottom vs. crease offset (± borderline band) |
| 🔵 | Knees not locked at start / finish | knee angle ≥ 160° |
| 🔵 | Double bounce | rise-then-drop of the hips in the hole |
| 🔵 | Downward movement on the way up | any drop below the highest point reached so far |
| 🟡 | Stepping | ankle displacement during the rep |

It also reports coaching metrics: descent/ascent time, sticking-point speed and horizontal sway.

## Deadlift checks

| Card | Failure | Check |
|---|---|---|
| 🔴 | Knees not locked / not standing erect | knee and hip angle at lockout ≥ 165° |
| 🔵 | Downward movement before lockout | bar (wrists) dropping during the pull |
| 🔵 | Hitching | knees re-bending with the bar on the thighs |
| 🟡 | Stepping | ankle displacement |

## Camera angle and crowded platforms

- **Camera-angle detection**: from the side, the left and right shoulders overlap in the picture;
  from the front they're a body-width apart. The judge measures this and **refuses to judge depth
  from a front view** (the hip and knee overlap when the thighs point at the camera) instead of
  giving a confident wrong answer. A 45° view gets a wider borderline zone.
- **Following the lifter through spotters**: the lifter is the biggest person who stays near the
  middle of the frame; when spotters crowd in and the tracker loses them, the judge re-attaches
  to whoever overlaps the lifter's last position.
- **[Filming guide](docs/filming_guide.md)**: where to put the camera, also built into the app.

## Web demo

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

Upload a side-view squat video and get the verdict per rep with red/blue/yellow cards and reasons,
the annotated video (H.264, plays in any browser), the depth curve over the set, and the neural network's
second opinion. It flags reps where the network and the rules disagree. Sliders let you change the pose
model size and the depth threshold live. **Try a sample set** works without a video.

## Mobile app (Flutter)

Record a set on your phone and get the verdict per rep: referee lights, card colours, reasons,
the annotated video, the depth curve and the network's second opinion. The phone uploads the video
to a FastAPI server that runs this pipeline (`uvicorn liftjudge.api:app --host 0.0.0.0`).
Setup and structure: [mobile/README.md](mobile/README.md).

## Labelling real videos in minutes

The judge drafts the labels and you only check them:

```bash
liftjudge label data/raw/squat --out data/labels_draft.csv --snapshots data/snapshots
```

- one row per rep with the rule verdict and fault columns already filled in, plus the rep's time in the video;
- **close calls are left empty** with a `CHECK:` note, because those are the reps a human must decide
  (and the most valuable ones for training);
- a snapshot of each rep's bottom position with the hip and knee lines drawn, for fast depth checks.

Fix what's wrong, put your name in `judge`, save as `data/labels.csv`. Then calibrate (notebook 02)
and fine-tune the network (notebook 03).

## Deep learning: a second opinion that reads the whole rep

The rules check a few hand-picked numbers. A **Keras sequence model** reads the entire rep instead
(96 frames × 10 signals) and outputs one probability per fault: depth, lockout, bounce, downward
movement, feet. A rep can have several faults at once, so it's multi-label.

```
rep signals (96×10) ─► Normalization ─► Conv1D ×2 ─► MaxPool ─► Bidirectional GRU ─► Dense ─► 5 × sigmoid
                                         short shapes          order in time               one per fault
                                         (bounce, dip, step)   (before / after the bottom)
```

- **Sim-to-real training:** pre-trained on 2,000 synthetic reps from 60 simulated lifters (labels are free),
  then fine-tuned on real labelled reps with the early layers frozen.
- **Evaluated on lifters it has never seen** (train/val/test split by lifter).
- On synthetic data it **learns the hidden referee depth line from labels alone**: depth F1 0.97 vs 0.91 for
  the fixed-threshold rules; good-lift / no-lift accuracy 99.5% vs 96%. Those are upper bounds until it's
  tested on real lifts.
- The rules stay as the explainable verdict. Where the network and the rules disagree is where a human should look.

```bash
liftjudge train-dl --out models/rep_fault_net.keras            # ~2 min on a laptop CPU
liftjudge analyze squat.mp4 --dl-model models/rep_fault_net.keras
```
```
Rep 1: NO LIFT  [blue card]
   X  Downward movement during the ascent (0.23 thighs)

Neural network (second opinion):
Rep 1: NO LIFT - downward movement on the way up (100%)
```

Full walkthrough: [notebooks/03_deep_learning_sequence_model.ipynb](notebooks/03_deep_learning_sequence_model.ipynb).

## Quick start

```bash
git clone https://github.com/ze0o/powerlifting-lift-judge.git
cd powerlifting-lift-judge
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# judge a video (the pose model downloads on first run)
liftjudge analyze path/to/squat.mp4 --lift squat --out results/

# train the deep learning model (Keras uses TensorFlow if you have it;
# otherwise set KERAS_BACKEND=torch, PyTorch is already installed with Ultralytics)
liftjudge train-dl

# run the tests
pytest
```

Example output:

```
Rep 1: GOOD LIFT
   ok Depth: hip crease below the knee (+0.19 thighs)
   ok Knees locked at the start (178°)
   ok Knees locked at the finish (180°)
   ok Single, controlled bottom position
   ok Continuous ascent, no downward movement
   ok Feet stayed in place
Rep 2: NO LIFT  [red card]
   X  High: hip crease did not go below the top of the knee (-0.24 thighs)
   ...
```

**Recording tips** (they matter more than the model): side view, camera at hip height and square to the
lifter, whole body in frame, phone on a tripod. See the [filming guide](docs/filming_guide.md).
For close depth calls use a bigger pose model: `--model yolo26s-pose.pt` or `yolo26m-pose.pt`.

## Project structure

```
src/liftjudge/
  keypoints.py     video → lifter keypoints (YOLO26-pose + tracking)
  signals.py       keypoints → depth, joint angles, velocities
  reps.py          rep segmentation
  rules/           IPF rules per lift (squat, deadlift; bench planned)
  view.py          camera-angle detection (side / angled / front)
  features.py      rep-level feature table for ML / calibration
  dl.py            Keras sequence model: rep → fault probabilities, synthetic pre-training, fine-tuning
  render.py        annotated output video (H.264)
  api.py           FastAPI server for the mobile app
  labeling.py      draft labels + bottom-position snapshots for fast human labelling
  synthetic.py     synthetic squats for tests and demos
  cli.py           `liftjudge analyze ...`
app/
  streamlit_app.py web demo
mobile/            Flutter app (talks to src/liftjudge/api.py)
notebooks/
  01_squat_judge_walkthrough.ipynb   problem, approach, signals, rules
  02_calibration_and_ml.ipynb        calibrating the offset + rep classifier
  03_deep_learning_sequence_model.ipynb   Conv1D + GRU network, rules vs deep learning
docs/              rules mapping, roadmap
tests/             rule and model tests on synthetic lifts
```

## Limitations

- 2-D video from one camera: a camera that isn't square to the lifter shifts the apparent depth
  (the judge detects front and angled views and says so).
- Commands from the Chief Referee can't be judged from video.
- Rules are validated on synthetic data so far; accuracy on real lifts will be reported after labelling (phase 1b).
- Intended as a **training aid and second opinion**, not a replacement for referees.

## Tech

Python · Ultralytics YOLO26 · OpenCV · FastAPI · Streamlit · Flutter/Dart · NumPy · pandas · SciPy · scikit-learn · Keras 3 (Conv1D + GRU) · pytest

## License

AGPL-3.0, matching the Ultralytics models this project builds on.

---

Made by **Abdulaziz Abdulhadi**, Software Engineering student (University of Bahrain), Data Science (General Assembly) and powerlifter.
[GitHub](https://github.com/ze0o) · [LinkedIn](https://www.linkedin.com/in/abdulaziz-abdulhadi-5510a1332) · [Portfolio](https://3ziz-portfolio.netlify.app/)
