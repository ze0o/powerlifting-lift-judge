# Data

Videos and keypoints are **not committed to git** (they're large and show real
people). Keep them in this folder locally. `.gitignore` ignores everything here
except this file and `labels_template.csv`.

```
data/
  raw/<lift>/<video>.mp4        your recordings
  raw/<lift>/<video>_keypoints.csv   cached pose output (created automatically)
  labels.csv                    your human judgement per rep
```

## How to record (this decides the accuracy)

- **Side view** for the squat and deadlift, camera at **hip height**, about 3–5 m away.
  The far-side leg is hidden from the side, so the near side is used.
- Camera **square to the lifter** (perpendicular to the line between the feet).
  A camera angled 30° from the side makes the hip look higher or lower than it is.
- **Whole body in frame** for the whole set, including feet and the bar.
- Phone on a tripod or a fixed stand, not hand-held. 1080p, 30 or 60 fps.
- Plain background if possible; ask spotters to stand behind the lifter, not in front.
- **Get consent** from everyone you film. Don't upload other people's videos to the repo.

## Labelling workflow

1. Put the videos in `data/raw/squat/`.
2. `liftjudge label data/raw/squat`: drafts `data/labels_draft.csv` and saves a bottom-position
   snapshot per rep in `data/snapshots/`.
3. Open the CSV in Excel/VS Code next to the snapshots. Check each row. The **empty cells with a
   `CHECK:` note** are close calls the judge refused to make: watch the rep and decide.
4. Put your name in `judge` and save as `data/labels.csv`.

Label honestly, as a referee would. Don't copy the draft's opinion when you're unsure;
write `unsure` in `notes` instead.

## Labels (`labels.csv`)

One row per rep, judged by you (and ideally a second person, to measure agreement).
Copy `labels_template.csv` to start.

| column | meaning |
|---|---|
| `video_id` | video file name without extension |
| `rep` | rep number in the video, starting at 1 |
| `lift` | squat / bench / deadlift |
| `judge` | who labelled it (e.g. `aziz`, `coach1`) |
| `depth_ok` | 1 = hip crease below top of knee, 0 = high |
| `good_lift` | 1 = white light, 0 = no lift |
| `cards` | failed card colours, e.g. `red` or `red+blue`, empty if good |
| `camera` | side / front / angled-45 |
| `depth`, `lockout`, `bounce`, `downward`, `feet` | 1 if the rep has that fault, else 0 (used to fine-tune the deep learning model) |
| `notes` | free text |

Aim for **at least 150 squat reps** with a good number of high and borderline
ones; deliberately include some above-parallel reps with light weight, or the
dataset will be almost all good lifts.

## Public datasets worth requesting

- **Fitness-AQA** (Parmar et al.): squat videos with expert labels for errors
  including shallow depth. Request form: https://github.com/ParitoshParmar/Fitness-AQA
