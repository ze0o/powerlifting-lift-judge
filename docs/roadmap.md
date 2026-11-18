# Roadmap

| Phase | Goal | Status |
|---|---|---|
| 1 | **Squat MVP**: pose extraction, rep splitting, IPF squat rules, annotated video, CLI | ✅ done (rules tested on synthetic data) |
| 1b | **Real data**: record ~150 labelled squat reps, run the pipeline, measure agreement with human calls | 🔧 labelling tool ready (`liftjudge label`); recording ⏳ |
| 2 | **Calibration + ML**: calibrate the crease offset; rep classifier (logistic regression, random forest) on rep features | ✅ pipeline done (synthetic) |
| 2a | **Deep learning**: Keras Conv1D + GRU sequence model, multi-label faults, synthetic pre-training + fine-tuning recipe | ✅ done (synthetic); fine-tune on real reps next |
| 2b | **Barbell tracking**: fine-tune YOLO26 to detect plates/bar ends; bar path + velocity | ⏳ |
| 3 | **Deadlift** (lockout, hitching) and **bench press** (pause, sinking, butt lift) | ⏳ |
| 4 | **Demo app**: Streamlit page, upload a video, get the judged video back | ✅ done |
| 5 | **Mobile**: Flutter app + FastAPI server | ✅ built (`mobile/`); on-device ML Kit mode ⏳ |
| 6 | **Portfolio**: demo GIF, results table, LinkedIn post, CV bullet | ⏳ |

## Evaluation plan

- Ground truth: human labels in `data/labels.csv` (two judges where possible;
  report their agreement as the ceiling).
- Main metric: **recall on no-lifts** (missing a bad lift is worse than flagging a good one),
  plus precision, F1 and the borderline rate (how often the tool refuses to call).
- Split by **video/lifter**, not by rep, so the same lifter never appears in train and test.
- Error analysis: list every miss with its frame and camera angle.
