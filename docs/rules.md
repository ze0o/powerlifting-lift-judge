# From IPF rules to measurable checks

Source: IPF Technical Rulebook 2026 (effective 1 March 2026), referee card chart.
Each card colour groups a set of failures for a lift. A lift is good when at least
two of three referees give a white light.

> Video can't hear the Chief Referee's commands ("Squat", "Rack", "Start", "Press",
> "Down"), so failures that are only about commands are out of scope. The tool is
> a **training aid and second opinion**, not a replacement for referees.

All distances are in **thigh lengths** (hip-to-knee distance), so results don't
depend on camera distance. Image `y` points down.

## Squat (built: phase 1)

| Card | IPF failure | How we measure it | Default threshold |
|---|---|---|---|
| 🔴 Red | Top surface of the legs at the hip joint not below the top of the knees | `depth = (hip_y − knee_y) / thigh` at the bottom frame, minus a **crease offset** | pass if margin > +0.04, fail if < −0.04, otherwise **borderline** |
| 🔵 Blue | Not upright with knees locked at the start/finish | Max knee angle in a 0.3 s window before the descent and after the rep | ≥ 160° |
| 🔵 Blue | Double bouncing / more than one recovery attempt | In the hole (within 0.35 thighs of the bottom), the largest rise-then-drop of the hips | > 0.08 thighs |
| 🔵 Blue | Any downward movement during the ascent | After the bottom, the largest drop below the running highest point | > 0.04 thighs (~2 cm) |
| 🟡 Yellow | Stepping or moving the feet | Largest ankle displacement during the rep | > 0.15 thighs |

### Why the crease offset exists

Pose models (YOLO, MediaPipe) find the **centre of the hip joint** and the **centre
of the knee joint**. The rule compares **surfaces**: the top of the thigh at the hip
crease against the top of the knee. The thigh is thicker at the hip than at the knee,
so the hip-joint centre has to be a little *below* the knee-joint centre before the
surfaces pass. That offset (default 0.07 thighs ≈ 3 cm) differs between lifters and
camera angles, so it is **calibrated on labelled videos** (`notebooks/02`).

The borderline band is there on purpose: keypoints jitter by a few pixels, and a
tool that admits "too close to call" is more trustworthy than one that guesses.

### Coaching metrics (not judged)

`descent_s`, `ascent_s`, `sticking_speed` (slowest point of the ascent),
`shoulder_drift` (horizontal sway of the bar position), `depth_margin`.

## Bench press (planned: phase 3)

| Card | IPF failure | Planned measurement |
|---|---|---|
| 🔴 Red | Bar not lowered to the chest/abdomen; elbows not at or below shoulder level | Bar height at the bottom vs chest line; elbow y vs shoulder y |
| 🔵 Blue | Downward movement of the whole bar while pressing; elbows not locked | Bar height signal (plate detector); elbow angle at lockout |
| 🟡 Yellow | Heaving/sinking after the pause; head, shoulders or buttocks lifting; feet on the bench | Bar speed ≈ 0 for N frames, then any further drop; hip keypoint leaving the bench line |

Needs a **barbell/plate detector**: from the side, the wrists are often hidden behind the plates.

## Deadlift (built)

The bar is tracked through the wrists (arms hang straight). A rep runs from the bar leaving
the floor, up to its highest point (lockout), and back down.

| Card | IPF failure | How we measure it | Default threshold |
|---|---|---|---|
| 🔴 Red | Knees not locked at completion | Max knee angle while the bar is at its highest | ≥ 165° |
| 🔴 Red | Not standing erect with shoulders back | Max hip angle (shoulder–hip–knee) at lockout | ≥ 165° |
| 🔵 Blue | Downward movement of the bar before the final position | Largest drop of the bar below its highest point so far, during the pull | > 0.05 thighs |
| 🔵 Blue | Supporting the bar on the thighs (hitching) | Knees re-bending during the pull while the bar is above the knees | > 8° |
| 🟡 Yellow | Stepping / moving the feet | Largest ankle displacement | > 0.15 thighs |

Not judged from video: lowering before the Chief Referee's "Down" signal, dropping the bar.

Coaching metrics: `pull_s`, `hold_s`, `lockout_knee`, `lockout_hip`, `bar_drift`, `sticking_speed`.

## Camera angle

Before judging, the camera angle is detected from the keypoints (`view.py`): body width
(left–right shoulders and hips) divided by torso length. From the side the two sides overlap
(ratio < 0.25); from the front they're a body-width apart (> 0.45; competition front footage
measured 0.52 standing and 0.73 in the hole).

- **Side**: everything is judged.
- **Angled**: the squat depth borderline band is doubled.
- **Front/back**: squat depth, deadlift lockout and hitching are not judged (status
  *unknown*, so the rep becomes *borderline*); left and right legs are averaged.

See [filming_guide.md](filming_guide.md).
