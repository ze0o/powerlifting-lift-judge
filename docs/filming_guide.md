# How to film so the judge sees what a referee sees

The judge measures where your **hip** is compared with your **knee**, and how straight your
knees and hips are. A camera only sees that **from the side**, where a side referee sits.

```
                 FRONT  ✗  depth not judged
                    ▼
        45° ⚠ ↘  ┌─────────┐
                 │  lifter │
   SIDE ✓  ──►   │ ═══bar═══│   ◄──  SIDE ✓
                 └─────────┘
                    ▲
                 BACK  ✗
```

## Why not the front?

At the bottom of a squat the thighs point straight at a front camera, so the hip and the knee
sit on top of each other in the picture. Their height difference, which is the whole depth
rule, disappears. Competition broadcast footage is usually filmed from the front, so it's the
wrong angle for this tool even though it looks good.

The app **detects the camera angle** from the keypoints: from the side the left and right
shoulders overlap, from the front they're a body-width apart.

| Detected view | What the judge does |
|---|---|
| Side | judges everything |
| Angled (about 45°) | doubles the borderline zone for depth and warns you |
| Front / back | doesn't judge depth (squat) or lockout and hitching (deadlift); still checks bounces, downward movement and steps, and warns you |

## Camera

- **Hip height**, about 80–100 cm off the floor. A low camera makes the hip look lower
  (deeper) than it is; a high one makes it look higher.
- **3–5 m away**, with head, feet and the whole bar in frame for the full set.
- **Square to the lifter**: at 90° to the line between the feet.
- **Steady**: tripod or a fixed stand. Landscape, 30 or 60 fps, 720p or more.
- Start recording before the unrack or the pull; stop after the rack or the bar is down.

## In the frame

- **Whole body, every rep.** If the feet leave the frame, steps can't be checked.
- **Spotters behind the lifter**, not between the camera and the lifter. At the bottom of a
  squat they cover the knees at the exact moment that matters, and the tracker can jump to them.
- **Clothes that show the hip and knee**: a singlet or fitted shorts. Baggy shorts or a long
  shirt hide the hip crease. A belt is fine.
- **Light on the lifter**, not behind them (no window behind you).
- Fewer people walking behind = fewer tracking mistakes.

## Per lift

| Lift | Camera | Watch out for |
|---|---|---|
| Squat | side, hip height | plates hiding the hip on the camera side: film the other side, or a little in front of the bar |
| Deadlift | side, hip height, slightly behind the bar | plates covering the knees and hips at lockout |
| Bench press | not supported yet | needs bar tracking: the plates hide the wrists from the side |

## Settings

- **Pose model size**: Nano is fast but its hip and knee points wobble a little between
  frames. For close depth calls use **Small** or **Medium**.
- **Depth line** (squat): default 0.07 thigh lengths. Calibrate it on your own labelled
  videos (notebook 02) once you have them.

## Checklist before you lift

- [ ] Phone on a tripod, landscape, at hip height
- [ ] Side view, square to the lifter
- [ ] Head, feet and bar in frame
- [ ] Spotters behind
- [ ] Recording started
