# Lift Judge: Flutter app

Record or pick a squat video on your phone and get the verdict for every rep: referee
lights, IPF card colours, the reason for each call, the annotated video, the depth curve
and the neural network's second opinion.

The judging runs on the Python server in this repo (YOLO26-pose + IPF rules + Keras model);
the app uploads the video and shows the result. **See a sample result** works offline.

```
phone (Flutter) ──video──► FastAPI server on your PC ──► pose → rules → neural net
                ◄──JSON + annotated video──
```

## Run it

**1. Start the server** (repo root, on your PC):

```bash
pip install -e ".[api]"
liftjudge train-dl                                  # once: neural-network second opinion (optional)
uvicorn liftjudge.api:app --host 0.0.0.0 --port 8000
```

Find your PC's Wi-Fi address with `ipconfig` (e.g. `192.168.1.20`). Allow Python through the
Windows firewall when asked (private networks).

**2. Create the platform folders and run the app** (first time only):

```bash
cd mobile
flutter create . --project-name lift_judge --org com.ze0o --platforms android,ios
python tool/setup_platforms.py      # adds internet/camera permissions + local HTTP
flutter pub get
flutter test
flutter run
```

**3. In the app:** Settings → server address → *Test connection*:

| Running the app in | Server address |
|---|---|
| Chrome (`flutter run -d chrome`) | `http://localhost:8000` (default on web) |
| Android emulator | `http://10.0.2.2:8000` (default on Android) |
| A real phone | your PC's Wi-Fi address, e.g. `http://192.168.1.20:8000` |

In Chrome there's no camera button: use *Choose a video* to pick a file from your computer.

## Structure

```
lib/
  main.dart                 app entry, loads settings
  api.dart                  HTTP client for the server (health, sample, analyze)
  models.dart               Analysis / RepVerdict / Check / DlOpinion from JSON
  settings.dart             server URL, pose model size, depth line (saved on the phone)
  theme.dart                dark "platform" theme, referee light + card colours
  screens/
    home_screen.dart        record / pick / sample, server status, filming tips
    analyzing_screen.dart   upload + wait, friendly errors with retry
    results_screen.dart     scoreboard, annotated video, depth chart, rep cards
    settings_screen.dart    server, model size, depth line
  widgets/
    ref_lights.dart         three referee lights + failure cards
    depth_chart.dart        depth over the set (CustomPainter, no chart package)
    rep_card.dart           checks, coaching metrics, network opinion, disagreement flag
assets/sample.json          a real API response (synthetic set) for offline demo + tests
test/app_test.dart          model parsing, API client (mocked), widget tests
tool/setup_platforms.py     Android/iOS permissions after `flutter create`
```

## Roadmap

- On-device mode: Google ML Kit pose detection + the rules in Dart, so it works without a server
- Share a judged rep as a clip; history of your sets
- Link to the Meet Prep Log app
