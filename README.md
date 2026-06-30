# StudyGuard AI — Personal Attention & Posture Monitoring System

A browser-based study monitoring tool that uses your laptop webcam and computer
vision to track posture, drowsiness, phone usage, and attention in real time —
no external hardware required.

## What's inside

```
studyguard/
├── app.py              Flask + SocketIO backend (routes, camera thread, session stats)
├── detector.py          Computer vision engine (MediaPipe + YOLO + state machine)
├── requirements.txt     Python dependencies
└── templates/
    └── index.html       Full dashboard frontend (HTML + CSS + JS, one file)
```

## How to run

```bash
# 1. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py

# 4. Open in your browser
http://localhost:5000
```

The first time you run it, `ultralytics` will auto-download the YOLOv8n
weights file (`yolov8n.pt`, ~6MB) — you need an internet connection for this
one-time download. After that it works fully offline.

## How to use it

1. Click **Start study mode**.
2. A 5-second countdown begins — sit naturally in your normal study position.
3. The system records a 5-second baseline of your posture, head angle, and
   eye openness.
4. Once calibration finishes, live monitoring starts automatically.
5. Click **Stop session** at any time to see your session report, with stats
   and a downloadable `.txt` summary.

## Why some pip warnings are normal

When installing, you may see a message like:

```
ultralytics 8.2.50 requires opencv-python, mediapipe requires opencv-contrib-python...
```

This is expected. Both packages pull in their own OpenCV build as a
dependency, and both provide the same `cv2` module — the app has been tested
and works correctly with this combination. You can safely ignore that
warning.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Could not access webcam" | Close any other app using the camera (Zoom, Teams, browser tab) and refresh the page. |
| Calibration keeps failing | Make sure your face is well-lit and centered in frame, then click Retry. |
| Page loads but feed never appears | Check the terminal running `python app.py` for errors — most often a camera permission issue at the OS level (System Settings → Privacy → Camera, on macOS). |
| YOLO/phone detection seems to do nothing | Check your terminal for `yolov8n.pt` download errors — it needs internet access on first run only. |
| `ModuleNotFoundError` on launch | Run `pip install -r requirements.txt` again inside the same environment you're using to run `python app.py`. |

## Extending this later

The code was structured so each of these is a small change, not a rewrite:

- **Multi-user support**: `session_stats` in `app.py` is already a clean,
  isolated dict — give it a `session_id` key and store one per user.
- **Database storage**: the `session_stats` dict and `/report` JSON map
  directly onto a simple SQL table.
- **AI-generated coaching**: swap the hardcoded strings in the
  `recommendations` list (in `app.py`, inside `/report`) for a call to an
  LLM API using the same stats as input.
- **detector.py has zero Flask imports on purpose** — it's a plain Python
  class, so it can be reused in a CLI tool, a test suite, or a different
  backend without changes.

## Notes for submission

All computer vision runs locally in your browser session via the Python
backend — no video or images are ever uploaded anywhere. This is a
single-user, rule-based monitoring system (not emotion recognition or
medical-grade drowsiness detection) — it works off measurable visual signals:
head angle, eye aspect ratio, shoulder slope, and object detection for phones.
