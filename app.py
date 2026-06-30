# === app.py ===
"""
StudyGuard AI - Flask + SocketIO backend.
Run with: python app.py
Then open: http://localhost:5000
"""

from gevent import monkey
monkey.patch_all()
import gevent

import base64
import threading
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

from detector import StudyDetector

app = Flask(__name__)
app.config["SECRET_KEY"] = "studyguard-secret-key"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")

# ---------------------------------------------------------------------------
# Global state (single-user app -> module-level globals are fine here)
# ---------------------------------------------------------------------------
state_lock = threading.RLock()
detector = StudyDetector()

session_state = "IDLE"  # IDLE | CALIBRATING_CAPTURING | MONITORING
calib_frames = []
last_emit_time = 0.0
last_stat_tick = 0.0
calibration_greenlet = None

session_stats = {
    "total_seconds": 0,
    "focused_seconds": 0,
    "distracted_seconds": 0,
    "phone_detections": 0,
    "posture_alerts": 0,
    "drowsiness_events": 0,
    "score_samples": [],          # list of attention_score values over time
    "peak_focus_start": None,     # datetime
    "peak_focus_end": None,
    "peak_focus_duration": 0,     # seconds
    "session_start_time": None,
    "_current_focus_streak_start": None,
    "_last_alert": None,
    "_last_phone_state": False,
}


def reset_session_stats():
    with state_lock:
        session_stats.update({
            "total_seconds": 0,
            "focused_seconds": 0,
            "distracted_seconds": 0,
            "phone_detections": 0,
            "posture_alerts": 0,
            "drowsiness_events": 0,
            "score_samples": [],
            "peak_focus_start": None,
            "peak_focus_end": None,
            "peak_focus_duration": 0,
            "session_start_time": datetime.now(),
            "_current_focus_streak_start": None,
            "_last_alert": None,
            "_last_phone_state": False,
        })


def _update_session_stats(result, dt_seconds):
    """Called once per processed frame to roll stats forward in time."""
    with state_lock:
        session_stats["total_seconds"] += dt_seconds
        session_stats["score_samples"].append(result.get("score", 0))

        attention = result.get("attention")
        now = datetime.now()

        if attention == "Focused":
            session_stats["focused_seconds"] += dt_seconds
            if session_stats["_current_focus_streak_start"] is None:
                session_stats["_current_focus_streak_start"] = now
            streak_duration = (now - session_stats["_current_focus_streak_start"]).total_seconds()
            if streak_duration > session_stats["peak_focus_duration"]:
                session_stats["peak_focus_duration"] = streak_duration
                session_stats["peak_focus_start"] = session_stats["_current_focus_streak_start"]
                session_stats["peak_focus_end"] = now
        else:
            session_stats["_current_focus_streak_start"] = None
            if attention in ("Distracted", "Looking Away"):
                session_stats["distracted_seconds"] += dt_seconds

        # Count phone detections on rising edge (off -> on) so we count
        # "pickups" rather than every frame the phone is visible.
        phone_now = result.get("phone_detected", False)
        if phone_now and not session_stats["_last_phone_state"]:
            session_stats["phone_detections"] += 1
        session_stats["_last_phone_state"] = phone_now

        alert = result.get("alert")
        if alert and alert != session_stats["_last_alert"]:
            if alert == "Posture Alert":
                session_stats["posture_alerts"] += 1
            elif alert == "Drowsiness Alert":
                session_stats["drowsiness_events"] += 1
        session_stats["_last_alert"] = alert


def _decode_frame(b64_str):
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"[_decode_frame] error: {e}")
        return None


def _calibration_greenlet():
    """Runs the countdown + capture phase entirely server-side on a timer.
    Completely independent of frame arrival rate."""
    global session_state, calib_frames, last_emit_time, last_stat_tick

    # -- Countdown phase (5 seconds) --
    for seconds_left in range(5, 0, -1):
        with state_lock:
            if session_state == "IDLE":
                return  # Session was stopped
        socketio.emit("calibration_status", {"phase": "countdown", "seconds_left": seconds_left})
        gevent.sleep(1)

    # -- Switch to capturing phase --
    with state_lock:
        if session_state == "IDLE":
            return
        session_state = "CALIBRATING_CAPTURING"
        calib_frames = []
    socketio.emit("calibration_status", {"phase": "capturing", "progress": 0})

    # -- Collect frames for up to 15 seconds (30 frames at ~10fps) --
    deadline = time.time() + 15.0
    while time.time() < deadline:
        with state_lock:
            if session_state == "IDLE":
                return
            n = len(calib_frames)
        if n >= 30:
            break
        gevent.sleep(0.1)

    # -- Run baseline calculation --
    with state_lock:
        frames_copy = list(calib_frames)

    baseline = detector.record_baseline(frames_copy)
    with state_lock:
        if session_state == "IDLE":
            return
        if baseline is None:
            session_state = "IDLE"
            socketio.emit("calibration_status", {"phase": "complete", "success": False})
        else:
            session_state = "MONITORING"
            socketio.emit("calibration_status", {"phase": "complete", "success": True})
            last_stat_tick = time.time()
            last_emit_time = time.time()


@socketio.on("client_frame")
def handle_client_frame(data):
    global last_emit_time, last_stat_tick

    b64_frame = data.get("image")
    if not b64_frame:
        return

    frame = _decode_frame(b64_frame)
    if frame is None:
        return

    now = time.time()

    with state_lock:
        current_state = session_state

    if current_state == "CALIBRATING_COUNTDOWN":
        # Just show the raw frame; the greenlet handles countdown ticks
        _emit_raw_frame(frame)

    elif current_state == "CALIBRATING_CAPTURING":
        with state_lock:
            calib_frames.append(frame.copy())
            n = len(calib_frames)
        progress = int(min(n / 30, 1.0) * 100)
        socketio.emit("calibration_status", {"phase": "capturing", "progress": progress})
        _emit_raw_frame(frame)

    elif current_state == "MONITORING":
        try:
            result = detector.process_frame(frame)
        except Exception as e:
            print(f"[handle_client_frame] detector processing error: {e}")
            return

        dt = now - last_stat_tick
        last_stat_tick = now
        _update_session_stats(result, dt)

        if now - last_emit_time >= 0.1:  # throttle to ~10fps
            frame_b64 = _encode_frame(frame)
            payload = dict(result)
            payload["frame"] = frame_b64
            payload["session_seconds"] = int(session_stats["total_seconds"])
            socketio.emit("update", payload)
            last_emit_time = now


def _encode_frame(frame):
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return None
    frame_b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{frame_b64}"


def _emit_raw_frame(frame):
    """Used during calibration countdown/capture - no CV overlays yet."""
    frame_b64 = _encode_frame(frame)
    socketio.emit("update", {
        "frame": frame_b64,
        "attention": "Calibrating",
        "posture": "Calibrating",
        "fatigue_index": 0,
        "phone_detected": False,
        "alert": None,
        "score": 100,
        "session_seconds": 0,
        "state": "IDLE",
    })


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    global session_state, calib_frames, calibration_greenlet
    with state_lock:
        # Kill any existing calibration greenlet
        if calibration_greenlet is not None and not calibration_greenlet.dead:
            calibration_greenlet.kill()
        session_state = "CALIBRATING_COUNTDOWN"
        calib_frames = []
        reset_session_stats()
        calibration_greenlet = gevent.spawn(_calibration_greenlet)
    return jsonify({"status": "started"})


@app.route("/stop", methods=["POST"])
def stop():
    global session_state, calibration_greenlet
    with state_lock:
        session_state = "IDLE"
        if calibration_greenlet is not None and not calibration_greenlet.dead:
            calibration_greenlet.kill()
            calibration_greenlet = None
    return jsonify({"status": "stopped"})


@app.route("/report")
def report():
    with state_lock:
        total = max(session_stats["total_seconds"], 0.001)
        focused_pct = round((session_stats["focused_seconds"] / total) * 100, 1)
        distracted_pct = round((session_stats["distracted_seconds"] / total) * 100, 1)
        samples = session_stats["score_samples"]
        avg_score = round(sum(samples) / len(samples), 1) if samples else 0

        duration_td = timedelta(seconds=int(total))
        duration_str = str(duration_td)
        if len(duration_str) == 7:  # H:MM:SS -> HH:MM:SS
            duration_str = "0" + duration_str

        peak_str = "N/A"
        if session_stats["peak_focus_start"] and session_stats["peak_focus_duration"] > 0:
            start_t = session_stats["peak_focus_start"].strftime("%H:%M:%S")
            end_t = session_stats["peak_focus_end"].strftime("%H:%M:%S")
            mins = round(session_stats["peak_focus_duration"] / 60, 1)
            peak_str = f"{start_t} - {end_t} ({mins} min)"

        recommendations = []
        if session_stats["posture_alerts"] > 3:
            recommendations.append("Take breaks to stretch your back and reset your posture.")
        if session_stats["drowsiness_events"] > 1:
            recommendations.append("Consider a short nap or a coffee break before your next session.")
        if session_stats["phone_detections"] > 2:
            recommendations.append("Keep your phone face-down or in another room while studying.")
        if focused_pct > 80:
            recommendations.append("Excellent focus! Keep up the great work.")
        recommendations.append("Try the Pomodoro technique: 25 min focus, 5 min break.")

        data = {
            "duration": duration_str,
            "avg_attention_score": avg_score,
            "focused_percent": focused_pct,
            "distracted_percent": distracted_pct,
            "phone_detections": session_stats["phone_detections"],
            "posture_alerts": session_stats["posture_alerts"],
            "drowsiness_events": session_stats["drowsiness_events"],
            "peak_focus_period": peak_str,
            "recommendations": recommendations,
        }
    return jsonify(data)


if __name__ == "__main__":
    print("=" * 60)
    print(" StudyGuard AI starting at http://localhost:5000")
    print("=" * 60)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
