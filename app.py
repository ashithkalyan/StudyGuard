# === app.py ===
"""
StudyGuard AI - Flask + SocketIO backend.
Run with: python app.py
Then open: http://localhost:5000
"""

from gevent import monkey
monkey.patch_all()

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
state_lock = threading.Lock()
detector = StudyDetector()

session_state = "IDLE"  # IDLE | CALIBRATING_COUNTDOWN | CALIBRATING_CAPTURING | MONITORING
calibration_start_time = 0.0
calib_frames = []
last_countdown_sec = 6
last_emit_time = 0.0
last_stat_tick = 0.0

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

@socketio.on("client_frame")
def handle_client_frame(data):
    global session_state, calibration_start_time, calib_frames, last_countdown_sec, last_emit_time, last_stat_tick
    
    b64_frame = data.get("image")
    if not b64_frame:
        return
        
    frame = _decode_frame(b64_frame)
    if frame is None:
        return
        
    now = time.time()
    
    with state_lock:
        if session_state == "CALIBRATING_COUNTDOWN":
            elapsed = now - calibration_start_time
            seconds_left = 5 - int(elapsed)
            
            if seconds_left <= 0:
                session_state = "CALIBRATING_CAPTURING"
                calib_frames = []
                socketio.emit("calibration_status", {"phase": "capturing", "progress": 0})
            else:
                if seconds_left != last_countdown_sec:
                    last_countdown_sec = seconds_left
                    socketio.emit("calibration_status", {"phase": "countdown", "seconds_left": seconds_left})
                
                # Emit the raw frame during countdown
                _emit_raw_frame(frame)
                
        elif session_state == "CALIBRATING_CAPTURING":
            calib_frames.append(frame.copy())
            progress = int((len(calib_frames) / 30) * 100)
            socketio.emit("calibration_status", {"phase": "capturing", "progress": progress})
            _emit_raw_frame(frame)
            
            if len(calib_frames) >= 30:
                baseline = detector.record_baseline(calib_frames)
                if baseline is None:
                    session_state = "IDLE"
                    socketio.emit("calibration_status", {"phase": "complete", "success": False})
                else:
                    session_state = "MONITORING"
                    socketio.emit("calibration_status", {"phase": "complete", "success": True})
                    last_stat_tick = time.time()
                    last_emit_time = time.time()
                    
        elif session_state == "MONITORING":
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
    global session_state, calibration_start_time, calib_frames, last_countdown_sec, last_stat_tick
    with state_lock:
        session_state = "CALIBRATING_COUNTDOWN"
        calibration_start_time = time.time()
        calib_frames = []
        last_countdown_sec = 6
        reset_session_stats()
    return jsonify({"status": "started"})


@app.route("/stop", methods=["POST"])
def stop():
    global session_state
    with state_lock:
        session_state = "IDLE"
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
