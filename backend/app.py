import json
import os
import re
import secrets
import urllib.request
import time
import base64
import threading
from datetime import datetime, timedelta

import numpy as np
import cv2
import torch
import joblib

import bcrypt
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    get_jwt_identity,
    jwt_required,
    decode_token,
)
from flask_socketio import SocketIO, emit, join_room, leave_room
from mysql.connector import IntegrityError

from db import get_db, close_db

# ML Imports & Initialization (Takes time on startup)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

print("Loading ML Models. This may take a moment...")
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate, Multiply, Dropout, Lambda
import tensorflow as tf
from transformers import HubertModel, Wav2Vec2Processor

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
SAMPLE_RATE = 16000
DURATION = 3 # 3-second audio buffer for inferences
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/hubert-large-ls960-ft"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
hubert_model = HubertModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
hubert_model.eval()

voice_scaler = joblib.load(r"d:\S6 Mini Project\HuBERT Model\scaler_hubert_large.pkl")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

base_face = load_model(r"d:\S6 Mini Project\Facial Detection\emotion_model.keras")
for layer in base_face.layers: layer.trainable = False
base_voice = load_model(r"d:\S6 Mini Project\HuBERT Model\voice_emotion_detection_hubert_large.keras")
for layer in base_voice.layers: layer.trainable = False

face_extractor = Model(inputs=base_face.inputs, outputs=base_face.layers[-2].output)
voice_extractor = Model(inputs=base_voice.inputs, outputs=base_voice.layers[-2].output)
input_face = Input(shape=(48, 48, 1))
input_voice = Input(shape=(1024,))
feat_face = face_extractor(input_face)
feat_voice = voice_extractor(input_voice)
proj_face = Dense(256, activation='relu')(feat_face)
proj_voice = Dense(256, activation='relu')(feat_voice)
joint = Concatenate()([proj_face, proj_voice])
att_weights = Dense(128, activation='relu')(joint)
att_weights = Dense(2, activation='softmax')(att_weights)
weight_face = Lambda(lambda x: tf.expand_dims(x[:, 0], axis=-1))(att_weights)
weight_voice = Lambda(lambda x: tf.expand_dims(x[:, 1], axis=-1))(att_weights)
attended_face = Multiply()([proj_face, weight_face])
attended_voice = Multiply()([proj_voice, weight_voice])
fused = Concatenate()([attended_face, attended_voice])
x = Dropout(0.4)(fused)
x = Dense(128, activation='relu')(x)
final_output = Dense(7, activation='softmax')(x)
fused_model = Model(inputs=[input_face, input_voice], outputs=final_output)
fused_model.load_weights(r"d:\S6 Mini Project\attention_fused_emotion_model.keras")

# Global dict to store processing stream states per WS sid
active_sessions = {}

# ── Noise-Robust Audio Helpers ────────────────────────────────────────────────

def estimate_noise_floor(audio_clip, noise_percentile=15):
    """Estimate background noise floor from the lower percentile of frame energies."""
    frame_size = 512
    hop_size = 256
    frames = []
    for start in range(0, len(audio_clip) - frame_size, hop_size):
        frame = audio_clip[start:start + frame_size]
        frames.append(float(np.sqrt(np.mean(frame ** 2))))
    if not frames:
        return 0.0
    return float(np.percentile(frames, noise_percentile))


def compute_snr(audio_clip, noise_floor):
    """Estimate SNR in dB given a noise floor measurement."""
    signal_rms = float(np.sqrt(np.mean(audio_clip ** 2)))
    if noise_floor < 1e-9:
        return 60.0  # Essentially clean signal
    return float(20 * np.log10(max(signal_rms, 1e-9) / max(noise_floor, 1e-9)))



def save_timeline_graph(timeline_probs):
    """Render latest session timeline graph to project root for analytics page."""
    if not timeline_probs:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Timeline Graph] Matplotlib unavailable: {e}")
        return

    try:
        all_probs = np.array(timeline_probs, dtype=float)
        if all_probs.ndim != 2 or all_probs.shape[1] != len(EMOTIONS):
            return

        # X-axis uses per-second inference cadence from session loop.
        times = np.arange(all_probs.shape[0], dtype=float)

        plt.figure(figsize=(12, 4.8))
        for idx, emotion in enumerate(EMOTIONS):
            plt.plot(times, all_probs[:, idx], label=emotion.capitalize(), linewidth=1.8)

        plt.title("Emotion Timeline")
        plt.xlabel("Time (s)")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        plt.grid(alpha=0.25, linestyle="--", linewidth=0.6)
        plt.legend(ncol=4, fontsize=8, frameon=False)
        plt.tight_layout()

        graph_path = os.path.join(PROJECT_ROOT, "timeline_graph.png")
        plt.savefig(graph_path, dpi=180)
        plt.close()
        print(f"[Timeline Graph] Updated: {graph_path}")
    except Exception as e:
        print(f"[Timeline Graph] Failed to save graph: {e}")


# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
# Upgrading fallback secret to 32 bytes to avoid PyJWT exception: InsecureKeyError
app.config["JWT_SECRET_KEY"] = os.getenv("SECRET_KEY", "fallback-dev-secret-must-be-32-bytes-long")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)

CORS(app)
jwt = JWTManager(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Brevo REST API config (uses API key directly — no separate SMTP key needed)
BREVO_API_KEY    = os.getenv("BREVO_API_KEY", "")
BREVO_FROM_EMAIL = os.getenv("BREVO_SENDER_EMAIL", "")
BREVO_FROM_NAME  = os.getenv("BREVO_SENDER_NAME", "EmoSense")
BREVO_SEND_URL   = "https://api.brevo.com/v3/smtp/email"


# ── Helpers ───────────────────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def generate_otp() -> str:
    """Return a 6-digit numeric OTP string."""
    return str(secrets.randbelow(900000) + 100000)


def send_otp_email(to_email: str, to_name: str, otp_code: str) -> None:
    """Send a styled OTP email via Brevo Transactional Email REST API."""
    html_text = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width"></head>
<body style="margin:0;padding:0;background:#0c0e14;font-family:'Segoe UI',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0c0e14;padding:40px 20px;">
    <tr><td align="center">
      <table width="480" cellpadding="0" cellspacing="0"
             style="background:#11141e;border:1px solid rgba(99,179,237,0.15);border-radius:8px;overflow:hidden;">
        <tr>
          <td style="background:linear-gradient(135deg,#1a1f2e,#0f1318);padding:28px 32px;
                     border-bottom:1px solid rgba(99,179,237,0.1);">
            <p style="margin:0;font-size:11px;letter-spacing:0.2em;color:rgba(99,179,237,0.6);
                       text-transform:uppercase;font-family:monospace;">EmoSense &middot; v3.2.1</p>
            <h1 style="margin:8px 0 0;font-size:22px;font-weight:800;color:#e2e8f0;
                        letter-spacing:-0.02em;">Verify your identity</h1>
          </td>
        </tr>
        <tr>
          <td style="padding:32px;">
            <p style="margin:0 0 8px;font-size:14px;color:#a0aec0;line-height:1.7;">
              Hi <strong style="color:#e2e8f0;">{to_name}</strong>,
            </p>
            <p style="margin:0 0 28px;font-size:14px;color:#a0aec0;line-height:1.7;">
              Use the code below to complete your EmoSense registration.
              It expires in <strong style="color:#e2e8f0;">10 minutes</strong>.
            </p>
            <div style="background:#0c0e14;border:1px solid rgba(183,148,244,0.35);
                        border-radius:6px;padding:24px 0;text-align:center;margin-bottom:28px;">
              <p style="margin:0 0 6px;font-size:10px;letter-spacing:0.18em;
                         color:rgba(183,148,244,0.6);text-transform:uppercase;font-family:monospace;">
                Verification Code
              </p>
              <span style="font-family:monospace;font-size:38px;font-weight:700;
                            letter-spacing:0.25em;color:#b794f4;">{otp_code}</span>
            </div>
            <p style="margin:0;font-size:12px;color:#718096;line-height:1.7;">
              If you did not attempt to register, please ignore this email.<br>
              Never share this code with anyone.
            </p>
          </td>
        </tr>
        <tr>
          <td style="padding:16px 32px;border-top:1px solid rgba(255,255,255,0.06);">
            <p style="margin:0;font-size:11px;color:#4a5568;">
              &copy; 2025 EmoSense &middot; Multimodal Emotion Detection Platform
            </p>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    plain_text = (
        f"Hi {to_name},\n\n"
        f"Your EmoSense verification code is: {otp_code}\n\n"
        f"It expires in 10 minutes. Do not share it with anyone.\n\n"
        f"-- EmoSense Team"
    )

    payload = json.dumps({
        "sender":      {"name": BREVO_FROM_NAME, "email": BREVO_FROM_EMAIL},
        "to":          [{"email": to_email, "name": to_name}],
        "subject":     f"EmoSense — Your verification code: {otp_code}",
        "htmlContent": html_text,
        "textContent": plain_text,
    }).encode("utf-8")

    req = urllib.request.Request(
        BREVO_SEND_URL,
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Accept":        "application/json",
            "api-key":       BREVO_API_KEY,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        if resp.status not in (200, 201):
            raise RuntimeError(f"Brevo API error {resp.status}: {resp.read()}")




# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "online", "service": "EmoSense API v1.0"}), 200


# ─── SEND OTP ────────────────────────────────────────────────────────────────
@app.route("/send-otp", methods=["POST"])
def send_otp():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    first_name = (data.get("firstName") or "").strip()
    last_name  = (data.get("lastName")  or "").strip()
    email      = (data.get("email")     or "").strip().lower()
    password   = data.get("password")   or ""

    # Validate
    errors = []
    if not first_name:
        errors.append("First name is required.")
    if not last_name:
        errors.append("Last name is required.")
    if not email or not EMAIL_RE.match(email):
        errors.append("A valid email is required.")
    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")
    if errors:
        return jsonify({"error": " ".join(errors)}), 400

    password_hash = hash_password(password)
    otp_code      = generate_otp()
    expires_at    = datetime.utcnow() + timedelta(minutes=10)

    conn = None
    try:
        conn   = get_db()
        cursor = conn.cursor()

        # Upsert the user as pending (is_verified = 0)
        cursor.execute(
            """
            INSERT INTO users (first_name, last_name, email, password_hash, is_verified)
            VALUES (%s, %s, %s, %s, 0)
            ON DUPLICATE KEY UPDATE
                first_name    = VALUES(first_name),
                last_name     = VALUES(last_name),
                password_hash = VALUES(password_hash),
                is_verified   = 0
            """,
            (first_name, last_name, email, password_hash),
        )

        # Invalidate any old unused tokens for this email
        cursor.execute(
            "UPDATE otp_tokens SET used = 1 WHERE email = %s AND used = 0",
            (email,),
        )

        # Insert new token
        cursor.execute(
            "INSERT INTO otp_tokens (email, otp_code, expires_at) VALUES (%s, %s, %s)",
            (email, otp_code, expires_at),
        )
        conn.commit()

        # Send email via Brevo
        send_otp_email(email, first_name, otp_code)

        return jsonify({"message": "Verification code sent to your email."}), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

    finally:
        close_db(conn)


# ─── VERIFY OTP ──────────────────────────────────────────────────────────────
@app.route("/verify-otp", methods=["POST"])
def verify_otp():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    email    = (data.get("email") or "").strip().lower()
    otp_code = (data.get("otp")   or "").strip()

    if not email or not otp_code:
        return jsonify({"error": "Email and OTP are required."}), 400

    conn = None
    try:
        conn   = get_db()
        cursor = conn.cursor(dictionary=True)

        # Find the latest valid token
        cursor.execute(
            """
            SELECT * FROM otp_tokens
            WHERE email = %s AND used = 0 AND expires_at > UTC_TIMESTAMP()
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (email,),
        )
        token = cursor.fetchone()

        if not token or token["otp_code"] != otp_code:
            return jsonify({"error": "Invalid or expired verification code."}), 400

        # Mark token as used
        cursor.execute(
            "UPDATE otp_tokens SET used = 1 WHERE id = %s",
            (token["id"],),
        )

        # Mark user as verified
        cursor.execute(
            "UPDATE users SET is_verified = 1 WHERE email = %s",
            (email,),
        )
        conn.commit()

        # Fetch user for JWT
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        jwt_token = create_access_token(
            identity=str(user["id"]),
            additional_claims={
                "firstName": user["first_name"],
                "lastName":  user["last_name"],
                "email":     user["email"],
            },
        )

        return jsonify({
            "message": "Account verified successfully.",
            "token": jwt_token,
            "user": {
                "id":        user["id"],
                "firstName": user["first_name"],
                "lastName":  user["last_name"],
                "email":     user["email"],
            },
        }), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

    finally:
        close_db(conn)


# ─── LOGIN ────────────────────────────────────────────────────────────────────
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    email    = (data.get("email")    or "").strip().lower()
    password = data.get("password")  or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    conn = None
    try:
        conn   = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if not user or not check_password(password, user["password_hash"]):
            return jsonify({"error": "Invalid email or password."}), 401

        if not user.get("is_verified"):
            return jsonify({"error": "Please verify your email before logging in."}), 403

        token = create_access_token(
            identity=str(user["id"]),
            additional_claims={
                "firstName": user["first_name"],
                "lastName":  user["last_name"],
                "email":     user["email"],
            },
        )

        return jsonify({
            "message":   "Login successful.",
            "token":     token,
            "user": {
                "id":        user["id"],
                "firstName": user["first_name"],
                "lastName":  user["last_name"],
                "email":     user["email"],
            },
        }), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

    finally:
        close_db(conn)


# ─── PROFILE (protected) ─────────────────────────────────────────────────────
@app.route("/profile", methods=["GET"])
@jwt_required()
def profile():
    conn = None
    try:
        user_id = get_jwt_identity()
        conn    = get_db()
        cursor  = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, first_name, last_name, email, created_at FROM users WHERE id = %s",
            (user_id,),
        )
        user = cursor.fetchone()

        if not user:
            return jsonify({"error": "User not found."}), 404

        if user.get("created_at"):
            user["created_at"] = user["created_at"].isoformat()

        return jsonify({
            "id":        user["id"],
            "firstName": user["first_name"],
            "lastName":  user["last_name"],
            "email":     user["email"],
            "createdAt": user.get("created_at"),
        }), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

    finally:
        close_db(conn)


# ─── ANALYTICS & SESSIONS (Phase 6) ────────────────────────────────────────────

@app.route("/api/sessions", methods=["GET"])
@jwt_required()
def get_sessions():
    user_id = get_jwt_identity()
    conn = None
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, duration_seconds, dominant_emotion, created_at FROM sessions WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
        sessions = cursor.fetchall()
        for session in sessions:
            if session.get("created_at"):
                session["created_at"] = session["created_at"].isoformat()
        return jsonify(sessions), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn: close_db(conn)

@app.route("/api/analytics", methods=["GET"])
@jwt_required()
def get_analytics():
    user_id = get_jwt_identity()
    conn = None
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        # 1. Latest Session Breakdown for Pie Chart
        dist_path = os.path.join(PROJECT_ROOT, "latest_distribution.json")
        distribution = []
        if os.path.exists(dist_path):
            with open(dist_path, "r") as f:
                last_dist = json.load(f)
                labels = last_dist.get("labels", [])
                probs = last_dist.get("data", [])
                for lbl, prb in zip(labels, probs):
                    percentage = round(prb * 100, 1)
                    if percentage >= 1.0: # Filter out tiny noise
                        distribution.append({
                            "dominant_emotion": lbl,
                            "count": percentage
                        })
        else:
            # Fallback to historical count if no session run yet
            cursor.execute(
                "SELECT dominant_emotion, COUNT(*) as count FROM sessions WHERE user_id = %s GROUP BY dominant_emotion",
                (user_id,)
            )
            distribution = cursor.fetchall()
            
        # 2. Time series (historical trends)
        cursor.execute(
            """
            SELECT DATE(created_at) as date, COUNT(*) as count 
            FROM sessions 
            WHERE user_id = %s AND created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) 
            GROUP BY DATE(created_at) ORDER BY DATE(created_at) ASC
            """,
            (user_id,)
        )
        time_series = cursor.fetchall()
        for t in time_series:
            t["date"] = t["date"].isoformat() if hasattr(t["date"], "isoformat") else str(t["date"])
            
        return jsonify({"distribution": distribution, "time_series": time_series}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn: close_db(conn)


# ─── TIMELINE GRAPH ENDPOINT ─────────────────────────────────────────────────

def _timeline_graph_route():
    graph_path = os.path.join(PROJECT_ROOT, "timeline_graph.png")
    if not os.path.isfile(graph_path):
        return jsonify({"error": "Timeline graph not yet generated. Run a session first."}), 404
    # Disable all caching so the browser always gets the freshest PNG
    response = send_file(graph_path, mimetype="image/png")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Register with jwt_required wrapper (authenticated users only)
from flask_jwt_extended import jwt_required as _jwt_required

@jwt_required()
def timeline_graph_protected():
    return _timeline_graph_route()

# Also expose a public variant (no auth) for easy embedding
def timeline_graph_public():
    return _timeline_graph_route()

try:
    # Flask 2.x+
    app.add_url_rule("/api/timeline-graph", "timeline_graph", timeline_graph_public, methods=["GET"])
except Exception:
    pass


# ─── WEBSOCKETS (Phase 5 Data Streaming) ──────────────────────────────────────

def extract_features_and_predict(sid):
    session = active_sessions.get(sid)
    if not session or not session.get("running"): return

    try:
        audio_clip = session["audio_buffer"].copy()
        max_amp = float(np.max(np.abs(audio_clip)))

        # 1. Dynamic Noise Floor & SNR Calculation
        noise_floor = estimate_noise_floor(audio_clip, noise_percentile=10)
        snr_db = compute_snr(audio_clip, noise_floor)

        # 2. Determine if it's genuine speech. (SNR > 8.5 dB + amplitude 0.03)
        # Tightened from 4.0 to 8.5 to prevent background fan noise from triggering "Sad" bias.
        is_speech = (snr_db >= 8.5) and (max_amp >= 0.03)

        # 3. Handle background noise gracefully WITHOUT normalizing it!
        if is_speech:
            audio_clip = audio_clip / max_amp

        # 4. Face feature extraction
        is_masked = False
        if session["latest_face_crop"] is not None:
            face_img = session["latest_face_crop"].copy()
        else:
            face_img = np.zeros((48, 48, 1), dtype='float32')
            is_masked = True

        face_input = np.reshape(face_img, (1, 48, 48, 1))

        # 5. Audio feature extraction through HuBERT
        inputs = processor(audio_clip, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = hubert_model(**inputs)

        hidden_states = outputs.hidden_states
        layer_mean = torch.stack(hidden_states[-4:]).mean(dim=0)
        pooled = layer_mean.mean(dim=1).squeeze().cpu().numpy()
        voice_input = voice_scaler.transform([pooled])

        # 6. Run all model predictions
        fused_pred = fused_model.predict([face_input, voice_input], verbose=0)[0]
        fused_idx = np.argmax(fused_pred)

        base_face_pred = base_face.predict(face_input, verbose=0)[0] if not is_masked else np.zeros((7,))
        base_voice_pred = base_voice.predict(voice_input, verbose=0)[0]

        # Remap face model label order → fused model label order
        mapped_pred = np.zeros(7, dtype=float)
        mapped_pred[0] = base_face_pred[0]  # Angry
        mapped_pred[1] = base_face_pred[1]  # Disgust
        mapped_pred[2] = base_face_pred[2]  # Fear
        mapped_pred[3] = base_face_pred[3]  # Happy
        mapped_pred[4] = base_face_pred[6]  # Neutral
        mapped_pred[5] = base_face_pred[4]  # Sad
        mapped_pred[6] = base_face_pred[5]  # Surprise

        # Remap voice model (8 classes: angry, calm, disgust, fear, happy, neutral, sad, surprise) to 7
        mapped_voice = np.zeros(7, dtype=float)
        if len(base_voice_pred) == 8:
            mapped_voice[0] = base_voice_pred[0]  # angry
            mapped_voice[1] = base_voice_pred[2]  # disgust
            mapped_voice[2] = base_voice_pred[3]  # fear
            mapped_voice[3] = base_voice_pred[4]  # happy
            mapped_voice[4] = base_voice_pred[5] + base_voice_pred[1]  # neutral + calm
            mapped_voice[5] = base_voice_pred[6]  # sad
            mapped_voice[6] = base_voice_pred[7]  # surprise
        else:
            # Fallback if it's already 7
            mapped_voice = base_voice_pred[:7]

        # 7. Clean binary routing
        latest_emotion = "Neutral"
        prediction = np.zeros(7)
        prediction[4] = 1.0  # default to Neutral

        if is_speech:
            if is_masked:
                # Voice only
                prediction = mapped_voice
                latest_emotion = EMOTIONS[np.argmax(mapped_voice)].capitalize()
            else:
                # Fused - logic check: if SNR is borderline and Face is strong, fallback to Face
                fused_conf = float(np.max(fused_pred))
                face_conf = float(np.max(mapped_pred))
                
                if fused_conf < 0.35 and face_conf > 0.60:
                    # Fusion is unsure (likely noise interference), Face is very sure. Trust Face.
                    prediction = mapped_pred
                    latest_emotion = EMOTIONS[np.argmax(mapped_pred)].capitalize()
                else:
                    prediction = fused_pred
                    latest_emotion = EMOTIONS[fused_idx].capitalize()
        else:
            if is_masked:
                # No face, no speech -> Neutral
                pass
            else:
                # Face only
                prediction = mapped_pred
                latest_emotion = EMOTIONS[np.argmax(mapped_pred)].capitalize()

        socketio.emit("emotion_update", {
            "emotion": latest_emotion,
            "probabilities": prediction.tolist(),
            "status": "evaluating",
            "debug": {
                "snr_db": round(snr_db, 1),
                "is_speech": is_speech,
                "max_amp": round(max_amp, 3)
            }
        }, to=sid)

        session["timeline"].append(prediction.tolist())

    except Exception as e:
        print(f"[WS Worker Error] Inference failed: {e}")
        import traceback
        traceback.print_exc()

def session_inference_loop(sid):
    """Background loop that runs predictions every ~1 second per socket while running."""
    while True:
        session = active_sessions.get(sid)
        if not session or not session.get("running"):
            break
        extract_features_and_predict(sid)
        time.sleep(1.0) # predict once a second

@socketio.on('start_session')
def handle_start_session(data):
    sid = request.sid
    token = data.get("token")
    if not token:
        emit("session_error", {"error": "Authentication required."})
        return
        
    try:
        decoded = decode_token(token)
        user_id = decoded["sub"]
    except Exception as e:
        emit("session_error", {"error": f"Invalid token: {str(e)}"})
        return

    active_sessions[sid] = {
        "user_id": user_id,
        "running": True,
        "audio_buffer": np.zeros(SAMPLE_RATE * DURATION, dtype=np.float32),
        "latest_face_crop": None,
        "timeline": [],
        "start_time": time.time()
    }
    join_room(sid)
    
    # Start inference loop
    executor = threading.Thread(target=session_inference_loop, args=(sid,))
    executor.daemon = True
    executor.start()
    
    emit("session_started", {"status": "ok"})


@socketio.on('video_frame')
def handle_video_frame(data):
    sid = request.sid
    session = active_sessions.get(sid)
    if not session or not session["running"]: return

    try:
        # data is expected to be a data URL: "data:image/jpeg;base64,....."
        b64 = data.split(',')[1] if ',' in data else data
        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Mirror flip like the original script
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_crop = gray[y:y+h, x:x+w]
            session["latest_face_crop"] = cv2.resize(face_crop, (48, 48)).astype('float32') / 255.0
        else:
            session["latest_face_crop"] = None
    except Exception as e:
        pass # Ignore corrupt frames

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Expects raw float32 PCM data passed as list of floats directly from JS Audio context"""
    sid = request.sid
    session = active_sessions.get(sid)
    if not session or not session["running"]: return
    
    try:
        new_audio = np.array(data, dtype=np.float32)
        shift = len(new_audio)
        if shift > 0 and shift <= len(session["audio_buffer"]):
            session["audio_buffer"] = np.roll(session["audio_buffer"], -shift)
            session["audio_buffer"][-shift:] = new_audio
    except Exception as e:
        pass


@socketio.on('stop_session')
def handle_stop_session(data):
    sid = request.sid
    end_session_internal(sid)

@socketio.on('disconnect')
def handle_disconnect():
    end_session_internal(request.sid)
    
def end_session_internal(sid):
    session = active_sessions.get(sid)
    if not session or not session["running"]: return
    
    session["running"] = False
    duration = int(time.time() - session["start_time"])
    
    # Compute dominant based on averages
    dom_emo = "Unknown"
    
    if session["timeline"]:
        all_probs = np.array(session["timeline"])
        mean_probs = np.mean(all_probs, axis=0)
        max_idx = np.argmax(mean_probs)
        dom_emo = EMOTIONS[max_idx].capitalize()
        
        # Save exact probability distribution of this session for the Analytics pie chart
        try:
            with open(os.path.join(PROJECT_ROOT, "latest_distribution.json"), "w") as f:
                json.dump({
                    "labels": [e.capitalize() for e in EMOTIONS],
                    "data": [float(p) for p in mean_probs]
                }, f)
        except Exception as e:
            print("[Analytics] Failed to save latest distribution:", e)
        
    user_id = session["user_id"]
    
    if duration > 0 and dom_emo != "Unknown":
        conn = None
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (user_id, duration_seconds, dominant_emotion) VALUES (%s, %s, %s)",
                (user_id, duration, dom_emo)
            )
            conn.commit()
        except Exception as e:
            print("[Database Error] Failed to save session:", e)
        finally:
            if conn: close_db(conn)

    # Always attempt to refresh the timeline graph for analytics page.
    save_timeline_graph(session.get("timeline", []))
            
    try:
        socketio.emit("session_ended", {
            "duration": duration,
            "dominant_emotion": dom_emo
        }, to=sid)
    except:
        pass
        
    del active_sessions[sid]


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
