"""
Real-time face analyzer — eyes, smile, all 7 emotions, age, gender.

Usage:
    python eye_detector.py
    python eye_detector.py --source video.mp4
    python eye_detector.py --no-deepface
"""

import argparse
import os
import queue
import ssl
import subprocess
import threading
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

MODEL_PATH = "face_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

EMOTION_COLORS = {
    "Happy":     (0,  210,  80),
    "Surprised": (0,  210, 210),
    "Angry":     (30,  50, 230),
    "Sad":       (170, 100, 230),
    "Fear":      (0,  140, 255),
    "Disgust":   (0,  180, 130),
    "Neutral":   (180, 180, 180),
}

_df_cache: dict = {}
_df_lock = threading.Lock()


# ── model download ──────────────────────────────────────────────────────────

def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading face landmarker model (~5 MB)...")
    try:
        subprocess.run(["curl", "-L", "-o", MODEL_PATH, MODEL_URL],
                       check=True, capture_output=True)
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(MODEL_URL, context=ctx) as r, \
             open(MODEL_PATH, "wb") as f:
            f.write(r.read())
    print("Model ready.")


# ── EAR ────────────────────────────────────────────────────────────────────

def eye_aspect_ratio(lm, indices, w, h):
    pts = np.array([(lm[i].x * w, lm[i].y * h) for i in indices], np.float32)
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def draw_eye_contour(frame, lm, indices, w, h, color):
    pts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in indices], np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)


# ── blendshape helpers ──────────────────────────────────────────────────────

def bsmap(blendshapes):
    return {b.category_name: b.score for b in blendshapes}


def detect_emotion(bs):
    """Score all 7 emotions from blendshapes, return the dominant one."""
    smile     = (bs.get("mouthSmileLeft", 0)    + bs.get("mouthSmileRight", 0))    / 2
    frown     = (bs.get("mouthFrownLeft", 0)    + bs.get("mouthFrownRight", 0))    / 2
    brow_dn   = (bs.get("browDownLeft", 0)      + bs.get("browDownRight", 0))      / 2
    brow_up   =  bs.get("browInnerUp", 0)
    eye_wide  = (bs.get("eyeWideLeft", 0)       + bs.get("eyeWideRight", 0))       / 2
    jaw_open  =  bs.get("jawOpen", 0)
    sneer     = (bs.get("noseSneerLeft", 0)     + bs.get("noseSneerRight", 0))     / 2
    cheek_sq  = (bs.get("cheekSquintLeft", 0)   + bs.get("cheekSquintRight", 0))   / 2
    stretch   = (bs.get("mouthStretchLeft", 0)  + bs.get("mouthStretchRight", 0))  / 2
    lip_up    = (bs.get("mouthUpperUpLeft", 0)  + bs.get("mouthUpperUpRight", 0))  / 2
    press     = (bs.get("mouthPressLeft", 0)    + bs.get("mouthPressRight", 0))    / 2

    scores = {
        "Happy":     smile * 2.0 + cheek_sq * 0.5,
        "Surprised": jaw_open * 1.5 + eye_wide * 1.0 + brow_up * 0.5,
        "Angry":     brow_dn * 1.5 + press * 0.5 + sneer * 0.3,
        "Sad":       frown * 1.5 + brow_up * 0.4,
        "Fear":      eye_wide * 1.2 + brow_up * 0.8 + stretch * 0.5,
        "Disgust":   sneer * 2.0 + lip_up * 0.8,
        "Neutral":   0.25,  # baseline wins only when nothing else fires
    }

    best = max(scores, key=scores.get)
    # suppress weak detections back to Neutral
    if best != "Neutral" and scores[best] < 0.30:
        best = "Neutral"
    return best, EMOTION_COLORS[best]


def face_bbox(lm, w, h, pad=0.18):
    xs = [p.x * w for p in lm]
    ys = [p.y * h for p in lm]
    pw = (max(xs) - min(xs)) * pad
    ph = (max(ys) - min(ys)) * pad
    return (max(0, int(min(xs) - pw)), max(0, int(min(ys) - ph)),
            min(w, int(max(xs) + pw)), min(h, int(max(ys) + ph)))


# ── DeepFace background thread ──────────────────────────────────────────────

def deepface_worker(job_queue):
    try:
        from deepface import DeepFace
        print("[DeepFace] loaded — age / gender active")
    except ImportError:
        print("[DeepFace] not installed. Run: pip install deepface")
        return

    while True:
        item = job_queue.get()
        if item is None:
            break
        face_img, face_idx = item
        try:
            res = DeepFace.analyze(
                face_img,
                actions=["age", "gender", "emotion"],
                enforce_detection=False,
                silent=True,
            )
            r = res[0] if isinstance(res, list) else res

            # Handle both old ("Male"/"Female") and new ("Man"/"Woman") API
            gender_raw = r.get("dominant_gender", r.get("gender", "?"))
            if isinstance(gender_raw, dict):
                gender_raw = max(gender_raw, key=gender_raw.get)
            gender_map = {"Man": "Male", "Woman": "Female",
                          "Male": "Male", "Female": "Female"}
            gender = gender_map.get(str(gender_raw), str(gender_raw))

            with _df_lock:
                _df_cache[face_idx] = {
                    "age":     int(r.get("age", 0)),
                    "gender":  gender,
                    "emotion": r.get("dominant_emotion", "?").capitalize(),
                }
        except Exception as e:
            print(f"[DeepFace error] {e}")


# ── text rendering helpers ──────────────────────────────────────────────────

def put_label(frame, text, x, y, color, scale=0.52, thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_panel(frame, x1, y1, x2, y2, labels):
    """Render a semi-transparent label panel above the face box."""
    line_h  = 22
    pad     = 6
    panel_h = len(labels) * line_h + pad * 2
    panel_w = 210
    px1     = x1
    px2     = min(frame.shape[1], x1 + panel_w)
    py2     = max(0, y1 - 4)
    py1     = max(0, py2 - panel_h)

    if py1 < py2 and px1 < px2:
        roi = frame[py1:py2, px1:px2]
        bg  = np.zeros_like(roi)
        cv2.addWeighted(bg, 0.55, roi, 0.45, 0, roi)
        frame[py1:py2, px1:px2] = roi

    for i, (text, color) in enumerate(labels):
        put_label(frame, text, px1 + pad, py1 + pad + (i + 1) * line_h - 4, color)


# ── main loop ───────────────────────────────────────────────────────────────

def run(source=0, use_deepface=True):
    download_model()

    job_queue = queue.Queue(maxsize=4)
    if use_deepface:
        threading.Thread(target=deepface_worker, args=(job_queue,), daemon=True).start()

    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=4,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = mp_vision.FaceLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    ear_threshold = 0.15   # works for most eyes; tune with + / - keys
    frame_n       = 0

    print("Running — press +/- to adjust EAR threshold, Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if source == 0:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result  = detector.detect(mp_img)
        frame_n += 1

        for fi, face_lm in enumerate(result.face_landmarks or []):
            blendshapes = (result.face_blendshapes or [[]])[fi] \
                          if result.face_blendshapes else []
            bs = bsmap(blendshapes)

            # ── eyes ──
            l_ear = eye_aspect_ratio(face_lm, LEFT_EYE,  w, h)
            r_ear = eye_aspect_ratio(face_lm, RIGHT_EYE, w, h)
            avg   = (l_ear + r_ear) / 2.0

            eye_open  = avg >= ear_threshold
            eye_color = (0, 215, 0) if eye_open else (30, 30, 230)
            eye_label = "Open" if eye_open else "Closed"

            draw_eye_contour(frame, face_lm, LEFT_EYE,  w, h, eye_color)
            draw_eye_contour(frame, face_lm, RIGHT_EYE, w, h, eye_color)

            # ── smile ──
            smile_score = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
            is_smiling  = smile_score > 0.30

            # ── blendshape emotion ──
            bs_emotion, bs_color = detect_emotion(bs)

            # ── face box + DeepFace feed ──
            x1, y1, x2, y2 = face_bbox(face_lm, w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (110, 110, 110), 1)

            if use_deepface and frame_n % 25 == 0:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    try:
                        job_queue.put_nowait((crop.copy(), fi))
                    except queue.Full:
                        pass

            # ── pull DeepFace result ──
            with _df_lock:
                df = _df_cache.get(fi, {})

            age        = df.get("age",     "...")
            gender     = df.get("gender",  "...")
            df_emotion = df.get("emotion", "")

            # Prefer DeepFace emotion when available (more classes)
            emotion       = df_emotion if df_emotion and df_emotion not in ("...", "?", "") \
                            else bs_emotion
            emotion_color = EMOTION_COLORS.get(emotion, bs_color)

            # ── label panel ──
            labels = [
                (f"Eyes    : {eye_label}  ({avg:.2f})",
                 eye_color),
                (f"Smile   : {'Yes :)' if is_smiling else 'No'}",
                 (0, 200, 90) if is_smiling else (160, 160, 160)),
                (f"Emotion : {emotion}",
                 emotion_color),
                (f"Gender  : {gender}",
                 (200, 140, 255)),
                (f"Age     : {age}",
                 (255, 190, 60)),
            ]
            draw_panel(frame, x1, y1, x2, y2, labels)

        # ── status bar ──
        put_label(frame, f"thresh={ear_threshold:.2f}  +/- to adjust  Q=quit",
                  8, h - 10, (80, 80, 80), scale=0.42)

        cv2.imshow("Face Analyzer", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("+") or key == ord("="):
            ear_threshold = round(min(ear_threshold + 0.01, 0.40), 2)
            print(f"EAR threshold -> {ear_threshold:.2f}")
        elif key == ord("-"):
            ear_threshold = round(max(ear_threshold - 0.01, 0.05), 2)
            print(f"EAR threshold -> {ear_threshold:.2f}")

    job_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",      default=0)
    parser.add_argument("--no-deepface", action="store_true")
    args = parser.parse_args()

    src = args.source
    try:
        src = int(src)
    except (ValueError, TypeError):
        pass

    run(source=src, use_deepface=not args.no_deepface)
