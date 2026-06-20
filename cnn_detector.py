"""
Real-time eye detector using the trained CNN model + MediaPipe for eye-region cropping.

Usage:
    python cnn_detector.py --model eye_model.pth
    python cnn_detector.py --model eye_model.pth --source video.mp4
"""

import argparse

import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms

from model import load_model

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

IMG_SIZE = 64
PADDING  = 0.4  # extra margin around eye crop

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def crop_eye(frame, landmarks, indices, w, h):
    pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in indices],
        dtype=np.float32,
    )
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    pad_x = (x_max - x_min) * PADDING
    pad_y = (y_max - y_min) * PADDING

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(w,  int(x_max + pad_x))
    y2 = min(h,  int(y_max + pad_y))

    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def predict(model, crop, device):
    if crop.size == 0:
        return "unknown", 0.0
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    # class 0 = closed, class 1 = open (ImageFolder sorts alphabetically)
    open_prob = probs[1].item()
    return ("OPEN" if open_prob >= 0.5 else "CLOSED"), open_prob


def run(model_path, source=0):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    model, _ = load_model(model_path, device)

    cap = cv2.VideoCapture(source)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("Press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if source == 0:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_lm in results.multi_face_landmarks:
                lm = face_lm.landmark

                for name, indices in [("L", LEFT_EYE_IDX), ("R", RIGHT_EYE_IDX)]:
                    crop, (x1, y1, x2, y2) = crop_eye(frame, lm, indices, w, h)
                    status, conf = predict(model, crop, device)
                    color = (0, 220, 0) if status == "OPEN" else (0, 0, 220)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(
                        frame, f"{name}:{status} {conf:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    )

        cv2.imshow("Eye Detector (CNN)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="Path to eye_model.pth")
    parser.add_argument("--source", default=0,     help="Webcam index or video path")
    args = parser.parse_args()

    src = args.source
    try:
        src = int(src)
    except (ValueError, TypeError):
        pass

    run(args.model, src)
