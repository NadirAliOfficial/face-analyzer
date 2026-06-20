"""
Helper script to collect your own eye training data using your webcam.

Usage:
    python collect_data.py --label open   --count 500 --split train
    python collect_data.py --label closed --count 500 --split train
    python collect_data.py --label open   --count 100 --split val
    python collect_data.py --label closed --count 100 --split val

Saves cropped eye images to data/<split>/<label>/
"""

import argparse
import os

import cv2
import mediapipe as mp

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
PADDING = 0.4


def crop_eye(frame, landmarks, indices, w, h):
    import numpy as np
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
    x2 = min(w, int(x_max + pad_x))
    y2 = min(h, int(y_max + pad_y))
    return frame[y1:y2, x1:x2]


def collect(label, count, split, out_dir):
    save_dir = os.path.join(out_dir, split, label)
    os.makedirs(save_dir, exist_ok=True)

    existing = len(os.listdir(save_dir))
    print(f"Saving to {save_dir}  (already have {existing})")
    print("Press SPACE to capture, Q to quit.")

    cap = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    captured = 0

    while captured < count:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        preview = frame.copy()
        cv2.putText(preview, f"Label: {label} | {captured}/{count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        cv2.putText(preview, "SPACE=capture  Q=quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            for indices in [LEFT_EYE_IDX, RIGHT_EYE_IDX]:
                import numpy as np
                pts = np.array(
                    [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices], dtype=np.int32,
                )
                cv2.polylines(preview, [pts], True, (0, 220, 0), 1)

        cv2.imshow("Data Collector", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            for side, indices in [("L", LEFT_EYE_IDX), ("R", RIGHT_EYE_IDX)]:
                crop = crop_eye(frame, lm, indices, w, h)
                if crop.size == 0:
                    continue
                fname = os.path.join(save_dir, f"{existing + captured:05d}_{side}.jpg")
                cv2.imwrite(fname, crop)
            captured += 1
            print(f"  Captured {captured}/{count}", end="\r")
        elif key == ord("q"):
            break

    print(f"\nDone. Collected {captured} samples.")
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=["open", "closed"])
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--out",   default="data")
    args = parser.parse_args()
    collect(args.label, args.count, args.split, args.out)
