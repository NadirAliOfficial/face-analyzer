# Face Analyzer

Real-time face analysis using MediaPipe and DeepFace. Detects eye state, smile, emotions, age, and gender from a webcam feed.

## Features

- **Eye detection** — open / closed using Eye Aspect Ratio (EAR)
- **Smile detection** — via facial blendshapes
- **Emotion recognition** — Happy, Sad, Angry, Surprised, Fear, Disgust, Neutral
- **Age estimation** — DeepFace model, updates every second
- **Gender detection** — Male / Female via DeepFace
- Supports up to 4 faces simultaneously
- Press `+` / `-` to tune the EAR threshold live
- Press `Q` to quit

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Webcam (default)
python eye_detector.py

# Video file
python eye_detector.py --source video.mp4

# Skip age/gender (faster, no DeepFace)
python eye_detector.py --no-deepface
```

## Train Your Own CNN

```bash
# Collect eye images from webcam
python collect_data.py --label open   --count 500 --split train
python collect_data.py --label closed --count 500 --split train
python collect_data.py --label open   --count 100 --split val
python collect_data.py --label closed --count 100 --split val

# Train
python train_cnn.py --data data/ --epochs 20

# Run with CNN model
python cnn_detector.py --model eye_model.pth
```

## How It Works

| Feature | Method |
|---|---|
| Eye open/closed | MediaPipe Face Mesh landmarks + EAR formula |
| Smile | MediaPipe blendshape scores |
| Emotion (fast) | Weighted blendshape scoring across 7 emotions |
| Emotion / Age / Gender | DeepFace (background thread, every 25 frames) |

## Requirements

- Python 3.10+
- Webcam with macOS camera permission granted
- ~1.1 GB disk for DeepFace models (downloaded on first run)

