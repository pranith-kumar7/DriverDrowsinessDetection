# Driver Drowsiness Detection

A Python desktop application that uses a webcam, facial landmarks, and eye/mouth measurements to alert when the person in front of the camera shows signs associated with drowsiness or yawning.

**Safety note:** This is an educational prototype, not a certified driver-monitoring system. Do not rely on it to decide whether it is safe to drive, and never test it while driving.

## Features

- Live webcam feed in a Tkinter interface
- Facial landmark detection using dlib's 68-point predictor
- Eye Aspect Ratio (EAR) to detect prolonged eye closure
- Mouth Aspect Ratio (MAR) to detect sustained yawning
- On-screen and spoken alerts
- Timestamped alert log
- Light/dark interface theme

## Tech stack

Python, OpenCV, dlib, NumPy, Pillow, imutils, SciPy, Tkinter, and pyttsx3.

## Requirements

- Python 3.10 or 3.11 is a reasonable starting point; dlib installation depends on your operating system and Python version.
- A webcam
- Tkinter support in your Python installation
- The `shape_predictor_68_face_landmarks.dat` file included in this repository

## Run locally

Clone the repository and open a terminal in its directory.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python drowsiness_detect.py
```

If installing `dlib` fails on Windows, use a compatible Python environment and follow the dlib installation instructions for your platform. Do not install unofficial binary packages from untrusted sources.

Click **Start Detection** and allow webcam access. Press **Esc** in the video window to stop capture. Alert events are appended to `alert_log.txt`.

## How detection works

- The application detects faces and obtains 68 facial landmarks.
- EAR is calculated from vertical and horizontal eye distances. A low EAR over a consecutive-frame threshold triggers a drowsiness alert.
- MAR measures mouth opening. A high MAR over a consecutive-frame threshold triggers a yawning alert.
- A cooldown reduces repeated alerts.

The current thresholds are hand-set values in the source code (`EAR_THRESHOLD`, `MAR_THRESHOLD`, `CONSEC_FRAMES`, and `YAWN_FRAMES`). They have not been established as clinically or scientifically validated thresholds.

## Limitations

Lighting, camera placement, glasses, face angle, and landmark-detection errors can affect results. False positives and missed detections are possible. The application should not be used as a substitute for sleep, rest breaks, or safe-driving decisions.

## Useful next improvements

- Add screenshots and a short demo recorded while stationary.
- Extract EAR/MAR calculations into a module with unit tests.
- Measure false positives and false negatives on a clearly documented test set.
- Move webcam capture and GUI updates into a thread-safe design; Tkinter widgets should be updated on the main thread.
- Document supported Python/OS combinations and model-file licensing.
