# Driver Drowsiness Detection

A Python computer-vision project that uses facial landmarks and eye-related signals to estimate signs of driver drowsiness through a webcam. This is an educational prototype, not a certified vehicle-safety system.

## Technologies

- Python
- OpenCV
- dlib
- NumPy and SciPy
- Tkinter for the desktop interface

## How it works

1. Capture frames from a webcam.
2. Detect facial landmarks.
3. Calculate eye-related measurements from the detected landmarks.
4. Track the measurements over time and trigger an alert when the configured drowsiness condition is met.

The result depends on camera position, lighting, landmark detection quality, and the thresholds used by the implementation.

## Requirements

- Python installed on your system
- A working webcam
- The Python packages listed in the project's imports / dependency file
- The dlib facial-landmark model file expected by the code

## Run locally

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install the project dependencies. If a `requirements.txt` file is present, run:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Download or place the facial-landmark model file required by the code in the configured location. Follow the model provider's license and terms.
5. Run the main Python script identified in the repository.
6. Allow webcam access and test in a well-lit environment.

If the project does not yet include a dependency file or model file, add the exact files and setup steps before describing the application as reproducible.

## Limitations and safety

- This prototype may miss drowsiness or produce false alarms.
- Glasses, shadows, camera placement, and individual differences can affect landmark detection.
- Do not use it as a replacement for adequate rest, safe driving practices, or a certified driver-monitoring system.
- Do not test it while driving on public roads.

## Improvements to demonstrate

- Add sample screenshots or a short demo video.
- Document the eye-aspect-ratio or other threshold formula actually used in the source code.
- Add tests for landmark calculations and alert-state logic.
- Measure false-positive and false-negative rates on a clearly described test set.
