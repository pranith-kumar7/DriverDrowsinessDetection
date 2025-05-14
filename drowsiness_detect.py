import cv2
import dlib
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from imutils import face_utils
from datetime import datetime
from scipy.spatial import distance as dist
import pyttsx3
import time

# Initialize TTS engine once
engine = pyttsx3.init()

# ------------------------ Detection Logic ------------------------

def speak_alert(message):
    print(f"[Voice Alert]: {message}")
    engine.say(message)
    engine.runAndWait()

def log_alert(alert_type):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("alert_log.txt", "a") as f:
        f.write(f"[{timestamp}] {alert_type} Alert Triggered\n")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  # 62–66
    B = dist.euclidean(mouth[3], mouth[9])   # 63–65
    C = dist.euclidean(mouth[4], mouth[8])   # 64–64
    D = dist.euclidean(mouth[0], mouth[6])   # 60–64
    return (A + B + C) / (3.0 * D)

# ------------------------ GUI App ------------------------

class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Detection")
        self.current_theme = "light"

        self.themes = {
            "light": {"bg": "#FFFFFF", "fg": "#000000", "btn_bg": "#E0E0E0", "alert_fg": "red"},
            "dark": {"bg": "#1e1e1e", "fg": "#FFFFFF", "btn_bg": "#333333", "alert_fg": "orange"}
        }

        self.setup_widgets()
        self.apply_theme()

    def setup_widgets(self):
        self.label = tk.Label(self.root, text="Driver Drowsiness Detection", font=("Arial", 16))
        self.label.pack(pady=5)

        self.video_frame = tk.Label(self.root)
        self.video_frame.pack()

        self.alert_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.alert_label.pack(pady=5)

        self.start_button = tk.Button(self.root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=5)

        self.log_button = tk.Button(self.root, text="View Alert Log", command=self.view_log)
        self.log_button.pack(pady=5)

        self.toggle_button = tk.Button(self.root, text="Toggle Theme", command=self.toggle_theme)
        self.toggle_button.pack(pady=5)

    def apply_theme(self):
        theme = self.themes[self.current_theme]
        self.root.configure(bg=theme["bg"])
        self.label.config(bg=theme["bg"], fg=theme["fg"])
        self.alert_label.config(bg=theme["bg"], fg=theme["alert_fg"])
        self.video_frame.config(bg=theme["bg"])
        self.start_button.config(bg=theme["btn_bg"], fg=theme["fg"])
        self.log_button.config(bg=theme["btn_bg"], fg=theme["fg"])
        self.toggle_button.config(bg=theme["btn_bg"], fg=theme["fg"])

    def toggle_theme(self):
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        self.apply_theme()

    def view_log(self):
        try:
            with open("alert_log.txt", "r") as f:
                logs = f.read()
            messagebox.showinfo("Alert Log", logs if logs else "No alerts logged yet.")
        except FileNotFoundError:
            messagebox.showinfo("Alert Log", "No log file found.")

    def start_detection(self):
        threading.Thread(target=self.detect_drowsiness).start()

    def detect_drowsiness(self):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        cap = cv2.VideoCapture(0)

        EAR_THRESHOLD = 0.25
        MAR_THRESHOLD = 0.55
        CONSEC_FRAMES = 20
        YAWN_FRAMES = 15
        ALERT_COOLDOWN = 5  # seconds

        frame_counter = 0
        yawn_counter = 0
        last_alert_time = time.time()

        def clear_alert():
            self.alert_label.config(text="")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            current_time = time.time()

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                mouth = shape[48:68]  # full mouth

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                mar = mouth_aspect_ratio(mouth)

                print(f"EAR: {ear:.2f}, MAR: {mar:.2f}")

                # Drowsiness Detection
                if ear < EAR_THRESHOLD:
                    frame_counter += 1
                    if frame_counter >= CONSEC_FRAMES and (current_time - last_alert_time > ALERT_COOLDOWN):
                        last_alert_time = current_time
                        log_alert("Drowsiness")
                        threading.Thread(target=speak_alert, args=("You seem drowsy. Please stay alert.",), daemon=True).start()
                        self.alert_label.config(text="DROWSINESS ALERT!")
                        self.root.after(3000, clear_alert)
                else:
                    frame_counter = 0

                # Yawning Detection
                if mar > MAR_THRESHOLD:
                    yawn_counter += 1
                    if yawn_counter >= YAWN_FRAMES and (current_time - last_alert_time > ALERT_COOLDOWN):
                        last_alert_time = current_time
                        log_alert("Yawning")
                        threading.Thread(target=speak_alert, args=("You are yawning. Please take a break.",), daemon=True).start()
                        self.alert_label.config(text="YAWNING ALERT!")
                        self.root.after(3000, clear_alert)
                else:
                    yawn_counter = 0

            # Display video feed in GUI
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

            if cv2.waitKey(1) == 27:  # ESC to exit
                break

        cap.release()
        cv2.destroyAllWindows()

# ------------------------ Run the App ------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessApp(root)
    root.mainloop()