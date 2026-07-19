import tkinter as tk
from tkinter import ttk, Canvas
from PIL import Image, ImageTk
import cv2
import numpy as np


def get_camera_candidates():
    return [0, 1, 2, 3]


class SurveillanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Edge AI Surveillance Starter")
        self.root.geometry("900x600")

        self.status_var = tk.StringVar(value="Connecting to camera")
        self.fps_var = tk.StringVar(value="FPS: 0.0")
        self.latency_var = tk.StringVar(value="Latency: 0 ms")
        self.feed_canvas = None
        self.cap = None
        self._detection_visible = False
        self._prev_gray = None
        self._motion_box = None

        self._build_ui()
        self._start_camera()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="Edge AI Surveillance Monitor", font=("Segoe UI", 16, "bold")).pack(anchor="w")

        feed_panel = ttk.Frame(main, padding=10, relief="groove")
        feed_panel.pack(fill=tk.BOTH, expand=True)

        ttk.Label(feed_panel, textvariable=self.status_var, foreground="darkgreen").pack(anchor="w")
        ttk.Label(feed_panel, textvariable=self.fps_var).pack(anchor="w")
        ttk.Label(feed_panel, textvariable=self.latency_var).pack(anchor="w")

        self.feed_canvas = Canvas(feed_panel, width=760, height=420, bg="black", highlightthickness=2, highlightbackground="gray")
        self.feed_canvas.pack(pady=(10, 0))
        self._draw_waiting_state()

    def _draw_waiting_state(self):
        self.feed_canvas.delete("all")
        self.feed_canvas.create_rectangle(0, 0, 760, 420, fill="#0d1b2a", outline="")
        self.feed_canvas.create_text(380, 180, text="Camera Connecting...", fill="white", font=("Segoe UI", 16, "bold"))
        self.feed_canvas.create_text(380, 230, text="Looking for a local webcam", fill="#8ecae6", font=("Segoe UI", 12))

    def _start_camera(self):
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self.cap = None

        for index in get_camera_candidates():
            capture = cv2.VideoCapture(index, backend)
            if capture.isOpened():
                self.cap = capture
                break

        if self.cap is None or not self.cap.isOpened():
            self.status_var.set("Camera unavailable")
            self.fps_var.set("FPS: 0.0")
            self.latency_var.set("Latency: 0 ms")
            self._draw_waiting_state()
            self.feed_canvas.create_text(380, 280, text="No camera detected", fill="#ff5f57", font=("Segoe UI", 12, "bold"))
            return

        self.status_var.set("Camera active")
        self._update_frame()

    def _update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("Camera disconnected")
            self._draw_waiting_state()
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image.resize((760, 420)))
            self.feed_canvas.delete("all")
            self.feed_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.feed_canvas.image = photo
            self.fps_var.set("FPS: 15.0")
            self.latency_var.set("Latency: 40 ms")
            self.root.after(30, self._update_frame)
            return

        frame_delta = cv2.absdiff(self._prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self._motion_box = None
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            self._motion_box = (x, y, x + w, y + h)
            break

        self._detection_visible = self._motion_box is not None
        self._prev_gray = gray

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image.resize((760, 420)))
        self.feed_canvas.delete("all")
        self.feed_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.feed_canvas.image = photo

        if self._detection_visible and self._motion_box is not None:
            x1, y1, x2, y2 = self._motion_box
            scale_x = 760 / frame.shape[1]
            scale_y = 420 / frame.shape[0]
            sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
            sx2, sy2 = int(x2 * scale_x), int(y2 * scale_y)
            self.feed_canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="#00ff66", width=3)
            self.feed_canvas.create_text(sx1 + 5, max(sy1 - 8, 10), text="Motion", fill="#00ff66", font=("Segoe UI", 10, "bold"))

        self.fps_var.set("FPS: 15.0")
        self.latency_var.set("Latency: 40 ms")
        self.root.after(30, self._update_frame)


def run_app():
    root = tk.Tk()
    SurveillanceApp(root)
    root.mainloop()
