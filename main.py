# This is your main.py file

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import dlib
from imutils import face_utils
import numpy as np
import threading
import os


# Alarm function for notifications
def alarm(msg):
    while True:
        os.system(f"espeak '{msg}'")
        break


# Define the main Kivy app class
class DrowsinessDetectorApp(App):
    def build(self):
        # Set up the camera feed and layout
        return DrowsinessDetector()


# DrowsinessDetector handles the logic for the drowsiness detection app
class DrowsinessDetector(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img = Image()
        self.add_widget(self.img)

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        self.ear_thresh = 0.25
        self.eye_frame_count = 30
        self.yawn_thresh = 35
        self.alarm_status = False
        self.saying = False
        self.counter = 0

        # Set up the face detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        # Process the frame
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:
            shape = self.predictor(gray, face)
            shape_np = face_utils.shape_to_np(shape)

            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            left_eye = shape_np[lStart:lEnd]
            right_eye = shape_np[rStart:rEnd]

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            # Check for drowsiness
            if ear < self.ear_thresh:
                self.counter += 1
                if self.counter >= self.eye_frame_count and not self.alarm_status:
                    self.alarm_status = True
                    thread = threading.Thread(target=alarm, args=("Wake up!",))
                    thread.daemon = True
                    thread.start()
            else:
                self.counter = 0
                self.alarm_status = False

            # Draw contours on the image
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

            lip = shape_np[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            # Convert back to Kivy texture
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
            texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
            self.img.texture = texture


# Main function to start the app
if __name__ == "__main__":
    DrowsinessDetectorApp().run()
