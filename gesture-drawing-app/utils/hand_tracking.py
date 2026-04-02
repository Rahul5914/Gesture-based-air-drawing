import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=max_num_hands,
                                               min_hand_detection_confidence=min_detection_confidence,
                                               min_hand_presence_confidence=min_tracking_confidence,
                                               min_tracking_confidence=min_tracking_confidence,
                                               running_mode=vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO)
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky
        self.results = None

    def find_hands(self, img, timestamp_ms=0, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        if self.detector._running_mode == vision.RunningMode.VIDEO:
            self.results = self.detector.detect_for_video(mp_image, timestamp_ms)
        else:
            self.results = self.detector.detect(mp_image)

        if self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                if draw:
                    # Very basic drawing as mp.solutions.drawing_utils is removed
                    h, w, c = img.shape
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return img

    def get_finger_states(self, img):
        lm_list = []
        fingers_up = []

        if self.results and self.results.hand_landmarks:
            my_hand = self.results.hand_landmarks[0]
            for id, lm in enumerate(my_hand):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            if len(lm_list) > 20:
                # Thumb (left/right check instead of up/down)
                # assuming right hand for simplicity, logic might need adjustment for robust thumb tracking
                if lm_list[self.tip_ids[0]][1] > lm_list[self.tip_ids[0] - 1][1]:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

                # 4 Fingers
                for id in range(1, 5):
                    if lm_list[self.tip_ids[id]][2] < lm_list[self.tip_ids[id] - 2][2]:
                        fingers_up.append(1)
                    else:
                        fingers_up.append(0)

        return lm_list, fingers_up
