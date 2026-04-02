import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    def get_finger_states(self, img):
        lm_list = []
        fingers_up = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            # Thumb
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
