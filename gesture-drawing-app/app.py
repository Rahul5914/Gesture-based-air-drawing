import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import time
import threading

from utils.hand_tracking import HandTracker
from utils.drawing_utils import create_canvas, draw_line_with_glow, blend_canvas

st.set_page_config(page_title="AI Air Drawing App", page_icon="✏️", layout="wide")

# Setup session state for global variables
if 'canvas' not in st.session_state:
    st.session_state['canvas'] = None
if 'clear_canvas' not in st.session_state:
    st.session_state['clear_canvas'] = False

# --- UI Sidebar ---
st.sidebar.title("🎨 Drawing Controls")

# Color Picker
color_map = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "Purple": (255, 0, 255),
    "Cyan": (255, 255, 0),
    "White": (255, 255, 255)
}
selected_color_name = st.sidebar.selectbox("Select Color", list(color_map.keys()))
draw_color = color_map[selected_color_name]

# Brush Thickness Slider
brush_thickness = st.sidebar.slider("Brush Thickness", 2, 20, 5)
glow_effect = st.sidebar.checkbox("Neon Glow Effect", value=True)

# Clear Canvas Button
if st.sidebar.button("Clear Canvas"):
    st.session_state['clear_canvas'] = True

# Instructions
st.sidebar.markdown("### 🖐️ Gesture Guide:")
st.sidebar.markdown("- ☝️ **Index finger only**: Draw")
st.sidebar.markdown("- ✌️ **Index + Middle**: Stop Drawing / Move")
st.sidebar.markdown("- ✋ **Open Hand (all fingers)**: Clear Canvas")

st.title("AI Air Drawing App")
st.write("Draw in the air using your webcam and hand gestures!")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.tracker = HandTracker(min_detection_confidence=0.8)
        self.px, self.py = 0, 0
        self.canvas = None
        self.prev_time = 0
        self.draw_color = (0, 0, 255)
        self.brush_thickness = 5
        self.glow_effect = True
        self.clear_canvas = False
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror image
        h, w, c = img.shape

        with self.lock:
            # Initialize canvas if needed
            if self.canvas is None or self.canvas.shape[:2] != (h, w):
                self.canvas = create_canvas(h, w)

            # Check if clear was requested from UI
            if self.clear_canvas:
                self.canvas = create_canvas(h, w)
                self.clear_canvas = False

            color = self.draw_color
            thickness = self.brush_thickness
            neon = self.glow_effect

        # Find hands
        img = self.tracker.find_hands(img)
        lm_list, fingers = self.tracker.get_finger_states(img)

        if len(lm_list) != 0:
            x1, y1 = lm_list[8][1:] # Index finger tip
            x2, y2 = lm_list[12][1:] # Middle finger tip

            # Gesture: Open Hand -> Clear
            if sum(fingers) == 5:
                with self.lock:
                    self.canvas = create_canvas(h, w)
                cv2.putText(img, "Canvas Cleared!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Gesture: Index + Middle -> Move/Hover
            elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                self.px, self.py = 0, 0 # Reset points so it doesn't connect
                cv2.circle(img, (x1, y1), 15, color, cv2.FILLED)
                cv2.putText(img, "Hover Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Gesture: Index Only -> Draw
            elif fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, color, cv2.FILLED)
                cv2.putText(img, "Drawing Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                if self.px == 0 and self.py == 0:
                    self.px, self.py = x1, y1

                # Draw on canvas
                with self.lock:
                    self.canvas = draw_line_with_glow(
                        self.canvas,
                        (self.px, self.py),
                        (x1, y1),
                        color,
                        thickness=thickness,
                        glow_thickness=thickness * 3,
                        is_neon=neon
                    )
                self.px, self.py = x1, y1
            else:
                 self.px, self.py = 0, 0
        else:
            self.px, self.py = 0, 0

        # Blend canvas with frame
        with self.lock:
            if self.canvas is not None:
                img = blend_canvas(img, self.canvas)

        # FPS Counter
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
        self.prev_time = curr_time
        cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

ctx = webrtc_streamer(
    key="drawing-app",
    video_processor_factory=VideoProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_processor:
    ctx.video_processor.draw_color = draw_color
    ctx.video_processor.brush_thickness = brush_thickness
    ctx.video_processor.glow_effect = glow_effect
    if st.session_state['clear_canvas']:
        ctx.video_processor.clear_canvas = True
        st.session_state['clear_canvas'] = False

# To allow downloading, we need a snapshot of the canvas from the video processor.
# Note: Streamlit WebRTC isolates frames, so getting a clean download requires
# extracting it via a button click that grabs the current canvas property.

if ctx.video_processor and ctx.video_processor.canvas is not None:
    # Use a lock-safe copy if available
    with ctx.video_processor.lock:
        current_canvas = ctx.video_processor.canvas.copy()

    is_success, buffer = cv2.imencode(".png", current_canvas)
    if is_success:
        st.download_button(
            label="Download Your Drawing!",
            data=buffer.tobytes(),
            file_name="air_drawing.png",
            mime="image/png"
        )
