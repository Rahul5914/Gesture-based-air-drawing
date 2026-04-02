import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import time
import threading

from utils.hand_tracking import HandTracker
from utils.drawing_utils import create_canvas, draw_line_with_glow, blend_canvas

st.set_page_config(page_title="AI Air Drawing App", page_icon="✏️", layout="wide")

# Setup session state for UI signals
if 'clear_canvas' not in st.session_state:
    st.session_state['clear_canvas'] = False

# --- Global Thread-Safe State for WebRTC ---
class DrawingState:
    def __init__(self):
        self.tracker = HandTracker(min_detection_confidence=0.8)
        self.canvas = None
        self.px = 0
        self.py = 0
        self.prev_time = 0
        self.lock = threading.Lock()

        # UI synced properties
        self.draw_color = (0, 0, 255)
        self.brush_thickness = 5
        self.glow_effect = True
        self.clear_requested = False

@st.cache_resource
def get_drawing_state():
    return DrawingState()

state = get_drawing_state()

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

# Update background state from UI
with state.lock:
    state.draw_color = draw_color
    state.brush_thickness = brush_thickness
    state.glow_effect = glow_effect
    if st.session_state['clear_canvas']:
        state.clear_requested = True
        st.session_state['clear_canvas'] = False

# Instructions
st.sidebar.markdown("### 🖐️ Gesture Guide:")
st.sidebar.markdown("- ☝️ **Index finger only**: Draw")
st.sidebar.markdown("- ✌️ **Index + Middle**: Stop Drawing / Move")
st.sidebar.markdown("- ✋ **Open Hand (all fingers)**: Clear Canvas")

st.title("AI Air Drawing App")
st.write("Draw in the air using your webcam and hand gestures!")

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) # Mirror image
    h, w, c = img.shape

    timestamp_ms = int(frame.time * 1000)

    with state.lock:
        # Initialize canvas if needed
        if state.canvas is None or state.canvas.shape[:2] != (h, w):
            state.canvas = create_canvas(h, w)

        # Check if clear was requested from UI
        if state.clear_requested:
            state.canvas = create_canvas(h, w)
            state.clear_requested = False

        color = state.draw_color
        thickness = state.brush_thickness
        neon = state.glow_effect

    # Find hands
    img = state.tracker.find_hands(img, timestamp_ms=timestamp_ms)
    lm_list, fingers = state.tracker.get_finger_states(img)

    with state.lock:
        if len(lm_list) != 0:
            x1, y1 = lm_list[8][1:] # Index finger tip
            x2, y2 = lm_list[12][1:] # Middle finger tip

            # Gesture: Open Hand -> Clear
            if sum(fingers) == 5:
                state.canvas = create_canvas(h, w)
                cv2.putText(img, "Canvas Cleared!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Gesture: Index + Middle -> Move/Hover
            elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                state.px, state.py = 0, 0 # Reset points so it doesn't connect
                cv2.circle(img, (x1, y1), 15, color, cv2.FILLED)
                cv2.putText(img, "Hover Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Gesture: Index Only -> Draw
            elif fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, color, cv2.FILLED)
                cv2.putText(img, "Drawing Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                if state.px == 0 and state.py == 0:
                    state.px, state.py = x1, y1

                # Draw on canvas
                state.canvas = draw_line_with_glow(
                    state.canvas,
                    (state.px, state.py),
                    (x1, y1),
                    color,
                    thickness=thickness,
                    glow_thickness=thickness * 3,
                    is_neon=neon
                )
                state.px, state.py = x1, y1
            else:
                 state.px, state.py = 0, 0
        else:
            state.px, state.py = 0, 0

        # Blend canvas with frame
        if state.canvas is not None:
            img = blend_canvas(img, state.canvas)

    # FPS Counter
    curr_time = time.time()
    fps = 1 / (curr_time - state.prev_time) if (curr_time - state.prev_time) > 0 else 0
    state.prev_time = curr_time
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="drawing-app",
    video_frame_callback=video_frame_callback,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
)

# To allow downloading
if state.canvas is not None:
    with state.lock:
        current_canvas = state.canvas.copy()

    is_success, buffer = cv2.imencode(".png", current_canvas)
    if is_success:
        st.download_button(
            label="Download Your Drawing!",
            data=buffer.tobytes(),
            file_name="air_drawing.png",
            mime="image/png"
        )