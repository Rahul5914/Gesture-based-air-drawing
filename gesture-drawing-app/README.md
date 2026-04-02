# AI Air Drawing App

A gesture-based drawing web application built with Streamlit, OpenCV, and MediaPipe.

## Description
This app allows you to draw in the air using your webcam. It tracks your hand landmarks in real-time, letting your index finger act as a virtual pen. You can change colors, brush thickness, use a neon glow effect, and clear or download the canvas!

## Features
- Real-time webcam feed via Streamlit (using `streamlit-webrtc` to bypass Streamlit deployment limitations)
- Hand tracking using MediaPipe
- Smooth line drawing with optional Neon Glow effect
- FPS optimization and display
- Download capability to save your artwork

### Gesture Controls
- ☝️ **Index finger only** -> Draw
- ✌️ **Index + Middle finger** -> Move/Hover without drawing
- ✋ **Open hand (all 5 fingers)** -> Clear canvas

## Setup Locally

1. **Clone the repo** (or download the files)
2. **Navigate into the folder**:
   ```bash
   cd gesture-drawing-app
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## Deploying on Streamlit Cloud
1. Push this folder to a GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click "New App".
4. Select your repository, branch, and set the main file path to `app.py`.
5. Click **Deploy!**

*(Note: Since `streamlit-webrtc` handles the webcam stream natively over the browser, it seamlessly supports cloud deployment without server-side OpenCV capture issues.)*

## Screenshots
*(Coming soon)*
