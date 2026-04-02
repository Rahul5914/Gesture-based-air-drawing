import cv2
import numpy as np

def create_canvas(height, width):
    """Creates an empty black canvas."""
    return np.zeros((height, width, 3), np.uint8)

def draw_line_with_glow(canvas, start_pos, end_pos, color, thickness=5, glow_thickness=15, is_neon=True):
    """Draws a smooth line on the canvas, optionally with a neon glow effect."""
    if is_neon:
        # Draw the glow (thicker, faded line)
        glow_color = color
        cv2.line(canvas, start_pos, end_pos, glow_color, glow_thickness, cv2.LINE_AA)

        # Draw the core (thinner, white line)
        core_color = (255, 255, 255)
        cv2.line(canvas, start_pos, end_pos, core_color, thickness, cv2.LINE_AA)
    else:
        # Standard draw
        cv2.line(canvas, start_pos, end_pos, color, thickness, cv2.LINE_AA)

    return canvas

def blend_canvas(frame, canvas):
    """Blends the drawing canvas over the video frame."""
    # Convert canvas to grayscale to find where drawn
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

    # Apply bitwise operations to combine frame and canvas
    frame = cv2.bitwise_and(frame, img_inv)
    frame = cv2.bitwise_or(frame, canvas)

    return frame
