import math
import numpy as np

# Landmark Indices for MediaPipe Hand Landmarker:
# 0: WRIST
# 1-4: THUMB (CMC, MCP, IP, TIP)
# 5-8: INDEX (MCP, PIP, DIP, TIP)
# 9-12: MIDDLE (MCP, PIP, DIP, TIP)
# 13-16: RING (MCP, PIP, DIP, TIP)
# 17-20: PINKY (MCP, PIP, DIP, TIP)

def distance_3d(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def classify_hand(landmarks):
    """
    Classifies single-hand signs into letters/words based on geometric landmarks.
    Handles:
    - Letters: A, B, C, D, I, L, V, W, Y
    - Common Gestures: Thumbs Up (YES), Open Palm (HELLO), Peace (PEACE), Fist (STOP), I Love You (ILY), OK
    """
    wrist = landmarks[0]
    
    # Finger tips and joint references
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    middle_mcp = landmarks[9]
    
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    ring_mcp = landmarks[13]
    
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    pinky_mcp = landmarks[17]

    # Calculate hand scale for normalization
    hand_size = distance_3d(wrist, middle_mcp)
    if hand_size < 1e-4:
        return "UNKNOWN", 0.0

    # Finger open/extended states (tip is significantly higher than pip relative to wrist)
    # Note: In screen coordinates, y=0 is TOP, so y_tip < y_pip means finger is pointing UP
    index_up = index_tip.y < index_pip.y
    middle_up = middle_tip.y < middle_pip.y
    ring_up = ring_tip.y < ring_pip.y
    pinky_up = pinky_tip.y < pinky_pip.y
    
    # Thumb extended out horizontally or upwards
    thumb_up = thumb_tip.y < thumb_mcp.y
    thumb_extended = distance_3d(thumb_tip, index_mcp) > (hand_size * 0.6)

    # Distances between key tips
    d_thumb_index = distance_3d(thumb_tip, index_tip) / hand_size
    d_index_middle = distance_3d(index_tip, middle_tip) / hand_size
    d_thumb_middle = distance_3d(thumb_tip, middle_tip) / hand_size

    # 1. OK Sign / Letter 'O' variant: Thumb & Index touching, Middle, Ring, Pinky extended UP
    if d_thumb_index < 0.28 and middle_up and ring_up and pinky_up:
        return "OK", 0.95

    # 2. 'I LOVE YOU' (ILY): Thumb, Index, and Pinky UP, Middle & Ring DOWN
    if index_up and pinky_up and not middle_up and not ring_up and thumb_extended:
        return "I LOVE YOU", 0.96

    # 3. Letter 'Y' / 'HANG LOOSE': Thumb & Pinky extended, Index, Middle, Ring curled
    if pinky_up and thumb_extended and not index_up and not middle_up and not ring_up:
        return "Y", 0.95

    # 4. Letter 'L': Index pointing UP, Thumb extended out 90 degrees, others curled
    if index_up and thumb_extended and not middle_up and not ring_up and not pinky_up:
        return "L", 0.95

    # 5. Letter 'W': Index, Middle, Ring all UP, Pinky and Thumb folded
    if index_up and middle_up and ring_up and not pinky_up:
        return "W", 0.95

    # 6. Letter 'V' / 'PEACE': Index & Middle UP, separated, others curled
    if index_up and middle_up and not ring_up and not pinky_up:
        if d_index_middle > 0.18:
            return "V (PEACE)", 0.95
        else:
            return "U", 0.90

    # 7. Letter 'D': Only Index finger pointing UP, thumb touching middle/ring/pinky tips
    if index_up and not middle_up and not ring_up and not pinky_up and not thumb_extended:
        return "D", 0.92

    # 8. Letter 'I': Only Pinky pointing UP, others curled
    if pinky_up and not index_up and not middle_up and not ring_up and not thumb_extended:
        return "I", 0.95

    # 9. Letter 'B' / 'HELLO / OPEN PALM': All 4 fingers straight UP
    if index_up and middle_up and ring_up and pinky_up:
        if thumb_tip.y < wrist.y:
            return "HELLO (OPEN PALM)", 0.95
        return "B", 0.92

    # 10. THUMBS UP / 'YES': Only thumb pointing straight UP, all 4 fingers firmly curled
    if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        # verify thumb is higher than all other fingers
        if thumb_tip.y < index_pip.y and thumb_tip.y < middle_pip.y:
            return "THUMBS UP (YES)", 0.95

    # 11. Letter 'C': Curved fingers forming a C shape
    if (not index_up and not middle_up and not ring_up and not pinky_up) and thumb_extended:
        if 0.35 < d_thumb_index < 0.85:
            return "C", 0.88

    # 12. Letter 'A' / 'FIST' / 'STOP': All fingers curled down into a fist
    if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
        return "A (FIST / STOP)", 0.90

    return "...", 0.50
