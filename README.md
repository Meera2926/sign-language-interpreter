# 🧏 AI Sign Language & Gesture Interpreter (Webcam)

A real-time Computer Vision application that uses your laptop's built-in webcam to recognize American Sign Language (ASL) alphabets and common conversational gestures, translating them into on-screen text and spoken audio.

---

## 🌟 Features
1. **Real-Time Hand Tracking**: Powered by MediaPipe 3D Hand Landmarks with 21 joint tracking points.
2. **Gesture & ASL Recognition**:
   - **Letters**: `A`, `B`, `C`, `D`, `I`, `L`, `V`, `W`, `Y`
   - **Common Signs**:
     - `HELLO` (Open Palm)
     - `PEACE / V` (Two fingers UP)
     - `OK` (Thumb + Index circle, 3 fingers UP)
     - `I LOVE YOU` (Thumb + Index + Pinky UP)
     - `THUMBS UP / YES` (Fist with thumb UP)
     - `STOP / FIST` (All fingers folded)
3. **Text-to-Speech (TTS)**: Built-in `pyttsx3` background speech synthesizer speaks words aloud.
4. **Sentence Builder**: Form sentences by pressing `[SPACE]` to append words, and press `[S]` to hear the full sentence spoken.
5. **HUD UI Overlay**: Clean translucent head-up display showing active sign, FPS, confidence, and sentence status.

---

## 🚀 How to Run

Open your terminal / PowerShell and run:

```bash
cd "C:\Users\MEERA KRISHNA\.gemini\antigravity-ide\scratch\sign_language_interpreter"
python main.py
```

---

## 🎮 Keyboard Controls

| Key | Action |
|---|---|
| **`[SPACE]`** | Add the currently detected letter/word to the sentence |
| **`[BACKSPACE]`** | Delete the last word from the sentence |
| **`[S]`** | Speak the entire constructed sentence aloud |
| **`[C]`** | Clear the current sentence |
| **`[Q]` or `[ESC]`** | Exit application |

---

## 🛠️ Project Structure

```
sign_language_interpreter/
├── hand_landmarker.task  # Pretrained MediaPipe 3D Hand Landmarker model
├── classifier.py         # 3D Geometric gesture classification rules
├── main.py               # Webcam feed, HUD overlay, speech engine, loop
└── README.md             # Documentation
```
