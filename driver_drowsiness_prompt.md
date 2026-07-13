# Claude Code Prompt — Driver Drowsiness Detection System

I am building a **real-time driver drowsiness and alertness detection system** intended for actual in-vehicle use. The system must run in Python and be structured so it can later be trained and submitted via **Google Colab**.

---

## Project Goal

Detect two states in real time:
- **Drowsy**: driver is yawning or head-nodding (eyes closing, head drooping)
- **Alert**: driver is talking and looking at the road

---

## System Architecture

Build a **dual-model pipeline** that runs concurrently:

### 1. Facial Expression Model (Video)
- Use **OpenCV** for video capture and frame processing
- Use **MediaPipe Face Mesh** to extract facial landmarks
- Detect yawning (mouth aspect ratio), eye closure (EAR - Eye Aspect Ratio), and head nodding (pitch angle from landmarks)
- The model should be trainable on a **combination of a self-collected video dataset and existing public datasets** (e.g. YawDD, NTHU-DDD)
- Output a per-frame classification: `DROWSY` or `ALERT`

---

### 2. Audio / Voice Model (concurrent)
- Use **Whisper** (or a lightweight alternative like `webrtcvad` + energy threshold) for real-time audio input
- Detect whether the driver is **speaking** (→ alert signal) vs. producing non-speech sounds like yawning (→ drowsy signal)
- Run in a **separate thread** alongside the video pipeline

---

### 3. Fusion Layer
Combine outputs from both models with a simple **rule-based or weighted fusion** logic:

- If audio detects **speech** AND video detects **open eyes** AND **mouth is moving** → `ALERT`
- If video detects **yawning** (mouth wide open, sustained) OR **head nodding** AND audio detects yawn sound or silence → `DROWSY`
- Mouth movement must be confirmed via **MediaPipe lip landmarks** — track the movement delta of key lip points across frames to distinguish active speech (small, rhythmic mouth motion) from a yawn (large, slow mouth opening)
- Output a final unified state every ~1–2 seconds

> **Note:** Mouth movement detection should use a **Mouth Movement Score (MMS)** — a per-frame metric derived from the displacement of upper and lower lip landmarks. Speech produces a high-frequency, low-amplitude signal; yawning produces a low-frequency, high-amplitude signal. Use this distinction to improve fusion accuracy.

---

## Output / Alerts
- **Audio alert** to the driver when drowsiness is detected (use `playsound` or `pyttsx3`)
- **Event logging**: save timestamped events (state, confidence, frame snapshot) to a CSV or JSON log file

---

## Code Structure Requirements
- Modular design: separate files for `video_model.py`, `audio_model.py`, `fusion.py`, `alert.py`, `logger.py`, and a `main.py` entry point
- Include a `config.py` for tunable thresholds (EAR threshold, yawn threshold, MMS threshold, alert cooldown, etc.)
- Add clear comments explaining each component — I have intermediate ML knowledge but need explanations for architecture decisions
- Include a `requirements.txt`
- The training code should be in a separate `train/` folder structured to run in **Google Colab** (with dataset mounting from Google Drive)

---

## Additional Notes
- Prioritize **low latency** — the system must respond within 2–3 seconds of drowsiness onset
- Use pretrained weights where possible (transfer learning) to minimize training data requirements
- Provide guidance on what video data to self-collect and what augmentation to apply
- Assume the camera is a standard webcam mounted on the dashboard

---

Please start by presenting the **full project plan and file structure**, then implement module by module, asking for my confirmation before moving to the next module.
