# STFTChordDetector

Chord detection from FFT/STFT in two forms: an offline notebook renderer and a real-time live detector app.

## Files
- `stft_chord_detection.ipynb`
- `live_chord_detector.py`
- `requirements.txt`

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Notebook (Offline Render)
```bash
jupyter notebook stft_chord_detection.ipynb
```
Run cells top-to-bottom.

### Notebook Inputs and Parameters
- `AUDIO_PATH` default is `audio/input.mp3`.
- Output video path is `out/chord_detection.mp4`.
- Core controls: `FPS`, `WINDOW_SIZE`, `FREQ_MIN`, `FREQ_MAX`, `DETECTION_THRESHOLD`, `VOLUME_THRESHOLD`.

## Run Live App (Real-Time)
```bash
python live_chord_detector.py
```

### Live App Controls
- Drag sliders for FFT window size, detection threshold, and volume gate.
- `ESC` quits.

## Notes
- This repo intentionally does not include audio/video assets.
- For Windows system-audio capture, enable Stereo Mix or use a loopback/virtual cable.
