# FFT1D

1D Fourier reconstruction of an audio signal with visual animation of frequency components.

## Files
- `fft1d_fourier_reconstruction.ipynb`
- `requirements.txt`

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
jupyter notebook fft1d_fourier_reconstruction.ipynb
```
Run cells top-to-bottom.

## Inputs and Parameters
- `audio_path` is the input audio file. Default in notebook: `FAHHH.mp3`.
- Sampling and reconstruction settings are controlled in the config cells.

## Notes
- This repo intentionally does not include audio/video assets.
- Place your audio file in this folder and update `audio_path` if needed.
