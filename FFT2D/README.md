# FFT2D

2D Fourier reconstruction of an image, showing how frequency terms rebuild the spatial image over time.

## Files
- `fft2d_fourier_reconstruction.ipynb`
- `requirements.txt`

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
jupyter notebook fft2d_fourier_reconstruction.ipynb
```
Run cells top-to-bottom.

## Inputs and Parameters
- `img_path` controls the input image filename. Default in notebook: `Water.jpg`.
- If `img_path` does not exist, the notebook falls back to a synthetic image.
- `target_size` and animation settings control output look and speed.

## Notes
- This repo intentionally does not include image assets.
- To use your own image, place it in this folder and update `img_path` if needed.
