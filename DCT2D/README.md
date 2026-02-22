# DCT2D

2D DCT basis and stripwise reconstruction notebook for visualizing progressive frequency-limited image recovery.

## Files
- `dct2d_fourier_sep_basis.ipynb`
- `requirements.txt`

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
jupyter notebook dct2d_fourier_sep_basis.ipynb
```
Run cells top-to-bottom.

## Controls and Parameters
- Image geometry and animation controls are defined in the config cells (`W`, `H`, angle, band/budget settings).
- The notebook uses a synthetic image generator by default, so no external image is required.

## Notes
- The notebook includes optional OpenCV export cells for writing video.
