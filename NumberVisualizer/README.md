# NumberVisualizer

Interactive browser-based visualization of a number's prime-factor structure using recursive polygons on canvas.

## Files
- `index.html`

## Run (Local Server)
From this folder:
```bash
python -m http.server 8000
```
Then open:
- `http://localhost:8000`

## Controls and Parameters
- `Left Arrow`: decrease number by 1.
- `Right Arrow`: increase number by 1.
- `Down Arrow`: decrease number by 50.
- `Up Arrow`: increase number by 50.
- The visual structure updates from prime factors of the current number.

## Notes
- No build step is required.
- Any modern browser with Canvas support should work.
