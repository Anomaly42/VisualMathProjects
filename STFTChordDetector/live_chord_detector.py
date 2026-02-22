"""
Live Chord Detection via FFT - Pygame App
Captures system audio and displays real-time chord detection.

Controls:
- FFT Window Size slider
- Detection Threshold slider
- Volume Gate slider (below this = noise, no chord detection)
"""

import numpy as np
import pygame
import sounddevice as sd
import threading
import queue
from collections import deque

# ============== Configuration ==============
WIDTH, HEIGHT = 0.5*1080, 0.5*1920
FPS = 30

# Default audio settings
DEFAULT_WINDOW_SIZE = 4096
DEFAULT_DETECTION_THRESHOLD = 0.4
DEFAULT_VOLUME_GATE = 0.02

# Frequency range for analysis
FREQ_MIN = 60
FREQ_MAX = 2000

# Audio buffer
SAMPLE_RATE = 44100
BUFFER_SIZE = 8192  # samples to keep in rolling buffer

# ============== Colors ==============
BG_COLOR = (10, 10, 10)
TEXT_COLOR = (255, 255, 255)
WAVEFORM_COLOR = (74, 158, 255)
FFT_COLOR = (74, 158, 255)
DETECTED_COLOR = (0, 255, 136)
UNDETECTED_COLOR = (255, 68, 68)
PLAYHEAD_COLOR = (255, 170, 0)
SLIDER_BG = (40, 40, 40)
SLIDER_FG = (100, 100, 100)
SLIDER_HANDLE = (200, 200, 200)

# ============== Note/Chord Data ==============
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def make_chord_template(root: int, intervals: list) -> np.ndarray:
    template = np.zeros(12)
    for interval in intervals:
        template[(root + interval) % 12] = 1.0
    return template

MAJOR_INTERVALS = [0, 4, 7]
MINOR_INTERVALS = [0, 3, 7]

CHORD_TEMPLATES = {}
for i, note in enumerate(NOTE_NAMES):
    CHORD_TEMPLATES[note] = make_chord_template(i, MAJOR_INTERVALS)
    CHORD_TEMPLATES[f"{note}m"] = make_chord_template(i, MINOR_INTERVALS)

# ============== Audio Processing ==============
def freq_to_chroma(freq: float) -> int:
    if freq <= 0:
        return -1
    midi = 12 * np.log2(freq / 440.0) + 69
    return int(round(midi)) % 12

def compute_chroma(spectrum: np.ndarray, freqs: np.ndarray, freq_min: float, freq_max: float) -> np.ndarray:
    chroma = np.zeros(12)
    for mag, freq in zip(spectrum, freqs):
        if freq < freq_min or freq > freq_max:
            continue
        pitch_class = freq_to_chroma(freq)
        if pitch_class >= 0:
            chroma[pitch_class] += mag ** 2
    max_val = np.max(chroma)
    if max_val > 0:
        chroma = chroma / max_val
    return chroma

def match_chord(chroma: np.ndarray) -> tuple:
    best_score = -1
    best_chord = "?"
    for name, template in CHORD_TEMPLATES.items():
        score = np.dot(chroma, template) / (np.linalg.norm(chroma) * np.linalg.norm(template) + 1e-8)
        if score > best_score:
            best_score = score
            best_chord = name
    return best_chord, best_score

def compute_fft(audio: np.ndarray, window_size: int, sample_rate: int) -> tuple:
    if len(audio) < window_size:
        audio = np.pad(audio, (0, window_size - len(audio)))
    else:
        audio = audio[-window_size:]
    
    segment = audio * np.hanning(window_size)
    spectrum = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(window_size, 1.0 / sample_rate)
    return spectrum, freqs

def get_note_freq_ranges():
    ranges = []
    for pitch_class in range(12):
        note_name = NOTE_NAMES[pitch_class]
        note_freqs = []
        for octave in range(2, 6):
            midi = pitch_class + (octave + 1) * 12
            freq = 440.0 * (2 ** ((midi - 69) / 12))
            if FREQ_MIN <= freq <= FREQ_MAX:
                freq_low = freq * (2 ** (-0.5/12))
                freq_high = freq * (2 ** (0.5/12))
                note_freqs.append((freq_low, freq_high, freq))
        if note_freqs:
            ranges.append((note_name, pitch_class, note_freqs))
    return ranges

NOTE_FREQ_RANGES = get_note_freq_ranges()

# ============== Slider Class ==============
class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label, format_str="{:.2f}"):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.format_str = format_str
        self.dragging = False
        self.handle_width = 20
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self._update_value(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_value(event.pos[0])
    
    def _update_value(self, mouse_x):
        rel_x = mouse_x - self.rect.x
        rel_x = max(0, min(rel_x, self.rect.width))
        self.value = self.min_val + (rel_x / self.rect.width) * (self.max_val - self.min_val)
    
    def draw(self, screen, font):
        # Background
        pygame.draw.rect(screen, SLIDER_BG, self.rect, border_radius=5)
        
        # Fill
        fill_width = int((self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width)
        fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_width, self.rect.height)
        pygame.draw.rect(screen, SLIDER_FG, fill_rect, border_radius=5)
        
        # Handle
        handle_x = self.rect.x + fill_width - self.handle_width // 2
        handle_rect = pygame.Rect(handle_x, self.rect.y - 5, self.handle_width, self.rect.height + 10)
        pygame.draw.rect(screen, SLIDER_HANDLE, handle_rect, border_radius=3)
        
        # Label
        label_text = font.render(f"{self.label}: {self.format_str.format(self.value)}", True, TEXT_COLOR)
        screen.blit(label_text, (self.rect.x, self.rect.y - 30))

# ============== Audio Capture ==============
class AudioCapture:
    def __init__(self, sample_rate=44100, buffer_size=8192):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.stream = None
        self.running = False
        
        # Initialize with silence
        for _ in range(buffer_size):
            self.audio_buffer.append(0.0)
    
    def _find_loopback_device(self):
        """Find a loopback/stereo mix device for capturing system audio."""
        devices = sd.query_devices()
        
        # Look for loopback devices (Windows WASAPI)
        for i, dev in enumerate(devices):
            name = dev['name'].lower()
            if dev['max_input_channels'] > 0:
                if any(keyword in name for keyword in ['loopback', 'stereo mix', 'what u hear', 'wave out']):
                    print(f"Found loopback device: {dev['name']}")
                    return i
        
        # If no loopback found, list available input devices
        print("\nNo loopback device found. Available input devices:")
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"  [{i}] {dev['name']}")
        
        # Return default input device
        default = sd.query_devices(kind='input')
        print(f"\nUsing default input: {default['name']}")
        return None
    
    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        
        # Convert to mono if stereo
        if len(indata.shape) > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata.flatten()
        
        with self.lock:
            for sample in mono:
                self.audio_buffer.append(sample)
    
    def start(self):
        device = self._find_loopback_device()
        
        try:
            self.stream = sd.InputStream(
                device=device,
                channels=2,
                samplerate=self.sample_rate,
                blocksize=1024,
                callback=self._audio_callback
            )
            self.stream.start()
            self.running = True
            print("Audio capture started!")
        except Exception as e:
            print(f"Error starting audio capture: {e}")
            print("\nTip: On Windows, enable 'Stereo Mix' in Sound settings,")
            print("or use a virtual audio cable like VB-Cable.")
    
    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.running = False
    
    def get_buffer(self) -> np.ndarray:
        with self.lock:
            return np.array(list(self.audio_buffer), dtype=np.float32)

# ============== Main App ==============
class ChordDetectorApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Live Chord Detection via FFT")
        
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 72)
        self.font_title = pygame.font.Font(None, 64)
        self.font_chord = pygame.font.Font(None, 144)
        self.font_small = pygame.font.Font(None, 36)
        self.font_tiny = pygame.font.Font(None, 28)
        
        # Layout regions (matching the video layout)
        self.title_rect = pygame.Rect(0, 0, WIDTH, int(HEIGHT * 0.06))
        self.wave_rect = pygame.Rect(50, int(HEIGHT * 0.06), WIDTH - 100, int(HEIGHT * 0.14))
        self.fft_rect = pygame.Rect(50, int(HEIGHT * 0.22), WIDTH - 100, int(HEIGHT * 0.38))
        self.chord_rect = pygame.Rect(0, int(HEIGHT * 0.62), WIDTH, int(HEIGHT * 0.12))
        self.controls_rect = pygame.Rect(50, int(HEIGHT * 0.76), WIDTH - 100, int(HEIGHT * 0.22))
        
        # Sliders
        slider_y = self.controls_rect.y + 60
        slider_spacing = 100
        slider_width = WIDTH - 150
        
        self.sliders = {
            'window_size': Slider(75, slider_y, slider_width, 30, 
                                  1024, 8192, DEFAULT_WINDOW_SIZE, 
                                  "FFT Window Size", "{:.0f}"),
            'threshold': Slider(75, slider_y + slider_spacing, slider_width, 30,
                               0.1, 0.9, DEFAULT_DETECTION_THRESHOLD,
                               "Detection Threshold", "{:.2f}"),
            'volume_gate': Slider(75, slider_y + slider_spacing * 2, slider_width, 30,
                                  0.001, 0.1, DEFAULT_VOLUME_GATE,
                                  "Volume Gate", "{:.3f}"),
        }
        
        # Audio capture
        self.audio = AudioCapture(SAMPLE_RATE, BUFFER_SIZE)
        
        # State
        self.current_chord = "?"
        self.confidence = 0.0
        self.chroma = np.zeros(12)
        self.spectrum = np.zeros(100)
        self.freqs = np.zeros(100)
        self.volume = 0.0
        self.is_silent = True
        
        self.running = True
    
    def run(self):
        self.audio.start()
        
        try:
            while self.running:
                self._handle_events()
                self._update()
                self._draw()
                self.clock.tick(FPS)
        finally:
            self.audio.stop()
            pygame.quit()
    
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
            
            for slider in self.sliders.values():
                slider.handle_event(event)
    
    def _update(self):
        # Get audio buffer
        audio_buffer = self.audio.get_buffer()
        
        # Calculate volume (RMS)
        self.volume = np.sqrt(np.mean(audio_buffer ** 2))
        
        # Get current settings
        window_size = int(self.sliders['window_size'].value)
        # Ensure power of 2
        window_size = 2 ** int(np.log2(window_size))
        threshold = self.sliders['threshold'].value
        volume_gate = self.sliders['volume_gate'].value
        
        # Check if silent
        self.is_silent = self.volume < volume_gate
        
        # Compute FFT
        self.spectrum, self.freqs = compute_fft(audio_buffer, window_size, SAMPLE_RATE)
        
        if not self.is_silent:
            # Compute chroma and match chord
            self.chroma = compute_chroma(self.spectrum, self.freqs, FREQ_MIN, FREQ_MAX)
            self.current_chord, self.confidence = match_chord(self.chroma)
        else:
            self.chroma = np.zeros(12)
            self.current_chord = "-"
            self.confidence = 0.0
    
    def _draw(self):
        self.screen.fill(BG_COLOR)
        
        self._draw_title()
        self._draw_waveform()
        self._draw_fft()
        self._draw_chord()
        self._draw_controls()
        
        pygame.display.flip()
    
    def _draw_title(self):
        title = self.font_title.render("Live Chord Detection via FFT", True, TEXT_COLOR)
        title_rect = title.get_rect(center=(WIDTH // 2, self.title_rect.centery))
        self.screen.blit(title, title_rect)
    
    def _draw_waveform(self):
        # Title
        label = self.font_small.render("Audio Waveform", True, TEXT_COLOR)
        self.screen.blit(label, (self.wave_rect.x, self.wave_rect.y - 30))
        
        # Get audio buffer
        audio_buffer = self.audio.get_buffer()
        
        # Downsample for display
        display_samples = 500
        step = max(1, len(audio_buffer) // display_samples)
        wave_display = audio_buffer[::step]
        
        if len(wave_display) < 2:
            return
        
        # Draw waveform
        rect = self.wave_rect
        center_y = rect.centery
        
        points = []
        for i, sample in enumerate(wave_display):
            x = int(rect.x + (i / len(wave_display)) * rect.width)
            y = int(center_y - sample * (rect.height // 2))
            y = max(rect.y, min(y, rect.bottom))  # clamp to rect
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, WAVEFORM_COLOR, False, points, 2)
        
        # Volume indicator
        vol_text = self.font_tiny.render(f"Volume: {self.volume:.4f}", True, 
                                         UNDETECTED_COLOR if self.is_silent else DETECTED_COLOR)
        self.screen.blit(vol_text, (rect.right - 200, rect.y - 30))
    
    def _draw_fft(self):
        # Title
        label = self.font_small.render("Frequency Spectrum (K-Space)", True, TEXT_COLOR)
        self.screen.blit(label, (self.fft_rect.x, self.fft_rect.y - 30))
        
        rect = self.fft_rect
        threshold = self.sliders['threshold'].value
        
        # Filter to frequency range
        freq_mask = (self.freqs >= FREQ_MIN) & (self.freqs <= FREQ_MAX)
        plot_freqs = self.freqs[freq_mask]
        plot_spectrum = self.spectrum[freq_mask]
        
        if len(plot_spectrum) == 0:
            return
        
        # Normalize
        max_spec = np.max(plot_spectrum)
        if max_spec > 0:
            plot_spectrum = plot_spectrum / max_spec
        
        # Draw note frequency bands
        for note_name, pitch_class, freq_ranges in NOTE_FREQ_RANGES:
            is_detected = self.chroma[pitch_class] > threshold
            band_color = DETECTED_COLOR if is_detected else UNDETECTED_COLOR
            alpha = 80 if is_detected else 30
            
            for freq_low, freq_high, center_freq in freq_ranges:
                # Convert frequency to x position (log scale)
                x1 = rect.x + (np.log2(freq_low / FREQ_MIN) / np.log2(FREQ_MAX / FREQ_MIN)) * rect.width
                x2 = rect.x + (np.log2(freq_high / FREQ_MIN) / np.log2(FREQ_MAX / FREQ_MIN)) * rect.width
                
                x1 = max(rect.x, min(x1, rect.right))
                x2 = max(rect.x, min(x2, rect.right))
                
                if x2 > x1:
                    band_surface = pygame.Surface((int(x2 - x1), rect.height), pygame.SRCALPHA)
                    band_surface.fill((*band_color, alpha))
                    self.screen.blit(band_surface, (x1, rect.y))
        
        # Draw FFT spectrum (log frequency scale)
        points = []
        for freq, mag in zip(plot_freqs, plot_spectrum):
            if freq > 0:
                x = int(rect.x + (np.log2(freq / FREQ_MIN) / np.log2(FREQ_MAX / FREQ_MIN)) * rect.width)
                y = int(rect.bottom - mag * rect.height)
                if rect.x <= x <= rect.right:
                    points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, FFT_COLOR, False, points, 2)
        
        # Frequency labels
        for freq in [100, 200, 400, 800, 1600]:
            x = rect.x + (np.log2(freq / FREQ_MIN) / np.log2(FREQ_MAX / FREQ_MIN)) * rect.width
            if rect.x <= x <= rect.right:
                label = self.font_tiny.render(f"{freq}", True, TEXT_COLOR)
                self.screen.blit(label, (x - 20, rect.bottom + 5))
        
        # Draw chroma bars at the bottom
        chroma_bar_height = 40
        chroma_y = rect.bottom - chroma_bar_height - 10
        bar_width = rect.width // 12
        
        for i, (note, val) in enumerate(zip(NOTE_NAMES, self.chroma)):
            x = rect.x + i * bar_width
            bar_height = int(val * chroma_bar_height)
            color = DETECTED_COLOR if val > threshold else UNDETECTED_COLOR
            
            pygame.draw.rect(self.screen, color, 
                           (x + 2, chroma_y + chroma_bar_height - bar_height, 
                            bar_width - 4, bar_height))
            
            # Note label
            note_label = self.font_tiny.render(note, True, TEXT_COLOR)
            self.screen.blit(note_label, (x + bar_width // 2 - 10, chroma_y + chroma_bar_height + 5))
    
    def _draw_chord(self):
        # Chord name
        chord_color = DETECTED_COLOR if self.confidence > 0.5 and not self.is_silent else TEXT_COLOR
        chord_text = self.font_chord.render(self.current_chord, True, chord_color)
        chord_rect = chord_text.get_rect(center=(WIDTH // 2, self.chord_rect.centery - 20))
        self.screen.blit(chord_text, chord_rect)
        
        # Confidence
        if not self.is_silent:
            conf_text = self.font_small.render(f"confidence: {self.confidence:.0%}", True, TEXT_COLOR)
        else:
            conf_text = self.font_small.render("(silent)", True, (128, 128, 128))
        conf_rect = conf_text.get_rect(center=(WIDTH // 2, self.chord_rect.centery + 50))
        self.screen.blit(conf_text, conf_rect)
    
    def _draw_controls(self):
        # Title
        label = self.font_small.render("Controls", True, TEXT_COLOR)
        self.screen.blit(label, (self.controls_rect.x, self.controls_rect.y))
        
        # Sliders
        for slider in self.sliders.values():
            slider.draw(self.screen, self.font_tiny)
        
        # Instructions
        instructions = [
            "ESC: Quit",
            "Drag sliders to adjust parameters",
        ]
        for i, text in enumerate(instructions):
            inst = self.font_tiny.render(text, True, (128, 128, 128))
            self.screen.blit(inst, (self.controls_rect.x, self.controls_rect.bottom - 60 + i * 25))

# ============== Entry Point ==============
if __name__ == "__main__":
    print("=" * 50)
    print("Live Chord Detection via FFT")
    print("=" * 50)
    print("\nTo capture system audio on Windows:")
    print("1. Right-click speaker icon -> Sound settings")
    print("2. Sound Control Panel -> Recording tab")
    print("3. Right-click -> Show Disabled Devices")
    print("4. Enable 'Stereo Mix' or similar")
    print("\nAlternatively, use a virtual audio cable (VB-Cable)")
    print("=" * 50)
    
    app = ChordDetectorApp()
    app.run()
