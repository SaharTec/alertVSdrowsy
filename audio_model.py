"""
audio_model.py - the microphone "sense" (runs concurrently with the camera).

A background thread listens to the mic and, about twice a second, tags the most
recent ~1 s of sound into a few buckets that matter for driver state:

    speech, yawn, singing, snore, breathing, silence

The fusion layer reads these to settle the one case the camera is structurally
blind to: an open mouth looks identical whether the driver is YAWNING or
TALKING, but it SOUNDS completely different. Audio breaks that tie.

Engine: **PANNs Cnn14** (Kong et al.) via `panns-inference`, a pretrained
AudioSet tagger that runs on **PyTorch**. This is a deliberate choice to stay
off TensorFlow (which breaks MediaPipe in this environment - see _tf_guard.py).

Design rules so audio can NEVER destabilise the video pipeline:
  * torch / panns_inference / sounddevice are imported LAZILY inside the thread,
    so importing this module is cheap and never forces those heavy deps.
  * ANY failure (package missing, no mic, download error) just disables the
    thread; .get() then returns all-zero scores and the video demo runs on,
    unaffected. Check .status for a human-readable reason.
  * Audio capture + inference share one dedicated thread. PortAudio and PyTorch
    both release the GIL during native work, so the video loop keeps running in
    true parallel without multiprocessing.

This adapts the parent project's src/audio_classifier.py; the public API
(start/stop/get/status, BUCKETS) is kept, and get_audio_state() is added so the
fusion layer can ask for a single word instead of interpreting raw scores.
"""
import threading

import numpy as np

from config import AUDIO_SPEECH_THRESH, AUDIO_YAWN_THRESH


# AudioSet display-name -> our bucket. Resolved against the tagger's own label
# list at load time, so a name missing in some version is just skipped (no crash).
_BUCKET_NAMES = {
    "yawn":      ["Yawn"],
    "speech":    ["Speech", "Conversation", "Narration, monologue",
                  "Speech synthesizer"],
    "singing":   ["Singing", "Choir", "Humming", "Chant"],
    "snore":     ["Snoring", "Snort"],
    "breathing": ["Breathing", "Gasp", "Sigh", "Wheeze"],
    "silence":   ["Silence"],
}
BUCKETS = list(_BUCKET_NAMES.keys())


class AudioModel(threading.Thread):
    """Background thread: microphone -> PANNs Cnn14 (PyTorch) -> bucket scores.

    Usage:
        audio = AudioModel()
        audio.start()
        scores = audio.get()           # {'speech': .., 'yawn': ..}, never blocks
        word   = audio.get_audio_state()  # 'speech' | 'yawn' | 'silence'
        audio.stop()
    """

    SAMPLE_RATE = 32000          # PANNs Cnn14 expects 32 kHz mono
    WINDOW_SEC = 1.0             # ~1 s of audio per inference
    HOP_SEC = 0.5                # refresh ~2x / second

    def __init__(self, device=None):
        super().__init__(daemon=True)   # daemon: never blocks process exit
        self._device = device           # input device index, None = default
        self._lock = threading.Lock()
        self._latest = {b: 0.0 for b in BUCKETS}
        self._stop = threading.Event()
        # 'starting' | 'loading model...' | 'ready' | 'disabled: <reason>'
        self.status = "starting"

    # ---- public API (called from the video thread) -----------------------
    def get(self):
        """Return a *copy* of the latest bucket scores. Never blocks."""
        with self._lock:
            return dict(self._latest)

    def get_audio_state(self, now=None):
        """Collapse the raw buckets into one word for the fusion layer.

        'speech'  - voice present (talking or singing) -> awake signal
        'yawn'    - an audible yawn                     -> drowsy signal
        'silence' - neither stands out                  -> no opinion
        Speech wins ties with yawn: a driver who is clearly talking is awake
        even if a yawn-ish sound briefly registers.

        `now` is accepted (and ignored) so this shares one call signature with
        FileAudioModel, whose recorded-clip audio must be queried by video time.
        """
        s = self.get()
        voice = max(s.get("speech", 0.0), s.get("singing", 0.0))
        yawn = s.get("yawn", 0.0)
        if voice >= AUDIO_SPEECH_THRESH and voice >= yawn:
            return "speech"
        if yawn >= AUDIO_YAWN_THRESH:
            return "yawn"
        return "silence"

    def stop(self):
        self._stop.set()

    # ---- internals (run inside the thread) -------------------------------
    def _load_tagger(self):
        """Load PANNs Cnn14 and map our buckets to its AudioSet class indices."""
        import torch  # lazy so importing this module stays light
        from panns_inference import AudioTagging, labels  # heavy: lazy, in-thread

        self.status = "loading model..."
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # checkpoint_path=None -> downloads Cnn14 (~300 MB) to ~/panns_data once.
        model = AudioTagging(checkpoint_path=None, device=device)
        name_to_idx = {name: i for i, name in enumerate(labels)}
        bucket_idx = {
            bucket: [name_to_idx[n] for n in names if n in name_to_idx]
            for bucket, names in _BUCKET_NAMES.items()
        }
        return model, bucket_idx

    def run(self):
        try:
            import sounddevice as sd
        except Exception as exc:  # noqa: BLE001 - any import problem disables us
            self.status = f"disabled: sounddevice missing ({exc})"
            return

        try:
            model, bucket_idx = self._load_tagger()
        except Exception as exc:  # noqa: BLE001 - torch / panns / download problem
            self.status = f"disabled: PANNs load failed ({exc})"
            return

        win = int(self.SAMPLE_RATE * self.WINDOW_SEC)
        hop = int(self.SAMPLE_RATE * self.HOP_SEC)
        buf = np.zeros(0, dtype=np.float32)

        try:
            with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1,
                                dtype="float32", blocksize=hop,
                                device=self._device) as stream:
                self.status = "ready"
                while not self._stop.is_set():
                    # Blocks ~HOP_SEC until a fresh chunk is ready - this is the
                    # thread's natural pace, so no sleep() is needed.
                    data, _overflow = stream.read(hop)     # (hop, 1) float32
                    buf = np.concatenate([buf, data[:, 0]])[-win:]
                    if len(buf) < win:
                        continue
                    # PANNs .inference returns (clipwise_output, embedding);
                    # clipwise_output is clip-level, shape (1, 527).
                    clipwise = model.inference(buf[None, :])[0]  # (1, 527)
                    scores = clipwise[0]                         # (527,)
                    latest = {
                        b: (float(scores[idx].max()) if idx else 0.0)
                        for b, idx in bucket_idx.items()
                    }
                    with self._lock:
                        self._latest = latest
        except Exception as exc:  # noqa: BLE001 - mic disappeared, etc.
            self.status = f"disabled: audio stream error ({exc})"


# ---------------------------------------------------------------------------
# Self-test: start the thread, sample its status/scores briefly, stop it. This
# never asserts a working mic/model (CI machines have neither) - it only proves
# the module imports, the thread starts, and .get()/.get_audio_state() return
# sane values without blocking. Run: python audio_model.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import time
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    print("Starting audio model (first ever run downloads PANNs Cnn14 ~300 MB)...")
    audio = AudioModel()
    audio.start()
    # Give it a moment to either load or disable itself, then read once.
    for _ in range(20):  # up to ~10 s
        time.sleep(0.5)
        if audio.status == "ready" or audio.status.startswith("disabled"):
            break
    print(f"  status        = {audio.status}")
    print(f"  scores        = {audio.get()}")
    print(f"  audio_state   = {audio.get_audio_state()}")
    audio.stop()
    print("audio_model self-test OK (thread started and .get() worked).")
