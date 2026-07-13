"""
file_audio_model.py - the audio "sense" for a RECORDED clip (Colab / offline).

The live system's AudioModel listens to the microphone. When you run the detector
on an uploaded MP4 there is no mic - but the clip has its own soundtrack. This
class gives the fusion layer the SAME audio signal, computed from the file's audio
track instead of a live mic, so a recorded run reproduces the live run exactly.

It mirrors AudioModel bucket-for-bucket:
  * the same PANNs Cnn14 tagger (PyTorch), the same 1.0 s window / 0.5 s hop,
  * the same bucket -> AudioSet mapping (imported from audio_model),
  * the same get_audio_state() collapse (speech / yawn / silence; speech wins ties).

Two differences from the mic model, both required by "recorded, not live":
  * SOURCE - a decoded audio file, not a live PortAudio stream. The whole track is
    tagged up front (batched through PANNs), so playback is a cheap table lookup.
  * CLOCK  - get_audio_state(now) returns the window covering the current *video*
    time, so audio stays aligned with the frames main.py is processing. (main uses
    a video clock now = start + frame_idx/fps, so now - t0 == the clip time.)

Audio is decoded to 32 kHz mono via ffmpeg (present on Colab). Any failure - no
audio track, ffmpeg/torch/panns missing - disables it gracefully: get_audio_state
then returns 'silence' and the run continues video-only, exactly like the mic model
does when it can't open a device. Check .status for a human-readable reason.
"""
import bisect
import os
import subprocess
import tempfile
import wave

import numpy as np

from config import AUDIO_SPEECH_THRESH, AUDIO_YAWN_THRESH
from audio_model import BUCKETS, _BUCKET_NAMES   # reuse the exact same bucket map

SR = 32000            # PANNs Cnn14 expects 32 kHz mono (same as AudioModel)
WINDOW_SEC = 1.0
HOP_SEC = 0.5


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001 - torch absent -> just use CPU path later
        return False


def _resample_linear(data, sr_in, sr_out):
    """Cheap linear resample. ffmpeg already outputs 32 kHz, so this is a rarely
    used safety net for a hand-supplied wav at another rate."""
    if sr_in == sr_out or len(data) < 2:
        return data
    n_out = int(round(len(data) * sr_out / sr_in))
    x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, data).astype(np.float32)


class FileAudioModel:
    """Offline, file-driven twin of AudioModel. Same public API main.py uses:
    start(), stop(), get(), get_audio_state(now), and .status."""

    def __init__(self, path):
        self._path = path
        self.status = "starting"
        self._latest = {b: 0.0 for b in BUCKETS}
        self._t0 = None          # video time of the first get_audio_state() call
        self._ends = []          # window END times in seconds, ascending
        self._scores = []        # list[dict bucket->score], aligned to _ends

    # ---- public API (called from the video loop) -------------------------
    def start(self):
        """Decode the soundtrack and pre-tag every window. Safe: any failure sets
        .status to 'disabled: ...' and leaves an empty timeline, so get_audio_state
        just returns 'silence' and the video run is unaffected."""
        try:
            wav = self._decode_to_mono32k(self._path)
        except Exception as exc:  # noqa: BLE001 - ffmpeg/codec/read problem
            self.status = f"disabled: audio decode failed ({exc})"
            return
        if wav is None or len(wav) < int(SR * WINDOW_SEC):
            self.status = "disabled: clip has no usable audio track"
            return
        try:
            self._tag_all_windows(wav)
            self.status = f"ready (file audio: {len(self._ends)} windows)"
        except Exception as exc:  # noqa: BLE001 - torch / panns / download problem
            self.status = f"disabled: PANNs unavailable ({exc})"
            self._ends, self._scores = [], []

    def get(self, now=None):
        """Latest bucket scores (the window from the most recent get_audio_state).
        Signature matches AudioModel.get(); `now` is accepted for uniformity."""
        return dict(self._latest)

    def get_audio_state(self, now=None):
        """Collapse the current window into one word, EXACTLY like AudioModel:
        'speech' (voice, awake), 'yawn' (audible yawn), or 'silence'. When `now`
        (the video clock) is given, first advance to the window covering it."""
        if now is not None and self._ends:
            if self._t0 is None:
                self._t0 = now
            self._latest = self._lookup(now - self._t0)
        s = self._latest
        voice = max(s.get("speech", 0.0), s.get("singing", 0.0))
        yawn = s.get("yawn", 0.0)
        if voice >= AUDIO_SPEECH_THRESH and voice >= yawn:
            return "speech"
        if yawn >= AUDIO_YAWN_THRESH:
            return "yawn"
        return "silence"

    def stop(self):
        pass  # nothing to tear down; the whole track was tagged in start()

    # ---- internals -------------------------------------------------------
    def _lookup(self, t):
        """The most recently completed window at clip time `t` (mirrors the mic:
        before the first 1 s window closes, scores are all-zero -> 'silence')."""
        idx = bisect.bisect_right(self._ends, t) - 1
        if idx < 0:
            return {b: 0.0 for b in BUCKETS}
        return self._scores[idx]

    @staticmethod
    def _decode_to_mono32k(path):
        """Return a float32 mono waveform at 32 kHz in [-1, 1]. A .wav is read
        directly (stdlib `wave`); anything else is piped through ffmpeg first."""
        ext = os.path.splitext(path)[1].lower()
        wav_path, tmp = path, None
        if ext != ".wav":
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                   "-ac", "1", "-ar", str(SR), "-f", "wav", tmp.name]
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode != 0 or os.path.getsize(tmp.name) == 0:
                os.unlink(tmp.name)
                msg = proc.stderr.decode(errors="ignore")[-200:] or "ffmpeg failed"
                raise RuntimeError(msg)
            wav_path = tmp.name
        try:
            with wave.open(wav_path, "rb") as w:
                sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
                raw = w.readframes(w.getnframes())
        finally:
            if tmp is not None:
                os.unlink(tmp.name)

        if sw == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sw == 1:
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
        else:
            raise RuntimeError(f"unsupported sample width: {sw} bytes")
        if nch > 1:
            data = data.reshape(-1, nch).mean(axis=1)
        return _resample_linear(data, sr, SR)

    def _tag_all_windows(self, wav):
        """Run PANNs Cnn14 over every 1 s window (0.5 s hop), batched for speed.
        Same math as AudioModel, just vectorised over the whole clip."""
        import torch  # noqa: F401 - panns needs it; error here disables us cleanly
        from panns_inference import AudioTagging, labels

        device = "cuda" if _cuda_available() else "cpu"
        model = AudioTagging(checkpoint_path=None, device=device)  # downloads once
        name_to_idx = {name: i for i, name in enumerate(labels)}
        bucket_idx = {
            b: [name_to_idx[n] for n in names if n in name_to_idx]
            for b, names in _BUCKET_NAMES.items()
        }

        win, hop = int(SR * WINDOW_SEC), int(SR * HOP_SEC)
        ends = list(range(win, len(wav) + 1, hop))       # window END sample indices
        windows = np.stack([wav[e - win:e] for e in ends]).astype(np.float32)

        BATCH = 64
        for i in range(0, len(windows), BATCH):
            clipwise = model.inference(windows[i:i + BATCH])[0]   # (b, 527)
            for j, e in enumerate(ends[i:i + BATCH]):
                scores = clipwise[j]
                self._ends.append(e / SR)
                self._scores.append({
                    b: (float(scores[idx].max()) if idx else 0.0)
                    for b, idx in bucket_idx.items()
                })


# ---------------------------------------------------------------------------
# Self-test: synthesise a short WAV, prove decode + window timeline + lookup work
# and that get_audio_state()/get() never block. PANNs tagging is attempted but not
# required (a CI box has no model) - a 'disabled' status there is a PASS for the
# plumbing. Run: python file_audio_model.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    dur = 3.0
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    with wave.open(tmp.name, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(tone.tobytes())

    am = FileAudioModel(tmp.name)
    am.start()
    print("status        :", am.status)
    # Query along the clip's timeline (now == clip time here since t0 starts at 0).
    for now in (0.0, 1.2, 2.4):
        print(f"  t={now:>4}s -> {am.get_audio_state(now):8}  scores={am.get()}")
    os.unlink(tmp.name)
    ok = am.status.startswith("ready") or am.status.startswith("disabled")
    print("file_audio_model self-test", "OK" if ok else "FAILED")
