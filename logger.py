"""
logger.py - keep a permanent record of what the system decided.

Every fused verdict can be written to two files (both human- and machine-
friendly):
  * events.csv   - one row per event, easy to open in Excel / pandas.
  * events.jsonl - one JSON object per line, easy to parse programmatically.

In addition, the moment the driver TRANSITIONS into DROWSY we save a snapshot
of that frame to logs/snapshots/, so a drowsy event can be reviewed later with a
picture, not just numbers. (We snapshot on the transition only, not every drowsy
frame, so we don't flood the disk.)

The image is written with cv2.imencode + numpy.tofile rather than cv2.imwrite,
because that path handles non-ASCII folder names on Windows safely.

Directories are created on construction, so the caller never has to.
"""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import LOG_CSV, LOG_JSONL, SNAPSHOT_DIR, SNAPSHOT_ON_DROWSY, STATE_DROWSY
from debug import dbg

# Column order for the CSV (and key order in each JSONL object).
_FIELDS = [
    "timestamp", "epoch", "source", "state", "confidence", "reasons",
    "ear", "mar", "mms_amplitude", "mms_oscillation", "pitch",
    "mouth_state", "audio_state", "drowsy_frac", "alert_frac", "snapshot",
]


class EventLogger:
    def __init__(self, csv_path=LOG_CSV, jsonl_path=LOG_JSONL,
                 snapshot_dir=SNAPSHOT_DIR, snapshot_on_drowsy=SNAPSHOT_ON_DROWSY,
                 source=""):
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_on_drowsy = snapshot_on_drowsy
        # Which clip / camera produced these rows. Constant for a run, so a batch
        # over many clips stays attributable to a clip instead of relying on the
        # order rows happen to land in the file.
        self.source = str(source)
        self._last_state = None

        # Make sure the folders exist and the CSV has a header.
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._rotate_if_stale_header()
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(_FIELDS)

    def _rotate_if_stale_header(self):
        """Retire a CSV whose header predates the current column set.

        The CSV is append-mode and long-lived, so a file written before a column
        was added keeps its old header and would silently take rows whose values
        no longer line up with it - corruption that only surfaces later, when
        someone reads it back. Rename it aside (never delete: those rows are
        someone's past run) and let a fresh file be created with today's header.
        The .jsonl goes with it so the pair keeps describing the same runs.
        """
        if not self.csv_path.exists():
            return
        try:
            with self.csv_path.open("r", newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), None)
        except OSError:
            return              # unreadable: leave it alone, the append will report
        if header is None or header == _FIELDS:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for path in (self.csv_path, self.jsonl_path):
            if path.exists():
                backup = path.with_suffix(path.suffix + f".{stamp}.bak")
                path.rename(backup)
                print(f"[logger] {path.name} used an older column set "
                      f"({len(header)} cols, now {len(_FIELDS)}); kept it as "
                      f"{backup.name} and started a fresh file.")

    def _save_snapshot(self, frame, now):
        """Write a JPEG snapshot of `frame`; return its path string (or '')."""
        if frame is None:
            return ""
        stamp = datetime.fromtimestamp(now).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = self.snapshot_dir / f"drowsy_{stamp}.jpg"
        try:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                buf.tofile(str(path))   # unicode-path-safe on Windows
                return str(path)
        except Exception as exc:  # noqa: BLE001 - a bad frame must not stop logging
            print(f"[logger] could not save snapshot ({exc})")
        return ""

    def log(self, fusion_result, video_result, frame=None, now=None):
        """Append one event. Saves a snapshot on the transition into DROWSY.

        Returns the row dict that was written (handy for tests / debugging).
        Intended to be called when fusion_result.is_new is True (once per
        ~1.5 s window), but works on any call.
        """
        if now is None:
            now = time.time()

        entered_drowsy = (fusion_result.state == STATE_DROWSY
                          and self._last_state != STATE_DROWSY)

        # Real-time debug trace (prints only when --debug is on). Read _last_state
        # BEFORE it's updated at the end of this method so we catch the transition.
        if fusion_result.state != self._last_state:
            dbg(f"State: {self._last_state or 'START'} -> {fusion_result.state}")
        if entered_drowsy:
            dbg("Driver is drowsy")

        snapshot = ""
        if entered_drowsy and self.snapshot_on_drowsy:
            snapshot = self._save_snapshot(frame, now)

        row = {
            "timestamp": datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
            "epoch": round(now, 3),
            "source": self.source,
            "state": fusion_result.state,
            "confidence": fusion_result.confidence,
            "reasons": "; ".join(fusion_result.reasons),
            "ear": round(float(video_result.get("ear", 0.0)), 4),
            "mar": round(float(video_result.get("mar", 0.0)), 4),
            "mms_amplitude": round(float(video_result.get("mms_amplitude", 0.0)), 4),
            "mms_oscillation": int(video_result.get("mms_oscillation", 0)),
            "pitch": round(float(video_result.get("pitch", 0.0)), 4),
            "mouth_state": video_result.get("mouth_state", ""),
            "audio_state": fusion_result.audio_state,
            "drowsy_frac": fusion_result.drowsy_frac,
            "alert_frac": fusion_result.alert_frac,
            "snapshot": snapshot,
        }

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row[k] for k in _FIELDS])
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        self._last_state = fusion_result.state
        return row


# ---------------------------------------------------------------------------
# Self-test: log a few synthetic events (incl. a DROWSY transition with a fake
# frame) into the system temp dir, then confirm the files + snapshot exist.
# Run: python logger.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import tempfile
    from types import SimpleNamespace
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    tmp = Path(tempfile.mkdtemp(prefix="dms_log_"))
    logger = EventLogger(csv_path=tmp / "events.csv",
                         jsonl_path=tmp / "events.jsonl",
                         snapshot_dir=tmp / "snapshots")

    vid = {"ear": 0.15, "mar": 0.7, "mms_amplitude": 0.3, "mms_oscillation": 1,
           "pitch": 0.1, "mouth_state": "yawn"}
    frame = np.full((480, 640, 3), 127, dtype=np.uint8)

    fr_alert = SimpleNamespace(state="ALERT", confidence=0.9, reasons=["speaking 80%"],
                               audio_state="speech", drowsy_frac=0.0, alert_frac=0.8)
    fr_drowsy = SimpleNamespace(state="DROWSY", confidence=0.95, reasons=["yawning 100%"],
                                audio_state="silence", drowsy_frac=1.0, alert_frac=0.0)

    t = time.time()
    logger.log(fr_alert, vid, frame=frame, now=t)
    logger.log(fr_drowsy, vid, frame=frame, now=t + 1.5)   # transition -> snapshot
    logger.log(fr_drowsy, vid, frame=frame, now=t + 3.0)   # still drowsy -> no new snap

    csv_lines = (tmp / "events.csv").read_text(encoding="utf-8").strip().splitlines()
    jsonl_lines = (tmp / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    snaps = list((tmp / "snapshots").glob("*.jpg"))

    print("logger self-test:")
    print(f"  dir          = {tmp}")
    print(f"  csv rows     = {len(csv_lines)-1} (expect 3, + header)")
    print(f"  jsonl rows   = {len(jsonl_lines)} (expect 3)")
    print(f"  snapshots    = {len(snaps)} (expect 1 - only on the DROWSY transition)")
    ok = (len(csv_lines) - 1 == 3 and len(jsonl_lines) == 3 and len(snaps) == 1)
    print("logger self-test", "OK" if ok else "FAILED")
