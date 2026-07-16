"""
fusion.py - combine the video and audio senses into ONE decision.

The camera reports a state every frame (fast but jittery) and the mic reports a
sound every half second. Neither alone should drive an alarm. The FusionEngine
collects both over a short window (~1.5 s) and then emits a single, stable
verdict: ALERT, DROWSY or NEUTRAL, with a confidence and a short list of the
reasons - so the decision is explainable, not a black box.

It implements the prompt's fusion rules as weighted evidence:

    * speech + eyes-open + facing-road        -> ALERT
    * (yawn OR nod OR eyes-closed) + (yawn-sound OR silence) -> DROWSY

The key tie-break: if the mouth is open but the mic hears SPEECH, the audio
nudges the verdict toward ALERT (talking/singing), not DROWSY (yawning) - the
exact case the camera can't resolve on its own.

Why a window instead of per-frame: a single noisy frame shouldn't flip the
verdict. Averaging over ~1.5 s removes flicker and matches the prompt's "output
a unified state every 1-2 seconds".
"""
import time
from dataclasses import dataclass, field

from config import (
    FUSION_INTERVAL_SECONDS, FUSION_MIN_EVIDENCE,
    WEIGHT_EYES, WEIGHT_YAWN, WEIGHT_NOD, WEIGHT_SPEECH, WEIGHT_AUDIO,
    STATE_ALERT, STATE_DROWSY, STATE_NEUTRAL,
)
from debug import dbg


@dataclass
class FusionResult:
    """One window's verdict. `is_new` is True only on the frame a fresh window
    closes, so callers (alert/logger) can act on the transition, not every call."""
    state: str
    confidence: float
    reasons: list = field(default_factory=list)
    is_new: bool = False
    # aggregates kept for the HUD / logs
    drowsy_frac: float = 0.0
    alert_frac: float = 0.0
    # Per-cue window fractions {eyes, yawn, nod, speech}: which cue(s) actually
    # drove this verdict. Lets a caller attribute a DROWSY window to a cause
    # (eyes vs yawn vs nod) without parsing the human-readable `reasons` strings.
    cue_fracs: dict = field(default_factory=dict)
    audio_state: str = "silence"
    # which sensor(s) heard the driver talk this window (Rule 5): "Audio",
    # "Video/Lip Movement", "Audio + Video", or None when no speech.
    speech_modality: str = None


class FusionEngine:
    """Feed it (video_result, audio_state) every frame; it returns the current
    FusionResult, recomputing the verdict once per ~1.5 s window."""

    def __init__(self, interval=FUSION_INTERVAL_SECONDS):
        self.interval = interval
        self._win_start = None
        self._reset_window()
        self._last = FusionResult(STATE_NEUTRAL, 0.0, ["warming up"])

    def _reset_window(self):
        self._win_start = None
        self._n = 0
        self._drowsy = self._alert = 0
        self._eyes = self._yawn = self._nod = self._speech = 0
        self._audio_votes = {"speech": 0, "yawn": 0, "silence": 0}

    def update(self, video_result, audio_state="silence", now=None):
        """Accumulate one frame. Returns the latest FusionResult every call;
        `is_new` flags the call where the window actually closed and re-decided."""
        if now is None:
            now = time.time()
        if self._win_start is None:
            self._win_start = now

        # ---- accumulate this frame's evidence ----------------------------
        self._n += 1
        st = video_result.get("state", STATE_NEUTRAL)
        if st == STATE_DROWSY:
            self._drowsy += 1
        elif st == STATE_ALERT:
            self._alert += 1
        if video_result.get("eyes_closed"):
            self._eyes += 1
        if video_result.get("yawning"):
            self._yawn += 1
        if video_result.get("nodding"):
            self._nod += 1
        if video_result.get("mouth_state") == "speech":
            self._speech += 1
        if audio_state not in self._audio_votes:
            audio_state = "silence"
        self._audio_votes[audio_state] += 1

        # ---- not time yet: hand back the cached verdict ------------------
        if now - self._win_start < self.interval:
            self._last.is_new = False
            return self._last

        # ---- window closed: decide, then start a fresh window ------------
        result = self._decide()
        self._reset_window()
        self._win_start = now
        self._last = result
        return result

    def _decide(self):
        n = max(1, self._n)
        f_eyes = self._eyes / n
        f_yawn = self._yawn / n
        f_nod = self._nod / n
        f_speech = self._speech / n
        f_drowsy = self._drowsy / n
        f_alert = self._alert / n
        audio_state = max(self._audio_votes, key=self._audio_votes.get)
        cue_fracs = {"eyes": round(f_eyes, 3), "yawn": round(f_yawn, 3),
                     "nod": round(f_nod, 3), "speech": round(f_speech, 3)}

        # ---- Speech modality: which sensor(s) actually heard talking --------
        # Computed from the RAW signals (audio vote vs. visual lip motion) so the
        # debug trace reports what was SENSED even when the rule below decides
        # NEUTRAL (e.g. audio-only speech -> the Rule 2 no-lips override).
        audio_speech = audio_state == "speech"
        video_speech = f_speech > 0
        if audio_speech and video_speech:
            speech_modality = "Audio + Video"
        elif audio_speech:
            speech_modality = "Audio"
        elif video_speech:
            speech_modality = "Video/Lip Movement"
        else:
            speech_modality = None
        if speech_modality:
            dbg(f"Speech detected ({speech_modality})")

        # Weighted evidence for each side (the prompt's rules made numeric).
        drowsy_ev = WEIGHT_EYES * f_eyes + WEIGHT_YAWN * f_yawn + WEIGHT_NOD * f_nod
        alert_ev = WEIGHT_SPEECH * f_speech

        # Audio nudges whichever side it supports. Silence is neutral: it does
        # NOT add drowsy evidence, it merely fails to argue for ALERT - which
        # is exactly the prompt's "yawn-sound OR silence" permission for DROWSY.
        reasons = []
        if audio_state == "speech":
            if f_speech > 0:
                alert_ev += WEIGHT_AUDIO
                reasons.append("audio: voice")
            else:
                # A voice with no lip movement is not THIS driver: a passenger, the
                # radio, someone outside. It therefore says nothing about the
                # driver's state and adds no evidence to either side.
                # It used to return NEUTRAL here, throwing away every drowsy cue
                # the camera had collected for the window - so a driver yawning
                # silently next to a talking passenger was reported NEUTRAL.
                # Crediting ALERT instead would be the opposite error: the driver
                # doesn't become alert because someone else spoke. So: abstain,
                # and let the camera's evidence decide.
                reasons.append("audio: a voice, but not the driver's")
        elif audio_state == "yawn":
            drowsy_ev += WEIGHT_AUDIO
            reasons.append("audio: yawn")

        # Explain the visual contributors that actually fired.
        if f_eyes > 0:
            reasons.append(f"eyes closed {f_eyes*100:.0f}%")
        if f_yawn > 0:
            reasons.append(f"yawning {f_yawn*100:.0f}%")
        if f_nod > 0:
            reasons.append(f"nodding {f_nod*100:.0f}%")
        if f_speech > 0:
            reasons.append(f"speaking {f_speech*100:.0f}%")

        # Decide + a confidence that reflects how dominant the winner is.
        # The min-evidence gate keeps us in NEUTRAL when neither side has any
        # real support, so we never report ALERT/DROWSY at high confidence off a
        # negligible signal (e.g. 2% speaking).
        total = drowsy_ev + alert_ev
        if total < FUSION_MIN_EVIDENCE:
            state, confidence = STATE_NEUTRAL, 0.0
            if not reasons:
                reasons = ["quiet / unclear"]
        elif drowsy_ev >= alert_ev:
            state = STATE_DROWSY
            confidence = drowsy_ev / total          # 0.5..1.0
        else:
            state = STATE_ALERT
            confidence = alert_ev / total           # 0.5..1.0

        return FusionResult(
            state=state,
            confidence=round(float(confidence), 3),
            reasons=reasons,
            is_new=True,
            drowsy_frac=round(f_drowsy, 3),
            alert_frac=round(f_alert, 3),
            cue_fracs=cue_fracs,
            audio_state=audio_state,
            speech_modality=speech_modality,
        )


# ---------------------------------------------------------------------------
# Self-test: drive the engine with synthetic per-frame readings and check it
# reaches the expected verdict for three scenarios. Uses a fake clock so it runs
# instantly. Run: python fusion.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    def frame(state, eyes=False, yawn=False, nod=False, mouth="still"):
        return {"state": state, "eyes_closed": eyes, "yawning": yawn,
                "nodding": nod, "mouth_state": mouth}

    def run(name, vframe, audio, expect):
        eng = FusionEngine(interval=1.0)
        t = 0.0
        res = None
        for _ in range(40):           # ~2 s of frames at dt=0.05
            res = eng.update(vframe, audio, now=t)
            t += 0.05
        ok = "OK" if res.state == expect else "FAIL"
        print(f"  [{ok}] {name}: -> {res.state} ({res.confidence:.2f})  "
              f"reasons={res.reasons}")
        return res.state == expect

    print("fusion self-test:")
    passed = True
    passed &= run("yawning + silence",
                  frame(STATE_DROWSY, yawn=True, mouth="yawn"), "silence",
                  STATE_DROWSY)
    passed &= run("talking + voice",
                  frame(STATE_ALERT, mouth="speech"), "speech",
                  STATE_ALERT)
    passed &= run("open mouth but mic hears speech (tie-break -> ALERT)",
                  frame(STATE_NEUTRAL, mouth="speech"), "speech",
                  STATE_ALERT)
    # The regression this pins: a driver yawning SILENTLY while a passenger or the
    # radio talks. The mic hears a voice, the driver's own lips aren't moving, and
    # fusion used to answer NEUTRAL - discarding a window that was 100% yawning.
    # The camera's evidence must survive a voice that isn't the driver's.
    passed &= run("silent yawn + a passenger talking -> still DROWSY",
                  frame(STATE_DROWSY, yawn=True, mouth="yawn"), "speech",
                  STATE_DROWSY)
    # ...but a voice with no lip movement must not make a drowsy driver look ALERT
    # either: with no visual evidence at all there is nothing to report.
    passed &= run("a voice, nobody's lips moving -> NEUTRAL",
                  frame(STATE_NEUTRAL, mouth="still"), "speech",
                  STATE_NEUTRAL)
    print("fusion self-test", "OK" if passed else "had FAILURES")
    # Exit non-zero on failure: this test printed FAIL and still exited 0, so
    # nothing (a CI job, a pre-commit hook, a human in a hurry) could ever catch a
    # regression by running it.
    sys.exit(0 if passed else 1)
