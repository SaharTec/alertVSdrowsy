"""
alert.py - warn the driver out loud when drowsiness is confirmed.

Turns the fused verdict into spoken alarms, by these rules:
  * The fused state must stay DROWSY continuously for DROWSY_HOLD_SECONDS before
    we speak - so a single noisy window never triggers a false alarm.
  * While it remains DROWSY we repeat the warning, but no more often than
    ALERT_COOLDOWN_SECONDS apart, so it never spams.
  * Once the driver is back to ALERT for RECOVERY_SECONDS, everything resets and
    the alarm can arm again from scratch.

Speech is produced by **pyttsx3** (offline text-to-speech) on a BACKGROUND
thread, so the video loop never blocks while the warning is spoken. If pyttsx3
is missing or fails, we fall back to a **winsound.Beep**, so an alarm is always
audible. This mirrors the parent project's streak/cooldown/background-player
pattern (src/alerts.py), simplified to the single binary DROWSY state.

All timing is wall-clock (time.time()), so it does not depend on frame rate.
"""
import queue
import threading
import time

from config import (
    STATE_ALERT, STATE_DROWSY, STATE_NEUTRAL,
    DROWSY_HOLD_SECONDS, ALERT_COOLDOWN_SECONDS, RECOVERY_SECONDS,
    ALERT_MESSAGE, YAWN_ALERT_MESSAGE, BEEP_FREQ_HZ, BEEP_MS,
)

try:
    import winsound  # Windows only; this project runs on Windows
except ImportError:  # pragma: no cover - keeps import working off-Windows
    winsound = None

STREAK_GRACE_SECONDS = 0.5     # ignore a brief flicker out of DROWSY
BANNER_SECONDS = 4.0           # how long the on-screen warning text lingers


class _Streak:
    """How long a condition has been continuously true (a short gap <= grace
    doesn't break it, absorbing per-frame jitter)."""

    def __init__(self, grace):
        self.grace = grace
        self.start = None
        self.last_active = None

    def update(self, active, now):
        if active:
            if self.start is None:
                self.start = now
            self.last_active = now
        elif self.last_active is not None and now - self.last_active > self.grace:
            self.start = None
            self.last_active = None

    def duration(self, now):
        return 0.0 if self.start is None else now - self.start

    def reset(self):
        self.start = None
        self.last_active = None


class VoiceAlerter:
    """Speaks queued messages on a background thread (non-blocking caller).

    A fresh pyttsx3 engine is created per utterance: pyttsx3's runAndWait() is
    notoriously unhappy being reused across calls, and re-init is cheap relative
    to the 8 s cooldown between alarms. Any failure falls back to a beep.
    """

    def __init__(self):
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def say(self, text):
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._q.put(text)

    def _worker(self):
        while True:
            text = self._q.get()
            if not self._speak(text):
                self._beep()

    @staticmethod
    def _speak(text):
        try:
            import pyttsx3  # lazy: missing TTS must never break import
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            return True
        except Exception as exc:  # noqa: BLE001 - any TTS problem -> beep instead
            print(f"[alert] TTS unavailable ({exc}); beeping instead.")
            return False

    @staticmethod
    def _beep():
        if winsound is not None:
            try:
                winsound.Beep(BEEP_FREQ_HZ, BEEP_MS)
                return
            except Exception:  # noqa: BLE001
                pass
        print("\a", end="", flush=True)  # terminal bell fallback


class AlertManager:
    """Call update(fused_state) every frame; it raises spoken alarms by the rules
    and returns the current on-screen banner text ('' when none)."""

    def __init__(self, voice=None, message=ALERT_MESSAGE):
        self.voice = voice or VoiceAlerter()
        self.message = message
        self.drowsy_streak = _Streak(STREAK_GRACE_SECONDS)
        self.recovery_streak = _Streak(0.0)
        self.armed = False          # True once the threshold has been met
        self.last_fire = 0.0
        self.banner = ""
        self.banner_until = 0.0
        self.fire_count = 0         # exposed for tests / logging

    def _fire(self, now, message=None):
        msg = message or self.message
        self.armed = True
        self.last_fire = now
        self.fire_count += 1
        self.voice.say(msg)
        self.banner = "DROWSY - WAKE UP & TAKE A REST"
        self.banner_until = now + BANNER_SECONDS
        print(f"[ALERT] {msg}")

    def _reset(self):
        self.armed = False
        self.drowsy_streak.reset()
        self.banner = ""
        self.banner_until = 0.0

    def update(self, state, reasons=(), now=None):
        if now is None:
            now = time.time()

        is_drowsy = state == STATE_DROWSY
        not_drowsy = state != STATE_DROWSY

        # Recovery: NOT drowsy long enough (ALERT *or* NEUTRAL both count) ->
        # disarm and clear timers, so the alarm arms again from scratch next time.
        self.recovery_streak.update(not_drowsy, now)
        if not_drowsy and self.recovery_streak.duration(now) >= RECOVERY_SECONDS:
            self._reset()

        # Pick the spoken message by CAUSE: a yawn-driven drowsy state gets the
        # yawn warning; eye-closure / head-nod drowsiness gets the generic one.
        # `reasons` is the fused verdict's reason list (e.g. "yawning 100%").
        msg = (YAWN_ALERT_MESSAGE if any("yawn" in str(r).lower() for r in reasons)
               else self.message)

        # Drowsy: hold the threshold to fire, then repeat on the cooldown.
        self.drowsy_streak.update(is_drowsy, now)
        if is_drowsy:
            if not self.armed:
                if self.drowsy_streak.duration(now) >= DROWSY_HOLD_SECONDS:
                    self._fire(now, msg)
            elif now - self.last_fire >= ALERT_COOLDOWN_SECONDS:
                self._fire(now, msg)

        if now > self.banner_until:
            self.banner = ""
        return self.banner


# ---------------------------------------------------------------------------
# Self-test: a fake clock + a fake voice (records messages instead of speaking)
# so we can verify the timing rules without making any sound. Run: python alert.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    class _FakeVoice:
        def __init__(self):
            self.spoken = []

        def say(self, text):
            self.spoken.append(text)

    voice = _FakeVoice()
    mgr = AlertManager(voice=voice)

    t = 0.0
    # 1) Drowsy for 3 s -> should fire exactly once (after the 1.5 s hold).
    while t < 3.0:
        mgr.update(STATE_DROWSY, now=t)
        t += 0.1
    fired_after_hold = len(voice.spoken)

    # 2) Keep drowsy past the 8 s cooldown -> one repeat.
    while t < 12.0:
        mgr.update(STATE_DROWSY, now=t)
        t += 0.1
    fired_after_cooldown = len(voice.spoken)

    # 3) Back to ALERT for >RECOVERY, then drowsy again -> re-arms and fires anew.
    while t < 15.0:
        mgr.update(STATE_ALERT, now=t)
        t += 0.1
    while t < 17.0:
        mgr.update(STATE_DROWSY, now=t)
        t += 0.1
    fired_after_rearm = len(voice.spoken)

    # 4) The real-world recovery the other cases miss: a quiet but AWAKE driver
    #    reads as NEUTRAL, not ALERT. NEUTRAL must still disarm, so a later BRIEF
    #    drowsy flicker waits the full HOLD again instead of insta-firing off the
    #    cooldown. This is the latch bug's fingerprint.
    voice4 = _FakeVoice()
    mgr4 = AlertManager(voice=voice4)
    t = 0.0
    while t < 2.0:                       # drowsy 2 s -> fires once at t=1.5
        mgr4.update(STATE_DROWSY, now=t); t += 0.1
    fired = len(voice4.spoken)           # expect 1

    while t < 11.0:                      # NEUTRAL 9 s: past RECOVERY (1.5) AND past cooldown (8)
        mgr4.update(STATE_NEUTRAL, now=t); t += 0.1

    while t < 11.5:                      # a 0.5 s drowsy blip — shorter than the 1.5 s HOLD
        mgr4.update(STATE_DROWSY, now=t); t += 0.1
    fired_after_blip = len(voice4.spoken)

    print(f"  neutral recovery: fired={fired}, after short blip={fired_after_blip} (expect 1 and 1)")
    neutral_ok = (fired == 1 and fired_after_blip == 1)


    print("alert self-test:")
    print(f"  after 1.5 s hold : fired={fired_after_hold} (expect 1)")
    print(f"  after 8 s cooldown: fired={fired_after_cooldown} (expect 2)")
    print(f"  after recovery+re : fired={fired_after_rearm} (expect 3)")
    ok = (fired_after_hold == 1 and fired_after_cooldown == 2
          and fired_after_rearm == 3)
    print("alert self-test", "OK" if ok else "FAILED")
