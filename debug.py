"""
debug.py - one on/off switch for all real-time debug prints.

Every module calls dbg(...) instead of print(...) for developer-facing trace
output ("Speech detected", "Driver is drowsy", yawn timestamps, state changes).
main.py flips the switch on with --debug, so a single flag silences or enables
ALL of it at once - the noisy prints stay out of normal runs, and "what is the
system seeing right now?" becomes a one-flag question.

Why a module-level flag (not a config constant): dbg() reads this module's
global at call time, and set_enabled() mutates that same global, so turning
debug on in main is seen everywhere immediately without import-order worries.
"""

# Off by default so normal runs stay quiet. main.set_enabled() turns it on.
_ENABLED = False


def set_enabled(on):
    """Turn the debug trace on or off (called once from main with --debug)."""
    global _ENABLED
    _ENABLED = bool(on)


def is_enabled():
    return _ENABLED


def dbg(msg):
    """Print one debug line, but only when debug mode is on."""
    if _ENABLED:
        print(f"[dbg] {msg}", flush=True)
