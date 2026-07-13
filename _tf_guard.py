"""
_tf_guard.py - keep MediaPipe importable in this environment.

THE PROBLEM (specific to the shared interpreter this project runs under):
TensorFlow 2.16 is installed alongside MediaPipe. MediaPipe's
`tasks/python/core/optional_dependencies.py` does:

    try:
        from tensorflow.tools.docs import doc_controls
    except ModuleNotFoundError:
        ...                      # graceful fallback if TF is absent

It only catches ModuleNotFoundError. But because TF *is* installed, that import
starts loading TensorFlow, which pulls in jax -> ml_dtypes and crashes with an
AttributeError (`module 'ml_dtypes' has no attribute 'float8_e3m4'`). An
AttributeError is NOT a ModuleNotFoundError, so the fallback can't catch it and
`import mediapipe` dies - taking the whole video pipeline down with it.

THE FIX (self-contained, no system changes):
Install a meta-path finder that makes `import tensorflow` raise
ModuleNotFoundError. MediaPipe then takes its intended "TF absent" branch and
imports cleanly. We touch nothing outside this folder and uninstall nothing -
import this guard BEFORE importing mediapipe and the problem is gone.

(The alternative durable fix is `pip uninstall -y tensorflow tensorflow-hub` in
the shared env; that's a global change and is left to the user - see README.)
"""
import importlib.abc
import sys


class _BlockTensorFlow(importlib.abc.MetaPathFinder):
    """Raises ModuleNotFoundError for any `tensorflow[...]` import."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "tensorflow" or fullname.startswith("tensorflow."):
            raise ModuleNotFoundError(
                f"{fullname} import blocked by alertVSdrowsy/_tf_guard so "
                "MediaPipe stays importable (TensorFlow breaks it in this env)."
            )
        return None  # not our concern -> let normal import proceed


def install():
    """Block TensorFlow imports. Safe to call more than once."""
    # Drop any half-loaded tensorflow modules so the block takes effect cleanly.
    for name in [n for n in sys.modules
                 if n == "tensorflow" or n.startswith("tensorflow.")]:
        del sys.modules[name]
    if not any(isinstance(f, _BlockTensorFlow) for f in sys.meta_path):
        sys.meta_path.insert(0, _BlockTensorFlow())


# Installing on import means `import _tf_guard` is enough - no need to remember
# to call install(). Modules that need MediaPipe simply import this first.
install()
