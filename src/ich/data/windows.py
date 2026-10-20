"""Fixed HU windows, the inputs of the image models."""
import numpy as np

# (level, width) in HU. Never the window in the file header, which is unreliable (01).
WINDOWS = {"brain": (40, 80), "subdural": (80, 200), "bone": (600, 2800)}
# Windows the image models may use besides those three (04): soft tissue, about twice as wide as subdural, the third
# window of the 2nd place. Kept apart from WINDOWS, whose three windows the stored formats of 02 are built from.
MODEL_WINDOWS = {**WINDOWS, "soft_tissue": (40, 380)}


def window(hu: np.ndarray, level: float, width: float) -> np.ndarray:
    """HU to float32 in [0, 1]: 0 up to level - width / 2, 1 from level + width / 2."""
    return np.clip((np.asarray(hu, dtype=np.float32) - (level - width / 2)) / width, 0, 1)


def to_uint8(x: np.ndarray) -> np.ndarray:
    """[0, 1] to 0-255, rounded."""
    return np.rint(x * 255).astype(np.uint8)
