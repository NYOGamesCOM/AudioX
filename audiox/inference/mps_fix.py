import torch
import platform
from contextlib import nullcontext

def get_autocast_context(device_type: str = None):
    """
    Returns a device-agnostic autocast context manager.
    Supports CUDA, MPS, and CPU.
    """
    if device_type is None:
        if torch.cuda.is_available():
            device_type = "cuda"
        elif platform.system() == "Darwin" and torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"

    if device_type == "cuda":
        return torch.cuda.amp.autocast()
    elif device_type == "mps":
        # MPS doesn't support amp.autocast in the same way yet, 
        # but we can return a nullcontext or handle specific local stability.
        # Actually, modern PyTorch has torch.autocast(device_type="mps", ...) 
        # in recent versions, but it might be unstable or missing.
        try:
            return torch.autocast(device_type="mps")
        except (AttributeError, RuntimeError):
            return nullcontext()
    else:
        return nullcontext()
