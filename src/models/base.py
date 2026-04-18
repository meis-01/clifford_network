from __future__ import annotations


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True
