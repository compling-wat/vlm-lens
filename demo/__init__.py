"""Demo package."""

from . import _bootstrap  # noqa: F401
from .lookup import get_model_info  # re-export for convenience

__all__ = ['get_model_info']
