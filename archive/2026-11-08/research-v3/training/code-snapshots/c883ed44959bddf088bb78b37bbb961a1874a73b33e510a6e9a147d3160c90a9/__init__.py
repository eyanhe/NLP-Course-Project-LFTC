"""Isolated, provenance-checked preparation for text-classification research."""

__version__ = '0.1.0'

# These Windows wheels require Arrow's native dataset extension to initialize
# before Torch is loaded by Transformers; reverse order causes 0xC0000005.
import sys

if sys.platform == 'win32':
    import pyarrow.dataset  # noqa: F401
