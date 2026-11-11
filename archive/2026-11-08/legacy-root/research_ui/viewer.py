"""Lightweight entry: browse current research without loading model dependencies."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research_frontend import render

render(ROOT)
