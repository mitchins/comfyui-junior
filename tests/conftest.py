import os
import sys
from pathlib import Path
import pytest

# Add src to sys.path
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

os.environ["SAFETY_ENABLED"] = "0"  # Default off for unit tests unless overridden
