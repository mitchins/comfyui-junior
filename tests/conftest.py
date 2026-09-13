import os
import sys
from pathlib import Path

# Add src to sys.path
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# GPU-less unit-test environment: pin an explicit stack (auto-detection needs
# a CUDA device) and exercise the appliance role unless a test overrides it.
os.environ.setdefault("JUNIOR_IMAGE_STACK", "ampere_fp8")
os.environ.setdefault("JUNIOR_EVALUATION_INSTANCE", "0")
