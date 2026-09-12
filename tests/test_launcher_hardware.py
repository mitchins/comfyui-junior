"""Hardware-contract tests for the stack-aware launcher (torch mocked)."""
import unittest
from unittest.mock import MagicMock, patch

from comfyui_junior import launcher


def _cuda_mock(major, minor):
    """Build a torch stand-in reporting the given compute capability."""
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = True
    torch_mock.cuda.get_device_name.return_value = f"Mock GPU SM{major}{minor}"
    torch_mock.cuda.get_device_capability.return_value = (major, minor)
    torch_mock.cuda.get_device_properties.return_value = MagicMock(total_memory=12 * 1024 ** 3)
    return torch_mock


class TestHardwareContract(unittest.TestCase):
    """Per-stack compute-capability enforcement."""

    def test_nvfp4_requires_sm120(self):
        """blackwell_nvfp4 rejects anything below SM120, including Ada/Ampere."""
        for major, minor in ((12, 0),):
            with patch.object(launcher, "torch", _cuda_mock(major, minor)):
                launcher.check_hardware_environment("blackwell_nvfp4")  # must not raise
        for major, minor in ((8, 6), (8, 9), (9, 0)):
            with patch.object(launcher, "torch", _cuda_mock(major, minor)):
                with self.assertRaises(RuntimeError):
                    launcher.check_hardware_environment("blackwell_nvfp4")

    def test_ampere_fp8_requires_sm86_or_newer(self):
        """ampere_fp8 accepts SM86+ and rejects SM80 (A100) and older."""
        for major, minor in ((8, 6), (8, 9), (9, 0), (12, 0)):
            with patch.object(launcher, "torch", _cuda_mock(major, minor)):
                launcher.check_hardware_environment("ampere_fp8")  # must not raise
        for major, minor in ((8, 0), (7, 5)):
            with patch.object(launcher, "torch", _cuda_mock(major, minor)):
                with self.assertRaises(RuntimeError):
                    launcher.check_hardware_environment("ampere_fp8")

    def test_no_cuda_rejected(self):
        """Without CUDA the supervisor refuses to start for any stack."""
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = False
        with patch.object(launcher, "torch", torch_mock):
            for stack in ("blackwell_nvfp4", "ampere_fp8"):
                with self.assertRaises(RuntimeError):
                    launcher.check_hardware_environment(stack)


class TestAllocatorDefaults(unittest.TestCase):
    """Per-stack allocator ceiling defaults."""

    def test_per_stack_defaults(self):
        """Unset env keeps the Blackwell 10 GiB fence and no Ampere cap."""
        import os
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(launcher.effective_allocator_cap_gib("blackwell_nvfp4"), 10.0)
            self.assertEqual(launcher.effective_allocator_cap_gib("ampere_fp8"), 0.0)

    def test_env_override_wins(self):
        """An explicit COMFY_MEMORY_CAP_GIB overrides the per-stack default."""
        import os
        with patch.dict(os.environ, {"COMFY_MEMORY_CAP_GIB": "7.5"}, clear=True):
            self.assertEqual(launcher.effective_allocator_cap_gib("ampere_fp8"), 7.5)


if __name__ == "__main__":
    unittest.main()
