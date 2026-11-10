"""Pytest bootstrap for running browser tests in a minimal environment.

Adds the repository source trees to ``sys.path`` and stubs the optional
heavyweight dependencies that ``RPA.Browser.Selenium`` imports at module
load time but which are irrelevant to the session-management unit tests.
"""

import importlib
import sys
import types
from pathlib import Path

MAIN_SRC = Path(__file__).resolve().parents[2] / "src"
CORE_SRC = Path(__file__).resolve().parents[3] / "core" / "src"
for path in (MAIN_SRC, CORE_SRC):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

# `RPA.core.notebook` pulls in optional notebook tooling; only the
# IPYTHON_AVAILABLE flag is read at import time.
try:
    importlib.import_module("RPA.core.notebook")
except Exception:  # pylint: disable=broad-except
    notebook = types.ModuleType("RPA.core.notebook")
    notebook.IPYTHON_AVAILABLE = False
    sys.modules["RPA.core.notebook"] = notebook

# `RPA.core.webdriver` requires requests + webdriver_manager, which are
# irrelevant to the session-management tests; the real module is kept when
# importable so browser-launching tests run unchanged.
try:
    importlib.import_module("RPA.core.webdriver")
except Exception:  # pylint: disable=broad-except
    core_webdriver = types.ModuleType("RPA.core.webdriver")

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "Browser webdriver download is unavailable in the stripped test"
            " environment."
        )

    core_webdriver.download = _unavailable
    core_webdriver.get_browser_version = lambda *a, **k: None
    sys.modules["RPA.core.webdriver"] = core_webdriver

# `RPA.Robocorp.utils` lives in the main source tree but imports vault and
# work-item packages that may be absent in a stripped-down environment.
try:
    importlib.import_module("RPA.Robocorp.utils")
except Exception:  # pylint: disable=broad-except
    robocorp = types.ModuleType("RPA.Robocorp")
    robocorp_utils = types.ModuleType("RPA.Robocorp.utils")

    def get_output_dir(default="output"):
        return Path(default)

    robocorp_utils.get_output_dir = get_output_dir
    robocorp.utils = robocorp_utils
    sys.modules["RPA.Robocorp"] = robocorp
    sys.modules["RPA.Robocorp.utils"] = robocorp_utils
