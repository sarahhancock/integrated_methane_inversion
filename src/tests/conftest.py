"""Make the IMI source importable for the test suite.

Adds the repo root (so ``src.inversion_scripts.*`` / ``src.utilities.*`` absolute imports resolve),
plus ``src/inversion_scripts`` and ``src/utilities`` (so the builders' bare ``import utils`` /
``import config_utils`` fallbacks resolve), to sys.path.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# NOTE ordering: src/inversion_scripts must end up BEFORE src/utilities on sys.path, because both
# contain a `utils.py` and the builders do a bare `import utils` expecting inversion_scripts/utils.py
# (get_mean_emissions, ensure_float_list). insert(0, ...) prepends, so insert inversion_scripts LAST.
for _p in (
    _ROOT,
    os.path.join(_ROOT, "src", "utilities"),
    os.path.join(_ROOT, "src", "inversion_scripts"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
