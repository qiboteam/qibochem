"""Pytest config.

Pre-imports numpy and scipy.optimize before any qibochem module is touched
to avoid a known interaction between pytest-cov instrumentation, numpy 2.x's
``_CopyMode`` enum, and editable installs that otherwise causes
``ValueError: _CopyMode.IF_NEEDED is neither True nor False`` mid-suite.
"""

import numpy  # noqa: F401
import scipy.optimize  # noqa: F401
