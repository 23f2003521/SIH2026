"""
Smoke tests: execute every console page and assert nothing raised.

Streamlit's AppTest runs the real script in-process, so these catch the kind of
runtime error a page only shows once a human clicks onto it.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / "ui" / "app.py")
PAGES = ["Overview", "AIS Anomaly Detection", "Route Deviation (LSTM)",
         "SAR Oil Spill Segmenter"]


def _run(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, f"{page}: initial run raised {at.exception}"
    at.radio[0].set_value(page).run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page: str) -> None:
    at = _run(page)
    assert not at.exception, f"{page} raised: {at.exception}"


def test_overview_names_the_suspect() -> None:
    at = _run("Overview")
    body = " ".join(m.value for m in at.markdown)
    assert "WAKASHIO" in body
