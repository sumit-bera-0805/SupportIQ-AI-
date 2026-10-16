"""Headless render of every page with Streamlit's AppTest."""
import glob
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
FILES = ["app.py"] + sorted(str(Path(p).relative_to(ROOT)) for p in glob.glob(str(ROOT / "pages" / "*.py")))


@pytest.mark.parametrize("rel", FILES)
def test_page_renders_without_exception(rel):
    at = AppTest.from_file(str(ROOT / rel), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_analyzer_end_to_end():
    at = AppTest.from_file(str(ROOT / "pages/3_AI_Ticket_Analyzer.py"), default_timeout=180).run()
    at.text_input(key="an_subject").set_value("Charged twice for my subscription")
    at.text_area(key="an_desc").set_value("Please refund the duplicate payment on my invoice.")
    next(b for b in at.button if b.label == "Analyze ticket").click().run()
    assert not at.exception
    assert any("Billing" in m.value for m in at.markdown)
