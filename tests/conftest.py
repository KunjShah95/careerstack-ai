"""Test-wide fixtures.

The important one is isolate_data_dir: it points app/store.py at a
temporary directory for the whole test session, so tests never write into
the real ./data/.

This is not hypothetical tidiness. Before this existed, running the suite
wrote real cache entries into ./data/role_profiles/ -- including a
"backend developer"/"Ahmedabad" profile containing a single skill
("python"), seeded by a route test. The demo resume has Python, so its
keyword coverage scored 1/1 = 100%, and demo_before.pdf came out at 81.7
instead of its true 45.4. The prep_demo numbers and the numbers the
running app produced silently disagreed, and nothing failed to flag it.

Tests that deliberately exercise the cache (test_cache.py,
test_analyze_degraded.py, the discovery route tests) still write real
files -- they just write them somewhere disposable now.
"""

import pytest

from app import store


@pytest.fixture(autouse=True, scope="session")
def isolate_data_dir(tmp_path_factory):
    """Redirect store.DATA_DIR at a temp directory for the whole session.

    Session-scoped rather than per-test: several tests write a record in
    one step and read it back in another through the service layer, so a
    per-test directory would break them for no benefit. The goal is
    isolation from ./data/, not isolation between tests.
    """
    original = store.DATA_DIR
    store.DATA_DIR = tmp_path_factory.mktemp("data")
    yield
    store.DATA_DIR = original
