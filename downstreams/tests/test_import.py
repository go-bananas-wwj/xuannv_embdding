import sys
from pathlib import Path


def test_import_downstreams():
    """The downstream package should be importable as a real package.

    The package source lives at ``downstreams/downstreams/``. When tests are
    executed from the project root, the top-level ``downstreams/`` directory
    shadows the installed package as a namespace package. We temporarily remove
    the project root from ``sys.path`` so the editable-install mapping is used.
    """
    project_root = Path(__file__).resolve().parents[2]
    original_path = sys.path[:]
    sys.path = [
        p
        for p in sys.path
        if p != "" and Path(p).resolve() != project_root
    ]
    try:
        import downstreams

        assert downstreams.__file__ is not None
        assert downstreams.__file__.endswith("downstreams/downstreams/__init__.py")

        import downstreams.data
        import downstreams.heads
        import downstreams.metrics
        import downstreams.tasks
        import downstreams.utils

        for sub in (downstreams.data, downstreams.heads, downstreams.metrics, downstreams.tasks, downstreams.utils):
            assert sub.__file__ is not None
    finally:
        sys.path[:] = original_path
