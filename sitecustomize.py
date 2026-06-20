"""Project-local import customization.

The ``downstreams`` package source lives at ``downstreams/downstreams/``
(following the same nested layout as ``src/<package>/``). When Python is
started from the project root, the top-level ``downstreams/`` directory is on
``sys.path`` and is interpreted as a namespace package, shadowing the real
editable-install mapping.

This module moves the editable-install finder to the front of
``sys.meta_path`` so that ``import downstreams`` resolves to the actual package
instead of the namespace shadow. It only runs when the project root happens to
be on ``sys.path`` (i.e. when working from this directory).
"""
import sys


def _reorder_editable_finder():
    target = None
    for finder in sys.meta_path:
        mod = getattr(finder, "__module__", "")
        if mod.startswith("__editable_") and mod.endswith("_finder"):
            target = finder
            break
        mapping = getattr(finder, "MAPPING", {})
        if isinstance(mapping, dict) and "downstreams" in mapping:
            target = finder
            break

    if target is None:
        return

    for idx, finder in enumerate(sys.meta_path):
        if finder is target:
            sys.meta_path.insert(0, sys.meta_path.pop(idx))
            break


_reorder_editable_finder()
del _reorder_editable_finder
