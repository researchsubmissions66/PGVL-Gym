"""Smoke-test that every method's adapter can be imported.

This does NOT load model weights or run forward passes -- it only
verifies that the registry, common modules, and adapter files are
all in good shape.

Useful when you've just modified `methods/__init__.py` or an adapter
and want a 1-second check.
"""
import importlib
import traceback
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from methods import list_methods, get_method


def main():
    print("=== sanity check: importing every adapter ===\n")
    failures = []
    for name in list_methods():
        try:
            cls = get_method(name)
            print(f"  {name:12s} OK   ({cls.__module__}.{cls.__name__})")
        except Exception as e:
            failures.append((name, e))
            print(f"  {name:12s} FAIL: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            print()

    print(f"\n{len(list_methods()) - len(failures)} / "
          f"{len(list_methods())} adapters importable")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
