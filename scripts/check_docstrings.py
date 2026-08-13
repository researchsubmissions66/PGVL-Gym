#!/usr/bin/env python3
"""Check docstrings on the framework's stable public API without imports.

The project contains vendored paper implementations whose internal functions
are intentionally outside the stable framework API. This check targets the
registry, adapter classes, encoder contracts, shared feature loaders, prompt
compiler, and composite extension interfaces rendered by MkDocs.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_API_FILES = (
    Path("methods/__init__.py"),
    Path("methods/base.py"),
    Path("common/backbones/interfaces.py"),
    Path("common/backbones/factory.py"),
    Path("common/datasets/bag_features.py"),
    Path("common/datasets/slide_embeddings.py"),
    Path("common/prompts/compiler.py"),
    Path("common/composite/interfaces.py"),
    Path("common/composite/model.py"),
    Path("methods/composite/adapter.py"),
    Path("methods/focus/adapter.py"),
    Path("methods/vila_mil/adapter.py"),
    Path("methods/cod_mil/adapter.py"),
    Path("methods/maple/adapter.py"),
    Path("methods/mscpt/adapter.py"),
    Path("methods/pathpt/adapter.py"),
    Path("methods/top/adapter.py"),
    Path("methods/slip/adapter.py"),
    Path("methods/wsi_five/adapter.py"),
    Path("methods/muse/adapter.py"),
    Path("methods/convlm/adapter.py"),
    Path("methods/sldpc/adapter.py"),
)


def audit_file(path: Path) -> list[str]:
    """Return missing module and top-level public-symbol docstrings."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    missing: list[str] = []
    if not ast.get_docstring(tree):
        missing.append(f"{path}: module")
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        if not ast.get_docstring(node):
            missing.append(f"{path}:{node.lineno}: {node.name}")
    return missing


def audit_paths(paths: Iterable[Path] = PUBLIC_API_FILES) -> list[str]:
    """Audit repository-relative API files and return all missing entries."""
    missing: list[str] = []
    for relative in paths:
        path = REPO_ROOT / relative
        if not path.is_file():
            missing.append(f"{relative}: file not found")
            continue
        missing.extend(audit_file(path))
    return missing


def main() -> int:
    """Run the public API audit and return a command-line exit status."""
    missing = audit_paths()
    if missing:
        print("Missing public API docstrings:")
        for item in missing:
            print(f"  - {item}")
        return 1
    print(f"Public API docstrings: {len(PUBLIC_API_FILES)} files passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
