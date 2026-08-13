"""Documentation structure and public API docstring tests."""
from pathlib import Path

import yaml

from scripts.check_docstrings import audit_paths


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stable_public_api_has_docstrings():
    """All symbols rendered in the stable API reference have docstrings."""
    assert audit_paths() == []


def test_mkdocs_navigation_targets_exist():
    """Every Markdown page named in the MkDocs navigation exists."""
    config = yaml.safe_load((REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    docs_root = REPO_ROOT / config.get("docs_dir", "docs")

    def targets(items):
        for item in items:
            if isinstance(item, str):
                yield item
            elif isinstance(item, dict):
                for value in item.values():
                    if isinstance(value, str):
                        yield value
                    else:
                        yield from targets(value)

    missing = [target for target in targets(config["nav"])
               if not (docs_root / target).is_file()]
    assert missing == []
