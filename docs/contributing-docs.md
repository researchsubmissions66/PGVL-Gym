# Writing documentation

The documentation website combines curated Markdown with API pages generated
from Python docstrings.

## Build locally

```bash
python -m pip install -r requirements-docs.txt
python -m mkdocs build --strict
```

For a live preview:

```bash
python -m mkdocs serve
```

The strict build treats broken internal links, invalid navigation entries, and
API rendering problems as failures.

## Docstring convention

Use an imperative one-line summary, then add context and Google-style sections
where they clarify the contract:

```python
def encode_text(texts: Sequence[str], normalize: bool = True) -> Tensor:
    """Encode text in the backbone's paired comparison space.

    Args:
        texts: Ordered text strings to encode.
        normalize: Whether to L2-normalize the returned rows.

    Returns:
        A tensor shaped ``[len(texts), shared_dim]``.

    Raises:
        BackboneCompatibilityError: If text encoding is unavailable.
    """
```

Document semantic tensor shapes, device/dtype behavior, feature-space
assumptions, config keys, and raised validation errors. Avoid repeating types
already present in annotations.

## What belongs in API documentation

- stable registry, adapter, encoder, prompt, and loader interfaces;
- public extension points intended for downstream use;
- invariants that cannot be inferred from the signature.

Vendored paper internals are source-readable but are not part of the stable
framework API. Document their adapter boundary instead.

## Continuous deployment

The documentation workflow builds every pull request in strict mode. Pushes to
`main` or `master` also publish the generated `site/` artifact through GitHub
Pages. Enable **GitHub Actions** as the Pages source once for the repository;
subsequent updates are automatic.
