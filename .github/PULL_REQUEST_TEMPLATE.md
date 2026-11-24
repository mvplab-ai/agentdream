## What

<!-- A short description of the change. -->

## Why

<!-- The bug, limitation, or use case this addresses. Link the issue if there is one. -->

## How you checked

- [ ] `python -m pytest -q`
- [ ] `ruff check src tests examples`
- [ ] `mypy`
- [ ] Docs updated if the change is user-visible (`README.md` or `docs/`)

The core must stay stdlib-only. If you added an adapter or backend, include tests and (for adapters) the hash-before/after check in `tests/test_safety.py`.
