"""Top-level entry so ``python -m qibochem`` invokes the CLI."""

# This shim is exercised only via ``python -m qibochem``, which spawns a fresh
# subprocess that pytest's coverage tracker cannot follow.
from qibochem.cli.main import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
