"""Top-level entry so ``python -m qibochem`` invokes the CLI."""

from qibochem.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
