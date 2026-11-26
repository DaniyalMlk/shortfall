"""Run `--help` on the top-level parser and every subcommand.

argparse formats help strings with %-substitution at call time, so a literal
"%" in a help string only fails when someone asks for help. This catches that
in CI instead of on a user's terminal.
"""
from __future__ import annotations

import re
import subprocess
import sys

MODULE = "shortfall.cli"


def main() -> int:
    top = subprocess.run([sys.executable, "-m", MODULE, "--help"], capture_output=True, text=True)
    if top.returncode != 0:
        print(top.stderr)
        return 1
    match = re.search(r"\{([a-z0-9,_-]+)\}", top.stdout)
    subcommands = match.group(1).split(",") if match else []
    failures = []
    for name in subcommands:
        result = subprocess.run(
            [sys.executable, "-m", MODULE, name, "--help"], capture_output=True, text=True
        )
        if result.returncode != 0:
            lines = result.stderr.strip().splitlines()
            failures.append((name, lines[-1] if lines else "no output"))
    for name, error in failures:
        print(f"{name} --help failed: {error}")
    print(f"{len(subcommands)} subcommands, {len(failures)} failing")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
