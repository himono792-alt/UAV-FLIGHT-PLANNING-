"""Launcher for the generated HCM 3D UAV demo.

The maintained browser artifact is ``outputs/hcm_3d_demo.html``. This file is
kept as a safe entry point so commands that reference ``demo_hcm_3d.py`` do not
fail on a corrupted source file.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_OUTPUT = Path("outputs/hcm_3d_demo.html")
DEFAULT_URL = "http://127.0.0.1:8765/outputs/hcm_3d_demo.html"


def main() -> int:
    parser = argparse.ArgumentParser(description="Show the HCM 3D UAV demo artifact.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to the generated HTML demo artifact.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Local URL to open after starting a static server.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if not output.exists():
        raise SystemExit(f"Demo HTML not found: {output}")

    print(f"Demo HTML: {output.resolve()}")
    print(f"Local URL: {args.url}")
    print("Run a static server from the repo root, for example:")
    print("  python -m http.server 8765 --bind 127.0.0.1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
