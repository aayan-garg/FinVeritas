#!/usr/bin/env python3
"""
FinVeritas: CLI entry point for EvasionBench baseline experiments.
Thin wrapper around finveritas.nlp_evasion.baselines.main.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from finveritas.nlp_evasion.baselines import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
