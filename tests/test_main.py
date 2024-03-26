# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

import subprocess
import sys


def test_help() -> None:
    subprocess.run([sys.executable, "-m", "uq-box", "--help"], check=True)
