# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

import subprocess


def test_uq_box_script() -> None:
    subprocess.run(["uq-box", "--help"], check=True)
