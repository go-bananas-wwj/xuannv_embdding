from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_direct_launcher_clears_proxy_environment() -> None:
    launcher = Path(__file__).parents[1] / "scripts/data/run_china_v1_direct.sh"
    result = subprocess.run(
        [str(launcher), "bash", "-c", "printf '%s|%s|%s' \"${HTTPS_PROXY-unset}\" \"${http_proxy-unset}\" \"${NO_PROXY-unset}\""],
        env={**os.environ, "HTTPS_PROXY": "http://127.0.0.1:7891", "http_proxy": "http://127.0.0.1:7891"},
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout == "unset|unset|*"
