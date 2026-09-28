import subprocess
import sys


def test_standalone_pipeline_script_runs():
    result = subprocess.run(
        [sys.executable, "src/climate_xai/pipeline.py"],
        cwd="/workspaces/weather-prediction",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Forecast rows:" in result.stdout
