import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from test_adapter import fixture


SCRIPT = Path(__file__).resolve().parents[1] / "predict.py"


def test_cli_prepares_only_visited_views_and_reuses_saved_frame(tmp_path):
    fusion, raw = fixture(tmp_path)
    output = tmp_path / "output"
    base = [sys.executable, str(SCRIPT), "--fusion-dir", str(fusion),
            "--raw-session", str(raw), "--output-dir", str(output),
            "--prepare-only"]
    first = subprocess.run(base + ["--visited", "0", "6",
                                   "--table-height-axis-mm", "0"],
                           capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["observed_points"] == 3
    saved = np.load(output / "observed.npy")
    assert saved.shape == (3, 3)
    second = subprocess.run(base + ["--visited", "0", "6", "15"],
                            capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["observed_points"] == 4
    np.testing.assert_allclose(np.load(output / "observed.npy")[:3], saved)
    assert json.loads((output / "object_frame.json").read_text(encoding="utf-8"))["initial_views"] == [0, 6]


def test_prepare_only_does_not_import_torch(tmp_path):
    fusion, raw = fixture(tmp_path)
    output = tmp_path / "prepare_only"
    script = ("import runpy,sys,pathlib; target=sys.argv[4]; "
              "sys.path.insert(0,str(pathlib.Path(target).parent)); sys.modules['torch']=None; "
              "sys.argv=['predict.py','--fusion-dir',sys.argv[1],"
              "'--raw-session',sys.argv[2],'--visited','0','6',"
              "'--table-height-axis-mm','0','--output-dir',sys.argv[3],"
              "'--prepare-only']; runpy.run_path(target,run_name='__main__')")
    result = subprocess.run([sys.executable, "-c", script, str(fusion), str(raw),
                             str(output), str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
