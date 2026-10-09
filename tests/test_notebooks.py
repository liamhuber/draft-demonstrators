import os
import pathlib
import shutil
import tempfile
import unittest

import papermill

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_RESOURCES = _REPO_ROOT / "resources"
_RESULTS = _REPO_ROOT / "results"


class TestNotebooks(unittest.TestCase):
    """
    Execute the demonstrator notebooks end to end.

    Each notebook runs in a fresh temporary directory so that run artefacts
    (e.g. `demo_runs/`) stay out of the repo and cached results (e.g.
    `results/phase_run.h5`) are not reused.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work_dir = pathlib.Path(self._tmp.name)
        shutil.copytree(_RESOURCES, self.work_dir / _RESOURCES.name)
        shutil.copytree(_RESULTS, self.work_dir / _RESULTS.name)

        self._old_pythonpath = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(_REPO_ROOT), self._old_pythonpath) if p
        )

    def tearDown(self):
        if self._old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = self._old_pythonpath
        self._tmp.cleanup()

    def run_notebook(self, name: str, parameters: dict | None = None):
        papermill.execute_notebook(
            _REPO_ROOT / f"{name}.ipynb",
            self.work_dir / f"{name}.out.ipynb",
            cwd=self.work_dir,
            kernel_name="python3",
            parameters=parameters,
            progress_bar=False,
        )

    def test_elastic(self):
        self.run_notebook("ex1-elastic")

    def test_grain_boundary(self):
        self.run_notebook(
            "ex2-grain_boundary",
            parameters={"cheap": True, "use_result": False, "write_result": False}
        )

    def test_phase_stability(self):
        self.run_notebook(
            "ex3-phase_stability",
            parameters={"cheap": True, "use_result": False, "write_result": False}
        )

    def test_grain_boundary_loading(self):
        self.run_notebook(
            "ex2-grain_boundary",
            parameters={"cheap": True, "use_result": True, "write_result": False}
        )

    def test_phase_stability_loading(self):
        self.run_notebook(
            "ex3-phase_stability",
            parameters={"cheap": True, "use_result": True, "write_result": False}
        )


if __name__ == "__main__":
    unittest.main()
