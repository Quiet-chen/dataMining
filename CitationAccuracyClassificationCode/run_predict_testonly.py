"""Run multivers/predict.py but build only the test fold.

`predict.py` calls `ConcatDataModule.setup()`, which tokenizes train and dev as
well, even though inference only reads test. This wrapper patches `setup()` for
this process only. It does not modify repository code or change the test-fold
construction, so predictions are identical to the stock entry point; it merely
skips preprocessing that inference never uses.

Usage: same arguments as `multivers/predict.py`.
"""

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR / "multivers"))

import data_train as dm  # noqa: E402


def setup_test_only(self, stage=None):
    self.folds = {"test": self._process_fold("test")}


dm.ConcatDataModule.setup = setup_test_only

import predict  # noqa: E402

sys.argv = ["predict.py", *sys.argv[1:]]
predict.main()
