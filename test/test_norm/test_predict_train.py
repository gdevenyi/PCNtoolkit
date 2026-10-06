"""Tests for the opt-in NormativeModel option predict_train (#576 row 10)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import BLR

OUTPUTS = ("Z", "centiles", "logp", "Yhat")


def _data(name: str, seed: int, n: int = 60) -> NormData:
    rng = np.random.default_rng(seed)
    X = rng.uniform(20, 80, (n, 1))
    Y = 0.05 * X + rng.normal(0, 1, (n, 2))
    be = rng.choice(["a", "b"], (n, 1))
    return NormData.from_ndarrays(name, X, Y, be)


def _model(save_dir: Path, **kwargs) -> NormativeModel:
    return NormativeModel(BLR(), save_dir=str(save_dir), **kwargs)


def _result_files(save_dir: Path, name: str) -> list[str]:
    return sorted(p.name for p in (save_dir / "results").glob(f"*_{name}.csv"))


def test_fit_should_predictTrainingData_when_default(tmp_path: Path) -> None:
    """
    Arrange: a model with default options.
    Act: fit.
    Assert: the training data has all outputs and result files (as before).
    """
    train = _data("train", 0)
    model = _model(tmp_path)
    assert model.predict_train is True
    model.fit(train)
    assert all(v in train.data_vars for v in OUTPUTS)
    assert _result_files(tmp_path, "train")


def test_fit_should_skipPredict_when_predictTrainIsFalse(tmp_path: Path) -> None:
    """
    Arrange: two models, with predict_train True and False.
    Act: fit both on the same data, then predict the same test data.
    Assert: with False, the training data gets no outputs and no result
        files, the model is saved and fitted, and the test predictions equal
        those of the default model bit for bit.
    """
    train, test = _data("train", 0), _data("test", 1)
    full = _model(tmp_path / "full")
    full.fit(_data("train", 0))
    full.predict(expected := _data("test", 1))
    model = _model(tmp_path / "skip", predict_train=False)
    model.fit(train)
    assert model.is_fitted
    assert not any(v in train.data_vars for v in OUTPUTS)
    assert not _result_files(tmp_path / "skip", "train")
    assert (tmp_path / "skip" / "model" / "normative_model.json").is_file()
    model.predict(test)
    for v in OUTPUTS:
        np.testing.assert_array_equal(test[v].values, expected[v].values, v)


def test_predictTrain_should_beSavedAndLoaded_when_modelIsSaved(tmp_path: Path) -> None:
    """
    Arrange: a model with predict_train=False, fitted and saved.
    Act: load it; then remove the key from the file (an old saved model) and
        load again.
    Assert: False after the first load; True (the old behaviour) without the
        key.
    """
    model = _model(tmp_path, predict_train=False)
    model.fit(_data("train", 0))
    assert NormativeModel.load(str(tmp_path)).predict_train is False
    meta_path = tmp_path / "model" / "normative_model.json"
    meta = json.loads(meta_path.read_text())
    del meta["predict_train"]
    meta_path.write_text(json.dumps(meta))
    assert NormativeModel.load(str(tmp_path)).predict_train is True


def test_fromArgs_should_readPredictTrain_when_givenOnCommandLine() -> None:
    """
    Arrange: command-line style arguments.
    Act: NormativeModel.from_args.
    Assert: predict_train is True by default and False for False or "False".
    """
    args = {"alg": "blr", "save_dir": "unused"}
    assert NormativeModel.from_args(**args).predict_train is True
    for value in (False, "False"):
        model = NormativeModel.from_args(**(args | {"predict_train": value}))
        assert model.predict_train is False


def test_transferAndExtend_should_keepPredictTrain_when_false(tmp_path: Path) -> None:
    """
    Arrange: a fitted model with predict_train=False.
    Act: transfer and extend it to new data.
    Assert: the new models keep the option, and neither predicts or saves
        results for the data it was fit on.
    """
    model = _model(tmp_path / "base", predict_train=False)
    model.fit(_data("train", 0))
    new_data = _data("new", 2)
    transferred = model.transfer(new_data, save_dir=str(tmp_path / "transfer"))
    assert transferred.predict_train is False
    assert not any(v in new_data.data_vars for v in OUTPUTS)
    assert not _result_files(tmp_path / "transfer", "new")
    extended = model.extend(_data("more", 3), save_dir=str(tmp_path / "extend"))
    assert extended.predict_train is False
    assert not list((tmp_path / "extend" / "results").glob("Z_*.csv"))
