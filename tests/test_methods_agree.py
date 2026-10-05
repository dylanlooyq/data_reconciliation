"""Every method must reproduce the generator's recorded ground truth."""
import importlib

import pytest

from tabrecon.data import generate
from tabrecon.methods.registry import METHODS

N_ROWS = 40_000
ROW_GROUP = 10_000


def _run(method, path_a, path_b):
    mod = importlib.import_module(method.module)
    if hasattr(mod, "prepare"):
        mod.prepare()
    return mod.run(str(path_a), str(path_b), **method.params)


@pytest.fixture(scope="module")
def scattered(tmp_path_factory):
    out = tmp_path_factory.mktemp("scattered")
    truth = generate(out, N_ROWS, change_fraction=0.005, row_group_size=ROW_GROUP)
    return out, truth


@pytest.fixture(scope="module")
def identical(tmp_path_factory):
    out = tmp_path_factory.mktemp("identical")
    truth = generate(out, N_ROWS, row_group_size=ROW_GROUP, dirty_groups=0)
    return out, truth


@pytest.fixture(scope="module")
def one_dirty_group(tmp_path_factory):
    out = tmp_path_factory.mktemp("one_dirty")
    truth = generate(out, N_ROWS, change_fraction=0.001, row_group_size=ROW_GROUP, dirty_groups=1)
    return out, truth


@pytest.mark.parametrize("method", METHODS, ids=lambda m: m.key)
@pytest.mark.parametrize("fixture", ["scattered", "identical", "one_dirty_group"])
def test_method_matches_ground_truth(method, fixture, request):
    out, truth = request.getfixturevalue(fixture)
    rate = _run(method, out / "a.parquet", out / "b.parquet")
    assert rate == pytest.approx(truth["expected_match_rate"], abs=0.5 / truth["n_rows"])


def test_generator_records_exact_changes(scattered):
    out, truth = scattered
    assert truth["n_changed"] == len(truth["changed_row_ids"])
    assert truth["expected_match_rate"] == (truth["n_rows"] - truth["n_changed"]) / truth["n_rows"]


def test_dirty_groups_confines_changes(one_dirty_group):
    _, truth = one_dirty_group
    groups = {(i - 1) // ROW_GROUP for i in truth["changed_row_ids"]}
    assert len(groups) == 1
