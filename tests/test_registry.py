"""Basic tests for the dataset registry."""

import pytest

from detectionbench.datasets import get, get_spec, list_datasets
from detectionbench.datasets.base import DatasetAdapter, DatasetSpec
from detectionbench.datasets.registry import register

_KNOWN_KEYS = {
    "brackish",
    "doclaynet",
    "exdark",
    "gwhd",
    "lisa",
    "seadronessee",
    "visdrone-det",
}


def test_list_datasets_includes_known_keys() -> None:
    datasets = list_datasets()
    assert _KNOWN_KEYS.issubset(set(datasets))
    assert datasets == sorted(datasets)


def test_get_spec_num_classes_matches_class_list() -> None:
    spec = get_spec("doclaynet")
    assert spec.num_classes == len(spec.classes) == 11


def test_get_returns_adapter_with_matching_spec() -> None:
    adapter = get("doclaynet")
    assert isinstance(adapter, DatasetAdapter)
    assert adapter.spec is get_spec("doclaynet")


def test_get_spec_unknown_key_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_spec("not-a-real-dataset")


def test_register_duplicate_key_raises_valueerror() -> None:
    class _DummyAdapter(DatasetAdapter):
        spec = DatasetSpec(key="doclaynet", display_name="Dummy", classes=["a"])

        def prepare_coco(self, raw_dir, output_dir) -> None:
            raise NotImplementedError

    with pytest.raises(ValueError, match="already registered"):
        register(_DummyAdapter)
