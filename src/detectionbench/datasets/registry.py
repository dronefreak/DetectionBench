"""Central registry mapping a dataset key to its adapter class and metadata."""

from __future__ import annotations

from typing import TypeVar

from detectionbench.datasets.base import DatasetAdapter, DatasetSpec

_REGISTRY: dict[str, type[DatasetAdapter]] = {}

AdapterT = TypeVar("AdapterT", bound=type[DatasetAdapter])


def register(adapter_cls: AdapterT) -> AdapterT:
    """Register a dataset adapter class under its ``spec.key`` (a class decorator)."""
    key = adapter_cls.spec.key
    if key in _REGISTRY:
        raise ValueError(f"Dataset '{key}' is already registered.")
    _REGISTRY[key] = adapter_cls
    return adapter_cls


def get(key: str) -> DatasetAdapter:
    """Instantiate the registered adapter for ``key``."""
    return _get_class(key)()


def get_spec(key: str) -> DatasetSpec:
    """Return the registered ``DatasetSpec`` for ``key``."""
    return _get_class(key).spec


def list_datasets() -> list[str]:
    """Return all registered dataset keys, sorted."""
    return sorted(_REGISTRY)


def _get_class(key: str) -> type[DatasetAdapter]:
    try:
        return _REGISTRY[key]
    except KeyError as err:
        supported = ", ".join(list_datasets())
        raise KeyError(f"Unknown dataset '{key}'. Supported: {supported}.") from err
