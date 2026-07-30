"""Dataset registry: importing this package registers every known adapter."""

from detectionbench.datasets import (  # noqa: F401
    aquarium,
    brackish,
    doclaynet,
    exdark,
    gwhd,
    lisa,
    seadronessee,
)
from detectionbench.datasets.registry import get, get_spec, list_datasets

__all__ = ["get", "get_spec", "list_datasets"]
