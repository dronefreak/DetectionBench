"""Dataset registry: importing this package registers every known adapter."""

from detectionbench.datasets import (  # noqa: F401
    animalsdet,
    bdd100k,
    brackish,
    ceymo,
    doclaynet,
    duo,
    exdark,
    gc10det,
    gwhd,
    hrp4k,
    hrsid,
    kitti,
    lisa,
    llvip,
    marida,
    neudet,
    pklot,
    publaynet,
    rdd2022,
    seadronessee,
    seaships,
    sku110k,
    ssdd,
    uavdt,
    visdrone,
)
from detectionbench.datasets.registry import get, get_spec, list_datasets

__all__ = ["get", "get_spec", "list_datasets"]
