"""Tests for the BDD100K (native per-image JSON) -> canonical COCO adapter."""

import json
from pathlib import Path

from detectionbench.datasets.bdd100k import BDD100KAdapter


def _entry(stem: str, categories: list[str]) -> dict:
    return {
        "name": stem,
        "attributes": {
            "weather": "clear",
            "scene": "city street",
            "timeofday": "daytime",
        },
        "frames": [
            {
                "timestamp": 10000,
                "objects": [
                    {
                        "category": cat,
                        "id": i,
                        "attributes": {"occluded": False, "truncated": False},
                        "box2d": {"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0},
                    }
                    for i, cat in enumerate(categories)
                ]
                # lane/drivable-area style poly2d entries, which must be skipped
                + [
                    {
                        "category": "area/drivable",
                        "id": 99,
                        "attributes": {},
                        "poly2d": [],
                    }
                ],
            }
        ],
    }


def _write_split(
    raw_dir: Path,
    official_split: str,
    shard: bool,
    names_and_cats: list[tuple[str, list[str]]],
) -> None:
    img_dir = raw_dir / "images_100k" / "100k" / official_split
    lbl_dir = raw_dir / "labels" / "100k" / official_split
    if shard:
        img_dir = img_dir / "shard_000"
        lbl_dir = lbl_dir / "shard_000"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for stem, cats in names_and_cats:
        (img_dir / f"{stem}.jpg").write_bytes(b"fake")
        (lbl_dir / f"{stem}.json").write_text(json.dumps(_entry(stem, cats)))


def _write_bdd100k(raw_dir: Path) -> None:
    # Train images sharded into shard_NNN/ (mirrors the Hugging Face layout);
    # val/test left flat -- the adapter must handle both via recursive glob.
    _write_split(
        raw_dir,
        "train",
        shard=True,
        names_and_cats=[(f"train_{i:03d}", ["car", "person"]) for i in range(20)],
    )
    _write_split(
        raw_dir,
        "val",
        shard=False,
        names_and_cats=[(f"val_{i:03d}", ["rider"]) for i in range(5)],
    )
    _write_split(
        raw_dir,
        "test",
        shard=False,
        names_and_cats=[(f"test_{i:03d}", ["traffic light"]) for i in range(6)],
    )


def test_prepare_coco_splits_multiclass(tmp_path: Path) -> None:
    raw_dir = tmp_path / "bdd100k"
    out = tmp_path / "canonical"
    _write_bdd100k(raw_dir)

    BDD100KAdapter().prepare_coco(raw_dir, out)

    counts = {}
    for split in ("train", "valid", "test"):
        payload = json.loads((out / split / "_annotations.coco.json").read_text())
        assert len(payload["categories"]) == 10
        assert all(a["segmentation"] == [] for a in payload["annotations"])
        counts[split] = len(payload["images"])
        for im in payload["images"]:
            assert (out / split / im["file_name"]).exists()
            assert im["width"] == 1280
            assert im["height"] == 720

    # Official train/val/test map directly onto canonical train/valid/test --
    # no more carving a validation slice out of train.
    assert counts["train"] == 20
    assert counts["valid"] == 5
    assert counts["test"] == 6

    # 2 box2d labels/image in train (poly2d dropped) -> 40 boxes.
    train_payload = json.loads((out / "train" / "_annotations.coco.json").read_text())
    assert len(train_payload["annotations"]) == 40

    valid_payload = json.loads((out / "valid" / "_annotations.coco.json").read_text())
    assert len(valid_payload["annotations"]) == 5
    assert {a["category_id"] for a in valid_payload["annotations"]} == {
        BDD100KAdapter.spec.classes.index("rider")
    }

    test_payload = json.loads((out / "test" / "_annotations.coco.json").read_text())
    assert len(test_payload["annotations"]) == 6
    assert {a["category_id"] for a in test_payload["annotations"]} == {
        BDD100KAdapter.spec.classes.index("traffic light")
    }
