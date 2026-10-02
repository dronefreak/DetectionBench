"""Tests for the Animals Detection (OIDv4-layout) -> canonical COCO adapter."""

import json
from pathlib import Path

from PIL import Image

from detectionbench.datasets.animalsdet import AnimalsDetAdapter


def _write_image(path: Path, size: tuple[int, int] = (100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path)


def _write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_raw_release(raw_dir: Path) -> None:
    # train/Zebra: two distinct images, one of them ("shared.jpg") also fetched
    # into train/Horse -- the classic OIDv4 cross-class duplication case.
    _write_image(raw_dir / "train" / "Zebra" / "only_zebra.jpg")
    _write_label(
        raw_dir / "train" / "Zebra" / "Label" / "only_zebra.txt",
        ["Zebra 1.0 2.0 50.0 40.0"],
    )
    _write_image(raw_dir / "train" / "Zebra" / "shared.jpg")
    _write_label(
        raw_dir / "train" / "Zebra" / "Label" / "shared.txt",
        ["Zebra 0.0 0.0 10.0 10.0"],
    )
    _write_image(raw_dir / "train" / "Horse" / "shared.jpg")  # same filename
    _write_label(
        raw_dir / "train" / "Horse" / "Label" / "shared.txt",
        ["Horse 20.0 20.0 60.0 60.0"],
    )
    # A multi-word class name, to exercise the "last 4 tokens are the box" parsing.
    _write_image(raw_dir / "train" / "Brown bear" / "bear.jpg")
    _write_label(
        raw_dir / "train" / "Brown bear" / "Label" / "bear.txt",
        ["Brown bear 5.0 5.0 15.0 15.0"],
    )
    # test split: one image, kept as-is (no val carve-out from test).
    _write_image(raw_dir / "test" / "Zebra" / "test_zebra.jpg")
    _write_label(
        raw_dir / "test" / "Zebra" / "Label" / "test_zebra.txt",
        ["Zebra 1.0 1.0 20.0 20.0"],
    )


def test_cross_class_duplicate_image_merges_into_one_entry(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "canonical"
    _write_raw_release(raw_dir)

    AnimalsDetAdapter().prepare_coco(raw_dir, output_dir)

    train_images: list[dict] = []
    valid_images: list[dict] = []
    for split, bucket in (("train", train_images), ("valid", valid_images)):
        payload = json.loads(
            (output_dir / split / "_annotations.coco.json").read_text()
        )
        bucket.extend(payload["images"])

    all_train_valid = train_images + valid_images
    filenames = [img["file_name"] for img in all_train_valid]
    # 3 unique train-side filenames (only_zebra, shared, bear.jpg) -- "shared.jpg"
    # must appear exactly once despite existing under both Zebra and Horse.
    assert sorted(filenames) == ["bear.jpg", "only_zebra.jpg", "shared.jpg"]
    assert filenames.count("shared.jpg") == 1


def test_shared_image_gets_boxes_from_every_class_folder(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "canonical"
    _write_raw_release(raw_dir)

    AnimalsDetAdapter().prepare_coco(raw_dir, output_dir)

    # With only 3 train-side images and a 15% val fraction, the single held-out
    # validation image could land in either canonical split -- check both.
    boxes_for_shared: list[dict] = []
    categories: dict[int, str] = {}
    for split in ("train", "valid"):
        payload = json.loads(
            (output_dir / split / "_annotations.coco.json").read_text()
        )
        categories.update({c["id"]: c["name"] for c in payload["categories"]})
        shared = next(
            (img for img in payload["images"] if img["file_name"] == "shared.jpg"),
            None,
        )
        if shared is None:
            continue
        boxes_for_shared = [
            a for a in payload["annotations"] if a["image_id"] == shared["id"]
        ]

    assert {categories[b["category_id"]] for b in boxes_for_shared} == {
        "Zebra",
        "Horse",
    }


def test_multiword_class_name_is_parsed_correctly(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "canonical"
    _write_raw_release(raw_dir)

    AnimalsDetAdapter().prepare_coco(raw_dir, output_dir)

    found = False
    for split in ("train", "valid"):
        payload = json.loads(
            (output_dir / split / "_annotations.coco.json").read_text()
        )
        names = {c["id"]: c["name"] for c in payload["categories"]}
        bear_img = next(
            (img for img in payload["images"] if img["file_name"] == "bear.jpg"), None
        )
        if bear_img is None:
            continue
        found = True
        anns = [a for a in payload["annotations"] if a["image_id"] == bear_img["id"]]
        assert len(anns) == 1
        assert names[anns[0]["category_id"]] == "Brown bear"
        assert anns[0]["bbox"] == [5.0, 5.0, 10.0, 10.0]
    assert found


def test_test_split_kept_official_and_image_dims_read_from_file(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "canonical"
    _write_raw_release(raw_dir)

    AnimalsDetAdapter().prepare_coco(raw_dir, output_dir)

    payload = json.loads((output_dir / "test" / "_annotations.coco.json").read_text())
    assert len(payload["images"]) == 1
    image = payload["images"][0]
    assert image["file_name"] == "test_zebra.jpg"
    assert (image["width"], image["height"]) == (100, 80)
    assert (output_dir / "test" / "test_zebra.jpg").exists()
