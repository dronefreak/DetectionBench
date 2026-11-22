"""Offline tests for detectionbench.scripts.download_dataset (no network)."""

import sys
import urllib.request
from pathlib import Path

import pytest
from rich.console import Console

from detectionbench.datasets import list_datasets
from detectionbench.scripts import download_dataset
from detectionbench.scripts.download_dataset import _SOURCES, DownloadOption


@pytest.fixture
def console(monkeypatch: pytest.MonkeyPatch) -> Console:
    recorder = Console(record=True, width=200)
    monkeypatch.setattr(
        download_dataset.RichConsoleManager, "get_console", lambda *a, **k: recorder
    )
    return recorder


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["download", *argv])
    download_dataset.main()


def test_sources_are_registered_and_well_formed() -> None:
    registered = set(list_datasets())
    for key, source in _SOURCES.items():
        assert key == source.key
        assert key in registered
        assert source.options
        for option in source.options:
            assert option.method in {"gdrive", "direct", "baidu", "manual"}
            if option.method == "gdrive":
                assert option.gdrive_id


def test_manual_instructions_name_the_real_dataset(
    monkeypatch: pytest.MonkeyPatch, console: Console, tmp_path: Path
) -> None:
    _run(
        monkeypatch,
        "--dataset",
        "uavdt",
        "--method",
        "manual",
        "--output-dir",
        str(tmp_path),
    )
    text = console.export_text()
    assert "detectionbench-prepare-coco --dataset uavdt" in text
    assert "<key>" not in text


def test_list_only_prints_options(
    monkeypatch: pytest.MonkeyPatch, console: Console
) -> None:
    _run(monkeypatch, "--dataset", "sku110k", "--list")
    text = console.export_text()
    assert "Official download options" in text
    assert "Manual download" not in text


def test_explicit_method_without_matching_option_errors(
    monkeypatch: pytest.MonkeyPatch, console: Console
) -> None:
    methods = {o.method for o in _SOURCES["sku110k"].options}
    assert "gdrive" not in methods  # precondition for this test
    with pytest.raises(SystemExit, match="no gdrive download option"):
        _run(monkeypatch, "--dataset", "sku110k", "--method", "gdrive")


def test_failed_direct_download_removes_partial_file(
    monkeypatch: pytest.MonkeyPatch, console: Console, tmp_path: Path
) -> None:
    class _BrokenResponse:
        def __enter__(self) -> "_BrokenResponse":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self, *args: object) -> bytes:
            raise OSError("connection reset")

    monkeypatch.setattr(urllib.request, "urlopen", lambda url: _BrokenResponse())
    option = DownloadOption(
        label="archive", method="direct", url="https://example.invalid/data.zip"
    )

    assert download_dataset.try_direct_download(option, tmp_path) is False
    assert not (tmp_path / "data.zip").exists()


def test_auto_success_skips_manual_instructions(
    monkeypatch: pytest.MonkeyPatch, console: Console, tmp_path: Path
) -> None:
    calls: list[str] = []

    def _succeed(option: DownloadOption, out: Path) -> bool:
        calls.append(option.method)
        return True

    monkeypatch.setattr(download_dataset, "try_direct_download", _succeed)
    monkeypatch.setattr(download_dataset, "try_gdrive_download", _succeed)
    _run(monkeypatch, "--dataset", "sku110k", "--output-dir", str(tmp_path))
    assert len(calls) == 1
    assert "Manual download" not in console.export_text()
