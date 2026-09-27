"""Download -> verify -> prepare -> verify on a synthetic bucket served over local HTTP."""

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from conftest import FILES, make_tif

import download

CHIP = "data/flood_events/HandLabeled/{0}/Bolivia_1_{0}.tif"


def _bucket() -> dict[str, bytes]:
    s1 = np.full((2, 512, 512), -12.5)
    s1[:, :100] = np.nan  # swath edge: the first rows of a valid S1 chip hold no value
    label = np.resize(np.array([-1, 0, 1]), (1, 512, 512))
    return {
        CHIP.format("S1Hand"): make_tif(s1, dtype="float32"),
        CHIP.format("S2Hand"): make_tif(np.full((13, 512, 512), 1500), dtype="uint16"),
        CHIP.format("LabelHand"): make_tif(label, dtype="int16"),
        "splits/flood_handlabeled/flood_test_data.csv": b"Bolivia_1_S1Hand.tif,"
        b"Bolivia_1_LabelHand.tif\n",
    }


@pytest.fixture
def bucket(server: str, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Serve the files under v1.1/ and list them as the bucket would."""
    listing = []
    for path, body in _bucket().items():
        FILES[f"/{download.PREFIX}{path}"] = body
        digest = base64.b64encode(hashlib.md5(body).digest()).decode()
        listing.append({"path": path, "size": len(body), "md5": digest})
    monkeypatch.setattr(download, "URL", f"{server}/")
    monkeypatch.setattr(download, "list_bucket", lambda: [dict(f) for f in listing])
    return listing


def test_download_lands_only_md5_matches_then_prepare_keeps_labels(
    tmp_path: Path, bucket: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    label = CHIP.format("LabelHand")
    good = {f["path"]: f["md5"] for f in bucket}
    monkeypatch.setattr(
        download,
        "list_bucket",
        lambda: [f | {"md5": "wrong"} if f["path"] == label else f for f in bucket],
    )
    assert download.download(tmp_path, workers=2) == {label: "error: MD5 mismatch"}
    assert not (tmp_path / label).exists()
    assert not download.verify(tmp_path, sample=10)
    assert json.loads((tmp_path / "verify_report.json").read_text())["missing"] == [label]

    monkeypatch.setattr(download, "list_bucket", lambda: bucket)
    assert download.download(tmp_path, workers=2) == {}  # re-run fetches only what is missing
    assert all(download.md5(tmp_path / p) == m for p, m in good.items())
    assert download.verify(tmp_path, sample=10)  # NaN rows at the top of S1 are not corrupt

    assert download.compact(tmp_path, workers=1) == {}
    with rasterio.open(tmp_path / label) as src:
        assert src.dtypes[0] == "int16" and set(np.unique(src.read())) == {-1, 0, 1}
    assert download.verify(tmp_path, sample=10)


def test_verify_without_a_manifest_fails(tmp_path: Path) -> None:
    assert not download.verify(tmp_path)
    assert json.loads((tmp_path / "verify_report.json").read_text())["missing"] == ["manifest.json"]
