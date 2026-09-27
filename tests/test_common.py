"""Transfer and compaction. Identical copy in meiazero/{allclear,sen12mscrts,sen1floods11}."""

import tarfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from conftest import FILES, RangeHandler, make_tar, make_tif

import common


@pytest.mark.parametrize("honor_range", [True, False])
def test_fetch_completes_an_interrupted_download(
    server: str, tmp_path: Path, honor_range: bool
) -> None:
    RangeHandler.honor_range = honor_range
    FILES["/big.bin"] = bytes(range(256)) * 400
    dest = tmp_path / "big.bin"
    (tmp_path / "big.bin.part").write_bytes(FILES["/big.bin"][:1000])
    common.fetch(f"{server}/big.bin", dest)
    assert dest.read_bytes() == FILES["/big.bin"]
    RangeHandler.honor_range = True


def test_corrupt_archive_never_reaches_its_final_place(tmp_path: Path) -> None:
    archive = tmp_path / "roi1.tar.gz"
    archive.write_bytes(make_tar({"roi1/a.txt": b"x"})[:40])
    with pytest.raises((tarfile.TarError, EOFError, OSError)):
        common.extract_atomic(archive, tmp_path / ".staging")
    assert not (tmp_path / "roi1").exists() and archive.exists()


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        (np.arange(12_000, dtype=np.float64), "uint16"),  # S2 digital numbers
        (np.arange(0, 100.5, 0.5), "float32"),  # composited mask: 0.5 steps
        (np.array([0, 1, 100.0]), "uint8"),
        (np.array([-3.0, 7.0]), "int16"),
        (np.array([1.0, np.nan]), "float32"),  # NaN only fits floats
        (np.array([0.1]), "float32"),  # float64 is stored at float32 precision
    ],
)
def test_compaction_picks_the_smallest_dtype_that_keeps_every_value(
    tmp_path: Path, values: np.ndarray, dtype: str
) -> None:
    arr = np.resize(values, (2, 256, 256))
    path = tmp_path / "x.tif"
    path.write_bytes(make_tif(arr))
    assert common.compact_raster(path)
    with rasterio.open(path) as src:
        assert src.dtypes[0] == dtype and src.compression.name.lower() == "zstd"
        assert np.array_equal(src.read(), arr.astype(np.float32), equal_nan=True)
        assert src.transform == rasterio.transform.from_origin(0, 256, 1, 1)
    assert not common.compact_raster(path)  # tagged: a re-run skips it


def test_compaction_keeps_negative_integer_labels(tmp_path: Path) -> None:
    arr = np.resize(np.array([-1, 0, 1], dtype=np.int16), (1, 256, 256))  # Sen1Floods11 labels
    path = tmp_path / "label.tif"
    path.write_bytes(make_tif(arr, dtype="int16"))
    assert common.compact_raster(path)
    with rasterio.open(path) as src:
        assert src.dtypes[0] == "int16" and np.array_equal(src.read(), arr)
