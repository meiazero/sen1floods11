"""Resumable transfer, lossless compaction of rasters and the per-raster check used by `verify`.

Identical copy in meiazero/allclear, meiazero/sen12mscrts and meiazero/sen1floods11: change all.

Every step is resumable: archives are fetched with HTTP Range into `.part` files, integrity
is checked by extracting into a staging directory, and only a complete result is moved into
place, so "exists" means "complete" and a re-run skips it.
"""

import json
import random
import shutil
import tarfile
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio

COMPACT_TAG = "datasets_compact"  # GeoTIFF tag set on rewritten files, so a re-run skips them
# Tried in order; the first one that holds every value exactly wins (NaN only fits floats).
DTYPES = ("uint8", "uint16", "int16", "float32", "float64")
MIN_SIDE = 256


# ---------------------------------------------------------------- transfer


def fetch(url: str, dest: Path, timeout: float = 60) -> None:
    """Resumable download to `dest` via `dest.part` and HTTP Range."""
    part = dest.with_name(f"{dest.name}.part")
    done = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        mode = "ab" if done and resp.status == 206 else "wb"  # server ignored Range: restart
        with part.open(mode) as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
    part.rename(dest)


def extract_atomic(archive: Path, staging: Path) -> Path:
    """Extract a .tar.gz into a fresh staging dir, then delete the archive.

    A corrupt archive raises before anything reaches its final place.
    """
    if staging.exists():
        shutil.rmtree(staging)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(staging, filter="data")
    archive.unlink()
    return staging


def move_tree(src: Path, dst: Path) -> None:
    """Move every file under `src` to the same relative path under `dst` (merging dirs)."""
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        f.replace(target)
    shutil.rmtree(src)


# ---------------------------------------------------------------- compaction


def fits(a: np.ndarray, dtype: str) -> bool:
    """True if `dtype` stores every value of `a` (NaN included) unchanged."""
    # Compared in float64: casting back to an integer `a.dtype` would wrap -1 -> 65535 -> -1.
    with np.errstate(invalid="ignore", over="ignore"):
        return np.array_equal(
            a.astype(dtype).astype("float64"), a.astype("float64"), equal_nan=True
        )


def smallest_exact_dtype(a: np.ndarray) -> str:
    return next((d for d in DTYPES if fits(a, d)), str(a.dtype))


def compact_raster(path: Path) -> bool:
    """Rewrite `path` in place as tiled ZSTD GeoTIFF in its smallest exact dtype.

    Lossless at float32 precision, the precision every reader works in: float64 becomes
    float32 (AllClear S1 changes by <= 6e-8 relative), everything else round-trips exactly.
    The rewritten file is read back and compared before it replaces the original.
    Returns False when the file was already compact.
    """
    with rasterio.open(path) as src:
        if src.tags().get(COMPACT_TAG):
            return False
        a, profile, tags = src.read(), src.profile, src.tags()
        band_tags = [src.tags(i) for i in range(1, src.count + 1)]
        descriptions = src.descriptions
    if a.dtype == np.float64:
        a = a.astype(np.float32)
    dtype = smallest_exact_dtype(a)
    nodata = profile.get("nodata")
    if nodata is not None and not fits(np.array([nodata], "float64"), dtype):
        nodata = None  # not representable in the new dtype, e.g. NaN in uint16
    profile.update(
        driver="GTiff",
        dtype=dtype,
        nodata=nodata,
        compress="zstd",
        zstd_level=9,
        predictor=3 if dtype.startswith("float") else 2,
        interleave="pixel",
        tiled=min(a.shape[1:]) >= 256,
    )
    if profile["tiled"]:
        profile.update(blockxsize=256, blockysize=256)
    tmp = path.with_name(f".{path.name}.compact")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(a.astype(dtype))
        dst.update_tags(**tags, **{COMPACT_TAG: "1"})
        for i, (bt, desc) in enumerate(zip(band_tags, descriptions, strict=True), 1):
            dst.update_tags(i, **bt)
            if desc:
                dst.set_band_description(i, desc)
    with rasterio.open(tmp) as check:
        if not np.array_equal(check.read().astype(a.dtype), a, equal_nan=True):
            tmp.unlink()
            raise RuntimeError(f"{path}: compaction round trip changed values")
    tmp.replace(path)
    return True


def _compact_dir(d: Path) -> tuple[str, int, str]:
    try:
        return str(d), sum(compact_raster(f) for f in sorted(d.rglob("*.tif"))), "ok"
    except Exception as e:  # keep going; the directory is retried on re-run
        return str(d), 0, f"error: {e}"


def compact_dirs(dirs: Iterable[Path], workers: int = 8) -> dict[str, str]:
    """Compact every .tif under each directory in parallel -> {dir: error} of failures."""
    dirs = sorted(dirs)
    failed, rewritten = {}, 0
    print(f"compact: {len(dirs)} directories, {workers} workers", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for i, (d, n, status) in enumerate(pool.map(_compact_dir, dirs, chunksize=4), 1):
            rewritten += n
            if status != "ok":
                failed[d] = status
                print(f"[{i}/{len(dirs)}] {d} {status}", flush=True)
            elif i % 500 == 0:
                print(f"[{i}/{len(dirs)}] {rewritten} rasters rewritten", flush=True)
    print(f"compact: {rewritten} rasters rewritten, {len(failed)} directories failed")
    return failed


# ---------------------------------------------------------------- verification


def check_raster(path: Path, bands: int | None) -> str | None:
    """None if the raster opens with the expected band count, size and a finite value in band 1."""
    try:
        with rasterio.open(path) as src:
            if bands and src.count != bands:
                return f"{src.count} bands, expected {bands}"
            if min(src.width, src.height) < MIN_SIDE:
                return f"{src.width}x{src.height} < {MIN_SIDE}"
            # The whole band: a valid S1 chip can be NaN over any corner (Sen1Floods11).
            if not np.isfinite(src.read(1)).any():
                return "no finite value in band 1"
    except Exception as e:
        return f"unreadable: {e}"
    return None


def verify_files(
    name: str,
    root: Path,
    files: list[Path],
    missing: list[str],
    bands_of: Callable[[Path], int | None],
    extra: dict,
    sample: int = 500,
    seed: int = 0,
) -> bool:
    """Report missing files + a deep read of `sample` random ones to <root>/verify_report.json.

    `extra` holds dataset-specific findings; any non-empty list in it fails the check.
    """
    rng = random.Random(seed)
    checked = rng.sample(files, min(sample, len(files)))
    bad = {str(p): err for p in checked if (err := check_raster(p, bands_of(p)))}
    problems = [k for k, v in extra.items() if isinstance(v, list) and v]
    ok = not missing and not bad and not problems
    report = {
        "dataset": name,
        "root": str(root),
        **extra,
        "expected": len(files) + len(missing),
        "n_missing": len(missing),
        "missing": missing[:1000],
        "deep_checked": len(checked),
        "corrupt": bad,
        "ok": ok,
    }
    (root / "verify_report.json").write_text(json.dumps(report, indent=2))
    print(
        f"{name}: expected {report['expected']}, missing {len(missing)}, "
        f"deep-checked {len(checked)}, corrupt {len(bad)}"
        + "".join(f", {k} {len(extra[k])}" for k in problems)
        + f" -> {'OK' if ok else 'NOT OK'} ({root / 'verify_report.json'})"
    )
    return ok
