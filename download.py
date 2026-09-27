"""Sen1Floods11: download, compact and verify under the directory given by --dir.

    python download.py download --dir <dir> --workers 16  # login node: download + verify, ~36 GB
    python download.py prepare --dir <dir> --workers 16   # cpuq: compact + verify
    python download.py verify --dir <dir>

Each action exits 0 only when verify passes; re-run the same command to resume.

Layout, the bucket's v1.1/ as is, without the model checkpoints and the STAC catalog:
    <dir>/data/flood_events/HandLabeled/<layer>/<chip>_<layer>.tif     446 chips x 5 layers
    <dir>/data/flood_events/WeaklyLabeled/<layer>/<chip>_<layer>.tif   4,384 chips x 4 layers
    <dir>/data/perm_water/{S1Perm,JRCPerm}/*.tif                       814 chips x 2 layers
    <dir>/splits/**/*.csv, <dir>/Sen1Floods11_Metadata.geojson
    <dir>/manifest.json    the bucket listing: what verify expects, offline

Source: the public bucket gs://sen1floods11 (Bonafilia et al., 2020), read through its HTTPS
JSON API, so neither gsutil nor credentials are needed. A file lands only after it matches the
bucket's MD5.
"""

import argparse
import base64
import hashlib
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common

LIST = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"
URL = "https://storage.googleapis.com/sen1floods11/"
PREFIX = "v1.1/"
KEEP = ("data/", "splits/", "Sen1Floods11_Metadata.geojson")
MANIFEST = "manifest.json"
BANDS = {"S1Hand": 2, "S1Weak": 2, "S1Perm": 2, "S2Hand": 13, "S2Weak": 13}  # others: 1


# ---------------------------------------------------------------- download


def list_bucket() -> list[dict]:
    """{path (relative to v1.1/), size, md5} of every file kept."""
    files, token = [], None
    while True:
        query = {"prefix": PREFIX, "fields": "nextPageToken,items(name,size,md5Hash)"}
        if token:
            query["pageToken"] = token
        with urllib.request.urlopen(f"{LIST}?{urllib.parse.urlencode(query)}", timeout=60) as r:
            page = json.load(r)
        for item in page.get("items", []):
            path = item["name"].removeprefix(PREFIX)
            if path.startswith(KEEP) and path.endswith((".tif", ".csv", ".geojson")):
                files.append({"path": path, "size": int(item["size"]), "md5": item["md5Hash"]})
        token = page.get("nextPageToken")
        if not token:
            return files


def md5(path: Path) -> str:
    """Base64 MD5, as the bucket reports it."""
    with path.open("rb") as f:
        return base64.b64encode(hashlib.file_digest(f, "md5").digest()).decode()


def _fetch(root: Path, entry: dict) -> tuple[str, str]:
    dest = root / entry["path"]
    if dest.exists():
        return entry["path"], "ok"
    tmp = dest.with_name(f".{dest.name}.dl")  # resumes through tmp.part
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        common.fetch(URL + urllib.parse.quote(PREFIX + entry["path"]), tmp, timeout=300)
        if md5(tmp) != entry["md5"]:
            tmp.unlink()
            return entry["path"], "error: MD5 mismatch"
        tmp.replace(dest)
        return entry["path"], "ok"
    except Exception as e:  # keep going; the file is retried on re-run
        return entry["path"], f"error: {e}"


def download(root: Path, workers: int = 8) -> dict[str, str]:
    """Every kept file of the bucket, MD5-checked, plus manifest.json."""
    root.mkdir(parents=True, exist_ok=True)
    files = list_bucket()
    tmp = root / f".{MANIFEST}.tmp"
    tmp.write_text(json.dumps(files, indent=1))
    tmp.replace(root / MANIFEST)
    size = sum(f["size"] for f in files) / 1e9
    print(f"download: {len(files)} files, {size:.1f} GB, {workers} workers", flush=True)
    failed = {}
    with ThreadPoolExecutor(workers) as pool:
        for i, (path, status) in enumerate(pool.map(lambda f: _fetch(root, f), files), 1):
            if status != "ok":
                failed[path] = status
                print(f"[{i}/{len(files)}] {path} {status}", flush=True)
            elif i % 1000 == 0:
                print(f"[{i}/{len(files)}]", flush=True)
    print(f"download: {len(failed)} files failed")
    return failed


def compact(root: Path, workers: int = 8) -> dict[str, str]:
    # ponytail: one worker per layer directory, so the 4,384 S2Weak chips bound the run
    # (~15 min); split into per-file tasks if that ever matters.
    dirs = {(root / p).parent for p in manifest_paths(root) if p.endswith(".tif")}
    return common.compact_dirs([d for d in dirs if d.is_dir()], workers)


# ---------------------------------------------------------------- verify


def manifest_paths(root: Path) -> list[str]:
    manifest = root / MANIFEST
    return [f["path"] for f in json.loads(manifest.read_text())] if manifest.is_file() else []


def verify(root: Path, sample: int = 500) -> bool:
    paths = manifest_paths(root)
    missing = [p for p in paths if not (root / p).is_file()] if paths else [MANIFEST]
    files = [root / p for p in paths if p.endswith(".tif") and (root / p).is_file()]
    return common.verify_files(
        "sen1floods11",
        root,
        files,
        missing,
        lambda p: BANDS.get(p.parent.name, 1),
        {"source": URL + PREFIX, "manifest_files": len(paths)},
        sample,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("download", "prepare", "verify"))
    ap.add_argument(
        "--dir", type=Path, required=True, help="where the dataset is created and saved"
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample", type=int, default=500, help="rasters opened by verify")
    args = ap.parse_args()
    root = args.dir.expanduser().resolve()
    failed = {}
    if args.action == "download":
        failed = download(root, args.workers)
    elif args.action == "prepare":
        failed = compact(root, args.workers)
    ok = verify(root, args.sample)  # after a partial failure too: the report lists it
    raise SystemExit(0 if ok and not failed else 1)


if __name__ == "__main__":
    main()
