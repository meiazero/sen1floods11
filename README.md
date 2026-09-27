# sen1floods11

Download, compact and verify [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)
(Bonafilia et al., 2020). The data is not in this repository: every action takes `--dir`, the
directory where the dataset is created and saved.

| File | What |
|---|---|
| `download.py` | `download` (every raster, split and the metadata of the public bucket `gs://sen1floods11/v1.1`, MD5-checked) + verify, `prepare` (compact) + verify, `verify` (every file in `manifest.json`) |
| `common.py` | resumable transfer (HTTP Range), lossless compaction, raster check and verify report; identical copy in [meiazero/allclear](https://github.com/meiazero/allclear) and [meiazero/sen12mscrts](https://github.com/meiazero/sen12mscrts) |
| `tests/` | local HTTP server: interrupted download resumes, an MD5 mismatch never lands and a re-run fetches it, compaction keeps negative integer labels, S1 chips with NaN rows pass verify |

## Run

```bash
uv sync && uv run pytest -q && uvx ruff check .

# 1. download + verify (~36 GB, 31k files); needs internet, so on a cluster run it on the login node
nohup uv run python -u download.py download --dir /path/to/sen1floods11 --workers 16 \
    > download.log 2>&1 &

# 2. compact + verify: CPU work (SLURM example; SLURM only launches the script)
mkdir -p logs
sbatch -p <cpu partition> -c 16 --mem=16G --time=04:00:00 -o logs/%x_%j.out -J sen1floods11-prepare \
    --wrap "uv run python -u download.py prepare --dir /path/to/sen1floods11 --workers 16"

# verify only
uv run python download.py verify --dir /path/to/sen1floods11
```

Every action is resumable: re-run the same command after a failure or timeout. Each one
ends with verify and exits 0 only when that step had no failure and verify passes; the
report is `<dir>/verify_report.json`.

## Layout

The bucket's `v1.1/` as is, minus the model checkpoints and the STAC catalog (the metadata
GeoJSON holds the per-event dates, orbit and location).

```text
<dir>/
├── data/flood_events/HandLabeled/{S1Hand,S2Hand,LabelHand,JRCWaterHand,S1OtsuLabelHand}/<chip>_<layer>.tif   446 chips
├── data/flood_events/WeaklyLabeled/{S1Weak,S2Weak,S1OtsuLabelWeak,S2IndexLabelWeak}/<chip>_<layer>.tif      4,384 chips
├── data/perm_water/{S1Perm,JRCPerm}/*.tif                                                                   814 chips
├── splits/{flood_handlabeled,perm_water}/*.csv
├── Sen1Floods11_Metadata.geojson   one feature per flood event (12 events)
├── manifest.json                   bucket listing (path, size, MD5): what verify expects, offline
└── verify_report.json
```

Flood-event chips are 512 × 512 at 10 m (EPSG:4326; `perm_water` chips vary): S1 has 2 bands
(VV, VH, dB, NaN outside the swath), S2 has 13 (L1C), and the labels have 1 (`-1` no data,
`0` not water, `1` water).

## How rasters are stored

`prepare` rewrites every GeoTIFF in place as tiled (256 × 256) ZSTD with a predictor, in the
smallest dtype that holds every value at float32 precision (`uint8`, `uint16`, `int16`,
`float32`). The file is read back and compared before it replaces the original, and a
GeoTIFF tag marks it so a re-run skips it. The dtype can differ between files of the same
layer, so readers must cast (`src.read().astype(np.float32)`). The gain is in `S2Weak`
(LZW upstream, ~2.4× smaller); the other layers are already DEFLATE.

## Checks before push

Run `git config core.hooksPath .githooks` once per clone. `.githooks/pre-push` then
refuses a push that fails `uvx ruff format --check .` or `uvx ruff check .`.
