from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Literal, Optional

import pandas as pd

Preset = Literal["dtm", "dsm", "intensity"]


@dataclass(frozen=True)
class RasterOptions:
    resolution: float = 1.0
    output_type: str = "mean"
    data_type: str = "float32"
    nodata: float = -9999.0
    dimension: Optional[str] = None
    classification_filter: Optional[str] = None


def _default_options(preset: Preset) -> RasterOptions:
    if preset == "dtm":
        return RasterOptions(
            resolution=1.0,
            output_type="min",
            data_type="float32",
            nodata=-9999.0,
            dimension="Z",
            classification_filter="Classification[2:2]",
        )
    if preset == "dsm":
        return RasterOptions(
            resolution=1.0,
            output_type="max",
            data_type="float32",
            nodata=-9999.0,
            dimension="Z",
            classification_filter=None,
        )
    return RasterOptions(
        resolution=1.0,
        output_type="mean",
        data_type="float32",
        nodata=-9999.0,
        dimension="Intensity",
        classification_filter=None,
    )


def _require_pdal() -> str:
    pdal_path = shutil.which("pdal")
    if not pdal_path:
        raise RuntimeError(
            "PDAL CLI not found in PATH. Install PDAL and retry."
        )
    return pdal_path


def _build_translate_command(
    input_laz: Path,
    output_tif: Path,
    options: RasterOptions,
) -> list[str]:
    cmd = [
        _require_pdal(),
        "translate",
        str(input_laz),
        str(output_tif),
        f"--writers.gdal.resolution={options.resolution}",
        f"--writers.gdal.output_type={options.output_type}",
        f"--writers.gdal.data_type={options.data_type}",
        f"--writers.gdal.nodata={options.nodata}",
    ]

    if options.dimension:
        cmd.append(f"--writers.gdal.dimension={options.dimension}")

    if options.classification_filter:
        cmd.append(f"--filters.range.limits={options.classification_filter}")

    return cmd


def laz_to_tif(
    input_laz: str | Path,
    output_tif: str | Path,
    preset: Preset = "dtm",
    resolution: Optional[float] = None,
) -> Path:
    """
    Rasterize a LAZ/LAS point cloud to GeoTIFF using PDAL CLI.

    Presets:
    - dtm: terrain model using ground class only (Classification==2).
    - dsm: surface model using all points.
    - intensity: raster from Intensity dimension.
    """
    src = Path(input_laz)
    dst = Path(output_tif)

    if not src.exists():
        raise FileNotFoundError(f"Input LAZ/LAS not found: {src}")

    opts = _default_options(preset)
    if resolution is not None:
        opts = RasterOptions(
            resolution=resolution,
            output_type=opts.output_type,
            data_type=opts.data_type,
            nodata=opts.nodata,
            dimension=opts.dimension,
            classification_filter=opts.classification_filter,
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_translate_command(src, dst, opts)

    subprocess.run(cmd, check=True)
    return dst


def dtm_from_laz(
    input_laz: str | Path,
    output_tif: str | Path,
    resolution: float = 1.0,
) -> Path:
    return laz_to_tif(input_laz, output_tif, preset="dtm", resolution=resolution)


def dsm_from_laz(
    input_laz: str | Path,
    output_tif: str | Path,
    resolution: float = 1.0,
) -> Path:
    return laz_to_tif(input_laz, output_tif, preset="dsm", resolution=resolution)


def intensity_from_laz(
    input_laz: str | Path,
    output_tif: str | Path,
    resolution: float = 1.0,
) -> Path:
    return laz_to_tif(input_laz, output_tif, preset="intensity", resolution=resolution)


def laz_to_parquet(
    input_laz: str | Path,
    output_parquet: str | Path,
    dimensions: Optional[list[str]] = None,
) -> Path:
    """
    Convert a LAZ/LAS point cloud to Parquet.

    Uses laspy (with lazrs or laszip backend) when available, falls back to
    the PDAL Python bindings otherwise. Each point becomes a row.
    Pass `dimensions` to select a subset of fields.
    """
    src = Path(input_laz)
    dst = Path(output_parquet)

    if not src.exists():
        raise FileNotFoundError(f"Input LAZ/LAS not found: {src}")

    try:
        import laspy
        import numpy as np
        with laspy.open(str(src)) as f:
            las = f.read()
        data = {dim: np.asarray(las[dim]) for dim in las.point_format.dimension_names}
        df = pd.DataFrame(data)
    except ImportError:
        try:
            import pdal
        except ImportError:
            raise RuntimeError(
                "No LAZ reader found. Install laspy + lazrs-python:  uv pip install laspy lazrs-python"
            )
        pipeline = pdal.Pipeline(json.dumps({"pipeline": [str(src)]}))
        pipeline.execute()
        df = pd.DataFrame(pipeline.arrays[0])

    if dimensions:
        missing = [d for d in dimensions if d not in df.columns]
        if missing:
            raise ValueError(f"Dimensions not found: {missing}. Available: {list(df.columns)}")
        df = df[dimensions]

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return dst
