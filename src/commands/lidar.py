from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from ..lidar import laz_to_parquet, laz_to_tif, parquet_to_tif

console = Console()
app = typer.Typer(help="LiDAR helpers")


@app.command(help="Convert LAZ/LAS to GeoTIFF using PDAL.")
def convert(
    input_laz: str = typer.Argument(..., help="Input .laz or .las path."),
    output_tif: str = typer.Argument(
        None, help="Output .tif path. Defaults to data/<input_stem>_<preset>.tif."
    ),
    preset: str = typer.Option(
        "dtm", "--preset", "-p", help="Raster preset: dtm, dsm, intensity."
    ),
    resolution: float = typer.Option(1.0, "--resolution", "-r", help="Pixel size."),
):
    if preset not in {"dtm", "dsm", "intensity"}:
        raise typer.BadParameter("preset must be one of: dtm, dsm, intensity")

    if output_tif:
        dst = Path(output_tif)
    else:
        stem = Path(input_laz).stem
        dst = Path("data") / f"{stem}_{preset}.tif"

    try:
        output = laz_to_tif(
            input_laz=input_laz,
            output_tif=dst,
            preset=preset,
            resolution=resolution,
        )
    except Exception as exc:
        console.print(f"[red]Conversion failed:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(f"[green]Created:[/green] {output}")


@app.command(name="to-parquet", help="Convert LAZ/LAS to Parquet using PDAL Python bindings.")
def to_parquet(
    input_laz: str = typer.Argument(..., help="Input .laz or .las path."),
    output_parquet: str = typer.Argument(
        None, help="Output .parquet path. Defaults to data/<input_stem>.parquet."
    ),
    dimensions: str = typer.Option(
        None,
        "--dimensions",
        "-d",
        help="Comma-separated list of dimensions to keep, e.g. X,Y,Z,Intensity.",
    ),
):
    if output_parquet:
        dst = Path(output_parquet)
    else:
        stem = Path(input_laz).stem
        dst = Path("data") / f"{stem}.parquet"

    dims = [d.strip() for d in dimensions.split(",")] if dimensions else None

    try:
        output = laz_to_parquet(input_laz=input_laz, output_parquet=dst, dimensions=dims)
    except Exception as exc:
        console.print(f"[red]Conversion failed:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(f"[green]Created:[/green] {output}")


@app.command(help="Describe a parquet/csv dataset and save a .txt report alongside it.")
def describe(
    file_path: str = typer.Argument(..., help="Input dataset path (.parquet or .csv)."),
    include_all: bool = typer.Option(
        False, "--all", help="Include non-numeric columns in describe output."
    ),
):
    path = Path(file_path)
    if not path.exists():
        raise typer.BadParameter(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise typer.BadParameter("Unsupported file type. Use .parquet or .csv.")

    summary = df.describe(include="all") if include_all else df.describe()

    lines = [
        f"File   : {path}",
        f"Shape  : {df.shape[0]:,} rows x {df.shape[1]} columns",
        f"Columns: {', '.join(df.columns.tolist())}",
        "",
        "── describe ──",
        summary.to_string(),
        "",
        "── sample (10 rows) ──",
        df.sample(min(10, len(df)), random_state=42).to_string(index=True),
    ]
    report = "\n".join(lines)

    out = path.with_name(path.stem + "_desc_sample.txt")
    out.write_text(report)

    console.print(report)
    console.print(f"\n[green]Saved:[/green] {out}")


@app.command(name="parquet-to-tif", help="Rasterize a point-cloud Parquet to GeoTIFF.")
def parquet_to_tif_cmd(
    input_parquet: str = typer.Argument(..., help="Input .parquet path."),
    output_tif: str = typer.Argument(
        None, help="Output .tif path. Defaults to data/<stem>.tif."
    ),
    value_col: str = typer.Option("Z", "--col", "-c", help="Column to rasterize."),
    resolution: float = typer.Option(1.0, "--resolution", "-r", help="Pixel size in metres."),
    aggregation: str = typer.Option("mean", "--agg", "-a", help="Aggregation: mean, min, max."),
):
    if aggregation not in {"mean", "min", "max"}:
        raise typer.BadParameter("agg must be one of: mean, min, max")

    if output_tif:
        dst = Path(output_tif)
    else:
        stem = Path(input_parquet).stem
        dst = Path("data") / f"{stem}_{value_col.lower()}_{aggregation}.tif"

    try:
        output = parquet_to_tif(
            input_parquet=input_parquet,
            output_tif=dst,
            value_col=value_col,
            resolution=resolution,
            aggregation=aggregation,
        )
    except Exception as exc:
        console.print(f"[red]Conversion failed:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(f"[green]Created:[/green] {output}")
