from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .. import config
from ..utils.storage_client import StorageClient

console = Console()
app = typer.Typer(help="Storage helpers (local, S3, GCS) via StorageClient")


def _resolve_gcp_credentials(credentials_path: Optional[str]):
    if credentials_path:
        return credentials_path
    return config.load_gcp_service_account_dict()


def _build_client(
    credentials_path: Optional[str],
    aws_key_id: Optional[str],
    aws_secret: Optional[str],
    aws_region: Optional[str],
) -> StorageClient:
    return StorageClient(
        gcp_credentials=_resolve_gcp_credentials(credentials_path),
        aws_key_id=aws_key_id,
        aws_secret=aws_secret,
        aws_region=aws_region,
    )


def _normalize_path(path: str) -> str:
    if path.startswith("gcs://"):
        return "gs://" + path[len("gcs://") :]
    return path


@app.command(help="Show resolved credential sources for StorageClient.")
def status(
    credentials_path: Optional[str] = typer.Option(
        None, "--credentials-path", "-c", help="Path to GCP service account JSON."
    )
):
    resolved = _resolve_gcp_credentials(credentials_path)
    if resolved:
        console.print(f"[green]GCP credentials:[/green] {resolved}")
    else:
        console.print("[yellow]GCP credentials not configured.[/yellow]")


@app.command(help="List files in a directory/path with optional regex filter.")
def ls(
    dir_path: str = typer.Argument(..., help="Directory path (local, s3://, gcs://)."),
    pattern: Optional[str] = typer.Option(None, "--pattern", "-p", help="Regex filter."),
    credentials_path: Optional[str] = typer.Option(None, "--credentials-path", "-c"),
    aws_key_id: Optional[str] = typer.Option(None, "--aws-key-id"),
    aws_secret: Optional[str] = typer.Option(None, "--aws-secret"),
    aws_region: Optional[str] = typer.Option(None, "--aws-region"),
):
    client = _build_client(credentials_path, aws_key_id, aws_secret, aws_region)
    files = client.ls_dir(dir_path, pattern)
    for file in files:
        console.print(file)


@app.command(help="Check if path exists.")
def exists(
    path: str = typer.Argument(..., help="Path (local, s3://, gcs://)."),
    credentials_path: Optional[str] = typer.Option(None, "--credentials-path", "-c"),
    aws_key_id: Optional[str] = typer.Option(None, "--aws-key-id"),
    aws_secret: Optional[str] = typer.Option(None, "--aws-secret"),
    aws_region: Optional[str] = typer.Option(None, "--aws-region"),
):
    client = _build_client(credentials_path, aws_key_id, aws_secret, aws_region)
    result = client.exists(path)
    console.print(result)


@app.command(help="Check read access for a file or directory.")
def check_access(
    path: str = typer.Argument(..., help="Path (local, s3://, gcs://)."),
    credentials_path: Optional[str] = typer.Option(None, "--credentials-path", "-c"),
    aws_key_id: Optional[str] = typer.Option(None, "--aws-key-id"),
    aws_secret: Optional[str] = typer.Option(None, "--aws-secret"),
    aws_region: Optional[str] = typer.Option(None, "--aws-region"),
):
    client = _build_client(credentials_path, aws_key_id, aws_secret, aws_region)
    result = client.check_access(path)
    console.print(result)

@app.command(help="Download file from storage (s3:// or gs://) to local path.")
def download(
    source_path: str = typer.Argument(..., help="Source path (s3://, gs://, gcs://, or local)."),
    destination_path: Optional[str] = typer.Argument(
        None, help="Local destination path. Defaults to data/<source filename>."
    ),
    credentials_path: Optional[str] = typer.Option(None, "--credentials-path", "-c"),
    aws_key_id: Optional[str] = typer.Option(None, "--aws-key-id"),
    aws_secret: Optional[str] = typer.Option(None, "--aws-secret"),
    aws_region: Optional[str] = typer.Option(None, "--aws-region"),
):
    client = _build_client(credentials_path, aws_key_id, aws_secret, aws_region)
    source_path = _normalize_path(source_path)

    if destination_path:
        destination = Path(destination_path)
    else:
        source_name = Path(source_path).name
        if not source_name:
            raise typer.BadParameter(
                "Could not infer filename from source_path. Provide destination_path explicitly."
            )
        destination = Path("data") / source_name

    destination.parent.mkdir(parents=True, exist_ok=True)

    content = client.read(source_path)
    destination.write_bytes(content)
    console.print(f"[green]Downloaded:[/green] {source_path} -> {destination}")


@app.command(help="Read file content directly from storage (s3:// or gs://).")
def read(
    source_path: str = typer.Argument(
        ..., help="Source file path (s3://, gs://, gcs://, or local)."
    ),
    output_path: Optional[str] = typer.Option(
        None, "--output", "-o", help="Optional local output path for raw bytes."
    ),
    encoding: str = typer.Option(
        "utf-8", "--encoding", "-e", help="Text encoding when printing to stdout."
    ),
    max_chars: int = typer.Option(
        20000, "--max-chars", help="Maximum number of characters to print."
    ),
    credentials_path: Optional[str] = typer.Option(None, "--credentials-path", "-c"),
    aws_key_id: Optional[str] = typer.Option(None, "--aws-key-id"),
    aws_secret: Optional[str] = typer.Option(None, "--aws-secret"),
    aws_region: Optional[str] = typer.Option(None, "--aws-region"),
):
    client = _build_client(credentials_path, aws_key_id, aws_secret, aws_region)
    source_path = _normalize_path(source_path)
    content = client.read(source_path)

    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        console.print(f"[green]Saved:[/green] {source_path} -> {destination}")
        return

    try:
        text = content.decode(encoding)
    except UnicodeDecodeError:
        raise typer.BadParameter(
            "Could not decode file as text with selected encoding. "
            "Use --output to save raw bytes locally."
        )

    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
        console.print(text)
        console.print(
            f"\n[yellow]Output truncated to {max_chars} characters. "
            "Use --max-chars to increase.[/yellow]"
        )
    else:
        console.print(text)
