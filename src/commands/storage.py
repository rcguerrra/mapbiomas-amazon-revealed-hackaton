from __future__ import annotations

import os
from typing import Optional

import typer
from rich.console import Console

from .. import config
from ..utils.storage_client import StorageClient

console = Console()
app = typer.Typer(help="Storage helpers (local, S3, GCS) via StorageClient")


def _resolve_gcp_credentials(credentials_path: Optional[str]) -> Optional[str]:
    return (
        credentials_path
        or config.MBENGINE_GCP_SERVICE_ACCOUNT
        or os.getenv("GCP_PKEY")
    )


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


@app.command(help="Read text content from a file.")
def read_text(
    file_path: str = typer.Argument(..., help="File path (local, s3://, gcs://)."),
    max_chars: int = typer.Option(5000, "--max-chars", help="Maximum chars to print."),
    credentials_path: Optional[str] = typer.Option(None, "--credentials-path", "-c"),
    aws_key_id: Optional[str] = typer.Option(None, "--aws-key-id"),
    aws_secret: Optional[str] = typer.Option(None, "--aws-secret"),
    aws_region: Optional[str] = typer.Option(None, "--aws-region"),
):
    client = _build_client(credentials_path, aws_key_id, aws_secret, aws_region)
    text = client.read_text(file_path)
    console.print(text[:max_chars])


@app.command(help="Save text content to a path.")
def save_text(
    file_path: str = typer.Argument(..., help="Destination file path."),
    content: str = typer.Argument(..., help="Text content to save."),
    credentials_path: Optional[str] = typer.Option(None, "--credentials-path", "-c"),
    aws_key_id: Optional[str] = typer.Option(None, "--aws-key-id"),
    aws_secret: Optional[str] = typer.Option(None, "--aws-secret"),
    aws_region: Optional[str] = typer.Option(None, "--aws-region"),
):
    client = _build_client(credentials_path, aws_key_id, aws_secret, aws_region)
    client.save_text(content, file_path)
    console.print(f"[green]Saved:[/green] {file_path}")


@app.command(help="Make remote object public and return URL when supported.")
def make_public(
    file_path: str = typer.Argument(..., help="Path must be s3:// or gcs://."),
    credentials_path: Optional[str] = typer.Option(None, "--credentials-path", "-c"),
    aws_key_id: Optional[str] = typer.Option(None, "--aws-key-id"),
    aws_secret: Optional[str] = typer.Option(None, "--aws-secret"),
    aws_region: Optional[str] = typer.Option(None, "--aws-region"),
):
    client = _build_client(credentials_path, aws_key_id, aws_secret, aws_region)
    url = client.make_public(file_path)
    if url:
        console.print(f"[green]{url}[/green]")
    else:
        console.print("[yellow]Not supported for this path.[/yellow]")
