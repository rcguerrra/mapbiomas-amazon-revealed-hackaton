from typing import Optional

import ee
import typer
from rich.console import Console

from .. import config
from ..gee import auth as gee_auth
from ..gee import init as gee_init

console = Console()
app = typer.Typer(help="Google Earth Engine helpers")


@app.command(help="Authenticate Earth Engine with a service account JSON.")
def auth(
    credentials_path: Optional[str] = typer.Option(
        None, "--credentials-path", "-c", help="Path to service account JSON."
    )
):
    creds = gee_auth(credentials_path)
    console.print(f"[green]Auth OK[/green] ({creds.service_account_email})")


@app.command(help="Initialize Earth Engine and run a health check request.")
def init(
    credentials_path: Optional[str] = typer.Option(
        None, "--credentials-path", "-c", help="Path to service account JSON."
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Google Cloud project id."
    ),
):
    gee_init(credentials_path=credentials_path, project=project)
    # If this request works, initialization is valid and API access is active.
    value = ee.Number(1).getInfo()
    console.print(f"[green]Init OK[/green] (healthcheck={value})")


@app.command(help="Show current configured credential path.")
def status():
    configured = config.GOOGLE_APPLICATION_CREDENTIALS or config.MBENGINE_GCP_SERVICE_ACCOUNT
    if configured:
        console.print(f"[blue]GOOGLE_APPLICATION_CREDENTIALS:[/blue] {configured}")
    else:
        console.print(
            "[yellow]GOOGLE_APPLICATION_CREDENTIALS is not set. "
            "Using default path in src.gee._resolve_credentials_path().[/yellow]"
        )
