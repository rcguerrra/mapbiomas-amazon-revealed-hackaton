from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import ee

from . import config


def _resolve_credentials_path(credentials_path: Optional[Union[str, Path]] = None) -> Path:
    env_path = (
        credentials_path
        or config.MBENGINE_GCP_SERVICE_ACCOUNT
        or "credentials/mapbiomas-workspace-10-gee.json"
    )
    path = Path(env_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"GEE credentials file not found: {path}. "
            "Set MBENGINE_GCP_SERVICE_ACCOUNT in .env or pass credentials_path."
        )
    return path


def auth(credentials_path: Optional[Union[str, Path]] = None) -> ee.ServiceAccountCredentials:
    """
    Authenticate Google Earth Engine using a service account JSON key file.
    Returns initialized credentials object.
    """
    path = _resolve_credentials_path(credentials_path)

    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    client_email = payload.get("client_email")
    if not client_email:
        raise ValueError(f"Invalid service account file (missing client_email): {path}")

    return ee.ServiceAccountCredentials(client_email, str(path))


def init(
    credentials_path: Optional[Union[str, Path]] = None,
    project: Optional[str] = None,
) -> None:
    """
    Authenticate and initialize Earth Engine.
    Authenticate and initialize Earth Engine.
    """
    credentials = auth(credentials_path)

    if project:
        ee.Initialize(credentials=credentials, project=project)
    else:
        ee.Initialize(credentials=credentials)

    return None
