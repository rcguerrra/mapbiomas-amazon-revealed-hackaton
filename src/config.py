import json
import os
import re
from typing import Any, Dict, Optional

from dotenv import load_dotenv

if os.getenv("ENV") == None:
    load_dotenv()

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# Backward compatibility with legacy variable name.
MBENGINE_GCP_SERVICE_ACCOUNT = (
    os.getenv("MBENGINE_GCP_SERVICE_ACCOUNT") or GOOGLE_APPLICATION_CREDENTIALS
)


def _normalize_multiline_private_key(raw_json: str) -> str:
    """
    Normalize invalid literal line breaks inside JSON private_key values.

    Some deployments paste service-account JSON with real newlines in private_key.
    JSON requires escaped newlines (\\n), otherwise json.loads raises
    "Invalid control character".
    """
    pattern = r'("private_key"\s*:\s*")(.*?)(")'

    def _replace(match: re.Match[str]) -> str:
        value = match.group(2)
        value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
        return f'{match.group(1)}{value}{match.group(3)}'

    return re.sub(pattern, _replace, raw_json, flags=re.DOTALL)


def _parse_service_account_json(raw: str) -> Dict[str, Any]:
    """
    Parse service-account JSON that may contain trailing garbage (duplicate JSON),
    real newlines inside private_key, or be double-encoded as a JSON string.
    """
    raw = raw.strip().lstrip("\ufeff")
    decoder = json.JSONDecoder()

    def _first_object(payload: str) -> Any:
        obj, end = decoder.raw_decode(payload)
        # Trailing content after the first JSON value is ignored (duplicate paste, etc.).
        _ = payload[end:].strip()
        return obj

    def _to_dict(payload: str) -> Dict[str, Any]:
        try:
            obj = _first_object(payload)
        except json.JSONDecodeError:
            normalized = _normalize_multiline_private_key(payload)
            if normalized == payload:
                raise
            obj = _first_object(normalized)
        if isinstance(obj, str) and obj.strip().startswith("{"):
            obj = _first_object(obj.strip())
        if not isinstance(obj, dict):
            raise ValueError("GCP credentials must be a JSON object (service account).")
        return obj

    return _to_dict(raw)


def load_gcp_service_account_dict() -> Optional[Dict[str, Any]]:
    """
    Load GCP service account JSON for use with google-cloud and gcs/fsspec.

    Resolution order (for Streamlit Cloud, use inline JSON in secrets / env):
    1. GCP_SERVICE_ACCOUNT_JSON — full JSON string
    2. GOOGLE_APPLICATION_CREDENTIALS_JSON — full JSON string
    3. GOOGLE_APPLICATION_CREDENTIALS — JSON string starting with '{' or path to a JSON file
    4. MBENGINE_GCP_SERVICE_ACCOUNT — same as (3)
    """
    for env_key in ("GCP_SERVICE_ACCOUNT_JSON", "GOOGLE_APPLICATION_CREDENTIALS_JSON"):
        raw = os.getenv(env_key)
        if raw and raw.strip():
            return _parse_service_account_json(raw)

    for env_key in ("GOOGLE_APPLICATION_CREDENTIALS", "MBENGINE_GCP_SERVICE_ACCOUNT"):
        raw = os.getenv(env_key)
        if not raw or not raw.strip():
            continue
        raw = raw.strip()
        if raw.startswith("{"):
            return _parse_service_account_json(raw)
        if os.path.isfile(raw):
            with open(raw, encoding="utf-8") as f:
                return json.load(f)

    return None
