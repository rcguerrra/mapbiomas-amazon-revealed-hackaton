import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

if os.getenv("ENV") == None:
    load_dotenv()

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# Backward compatibility with legacy variable name.
MBENGINE_GCP_SERVICE_ACCOUNT = (
    os.getenv("MBENGINE_GCP_SERVICE_ACCOUNT") or GOOGLE_APPLICATION_CREDENTIALS
)


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
            return json.loads(raw.strip())

    for env_key in ("GOOGLE_APPLICATION_CREDENTIALS", "MBENGINE_GCP_SERVICE_ACCOUNT"):
        raw = os.getenv(env_key)
        if not raw or not raw.strip():
            continue
        raw = raw.strip()
        if raw.startswith("{"):
            return json.loads(raw)
        if os.path.isfile(raw):
            with open(raw, encoding="utf-8") as f:
                return json.load(f)

    return None
