python -m src.console storage download gs://amazon-revealed/Point-Cloud/01_ENTREGA_23_08_2023/NP/ACRE_005_NP_8973-536.laz

# Read content directly from storage and print in terminal
python -m src.console storage read gs://bucket/caminho/arquivo.txt

# Read from storage and save locally
python -m src.console storage read gs://bucket/caminho/arquivo.laz --output data/arquivo.laz

# Run Streamlit app (supports direct Parquet from gs:// or s3:// in sidebar)
streamlit run src/main.py

# Convert local LAZ/LAS to Parquet
python -m src.console lidar to-parquet data/ACRE_005_NP_8973-536.laz

# Streamlit runtime flow for LAZ on storage:
# 1) Check remote .laz existence
# 2) Download to local data/
# 3) Convert to local .parquet
# 4) Load data from converted Parquet

## Streamlit Cloud Secrets

In Streamlit Cloud, open **App settings -> Secrets** and add:

```toml
GCP_SERVICE_ACCOUNT_JSON = """
{
  "type": "service_account",
  "project_id": "YOUR_PROJECT_ID",
  "private_key_id": "YOUR_PRIVATE_KEY_ID",
  "private_key": "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY\n-----END PRIVATE KEY-----\n",
  "client_email": "your-service-account@your-project.iam.gserviceaccount.com",
  "client_id": "YOUR_CLIENT_ID",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/..."
}
"""
```

Notes:
- Keep `\n` line breaks inside `private_key`.
- If `private_key` is pasted with real line breaks, runtime now normalizes it automatically.
- Do not commit real credentials to the repository.
