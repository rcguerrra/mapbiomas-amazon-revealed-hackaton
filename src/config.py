import os
from dotenv import load_dotenv

if os.getenv("ENV") == None:
    load_dotenv()

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# Backward compatibility with legacy variable name.
MBENGINE_GCP_SERVICE_ACCOUNT = (
    os.getenv("MBENGINE_GCP_SERVICE_ACCOUNT") or GOOGLE_APPLICATION_CREDENTIALS
)
