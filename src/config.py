import os
from dotenv import load_dotenv

if os.getenv("ENV") == None:
    load_dotenv()

MBENGINE_GCP_SERVICE_ACCOUNT = os.getenv("MBENGINE_GCP_SERVICE_ACCOUNT")