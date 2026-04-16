"""
Test BigQuery access for the LiDAR Amazonia table.

Usage:
    uv run python scripts/test_bigquery.py
    uv run python scripts/test_bigquery.py --credentials credentials/rcguerra82-dev.json
"""
import argparse
import os
import sys
import time

from dotenv import load_dotenv

BQ_PROJECT = "mapbiomas"
BQ_TABLE   = "mapbiomas.mapbiomas_lidar.amazonia_3d"
load_dotenv()
CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials/mapbiomas-workspace-10-gee.json")


def main(credentials: str) -> None:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials

    print(f"Credentials : {credentials}")
    print(f"Project     : {BQ_PROJECT}")
    print(f"Table       : {BQ_TABLE}")
    print()

    try:
        from google.cloud import bigquery
    except ImportError:
        print("ERROR: google-cloud-bigquery not installed. Run: uv pip install google-cloud-bigquery")
        sys.exit(1)

    client = bigquery.Client(project=BQ_PROJECT)

    # 1. Check table existence
    print("1. Checking table existence...")
    try:
        table = client.get_table(BQ_TABLE)
        print(f"   OK — {table.num_rows:,} rows, {len(table.schema)} columns")
        print(f"   Schema: {[f.name for f in table.schema]}")
    except Exception as exc:
        print(f"   FAIL — {exc}")
        sys.exit(1)

    print()

    # 2. Fetch a small sample
    print("2. Fetching 5 rows...")
    t0 = time.time()
    try:
        query = f"SELECT * FROM `{BQ_TABLE}` LIMIT 5"
        df = client.query(query).result().to_dataframe()
        elapsed = time.time() - t0
        print(f"   OK — {len(df)} rows in {elapsed:.2f}s")
        print(df.to_string(index=False))
    except Exception as exc:
        print(f"   FAIL — {exc}")
        sys.exit(1)

    print()

    # 3. Count rows
    print("3. Counting total rows...")
    t0 = time.time()
    try:
        count = client.query(f"SELECT COUNT(*) AS n FROM `{BQ_TABLE}`").result().to_dataframe()["n"][0]
        elapsed = time.time() - t0
        print(f"   OK — {count:,} rows ({elapsed:.2f}s)")
    except Exception as exc:
        print(f"   FAIL — {exc}")
        sys.exit(1)

    print()
    print("All checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", default=CREDENTIALS)
    args = parser.parse_args()
    main(args.credentials)
