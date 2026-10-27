"""Sync new telemetry parquet files from outputs/ to S3.

Differences from sync_outputs.py:
* Uploads a file only when it is new or has changed (the old script re-uploaded EVERYTHING
  every 10s, which saturated the upload link and reset every file's S3 'LastModified', so the
  dashboard could no longer tell which run was newest or whether a sim was live).
* Only uploads *.parquet (what the dashboard reads), not checkpoints / tensorboard / wandb files.
* Skips files still being written so the dashboard never reads a half-written parquet.
* Prints upload errors instead of silently swallowing them.
"""

import os
import time

import boto3

BUCKET_NAME = "samuelojo-ac-telemetry"
LOCAL_DIR = r"C:\Users\samue\assetto_corsa_gym\outputs"
EXTENSIONS = (".parquet",)
SETTLE_SECONDS = 3   # ignore files modified in the last N seconds (still being written)
POLL_SECONDS = 5

print("Starting automatic output sync to AWS S3...")
s3 = boto3.client("s3")
uploaded = {}  # local path -> (size, mtime_ns) at last successful upload

while True:
    count = 0
    try:
        if os.path.exists(LOCAL_DIR):
            for root, _dirs, files in os.walk(LOCAL_DIR):
                for name in files:
                    if not name.lower().endswith(EXTENSIONS):
                        continue
                    path = os.path.join(root, name)
                    try:
                        st = os.stat(path)
                    except OSError:
                        continue
                    if time.time() - st.st_mtime < SETTLE_SECONDS:
                        continue
                    sig = (st.st_size, st.st_mtime_ns)
                    if uploaded.get(path) == sig:
                        continue
                    key = "outputs/" + os.path.relpath(path, LOCAL_DIR).replace(os.sep, "/")
                    try:
                        s3.upload_file(path, BUCKET_NAME, key)
                        uploaded[path] = sig
                        count += 1
                    except Exception as e:  # noqa: BLE001
                        print(f"[{time.strftime('%H:%M:%S')}] upload failed for {key}: {e}")
        if count:
            print(f"[{time.strftime('%H:%M:%S')}] Synced {count} new/changed file(s) to S3.")
    except Exception as e:  # noqa: BLE001
        print(f"[{time.strftime('%H:%M:%S')}] S3 Sync error: {e}")
    time.sleep(POLL_SECONDS)
