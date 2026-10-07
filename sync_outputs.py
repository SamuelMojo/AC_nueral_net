import os
import time
import boto3

BUCKET_NAME = "samuelojo-ac-telemetry"
LOCAL_DIR = r"C:\Users\samue\assetto_corsa_gym\outputs"

print("Starting automatic output sync to AWS S3...")
s3_client = boto3.client('s3')

while True:
    try:
        upload_count = 0
        if os.path.exists(LOCAL_DIR):
            for root, dirs, files in os.walk(LOCAL_DIR):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, LOCAL_DIR)
                    s3_key = f"outputs/{relative_path.replace(os.sep, '/')}"
                    
                    try:
                        s3_client.upload_file(local_path, BUCKET_NAME, s3_key)
                        upload_count += 1
                    except Exception:
                        pass
                        
        if upload_count > 0:
            print(f"[{time.strftime('%H:%M:%S')}] Successfully synced outputs to S3 ({upload_count} files).")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] S3 Sync error: {e}")
    
    # Check for new files every 10 seconds
    time.sleep(10)