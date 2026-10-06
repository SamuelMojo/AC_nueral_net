import time
import subprocess
import os

REMOTE_HOST = "ec2-user@16.61.38.122"
KEY_PATH = "samuelojo-key.pem"
LOCAL_DIR = r"C:\Users\samue\assetto_corsa_gym\outputs"
REMOTE_DIR = "~/assetto_corsa_gym/outputs/"

print("Starting automatic output sync to EC2...")

while True:
    try:
        # Recursively copy all parquet files and summaries to EC2
        cmd = f'scp -i "{KEY_PATH}" -r "{LOCAL_DIR}\\*" {REMOTE_HOST}:{REMOTE_DIR}'
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[{time.strftime('%H:%M:%S')}] Successfully synced outputs to EC2.")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Sync error: {e}")
    
    # Wait 10 seconds before checking and syncing again
    time.sleep(10)