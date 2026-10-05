import glob
import os
import pandas as pd

# Find the latest parquet file
latest_file = max(
    glob.glob("outputs/**/laps/*.parquet", recursive=True),
    key=os.path.getctime,
)
df = pd.read_parquet(latest_file)

# Print every single metric the game is tracking
print(list(df.columns))