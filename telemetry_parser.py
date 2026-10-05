import glob
import os
import pandas as pd


class TelemetryParser:

  def __init__(self, outputs_dir="outputs"):
    self.outputs_dir = outputs_dir

  def get_latest_parquet_file(self):
    """Finds the most recently created parquet telemetry file."""
    pattern = os.path.join(self.outputs_dir, "**", "laps", "*.parquet")
    files = glob.glob(pattern, recursive=True)
    if not files:
      raise FileNotFoundError(
          f"No parquet telemetry files found in {self.outputs_dir}"
      )
    return max(files, key=os.path.getctime)

  def get_all_parquet_files(self):
    """Finds all parquet telemetry files across all training sessions."""
    pattern = os.path.join(self.outputs_dir, "**", "laps", "*.parquet")
    return glob.glob(pattern, recursive=True)

  def get_global_training_stats(self):
    """Calculates total attempts, cumulative steps, and estimated running time."""
    files = self.get_all_parquet_files()
    if not files:
      return {"total_attempts": 0, "cumulative_steps": 0, "estimated_time_min": 0.0}

    total_attempts = len(files)
    cumulative_steps = 0

    for file_path in files:
      try:
        df = pd.read_parquet(file_path)
        cumulative_steps += len(df)
      except Exception:
        pass

    estimated_time_seconds = cumulative_steps / 25.0
    estimated_time_minutes = estimated_time_seconds / 60.0

    return {
        "total_attempts": total_attempts,
        "cumulative_steps": cumulative_steps,
        "estimated_time_min": round(estimated_time_minutes, 2),
    }

  def parse_latest_run(self):
    """Loads the latest parquet file and extracts aggregated summary statistics."""
    file_path = self.get_latest_parquet_file()
    df = pd.read_parquet(file_path)

    # Extract high-level summary metrics
    summary = {
        "file_path": file_path,
        "total_steps": int(len(df)),
        "max_speed_ms": (
            float(df["speed"].max()) if "speed" in df.columns else 0.0
        ),
        "mean_speed_ms": (
            float(df["speed"].mean()) if "speed" in df.columns else 0.0
        ),
        "max_track_progress": (
            float(df["NormalizedSplinePosition"].max() * 100)
            if "NormalizedSplinePosition" in df.columns
            else 0.0
        ),
        "max_lateral_g": (
            float(df["accelY"].abs().max()) if "accelY" in df.columns else 0.0
        ),
        "mean_rear_slip": (
            float(df["tyre_slip_ratio_rr"].mean())
            if "tyre_slip_ratio_rr" in df.columns
            else 0.0
        ),
        "went_off_track": (
            bool(df["out_of_track"].any())
            if "out_of_track" in df.columns
            else False
        ),
    }

    return summary, df


if __name__ == "__main__":
  parser = TelemetryParser()
  try:
    stats, raw_df = parser.parse_latest_run()
    global_stats = parser.get_global_training_stats()

    print("Successfully parsed latest telemetry file:")
    for key, value in stats.items():
      print(f"  - {key}: {value}")

    print("\nGlobal Training Progress:")
    for key, value in global_stats.items():
      print(f"  - {key}: {value}")
      
  except Exception as e:
    print(f"Error parsing telemetry: {e}")