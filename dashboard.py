import os
import time
from datetime import timezone
import s3fs
import ollama
import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit Page Configuration
st.set_page_config(
    page_title="Assetto Corsa Telemetry Dashboard",
    page_icon=None,
    layout="wide",
)

# Custom CSS for Square Tiles Styling (Fastlytics / Modern Dark Theme Style)
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #2d2d2d;
        border-radius: 4px;
        padding: 20px;
        text-align: center;
        aspect-ratio: 1 / 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 1rem;
        font-weight: 500;
        color: #888888;
        margin-top: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .summary-box {
        background-color: #1e1e1e;
        border: 1px solid #2d2d2d;
        border-radius: 4px;
        padding: 15px;
        margin-bottom: 20px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# S3 Configuration
S3_BUCKET_PATH = "samuelojo-ac-telemetry/outputs"

# Precise Monza Track Sector Mapping
MONZA_CORNERS = [
    {"name": "Variante del Rettifilo (Turn 1 Chicane)", "start": 0.00, "end": 0.12},
    {"name": "Curva Grande (Turn 3)", "start": 0.12, "end": 0.28},
    {"name": "Variante della Roggia (Turn 4 Chicane)", "start": 0.28, "end": 0.45},
    {"name": "Lesmo Curves (Turns 6-7)", "start": 0.45, "end": 0.65},
    {"name": "Variante Ascari (Turn 8 Chicane)", "start": 0.65, "end": 0.82},
    {"name": "Curva Parabolica (Turn 11)", "start": 0.82, "end": 1.00},
]


def identify_monza_corner(progress_val):
    for corner in MONZA_CORNERS:
        if corner["start"] <= progress_val < corner["end"]:
            return corner["name"]
    return "Unknown Section"


@st.cache_data(ttl=5)
def load_all_runs():
    """Scans S3 bucket and loads all parquet files for global aggregation."""
    try:
        fs = s3fs.S3FileSystem(anon=False)
        files = fs.glob(f"{S3_BUCKET_PATH}/**/*.parquet")
        if not files:
            return []

        # Sort by AWS LastModified time (newest first)
        files.sort(key=lambda x: fs.info(x)["LastModified"], reverse=True)
        return [f"s3://{f}" for f in files]
    except Exception as e:
        st.error(f"S3 Connection Error: {e}")
        return []


def is_simulation_live(parquet_files, threshold_seconds=120):
    """Checks if the most recent S3 parquet file was modified within the threshold."""
    if not parquet_files:
        return False
    try:
        fs = s3fs.S3FileSystem(anon=False)
        latest_file = parquet_files[0].replace("s3://", "")
        file_info = fs.info(latest_file)
        
        # S3 LastModified is timezone-aware. Compare against current UTC time.
        mtime = file_info["LastModified"]
        current_time = pd.Timestamp.now(tz=timezone.utc)
        diff = (current_time - mtime).total_seconds()
        return diff <= threshold_seconds
    except Exception:
        return False


def get_historical_context(parquet_files, current_file, max_history=10):
    """Aggregates summary statistics from the previous 10 runs for rolling memory comparison."""
    if not parquet_files:
        return "No historical run data available."

    historical_files = [f for f in parquet_files if f != current_file][:max_history]
    if not historical_files:
        return "This is the first recorded training run."

    history_lines = []
    for fpath in historical_files:
        try:
            df_prev = pd.read_parquet(fpath)
            p_speed = df_prev["speed"].max() if "speed" in df_prev.columns else 0.0
            m_speed = df_prev["speed"].mean() if "speed" in df_prev.columns else 0.0
            max_g = df_prev["accelY"].abs().max() if "accelY" in df_prev.columns else 0.0
            off = bool(df_prev["out_of_track"].any()) if "out_of_track" in df_prev.columns else False
            
            lap_str = "N/A"
            for col in ["lap_time", "lap_duration", "lapTime"]:
                if col in df_prev.columns:
                    val = df_prev[col].iloc[-1] if not df_prev[col].empty else 0.0
                    if val > 0:
                        lap_str = f"{val:.3f}s"
                        break

            history_lines.append(
                f"- Run {os.path.basename(fpath)} | Lap Time: {lap_str} | Peak Speed: {p_speed:.1f} m/s | Mean Speed: {m_speed:.1f} m/s | Max G: {max_g:.2f} G | Off-track: {off}"
            )
        except Exception:
            continue

    return "\n".join(history_lines)


def get_reward_progression(parquet_files):
    """Extracts cumulative reward trends across all available runs for macro convergence plotting."""
    run_data = []
    # parquet_files are sorted newest-first by load_all_runs. Reverse for chronological chart.
    sorted_files = list(reversed(parquet_files))
    for idx, fpath in enumerate(sorted_files):
        try:
            df_run = pd.read_parquet(fpath)
            total_rew = df_run["reward"].sum() if "reward" in df_run.columns else float(len(df_run))
            run_data.append({"Run": f"Run {idx+1}", "Total Reward": total_rew})
        except Exception:
            continue
    return pd.DataFrame(run_data)


def query_telemetry_context(df, file_path, total_runs, historical_context):
    if df is None or df.empty:
        return "No telemetry data available yet."

    speed_max = df["speed"].max() if "speed" in df.columns else 0.0
    speed_mean = df["speed"].mean() if "speed" in df.columns else 0.0