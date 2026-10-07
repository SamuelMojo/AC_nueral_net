import os
import time
from datetime import timezone
import s3fs
import ollama
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Streamlit Page Configuration (Must be the absolute first Streamlit command)
st.set_page_config(
    page_title="Assetto Corsa Telemetry Dashboard",
    page_icon=None,
    layout="wide",
)

# Automatically rerun the script every 10 seconds to pull fresh data from S3
count = st_autorefresh(interval=600000, limit=None, key="datarefresh")

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
    """Scans S3 bucket and loads all parquet files safely, handling active uploads."""
    try:
        fs = s3fs.S3FileSystem(anon=False)
        files = fs.glob(f"{S3_BUCKET_PATH}/**/*.parquet")
        if not files:
            return []

        valid_files = []
        for f in files:
            try:
                info = fs.info(f)
                mtime = info.get("LastModified")
                if mtime:
                    valid_files.append((f, mtime))
            except Exception:
                continue

        valid_files.sort(key=lambda x: x[1], reverse=True)
        return [f"s3://{f}" for f, _ in valid_files]
    except Exception as e:
        st.error(f"S3 Connection Error: {e}")
        return []


def is_simulation_live(parquet_files, threshold_seconds=120):
    if not parquet_files:
        return False
    try:
        fs = s3fs.S3FileSystem(anon=False)
        latest_file = parquet_files[0].replace("s3://", "")
        file_info = fs.info(latest_file)
        
        mtime = file_info["LastModified"]
        current_time = pd.Timestamp.now(tz=timezone.utc)
        diff = (current_time - mtime).total_seconds()
        return diff <= threshold_seconds
    except Exception:
        return False


def get_historical_context(parquet_files, current_file, max_history=10):
    if not parquet_files:
        return "No historical run data available."

    historical_files = [f for f in parquet_files if f != current_file][:max_history]
    if not historical_files:
        return "This is the first recorded training run."

    fs = s3fs.S3FileSystem(anon=False)
    history_lines = []
    for fpath in historical_files:
        try:
            clean_path = fpath.replace("s3://", "")
            with fs.open(clean_path, "rb") as f:
                df_prev = pd.read_parquet(f)
            
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
    run_data = []
    fs = s3fs.S3FileSystem(anon=False)
    sorted_files = list(reversed(parquet_files))
    for idx, fpath in enumerate(sorted_files):
        try:
            clean_path = fpath.replace("s3://", "")
            with fs.open(clean_path, "rb") as f:
                df_run = pd.read_parquet(f)
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
    lat_g = df["accelY"].abs().max() if "accelY" in df.columns else 0.0
    off_track = bool(df["out_of_track"].any()) if "out_of_track" in df.columns else False
    
    lap_time_str = "Not Completed / In Progress"
    for col in ["lap_time", "lap_duration", "lapTime"]:
        if col in df.columns:
            val = df[col].iloc[-1] if not df[col].empty else 0.0
            if val > 0:
                lap_time_str = f"{val:.3f} seconds"
                break

    off_track_location = "None"
    if off_track and "progress" in df.columns:
        off_frames = df[df["out_of_track"] == True]
        if not off_frames.empty:
            off_frames = off_frames.copy()
            off_frames["corner"] = off_frames["progress"].apply(identify_monza_corner)
            dominant_corner = off_frames["corner"].mode()
            if not dominant_corner.empty and pd.notna(dominant_corner[0]):
                off_track_location = str(dominant_corner[0])

    context = f"""
    [ACTIVE RUN METRICS]
    - Total Training Runs So Far: {total_runs}
    - Active Log File: {os.path.basename(file_path)}
    - Completed Lap Time: {lap_time_str}
    - Total Frames: {len(df)}
    - Peak Speed: {speed_max:.1f} m/s (Average Speed: {speed_mean:.1f} m/s)
    - Max Cornering G-Force: {lat_g:.2f} G
    - Went Off Track: {off_track} (Primary Off-Track Corner: {off_track_location})

    [LAST 10 RUNS ROLLING MEMORY]
    {historical_context}
    """
    return context


def generate_summary_text(selected_file, parquet_files, df, total_events):
    summary_file = os.path.basename(selected_file) + ".summary.txt"
    historical_context = get_historical_context(parquet_files, selected_file, max_history=10)
    telemetry_context = query_telemetry_context(df, selected_file, total_events, historical_context)
    
    summary_prompt = f"""
        You are an R&D simulation engineer explaining how an AI driving model is learning. 
        Provide a simple, clear 2-sentence summary using everyday language.
        
        {telemetry_context}
        
        Strict Instructions:
        - NEVER start your response with filler phrases like "Here is a 2-sentence summary", "Sure", or introductory text. Jump immediately into the analysis.
        - Call the agent solely "driver".
        - If a lap time is recorded, include it explicitly.
        - If an off-track excursion or wall crash occurred, mention ONLY the single most frequent or critical Monza corner where it happened based on the mapping.
        - Include specific numbers and figures (such as lap time, peak speed, average speed, or G-forces) directly in the text.
        - Keep it simple so non-technical people can easily understand how performance compares to past runs.
        """
    try:
        client = ollama.Client(host="http://100.72.210.89:11434") 
        response = client.chat(
            model="llama3.1:latest",
            messages=[{"role": "system", "content": summary_prompt}],
        )
        summary_text = response["message"]["content"].strip()
        
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary_text)
            
        return summary_text
    except Exception as e:
        return f"AI Offline Error: {e}"


def main():
    st.title("Assetto Corsa Telemetry Dashboard")
    st.markdown("---")

    parquet_files = load_all_runs()
    total_events = len(parquet_files)

    live_status = is_simulation_live(parquet_files)
    if not live_status:
        st.warning("No live simulation right now — viewing historical data and manual debriefs.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total_events}</div>
                <div class="metric-label">Total Events</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if total_events == 0:
        st.info("No telemetry logs found in the S3 bucket.")
        return

    selected_file = st.selectbox(
        "Select Training Run Log",
        parquet_files,
        format_func=lambda x: os.path.basename(x),
    )
    
    try:
        fs = s3fs.S3FileSystem(anon=False)
        clean_selected = selected_file.replace("s3://", "")
        with fs.open(clean_selected, "rb") as f:
            df = pd.read_parquet(f)
    except Exception as e:
        st.error(f"Failed to load selected telemetry file: {e}")
        return

    with col2:
        max_speed = df["speed"].max() if "speed" in df.columns else 0.0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{max_speed:.1f}</div>
                <div class="metric-label">Peak Speed (m/s)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        max_lat_g = (
            df["accelY"].abs().max() if "accelY" in df.columns else 0.0
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{max_lat_g:.2f}G</div>
                <div class="metric-label">Max Lateral G</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        total_frames = len(df)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total_frames}</div>
                <div class="metric-label">Selected Run Frames</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    chart_col, chat_col = st.columns([1.3, 0.7])

    with chart_col:
        st.subheader("Telemetry Traces")
        
        df_plot = df.iloc[::10, :] if len(df) > 5000 else df
        
        if "speed" in df_plot.columns:
            fig_speed = px.line(
                df_plot,
                y="speed",
                title="Speed Trace Over Time",
                labels={"index": "Frame", "value": "Speed (m/s)"},
            )
            fig_speed.update_layout(
                template="plotly_dark",
                plot_bgcolor="#1e1e1e",
                paper_bgcolor="#111111",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_speed, use_container_width=True)

        if "accStatus" in df_plot.columns or "brakeStatus" in df_plot.columns:
            input_cols = [c for c in ["accStatus", "brakeStatus"] if c in df_plot.columns]
            if input_cols:
                fig_inputs = px.line(
                    df_plot,
                    y=input_cols,
                    title="Driver Input Trace (Throttle & Brake Overlay)",
                    labels={"index": "Frame", "value": "Input Level (0.0 - 1.0)"},
                )
                fig_inputs.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#1e1e1e",
                    paper_bgcolor="#111111",
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_inputs, use_container_width=True)

        reward_df = get_reward_progression(parquet_files)
        if not reward_df.empty:
            fig_reward = px.line(
                reward_df,
                x="Run",
                y="Total Reward",
                markers=True,
                title="Cumulative Reward Progression Across Training Runs",
            )
            fig_reward.update_layout(
                template="plotly_dark",
                plot_bgcolor="#1e1e1e",
                paper_bgcolor="#111111",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_reward, use_container_width=True)

    with chat_col:
        st.subheader("Automated Engineer Brief")

        summary_file = os.path.basename(selected_file) + ".summary.txt"
        summary_text = ""
        if os.path.exists(summary_file):
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary_text = f.read()
            except Exception:
                pass

        if summary_text:
            st.markdown(
                f"""
                <div class="summary-box">
                    <strong>Run Analysis:</strong><br><br>
                    {summary_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("Generate AI Debrief for Selected Run"):
            with st.spinner("Generating debriefing via Windows GPU..."):
                summary_text = generate_summary_text(selected_file, parquet_files, df, total_events)
                st.rerun()

        st.markdown("---")
        st.subheader("Pit Radio Channel")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_prompt := st.chat_input("Ask engineer about session performance..."):
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            historical_context = get_historical_context(parquet_files, selected_file, max_history=10)
            telemetry_context = query_telemetry_context(df, selected_file, total_events, historical_context)
            system_prompt = f"""
                You are a helpful R&D simulation engineer explaining training progress over the pit radio. 
                Answer in clear, simple, non-technical language that is easy for anyone to understand. 
                Always refer to the AI model as "driver". 
                Rely heavily on the rolling memory history of the previous 10 runs and lap times to explain how the driver is improving, struggling, or repeatedly crashing into walls.
                Always include exact numbers, stats, and figures from the data to support your answers. Use real Monza corner names when discussing track positions.
                
                Context provided:
                {telemetry_context}
                
                Keep responses short, clear, and friendly.
                """

            with st.chat_message("assistant"):
                with st.spinner("Analyzing telemetry across recent runs..."):
                    try:
                        client = ollama.Client(host="http://100.72.210.89:11434")
                        response = client.chat(
                            model="llama3.1:latest",
                            messages=[
                                {"role": "system", "content": system_prompt},
                            ]
                            + [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ],
                        )
                        reply = response["message"]["content"]
                        st.markdown(reply)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": reply}
                        )
                    except Exception as e:
                        error_msg = f"AI offline. (Radio error: {e})"
                        st.markdown(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )


if __name__ == "__main__":
    main()