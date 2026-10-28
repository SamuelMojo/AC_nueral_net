"""Assetto Corsa telemetry dashboard (v2).

What changed vs dashboard.py
----------------------------
* The pit radio chat is rendered FIRST and never waits on S3 - previously it sat
  underneath get_reward_progression(), which downloaded every parquet file in the
  bucket before the chat column was ever drawn.
* Only the columns the dashboard needs are read from each parquet file, and every
  read is cached. Finished runs are never downloaded twice.
* Listing the bucket is one S3 call (fs.find(detail=True)) instead of one call per file.
* The live panel auto-refreshes on its own timer (st.fragment) without re-running
  the whole page or touching the chat.
* Ollama calls have a timeout, stream tokens as they arrive, and keep the model
  loaded between questions. An online/offline indicator shows if the AI box is unreachable.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import ollama
import pandas as pd
import plotly.express as px
import pyarrow.parquet as pq
import s3fs
import streamlit as st

st.set_page_config(page_title="Assetto Corsa Telemetry Dashboard", page_icon=None, layout="wide")

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e1e1e; border: 1px solid #2d2d2d; border-radius: 4px;
        padding: 12px; text-align: center; aspect-ratio: 1 / 1;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #ffffff; line-height: 1.2; }
    .metric-label { font-size: 0.8rem; font-weight: 500; color: #888888; margin-top: 8px;
                    text-transform: uppercase; letter-spacing: 1px; }
    .summary-box { background-color: #1e1e1e; border: 1px solid #2d2d2d; border-radius: 4px;
                   padding: 15px; margin-bottom: 20px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- config
S3_BUCKET_PATH = os.environ.get("AC_S3_PATH", "samuelojo-ac-telemetry/outputs")
OLLAMA_URL = os.environ.get("AC_OLLAMA_URL", "http://100.72.210.89:11434")
OLLAMA_MODEL = os.environ.get("AC_OLLAMA_MODEL", "llama3.1:latest")
OLLAMA_TIMEOUT_S = 120  # first token can be slow if the model has to be loaded onto the GPU
LIVE_THRESHOLD_S = 120
MAX_PLOT_POINTS = 2000
LATEST = "__latest__"

LAP_COLS = ["lap_time", "lap_duration", "lapTime"]
NEEDED_COLS = ["speed", "accelY", "out_of_track", "progress", "accStatus", "brakeStatus", "reward"] + LAP_COLS

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


# --------------------------------------------------------------------------- S3 access
@st.cache_resource
def get_fs():
    return s3fs.S3FileSystem(anon=False)


@st.cache_data(ttl=10, show_spinner=False)
def list_runs():
    """One S3 listing call. Returns (runs newest-first, error string or '')."""
    fs = get_fs()
    fs.invalidate_cache()  # s3fs caches listings per process; without this new files appear late
    try:
        found = fs.find(S3_BUCKET_PATH, detail=True)
    except Exception as e:  # noqa: BLE001
        return [], str(e)
    runs = []
    for path, info in found.items():
        modified = info.get("LastModified")
        if path.endswith(".parquet") and modified is not None:
            runs.append({
                "path": path,
                "mtime": modified.timestamp(),
                "name": os.path.basename(path),
                "session": session_of(path),
            })
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs, ""


CURRENT = "__current__"  # newest training session (resets every time train.py is restarted)
ALL = "__all__"


def session_of(path):
    """outputs/<session folder>/laps/x.parquet -> '<session folder>' (one per train.py start)."""
    rel = path[len(S3_BUCKET_PATH):].lstrip("/") if path.startswith(S3_BUCKET_PATH) else path
    parts = rel.split("/")
    return parts[0] if len(parts) > 1 else "(root)"


def scoped_runs(scope):
    """Runs (newest first) limited to one training session, or all of them."""
    runs, err = list_runs()
    if err or not runs or scope == ALL:
        return runs, err
    session = runs[0]["session"] if scope == CURRENT else scope
    return [r for r in runs if r["session"] == session], err


def read_cols(path, wanted):
    """Read only the requested columns that exist in the file."""
    with get_fs().open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        names = set(pf.schema_arrow.names)
        cols = [c for c in wanted if c in names]
        return pf.read(columns=cols).to_pandas()


@st.cache_data(ttl=300, show_spinner=False, max_entries=64)
def load_run(path, mtime):
    # mtime is part of the cache key: a run that is still being uploaded gets re-read, finished runs don't.
    return read_cols(path, NEEDED_COLS)


def lap_time_of(df):
    for col in LAP_COLS:
        if col in df.columns and not df[col].empty:
            val = df[col].iloc[-1]
            if pd.notna(val) and val > 0:
                return float(val)
    return None


def _compute_stats(run):
    try:
        df = read_cols(run["path"], ["speed", "accelY", "out_of_track", "reward"] + LAP_COLS)
        return {
            "name": run["name"],
            "lap_time": lap_time_of(df),
            "peak": float(df["speed"].max()) if "speed" in df else 0.0,
            "mean": float(df["speed"].mean()) if "speed" in df else 0.0,
            "max_g": float(df["accelY"].abs().max()) if "accelY" in df else 0.0,
            "off": bool(df["out_of_track"].any()) if "out_of_track" in df else False,
            "reward": float(df["reward"].sum()) if "reward" in df else float(len(df)),
        }
    except Exception:  # noqa: BLE001
        return None


@st.cache_resource
def _stats_store():
    return {}


def get_all_stats(runs):
    """Per-run summary stats. Each (file, mtime) is computed once, new files in parallel."""
    store = _stats_store()
    todo = [r for r in runs if (r["path"], r["mtime"]) not in store]
    if todo:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for run, stats in zip(todo, pool.map(_compute_stats, todo)):
                store[(run["path"], run["mtime"])] = stats
    return [store.get((r["path"], r["mtime"])) for r in runs]


def resolve(runs, key):
    if not runs:
        return None
    if key == LATEST:
        return runs[0]
    return next((r for r in runs if r["path"] == key), runs[0])


def history_text(runs, stats, current, max_history=10):
    lines = []
    for run, s in zip(runs, stats):
        if run["path"] == current["path"] or s is None:
            continue
        lap = f"{s['lap_time']:.3f}s" if s["lap_time"] else "N/A"
        lines.append(
            f"- Run {s['name']} | Lap Time: {lap} | Peak Speed: {s['peak']:.1f} m/s | "
            f"Mean Speed: {s['mean']:.1f} m/s | Max G: {s['max_g']:.2f} G | Off-track: {s['off']}"
        )
        if len(lines) >= max_history:
            break
    return "\n".join(lines) if lines else "This is the first recorded training run."


def build_context(selected_key, scope):
    """Text block handed to the LLM. Only called when the user asks something / clicks debrief."""
    runs, _ = scoped_runs(scope)
    run = resolve(runs, selected_key)
    if run is None:
        return "No telemetry data available yet."
    df = load_run(run["path"], run["mtime"])
    history = history_text(runs, get_all_stats(runs), run)

    lap = lap_time_of(df)
    lap_str = f"{lap:.3f} seconds" if lap else "Not Completed / In Progress"
    off_track = bool(df["out_of_track"].any()) if "out_of_track" in df else False
    corner = "None"
    if off_track and "progress" in df:
        off_frames = df[df["out_of_track"] == True]  # noqa: E712
        if not off_frames.empty:
            modes = off_frames["progress"].apply(identify_monza_corner).mode()
            if not modes.empty:
                corner = str(modes[0])
    speed = df["speed"] if "speed" in df else pd.Series([0.0])
    g = df["accelY"].abs().max() if "accelY" in df else 0.0
    return f"""
    [ACTIVE RUN METRICS]
    - Total Training Runs So Far: {len(runs)}
    - Active Log File: {run['name']}
    - Completed Lap Time: {lap_str}
    - Total Frames: {len(df)}
    - Peak Speed: {speed.max():.1f} m/s (Average Speed: {speed.mean():.1f} m/s)
    - Max Cornering G-Force: {g:.2f} G
    - Went Off Track: {off_track} (Primary Off-Track Corner: {corner})

    [LAST 10 RUNS ROLLING MEMORY]
    {history}
    """


# --------------------------------------------------------------------------- Ollama
def ollama_client(timeout=OLLAMA_TIMEOUT_S):
    return ollama.Client(host=OLLAMA_URL, timeout=timeout)


@st.cache_data(ttl=20, show_spinner=False)
def ollama_status():
    try:
        ollama_client(timeout=3).list()
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def stream_reply(messages):
    for chunk in ollama_client().chat(model=OLLAMA_MODEL, messages=messages, stream=True, keep_alive="30m"):
        yield chunk["message"]["content"]


def generate_summary_text(selected_key, scope, run_name):
    prompt = f"""
        You are an R&D simulation engineer explaining how an AI driving model is learning.
        Provide a simple, clear 2-sentence summary using everyday language.

        {build_context(selected_key, scope)}

        Strict Instructions:
        - NEVER start your response with filler phrases like "Here is a 2-sentence summary", "Sure", or introductory text. Jump immediately into the analysis.
        - Call the agent solely "driver".
        - If a lap time is recorded, include it explicitly.
        - If an off-track excursion or wall crash occurred, mention ONLY the single most frequent or critical Monza corner where it happened based on the mapping.
        - Include specific numbers and figures (such as lap time, peak speed, average speed, or G-forces) directly in the text.
        - Keep it simple so non-technical people can easily understand how performance compares to past runs.
        """
    try:
        resp = ollama_client().chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "system", "content": prompt}],
            keep_alive="30m",
        )
        text = resp["message"]["content"].strip()
        with open(run_name + ".summary.txt", "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except Exception as e:  # noqa: BLE001
        return f"AI Offline Error: {e}"


# --------------------------------------------------------------------------- UI pieces
def card(value, label):
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def style(fig):
    fig.update_layout(
        template="plotly_dark", plot_bgcolor="#1e1e1e", paper_bgcolor="#111111",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def render_chat(selected_key, scope):
    """Right-hand column. Must stay free of any slow S3 work so it always appears instantly."""
    st.subheader("Automated Engineer Brief")
    runs, _ = scoped_runs(scope)
    run = resolve(runs, selected_key)

    if run is not None:
        summary_file = run["name"] + ".summary.txt"
        if os.path.exists(summary_file):
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    st.markdown(
                        f'<div class="summary-box"><strong>Run Analysis:</strong><br><br>{f.read()}</div>',
                        unsafe_allow_html=True,
                    )
            except Exception:  # noqa: BLE001
                pass
        if st.button("Generate AI Debrief for Selected Run", key="gen_debrief_btn"):
            with st.spinner("Generating debrief..."):
                generate_summary_text(selected_key, scope, run["name"])
            st.rerun()

    st.markdown("---")
    st.subheader("Pit Radio Channel")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask engineer about session performance...", key="pit_radio_input")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        system_prompt = f"""
            You are a helpful R&D simulation engineer explaining training progress over the pit radio.
            Answer in clear, simple, non-technical language that is easy for anyone to understand.
            Always refer to the AI model as "driver".
            Rely heavily on the rolling memory history of the previous 10 runs and lap times to explain how the driver is improving, struggling, or repeatedly crashing into walls.
            Always include exact numbers, stats, and figures from the data to support your answers. Use real Monza corner names when discussing track positions.

            Context provided:
            {build_context(selected_key, scope)}

            Keep responses short, clear, and friendly.
            """
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]]
        with st.chat_message("assistant"):
            try:
                reply = st.write_stream(stream_reply([{"role": "system", "content": system_prompt}] + history))
            except Exception as e:  # noqa: BLE001
                reply = f"AI offline. (Radio error: {e})"
                st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    online, err = ollama_status()
    st.caption("Engineer online" if online else f"Engineer offline - cannot reach {OLLAMA_URL} ({err})")


def render_live(selected_key, scope):
    """Left-hand column: tiles + charts. Re-runs on its own timer, independent of the chat."""
    runs, err = scoped_runs(scope)
    if err:
        st.error(f"S3 Connection Error: {err}")
        return
    run = resolve(runs, selected_key)
    if run is None:
        st.info("No telemetry logs found in the S3 bucket.")
        return

    age = pd.Timestamp.now(tz="UTC").timestamp() - list_runs()[0][0]["mtime"]  # newest file overall
    if age > LIVE_THRESHOLD_S:
        st.warning("No live simulation right now - viewing historical data and manual debriefs.")

    try:
        df = load_run(run["path"], run["mtime"])
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to load selected telemetry file: {e}")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card(len(runs), "Total Events")
    with c2:
        card(f"{df['speed'].max():.1f}" if "speed" in df else "0.0", "Peak Speed (m/s)")
    with c3:
        card(f"{df['accelY'].abs().max():.2f}G" if "accelY" in df else "0.00G", "Max Lateral G")
    with c4:
        card(len(df), "Run Frames")

    st.subheader("Telemetry Traces")
    step = max(1, len(df) // MAX_PLOT_POINTS)
    plot = df.iloc[::step]

    if "speed" in plot:
        st.plotly_chart(
            style(px.line(plot, y="speed", title="Speed Trace Over Time", labels={"index": "Frame", "value": "Speed (m/s)"})),
            use_container_width=True,
        )
    inputs = [c for c in ["accStatus", "brakeStatus"] if c in plot]
    if inputs:
        st.plotly_chart(
            style(px.line(plot, y=inputs, title="Driver Input Trace (Throttle & Brake Overlay)",
                          labels={"index": "Frame", "value": "Input Level (0.0 - 1.0)"})),
            use_container_width=True,
        )

    oldest_first = list(reversed(list(zip(runs, get_all_stats(runs)))))
    rows = [{"Run": f"Run {i + 1}", "Total Reward": s["reward"]} for i, (_, s) in enumerate(oldest_first) if s]
    if rows:
        st.plotly_chart(
            style(px.line(pd.DataFrame(rows), x="Run", y="Total Reward", markers=True,
                          title="Cumulative Reward Progression Across Training Runs")),
            use_container_width=True,
        )


# --------------------------------------------------------------------------- main
def main():
    st.title("Assetto Corsa Telemetry Dashboard")

    auto = st.sidebar.toggle("Auto-refresh live data", value=True)
    interval = st.sidebar.select_slider("Refresh every (seconds)", options=[5, 10, 15, 30, 60], value=10)
    if st.sidebar.button("Refresh now"):
        list_runs.clear()
        st.rerun()

    all_runs, _ = list_runs()
    sessions = list(dict.fromkeys(r["session"] for r in all_runs))  # newest first
    live = bool(all_runs) and (pd.Timestamp.now(tz="UTC").timestamp() - all_runs[0]["mtime"]) <= LIVE_THRESHOLD_S
    # "Current session" only exists while the simulation is actually uploading laps.
    choices = ([CURRENT] if live else []) + [ALL] + sessions
    default = 0 if live else (1 if sessions else 0)  # offline: open on the most recent finished session

    def label(s):
        if s == CURRENT:
            return "Current session (live, resets each restart)"
        if s == ALL:
            return "All sessions"
        return f"{s} (last session)" if (not live and s == sessions[0]) else s

    scope = st.sidebar.selectbox("Training session", choices, index=default, format_func=label)
    selected_key = LATEST  # always the newest run of the chosen session; no second file dropdown
    st.markdown("---")

    chart_col, chat_col = st.columns([1.3, 0.7])

    # Chat first: containers are laid out by position, not by execution order, so this appears
    # instantly in the right column while the left column is still loading.
    with chat_col:
        render_chat(selected_key, scope)

    with chart_col:
        if hasattr(st, "fragment"):  # Streamlit >= 1.37
            live_panel = st.fragment(run_every=f"{interval}s" if auto else None)(render_live)
        else:
            live_panel = render_live
        live_panel(selected_key, scope)


if __name__ == "__main__":
    main()
