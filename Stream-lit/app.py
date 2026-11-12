"""
Understanding Anomalies in Chemical Processes

Run using:
streamlit run app.py
"""

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yaml


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Chemical Process Anomaly Explorer",
    page_icon="🔬",
    layout="wide"
)

st.html("""
<style>
[data-testid="stMainBlockContainer"] {
    padding-top: 2rem !important;
    padding-bottom: 1rem !important;
}
</style>
""")


# ==================================================
# DATA DIRECTORY
# ==================================================

DATA_DIR = Path(__file__).parent / "batch_distillation_dataset"

OPERATING_POINTS = [
    "OP10",
    "OP17",
    "OP18",
    "OP22",
    "OP29",
    "OP31"
]

NORMAL_COLOR = "#0072B2"
ANOMALOUS_COLOR = "#D55E00"
HIGHLIGHT_COLOR = "rgba(255, 0, 0, 0.15)"


# ==================================================
# LOAD MEASUREMENT DATA
# ==================================================

@st.cache_data
def load_measurements(op, experiment):

    folder = DATA_DIR / op / "measurements"

    if experiment == "Normal":
        files = sorted(folder.glob("train_normal*.csv"))
    else:
        files = sorted(folder.glob("test_anormal*.csv"))

    if not files:
        return pd.DataFrame()

    dfs = []

    for file in files:

        df = pd.read_csv(file)

        df["Experiment"] = file.name

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ==================================================
# LOAD ANOMALY ANNOTATIONS
# ==================================================

@st.cache_data
def load_annotations(op):

    folder = DATA_DIR / op / "annotations"

    anomalies = []

    for file in sorted(folder.glob("*.yaml")):

        with open(file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            continue

        for anomaly in data.get("anomalies", []):

            mode = anomaly.get("PerturbationMode", {}) or {}

            local_effect = (
                (mode.get("hasLocalEffect") or {})
                .get("LocalEffect") or {}
            )

            sensor = (
                (local_effect.get("madeBySensor") or {})
                .get("Sensor") or {}
            ).get("id", "Unknown")

            anomalies.append({
                "ID": anomaly.get("id"),
                "Start": anomaly.get("hasBeginning"),
                "End": anomaly.get("hasEnd"),
                "Category": anomaly.get("hasCategory"),
                "Perturbation": anomaly.get("Perturbation"),
                "Affected Sensor": sensor,
                "Local Effect": local_effect.get("label"),
                "Recovery Action": anomaly.get("hasRecoveryAction"),
                "Source File": file.name
            })

    return pd.DataFrame(anomalies)


# ==================================================
# TIME PROCESSING
# ==================================================

def prepare_time(df):

    df = df.copy()

    # Convert sensor values to numeric values
    sensor_columns = [
        col for col in df.columns
        if col not in ["Time", "Experiment"]
    ]

    for col in sensor_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert time strings into datetime values
    df["Time"] = pd.to_datetime(
        df["Time"].astype(str),
        errors="coerce"
    )

    # Sort recordings separately
    # and calculate elapsed time for comparison

    processed = []

    for experiment_file, group in df.groupby(
        "Experiment", sort=False
    ):

        group = group.copy()

        # Use the original recording order
        group["Elapsed Minutes"] = (
            group["Time"] - group["Time"].iloc[0]
        ).dt.total_seconds() / 60

        # Correct for recordings crossing midnight
        elapsed = group["Elapsed Minutes"].copy()

        if elapsed.notna().all():
            seconds = (
                group["Time"].dt.hour * 3600
                + group["Time"].dt.minute * 60
                + group["Time"].dt.second
            )

            rollover = (
                seconds.diff().fillna(0) < -43200
            ).cumsum()

            elapsed = (
                seconds
                + rollover * 86400
                - seconds.iloc[0]
            ) / 60

        group["Elapsed Minutes"] = elapsed

        processed.append(group)

    return pd.concat(processed, ignore_index=True)


def time_to_seconds(value):

    if value is None:
        return None

    try:
        parsed = pd.to_datetime(str(value))

        return (
            parsed.hour * 3600
            + parsed.minute * 60
            + parsed.second
        )

    except (ValueError, TypeError):
        return None


def calculate_duration(start, end):

    start_seconds = time_to_seconds(start)
    end_seconds = time_to_seconds(end)

    if start_seconds is None or end_seconds is None:
        return None

    duration = end_seconds - start_seconds

    if duration < 0:
        duration += 86400

    return duration / 60


# ==================================================
# DASHBOARD HEADER
# ==================================================

st.title("🔬 Understanding Anomalies in Chemical Processes")

st.caption(
    "Interactive visual analysis of normal and anomalous "
    "operating conditions in a batch distillation process."
)


# ==================================================
# SIDEBAR CONTROLS
# ==================================================

st.sidebar.header("Dashboard Controls")

selected_op = st.sidebar.selectbox(
    "Select Operating Point",
    OPERATING_POINTS
)

experiment = st.sidebar.radio(
    "Select Experiment",
    ["Anomalous", "Normal"]
)


# ==================================================
# LOAD DATA
# ==================================================

df = load_measurements(selected_op, experiment)

normal_df = load_measurements(selected_op, "Normal")

anomalous_df = load_measurements(selected_op, "Anomalous")

annotations = load_annotations(selected_op)

if df.empty:
    st.error("No measurement data found.")
    st.stop()

df = prepare_time(df)

if not normal_df.empty:
    normal_df = prepare_time(normal_df)

if not anomalous_df.empty:
    anomalous_df = prepare_time(anomalous_df)


# ==================================================
# SENSOR SELECTION
# ==================================================

sensor_columns = [
    col for col in df.columns
    if col not in [
        "Time",
        "Experiment",
        "Elapsed Minutes"
    ]
]

default_sensor = "T712" if "T712" in sensor_columns else sensor_columns[0]

selected_sensor = st.sidebar.selectbox(
    "Select Sensor",
    sensor_columns,
    index=sensor_columns.index(default_sensor)
)


# ==================================================
# DATASET OVERVIEW
# ==================================================

st.header("Overview")

st.caption(
    f"{selected_op} · {experiment} experiment · "
    f"{len(df):,} measurements"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Operating Point",
    selected_op
)

col2.metric(
    "Measurements",
    f"{len(df):,}"
)

col3.metric(
    "Sensors",
    len(sensor_columns)
)

col4.metric(
    "Recorded Anomalies",
    len(annotations)
)


# ==================================================
# CHART 1: SENSOR BEHAVIOR
# ==================================================

fig1 = go.Figure()

for experiment_file, group in df.groupby(
    "Experiment", sort=False
):

    fig1.add_trace(
        go.Scatter(
            x=group["Elapsed Minutes"],
            y=group[selected_sensor],
            mode="lines",
            name=experiment_file,
            line=dict(
                color=(
                    ANOMALOUS_COLOR
                    if experiment == "Anomalous"
                    else NORMAL_COLOR
                )
            )
        )
    )


# Highlight anomalies using elapsed time
if experiment == "Anomalous" and not annotations.empty:

    for _, anomaly in annotations.iterrows():

        measurement_file = str(
            anomaly["Source File"]
        ).replace(".yaml", ".csv")

        matching_data = df[
            df["Experiment"] == measurement_file
        ]

        if matching_data.empty:
            continue

        start_seconds = time_to_seconds(anomaly["Start"])
        end_seconds = time_to_seconds(anomaly["End"])

        if start_seconds is None or end_seconds is None:
            continue

        first_time = matching_data["Time"].iloc[0]

        first_seconds = (
            first_time.hour * 3600
            + first_time.minute * 60
            + first_time.second
        )

        start_minutes = (
            start_seconds - first_seconds
        ) / 60

        end_minutes = (
            end_seconds - first_seconds
        ) / 60

        if start_minutes < 0:
            start_minutes += 1440

        if end_minutes < start_minutes:
            end_minutes += 1440

        fig1.add_vrect(
            x0=start_minutes,
            x1=end_minutes,
            fillcolor="red",
            opacity=0.15,
            line_width=0,
            annotation_text="Anomaly",
            annotation_position="top left"
        )

fig1.update_layout(
    title="Sensor Behavior Over Time",
    xaxis_title="Elapsed Time (minutes)",
    yaxis_title=selected_sensor,
    hovermode="x unified"
)


# ==================================================
# CHART 2: NORMAL VS ANOMALOUS
# ==================================================

fig2 = go.Figure()

if not normal_df.empty:

    for experiment_file, group in normal_df.groupby(
        "Experiment", sort=False
    ):

        fig2.add_trace(
            go.Scatter(
                x=group["Elapsed Minutes"],
                y=group[selected_sensor],
                mode="lines",
                name="Normal",
                line=dict(color=NORMAL_COLOR),
                legendgroup="Normal",
                showlegend=(
                    experiment_file
                    == normal_df["Experiment"].iloc[0]
                )
            )
        )

if not anomalous_df.empty:

    for experiment_file, group in anomalous_df.groupby(
        "Experiment", sort=False
    ):

        fig2.add_trace(
            go.Scatter(
                x=group["Elapsed Minutes"],
                y=group[selected_sensor],
                mode="lines",
                name="Anomalous",
                line=dict(color=ANOMALOUS_COLOR),
                legendgroup="Anomalous",
                showlegend=(
                    experiment_file
                    == anomalous_df["Experiment"].iloc[0]
                )
            )
        )

fig2.update_layout(
    title="Normal vs Anomalous Behavior",
    xaxis_title="Elapsed Time (minutes)",
    yaxis_title=selected_sensor,
    hovermode="x unified"
)


# ==================================================
# CHART 3: ANOMALY DURATION
# ==================================================

fig3 = go.Figure()

if not annotations.empty:

    duration_data = annotations.copy()

    duration_data["Duration"] = duration_data.apply(
        lambda row: calculate_duration(
            row["Start"],
            row["End"]
        ),
        axis=1
    )

    duration_data["Anomaly"] = (
        duration_data["ID"].astype(str)
    )

    fig3 = px.bar(
        duration_data,
        x="Anomaly",
        y="Duration",
        color="Category",
        title="Recorded Anomaly Durations",
        labels={
            "Duration": "Duration (minutes)",
            "Anomaly": "Anomaly ID"
        },
        hover_data=[
            "Perturbation",
            "Affected Sensor"
        ]
    )

else:

    fig3.update_layout(
        title="Recorded Anomaly Durations"
    )


# ==================================================
# CHART STYLING
# ==================================================

for chart in [fig1, fig2, fig3]:

    chart.update_layout(
        height=340,
        margin=dict(
            l=10,
            r=10,
            t=70,
            b=50
        ),
        legend=dict(
            orientation="h",
            y=1.15,
            x=0
        ),
        legend_title_text=""
    )


# ==================================================
# DISPLAY THREE CHARTS SIDE BY SIDE
# ==================================================

left, middle, right = st.columns(3)

with left:
    st.plotly_chart(fig1, use_container_width=True)

with middle:
    st.plotly_chart(fig2, use_container_width=True)

with right:
    st.plotly_chart(fig3, use_container_width=True)


# ==================================================
# ANOMALY DETAILS
# ==================================================

with st.expander("🚨 View Recorded Anomaly Details"):

    if annotations.empty:

        st.info("No anomaly annotations available.")

    else:

        st.dataframe(
            annotations,
            use_container_width=True,
            hide_index=True
        )


# ==================================================
# RAW DATA
# ==================================================

with st.expander("📁 View Raw Sensor Data"):

    st.dataframe(
        df,
        use_container_width=True
    )