from pathlib import Path
import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

st.set_page_config(page_title="DistillSense", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")
DATA_DIR = Path(__file__).parent / "batch_distillation_dataset"
OPS = ["OP10", "OP17", "OP18", "OP22", "OP29", "OP31"]
BLUE, RED, INK, MUTED = "#299bd6", "#f43f65", "#171c45", "#64748b"
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*=css],.stApp {font-family:'Inter',sans-serif}
.stApp {background:#f7f8fe;color:#171c45}
.block-container {max-width:1570px;padding-top:1.25rem;padding-bottom:2rem}
[data-testid="stSidebar"] {background:linear-gradient(160deg,#151a43 0%,#22205b 65%,#43209b 100%);border:0}
[data-testid="stSidebar"] * {color:#f5f4ff!important}
[data-testid="stSidebar"] .block-container {padding-top:2rem}
[data-testid="stSidebar"] [data-testid="stRadio"] label {font-size:1rem}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:9px 8px;border-radius:9px}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:rgba(148,111,255,.35)}
/* Keep dark sidebar labels, but make the select input readable. */
[data-testid="stSidebar"] [data-baseweb="select"] {background:#fff!important;border-radius:9px!important}
[data-testid="stSidebar"] [data-baseweb="select"] * {color:#172554!important;-webkit-text-fill-color:#172554!important}
[data-testid="stSidebar"] [data-baseweb="select"] svg {fill:#172554!important;color:#172554!important}
[role="listbox"] {background:#fff!important}
[role="option"], [role="option"] * {color:#172554!important;-webkit-text-fill-color:#172554!important}
/* Streamlit versions with baseweb combobox input */
[data-testid="stSidebar"] [role="combobox"],
[data-testid="stSidebar"] [role="combobox"] * {color:#172554!important;-webkit-text-fill-color:#172554!important}
[data-testid="stSidebar"] hr {border-color:rgba(255,255,255,.16)}
h1,h2,h3 {color:#171c45!important;letter-spacing:-.035em}
.hero {padding:21px 27px;border-radius:18px;background:linear-gradient(110deg,#171c4a,#5030c0);color:white;margin:0 0 17px;box-shadow:0 10px 25px rgba(34,25,103,.13)}
.hero h1 {color:white!important;font-size:2rem;margin:3px 0 6px}
.hero p {color:#e4e4ff;margin:0;font-size:.93rem}
.kicker {color:#9ee8ff;font-weight:800;font-size:.68rem;letter-spacing:.15em}
.note {color:#64748b;font-size:.9rem;margin-bottom:14px}
[data-testid="stMetric"] {background:white;border:1px solid #e4e5f2;border-radius:15px;padding:12px 16px;box-shadow:0 5px 18px rgba(31,29,88,.055);min-height:110px}
[data-testid="stMetricLabel"] {color:#65718e}
[data-testid="stMetricValue"] {color:#161d48;font-weight:800}
div.stButton>button {border-radius:10px;min-height:40px;font-weight:650;border:1px solid #d9dcf1}
div.stButton>button[kind="primary"] {background:#6b34d6;border-color:#6b34d6}
[data-testid="stPlotlyChart"] {background:white;border:1px solid #e3e4f3;border-radius:16px;overflow:hidden;box-shadow:0 5px 18px rgba(31,29,88,.045)}
.callout {padding:13px 17px;border:1px solid #e0dcff;border-left:4px solid #7b43e8;background:#f2efff;border-radius:12px;margin:8px 0 14px}
.step {background:white;border:1px solid #e3e4f3;padding:16px 18px;border-radius:14px;margin-bottom:12px}
.smallcap {color:#6557a5;font-size:.82rem;letter-spacing:.04em;text-transform:uppercase;font-weight:800}
.panel-title {font-weight:800;font-size:1.13rem;color:#171c45;margin:2px 0 3px}
.panel-caption {font-size:.84rem;color:#65718e;margin-bottom:10px}
</style>""",unsafe_allow_html=True)
@st.cache_data(show_spinner=False)
def measurements(op, kind):
    folder = DATA_DIR / op / "measurements"
    pattern = "train_normal*.csv" if kind == "Normal" else "test_anormal*.csv"
    frames = []
    for file in sorted(folder.glob(pattern)):
        frame = pd.read_csv(file)
        if "Time" not in frame.columns:
            continue
        frame["Time"] = pd.to_datetime(frame["Time"], errors="coerce")
        frame = frame.dropna(subset=["Time"]).copy()
        if frame.empty:
            continue
        frame["Experiment"] = file.name
        # Elapsed time based on recording order, including midnight rollovers.
        clock = frame["Time"].dt.hour * 3600 + frame["Time"].dt.minute * 60 + frame["Time"].dt.second
        rollovers = (clock.diff().fillna(0) < -43200).cumsum()
        frame["Elapsed Minutes"] = (clock + 86400 * rollovers - clock.iloc[0]) / 60
        for col in frame.columns.difference(["Time", "Experiment", "Elapsed Minutes"]):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

@st.cache_data(show_spinner=False)
def annotations_for(op):
    rows = []
    for file in sorted((DATA_DIR / op / "annotations").glob("*.yaml")):
        with file.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for item in data.get("anomalies", []):
            mode = item.get("PerturbationMode") or {}
            effect = ((mode.get("hasLocalEffect") or {}).get("LocalEffect") or {})
            sensor = ((effect.get("madeBySensor") or {}).get("Sensor") or {}).get("id")
            rows.append({"ID": item.get("id", "Unknown"), "Start": item.get("hasBeginning"),
                         "End": item.get("hasEnd"), "Category": item.get("hasCategory", ""),
                         "Perturbation": item.get("Perturbation", ""), "Affected Sensor": sensor,
                         "Local Effect": effect.get("label", ""),
                         "Recovery Action": item.get("hasRecoveryAction", ""), "Source File": file.name})
    return pd.DataFrame(rows)

def sensors_of(frame):
    return [c for c in frame.columns if c not in ("Time", "Experiment", "Elapsed Minutes") and pd.api.types.is_numeric_dtype(frame[c])]

def clock_seconds(value):
    try:
        stamp = pd.to_datetime(str(value))
        return stamp.hour * 3600 + stamp.minute * 60 + stamp.second
    except (TypeError, ValueError):
        return None

def event_window(row, frame):
    """Convert YAML wall-clock event times into elapsed minutes for its matching CSV."""
    if frame.empty:
        return None
    source = str(row.get("Source File", "")).replace(".yaml", ".csv")
    match = frame.loc[frame["Experiment"] == source]
    if match.empty:
        return None
    a, b = clock_seconds(row.get("Start")), clock_seconds(row.get("End"))
    if a is None or b is None:
        return None
    first = match["Time"].iloc[0]
    origin = first.hour * 3600 + first.minute * 60 + first.second
    start = (a - origin) / 60
    if start < 0:
        start += 1440
    end = (b - origin) / 60
    while end < start:
        end += 1440
    return start, end, source

def robust_scale(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 1.0
    q1, q3 = s.quantile([.25, .75])
    return max(float((q3 - q1) / 1.349), float(s.std()) * .15, 1e-5)

def response_profile(normal, anomalous, sensors, event):
    """Compare within-run change around the documented event, not raw run levels."""
    window = event_window(event, anomalous)
    if window is None:
        return pd.DataFrame()
    start, end, _ = window
    rows = []
    for sensor in sensors:
        item = {"Sensor": sensor}
        for label, frame in [("Normal", normal), ("Anomalous", anomalous)]:
            if frame.empty or sensor not in frame:
                item[label] = np.nan
                continue
            pre = frame.loc[(frame["Elapsed Minutes"] >= max(0, start-10)) & (frame["Elapsed Minutes"] < start), sensor].dropna()
            during = frame.loc[(frame["Elapsed Minutes"] >= start) & (frame["Elapsed Minutes"] <= end), sensor].dropna()
            if len(pre) < 5 or len(during) < 5:
                item[label] = np.nan
                continue
            # Change in event-period median relative to the 10-minute pre-event median.
            # Scaling uses pre-event variability, with a floor to prevent division by zero.
            baseline = float(pre.median())
            change = abs(float(during.median()) - baseline)
            scale = max(robust_scale(pre), abs(baseline)*0.001, 0.001)
            item[label] = min(change/scale, 25.0)
            item[label+" shift"] = float(during.median()) - baseline
        rows.append(item)
    return pd.DataFrame(rows).sort_values("Anomalous", ascending=True, na_position="first")

def response_constellation(profile, annotated_sensor=None):
    """Quadrant scatter: compare event-period shifts within each recording."""
    data = profile.dropna(subset=["Normal", "Anomalous"]).copy()
    if data.empty:
        return None
    data["Difference"] = data["Anomalous"] - data["Normal"]
    data["Group"] = np.where(data["Difference"] > 0.7, "Stronger in anomalous", np.where(data["Difference"] < -0.7, "Stronger in normal", "Similar response"))
    limit = max(2.0, float(data[["Normal", "Anomalous"]].max().max()) * 1.13)
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=limit, y1=limit, fillcolor="#fbfdff", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=[0, limit], y=[0, limit], mode="lines", line=dict(color="#94a3b8", dash="dash", width=1.5), name="Equal response", hoverinfo="skip"))
    palette = {"Stronger in anomalous":"#ed6670", "Stronger in normal":"#4b9cc9", "Similar response":"#9aa9bb"}
    for group, color in palette.items():
        subset = data[data["Group"] == group]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["Normal"], y=subset["Anomalous"], mode="markers+text", name=group,
            text=subset["Sensor"], textposition="top center", textfont=dict(size=11,color="#233852"),
            marker=dict(size=[21 if str(x)==str(annotated_sensor) else 13 for x in subset["Sensor"]],
                        color=color, opacity=.9, line=dict(color="white", width=2)),
            customdata=np.column_stack([subset["Sensor"],subset["Normal shift"],subset["Anomalous shift"]]),
            hovertemplate="<b>%{customdata[0]}</b><br>Normal response: %{x:.2f}<br>Anomalous response: %{y:.2f}<br>Normal median shift: %{customdata[1]:.3g}<br>Anomalous median shift: %{customdata[2]:.3g}<extra></extra>"))
    fig.add_annotation(x=limit*.22,y=limit*.92,text="STRONGER DURING ANOMALOUS RUN ↑",showarrow=False,font=dict(size=12,color="#c24d60"))
    fig.add_annotation(x=limit*.78,y=limit*.08,text="STRONGER DURING NORMAL RUN →",showarrow=False,font=dict(size=12,color="#377ba4"))
    fig.update_layout(height=590, margin=dict(l=20,r=20,t=50,b=70),
        xaxis=dict(title="Normal-run response · relative to pre-event variation",range=[-.05*limit,limit],gridcolor="#edf1f6",zeroline=False),
        yaxis=dict(title="Anomalous-run response · relative to pre-event variation",range=[-.05*limit,limit],gridcolor="#edf1f6",zeroline=False,scaleanchor="x",scaleratio=1),
        paper_bgcolor="white",plot_bgcolor="white",font=dict(color=INK),
        legend=dict(orientation="h",y=1.13,x=0),hovermode="closest")
    return fig

def impact_ranking(profile, annotated_sensor=None):
    """Readable ranked lollipop chart; values are observed, not simulated."""
    d=profile.dropna(subset=["Normal","Anomalous"]).copy()
    if d.empty:
        return None, d
    d["Excess"] = d["Anomalous"]-d["Normal"]
    d["Impact"] = d["Excess"].clip(lower=0)
    d=d.sort_values(["Impact","Anomalous"],ascending=[False,False])
    # Show every sensor, with a fixed dedicated row per sensor.
    fig=go.Figure()
    labels=d["Sensor"].astype(str).tolist()
    n=len(labels)
    maxval=max(1.0,float(d["Impact"].max())*1.15)
    for idx,(_,r) in enumerate(d.iterrows()):
        value=float(r["Impact"])
        col="#f43f65" if idx<3 and value>0 else ("#9252ed" if value>0 else "#b7b6e7")
        fig.add_trace(go.Scatter(x=[0,value],y=[idx,idx],mode="lines",line=dict(color=col,width=4),opacity=.55,showlegend=False,hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[value],y=[idx],mode="markers+text",marker=dict(color=col,size=12 if idx<3 else 9,line=dict(color="white",width=1)),text=[f" {value:.1f}"],textposition="middle right",textfont=dict(size=11,color="#171c45"),showlegend=False,
            customdata=[[r["Sensor"],r["Normal"],r["Anomalous"],r["Excess"]]],hovertemplate="<b>%{customdata[0]}</b><br>Normal response: %{customdata[1]:.2f}<br>Anomalous response: %{customdata[2]:.2f}<br>Excess response: %{customdata[3]:+.2f}<extra></extra>"))
    fig.update_layout(height=max(405,n*25+78),margin=dict(l=66,r=54,t=20,b=50),paper_bgcolor="white",plot_bgcolor="white",font=dict(family="Inter",color="#171c45"),
        xaxis=dict(title="Excess anomalous response (scaled units)",range=[0,maxval],showgrid=True,gridcolor="#edf0fa",zeroline=False),
        yaxis=dict(tickmode="array",tickvals=list(range(n)),ticktext=labels,range=[n-.5,-.5],showgrid=False,zeroline=False),hovermode="closest")
    return fig,d

def detail_figure(normal, anomalous, sensor, events, kind):
    fig = go.Figure()
    series = [("Normal", normal, BLUE), ("Anomalous", anomalous, RED)]
    if kind != "Both":
        series = [s for s in series if s[0] == kind]
    for label, frame, color in series:
        if frame.empty or sensor not in frame:
            continue
        for idx, (_, group) in enumerate(frame.groupby("Experiment", sort=False)):
            fig.add_trace(go.Scatter(x=group["Elapsed Minutes"], y=group[sensor], mode="lines",
                name=label, legendgroup=label, showlegend=idx == 0,
                line=dict(color=color, width=2.2), opacity=.92))
    if kind in ("Anomalous", "Both") and not events.empty:
        for _, row in events.iterrows():
            window = event_window(row, anomalous)
            if window:
                a, b, _ = window
                fig.add_vrect(x0=a,x1=b,fillcolor="rgba(228,75,89,.13)",line_width=0)
    fig.update_layout(height=410, margin=dict(l=12,r=18,t=22,b=45), hovermode="x unified",
        xaxis_title="Elapsed time (minutes)", yaxis_title=f"{sensor} reading",
        paper_bgcolor="white",plot_bgcolor="white", font=dict(color=INK),
        legend=dict(orientation="h",y=1.12,x=0), xaxis=dict(showgrid=True,gridcolor="#eef2f7"),
        yaxis=dict(showgrid=True,gridcolor="#eef2f7"))
    return fig


# ---------- Navigation ----------
if not DATA_DIR.is_dir():
    st.error("Dataset folder missing. Put `batch_distillation_dataset` beside `app_supervisor.py`.")
    st.stop()

@st.cache_data(show_spinner=False)
def overview_counts():
    return {op: len(annotations_for(op)) for op in OPS}

counts = overview_counts()
if "selected_op" not in st.session_state:
    st.session_state.selected_op = "OP10"
if "page" not in st.session_state:
    st.session_state.page = "Explore an anomaly"

if "pending_op" in st.session_state:
    st.session_state.op_selector = st.session_state.pop("pending_op")
    st.session_state.explore_op = st.session_state.op_selector

with st.sidebar:
    st.markdown("## ⚗️ DistillSense")
    st.divider()
    page = st.radio("NAVIGATION", ["Overview", "Explore an anomaly", "Process & dataset", "Reading guide"], key="page", label_visibility="visible")
    st.divider()
    st.markdown("**Operating point**")
    op = st.selectbox("Operating point", OPS, index=OPS.index(st.session_state.selected_op), label_visibility="collapsed", key="op_selector")
    if st.session_state.selected_op != op:
        st.session_state.selected_op = op
        # Update the main-panel selector before it is rendered.
        st.session_state.explore_op = op
    st.caption(f"{counts[op]} documented event(s) · choose an operating point to investigate")
    if page == "Explore an anomaly":
        st.info("Choose an event and its sensor in the main panel.", icon="🔎")
    st.divider()
    st.caption("Normal and anomalous recordings are separate experiments. Differences are exploratory, not proof of a fault.")

# ---------- Animated, conceptual batch-distillation diagram ----------
PROCESS_SVG = r"""
<!DOCTYPE html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;font-family:Inter,Arial,sans-serif;color:#172554;background:transparent}
.shell{border:1px solid #dfe5ff;border-radius:22px;background:radial-gradient(ellipse at 70% 10%,#edf0ff,#fff 60%);padding:14px 20px 12px;overflow:hidden}
svg{width:100%;height:auto;display:block}.pipe{fill:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round}.flow{fill:none;stroke-width:4;stroke-linecap:round;stroke-dasharray:5 19;animation:move 1.25s linear infinite}.reverse{animation-direction:reverse}
@keyframes move{to{stroke-dashoffset:-48}}@keyframes bubble{0%{transform:translateY(0);opacity:.1}30%{opacity:.9}100%{transform:translateY(-95px);opacity:0}}@keyframes glow{50%{opacity:.45}}.bubble{animation:bubble 3s ease-in infinite;transform-box:fill-box}.bubble:nth-child(even){animation-delay:1.4s}.heat{animation:glow 1.8s ease-in-out infinite}.tag{font-weight:800;font-size:19px}.small{font-size:15px}.call{fill:#fff;stroke:#dbe5ff;stroke-width:2}.foot{display:flex;align-items:center;justify-content:center;gap:12px;flex-wrap:wrap;background:#f7f6ff;border:1px solid #e1ddff;border-radius:16px;padding:12px;margin:0 7px 3px;font-size:14px;font-weight:700}.pill{border-radius:30px;padding:9px 13px;background:white;border:1px solid #e0ddfa}.arrow{color:#6541d7;font-size:22px}.hint{text-align:right;color:#71809b;font-size:11px;padding:7px 8px 0}@media(prefers-reduced-motion:reduce){*{animation:none!important}}
</style></head><body><div class="shell">
<svg viewBox="0 0 1200 590" role="img" aria-label="Animated conceptual batch distillation: heat in V001, separate in C001 to C003, condense in HE001 and HE002, collect in V002, and return reflux to column">
<defs>
<linearGradient id="glass" x2="1" y2="1"><stop stop-color="#f1fbff"/><stop offset=".52" stop-color="#b9e9ff"/><stop offset="1" stop-color="#e5f6ff"/></linearGradient>
<linearGradient id="boil" x2="0" y2="1"><stop stop-color="#ffd18b"/><stop offset="1" stop-color="#ff7a22"/></linearGradient>
<linearGradient id="cool" x2="0" y2="1"><stop stop-color="#d8f9ff"/><stop offset="1" stop-color="#54caff"/></linearGradient>
<filter id="shadow" x="-30%" y="-30%" width="160%" height="160%"><feDropShadow dx="0" dy="5" stdDeviation="6" flood-color="#6275a5" flood-opacity=".18"/></filter>
</defs>
<!-- pipes behind vessels -->
<path class="pipe" stroke="#ff9b38" d="M337 158 V125 H535 Q552 125 552 145 V165 H607"/>
<path class="flow" stroke="#fff3a5" d="M337 158 V125 H535 Q552 125 552 145 V165 H607"/>
<path class="pipe" stroke="#21a9f4" d="M837 165 H944 Q961 165 961 182 V277"/>
<path class="flow" stroke="#d5f7ff" d="M837 165 H944 Q961 165 961 182 V277"/>
<path class="pipe" stroke="#23b878" d="M959 392 V415 H470 V370 H405"/>
<path class="flow reverse" stroke="#d8ffdb" d="M959 392 V415 H470 V370 H405"/>
<path class="pipe" stroke="#8e45df" d="M1020 392 V484 H1110"/>
<path class="flow" stroke="#e7c9ff" d="M1020 392 V484 H1110"/>
<!-- reboiler -->
<rect x="238" y="423" width="211" height="113" rx="35" fill="url(#boil)" stroke="#36557c" stroke-width="3" filter="url(#shadow)"/>
<rect class="heat" x="251" y="457" width="185" height="65" rx="25" fill="#ff6b13" opacity=".55"/>
<path d="M279 536 V548 M407 536 V548" stroke="#36557c" stroke-width="8" stroke-linecap="round"/>
<text x="343" y="472" text-anchor="middle" font-size="29">🔥</text>
<text x="343" y="501" text-anchor="middle" class="tag">V001</text><text x="343" y="522" text-anchor="middle" class="small">Reboiler</text>
<!-- column -->
<rect x="286" y="157" width="119" height="256" rx="28" fill="url(#glass)" stroke="#36557c" stroke-width="3" filter="url(#shadow)"/>
<rect x="294" y="290" width="103" height="114" rx="16" fill="#3ab7e8" opacity=".22"/>
<g stroke="#80bcd9" stroke-width="4"><path d="M299 212 H391 M299 260 H391 M299 310 H391 M299 359 H391"/></g>
<g fill="#fff" stroke="#68d8f7"><circle class="bubble" cx="318" cy="367" r="6"/><circle class="bubble" cx="355" cy="355" r="5"/><circle class="bubble" cx="379" cy="379" r="4"/><circle class="bubble" cx="334" cy="392" r="4"/></g>
<text x="346" y="274" text-anchor="middle" class="tag">C001–C003</text><text x="346" y="294" text-anchor="middle" class="small">Column</text>
<path class="pipe" stroke="#ff9b38" d="M343 423 V410"/><path class="flow" stroke="#fff2a6" d="M343 423 V410"/>
<!-- condenser -->
<rect x="607" y="111" width="230" height="107" rx="28" fill="url(#cool)" stroke="#36557c" stroke-width="3" filter="url(#shadow)"/>
<g stroke="#06b5ef" stroke-width="10" stroke-linecap="round"><path d="M632 137 H812 M632 156 H812 M632 175 H812"/></g>
<text x="723" y="197" text-anchor="middle" class="tag">HE001 / HE002</text><text x="723" y="214" text-anchor="middle" font-size="13">Condensers</text>
<circle cx="824" cy="113" r="27" fill="#1686e7"/><text x="824" y="123" text-anchor="middle" fill="white" font-size="31">❄</text>
<!-- receiver -->
<rect x="902" y="277" width="150" height="116" rx="28" fill="url(#glass)" stroke="#36557c" stroke-width="3" filter="url(#shadow)"/>
<path d="M912 345 Q976 327 1042 345 V371 Q1042 383 1028 383 H927 Q912 383 912 369Z" fill="#20b9ee" opacity=".5"/>
<text x="977" y="326" text-anchor="middle" class="tag">V002</text><text x="977" y="349" text-anchor="middle" class="small">Receiver</text>
<!-- flow labels -->
<g font-size="17" font-weight="800"><text x="433" y="104" fill="#dc711c">Vapour →</text><text x="867" y="144" fill="#0989c8">Condensed liquid →</text><text x="645" y="397" fill="#12895b">← Reflux to column</text><text x="1080" y="464" fill="#7634b7">Product →</text></g>
<!-- four concise callouts -->
<g><rect class="call" x="20" y="160" width="229" height="110" rx="17" filter="url(#shadow)"/><text x="38" y="193" fill="#1762ba" class="tag">① Separate</text><text x="38" y="219" fill="#273b64" class="small">Vapour rises while liquid</text><text x="38" y="241" fill="#273b64" class="small">flows down the column.</text><path d="M249 217 H285" stroke="#6c54d5" stroke-width="2"/></g>
<g><rect class="call" x="20" y="435" width="195" height="100" rx="17" filter="url(#shadow)"/><text x="38" y="468" fill="#d65d18" class="tag">② Heat</text><text x="38" y="493" fill="#273b64" class="small">Reboiler heats mixture</text><text x="38" y="514" fill="#273b64" class="small">to produce vapour.</text><path d="M215 481 H237" stroke="#ef8a39" stroke-width="2"/></g>
<g><rect class="call" x="898" y="18" width="277" height="99" rx="17" filter="url(#shadow)"/><text x="917" y="51" fill="#1762ba" class="tag">③ Condense</text><text x="917" y="75" fill="#273b64" class="small">Cool vapour back into liquid</text><text x="917" y="96" fill="#273b64" class="small">inside the condensers.</text><path d="M898 83 L847 114" stroke="#6c54d5" stroke-width="2"/></g>
<g><rect class="call" x="1055" y="291" width="127" height="103" rx="17" filter="url(#shadow)"/><text x="1068" y="323" fill="#1762ba" font-size="17" font-weight="800">④ Collect</text><text x="1068" y="348" fill="#273b64" font-size="13">Liquid is</text><text x="1068" y="366" fill="#273b64" font-size="13">collected here.</text></g>
<!-- anomaly location -->
<circle cx="462" cy="460" r="9" fill="#ef4765"/><path d="M462 451 V439 H511" stroke="#ef4765" stroke-width="2" stroke-dasharray="5 5" fill="none"/>
<rect x="506" y="443" width="335" height="67" rx="13" fill="#fff0f3" stroke="#ffc4cf" stroke-width="2"/>
<text x="522" y="468" font-size="16" font-weight="800" fill="#d52d52">⚠ Example disturbance</text><text x="522" y="490" font-size="14" fill="#a92746">Nitrogen injection at reboiler (OP10)</text>
<text x="16" y="576" font-size="12" fill="#74829c">Conceptual animated schematic · not an engineering P&amp;ID · motion illustrates direction, not measured flow speed</text>
</svg>
<div class="foot"><span class="pill">🔥 Heating · V001</span><span class="arrow">→</span><span class="pill">🧪 Separation · C001–C003</span><span class="arrow">→</span><span class="pill">❄️ Condensation · HE001/HE002</span><span class="arrow">→</span><span class="pill">💧 Collection · V002</span></div>
<div class="hint">Illustrative process only; actual sensor behaviour comes from the dataset.</div>
</div></body></html>
"""

def hero(kicker, title, description):
    st.markdown(f'<div class="hero"><div class="kicker">{kicker}</div><h1>{title}</h1><p>{description}</p></div>', unsafe_allow_html=True)

def goto_explore(op_value):
    st.session_state.selected_op = op_value
    st.session_state.pending_op = op_value
    st.session_state.page = "Explore an anomaly"

if page == "Overview":
    hero("FROM PROCESS TO PATTERN", "DistillSense", "Explore how disturbances influence the sensors of a batch distillation system — from a plant-wide picture to a single event.")
    m = st.columns(4)
    m[0].metric("Operating points", len(OPS))
    m[1].metric("Annotated anomalies", sum(counts.values()))
    m[2].metric("Operating point with most events", max(counts, key=counts.get))
    m[3].metric("Experiment types", "2", help="Normal and anomalous recordings")
    st.markdown("### How the chemical process works")
    st.markdown('<div class="note">A simplified view of the equipment and where disturbances can influence sensor readings.</div>',unsafe_allow_html=True)
    st.components.v1.html(PROCESS_SVG, height=735, scrolling=False)
    st.caption("Follow the animated flows to see how heat, vapour, condensate and reflux connect. The red callout illustrates the OP10 nitrogen-injection disturbance; other operating points may involve different equipment.")
    st.markdown("### Choose an operating point")
    st.caption("Click a card to open its anomaly-response fingerprint and event-level sensor plots.")
    cards = st.columns(3)
    for i, item in enumerate(OPS):
        with cards[i%3]:
            st.markdown(f"**{item}**  ·  {counts[item]} annotated event(s)")
            st.button(f"Explore {item}  →", key=f"go_{item}", use_container_width=True, on_click=goto_explore, args=(item,))
    st.markdown('<div class="callout"><b>Investigation pathway</b><br>Choose an operating point → scan the sensor impact ranking → select a documented disturbance → inspect the linked sensor before, during and after the event.</div>',unsafe_allow_html=True)

elif page == "Explore an anomaly":
    selected_op=st.session_state.selected_op
    normal=measurements(selected_op,"Normal")
    anomalous=measurements(selected_op,"Anomalous")
    ann=annotations_for(selected_op)
    if normal.empty and anomalous.empty:
        st.warning(f"No measurements found for {selected_op}.")
        st.stop()
    all_sensors=sorted(set(sensors_of(normal))|set(sensors_of(anomalous)))
    shared=sorted(set(sensors_of(normal))&set(sensors_of(anomalous)))
    affected=[str(v) for v in ann.get("Affected Sensor",pd.Series(dtype=str)).dropna().unique() if str(v) in all_sensors]
    st.markdown("""<h1 style="margin:0;font-size:2.25rem">DistillSense</h1>
    <p style="color:#626e8c;margin:2px 0 18px">An interactive exploration of how a disturbance propagates through a batch distillation process.</p>""",unsafe_allow_html=True)
    m=st.columns(4)
    m[0].metric("Operating points",len(OPS))
    m[1].metric(f"{selected_op} measurements",f"{len(anomalous):,}")
    m[2].metric("Measured sensors",len(all_sensors))
    m[3].metric("Documented anomalies",len(ann))
    st.markdown("## ⚡ Explore an Anomaly")
    st.caption("Identify which sensors respond most strongly and inspect their readings during the documented disturbance.")
    c_op,c_event,c_view=st.columns([1,2.4,1.2],gap="medium")
    with c_op:
        if st.session_state.get("explore_op") not in OPS:
            st.session_state.explore_op = selected_op
        st.selectbox("Operating point", OPS, key="explore_op",
                     on_change=lambda: st.session_state.update(selected_op=st.session_state.explore_op,
                                                               op_selector=st.session_state.explore_op))
    if ann.empty:
        st.info("No documented events for this operating point.")
        st.stop()
    with c_event:
        idx=st.selectbox("Documented disturbance",list(range(len(ann))),format_func=lambda i:f"Event {i+1} · {str(ann.iloc[i].get('Perturbation') or ann.iloc[i].get('Category') or ann.iloc[i].get('ID'))[:85]}",key=f"event_{selected_op}")
    with c_view:
        mode=st.segmented_control("View",["Sensor Impact","Response Timeline"],default="Sensor Impact",key=f"mode_{selected_op}")
    event=ann.iloc[idx]
    suggested=str(event.get("Affected Sensor") or "")
    profile=response_profile(normal,anomalous,shared,event)
    ranking,ranked=impact_ranking(profile,suggested) if not profile.empty else (None,pd.DataFrame())
    if not ranked.empty:
        default_sensor=suggested if suggested in all_sensors else str(ranked.iloc[0]["Sensor"])
    else:
        default_sensor=suggested if suggested in all_sensors else (all_sensors[0] if all_sensors else None)
    if not all_sensors:
        st.warning("No numeric sensors available.")
        st.stop()
    left,right=st.columns([1.55,1],gap="medium")
    with left:
        if mode=="Sensor Impact":
            st.markdown('<div class="panel-title">🎯 Sensor Impact Ranking</div><div class="panel-caption">Sensors ranked by excess change in the anomalous run relative to the normal run.</div>',unsafe_allow_html=True)
            if ranking is not None:
                st.plotly_chart(ranking,width="stretch",config={"displaylogo":False})
            else:
                st.info("Insufficient measurements to calculate the response ranking.")
            st.caption("The score compares each run’s event-period median with its own 10-minute pre-event baseline. Zero means no excess anomalous response; it does not imply no change. This is not a validated anomaly detector.")
        else:
            st.markdown('<div class="panel-title">⏱️ Response Timeline</div><div class="panel-caption">See the annotated event window and selected sensor before, during and after the disturbance.</div>',unsafe_allow_html=True)
            st.plotly_chart(detail_figure(normal,anomalous,default_sensor,ann,"Anomalous"),width="stretch",config={"displaylogo":False})
            st.caption("Shaded region marks the documented event, not an automatically detected fault.")
    with right:
        options=all_sensors
        sensor=st.selectbox("🔎 Sensor detail",options,index=options.index(default_sensor) if default_sensor in options else 0,key=f"detail_sensor_{selected_op}_{idx}")
        st.markdown(f'<div class="panel-title">📈 Sensor detail: {html.escape(sensor)}</div><div class="panel-caption">Normal and anomalous runs · elapsed time from each recording start</div>',unsafe_allow_html=True)
        st.plotly_chart(detail_figure(normal,anomalous,sensor,ann,"Both"),width="stretch",config={"displaylogo":False})
        if not ranked.empty:
            positive=ranked[ranked["Impact"]>0]
            if not positive.empty:
                lead=positive.iloc[0]
                st.markdown(f'<div class="callout"><b>💡 Key observation</b><br><b>{html.escape(str(lead["Sensor"]))}</b> has the largest excess relative response ({lead["Impact"]:.1f} scaled units) for this event. Inspect the trace to understand the actual change.</div>',unsafe_allow_html=True)
            else:
                st.info("No sensor has a larger scaled median shift in the anomalous run for this event.")
        st.markdown(f'<div class="callout"><b>What happened?</b><br>{html.escape(str(event.get("Perturbation") or event.get("Category") or "Documented disturbance"))}<br><small>Annotated time: {html.escape(str(event.get("Start")))} – {html.escape(str(event.get("End")))}</small></div>',unsafe_allow_html=True)
        st.caption("The recordings were collected separately; matching elapsed times may not represent the same process stage.")
    with st.expander("📋 Documented anomaly details"):
        st.dataframe(ann,width="stretch",hide_index=True)

elif page == "Process & dataset":
    hero("UNDERSTAND THE SYSTEM","The batch distillation process","A conceptual equipment overview to connect disturbances to physical process components.")
    st.components.v1.html(PROCESS_SVG, height=735, scrolling=False)
    st.markdown("### Dataset coverage")
    st.dataframe(pd.DataFrame({"Operating point":OPS,"Annotated events":[counts[o] for o in OPS]}),hide_index=True,width="stretch")

else:
    hero("READ THE VISUAL STORY", "How to interpret the dashboard", "A short guide to the process schematic, sensor impact ranking, and event investigation.")
    st.markdown("### What each view answers")
    st.markdown("""<div class="step"><b>Process schematic</b><br>Where can a disturbance enter the distillation system, and which equipment may be involved?</div>
<div class="step"><b>Operating-point overview</b><br>Which experimental conditions are available, and how many documented events does each contain?</div>
<div class="step"><b>Sensor impact ranking</b><br>Which sensors shift most during a selected event compared with their own 10-minute pre-event baseline? Each row ranks the excess response during the anomalous recording compared with the normal recording.</div>
<div class="step"><b>Event-level time series</b><br>How does the selected sensor evolve before, during, and after the documented disturbance?</div>""",unsafe_allow_html=True)
    st.markdown("### Reading the colours")
    st.markdown("**Impact ranking:** longer lollipops show a larger excess response in the anomalous recording relative to the normal recording. **Time series:** blue is the normal run, coral is the anomalous run, and the pale coral band marks annotated event timing.")
    st.warning("Important interpretation limit: normal and anomalous experiments may have different starting conditions and process stages. A large difference does not, by itself, prove that an anomaly caused the change.",icon="⚠️")
    st.markdown("### Suggested supervisor walkthrough")
    st.markdown("**1.** Start with the process diagram. **2.** Select OP10. **3.** Show the sensor impact ranking. **4.** Select the nitrogen-injection annotation. **5.** Explain the sensor response within the shaded event interval. **6.** Switch to another operating point to show the design works across the dataset.")
