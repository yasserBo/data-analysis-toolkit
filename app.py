from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Data Analysis Toolkit",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .hero {
        padding: 2.2rem;
        border-radius: 28px;
        background: radial-gradient(circle at top left, #22c55e 0, transparent 25%),
                    linear-gradient(135deg, #0f172a 0%, #1e3a8a 45%, #4c1d95 100%);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 14px 36px rgba(15, 23, 42, 0.24);
    }

    .hero h1 {
        font-size: 2.5rem;
        line-height: 1.1;
        margin-bottom: 0.5rem;
        color: white;
    }

    .hero p {
        color: #e5edff;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .tool-card {
        background: #ffffff;
        border: 1px solid #e6edf7;
        border-radius: 22px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
    }

    .tool-card h3 {
        margin-top: 0;
    }

    .feature-pill {
        display: inline-block;
        background: #eef4ff;
        color: #1d4ed8;
        padding: 0.38rem 0.75rem;
        border-radius: 999px;
        font-size: 0.86rem;
        font-weight: 700;
        margin: 0.2rem 0.25rem 0.2rem 0;
    }

    .step-box {
        background: linear-gradient(135deg, #eff6ff, #ffffff);
        border: 1px solid #dbeafe;
        border-left: 6px solid #2563eb;
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    .success-box {
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #22c55e;
        border-radius: 16px;
        padding: 1rem;
        color: #064e3b;
        margin-bottom: 0.8rem;
    }

    .warning-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-left: 6px solid #f97316;
        border-radius: 16px;
        padding: 1rem;
        color: #7c2d12;
        margin-bottom: 0.8rem;
    }

    .danger-box {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 6px solid #ef4444;
        border-radius: 16px;
        padding: 1rem;
        color: #7f1d1d;
        margin-bottom: 0.8rem;
    }

    .mini-title {
        font-weight: 800;
        color: #0f172a;
        font-size: 1.05rem;
        margin-bottom: 0.35rem;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6edf7;
        padding: 1rem;
        border-radius: 18px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
    }

    [data-testid="stMetricValue"] {
        font-weight: 900;
        color: #0f172a;
    }

    .sidebar-footer {
        font-size: 0.82rem;
        opacity: 0.75;
        margin-top: 2rem;
    }

    .journey-wrap {
        display: flex;
        gap: 0.45rem;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }

    .journey-step-done {
        padding: 0.5rem 0.8rem;
        border-radius: 999px;
        background: #dcfce7;
        color: #166534;
        font-size: 0.85rem;
        font-weight: 800;
        border: 1px solid #86efac;
    }

    .journey-step-current {
        padding: 0.5rem 0.8rem;
        border-radius: 999px;
        background: #dbeafe;
        color: #1d4ed8;
        font-size: 0.85rem;
        font-weight: 900;
        border: 1px solid #93c5fd;
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.12);
    }

    .journey-step-waiting {
        padding: 0.5rem 0.8rem;
        border-radius: 999px;
        background: #f8fafc;
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 700;
        border: 1px solid #e2e8f0;
    }

    .action-card {
        background: linear-gradient(135deg, #ffffff, #f8fbff);
        border: 1px solid #dbeafe;
        border-radius: 20px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)


def find_sample_dataset():
    candidates = [
        "data/StudentsPerformance.csv",
        "data/students_performance.csv",
        "data/StudentsPerformance.CSV",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    data_folder = Path("data")
    if data_folder.exists():
        csv_files = list(data_folder.glob("*.csv"))
        if csv_files:
            return str(csv_files[0])

    return None


def get_column_types(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    return numeric_cols, categorical_cols


def clean_dataset(df, remove_duplicates=True, missing_method="Fill numeric mean + categorical mode"):
    cleaned = df.copy()

    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()

    numeric_cols, categorical_cols = get_column_types(cleaned)

    if missing_method == "Keep missing values":
        return cleaned

    if missing_method == "Drop rows with missing values":
        return cleaned.dropna()

    if missing_method == "Fill numeric mean + categorical mode":
        for col in numeric_cols:
            cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
        for col in categorical_cols:
            mode_value = cleaned[col].mode()
            cleaned[col] = cleaned[col].fillna(mode_value[0] if not mode_value.empty else "Unknown")

    if missing_method == "Fill numeric median + categorical mode":
        for col in numeric_cols:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        for col in categorical_cols:
            mode_value = cleaned[col].mode()
            cleaned[col] = cleaned[col].fillna(mode_value[0] if not mode_value.empty else "Unknown")

    return cleaned


def missing_table(df):
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2) if len(df) else missing
    return pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values,
        "Missing %": missing_pct.values
    })


def descriptive_statistics(df):
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return pd.DataFrame()

    return pd.DataFrame({
        "Mean": numeric_df.mean(),
        "Median": numeric_df.median(),
        "Mode": numeric_df.mode().iloc[0],
        "Min": numeric_df.min(),
        "Max": numeric_df.max(),
        "Range": numeric_df.max() - numeric_df.min(),
        "Variance": numeric_df.var(),
        "Std Dev": numeric_df.std(),
        "Skewness": numeric_df.skew(),
        "Kurtosis": numeric_df.kurtosis()
    }).round(3)


def dataset_health_score(df):
    if df.empty:
        return 0

    total_cells = df.shape[0] * df.shape[1]
    missing_rate = df.isna().sum().sum() / total_cells if total_cells else 0
    duplicate_rate = df.duplicated().sum() / len(df) if len(df) else 0

    numeric_cols, categorical_cols = get_column_types(df)
    type_balance_bonus = 5 if numeric_cols and categorical_cols else 0

    score = 100 - (missing_rate * 70) - (duplicate_rate * 35) + type_balance_bonus
    return int(max(0, min(100, round(score))))


def health_label(score):
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 55:
        return "Needs attention"
    return "Poor"


def smart_recommendations(raw_df, working_df):
    recommendations = []
    numeric_cols, categorical_cols = get_column_types(working_df)

    total_missing = raw_df.isna().sum().sum()
    duplicates = raw_df.duplicated().sum()

    if total_missing > 0:
        recommendations.append("Handle missing values before running statistical tests.")
    else:
        recommendations.append("No missing values detected. You can move directly to visualization and inference.")

    if duplicates > 0:
        recommendations.append("Remove duplicate rows to avoid biased summaries.")
    else:
        recommendations.append("No duplicate rows detected.")

    if len(numeric_cols) >= 2:
        recommendations.append("Use the Chart Builder or Correlation Heatmap to explore relationships between numerical variables.")

    if len(categorical_cols) >= 2:
        recommendations.append("Use the Chi-Square Test to check relationships between categorical variables.")

    if len(numeric_cols) >= 1:
        recommendations.append("Use Distribution Lab to check whether numerical variables look normally distributed.")

    if working_df.shape[0] < 30:
        recommendations.append("Your dataset is small. Be careful when interpreting statistical tests.")

    return recommendations


def apply_interactive_filters(df, categorical_cols, numeric_cols):
    filtered = df.copy()

    with st.sidebar.expander("🔎 Optional Interactive Filters", expanded=False):
        st.caption("Use filters to focus the analysis on a subset of the dataset.")

        selected_cat_filters = st.multiselect(
            "Categorical filters",
            categorical_cols,
            key="selected_cat_filters"
        )

        for col in selected_cat_filters:
            values = sorted(filtered[col].dropna().astype(str).unique().tolist())
            selected_values = st.multiselect(
                f"Keep values for {col}",
                values,
                default=values,
                key=f"cat_filter_{col}"
            )

            if selected_values:
                filtered = filtered[filtered[col].astype(str).isin(selected_values)]

        selected_num_filters = st.multiselect(
            "Numerical filters",
            numeric_cols,
            key="selected_num_filters"
        )

        for col in selected_num_filters:
            col_data = filtered[col].dropna()
            if not col_data.empty and col_data.min() != col_data.max():
                min_val = float(col_data.min())
                max_val = float(col_data.max())

                selected_range = st.slider(
                    f"Range for {col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key=f"num_filter_{col}"
                )

                filtered = filtered[
                    (filtered[col] >= selected_range[0]) &
                    (filtered[col] <= selected_range[1])
                ]

    return filtered


def info_box(text):
    st.markdown(f'<div class="step-box">{text}</div>', unsafe_allow_html=True)


def success_box(text):
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)


def warning_box(text):
    st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)


def danger_box(text):
    st.markdown(f'<div class="danger-box">{text}</div>', unsafe_allow_html=True)


def p_value_interpretation(p_value, alpha=0.05):
    if p_value < alpha:
        return "Reject the null hypothesis. The result is statistically significant."
    return "Fail to reject the null hypothesis. The result is not statistically significant."


def generate_markdown_report(dataset_name, raw_df, working_df, recommendations):
    numeric_cols, categorical_cols = get_column_types(working_df)
    score = dataset_health_score(raw_df)

    report = f"""# Data Analysis Toolkit Report

## Dataset

Dataset name: {dataset_name}

## Dataset Shape

Original dataset:
- Rows: {raw_df.shape[0]}
- Columns: {raw_df.shape[1]}

Current working dataset:
- Rows: {working_df.shape[0]}
- Columns: {working_df.shape[1]}

## Dataset Health

Health score: {score}/100
Health status: {health_label(score)}

## Column Types

Numerical columns:
{", ".join(numeric_cols) if numeric_cols else "None"}

Categorical columns:
{", ".join(categorical_cols) if categorical_cols else "None"}

## Missing Values

Total missing values in original dataset: {raw_df.isna().sum().sum()}

## Duplicate Rows

Duplicate rows in original dataset: {raw_df.duplicated().sum()}

## Smart Recommendations

"""
    for rec in recommendations:
        report += f"- {rec}\n"

    report += """

## Notes

This report was automatically generated by the Student-Friendly Data Analysis Toolkit.
"""
    return report



# ============================================================
# DYNAMIC UI FUNCTIONS
# ============================================================

JOURNEY_STEPS = [
    "🏠 Data Hub",
    "🧹 Cleaning Studio",
    "💡 Insight Center",
    "🎨 Chart Builder",
    "📊 Distribution Lab",
    "🧪 Test Lab",
    "📦 Export Center"
]


def init_session_state():
    if "visited_pages" not in st.session_state:
        st.session_state.visited_pages = []

    if "scan_count" not in st.session_state:
        st.session_state.scan_count = 0

    if "show_tips" not in st.session_state:
        st.session_state.show_tips = True

    if "last_action" not in st.session_state:
        st.session_state.last_action = "Dataset loaded"


def mark_page_visited(page_name):
    if page_name not in st.session_state.visited_pages:
        st.session_state.visited_pages.append(page_name)


def render_journey_tracker(current_page):
    visited = st.session_state.visited_pages
    completed_count = len(set(visited))
    progress_value = completed_count / len(JOURNEY_STEPS)

    st.write("### 🧭 Analysis Journey Progress")
    st.progress(progress_value, text=f"{completed_count}/{len(JOURNEY_STEPS)} workspaces explored")

    html = '<div class="journey-wrap">'
    for step in JOURNEY_STEPS:
        if step == current_page:
            css_class = "journey-step-current"
        elif step in visited:
            css_class = "journey-step-done"
        else:
            css_class = "journey-step-waiting"
        html += f'<span class="{css_class}">{step}</span>'
    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)


def run_dynamic_scan(raw_df, working_df, recommendations):
    st.session_state.scan_count += 1

    with st.status("Running dynamic dataset scan...", expanded=True) as status:
        st.write("Checking dataset shape...")
        time.sleep(0.2)

        st.write("Checking missing values and duplicate rows...")
        time.sleep(0.2)

        st.write("Detecting numerical and categorical columns...")
        time.sleep(0.2)

        st.write("Generating smart recommendations...")
        time.sleep(0.2)

        status.update(label="Dataset scan completed", state="complete", expanded=False)

    st.toast("Dataset scan completed successfully!", icon="✅")

    if dataset_health_score(raw_df) >= 90:
        st.balloons()

    st.session_state.last_action = "Dynamic scan completed"

    st.markdown('<div class="action-card">', unsafe_allow_html=True)
    st.write("### ⚡ Dynamic Scan Result")
    st.write(f"Scan number: **{st.session_state.scan_count}**")
    st.write(f"Original rows: **{raw_df.shape[0]:,}**")
    st.write(f"Working rows after cleaning/filtering: **{working_df.shape[0]:,}**")
    st.write(f"Missing values in original dataset: **{raw_df.isna().sum().sum():,}**")
    st.write(f"Duplicate rows in original dataset: **{raw_df.duplicated().sum():,}**")

    st.write("#### Recommended next actions")
    for rec in recommendations:
        st.write(f"✅ {rec}")

    st.markdown('</div>', unsafe_allow_html=True)


def dynamic_tip_box(page_name, numeric_cols, categorical_cols):
    if not st.session_state.show_tips:
        return

    tips = {
        "🏠 Data Hub": "Start here to understand whether your dataset is clean enough for analysis.",
        "🧹 Cleaning Studio": "Use this section before making conclusions. Bad data quality can produce misleading results.",
        "💡 Insight Center": "Use this section to quickly understand the main numerical patterns.",
        "🎨 Chart Builder": "Try different chart types. Visual exploration often reveals patterns before statistical tests.",
        "📊 Distribution Lab": "Use Q-Q plots and PDF/CDF curves to understand whether numerical data looks normally distributed.",
        "🧪 Test Lab": "Choose tests based on your variable types: numerical variables for t-tests/correlation, categorical variables for chi-square.",
        "📦 Export Center": "Export the cleaned dataset and generated report for your assignment submission."
    }

    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.write(f"💬 **Toolkit tip:** {tips.get(page_name, 'Explore this workspace interactively.')}")

    if numeric_cols and categorical_cols:
        st.caption("Your dataset has both numerical and categorical columns, so most toolkit features are available.")
    elif numeric_cols:
        st.caption("Your dataset has numerical columns. Visualization, distributions, and inference tests are available.")
    elif categorical_cols:
        st.caption("Your dataset has categorical columns. Bar charts and chi-square tests are available.")
    else:
        st.caption("No clear numerical or categorical columns were detected.")

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# HERO HEADER
# ============================================================

init_session_state()

st.markdown(
    """
    <div class="hero">
        <h1>🧠 Student-Friendly Data Analysis Toolkit</h1>
        <p>
        A modern guided workspace for exploring, cleaning, visualizing, and testing datasets.
        Built for students, beginners, and fast data storytelling.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <span class="feature-pill">Guided Workflow</span>
    <span class="feature-pill">Smart Recommendations</span>
    <span class="feature-pill">Interactive Filters</span>
    <span class="feature-pill">Chart Builder</span>
    <span class="feature-pill">Statistical Test Lab</span>
    <span class="feature-pill">Exportable Report</span>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧭 Analysis Journey")

page = st.sidebar.radio(
    "Choose a workspace",
    [
        "🏠 Data Hub",
        "🧹 Cleaning Studio",
        "💡 Insight Center",
        "🎨 Chart Builder",
        "📊 Distribution Lab",
        "🧪 Test Lab",
        "📦 Export Center"
    ],
    key="workspace_page"
)

mark_page_visited(page)

if st.sidebar.button("⚡ Run Dynamic Scan", use_container_width=True, key="quick_scan_button"):
    st.session_state.run_dynamic_scan = True

st.session_state.show_tips = st.sidebar.toggle(
    "Show smart tips",
    value=st.session_state.show_tips,
    key="show_tips_toggle"
)

st.sidebar.divider()
st.sidebar.subheader("📁 Dataset")

dataset_source = st.sidebar.radio(
    "Choose dataset source",
    ["Use sample dataset", "Upload your own CSV"],
    key="dataset_source"
)

raw_df = None
dataset_name = "No dataset loaded"

if dataset_source == "Use sample dataset":
    sample_path = find_sample_dataset()

    if sample_path:
        raw_df = load_csv(sample_path)
        dataset_name = Path(sample_path).name
        st.sidebar.success(f"Loaded: {dataset_name}")
    else:
        st.sidebar.error("No sample CSV found in the data folder.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        key="csv_uploader"
    )

    if uploaded_file is not None:
        raw_df = load_csv(uploaded_file)
        dataset_name = uploaded_file.name
        st.sidebar.success(f"Uploaded: {dataset_name}")

if raw_df is None:
    st.markdown('<div class="tool-card">', unsafe_allow_html=True)
    st.subheader("Welcome 👋")
    st.write(
        """
        Start by choosing the sample dataset or uploading your own CSV file from the sidebar.

        This toolkit is organized as an analysis journey:
        **Data Hub → Cleaning Studio → Insight Center → Chart Builder → Distribution Lab → Test Lab → Export Center**.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


st.sidebar.divider()
st.sidebar.subheader("🧹 Quick Cleaning")

remove_duplicates = st.sidebar.checkbox(
    "Remove duplicate rows",
    value=True,
    key="remove_duplicates"
)

missing_method = st.sidebar.selectbox(
    "Missing-value method",
    [
        "Keep missing values",
        "Drop rows with missing values",
        "Fill numeric mean + categorical mode",
        "Fill numeric median + categorical mode"
    ],
    index=2,
    key="missing_method"
)

cleaned_df = clean_dataset(raw_df, remove_duplicates, missing_method)
numeric_cols, categorical_cols = get_column_types(cleaned_df)
working_df = apply_interactive_filters(cleaned_df, categorical_cols, numeric_cols)
numeric_cols, categorical_cols = get_column_types(working_df)

st.sidebar.divider()
st.sidebar.markdown(
    '<div class="sidebar-footer">Created by Yasser Bouyaddid<br>Data Analysis Toolkit</div>',
    unsafe_allow_html=True
)


# ============================================================
# TOP STATUS BAR
# ============================================================

health_score = dataset_health_score(raw_df)
recommendations = smart_recommendations(raw_df, working_df)

render_journey_tracker(page)
dynamic_tip_box(page, numeric_cols, categorical_cols)

if st.session_state.get("run_dynamic_scan", False):
    run_dynamic_scan(raw_df, working_df, recommendations)
    st.session_state.run_dynamic_scan = False

metric1, metric2, metric3, metric4, metric5 = st.columns(5)

with metric1:
    st.metric("Health Score", f"{health_score}/100", health_label(health_score))

with metric2:
    st.metric("Rows", f"{working_df.shape[0]:,}")

with metric3:
    st.metric("Columns", f"{working_df.shape[1]:,}")

with metric4:
    st.metric("Numerical", len(numeric_cols))

with metric5:
    st.metric("Categorical", len(categorical_cols))

st.markdown("---")


# ============================================================
# PAGE: DATA HUB
# ============================================================

if page == "🏠 Data Hub":
    st.subheader("🏠 Data Hub")
    info_box("This is the main control center. It gives a quick overview of dataset quality, structure, and readiness for analysis.")

    st.markdown('<div class="action-card">', unsafe_allow_html=True)
    st.write("### ⚡ Dynamic Controls")
    dc1, dc2, dc3 = st.columns(3)

    with dc1:
        if st.button("Run Quick Health Scan", use_container_width=True, key="hub_scan"):
            run_dynamic_scan(raw_df, working_df, recommendations)

    with dc2:
        if st.button("Celebrate Clean Data", use_container_width=True, key="celebrate_clean"):
            if health_score >= 90:
                st.balloons()
                st.toast("Excellent dataset quality!", icon="🎉")
            else:
                st.toast("Clean the dataset more to improve the health score.", icon="🧹")

    with dc3:
        if st.button("Suggest Next Step", use_container_width=True, key="suggest_step"):
            st.toast(recommendations[0] if recommendations else "Explore the dataset first.", icon="💡")

    st.caption(f"Last action: {st.session_state.last_action}")
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown('<div class="tool-card">', unsafe_allow_html=True)
        st.write("### Dataset Preview")
        st.write(f"**Dataset:** {dataset_name}")
        st.dataframe(working_df.head(12), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="tool-card">', unsafe_allow_html=True)
        st.write("### Dataset Health Gauge")

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=health_score,
                title={"text": f"Health Status: {health_label(health_score)}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2563eb"},
                    "steps": [
                        {"range": [0, 55], "color": "#fee2e2"},
                        {"range": [55, 75], "color": "#ffedd5"},
                        {"range": [75, 90], "color": "#dbeafe"},
                        {"range": [90, 100], "color": "#dcfce7"},
                    ],
                },
            )
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        if health_score >= 90:
            success_box("The dataset looks very clean and ready for analysis.")
        elif health_score >= 75:
            success_box("The dataset is mostly ready, but review cleaning details.")
        elif health_score >= 55:
            warning_box("The dataset needs attention before strong conclusions.")
        else:
            danger_box("The dataset has quality issues that should be fixed first.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.write("### Column Inventory")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="tool-card">', unsafe_allow_html=True)
        st.write("#### Numerical Columns")
        st.write(numeric_cols if numeric_cols else "No numerical columns found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="tool-card">', unsafe_allow_html=True)
        st.write("#### Categorical Columns")
        st.write(categorical_cols if categorical_cols else "No categorical columns found.")
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PAGE: CLEANING STUDIO
# ============================================================

elif page == "🧹 Cleaning Studio":
    st.subheader("🧹 Cleaning Studio")
    info_box("Use this workspace to understand missing values, duplicates, and how preprocessing changes the dataset.")

    before1, before2, before3, after1 = st.columns(4)

    with before1:
        st.metric("Original Rows", f"{raw_df.shape[0]:,}")

    with before2:
        st.metric("Duplicate Rows", f"{raw_df.duplicated().sum():,}")

    with before3:
        st.metric("Missing Cells", f"{raw_df.isna().sum().sum():,}")

    with after1:
        st.metric("Working Rows", f"{working_df.shape[0]:,}")

    st.write("### Missing-Value Map")

    mt = missing_table(raw_df)
    st.dataframe(mt, use_container_width=True)

    if mt["Missing Values"].sum() > 0:
        fig = px.bar(
            mt,
            x="Column",
            y="Missing %",
            text="Missing %",
            title="Missing Percentage by Column",
            template="plotly_white"
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
    else:
        success_box("No missing values found in the original dataset.")

    st.write("### Before vs After Cleaning")

    comparison = pd.DataFrame({
        "Metric": ["Rows", "Columns", "Duplicate Rows", "Missing Cells"],
        "Before": [
            raw_df.shape[0],
            raw_df.shape[1],
            raw_df.duplicated().sum(),
            raw_df.isna().sum().sum()
        ],
        "After": [
            working_df.shape[0],
            working_df.shape[1],
            working_df.duplicated().sum(),
            working_df.isna().sum().sum()
        ]
    })

    st.dataframe(comparison, use_container_width=True)

    fig_compare = px.bar(
        comparison,
        x="Metric",
        y=["Before", "After"],
        barmode="group",
        title="Cleaning Impact",
        template="plotly_white"
    )
    st.plotly_chart(fig_compare, use_container_width=True)


# ============================================================
# PAGE: INSIGHT CENTER
# ============================================================

elif page == "💡 Insight Center":
    st.subheader("💡 Insight Center")
    info_box("This workspace automatically summarizes the dataset and suggests what analysis to do next.")

    st.write("### Smart Recommendations")

    for rec in recommendations:
        st.markdown(f"- ✅ {rec}")

    st.write("### Descriptive Statistics")

    stats_table = descriptive_statistics(working_df)

    if stats_table.empty:
        st.warning("No numerical columns available for descriptive statistics.")
    else:
        st.dataframe(stats_table, use_container_width=True)

        chosen_col = st.selectbox(
            "Choose one numerical column for an insight card",
            numeric_cols,
            key="insight_column"
        )

        data = working_df[chosen_col].dropna()

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Mean", round(data.mean(), 3))

        with c2:
            st.metric("Median", round(data.median(), 3))

        with c3:
            st.metric("Std Dev", round(data.std(), 3))

        with c4:
            st.metric("Skewness", round(data.skew(), 3))

        if data.skew() > 0.5:
            warning_box(f"{chosen_col} is positively skewed. Higher values stretch the distribution to the right.")
        elif data.skew() < -0.5:
            warning_box(f"{chosen_col} is negatively skewed. Lower values stretch the distribution to the left.")
        else:
            success_box(f"{chosen_col} looks fairly balanced based on skewness.")


# ============================================================
# PAGE: CHART BUILDER
# ============================================================

elif page == "🎨 Chart Builder":
    st.subheader("🎨 Interactive Chart Builder")
    info_box("Choose a chart type, select columns, and build visualizations dynamically. This is the most interactive part of the toolkit.")

    chart_type = st.radio(
        "Choose chart type",
        [
            "Histogram",
            "Boxplot",
            "Scatter Plot",
            "Bar Chart",
            "Correlation Heatmap",
            "Pairwise Scatter Matrix"
        ],
        horizontal=True,
        key="chart_type"
    )

    if chart_type == "Histogram":
        if numeric_cols:
            col = st.selectbox("Numerical column", numeric_cols, key="hist_col")
            color_col = st.selectbox("Optional category color", ["None"] + categorical_cols, key="hist_color")
            bins = st.slider("Number of bins", 5, 60, 25, key="hist_bins")

            if color_col == "None":
                fig = px.histogram(working_df, x=col, nbins=bins, marginal="box", template="plotly_white")
            else:
                fig = px.histogram(working_df, x=col, color=color_col, nbins=bins, marginal="box", template="plotly_white")

            fig.update_layout(title=f"Histogram of {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numerical columns available.")

    elif chart_type == "Boxplot":
        if numeric_cols:
            y_col = st.selectbox("Numerical column", numeric_cols, key="box_y")
            x_col = st.selectbox("Group by category", ["None"] + categorical_cols, key="box_x")

            if x_col == "None":
                fig = px.box(working_df, y=y_col, points="outliers", template="plotly_white")
            else:
                fig = px.box(working_df, x=x_col, y=y_col, points="outliers", template="plotly_white")

            fig.update_layout(title=f"Boxplot of {y_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numerical columns available.")

    elif chart_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Y axis", numeric_cols, key="scatter_y")
            color_col = st.selectbox("Color by", ["None"] + categorical_cols, key="scatter_color")
            size_col = st.selectbox("Optional size by", ["None"] + numeric_cols, key="scatter_size")
            add_trend = st.checkbox("Add trendline", value=True, key="scatter_trend")

            args = {
                "data_frame": working_df,
                "x": x_col,
                "y": y_col,
                "template": "plotly_white",
                "title": f"{x_col} vs {y_col}"
            }

            if color_col != "None":
                args["color"] = color_col

            if size_col != "None":
                args["size"] = size_col

            if add_trend:
                args["trendline"] = "ols"

            fig = px.scatter(**args)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("At least two numerical columns are required.")

    elif chart_type == "Bar Chart":
        if categorical_cols:
            cat_col = st.selectbox("Categorical column", categorical_cols, key="bar_cat")
            counts = working_df[cat_col].value_counts().reset_index()
            counts.columns = [cat_col, "Count"]

            fig = px.bar(
                counts,
                x=cat_col,
                y="Count",
                text="Count",
                template="plotly_white",
                title=f"Bar Chart of {cat_col}"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No categorical columns available.")

    elif chart_type == "Correlation Heatmap":
        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect(
                "Choose numerical columns",
                numeric_cols,
                default=numeric_cols,
                key="corr_cols"
            )

            if len(selected_cols) >= 2:
                corr = working_df[selected_cols].corr()
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    template="plotly_white",
                    title="Correlation Heatmap"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Choose at least two numerical columns.")
        else:
            st.warning("At least two numerical columns are required.")

    elif chart_type == "Pairwise Scatter Matrix":
        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect(
                "Choose numerical columns",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))],
                key="pair_cols"
            )
            color_col = st.selectbox("Color by", ["None"] + categorical_cols, key="pair_color")

            if len(selected_cols) >= 2:
                if color_col == "None":
                    fig = px.scatter_matrix(working_df, dimensions=selected_cols, template="plotly_white")
                else:
                    fig = px.scatter_matrix(working_df, dimensions=selected_cols, color=color_col, template="plotly_white")

                fig.update_layout(title="Pairwise Scatter Matrix")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Choose at least two numerical columns.")
        else:
            st.warning("At least two numerical columns are required.")


# ============================================================
# PAGE: DISTRIBUTION LAB
# ============================================================

elif page == "📊 Distribution Lab":
    st.subheader("📊 Distribution Lab")
    info_box("Compare observed data with a fitted normal distribution using PDF, CDF, Q-Q plot, and goodness-of-fit testing.")

    if not numeric_cols:
        st.warning("No numerical columns available.")
    else:
        col = st.selectbox("Choose numerical column", numeric_cols, key="dist_col")
        data = working_df[col].dropna()

        if len(data) < 3:
            st.warning("At least 3 numerical observations are required.")
        else:
            mean = data.mean()
            std = data.std()

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.metric("Mean", round(mean, 3))
            with d2:
                st.metric("Std Dev", round(std, 3))
            with d3:
                st.metric("Skewness", round(data.skew(), 3))
            with d4:
                st.metric("Kurtosis", round(data.kurtosis(), 3))

            if std == 0 or np.isnan(std):
                st.warning("Standard deviation is zero or undefined.")
            else:
                x = np.linspace(data.min(), data.max(), 250)
                pdf = stats.norm.pdf(x, mean, std)
                cdf = stats.norm.cdf(x, mean, std)

                plot_choice = st.radio(
                    "Choose distribution view",
                    ["PDF Comparison", "CDF Curve", "Q-Q Plot"],
                    horizontal=True,
                    key="dist_view"
                )

                if plot_choice == "PDF Comparison":
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=data, histnorm="probability density", name="Observed Data", opacity=0.65))
                    fig.add_trace(go.Scatter(x=x, y=pdf, mode="lines", name="Fitted Normal PDF"))
                    fig.update_layout(title=f"PDF Comparison for {col}", xaxis_title=col, yaxis_title="Density", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                elif plot_choice == "CDF Curve":
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=cdf, mode="lines", name="Normal CDF"))
                    fig.update_layout(title=f"CDF Curve for {col}", xaxis_title=col, yaxis_title="Cumulative Probability", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                elif plot_choice == "Q-Q Plot":
                    theoretical_quantiles, ordered_values = stats.probplot(data, dist="norm")[0]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=ordered_values, mode="markers", name="Data points"))
                    fig.add_trace(go.Scatter(
                        x=[min(theoretical_quantiles), max(theoretical_quantiles)],
                        y=[min(ordered_values), max(ordered_values)],
                        mode="lines",
                        name="Reference line"
                    ))
                    fig.update_layout(title=f"Q-Q Plot for {col}", xaxis_title="Theoretical Quantiles", yaxis_title="Observed Values", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                st.write("### Goodness-of-Fit Test")
                ks_stat, ks_p = stats.kstest(data, "norm", args=(mean, std))
                st.write("Kolmogorov-Smirnov statistic:", round(ks_stat, 4))
                st.write("P-value:", round(ks_p, 4))

                if ks_p > 0.05:
                    success_box("The data may follow a normal distribution.")
                else:
                    warning_box("The data probably does not follow a normal distribution.")


# ============================================================
# PAGE: TEST LAB
# ============================================================

elif page == "🧪 Test Lab":
    st.subheader("🧪 Statistical Test Lab")
    info_box("Choose a test, select the variables, and get beginner-friendly interpretation.")

    test = st.selectbox(
        "Choose statistical test",
        [
            "Normality Test",
            "Confidence Interval",
            "One-Sample T-Test",
            "Chi-Square Test",
            "Pearson Correlation"
        ],
        key="test_choice"
    )

    if test == "Normality Test":
        if numeric_cols:
            col = st.selectbox("Numerical column", numeric_cols, key="normality_col")
            data = working_df[col].dropna()

            if len(data) < 3:
                st.warning("At least 3 values are required.")
            else:
                stat, p = stats.shapiro(data)
                st.write("### Shapiro-Wilk Test")
                st.write("Statistic:", round(stat, 4))
                st.write("P-value:", round(p, 4))

                if p > 0.05:
                    success_box("The data may be normally distributed.")
                else:
                    warning_box("The data does not appear to be normally distributed.")
        else:
            st.warning("No numerical columns available.")

    elif test == "Confidence Interval":
        if numeric_cols:
            col = st.selectbox("Numerical column", numeric_cols, key="ci_col")
            confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="ci_level")
            data = working_df[col].dropna()

            if len(data) < 2:
                st.warning("At least 2 values are required.")
            else:
                interval = stats.t.interval(confidence, len(data) - 1, loc=data.mean(), scale=stats.sem(data))
                st.write(f"### {confidence * 100:.0f}% Confidence Interval")
                st.write("Mean:", round(data.mean(), 4))
                st.write("Lower bound:", round(interval[0], 4))
                st.write("Upper bound:", round(interval[1], 4))
                success_box("This range estimates where the true population mean may be located.")
        else:
            st.warning("No numerical columns available.")

    elif test == "One-Sample T-Test":
        if numeric_cols:
            col = st.selectbox("Numerical column", numeric_cols, key="ttest_col")
            hypothesized_mean = st.number_input("Hypothesized mean", value=70.0, key="hyp_mean")
            data = working_df[col].dropna()

            if len(data) < 2:
                st.warning("At least 2 values are required.")
            else:
                stat, p = stats.ttest_1samp(data, hypothesized_mean)
                st.write("### One-Sample T-Test")
                st.write("T-statistic:", round(stat, 4))
                st.write("P-value:", round(p, 4))
                st.write("Interpretation:", p_value_interpretation(p))
        else:
            st.warning("No numerical columns available.")

    elif test == "Chi-Square Test":
        if len(categorical_cols) >= 2:
            col1 = st.selectbox("First categorical variable", categorical_cols, key="chi_col1")
            col2 = st.selectbox("Second categorical variable", categorical_cols, key="chi_col2")

            table = pd.crosstab(working_df[col1], working_df[col2])

            st.write("### Contingency Table")
            st.dataframe(table, use_container_width=True)

            stat, p, dof, expected = stats.chi2_contingency(table)

            st.write("### Test Result")
            st.write("Chi-square statistic:", round(stat, 4))
            st.write("P-value:", round(p, 4))
            st.write("Degrees of freedom:", dof)
            st.write("Interpretation:", p_value_interpretation(p))
        else:
            st.warning("At least two categorical columns are required.")

    elif test == "Pearson Correlation":
        if len(numeric_cols) >= 2:
            col1 = st.selectbox("First numerical variable", numeric_cols, key="pearson_col1")
            col2 = st.selectbox("Second numerical variable", numeric_cols, key="pearson_col2")

            pair_data = working_df[[col1, col2]].dropna()

            if len(pair_data) < 2:
                st.warning("At least 2 paired values are required.")
            else:
                corr, p = stats.pearsonr(pair_data[col1], pair_data[col2])

                st.write("### Pearson Correlation")
                st.write("Correlation coefficient:", round(corr, 4))
                st.write("P-value:", round(p, 4))

                if abs(corr) >= 0.7:
                    strength = "strong"
                elif abs(corr) >= 0.4:
                    strength = "moderate"
                else:
                    strength = "weak"

                direction = "positive" if corr >= 0 else "negative"
                success_box(f"The relationship is {strength} and {direction}.")

                fig = px.scatter(pair_data, x=col1, y=col2, trendline="ols", template="plotly_white")
                fig.update_layout(title=f"{col1} vs {col2}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("At least two numerical columns are required.")


# ============================================================
# PAGE: EXPORT CENTER
# ============================================================

elif page == "📦 Export Center":
    st.subheader("📦 Export Center")
    info_box("Download cleaned data, descriptive statistics, or an automatically generated Markdown analysis report.")

    c1, c2, c3 = st.columns(3)

    with c1:
        csv_data = working_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Cleaned Dataset",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            key="download_cleaned"
        )

    with c2:
        stats_table = descriptive_statistics(working_df)
        if not stats_table.empty:
            stats_csv = stats_table.to_csv(index=True).encode("utf-8")
            st.download_button(
                "⬇️ Download Statistics",
                data=stats_csv,
                file_name="descriptive_statistics.csv",
                mime="text/csv",
                key="download_stats"
            )
        else:
            st.button("No statistics available", disabled=True)

    with c3:
        report = generate_markdown_report(dataset_name, raw_df, working_df, recommendations)
        st.download_button(
            "⬇️ Download Report",
            data=report.encode("utf-8"),
            file_name="analysis_report.md",
            mime="text/markdown",
            key="download_report"
        )

    st.write("### Auto-Generated Report Preview")
    st.code(generate_markdown_report(dataset_name, raw_df, working_df, recommendations), language="markdown")
