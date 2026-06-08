import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Student-Friendly Data Analysis Toolkit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #1f4e79 0%, #2563eb 45%, #7c3aed 100%);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(31, 78, 121, 0.25);
    }

    .hero h1 {
        color: white;
        font-size: 2.4rem;
        margin-bottom: 0.4rem;
    }

    .hero p {
        color: #eef4ff;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .section-card {
        background-color: white;
        padding: 1.3rem;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.07);
        border: 1px solid #e8eef7;
        margin-bottom: 1rem;
    }

    .small-card {
        background-color: white;
        padding: 1rem;
        border-radius: 16px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        border: 1px solid #e8eef7;
        margin-bottom: 0.8rem;
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background-color: #e0ecff;
        color: #1d4ed8;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    .warning-box {
        background-color: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 1rem;
        border-radius: 12px;
        color: #7c2d12;
    }

    .success-box {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 1rem;
        border-radius: 12px;
        color: #064e3b;
    }

    .info-box {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 1rem;
        border-radius: 12px;
        color: #1e3a8a;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.7rem;
        font-weight: 800;
    }

    [data-testid="stSidebar"] {
        background-color: #0f172a;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stFileUploader label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: white !important;
    }

    hr {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Helper Functions
# ============================================================

@st.cache_data
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def find_sample_dataset() -> str | None:
    """
    Finds a sample CSV file in the data folder.
    Priority:
    1. data/StudentsPerformance.csv
    2. data/students_performance.csv
    3. first CSV found in data/
    """
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


def missing_value_analysis(df: pd.DataFrame) -> pd.DataFrame:
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    return pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values,
        "Missing Percentage": missing_percentage.round(2).values
    })


def get_column_types(df: pd.DataFrame):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()
    return numeric_columns, categorical_columns


def clean_data(df: pd.DataFrame, remove_duplicates: bool, missing_method: str) -> pd.DataFrame:
    cleaned = df.copy()

    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()

    numeric_columns, categorical_columns = get_column_types(cleaned)

    if missing_method == "Keep missing values":
        return cleaned

    if missing_method == "Drop rows with missing values":
        cleaned = cleaned.dropna()

    elif missing_method == "Fill numerical with mean + categorical with mode":
        for col in numeric_columns:
            cleaned[col] = cleaned[col].fillna(cleaned[col].mean())

        for col in categorical_columns:
            if cleaned[col].mode().empty:
                cleaned[col] = cleaned[col].fillna("Unknown")
            else:
                cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])

    elif missing_method == "Fill numerical with median + categorical with mode":
        for col in numeric_columns:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

        for col in categorical_columns:
            if cleaned[col].mode().empty:
                cleaned[col] = cleaned[col].fillna("Unknown")
            else:
                cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])

    return cleaned


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        return pd.DataFrame()

    summary = pd.DataFrame({
        "Mean": numeric_df.mean(),
        "Median": numeric_df.median(),
        "Minimum": numeric_df.min(),
        "Maximum": numeric_df.max(),
        "Variance": numeric_df.var(),
        "Standard Deviation": numeric_df.std(),
        "Skewness": numeric_df.skew(),
        "Kurtosis": numeric_df.kurtosis()
    })

    return summary.round(3)


def interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
    if p_value < alpha:
        return "Statistically significant result. Reject the null hypothesis."
    return "Not statistically significant. Fail to reject the null hypothesis."


def show_info_box(text: str):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def show_success_box(text: str):
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)


def show_warning_box(text: str):
    st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📊 Student-Friendly Data Analysis Toolkit</h1>
        <p>
        A modern interactive toolkit for data exploration, cleaning, visualization,
        probability distributions, and statistical inference.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <span class="badge">Python</span>
    <span class="badge">Streamlit</span>
    <span class="badge">Pandas</span>
    <span class="badge">Plotly</span>
    <span class="badge">SciPy</span>
    <span class="badge">Beginner Friendly</span>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("⚙️ Toolkit Control Panel")
st.sidebar.write("Upload a dataset or use the built-in sample dataset.")

dataset_choice = st.sidebar.radio(
    "Dataset source",
    ["Use sample dataset", "Upload your own CSV"],
    key="dataset_source"
)

df = None
dataset_name = None

if dataset_choice == "Use sample dataset":
    sample_path = find_sample_dataset()

    if sample_path is not None:
        df = load_csv_from_path(sample_path)
        dataset_name = Path(sample_path).name
        st.sidebar.success(f"Loaded: {dataset_name}")
    else:
        st.sidebar.error("No sample CSV found in the data folder.")

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="csv_uploader"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name
        st.sidebar.success(f"Uploaded: {dataset_name}")


st.sidebar.divider()
st.sidebar.subheader("🧹 Preprocessing Options")

remove_duplicates = st.sidebar.checkbox(
    "Remove duplicate rows",
    value=True,
    key="remove_duplicates"
)

missing_method = st.sidebar.selectbox(
    "Missing-value handling",
    [
        "Keep missing values",
        "Drop rows with missing values",
        "Fill numerical with mean + categorical with mode",
        "Fill numerical with median + categorical with mode"
    ],
    index=2,
    key="missing_method"
)

st.sidebar.divider()
st.sidebar.caption("Created by Yasser Bouyaddid")


# ============================================================
# Main App
# ============================================================

if df is None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("👋 Welcome")
    st.write(
        """
        Choose **Use sample dataset** from the sidebar or upload your own CSV file to begin.

        This toolkit helps you complete a full data analysis workflow without writing code manually.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


analysis_df = clean_data(df, remove_duplicates, missing_method)
numeric_columns, categorical_columns = get_column_types(analysis_df)

# Top metrics
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("Rows", f"{analysis_df.shape[0]:,}")

with metric_col2:
    st.metric("Columns", f"{analysis_df.shape[1]:,}")

with metric_col3:
    st.metric("Numeric Columns", len(numeric_columns))

with metric_col4:
    st.metric("Categorical Columns", len(categorical_columns))


st.markdown("---")

tab_overview, tab_cleaning, tab_stats, tab_viz, tab_dist, tab_tests, tab_export = st.tabs(
    [
        "🏠 Overview",
        "🧹 Cleaning",
        "📈 Statistics",
        "🎨 Visualization",
        "📊 Distributions",
        "🧪 Inference Tests",
        "⬇️ Export"
    ]
)


# ============================================================
# Overview Tab
# ============================================================

with tab_overview:
    st.subheader("Dataset Overview")

    col_a, col_b = st.columns([1.4, 1])

    with col_a:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.write(f"**Dataset:** {dataset_name}")
        st.write("Preview of the dataset:")
        st.dataframe(analysis_df.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.write("### Quick Health Check")
        st.write(f"Original rows: **{df.shape[0]:,}**")
        st.write(f"Current rows after preprocessing: **{analysis_df.shape[0]:,}**")
        st.write(f"Duplicate rows in original data: **{df.duplicated().sum():,}**")
        st.write(f"Total missing values in original data: **{df.isnull().sum().sum():,}**")

        if df.isnull().sum().sum() == 0:
            show_success_box("Great! The original dataset has no missing values.")
        else:
            show_warning_box("The original dataset contains missing values. Check the Cleaning tab.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Column Types")

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="small-card">', unsafe_allow_html=True)
        st.write("### Numerical Columns")
        if numeric_columns:
            st.write(numeric_columns)
        else:
            st.info("No numerical columns found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="small-card">', unsafe_allow_html=True)
        st.write("### Categorical Columns")
        if categorical_columns:
            st.write(categorical_columns)
        else:
            st.info("No categorical columns found.")
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# Cleaning Tab
# ============================================================

with tab_cleaning:
    st.subheader("Data Cleaning and Missing-Value Analysis")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write("Current preprocessing settings are controlled from the sidebar.")
    st.write(f"Remove duplicates: **{remove_duplicates}**")
    st.write(f"Missing-value method: **{missing_method}**")
    st.markdown('</div>', unsafe_allow_html=True)

    col_e, col_f = st.columns(2)

    with col_e:
        st.write("### Original Dataset Shape")
        st.write(df.shape)

    with col_f:
        st.write("### Cleaned Dataset Shape")
        st.write(analysis_df.shape)

    st.write("### Missing-Value Table")
    missing_table = missing_value_analysis(df)
    st.dataframe(missing_table, use_container_width=True)

    if missing_table["Missing Values"].sum() > 0:
        fig_missing = px.bar(
            missing_table,
            x="Column",
            y="Missing Percentage",
            title="Missing Values by Column (%)",
            text="Missing Percentage"
        )
        fig_missing.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_missing.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_missing, use_container_width=True)

    st.write("### Cleaned Data Preview")
    st.dataframe(analysis_df.head(10), use_container_width=True)


# ============================================================
# Statistics Tab
# ============================================================

with tab_stats:
    st.subheader("Descriptive Statistics")

    if not numeric_columns:
        st.warning("No numerical columns available for descriptive statistics.")
    else:
        stats_table = descriptive_statistics(analysis_df)
        st.dataframe(stats_table, use_container_width=True)

        selected_stat_col = st.selectbox(
            "Choose a numerical column for a detailed summary",
            numeric_columns,
            key="stats_detail_column"
        )

        series = analysis_df[selected_stat_col].dropna()

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

        with stat_col1:
            st.metric("Mean", round(series.mean(), 3))

        with stat_col2:
            st.metric("Median", round(series.median(), 3))

        with stat_col3:
            st.metric("Standard Deviation", round(series.std(), 3))

        with stat_col4:
            st.metric("Range", round(series.max() - series.min(), 3))

        show_info_box(
            "Descriptive statistics summarize the center, spread, and shape of numerical data."
        )


# ============================================================
# Visualization Tab
# ============================================================

with tab_viz:
    st.subheader("Interactive Data Visualization")

    plot_type = st.selectbox(
        "Choose visualization type",
        [
            "Histogram",
            "Boxplot",
            "Scatter Plot",
            "Bar Chart",
            "Correlation Heatmap"
        ],
        key="visualization_type"
    )

    if plot_type == "Histogram":
        if numeric_columns:
            column = st.selectbox(
                "Choose numerical column",
                numeric_columns,
                key="histogram_column"
            )

            fig = px.histogram(
                analysis_df,
                x=column,
                nbins=25,
                marginal="box",
                title=f"Histogram of {column}",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            show_info_box("A histogram shows the distribution of a numerical variable.")
        else:
            st.warning("No numerical columns available.")

    elif plot_type == "Boxplot":
        if numeric_columns:
            column = st.selectbox(
                "Choose numerical column",
                numeric_columns,
                key="boxplot_column"
            )

            group_by = st.selectbox(
                "Optional: group by categorical column",
                ["None"] + categorical_columns,
                key="boxplot_group_column"
            )

            if group_by == "None":
                fig = px.box(
                    analysis_df,
                    y=column,
                    points="outliers",
                    title=f"Boxplot of {column}",
                    template="plotly_white"
                )
            else:
                fig = px.box(
                    analysis_df,
                    x=group_by,
                    y=column,
                    points="outliers",
                    title=f"Boxplot of {column} by {group_by}",
                    template="plotly_white"
                )

            st.plotly_chart(fig, use_container_width=True)
            show_info_box("A boxplot shows median, spread, and possible outliers.")
        else:
            st.warning("No numerical columns available.")

    elif plot_type == "Scatter Plot":
        if len(numeric_columns) >= 2:
            x_column = st.selectbox(
                "Choose X column",
                numeric_columns,
                key="scatter_x_column"
            )

            y_column = st.selectbox(
                "Choose Y column",
                numeric_columns,
                key="scatter_y_column"
            )

            color_column = st.selectbox(
                "Optional: color by categorical column",
                ["None"] + categorical_columns,
                key="scatter_color_column"
            )

            if color_column == "None":
                fig = px.scatter(
                    analysis_df,
                    x=x_column,
                    y=y_column,
                    trendline="ols",
                    title=f"{x_column} vs {y_column}",
                    template="plotly_white"
                )
            else:
                fig = px.scatter(
                    analysis_df,
                    x=x_column,
                    y=y_column,
                    color=color_column,
                    trendline="ols",
                    title=f"{x_column} vs {y_column}",
                    template="plotly_white"
                )

            st.plotly_chart(fig, use_container_width=True)
            show_info_box("A scatter plot shows the relationship between two numerical variables.")
        else:
            st.warning("At least two numerical columns are required.")

    elif plot_type == "Bar Chart":
        if categorical_columns:
            column = st.selectbox(
                "Choose categorical column",
                categorical_columns,
                key="bar_chart_column"
            )

            counts = analysis_df[column].value_counts().reset_index()
            counts.columns = [column, "Count"]

            fig = px.bar(
                counts,
                x=column,
                y="Count",
                text="Count",
                title=f"Bar Chart of {column}",
                template="plotly_white"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_tickangle=-35)

            st.plotly_chart(fig, use_container_width=True)
            show_info_box("A bar chart shows how often each category appears.")
        else:
            st.warning("No categorical columns available.")

    elif plot_type == "Correlation Heatmap":
        if len(numeric_columns) >= 2:
            corr = analysis_df[numeric_columns].corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            show_info_box(
                "Correlation values close to 1 or -1 indicate stronger relationships between variables."
            )
        else:
            st.warning("At least two numerical columns are required.")


# ============================================================
# Probability Distributions Tab
# ============================================================

with tab_dist:
    st.subheader("Probability Distributions")

    if not numeric_columns:
        st.warning("No numerical columns available.")
    else:
        column = st.selectbox(
            "Choose numerical column for distribution analysis",
            numeric_columns,
            key="distribution_column"
        )

        data = analysis_df[column].dropna()

        if data.empty:
            st.warning("Selected column has no valid numerical values.")
        else:
            mean = data.mean()
            std = data.std()

            x = np.linspace(data.min(), data.max(), 200)

            if std == 0 or np.isnan(std):
                st.warning("Standard deviation is zero or undefined. Distribution curve cannot be calculated.")
            else:
                pdf = stats.norm.pdf(x, mean, std)
                cdf = stats.norm.cdf(x, mean, std)

                fig_pdf = go.Figure()
                fig_pdf.add_trace(
                    go.Histogram(
                        x=data,
                        histnorm="probability density",
                        name="Observed Data",
                        opacity=0.65
                    )
                )
                fig_pdf.add_trace(
                    go.Scatter(
                        x=x,
                        y=pdf,
                        mode="lines",
                        name="Fitted Normal PDF"
                    )
                )
                fig_pdf.update_layout(
                    title=f"PDF Comparison for {column}",
                    xaxis_title=column,
                    yaxis_title="Density",
                    template="plotly_white"
                )
                st.plotly_chart(fig_pdf, use_container_width=True)

                fig_cdf = go.Figure()
                fig_cdf.add_trace(
                    go.Scatter(
                        x=x,
                        y=cdf,
                        mode="lines",
                        name="Normal CDF"
                    )
                )
                fig_cdf.update_layout(
                    title=f"CDF for {column}",
                    xaxis_title=column,
                    yaxis_title="Cumulative Probability",
                    template="plotly_white"
                )
                st.plotly_chart(fig_cdf, use_container_width=True)

                ks_stat, ks_p = stats.kstest(data, "norm", args=(mean, std))

                st.write("### Goodness-of-Fit Test")
                st.write("Kolmogorov-Smirnov statistic:", round(ks_stat, 4))
                st.write("P-value:", round(ks_p, 4))

                if ks_p > 0.05:
                    show_success_box("The data may follow a normal distribution.")
                else:
                    show_warning_box("The data probably does not follow a normal distribution.")

                show_info_box(
                    "PDF shows where values are more likely to occur. CDF shows the probability of getting a value less than or equal to a specific value."
                )


# ============================================================
# Statistical Inference Tab
# ============================================================

with tab_tests:
    st.subheader("Statistical Inference Tests")

    test_type = st.selectbox(
        "Choose statistical test",
        [
            "Normality Test",
            "Confidence Interval",
            "One-Sample T-Test",
            "Chi-Square Test",
            "Pearson Correlation"
        ],
        key="test_type"
    )

    if test_type == "Normality Test":
        if numeric_columns:
            column = st.selectbox(
                "Choose numerical column",
                numeric_columns,
                key="normality_column"
            )

            data = analysis_df[column].dropna()

            if len(data) < 3:
                st.warning("At least 3 observations are required for the Shapiro-Wilk test.")
            else:
                stat, p_value = stats.shapiro(data)

                st.write("### Shapiro-Wilk Normality Test")
                st.write("Statistic:", round(stat, 4))
                st.write("P-value:", round(p_value, 4))

                if p_value > 0.05:
                    show_success_box("The data may be normally distributed.")
                else:
                    show_warning_box("The data does not appear to be normally distributed.")

        else:
            st.warning("No numerical columns available.")

    elif test_type == "Confidence Interval":
        if numeric_columns:
            column = st.selectbox(
                "Choose numerical column",
                numeric_columns,
                key="confidence_column"
            )

            confidence = st.slider(
                "Confidence level",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01,
                key="confidence_slider"
            )

            data = analysis_df[column].dropna()

            if len(data) < 2:
                st.warning("At least 2 observations are required.")
            else:
                mean = np.mean(data)
                sem = stats.sem(data)

                interval = stats.t.interval(
                    confidence,
                    len(data) - 1,
                    loc=mean,
                    scale=sem
                )

                st.write(f"### {confidence * 100:.0f}% Confidence Interval for {column}")
                st.write("Mean:", round(mean, 4))
                st.write("Lower bound:", round(interval[0], 4))
                st.write("Upper bound:", round(interval[1], 4))

                show_info_box(
                    "A confidence interval gives a likely range for the true population mean."
                )

        else:
            st.warning("No numerical columns available.")

    elif test_type == "One-Sample T-Test":
        if numeric_columns:
            column = st.selectbox(
                "Choose numerical column",
                numeric_columns,
                key="ttest_column"
            )

            population_mean = st.number_input(
                "Hypothesized population mean",
                value=70.0,
                key="population_mean"
            )

            data = analysis_df[column].dropna()

            if len(data) < 2:
                st.warning("At least 2 observations are required.")
            else:
                stat, p_value = stats.ttest_1samp(data, population_mean)

                st.write("### One-Sample T-Test")
                st.write("T-statistic:", round(stat, 4))
                st.write("P-value:", round(p_value, 4))
                st.write("Interpretation:", interpret_p_value(p_value))

        else:
            st.warning("No numerical columns available.")

    elif test_type == "Chi-Square Test":
        if len(categorical_columns) >= 2:
            column1 = st.selectbox(
                "Choose first categorical column",
                categorical_columns,
                key="chi_square_column1"
            )

            column2 = st.selectbox(
                "Choose second categorical column",
                categorical_columns,
                key="chi_square_column2"
            )

            table = pd.crosstab(analysis_df[column1], analysis_df[column2])

            if table.empty:
                st.warning("The contingency table is empty.")
            else:
                stat, p_value, dof, expected = stats.chi2_contingency(table)

                st.write("### Contingency Table")
                st.dataframe(table, use_container_width=True)

                st.write("### Chi-Square Result")
                st.write("Chi-square statistic:", round(stat, 4))
                st.write("P-value:", round(p_value, 4))
                st.write("Degrees of freedom:", dof)
                st.write("Interpretation:", interpret_p_value(p_value))

        else:
            st.warning("At least two categorical columns are required.")

    elif test_type == "Pearson Correlation":
        if len(numeric_columns) >= 2:
            x_col = st.selectbox(
                "Choose first numerical column",
                numeric_columns,
                key="pearson_x_column"
            )

            y_col = st.selectbox(
                "Choose second numerical column",
                numeric_columns,
                key="pearson_y_column"
            )

            valid_data = analysis_df[[x_col, y_col]].dropna()

            if len(valid_data) < 2:
                st.warning("At least 2 paired observations are required.")
            else:
                corr, p_value = stats.pearsonr(valid_data[x_col], valid_data[y_col])

                st.write("### Pearson Correlation Result")
                st.write("Correlation coefficient:", round(corr, 4))
                st.write("P-value:", round(p_value, 4))

                if abs(corr) >= 0.7:
                    strength = "strong"
                elif abs(corr) >= 0.4:
                    strength = "moderate"
                else:
                    strength = "weak"

                direction = "positive" if corr > 0 else "negative"

                st.write(f"Interpretation: There is a **{strength} {direction}** relationship.")

                fig = px.scatter(
                    valid_data,
                    x=x_col,
                    y=y_col,
                    trendline="ols",
                    title=f"Pearson Correlation: {x_col} vs {y_col}",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("At least two numerical columns are required.")


# ============================================================
# Export Tab
# ============================================================

with tab_export:
    st.subheader("Export Results")

    st.write("Download the cleaned/preprocessed dataset using the current sidebar settings.")

    csv_data = analysis_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Cleaned Dataset",
        data=csv_data,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
        key="download_cleaned_dataset"
    )

    stats_table = descriptive_statistics(analysis_df)

    if not stats_table.empty:
        stats_csv = stats_table.to_csv(index=True).encode("utf-8")

        st.download_button(
            label="⬇️ Download Descriptive Statistics",
            data=stats_csv,
            file_name="descriptive_statistics.csv",
            mime="text/csv",
            key="download_stats"
        )

    show_success_box("Export tools are ready. You can download your cleaned dataset and summary statistics.")
