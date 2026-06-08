import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


st.set_page_config(
    page_title="Student-Friendly Data Analysis Toolkit",
    layout="wide"
)


st.title("Student-Friendly Data Analysis Toolkit")

st.write("""
This toolkit helps users explore, clean, visualize, and statistically analyze datasets.
It is designed for students and beginners with limited programming experience.
""")


# -----------------------------
# Helper Functions
# -----------------------------

def load_sample_data():
    return pd.read_csv("data/StudentsPerformance.csv")


def missing_value_analysis(df):
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    result = pd.DataFrame({
        "Missing Values": missing_values,
        "Missing Percentage": missing_percentage
    })

    return result


def clean_data(df, method):
    df_cleaned = df.copy()
    df_cleaned = df_cleaned.drop_duplicates()

    if method == "Drop missing rows":
        df_cleaned = df_cleaned.dropna()

    elif method == "Fill numerical missing values with mean":
        numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())

    elif method == "Fill numerical missing values with median":
        numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())

    return df_cleaned


def descriptive_statistics(df):
    numeric_df = df.select_dtypes(include=np.number)

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

    return summary


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("Dataset Options")

dataset_choice = st.sidebar.radio(
    "Choose dataset source:",
    ["Use sample dataset", "Upload your own CSV"]
)

df = None

if dataset_choice == "Use sample dataset":
    try:
        df = load_sample_data()
        st.sidebar.success("Sample dataset loaded successfully.")
    except Exception:
        st.sidebar.error("Sample dataset not found. Please upload data/students_performance.csv.")

else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Dataset uploaded successfully.")
    else:
        st.info("Please upload a CSV file to start.")

# Streamlit has a built-in file uploader widget for user file uploads. :contentReference[oaicite:1]{index=1}


if df is not None:

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Data Exploration",
        "Cleaning",
        "Descriptive Statistics",
        "Visualization",
        "Probability Distributions",
        "Statistical Inference",
        "Export"
    ])

    # -----------------------------
    # Tab 1: Data Exploration
    # -----------------------------
    with tab1:
        st.header("Data Exploration")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric("Duplicate Rows", df.duplicated().sum())

        st.subheader("Column Names")
        st.write(df.columns.tolist())

        st.subheader("Data Types")
        st.dataframe(df.dtypes.astype(str).reset_index().rename(columns={
            "index": "Column",
            0: "Data Type"
        }))

        st.subheader("Missing-Value Analysis")
        st.dataframe(missing_value_analysis(df))

        st.subheader("Numerical Columns")
        st.write(numeric_columns)

        st.subheader("Categorical Columns")
        st.write(categorical_columns)

    # -----------------------------
    # Tab 2: Cleaning
    # -----------------------------
    with tab2:
        st.header("Data Cleaning and Preprocessing")

        cleaning_method = st.selectbox(
            "Choose a cleaning method:",
            [
                "Drop missing rows",
                "Fill numerical missing values with mean",
                "Fill numerical missing values with median"
            ]
        )

        df_cleaned = clean_data(df, cleaning_method)

        st.success("Data cleaned successfully.")

        st.write("Original dataset shape:", df.shape)
        st.write("Cleaned dataset shape:", df_cleaned.shape)

        st.subheader("Cleaned Dataset Preview")
        st.dataframe(df_cleaned.head())

    # -----------------------------
    # Tab 3: Descriptive Statistics
    # -----------------------------
    with tab3:
        st.header("Descriptive Statistics")

        if len(numeric_columns) > 0:
            st.dataframe(descriptive_statistics(df))
            st.write("""
            These statistics summarize the numerical variables in the dataset.
            They help users understand the center, spread, and shape of the data.
            """)
        else:
            st.warning("No numerical columns found.")

    # -----------------------------
    # Tab 4: Visualization
    # -----------------------------
    with tab4:
        st.header("Data Visualization")

        plot_type = st.selectbox(
            "Choose plot type:",
            ["Histogram", "Boxplot", "Scatter Plot", "Bar Chart"]
        )

        if plot_type == "Histogram":
            if len(numeric_columns) > 0:
                column = st.selectbox("Choose numerical column:", numeric_columns)

                fig, ax = plt.subplots()
                ax.hist(df[column].dropna(), bins=20)
                ax.set_title(f"Histogram of {column}")
                ax.set_xlabel(column)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

                st.write("A histogram shows the distribution of a numerical variable.")
            else:
                st.warning("No numerical columns available.")

        elif plot_type == "Boxplot":
            if len(numeric_columns) > 0:
                column = st.selectbox("Choose numerical column:", numeric_columns)

                fig, ax = plt.subplots()
                ax.boxplot(df[column].dropna(), vert=False)
                ax.set_title(f"Boxplot of {column}")
                ax.set_xlabel(column)
                st.pyplot(fig)

                st.write("A boxplot helps identify spread, median, and possible outliers.")
            else:
                st.warning("No numerical columns available.")

        elif plot_type == "Scatter Plot":
            if len(numeric_columns) >= 2:
                x_column = st.selectbox("Choose X column:", numeric_columns)
                y_column = st.selectbox("Choose Y column:", numeric_columns)

                fig, ax = plt.subplots()
                ax.scatter(df[x_column], df[y_column])
                ax.set_title(f"{x_column} vs {y_column}")
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                st.pyplot(fig)

                st.write("A scatter plot shows the relationship between two numerical variables.")
            else:
                st.warning("At least two numerical columns are required.")

        elif plot_type == "Bar Chart":
            if len(categorical_columns) > 0:
                column = st.selectbox("Choose categorical column:", categorical_columns)

                counts = df[column].value_counts()

                fig, ax = plt.subplots()
                ax.bar(counts.index.astype(str), counts.values)
                ax.set_title(f"Bar Chart of {column}")
                ax.set_xlabel(column)
                ax.set_ylabel("Count")
                plt.xticks(rotation=45)
                st.pyplot(fig)

                st.write("A bar chart shows the frequency of each category.")
            else:
                st.warning("No categorical columns available.")

    # -----------------------------
    # Tab 5: Probability Distributions
    # -----------------------------
    with tab5:
        st.header("Probability Distributions")

        if len(numeric_columns) > 0:
            column = st.selectbox("Choose numerical column for distribution analysis:", numeric_columns)

            data = df[column].dropna()
            mean = data.mean()
            std = data.std()

            x = np.linspace(data.min(), data.max(), 100)
            pdf = stats.norm.pdf(x, mean, std)
            cdf = stats.norm.cdf(x, mean, std)

            st.subheader("Probability Density Function")
            fig1, ax1 = plt.subplots()
            ax1.plot(x, pdf)
            ax1.set_title(f"PDF of {column}")
            ax1.set_xlabel(column)
            ax1.set_ylabel("Probability Density")
            st.pyplot(fig1)

            st.write("""
            The PDF shows where values are more likely to occur.
            Higher parts of the curve mean values are more common around that area.
            """)

            st.subheader("Cumulative Distribution Function")
            fig2, ax2 = plt.subplots()
            ax2.plot(x, cdf)
            ax2.set_title(f"CDF of {column}")
            ax2.set_xlabel(column)
            ax2.set_ylabel("Cumulative Probability")
            st.pyplot(fig2)

            st.write("""
            The CDF shows the probability of getting a value less than or equal to a certain point.
            """)

            st.subheader("Goodness-of-Fit: Kolmogorov-Smirnov Test")

            ks_stat, ks_p_value = stats.kstest(data, "norm", args=(mean, std))

            st.write("KS Statistic:", ks_stat)
            st.write("P-value:", ks_p_value)

            if ks_p_value > 0.05:
                st.success("Interpretation: The data may follow a normal distribution.")
            else:
                st.warning("Interpretation: The data probably does not follow a normal distribution.")

        else:
            st.warning("No numerical columns available.")

    # -----------------------------
    # Tab 6: Statistical Inference
    # -----------------------------
    with tab6:
        st.header("Statistical Inference")

        test_type = st.selectbox(
            "Choose statistical test:",
            [
                "Normality Test",
                "Confidence Interval",
                "One-Sample T-Test",
                "Chi-Square Test"
            ]
        )

        if test_type == "Normality Test":
            if len(numeric_columns) > 0:
                column = st.selectbox("Choose numerical column:", numeric_columns)

                data = df[column].dropna()
                stat, p_value = stats.shapiro(data)

                st.write("Shapiro-Wilk Statistic:", stat)
                st.write("P-value:", p_value)

                if p_value > 0.05:
                    st.success("Interpretation: The data may be normally distributed.")
                else:
                    st.warning("Interpretation: The data does not appear to be normally distributed.")
            else:
                st.warning("No numerical columns available.")

        elif test_type == "Confidence Interval":
            if len(numeric_columns) > 0:
                column = st.selectbox("Choose numerical column:", numeric_columns)
                confidence = st.slider("Confidence level:", 0.80, 0.99, 0.95)

                data = df[column].dropna()
                mean = np.mean(data)
                sem = stats.sem(data)

                interval = stats.t.interval(
                    confidence,
                    len(data) - 1,
                    loc=mean,
                    scale=sem
                )

                st.write(f"{confidence * 100:.0f}% Confidence Interval:")
                st.write(interval)

                st.write("""
                A confidence interval gives a likely range for the true population mean.
                """)
            else:
                st.warning("No numerical columns available.")

        elif test_type == "One-Sample T-Test":
            if len(numeric_columns) > 0:
                column = st.selectbox("Choose numerical column:", numeric_columns)
                population_mean = st.number_input("Enter hypothesized population mean:", value=70.0)

                data = df[column].dropna()
                stat, p_value = stats.ttest_1samp(data, population_mean)

                st.write("T-statistic:", stat)
                st.write("P-value:", p_value)

                if p_value < 0.05:
                    st.success("Interpretation: Reject the null hypothesis.")
                else:
                    st.info("Interpretation: Fail to reject the null hypothesis.")
            else:
                st.warning("No numerical columns available.")

        elif test_type == "Chi-Square Test":
            if len(categorical_columns) >= 2:
                column1 = st.selectbox("Choose first categorical column:", categorical_columns)
                column2 = st.selectbox("Choose second categorical column:", categorical_columns)

                table = pd.crosstab(df[column1], df[column2])
                stat, p_value, dof, expected = stats.chi2_contingency(table)

                st.subheader("Contingency Table")
                st.dataframe(table)

                st.write("Chi-square statistic:", stat)
                st.write("P-value:", p_value)
                st.write("Degrees of freedom:", dof)

                if p_value < 0.05:
                    st.success("Interpretation: There is a significant relationship between the two variables.")
                else:
                    st.info("Interpretation: There is no significant relationship between the two variables.")
            else:
                st.warning("At least two categorical columns are required.")

    # -----------------------------
    # Tab 7: Export
    # -----------------------------
    with tab7:
        st.header("Export Results")

        cleaned_export = clean_data(df, "Fill numerical missing values with mean")

        csv = cleaned_export.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Cleaned Dataset",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

        st.write("Users can download the cleaned dataset as a CSV file.")
