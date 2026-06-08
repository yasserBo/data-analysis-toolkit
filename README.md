# Student-Friendly Data Analysis Toolkit

## Overview

The Student-Friendly Data Analysis Toolkit is an interactive web-based toolkit designed to help users explore, clean, visualize, and statistically analyze datasets.

This project was created for a Data Analysis Toolkit assignment. The main goal is to provide a simple and reusable tool that allows users with limited programming experience to perform common data analysis tasks.

The toolkit is built using Python and Streamlit.

---

## Main Features

### Data Exploration

- Upload a CSV dataset
- Use a built-in sample dataset
- Preview the first rows
- View dataset shape
- Display column names
- Identify numerical and categorical columns
- Check duplicate rows
- Analyze missing values

### Data Cleaning

- Remove duplicate rows
- Drop missing rows
- Fill numerical missing values using the mean
- Fill numerical missing values using the median

### Descriptive Statistics

The toolkit calculates:

- Mean
- Median
- Minimum
- Maximum
- Variance
- Standard deviation
- Skewness
- Kurtosis

### Data Visualization

The toolkit includes:

- Histograms
- Boxplots
- Scatter plots
- Bar charts

### Probability Distributions

The toolkit supports:

- PDF visualization
- CDF visualization
- Normal distribution comparison
- Kolmogorov-Smirnov goodness-of-fit test

### Statistical Inference

The toolkit includes:

- Shapiro-Wilk normality test
- Confidence intervals
- One-sample t-test
- Chi-square test

### Export

Users can download the cleaned dataset as a CSV file.

---

## Tools and Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- SciPy
- GitHub

---

## Repository Structure

```text
data-analysis-toolkit/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── StudentsPerformance.csv
│
├── notebooks/
│   └── Data_Analysis_Toolkit.ipynb
│
├── docs/
│   └── user_guide.md
│
└── screenshots/
    ├── home_page.png
    ├── data_exploration.png
    ├── visualization.png
    └── statistical_tests.png
</> Markdown
