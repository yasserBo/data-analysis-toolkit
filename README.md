# Student-Friendly Data Analysis Toolkit

## Overview

The Student-Friendly Data Analysis Toolkit is a beginner-friendly Python toolkit designed to help users explore, clean, visualize, and statistically analyze datasets.

This project was created as part of a Data Analysis Toolkit assignment. The goal is to provide a reusable and easy-to-use toolkit for common data analysis tasks.

The toolkit is especially useful for students and beginners who have limited programming experience but want to perform basic data analysis efficiently.

---

## Main Features

### 1. Data Exploration

- Load CSV datasets
- Preview the first rows of the dataset
- Display dataset shape
- Show column names
- Show data types
- Display general dataset information

### 2. Data Cleaning and Preprocessing

- Detect missing values
- Calculate missing-value percentages
- Remove duplicate rows
- Handle missing values using:
  - Drop method
  - Mean replacement
  - Median replacement

### 3. Descriptive Statistics

The toolkit calculates:

- Mean
- Median
- Mode
- Minimum
- Maximum
- Variance
- Standard deviation
- Skewness
- Kurtosis

### 4. Data Visualization

The toolkit includes common visualizations such as:

- Histograms
- Boxplots
- Scatter plots
- Bar charts

### 5. Probability Distributions

The toolkit supports probability distribution analysis using:

- PDF: Probability Density Function
- CDF: Cumulative Distribution Function
- Normal distribution visualization

### 6. Statistical Inference

The toolkit includes:

- Normality testing
- Confidence intervals
- One-sample t-test
- Chi-square test

### 7. Export Results

Users can export the cleaned dataset as a CSV file.

---

## Tools and Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Jupyter Notebook
- Google Colab
- GitHub

---

## Repository Structure

```text
data-analysis-toolkit/
│
├── README.md
├── Data_Analysis_Toolkit.ipynb
├── requirements.txt
│
├── data/
│   └── students_performance.csv
│
├── docs/
│   └── user_guide.md
│
└── screenshots/
    └── README.md
