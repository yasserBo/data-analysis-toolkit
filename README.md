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
- Preview the first rows of the dataset
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
├── Data_Analysis_Toolkit.ipynb
│
├── docs/
│   └── user_guide.md
│
└── screenshots/
    └── README.md
```

---

## Sample Dataset

The sample dataset used in this toolkit is the Students Performance in Exams dataset.

The dataset contains student performance information, including:

- gender
- race/ethnicity
- parental level of education
- lunch
- test preparation course
- math score
- reading score
- writing score

Dataset source:

```text
https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
```

---

## How to Use the Toolkit

1. Open the toolkit.
2. Choose one of the dataset options:
   - Use the built-in sample dataset
   - Upload your own CSV file
3. Explore the dataset using the Data Exploration section.
4. Clean the dataset using the Cleaning section.
5. View descriptive statistics.
6. Generate visualizations such as histograms, boxplots, scatter plots, and bar charts.
7. Analyze probability distributions using PDF and CDF plots.
8. Run statistical inference tests.
9. Export the cleaned dataset as a CSV file.

---

## How to Run the Toolkit Locally

If users want to run the toolkit on their own computer, they can follow these steps:

1. Download or clone this repository.

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Open the local URL shown in the terminal.

---

## Example Analysis

Using the Students Performance dataset, this toolkit can be used to:

- Explore the structure of the dataset
- Check whether there are missing values
- Clean the dataset by removing duplicates
- Calculate descriptive statistics for math, reading, and writing scores
- Create histograms to study score distributions
- Create boxplots to detect possible outliers
- Compare two numerical variables using scatter plots
- Analyze PDF and CDF curves for exam scores
- Test whether a score variable follows a normal distribution
- Calculate confidence intervals for average scores
- Perform a one-sample t-test
- Perform a chi-square test between categorical variables

---

## Screenshots

Screenshots of the toolkit interface and example outputs can be added in the `screenshots/` folder.

Recommended screenshots:

- Home page
- Data exploration section
- Visualization section
- Probability distribution section
- Statistical inference section

---

## Project Purpose

This toolkit was designed to make data analysis easier and more accessible for beginners.

Instead of writing code manually, users can interact with a simple web interface to perform common data analysis tasks.

---

## Author

Yasser Bouyaddid  
Computer Science Student  
SRH Leipzig University of Applied Sciences
