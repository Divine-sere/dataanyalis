# dataanyalis
# Iris Dataset Analysis

This project analyzes the famous Iris dataset using Python. It involves data loading, cleaning, basic analysis, and creating visualizations using `pandas`, `matplotlib`, and `seaborn`.

## Files Included

- `data.py`: Python script with all analysis and visualization code
- `sepal_length_line_chart.png`: Line chart of sepal length over index
- `sepal_width_histogram.png`: Histogram of sepal width.
- `sepal_vs_petal_scatter.png`: Scatter plot comparing sepal and petal lengths.

## Steps Performed

1. **Data Loading**: Used `sklearn.datasets.load_iris()` to load the dataset.
2. **Data Exploration**: Displayed structure, checked for missing values, and ensured clean data.
3. **Basic Analysis**: Calculated descriptive statistics and grouped data by species to find insights.
4. **Visualizations**:
   - Line chart showing sepal length trend.
   - Bar chart comparing petal lengths by species.
   - Histogram for sepal width distribution.
   - Scatter plot visualizing correlation between sepal and petal lengths.

## How to Run

1. Make sure you have `pandas`, `matplotlib`, `seaborn`, and `sklearn` installed.
2. Run the script with:

```bash
python iris_data_analysis.py
