
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Display the first few rows
print(df.head())

# Check data types and missing values
print(df.info())
print(df.isnull().sum())

# Clean the dataset (fill missing values if any)
df_cleaned = df.fillna(method='ffill')

# Basic statistics
print(df_cleaned.describe())

# Group by species (target) and compute mean
print(df_cleaned.groupby('target').mean())

# Map target values to species names
df_cleaned['species'] = df_cleaned['target'].map(dict(enumerate(iris.target_names)))

# Plot 1: Line Chart
plt.figure(figsize=(7, 5))
plt.plot(df_cleaned.index, df_cleaned['sepal length (cm)'], label='Sepal Length')
plt.title('Sepal Length Over Index')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.tight_layout()
plt.savefig("sepal_length_line_chart.png")
plt.close()

# Plot 2: Bar Chart
plt.figure(figsize=(7, 5))
sns.barplot(x='species', y='petal length (cm)', data=df_cleaned)
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.savefig("petal_length_bar_chart.png")
plt.close()

# Plot 3: Histogram
plt.figure(figsize=(7, 5))
plt.hist(df_cleaned['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("sepal_width_histogram.png")
plt.close()

# Plot 4: Scatter Plot
plt.figure(figsize=(7, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df_cleaned)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.savefig("sepal_vs_petal_scatter.png")
plt.close()
