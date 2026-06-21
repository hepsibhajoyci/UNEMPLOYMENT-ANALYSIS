# ==========================================================
# TASK 2: UNEMPLOYMENT ANALYSIS WITH PYTHON
# ==========================================================
# Author: Your Name
# Dataset: Unemployment in India.csv
# ==========================================================

# Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("Unemployment in India.csv")

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# ----------------------------------------------------------
# Data Cleaning
# ----------------------------------------------------------
df.dropna(inplace=True)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Rename columns for easy access
df.rename(columns={
    'Region': 'Region',
    'Date': 'Date',
    'Frequency': 'Frequency',
    'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate',
    'Area': 'Area'
}, inplace=True)

# Convert Date Column
df['Date'] = pd.to_datetime(df['Date'])

# Extract Month and Year
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

print("\n========== CLEANED DATA ==========")
print(df.head())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# ----------------------------------------------------------
# GRAPH 1 : Overall Unemployment Trend
# ----------------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(
    df['Date'],
    df['Unemployment_Rate'],
    marker='o'
)
plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# GRAPH 2 : Average Unemployment by State
# ----------------------------------------------------------
state_avg = df.groupby('Region')['Unemployment_Rate'].mean().sort_values(ascending=False)

plt.figure(figsize=(12,8))
sns.barplot(
    x=state_avg.values,
    y=state_avg.index
)

plt.title("Average Unemployment Rate by State")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# GRAPH 3 : COVID-19 Impact Analysis (2020)
# ----------------------------------------------------------
covid_df = df[df['Year'] == 2020]

plt.figure(figsize=(12,6))
sns.lineplot(
    data=covid_df,
    x='Month',
    y='Unemployment_Rate',
    marker='o'
)

plt.title("Covid-19 Impact on Unemployment in 2020")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# GRAPH 4 : Urban vs Rural Comparison
# ----------------------------------------------------------
plt.figure(figsize=(8,6))
sns.boxplot(
    x='Area',
    y='Unemployment_Rate',
    data=df
)

plt.title("Urban vs Rural Unemployment")
plt.xlabel("Area")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# GRAPH 5 : Labour Participation vs Unemployment
# ----------------------------------------------------------
plt.figure(figsize=(10,6))
sns.scatterplot(
    x='Labour_Participation_Rate',
    y='Unemployment_Rate',
    hue='Area',
    data=df
)

plt.title("Labour Participation vs Unemployment")
plt.xlabel("Labour Participation Rate (%)")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# GRAPH 6 : Monthly Average Unemployment
# ----------------------------------------------------------
monthly_avg = df.groupby('Month')['Unemployment_Rate'].mean()

plt.figure(figsize=(10,5))
sns.lineplot(
    x=monthly_avg.index,
    y=monthly_avg.values,
    marker='o'
)

plt.title("Monthly Average Unemployment Rate")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# GRAPH 7 : Correlation Heatmap
# ----------------------------------------------------------
numeric_data = df.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(8,6))
sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# TOP 5 STATES WITH HIGHEST UNEMPLOYMENT
# ----------------------------------------------------------
print("\n========== TOP 5 STATES WITH HIGHEST UNEMPLOYMENT ==========")

top_states = state_avg.head(5)
print(top_states)

# ----------------------------------------------------------
# TOP 5 STATES WITH LOWEST UNEMPLOYMENT
# ----------------------------------------------------------
print("\n========== TOP 5 STATES WITH LOWEST UNEMPLOYMENT ==========")

low_states = state_avg.tail(5)
print(low_states)

# ----------------------------------------------------------
# YEARLY UNEMPLOYMENT RATE
# ----------------------------------------------------------
print("\n========== YEARLY AVERAGE UNEMPLOYMENT ==========")

yearly_avg = df.groupby('Year')['Unemployment_Rate'].mean()
print(yearly_avg)

# ----------------------------------------------------------
# SAVE CLEANED DATASET
# ----------------------------------------------------------
df.to_csv("Cleaned_Unemployment_Data.csv", index=False)

print("\nCleaned dataset saved successfully.")
print("Project Completed Successfully!")
