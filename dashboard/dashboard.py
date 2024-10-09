import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Data
hour_df = pd.read_csv('https://raw.githubusercontent.com/karrengabriella/analisis-data_bike/main/data/hour.csv')
day_df = pd.read_csv('https://raw.githubusercontent.com/karrengabriella/analisis-data_bike/main/data/day.csv')

# Page Title
st.title("Bike Rental Analysis Dashboard")

# Section 1: Introduction and Data Preview
st.header("Data Preview")
st.write("""
This dashboard analyzes bike rental data by hours and days. 
We will explore rental trends by season, hour, and perform clustering analysis to uncover patterns. Below are previews of the data being used:
""")

st.subheader("Hourly Data")
st.dataframe(hour_df.head())

st.subheader("Daily Data")
st.dataframe(day_df.head())

# Function for data cleaning
def clear_data(df):
    df_cleaned = df.drop_duplicates()
    df_cleaned = df_cleaned.dropna()
    return df_cleaned

hour_df_cleaned = clear_data(hour_df)
day_df_cleaned = clear_data(day_df)

st.write("Data has been cleaned by removing duplicates and handling missing values.")

# Section 2: Descriptive Statistics
st.header("Descriptive Statistics")
st.write("""
Here, we present basic statistics such as mean, median, and standard deviation for the bike rental data. 
These statistics give us an overall idea of the distribution and variability in the dataset.
""")

st.subheader("Hourly Data Descriptive Statistics")
st.write(hour_df.describe())

st.subheader("Daily Data Descriptive Statistics")
st.write(day_df.describe())

# Section 3: Average Rentals by Season
st.header("Average Rentals by Season")
st.write("""
We analyze the average bike rentals per season to observe the influence of seasonal variations on rental trends.
The following bar chart illustrates the average rentals per season.
""")

hour_df['season'] = hour_df['season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})
season_avg = hour_df_cleaned.groupby('season')['cnt'].mean().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(x='season', y='cnt', data=season_avg, palette='Blues_d')
plt.title('Average Bike Rentals per Season')
plt.xlabel('Season')
plt.ylabel('Average Rentals')
st.pyplot(plt)

st.write("""
**Conclusion:** Summer and Fall seem to have higher average bike rentals compared to Winter and Spring. This could be due to better weather conditions, making biking more favorable during these seasons.
""")

# Section 4: Average Rentals by Hour
st.header("Average Rentals by Hour")
st.write("""
Next, we explore the bike rental patterns across different hours of the day to see at what time of the day the rentals peak.
""")

hour_avg = hour_df_cleaned.groupby('hr')['cnt'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='hr', y='cnt', data=hour_avg, palette='viridis')
plt.title('Average Bike Rentals by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Rentals')
st.pyplot(plt)

st.write("""
**Conclusion:** The peak hours for bike rentals appear to be during the morning and late afternoon, likely corresponding to commuting hours.
""")

# Section 5: Clustering Analysis
st.header("Clustering Analysis on Bike Rentals")
st.write("""
To further analyze the rental patterns, we perform clustering using KMeans to group similar rental patterns based on the hour and season.
This helps us uncover underlying patterns or groups in the rental behavior.
""")

X = hour_df_cleaned[['season', 'hr', 'cnt']]

# Scaling the data before clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
hour_df_cleaned['cluster'] = kmeans.fit_predict(X_scaled)

# Plot Clusters
cluster_avg = hour_df_cleaned.groupby(['hr', 'cluster'])['cnt'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(x='hr', y='cnt', hue='cluster', data=cluster_avg, palette='Set2')
plt.title('Average Rentals by Hour and Cluster')
plt.xlabel('Hour')
plt.ylabel('Average Rentals')
st.pyplot(plt)

st.write("""
**Conclusion:** The clustering analysis groups similar rental behaviors, showing distinct patterns during peak hours (morning and evening commutes) and off-peak hours. This information can help with resource allocation, such as ensuring there are enough bikes available during peak hours.
""")

# Section 6: Final Observations and Conclusion
st.header("Final Observations and Conclusion")
st.write("""
**Final Conclusion:**

From the analysis of the bike rental dataset, several key insights have emerged:

1. **Seasonal Influence:** 
   The data shows that bike rentals are strongly influenced by the season. **Summer** and **Fall** have the highest average rentals, likely due to favorable weather conditions that encourage biking. In contrast, **Winter** has the lowest rental numbers, likely due to harsher weather.

2. **Hourly Rental Patterns:**
   The rental behavior follows a clear pattern throughout the day. The peak rental times are during the **morning** (around 8 AM) and **late afternoon** (around 5-6 PM), corresponding to commuting hours. These peak hours indicate high demand during times when people are likely traveling to and from work or school.

3. **Clustering Analysis:**
   The KMeans clustering uncovered three distinct clusters of rental behavior. The analysis highlighted that different hours of the day show unique rental patterns:
   - **Cluster 1** represents peak commuting hours with the highest rentals.
   - **Cluster 2** captures midday and evening rentals.
   - **Cluster 3** groups the off-peak or late-night hours with minimal rentals.
   
   These clusters provide actionable insights for better **resource allocation**, such as deploying more bikes during peak hours and reducing inventory during off-peak times.

4. **Overall Patterns:**
   The data points to a cyclical pattern where **workday commuting hours** see a significant rise in rentals. The combination of hourly and seasonal analysis provides valuable information to optimize bike-sharing services, ensuring that there are enough bikes available when demand is highest.

**Business Implication:**
Bike rental services should focus on providing more availability and maintenance during the high-demand **morning and evening hours**, especially in the **Summer** and **Fall** seasons. They can also reduce fleet size during low-demand times like **Winter** and late-night hours to minimize costs.

This analysis provides an opportunity to improve **operational efficiency** and **customer satisfaction** by aligning resources with demand patterns.
""")