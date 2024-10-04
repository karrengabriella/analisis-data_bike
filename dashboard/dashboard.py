import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Step 1: Memuat dataset dari URL raw dari GitHub
hour_df = pd.read_csv('https://raw.githubusercontent.com/karrengabriella/analisis-data_bike/main/data/hour.csv')
day_df = pd.read_csv('https://raw.githubusercontent.com/karrengabriella/analisis-data_bike/main/data/day.csv')

# Step 2: Ekstraksi Fitur Tambahan
hour_df['datetime'] = pd.to_datetime(hour_df['dteday']) + pd.to_timedelta(hour_df['hr'], unit='h')
hour_df['year'] = hour_df['datetime'].dt.year
hour_df['month'] = hour_df['datetime'].dt.month
hour_df['day'] = hour_df['datetime'].dt.day
hour_df['hour'] = hour_df['datetime'].dt.hour
hour_df['season'] = hour_df['season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})

day_df['datetime'] = pd.to_datetime(day_df['dteday'])
day_df['year'] = day_df['datetime'].dt.year
day_df['month'] = day_df['datetime'].dt.month
day_df['season'] = day_df['season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})

# Step 3: Membuat Dashboard
st.title("Bike Rental Analysis")

# Deskripsi umum
st.markdown("""
    **Analisis Data Penyewaan Sepeda**
    
    Dashboard ini menampilkan hasil analisis data penyewaan sepeda berdasarkan dataset penyewaan sepeda selama beberapa waktu.
    Data mencakup penyewaan berdasarkan musim, waktu, dan kondisi cuaca. Tujuan dari analisis ini adalah untuk memahami pola
    penggunaan layanan penyewaan sepeda dan faktor-faktor apa yang memengaruhi tingkat penyewaan.
""")

# Visualisasi 1: Jumlah Penyewaan Berdasarkan Musim (hour_df)
st.header("Jumlah Penyewaan Berdasarkan Musim (Hour Data)")
st.markdown("""
    **Mengapa analisis ini?**
    
    Penyewaan sepeda bisa sangat dipengaruhi oleh musim. Melihat pola penyewaan berdasarkan musim dapat membantu kita
    memahami kapan layanan lebih banyak digunakan, sehingga perusahaan dapat menyesuaikan operasionalnya.
""")
season_counts_hour = hour_df.groupby('season')['cnt'].sum().reset_index()
fig1, ax1 = plt.subplots()
sns.barplot(x='season', y='cnt', data=season_counts_hour, palette='viridis', ax=ax1)
ax1.set_title('Jumlah Penyewaan Berdasarkan Musim')
ax1.set_xlabel('Musim')
ax1.set_ylabel('Jumlah Penyewaan')
st.pyplot(fig1)

# Visualisasi 2: Jumlah Penyewaan Berdasarkan Jam (hour_df)
st.header("Jumlah Penyewaan Berdasarkan Jam")
st.markdown("""
    **Mengapa?**
    
    Menganalisis jumlah penyewaan berdasarkan jam sangat penting untuk memahami kapan puncak penggunaan layanan terjadi.
    Hal ini penting bagi perusahaan untuk menentukan jam operasional dan alokasi sumber daya.
""")
hour_counts = hour_df.groupby('hour')['cnt'].sum().reset_index()
fig2, ax2 = plt.subplots()
sns.lineplot(x='hour', y='cnt', data=hour_counts, marker='o', color='b', ax=ax2)
ax2.set_title('Jumlah Penyewaan Berdasarkan Jam')
ax2.set_xlabel('Jam')
ax2.set_ylabel('Jumlah Penyewaan')
ax2.grid(True)
st.pyplot(fig2)

# Visualisasi 3: Jumlah Penyewaan Berdasarkan Cuaca (day_df)
st.header("Jumlah Penyewaan Berdasarkan Cuaca (Day Data)")
st.markdown("""
    **Mengapa?**
    
    Cuaca sangat mempengaruhi preferensi orang dalam menggunakan sepeda. Dalam cuaca cerah, penyewaan lebih tinggi,
    sementara cuaca buruk bisa mengurangi jumlah penyewaan. Memahami pola ini bisa membantu dalam perencanaan strategis.
""")
weather_counts_day = day_df.groupby('weathersit')['cnt'].sum().reset_index()
weather_map_day = {1: 'Clear, Few clouds', 2: 'Mist + Cloudy', 3: 'Light Snow/Rain', 4: 'Heavy Rain/Snow'}
weather_counts_day['weathersit'] = weather_counts_day['weathersit'].map(weather_map_day)
fig3, ax3 = plt.subplots()
sns.barplot(x='weathersit', y='cnt', data=weather_counts_day, palette='plasma', ax=ax3)
ax3.set_title('Jumlah Penyewaan Berdasarkan Cuaca')
ax3.set_xlabel('Cuaca')
ax3.set_ylabel('Jumlah Penyewaan')
st.pyplot(fig3)

# Statistik Deskriptif
st.header("Statistik Deskriptif Data Hour")
st.markdown("""
    **Deskripsi Analisis Statistik**
    
    Statistik deskriptif memberikan gambaran umum tentang distribusi data. Ini membantu memahami seberapa besar variasi dalam
    data penyewaan sepeda dan menentukan apakah ada anomali yang perlu dianalisis lebih lanjut.
""")
st.write(hour_df.describe())

st.header("Statistik Deskriptif Data Day")
st.write(day_df.describe())
