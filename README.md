<h1 align="center">🎬 Anime Recommendation & Analytics</h1>

---

# Overview 📊

This comprehensive project provides an end-to-end system for analyzing anime datasets, predicting anime popularity using machine learning, exploring insights via SQL and Python EDA, and visualizing trends in Power BI. It combines a Streamlit-based web application for advanced interactive analysis and recommendations with SQL queries for foundational exploration and a Power BI dashboard for high-level reporting. Ideal for anime enthusiasts, data analysts, and industry professionals seeking data-driven insights into anime trends and discoverability.

---

# Features ⚡

- 📈 Market Overview: Visualize rating distributions, popularity levels, genres and type distributions, and key metrics like average rating and median episodes
- 🔮 Popularity Predictions: Predict if an anime will be Unpopular or Popular/Super Hit using a Gradient Boosting Classifier
- 🧠 Exploratory Data Analysis: Python EDA with Seaborn, Matplotlib, Pandas for KDE plots, histograms, box plots, and correlation matrices and  Power BI dashboard with interactive slicers, line and bar charts, donut charts, and metrics cards
- 🗄️ SQL Analysis: Top 10 highest rated & most popular anime, Genre-wise average ratings, Most active users, Underrated high-quality anime detection and many more.
- 📊 Visualizations: Interactive histograms, box plots, KDE plots, SHAP explainability plots, and Power BI charts
- 💡 Advanced Insights: Explore raw data, summary stats, download filtered data as CSV, and inspect correlation heatmaps
- 🎛️ Custom Filters: Filter by rating range, anime type, popularity level, genres, episodes, and members
- 🎨 Theme Support: Choose Light, Dark, or Modern Blue themes in Streamlit for a tailored experience
- 🎬 Recommendations: Personalized content-based recommendations for similar anime based on normalized features and genres
 
---

# Technologies Used 🛠️

### 🐍 Python Libraries:
- Streamlit – Interactive web app framework
- Pandas, NumPy – Data manipulation, cleaning, feature creation
- Seaborn, Matplotlib – EDA plots (histograms, KDE, box plots)
- Plotly – Interactive charts (correlation, box, bar, heatmaps)
- Scikit-learn – Gradient Boosting Classifier, scaling, label encoding, metrics
- SHAP – Feature importance & explainability
- Logging, Pathlib, Re – Data processing and robust logging

### 🗄️ SQL:
- MySQL – Structured queries for data exploration
- Aggregations: AVG, COUNT, GROUP BY, ORDER BY
- Joins to combine anime & ratings data

### 📊 Power BI:
- Dashboard with cards, bar/line charts, pie/donut charts, slicers, and interactive visuals

### 📦 Machine Learning Models:
- Gradient Boosting Classifier (Scikit-learn) – for predicting popularity levels

### 🔄 Data Processing:
- `@st.cache_data`, `@st.cache_resource` – Efficient caching of processed data and trained models
- StandardScaler – Normalizes rating, episodes, members columns
- Binary genre features for similarity recommendations

---

# Installation 🧩

```bash
git clone https://github.com/YourUsername/anime-recommendation-analytics.git
cd anime-recommendation-analytics
```

---

# Data Requirements 📂

- Anime Dataset: anime.xlsx with a sheet named anime

- Key Columns: `name`, `type`, `genre.1`, `genre.2`, `genre.3`, `genre.4`, `rating`, `episodes`, `members`

### 🧠 Feature Engineering Includes:
- PopularityLevel – derived from quantiles of members into Unpopular, Popular, Super Hit

- rating_scaled, episodes_scaled, members_scaled – standardized numerical features

- Binary genre columns to enable content-based recommendations

---

# Usage Guide 🚀

 **Launch App**:  
  ```bash
 streamlit run app.py
  ```
 ### 🔎 Apply Filters

Use the sidebar to filter the dataset by:

🎯 Rating
📺 Anime Type (TV, Movie, OVA, etc.)
🎖 Popularity Level (Unpopular, Popular, Super Hit)
🎭 Genres
📚 Episodes
👥 Members

Click Apply Filters or Reset Filters to explore different segments.

### 🧭 Explore Tabs in Streamlit

- 📊 Anime Overview – Rating distributions via histograms, KDE plots, or box plots
- 🔮 Popularity Prediction – Input features to predict if an anime will be Unpopular or Popular/Super Hit
- 📉 Model Performance – Review accuracy, macro precision, recall, F1-score, confusion matrix, SHAP plots
- 💡 Advanced Insights – Explore raw data, data summaries, correlation heatmaps, download CSV
- 🎬 Recommendations – Select an anime and get top similar titles based on normalized features & genres

### 🧪 Train Models
Click the Train Model button to train the model on the filtered dataset. Results are cached for efficiency.

### ⚡ Power BI Dashboard

- View key metrics: total anime count, average ratings, member distributions
- See top 10 highest rated and most popular anime by members
- Analyze anime counts by type & genre via bar charts
- Slice by type, popularity level, rating, and genre
- Donut charts for overall popularity distribution

### 🗄️ SQL Queries

Executed in MySQL to build foundational insights:

Query Example	Insight

SELECT name, rating...	          Top 10 highest-rated anime
SELECT name, members...	          Top 10 most popular anime
SELECT type, COUNT(*)	            Anime counts by type
SELECT genre, AVG(rating)...	    Average ratings by genre
SELECT user_id, COUNT(*)...	      Most active users by rating count
SELECT name, rating, members...  	Detect underrated anime (high rating, low members)

---

# Notes 📝

- 📦 Caching
Processed dataset stored at: ./cache/processed_anime_data.pkl for faster reloads.

- 🔢 Training Requirements
Requires at least 50 filtered records to train the Gradient Boosting Classifier.

- 🎨 Themes
Switch between Light, Dark, and Modern Blue UI themes in Streamlit sidebar.

- 🧼 Error Handling
Includes robust logging & fallback alerts for missing columns or insufficient data.  

---

# Limitations ⚠️

i. Requires a properly structured anime.xlsx dataset.
ii. Machine learning accuracy depends on data quality & filter segments.
iii. SHAP computations can be slow on very large datasets.
iv. Recommendations are based on content similarity, not collaborative filtering.
v. No live updates from external anime APIs (yet).

---

# Future Improvements 🔮

- 🔄 Add collaborative filtering & hybrid recommender models
- 📊 More EDA visuals like scatter matrices and pair plots
- ⚙️ Include hyperparameter tuning & additional ML algorithms
- 🌐 Integrate live anime data via APIs (e.g., MyAnimeList, AniList)
- 💬 Add sentiment analysis from reviews
- 🔐 Implement user login to save preferences & personalized dashboards.
  
