import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
import logging
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create cache directory
cache_dir = Path("./cache")
cache_dir.mkdir(exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="🎬 Anime Recommendation & Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #F3F4F6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE;
        border-bottom: 2px solid #2563EB;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Theme toggle
st.sidebar.markdown('<div style="text-align: center; padding: 12px 0;"><h2 style="color: #1E3A8A;">🎬 Anime Analytics</h2><p style="font-size: 0.9rem; color: #6B7280;">Advanced Anime Analysis & Recommendations</p><hr style="margin: 10px 0;"></div>', unsafe_allow_html=True)
theme = st.sidebar.selectbox("Interface Theme", ["Light", "Dark", "Modern Blue"])
if theme == "Dark":
    st.markdown("""
    <style>
    body { background-color: #111827; color: #F9FAFB; }
    .stApp { background-color: #111827; }
    .card { background-color: #1F2937; }
    .metric-card { background-color: #111827; border-left: 4px solid #3B82F6; }
    .main-header { color: #60A5FA; }
    .sub-header { color: #93C5FD; }
    .stTabs [data-baseweb="tab"] { background-color: #374151; }
    .stTabs [aria-selected="true"] { background-color: #1F2937; border-bottom: 2px solid #3B82F6; }
    </style>
    """, unsafe_allow_html=True)
elif theme == "Modern Blue":
    st.markdown("""
    <style>
    body { background-color: #F0F9FF; color: #0F172A; }
    .stApp { background-color: #F0F9FF; }
    .card { background-color: #FFFFFF; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); }
    .metric-card { background-color: #DBEAFE; border-left: 4px solid #2563EB; }
    .main-header { color: #1E40AF; }
    .sub-header { color: #1D4ED8; }
    .stTabs [data-baseweb="tab"] { background-color: #EFF6FF; }
    .stTabs [aria-selected="true"] { background-color: #BFDBFE; border-bottom: 2px solid #1D4ED8; }
    </style>
    """, unsafe_allow_html=True)

# Data loading and preprocessing
@st.cache_data(ttl=3600)
def load_data():
    try:
        start_time = time.time()
        processed_file = cache_dir / "processed_anime_data.pkl"
        
        # Clear cache to ensure fresh data
        if processed_file.exists():
            logger.info("Clearing existing cache to ensure fresh data")
            processed_file.unlink()

        # Load dataset
        data = pd.read_excel("anime.xlsx", sheet_name="anime")
        data.columns = [col.strip().replace(' ', '_') for col in data.columns]
        logger.info("Loaded raw dataset")

        # Clean HTML entities in name
        data['name'] = data['name'].apply(lambda x: re.sub(r"'", "'", str(x)))
        logger.info("Cleaned HTML entities in names")

        # Handle missing values
        genre_columns = ['genre.1', 'genre.2', 'genre.3', 'genre.4']
        for col in genre_columns:
            data[col] = data[col].fillna('')
        logger.info("Handled missing values in genre columns")

        # Create binary genre columns
        genres = set()
        for col in genre_columns:
            genres.update(data[col].unique())
        genres.discard('')
        for genre in genres:
            data[genre] = data[genre_columns].apply(lambda x: 1 if genre in x.values else 0, axis=1)
        logger.info(f"Created binary columns for {len(genres)} genres")

        # Convert episodes and members to numeric, handle non-numeric values
        data['episodes'] = pd.to_numeric(data['episodes'], errors='coerce').fillna(data['episodes'].median())
        data['members'] = pd.to_numeric(data['members'], errors='coerce').fillna(data['members'].median())
        logger.info("Converted episodes and members to numeric")

        # Ensure numerical columns are float
        data['rating'] = pd.to_numeric(data['rating'], errors='coerce').fillna(data['rating'].median())
        logger.info("Ensured numerical columns are float")

        # Create PopularityLevel column based on rating
        quantiles = data['rating'].quantile([0.33, 0.66])
        data['PopularityLevel'] = pd.cut(
            data['rating'],
            bins=[-float('inf'), quantiles[0.33], quantiles[0.66], float('inf')],
            labels=['Unpopular', 'Popular', 'Super Hit'],
            include_lowest=True
        )
        logger.info("Created PopularityLevel column based on rating")

        # Normalize numerical features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[['episodes', 'members']])
        scaled_df = pd.DataFrame(
            scaled_features,
            columns=['episodes_scaled', 'members_scaled'],
            index=data.index
        )
        data = pd.concat([data, scaled_df], axis=1)
        logger.info("Created scaled columns: episodes_scaled, members_scaled")

        # Verify scaled columns exist
        if not all(col in data.columns for col in ['episodes_scaled', 'members_scaled']):
            raise ValueError("Scaled columns were not created properly")

        data.to_pickle(processed_file)
        logger.info(f"Processed dataset with {len(data)} rows in {time.time() - start_time:.2f} seconds")
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Data loading failed: {str(e)}")
        return None

# Load data
with st.spinner('Loading and optimizing data...'):
    data = load_data()

if data is None or data.empty:
    st.markdown('<p class="error-message">No valid data available. Please check the dataset.</p>', unsafe_allow_html=True)
    st.stop()

# Sidebar filters
st.sidebar.markdown('<div class="sub-header">Filter Options</div>', unsafe_allow_html=True)
with st.sidebar.expander("Anime Filters", expanded=True):
    rating_range = st.slider("Rating Range", float(data['rating'].min()), float(data['rating'].max()), 
                           (float(data['rating'].min()), float(data['rating'].max())), step=0.1)
    types = st.multiselect("Anime Type", sorted(data['type'].unique()), default=data['type'].unique())
    popularity_levels = st.multiselect("Popularity Level", sorted(data['PopularityLevel'].unique()), 
                                     default=data['PopularityLevel'].unique())
with st.sidebar.expander("Genre Filters", expanded=True):
    genres = set()
    for col in ['genre.1', 'genre.2', 'genre.3', 'genre.4']:
        genres.update(data[col].unique())
    genres.discard('')
    selected_genres = st.multiselect("Genres", sorted(genres), default=[])
with st.sidebar.expander("Advanced Filters", expanded=False):
    episodes_range = st.slider("Episodes", float(data['episodes'].min()), float(data['episodes'].max()), 
                             (float(data['episodes'].min()), float(data['episodes'].max())), step=1.0)

if st.sidebar.button("Reset Filters", use_container_width=True):
    rating_range = (float(data['rating'].min()), float(data['rating'].max()))
    types = data['type'].unique()
    popularity_levels = data['PopularityLevel'].unique()
    selected_genres = []
    episodes_range = (float(data['episodes'].min()), float(data['episodes'].max()))

# Apply filters
if st.sidebar.button("Apply Filters", type="primary", use_container_width=True):
    with st.spinner('Applying filters...'):
        filtered_data = data[
            (data['rating'].between(rating_range[0], rating_range[1])) &
            (data['type'].isin(types)) &
            (data['PopularityLevel'].isin(popularity_levels)) &
            (data['episodes'].between(episodes_range[0], episodes_range[1]))
        ]
        if selected_genres:
            genre_condition = filtered_data[selected_genres].sum(axis=1) > 0
            filtered_data = filtered_data[genre_condition]
        if filtered_data.empty or len(filtered_data) < 50:
            st.markdown('<p class="error-message">No data matches the selected filters or insufficient data (less than 50 anime).</p>', unsafe_allow_html=True)
            st.session_state['filtered_data'] = data
        else:
            st.session_state['filtered_data'] = filtered_data
            train_model.clear()
            logger.info(f"Filtered dataset to {len(filtered_data)} rows")

filtered_data = st.session_state.get('filtered_data', data)

# Model preparation
@st.cache_resource
def train_model(data):
    try:
        start_time = time.time()
        logger.info("Starting model training...")

        # Validate input data
        if data.empty or len(data) < 50:
            raise ValueError("Insufficient data for training. At least 50 rows are required.")

        # Prepare feature columns
        feature_columns = ['members_scaled', 'episodes_scaled']
        genre_columns = [col for col in data.columns if col not in data.columns.difference(['genre.1', 'genre.2', 'genre.3', 'genre.4']) and col not in ['genre.1', 'genre.2', 'genre.3', 'genre.4', 'name', 'type', 'rating', 'episodes', 'members', 'PopularityLevel', 'episodes_scaled', 'members_scaled']]
        feature_columns.extend(genre_columns)

        # Combined validation
        missing_cols = [col for col in feature_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
        if data[feature_columns].isna().any().any() or data['PopularityLevel'].isna().any():
            raise ValueError("NaN values detected in feature columns or PopularityLevel")
        if 'PopularityLevel' not in data.columns:
            raise ValueError("PopularityLevel column is missing")

        # Create binary target for training
        binary_target = data['PopularityLevel'].apply(lambda x: 0 if x == 'Unpopular' else 1)
        unique_classes = np.unique(binary_target)
        if len(unique_classes) < 2:
            raise ValueError(f"Binary target has only one class: {unique_classes}. At least two classes are required.")

        # Prepare features and binary target
        X = data[feature_columns]
        le = LabelEncoder()
        y = le.fit_transform(binary_target)

        # Verify binary classes after encoding
        if len(np.unique(y)) != 2:
            raise ValueError(f"Encoded target has {len(np.unique(y))} classes instead of 2")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model with optimized parameters
        model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Feature importances
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        # SHAP values
        sample_size = min(100, len(X_test))
        X_test_sample = X_test.sample(sample_size, random_state=42) if sample_size < len(X_test) else X_test
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_sample)

        logger.info(f"Model trained in {time.time() - start_time:.2f} seconds")
        return model, X_train, X_test, y_train, y_test, y_pred, report, feature_importances, shap_values, le

    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        logger.error(f"Model training failed: {str(e)}", exc_info=True)
        return None, None, None, None, None, None, None, None, None, None

# Content-based recommendation system
def get_recommendations(anime_name, df, top_n=5):
    try:
        feature_columns = ['members_scaled', 'episodes_scaled'] + \
                         [col for col in df.columns if col in set(df[['genre.1', 'genre.2', 'genre.3', 'genre.4']].values.ravel()) and col != '']
        
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

        feature_matrix = df[feature_columns]
        similarity_matrix = cosine_similarity(feature_matrix)
        idx = df[df['name'] == anime_name].index[0]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in sim_scores[1:top_n+1]]
        recommendations = df.iloc[top_indices][['name', 'type', 'rating', 'PopularityLevel']]
        return recommendations.reset_index(drop=True)
    except IndexError:
        return "Anime not found in the dataset."
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        logger.error(f"Recommendation generation failed: {str(e)}")
        return "Error generating recommendations."

# Main page header
st.markdown('<h1 class="main-header">Anime Recommendation & Analytics</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
    <p style="text-align: center;">Explore insights, analysis, visualizations, and recommendations for anime using advanced machine learning. Ideal for anime enthusiasts and industry professionals.</p>
</div>
""", unsafe_allow_html=True)

# Insights section
st.markdown('<div class="sub-header">Insights</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Anime", f"{len(filtered_data):,}")
col2.metric("Average Rating", f"{filtered_data['rating'].mean():.2f}")
col3.metric("Median Episodes", f"{filtered_data['episodes'].median():.0f}")
col4.metric("Popular Anime (≥8.5)", f"{(filtered_data['rating'] >= 8.5).mean() * 100:.1f}%")

# Train model
if st.button("Train Model", type="primary"):
    with st.spinner('Training model...'):
        model, X_train, X_test, y_train, y_test, y_pred, report, feature_importances, shap_values, label_encoder = train_model(filtered_data)
        if model:
            st.session_state['model'] = model
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
            st.session_state['report'] = report
            st.session_state['feature_importances'] = feature_importances
            st.session_state['shap_values'] = shap_values
            st.session_state['label_encoder'] = label_encoder
            st.success("Model trained successfully!")
        else:
            st.error("Model training failed. Please check logs for details.")

# Tabs for analysis, predictions, and recommendations
if len(filtered_data) >= 50:
    tabs = st.tabs(["📊 Anime Overview", "🔮 Popularity Prediction", "📉 Model Performance", "💡 Advanced Insights", "🎬 Recommendations"])

    with tabs[0]:
        st.markdown('<div class="sub-header">📊 Anime Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Rating Distribution")
        chart_type = st.radio("Select Chart Type", ["Histogram", "Box Plot", "KDE"], horizontal=True)
        if chart_type == "Histogram":
            fig = px.histogram(filtered_data, x="rating", nbins=20, title="Rating Distribution")
            fig.update_layout(xaxis_title="Rating", yaxis_title="Count", height=500)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Box Plot":
            fig = px.box(filtered_data, y="rating", x="type", title="Rating by Anime Type")
            fig.update_layout(xaxis_title="Anime Type", yaxis_title="Rating", height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.kdeplot(filtered_data['rating'], ax=ax, fill=True, color="#2563EB")
            ax.set_title("Rating Density Distribution")
            ax.set_xlabel("Rating")
            ax.set_ylabel("Density")
            st.pyplot(fig)
            plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sub-header">Feature Analysis</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            feature = st.selectbox("Select Feature for Analysis", ['rating', 'episodes', 'members'])
            fig = px.box(filtered_data, x="PopularityLevel", y=feature, title=f"{feature.replace('_', ' ').title()} by Popularity")
            fig.update_layout(xaxis_title="Popularity Level", yaxis_title=feature.replace('_', ' ').title(), height=500)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Key Feature Insights**")
            high_popularity = filtered_data[filtered_data['PopularityLevel'].isin(['Popular', 'Super Hit'])]
            low_popularity = filtered_data[filtered_data['PopularityLevel'] == 'Unpopular']
            st.markdown(f"• Avg {feature.replace('_', ' ').title()} (High Popularity): {high_popularity[feature].mean():.2f}")
            st.markdown(f"• Avg {feature.replace('_', ' ').title()} (Low Popularity): {low_popularity[feature].mean():.2f}")
            st.markdown(f"• Range: {filtered_data[feature].min():.2f} - {filtered_data[feature].max():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="sub-header">🔮 Popularity Prediction</div>', unsafe_allow_html=True)
        if 'model' in st.session_state:
            st.markdown(f'<div class="card"><p>Using Gradient Boosting Classifier with Accuracy: {st.session_state["report"]["accuracy"]*100:.1f}%</p></div>', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Predict Anime Popularity")
                with st.form("anime_popularity_form"):
                    members = st.number_input("Members", min_value=0.0, max_value=float(data['members'].max()), value=float(filtered_data['members'].median()), step=1000.0)
                    episodes = st.number_input("Episodes", min_value=1.0, max_value=1000.0, value=float(filtered_data['episodes'].median()), step=1.0)
                    form_genres = st.multiselect("Genres", sorted(genres), default=[])
                    submit_button = st.form_submit_button("Predict Popularity")
                if submit_button and 'model' in st.session_state:
                    try:
                        scaler = StandardScaler()
                        numerical_features = scaler.fit_transform([[episodes, members]])
                        genre_features = [1 if genre in form_genres else 0 for genre in sorted(genres)]
                        sample = np.concatenate([numerical_features[0], genre_features])
                        model = st.session_state['model']
                        prediction = model.predict([sample])[0]
                        predicted_label = 'Popular/Super Hit' if prediction == 1 else 'Unpopular'
                        st.success(f"Predicted Popularity: {predicted_label}")
                        if predicted_label == 'Popular/Super Hit':
                            comparables = filtered_data[filtered_data['PopularityLevel'].isin(['Popular', 'Super Hit'])]
                        else:
                            comparables = filtered_data[filtered_data['PopularityLevel'] == 'Unpopular']
                        if not comparables.empty:
                            st.markdown("##### Similar Anime")
                            for _, anime in comparables.head(3).iterrows():
                                st.markdown(f"• {anime['name']}: Rating {anime['rating']:.2f}, {anime['type']}, {anime['episodes']} episodes")
                    except Exception as e:
                        st.error(f"Error predicting popularity: {str(e)}")
                        logger.error(f"Popularity prediction failed: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Popularity Drivers")
                if 'feature_importances' in st.session_state:
                    fig = px.bar(st.session_state['feature_importances'].head(10), x='Importance', y='Feature', orientation='h', title="Top Factors Affecting Popularity")
                    fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
                    st.plotly_chart(fig, use_container_width=True)
                if 'shap_values' in st.session_state:
                    st.markdown("#### SHAP Analysis")
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(st.session_state['shap_values'], features=st.session_state['X_test'], feature_names=st.session_state['X_test'].columns, plot_type="bar", show=False)
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error rendering SHAP plot: {str(e)}")
                        logger.error(f"SHAP plot rendering failed: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="sub-header">📉 Model Performance</div>', unsafe_allow_html=True)
        if 'report' in st.session_state:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{st.session_state['report']['accuracy']*100:.2f}%")
            with col2:
                st.metric("Macro Precision", f"{st.session_state['report']['macro avg']['precision']*100:.2f}%")
            with col3:
                st.metric("Macro Recall", f"{st.session_state['report']['macro avg']['recall']*100:.2f}%")
            with col4:
                st.metric("Macro F1-Score", f"{st.session_state['report']['macro avg']['f1-score']*100:.2f}%")
            st.markdown("#### Classification Report")
            report_df = pd.DataFrame(st.session_state['report']).T
            report_df = report_df[['precision', 'recall', 'f1-score', 'support']]
            report_df = report_df.apply(lambda x: x.round(2) if x.dtype == "float64" else x)
            st.dataframe(report_df, hide_index=False)
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
            fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"), title="Confusion Matrix")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### Prediction Distribution")
            try:
                pred_dist = pd.DataFrame({
                    'Actual': ['Unpopular' if x == 0 else 'Popular/Super Hit' for x in st.session_state['y_test']],
                    'Predicted': ['Unpopular' if x == 0 else 'Popular/Super Hit' for x in st.session_state['y_pred']]
                })
                fig = px.histogram(pred_dist, x=['Actual', 'Predicted'], barmode='overlay', title="Actual vs Predicted Popularity Distribution")
                fig.update_layout(xaxis_title="Popularity Level", yaxis_title="Count", height=500)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering prediction distribution: {str(e)}")
                logger.error(f"Prediction distribution rendering failed: {str(e)}")

    with tabs[3]:
        st.markdown('<div class="sub-header">💡 Advanced Insights</div>', unsafe_allow_html=True)
        st.markdown("#### Raw Data")
        st.dataframe(filtered_data[['name', 'type', 'rating', 'episodes', 'PopularityLevel']], hide_index=True, height=400)
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        csv = convert_df(filtered_data)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="anime_data.csv",
            mime="text/csv",
        )
        st.markdown("#### Data Summary")
        summary = filtered_data[['rating', 'episodes', 'members']].describe().T
        summary['count'] = summary['count'].astype(int)
        for col in ['mean', '50%', 'min', 'max']:
            if col in summary.columns:
                summary[col] = summary[col].apply(lambda x: f"{x:.2f}")
        st.dataframe(summary, height=300)
        st.markdown("#### Correlation Matrix")
        corr = filtered_data[['rating', 'episodes', 'members']].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Feature Correlation Matrix")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.markdown('<div class="sub-header">🎬 Recommendations</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Anime Recommendations")
        anime_name = st.selectbox("Select an Anime", sorted(filtered_data['name'].unique()))
        if st.button("Get Recommendations"):
            recommendations = get_recommendations(anime_name, filtered_data, top_n=5)
            if isinstance(recommendations, str):
                st.warning(recommendations)
            else:
                st.markdown(f"**Recommendations for {anime_name}**")
                st.dataframe(recommendations, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<p class="error-message">Insufficient data (less than 50 anime). Please adjust filters.</p>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <p style="color: #5c5c5c;"> Built with Streamlit | Data updated as of 10:16 AM IST, July 17, 2025</p>
</div>
""", unsafe_allow_html=True)    