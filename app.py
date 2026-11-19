
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="Pakistan Social Audience Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("🇵🇰 Pakistan Social Audience & Campaign Intelligence Engine")
st.write("AI Based Social Media Marketing Analytics Dashboard")

st.info("This project uses sample simulated data for demonstration.")

# Load datasets
@st.cache_data
def load_data():
    posts = pd.read_csv("data/posts.csv")
    comments = pd.read_csv("data/comments.csv")
    return posts, comments

try:
    posts, comments = load_data()
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()

# Sidebar
st.sidebar.title("Project Controls")

k = st.sidebar.slider(
    "Number of Audience Clusters",
    min_value=2,
    max_value=8,
    value=5
)

# Main metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Posts", len(posts))
col2.metric("Total Comments", len(comments))
col3.metric("Total Brands", posts["brand"].nunique())
col4.metric("Platforms", posts["platform"].nunique())

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Audience Clustering",
    "Content Analysis",
    "Unusual Patterns"
])

# Overview
with tab1:
    st.subheader("Social Media Overview")

    col1, col2 = st.columns(2)

    with col1:
        platform_data = posts.groupby(
            "platform"
        )["likes"].sum().reset_index()

        fig = px.bar(
            platform_data,
            x="platform",
            y="likes",
            title="Likes by Platform"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        brand_data = posts.groupby(
            "brand"
        )["comments_count"].sum().reset_index()

        fig = px.pie(
            brand_data,
            names="brand",
            values="comments_count",
            title="Comments by Brand"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Posts Dataset")
    st.dataframe(posts, use_container_width=True)

# Audience Clustering
with tab2:
    st.subheader("AI Audience Segmentation")

    if len(comments) <= k:
        st.error("Not enough comments for selected clusters.")
    else:
        text_data = comments["comment_text"].fillna("").astype(str)

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            stop_words="english"
        )

        X = vectorizer.fit_transform(text_data)

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        comments["Cluster"] = model.fit_predict(X)

        st.success("Audience clustering completed!")

        cluster_data = comments["Cluster"].value_counts().reset_index()
        cluster_data.columns = ["Cluster", "Comments"]

        fig = px.bar(
            cluster_data,
            x="Cluster",
            y="Comments",
            title="Audience Segments"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Clustered Comments")
        st.dataframe(comments, use_container_width=True)

# Content Analysis
with tab3:
    st.subheader("Campaign Content Performance")

    posts["engagement"] = (
        posts["likes"]
        + posts["comments_count"]
        + posts["shares"]
    )

    posts["engagement_rate"] = np.where(
        posts["views"] > 0,
        posts["engagement"] / posts["views"] * 100,
        0
    )

    content_data = posts.groupby(
        "content_type"
    )["engagement_rate"].mean().reset_index()

    fig = px.bar(
        content_data,
        x="content_type",
        y="engagement_rate",
        title="Engagement Rate by Content Type"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(content_data, use_container_width=True)

# Unusual Patterns
with tab4:
    st.subheader("Unusual Campaign Detection")

    features = posts[
        ["likes", "comments_count", "shares", "views"]
    ]

    detector = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    posts["anomaly"] = detector.fit_predict(
        np.log1p(features)
    )

    unusual = posts[posts["anomaly"] == -1]

    st.metric("Unusual Posts Detected", len(unusual))

    st.caption(
        "These are statistical outliers requiring further investigation, "
        "not confirmed suspicious activity."
    )

    st.dataframe(unusual, use_container_width=True)

st.divider()

st.subheader("Marketing Recommendations")

st.write("""
1. Analyze which content type gets higher engagement.
2. Understand customer price and delivery concerns.
3. Identify customer groups through K-Means clustering.
4. Investigate unusual campaign performance.
5. Improve future marketing campaigns using collected evidence.
""")

st.success("Pakistan Social Audience Intelligence Engine is running!")
