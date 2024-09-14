from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import Birch, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from collections import Counter
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import time

df = pd.read_csv("C:/Users/Windows10/Downloads/processed_data.csv")

st.title("Amazon Sales")
st.header("Enter a Word or Phrase")
user_input = st.text_input("Enter a word or phrase:")

if 'cluster_labels' not in st.session_state:
    st.session_state['cluster_labels'] = None
if 'pca_transformed_data' not in st.session_state:
    st.session_state['pca_transformed_data'] = None
if 'clustering_model' not in st.session_state:
    st.session_state['clustering_model'] = None
if 'tfidf_vect' not in st.session_state:
    st.session_state['tfidf_vect'] = None

if user_input:
    st.header("Clustering Model")
    model_options = ["Birch", "Spectral", "Hierarchical"]
    selected_model = st.selectbox("Select a Clustering Model", model_options)

    if selected_model == "Birch":
        st.header("Birch Parameters")
        threshold = st.slider("Threshold", 0.0, 0.3, 0.05)
    elif selected_model == "Spectral":
        st.header("Spectral Parameters")
        n_clusters = st.selectbox("Number of Clusters", [2, 3])
        eigen_solver = st.selectbox("Eigen Solver", ["arpack", "lobpcg"])
    elif selected_model == "Hierarchical":
        st.header("Hierarchical Parameters")
        n_clusters = st.selectbox("Number of Clusters", [2, 3])
        linkage = st.selectbox("Linkage", ["ward", "complete", "average"])

    generate_clusters = st.button("Generate Clusters")

    if generate_clusters:
        tfidf_vect = TfidfVectorizer()
        tfidf_matrix = tfidf_vect.fit_transform(df['review'].values)

        pca = PCA(n_components=2)
        pca_transformed_data = pca.fit_transform(tfidf_matrix.toarray())

        if selected_model == "Birch":
            clustering_model = Birch(threshold=threshold)
        elif selected_model == "Spectral":
            clustering_model = SpectralClustering(n_clusters=n_clusters, eigen_solver=eigen_solver)
        elif selected_model == "Hierarchical":
            clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

        start_time = time.time()
        cluster_labels = clustering_model.fit_predict(pca_transformed_data)
        end_time = time.time()
        clustering_time = end_time - start_time
        print(f"Execution time: {clustering_time:.2f} seconds")

        df['Cluster'] = cluster_labels

        # Display clustering result in words
        st.header("Clustering Result:")
        st.write(f"Number of clusters: {len(set(cluster_labels))}")
        st.write(f"Clustering time: {clustering_time:.2f} seconds")

        # Display cluster size
        cluster_sizes = pd.Series(cluster_labels).value_counts()
        st.header("Cluster Sizes:")
        fig, ax = plt.subplots()
        ax.bar(cluster_sizes.index, cluster_sizes.values)
        ax.set_xlabel("Cluster Label")
        ax.set_ylabel("Count")
        ax.set_title("Cluster Sizes")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.scatterplot(x=pca_transformed_data[:, 0], y=pca_transformed_data[:, 1], hue=cluster_labels, palette="viridis", ax=ax)
        ax.set_title("Clusters Visualization")
        st.pyplot(fig)

        # Show top terms for each cluster
        for cluster in set(cluster_labels):
            cluster_data = df[df['Cluster'] == cluster].select_dtypes(include=[np.number])
            top_terms = cluster_data.mean().nlargest(10)
            st.write(f"Top terms for Cluster {cluster}:")
            st.write(top_terms.to_frame('Mean').style.format('{:.2f}'))

        # Word cloud visualization for each cluster
        for cluster in set(cluster_labels):
            cluster_data = df[df['Cluster'] == cluster]
            wordcloud = WordCloud(width=800, height=400, random_state=0).generate(' '.join(cluster_data['review']))
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.write(f"Word Cloud for Cluster {cluster}:")
            st.pyplot(fig)


        # Train the selected clustering model
        if selected_model == "Birch":
            clustering_model = Birch(threshold=threshold)
            cluster_labels = clustering_model.fit_predict(pca_transformed_data)
        elif selected_model == "Spectral":
            clustering_model = SpectralClustering(n_clusters=n_clusters, eigen_solver=eigen_solver)
            cluster_labels = clustering_model.fit_predict(pca_transformed_data)
        elif selected_model == "Hierarchical":
            clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            cluster_labels = clustering_model.fit_predict(pca_transformed_data)

        # Model evaluation metrics
        st.header("Model Evaluation Metrics:")
        silhouette_metric = metrics.silhouette_score(pca_transformed_data, cluster_labels)
        calinski_harabasz_metric = metrics.calinski_harabasz_score(pca_transformed_data, cluster_labels)
        st.write(f"Silhouette Score: {silhouette_metric:.3f}")
        st.write(f"Calinski-Harabasz Index: {calinski_harabasz_metric:.3f}")

        st.session_state['cluster_labels'] = cluster_labels
        st.session_state['pca_transformed_data'] = pca_transformed_data
        st.session_state['clustering_model'] = clustering_model
        st.session_state['tfidf_vect'] = tfidf_vect
        st.session_state['pca'] = pca

    # If user has entered a phrase and clustering model is available
    if user_input and st.session_state['clustering_model'] and st.session_state['tfidf_vect']:
        # Vectorize the user's input using the same TF-IDF vectorizer
        user_input_vectorized = st.session_state['tfidf_vect'].transform([user_input])

        # Apply PCA transformation to the user's input
        user_input_pca = st.session_state['pca'].transform(user_input_vectorized.toarray())

        # Predict the cluster of the user's input using the trained model
        if isinstance(st.session_state['clustering_model'], SpectralClustering):
            from sklearn.metrics import pairwise_distances
            cluster_centers = np.array([np.mean(st.session_state['pca_transformed_data'][st.session_state['cluster_labels'] == i], axis=0) 
                                        for i in np.unique(st.session_state['cluster_labels'])])
            distances = pairwise_distances(user_input_pca, cluster_centers)
            user_cluster = np.argmin(distances)
        elif isinstance(st.session_state['clustering_model'], AgglomerativeClustering):
            # For Hierarchical clustering, you can use the predict method directly
            from sklearn.metrics import pairwise_distances
            cluster_centers = np.array([np.mean(st.session_state['pca_transformed_data'][st.session_state['cluster_labels'] == i], axis=0) 
                                        for i in np.unique(st.session_state['cluster_labels'])])
            distances = pairwise_distances(user_input_pca, cluster_centers)
            user_cluster = np.argmin(distances)
        else:
            user_cluster = st.session_state['clustering_model'].predict(user_input_pca)[0]

        # Display which cluster the input belongs to
        st.header(f"The input text belongs to *Cluster {user_cluster}*")

        # Optionally, display reviews from the same cluster
        review_table = pd.concat([
            df[df['Cluster'] == user_cluster][['review_title']].head(10),
            df[df['Cluster'] == user_cluster][['review_content']].head(10)
        ], axis=1)
        st.table(review_table)