import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from urllib.parse import urlparse

# Article processor functions
def validate_url(url):
    """Validate that a URL is properly formatted"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_article_content(url):
    """Extract content from an article URL"""
    if not validate_url(url):
        raise ValueError("Invalid URL format")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get title
        title = soup.title.string if soup.title else "No title found"
        
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        
        # Simple cleanup
        content = re.sub(r'\s+', ' ', content).strip()
        
        return {
            "url": url,
            "title": title,
            "content": content,
            "source": urlparse(url).netloc
        }
    except Exception as e:
        raise Exception(f"Failed to extract content: {str(e)}")

def preprocess_text(text):
    """Simple text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Graph manager functions
class ArticleGraph:
    """Manages the article graph and provides analysis methods"""
    
    def __init__(self):
        """Initialize the graph"""
        self.G = nx.Graph()
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.vectors = {}
        self.contents = {}
        self.fitted = False
        
    def add_article(self, article):
        """Add an article to the graph"""
        # Generate an ID if not provided
        if 'id' not in article:
            article['id'] = re.sub(r'[^\w]', '_', article['url'])
        
        # Standardize cluster format (convert 0/1 to "good"/"bad")
        if 'cluster' in article:
            if article['cluster'] == 0 or article['cluster'] == '0':
                article['cluster'] = "good"
            elif article['cluster'] == 1 or article['cluster'] == '1':
                article['cluster'] = "bad"
        
        # Add node to graph
        self.G.add_node(
            article['id'], 
            url=article['url'],
            title=article['title'],
            content=article['content'],
            source=article['source'],
            cluster=article.get('cluster', None)
        )
        
        # Store content for later vectorization
        processed_content = preprocess_text(article['content'])
        self.contents[article['id']] = processed_content
        
        # Reset fitted flag since we added new content
        self.fitted = False
        
        return article['id']
    
    def _fit_vectorizer(self):
        """Fit the TF-IDF vectorizer on all documents"""
        corpus = list(self.contents.values())
        if not corpus:  # Check if corpus is empty
            return
            
        self.vectorizer.fit(corpus)
        self.fitted = True
        
        # Transform all documents
        for article_id, content in self.contents.items():
            self.vectors[article_id] = self.vectorizer.transform([content])
    
    def calculate_similarities(self, article_id):
        """Calculate similarities between an article and all others"""
        # Make sure vectorizer is fitted
        if not self.fitted:
            self._fit_vectorizer()
            
        similarities = {}
        if article_id not in self.vectors:
            # This can happen if vectorization failed
            return similarities
            
        target_vector = self.vectors[article_id]
        
        for node_id, vector in self.vectors.items():
            if node_id != article_id:
                sim = cosine_similarity(target_vector, vector)[0][0]
                similarities[node_id] = sim
                
        return similarities
    
    def update_edges(self, article_id, threshold=0.2):  # Lower threshold from 0.3 to 0.2
        """Update graph edges based on similarities"""
        similarities = self.calculate_similarities(article_id)
        
        # Add edges for similarities above threshold
        for node_id, sim in similarities.items():
            if sim > threshold:
                self.G.add_edge(article_id, node_id, weight=sim)
    
    def extract_features(self, article_id):
        """Extract graph features for classification"""
        # Initialize features with default values
        features = {
            'avg_sim_good': 0,
            'avg_sim_bad': 0,
            'max_sim_good': 0,
            'max_sim_bad': 0,
            'count_good': 0,
            'count_bad': 0
        }
        
        # Get neighbors
        neighbors = list(self.G.neighbors(article_id))
        
        if not neighbors:
            # No neighbors means no connections to analyze
            return features
        
        # Calculate average similarity to each known cluster
        cluster_similarities = {'good': [], 'bad': []}
        
        for neighbor in neighbors:
            cluster = self.G.nodes[neighbor].get('cluster')
            if cluster in ['good', 'bad']:
                edge_weight = self.G[article_id][neighbor]['weight']
                cluster_similarities[cluster].append(edge_weight)
        
        # Calculate metrics for each cluster
        for cluster, sims in cluster_similarities.items():
            if sims:  # Only if we have similarities for this cluster
                features[f'avg_sim_{cluster}'] = np.mean(sims)
                features[f'max_sim_{cluster}'] = np.max(sims)
                features[f'count_{cluster}'] = len(sims)
        
        return features
    
    def get_labeled_nodes(self):
        """Get nodes with cluster labels"""
        return {node: data['cluster'] 
                for node, data in self.G.nodes(data=True) 
                if data.get('cluster') is not None}
                
    def visualize(self, highlight_node=None):
        """Create a visualization of the graph"""
        plt.figure(figsize=(12, 8))
        
        # Define positions for nodes
        pos = nx.spring_layout(self.G, seed=42)  # Add seed for consistent layout
        
        # Define node colors based on clusters
        node_colors = []
        node_sizes = []
        
        for node in self.G.nodes():
            cluster = self.G.nodes[node].get('cluster')
            
            # Set node size - larger for highlighted node
            if node == highlight_node:
                node_sizes.append(500)
            else:
                node_sizes.append(300)
                
            # Set node color based on cluster
            if node == highlight_node:
                if cluster == "good":
                    node_colors.append('lightgreen')
                elif cluster == "bad":
                    node_colors.append('lightcoral')
                else:
                    node_colors.append('purple')  # Purple only for highlighted unknown nodes
            elif cluster == "good":
                node_colors.append('green')
            elif cluster == "bad":
                node_colors.append('red')
            else:
                node_colors.append('gray')
        
        # Draw the graph
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, node_size=node_sizes)
        
        # Adjust edge width based on weight
        edge_widths = [self.G[u][v].get('weight', 1) * 3 for u, v in self.G.edges()]
        nx.draw_networkx_edges(self.G, pos, width=edge_widths, alpha=0.6)
        
        # Add labels with smaller font
        nx.draw_networkx_labels(self.G, pos, font_size=8)
        
        plt.title("Article Graph")
        plt.axis('off')
        return plt.gcf()
        
# Classifier functions
class GraphClassifier:
    """Classifies articles based on graph features"""
    
    def predict(self, features):
        """Improved rule-based prediction based on graph features"""
        result = {
            'article_id': None,
            'predicted_cluster': None,
            'confidence': 0,
            'supporting_evidence': []
        }
        
        # Get similarity and count features with proper handling of None values
        good_sim = features.get('avg_sim_good', 0) or 0
        bad_sim = features.get('avg_sim_bad', 0) or 0
        
        good_count = features.get('count_good', 0) or 0
        bad_count = features.get('count_bad', 0) or 0
        
        # Decision logic - first try using max similarity
        max_good = features.get('max_sim_good', 0) or 0
        max_bad = features.get('max_sim_bad', 0) or 0
        
        # If we have strong connections to either class, use that
        if max_good > 0.6 and max_good > max_bad:
            result['predicted_cluster'] = "good"
            result['confidence'] = max_good
            return result
        
        if max_bad > 0.6 and max_bad > max_good:
            result['predicted_cluster'] = "bad"
            result['confidence'] = max_bad
            return result
        
        # Otherwise use average similarities if they differ significantly
        if abs(good_sim - bad_sim) > 0.05:
            if good_sim > bad_sim:
                result['predicted_cluster'] = "good"
                result['confidence'] = good_sim
            else:
                result['predicted_cluster'] = "bad"
                result['confidence'] = bad_sim
        # If similarities are very close, use counts
        elif good_count != bad_count:
            total_connections = good_count + bad_count
            if total_connections == 0:
                # No connections at all, default to unknown
                result['predicted_cluster'] = "unknown"
                result['confidence'] = 0.5
            elif good_count > bad_count:
                result['predicted_cluster'] = "good"
                result['confidence'] = 0.5 + (good_count / total_connections) * 0.5
            else:
                result['predicted_cluster'] = "bad"
                result['confidence'] = 0.5 + (bad_count / total_connections) * 0.5
        else:
            # If everything is tied, slightly favor "good" as a more neutral default
            result['predicted_cluster'] = "good"
            result['confidence'] = 0.51
            
        return result

# Main Streamlit application
def main():
    st.title("Samuel Identity Disambiguation System")
    st.markdown("""
    This system disambiguates between two individuals named Samuel Jackson - one with legitimate activities ("good Samuel") 
    and another involved in money laundering ("bad Samuel").
    """)
    
    # Initialize graph if not already done
    if 'graph' not in st.session_state:
        st.session_state.graph = ArticleGraph()
        st.session_state.classifier = GraphClassifier()
    
    # Sidebar for loading data and controls
    with st.sidebar:
        st.header("Controls")
        
        # Load CSV data
        st.subheader("Load Training Data")
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} articles")
                
                # Display CSV structure for debugging
                with st.expander("Show CSV Structure"):
                    st.write(df.head())
                    st.write("Columns:", df.columns.tolist())
                
                # Process CSV data button
                if st.button("Process CSV Data"):
                    with st.spinner("Processing articles..."):
                        # Clear existing graph
                        st.session_state.graph = ArticleGraph()
                        
                        # Count articles by type for reporting
                        good_count = 0
                        bad_count = 0
                        
                        for _, row in df.iterrows():
                            # Create article object with proper cluster assignment
                            try:
                                # Handle different column names and formats
                                if 'category' in df.columns:
                                    cluster = "good" if row['category'] == 0 else "bad"
                                elif 'decision' in df.columns:
                                    cluster = "good" if row['decision'] == 0 else "bad"
                                elif 'cluster' in df.columns:
                                    if isinstance(row['cluster'], (int, float)):
                                        cluster = "good" if row['cluster'] == 0 else "bad"
                                    else:
                                        cluster = row['cluster'].lower()
                                else:
                                    # Default if no cluster column found
                                    cluster = None
                                
                                # Track counts
                                if cluster == "good":
                                    good_count += 1
                                elif cluster == "bad":
                                    bad_count += 1
                                
                                # Get article URL
                                url = row['link'] if 'link' in df.columns else f"https://example.com/article-{_}"
                                
                                # Create simplified article object
                                article = {
                                    'id': f"article-{_}",  # Ensure unique ID
                                    'url': url,
                                    'title': row.get('title', f"Article {_} about Samuel Jackson"),
                                    'content': row.get('content', f"This is article {_} about Samuel Jackson with decision {cluster}"),
                                    'source': urlparse(url).netloc if 'link' in df.columns else "example.com",
                                    'cluster': cluster
                                }
                                
                                article_id = st.session_state.graph.add_article(article)
                            except Exception as e:
                                st.warning(f"Error processing row {_}: {str(e)}")
                                
                        # Update all edges
                        for article_id in st.session_state.graph.G.nodes():
                            st.session_state.graph.update_edges(article_id)
                        
                        st.success(f"Graph updated: {good_count} good, {bad_count} bad articles")
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
        
        # URL input for new articles
        st.subheader("Process New Article")
        new_url = st.text_input("Article URL")
        
        if st.button("Process Article") and new_url:
            try:
                with st.spinner("Processing article..."):
                    # Extract content
                    article_data = extract_article_content(new_url)
                    
                    # Add to graph
                    article_id = st.session_state.graph.add_article(article_data)
                    
                    # Update edges
                    st.session_state.graph.update_edges(article_id)
                    
                    # Classify
                    features = st.session_state.graph.extract_features(article_id)
                    result = st.session_state.classifier.predict(features)
                    result['article_id'] = article_id
                    
                    st.session_state.current_article = article_id
                    st.session_state.classification_result = result
                    st.session_state.current_features = features
                    
                    st.success("Article processed successfully")
                    
            except Exception as e:
                st.error(f"Error processing article: {str(e)}")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Graph Visualization", "Article Analysis", "Debug Info"])
    
    with tab1:
        st.header("Article Graph")
        
        # Check if we have a graph with nodes
        if 'graph' in st.session_state and len(st.session_state.graph.G.nodes()) > 0:
            current_article = st.session_state.get('current_article', None)
            fig = st.session_state.graph.visualize(highlight_node=current_article)
            st.pyplot(fig)
            
            # Display graph stats
            st.subheader("Graph Statistics")
            
            # Count node types
            good_nodes = sum(1 for _, data in st.session_state.graph.G.nodes(data=True) 
                             if data.get('cluster') == "good")
            bad_nodes = sum(1 for _, data in st.session_state.graph.G.nodes(data=True) 
                            if data.get('cluster') == "bad")
            unknown_nodes = len(st.session_state.graph.G.nodes()) - good_nodes - bad_nodes
            
            st.write(f"Total nodes: {len(st.session_state.graph.G.nodes())}")
            st.write(f"Good nodes: {good_nodes}")
            st.write(f"Bad nodes: {bad_nodes}")
            st.write(f"Unknown nodes: {unknown_nodes}")
            st.write(f"Total edges: {len(st.session_state.graph.G.edges())}")
        else:
            st.info("Load data from CSV or process articles to build the graph")
    
    with tab2:
        st.header("Article Classification")
        
        if 'classification_result' in st.session_state:
            result = st.session_state.classification_result
            
            st.subheader("Classification Result")
            cluster = result['predicted_cluster']
            confidence = result['confidence']
            
            # Show result with color
            if cluster == "good":
                st.success(f"This article refers to the **GOOD** Samuel Jackson (Confidence: {confidence:.2f})")
            elif cluster == "bad":
                st.error(f"This article refers to the **BAD** Samuel Jackson (Confidence: {confidence:.2f})")
            else:
                st.warning(f"Unable to determine which Samuel Jackson (Confidence: {confidence:.2f})")
                
            # Show article details
            if 'current_article' in st.session_state:
                article_id = st.session_state.current_article
                article_data = st.session_state.graph.G.nodes[article_id]
                
                st.subheader("Article Details")
                st.write(f"**Title:** {article_data['title']}")
                st.write(f"**Source:** {article_data['source']}")
                with st.expander("Content Preview"):
                    st.write(article_data['content'][:500] + "..." if len(article_data['content']) > 500 else article_data['content'])
        else:
            st.info("Process an article to see classification results")
    
    with tab3:
        st.header("Debug Information")
        
        if 'current_features' in st.session_state:
            st.subheader("Classification Features")
            features = st.session_state.current_features
            
            # Create a DataFrame for better display
            feature_df = pd.DataFrame({
                'Feature': features.keys(),
                'Value': features.values()
            })
            
            st.dataframe(feature_df)
            
            # Show key metrics for decision making
            st.subheader("Key Metrics")
            cols = st.columns(2)
            
            with cols[0]:
                st.metric("Good Similarity", f"{features.get('avg_sim_good', 0):.3f}")
                st.metric("Good Max Similarity", f"{features.get('max_sim_good', 0):.3f}")
                st.metric("Good Connections", features.get('count_good', 0))
                
            with cols[1]:
                st.metric("Bad Similarity", f"{features.get('avg_sim_bad', 0):.3f}")
                st.metric("Bad Max Similarity", f"{features.get('max_sim_bad', 0):.3f}")
                st.metric("Bad Connections", features.get('count_bad', 0))
                
            # Show neighboring articles
            if 'current_article' in st.session_state:
                st.subheader("Connected Articles")
                article_id = st.session_state.current_article
                
                # Get neighbors
                neighbors = list(st.session_state.graph.G.neighbors(article_id))
                if neighbors:
                    neighbor_data = []
                    for neighbor in neighbors:
                        node_data = st.session_state.graph.G.nodes[neighbor]
                        edge_data = st.session_state.graph.G[article_id][neighbor]
                        
                        neighbor_data.append({
                            'ID': neighbor,
                            'Title': node_data.get('title', 'Unknown'),
                            'Cluster': node_data.get('cluster', 'unknown'),
                            'Similarity': edge_data.get('weight', 0)
                        })
                    
                    # Create DataFrame and display sorted by similarity
                    neighbor_df = pd.DataFrame(neighbor_data)
                    neighbor_df = neighbor_df.sort_values('Similarity', ascending=False)
                    st.dataframe(neighbor_df)
                else:
                    st.info("No connected articles")
        else:
            st.info("Process an article to see debug information")

if __name__ == "__main__":
    main()