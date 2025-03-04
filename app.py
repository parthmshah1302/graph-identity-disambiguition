import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

# Graph manager class
class ArticleGraph:
    """Manages the article graph and provides analysis methods"""
    
    def __init__(self):
        """Initialize the graph"""
        self.G = nx.Graph()
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.vectors = {}
        self.contents = {}
        self.fitted = False
        self.good_centroid = None
        self.bad_centroid = None
        
    def add_article(self, article):
        """Add an article to the graph"""
        # Generate an ID if not provided
        if 'id' not in article:
            article['id'] = re.sub(r'[^\w]', '_', article['url'])
        
        # Standardize cluster format (convert 0/1 to "good"/"bad")
        if 'category' in article:
            if article['category'] in [0, '0']:
                article['cluster'] = "good"
            elif article['category'] in [1, '1']:
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
            self.vectors[article_id] = self.vectorizer.transform([content])[0]
        
        # Calculate centroids for each cluster
        self._calculate_centroids()
    
    def _calculate_centroids(self):
        """Calculate centroid vectors for good and bad clusters"""
        # Get nodes by cluster
        good_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('cluster') == 'good']
        bad_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('cluster') == 'bad']
        
        # Calculate good centroid if we have good nodes
        if good_nodes and all(node in self.vectors for node in good_nodes):
            good_vectors = [self.vectors[node] for node in good_nodes]
            self.good_centroid = sum(good_vectors) / len(good_vectors)
        else:
            self.good_centroid = None
        
        # Calculate bad centroid if we have bad nodes
        if bad_nodes and all(node in self.vectors for node in bad_nodes):
            bad_vectors = [self.vectors[node] for node in bad_nodes]
            self.bad_centroid = sum(bad_vectors) / len(bad_vectors)
        else:
            self.bad_centroid = None
    
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
                # Convert sparse vectors to dense for computation
                sim = cosine_similarity(target_vector.reshape(1, -1), 
                                        vector.reshape(1, -1))[0][0]
                similarities[node_id] = sim
                
        return similarities
    
    def update_edges(self, article_id, threshold=0.7):  # Lower threshold to 0.3
        """Update graph edges based on similarities"""
        similarities = self.calculate_similarities(article_id)
        
        # Add edges for similarities above threshold
        for node_id, sim in similarities.items():
            if sim > threshold:
                self.G.add_edge(article_id, node_id, weight=sim)
    
    def calculate_centroid_similarities(self, article_id):
        """Calculate similarities between article and cluster centroids"""
        # Make sure we have fitted the vectorizer and calculated centroids
        if not self.fitted:
            self._fit_vectorizer()
        
        # Initialize similarities
        similarities = {
            'good_similarity': 0,
            'bad_similarity': 0
        }
        
        # Get article vector
        if article_id not in self.vectors:
            return similarities
        
        article_vector = self.vectors[article_id]
        
        # Calculate similarity with good centroid
        if self.good_centroid is not None:
            good_sim = cosine_similarity(article_vector.reshape(1, -1), 
                                          self.good_centroid.reshape(1, -1))[0][0]
            similarities['good_similarity'] = good_sim
        
        # Calculate similarity with bad centroid
        if self.bad_centroid is not None:
            bad_sim = cosine_similarity(article_vector.reshape(1, -1), 
                                         self.bad_centroid.reshape(1, -1))[0][0]
            similarities['bad_similarity'] = bad_sim
        
        return similarities
                
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
        
# Classifier class
class GraphClassifier:
    """Classifies articles based solely on distance from centroids"""
    
    def predict(self, similarities):
        """Simple centroid-based prediction using cosine similarities"""
        result = {
            'predicted_cluster': None,
            'confidence': 0,
        }
        
        # Extract the centroid similarities
        good_sim = similarities.get('good_similarity', 0)
        bad_sim = similarities.get('bad_similarity', 0)
        
        # Compare distances to centroids - use whichever is higher
        if good_sim > bad_sim:
            result['predicted_cluster'] = "good"
            # Calculate confidence based on difference
            diff = good_sim - bad_sim
            result['confidence'] = 0.5 + min(0.5, diff)
        elif bad_sim > good_sim:
            result['predicted_cluster'] = "bad"
            diff = bad_sim - good_sim
            result['confidence'] = 0.5 + min(0.5, diff)
        else:
            # Equal distances
            result['predicted_cluster'] = "unknown"
            result['confidence'] = 0.5
                
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
                        
                        # Make sure to fit the vectorizer and calculate centroids
                        st.session_state.graph._fit_vectorizer()
                        
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
                    
                    # Make sure vectorizer is fitted and centroids calculated
                    if not st.session_state.graph.fitted:
                        st.session_state.graph._fit_vectorizer()
                    
                    # Calculate similarities
                    centroid_similarities = st.session_state.graph.calculate_centroid_similarities(article_id)
                    
                    # Classify
                    result = st.session_state.classifier.predict(centroid_similarities)
                    
                    st.session_state.current_article = article_id
                    st.session_state.classification_result = result
                    st.session_state.centroid_similarities = centroid_similarities
                    
                    st.success("Article processed successfully")
                    
            except Exception as e:
                st.error(f"Error processing article: {str(e)}")
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["Graph Visualization", "Article Classification"])
    
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
            
            # Display centroid similarities for debugging
            if 'centroid_similarities' in st.session_state:
                sims = st.session_state.centroid_similarities
                st.subheader("Debug: Centroid Similarities")
                st.write(f"Similarity to GOOD centroid: {sims.get('good_similarity', 0):.4f}")
                st.write(f"Similarity to BAD centroid: {sims.get('bad_similarity', 0):.4f}")
                
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

if __name__ == "__main__":
    main()